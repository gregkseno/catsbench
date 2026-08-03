from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from .csbm import CSBM
from ..utils import gumbel_sample


class ARCSBM(CSBM):
    """CSBM with an autoregressive reverse-transition denoiser.

    Each denoiser must implement ``model(x_t, t, x_prefix)`` and return logits
    with shape ``(*x_t.shape, num_categories)``.  ``x_prefix`` has the same
    shape as ``x_t``: generated/teacher-forced coordinates contain category
    ids and all remaining coordinates contain ``model.mask_token_id``.

    Training uses the chain-rule identity

    ``KL(q(z) || p(z)) = E_q[sum_i KL(q_i || p_i(. | z_<i))]``.

    Since the bridge posterior ``q(z | x_t, x_0)`` factorizes, one sample from
    it supplies all teacher-forced prefixes and the resulting loss is an
    unbiased Monte Carlo estimate of the joint transition KL.
    """

    @staticmethod
    def _mask_token_id(model: torch.nn.Module) -> int:
        mask_token_id = getattr(model, "mask_token_id", None)
        if mask_token_id is None:
            raise AttributeError(
                "An ARCSBM denoiser must define `mask_token_id` and accept "
                "model(x_t, t, x_prefix)."
            )
        return int(mask_token_id)

    @staticmethod
    def _flatten_logits(logits: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        expected_prefix = x_t.shape
        if logits.shape[:-1] != expected_prefix:
            raise ValueError(
                "Autoregressive denoiser logits must have shape "
                f"(*x_t.shape, num_categories); got {tuple(logits.shape)} for "
                f"x_t shape {tuple(x_t.shape)}."
            )
        return logits.reshape(x_t.shape[0], -1, logits.shape[-1])

    def _conditional_logits(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
        *,
        sample: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate AR conditionals along a teacher-forced or sampled prefix.

        When ``x_prev`` is supplied, coordinate ``i`` is predicted using only
        ``x_prev[..., :i]``.  With no supplied state, prefixes are constructed
        ancestrally.  Greedy prefixes are used for logits-only calls so that
        diagnostics are deterministic; ``sample=True`` uses Gumbel sampling.
        """
        if x_t.ndim < 2:
            raise ValueError(f"Expected a batched state, got shape {tuple(x_t.shape)}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(
                f"Expected t shape ({x_t.shape[0]},), got {tuple(t.shape)}"
            )
        if x_prev is not None and x_prev.shape != x_t.shape:
            raise ValueError(
                f"x_prev and x_t must have the same shape, got "
                f"{tuple(x_prev.shape)} and {tuple(x_t.shape)}"
            )

        teacher_forced_logits = getattr(model, "teacher_forced_logits", None)
        if x_prev is not None and not sample and callable(teacher_forced_logits):
            logits = teacher_forced_logits(x_t, t, x_prev)
            self._flatten_logits(logits, x_t)
            return logits

        batch_size = x_t.shape[0]
        event_size = x_t[0].numel()
        event_shape = x_t.shape[1:]
        prefix = torch.full_like(x_t, self._mask_token_id(model))
        prefix_flat = prefix.reshape(batch_size, event_size)
        teacher_flat = None if x_prev is None else x_prev.reshape(batch_size, event_size)

        selected_logits = []
        generated = []
        for index in range(event_size):
            # Some modules save integer inputs for their backward pass.  Pass a
            # clone so revealing the next token cannot invalidate that saved
            # tensor's autograd version counter.
            logits = model(x_t, t, prefix.clone())
            logits_i = self._flatten_logits(logits, x_t)[:, index]
            selected_logits.append(logits_i)

            if teacher_flat is not None:
                token = teacher_flat[:, index]
            elif sample:
                token = gumbel_sample(logits_i, tau=self.hparams.tau, dim=-1)
                if self.hparams.argmax_mode:
                    token = torch.where(t == 1, logits_i.argmax(dim=-1), token)
            else:
                token = logits_i.argmax(dim=-1)

            # Prefix tensors contain integer token ids and never participate in
            # autograd, so updating the next conditioning slot in-place is safe.
            prefix_flat[:, index] = token
            generated.append(token)

        logits = torch.stack(selected_logits, dim=1).reshape(
            batch_size, *event_shape, selected_logits[0].shape[-1]
        )
        tokens = torch.stack(generated, dim=1).reshape(batch_size, *event_shape)
        return (tokens, logits) if sample else logits

    def kl_loss(
        self, true_logits: torch.Tensor, pred_logits: torch.Tensor
    ) -> torch.Tensor:
        """Joint chain-rule KL: sum coordinates, then average the batch."""
        true_log_probs = true_logits.log_softmax(dim=-1)
        pred_log_probs = pred_logits.log_softmax(dim=-1)
        true_probs = true_log_probs.exp()
        support = true_probs > 0
        log_ratio = torch.where(
            support, true_log_probs - pred_log_probs, torch.zeros_like(true_probs)
        )
        coordinate_kl = (true_probs * log_ratio).sum(dim=-1)
        return coordinate_kl.flatten(start_dim=1).sum(dim=-1).mean()

    def ce_loss(
        self, true_x_prev: torch.Tensor, pred_logits: torch.Tensor
    ) -> torch.Tensor:
        """Joint transition NLL: sum coordinates, then average the batch."""
        per_coordinate = F.cross_entropy(
            pred_logits.movedim(-1, 1),
            true_x_prev,
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        )
        return per_coordinate.flatten(start_dim=1).sum(dim=-1).mean()

    def markovian_projection(
        self,
        fb: Literal["forward", "backward"],
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=1,
            high=self.hparams.num_timesteps + 2,
            size=(batch_size,),
            device=self.device,
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)
        true_logits = self.prior.posterior_logits(
            true_x_start, x_t, t, logits=False
        )

        # The sampled state is used only as a teacher-forced prefix.  The KL at
        # every coordinate still compares the complete categorical q_i and p_i.
        true_x_prev = gumbel_sample(true_logits, tau=1.0, dim=-1)
        pred_logits = self._conditional_logits(
            self.models[fb], x_t, t, x_prev=true_x_prev
        )

        loss = true_logits.new_zeros(())
        kl = true_logits.new_zeros(())
        ce = true_logits.new_zeros(())
        mse = true_logits.new_zeros(())
        if self.hparams.kl_loss_coeff > 0:
            kl = self.kl_loss(true_logits, pred_logits)
            loss = loss + self.hparams.kl_loss_coeff * kl
        if self.hparams.ce_loss_coeff > 0:
            # This is the sampled joint transition NLL.  Its gradient is the
            # same cross-entropy term as the chain-rule KL estimator.
            ce = self.ce_loss(true_x_prev, pred_logits)
            loss = loss + self.hparams.ce_loss_coeff * ce
        if self.hparams.mse_loss_coeff > 0:
            mse = self.mse_loss(true_logits, pred_logits)
            loss = loss + self.hparams.mse_loss_coeff * mse

        info = {
            f"kl_loss_{fb}": kl,
            f"ce_loss_{fb}": ce,
            f"mse_loss_{fb}": mse,
        }
        return loss, info

    @torch.no_grad()
    def get_transition_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        fb: Optional[Literal["forward", "backward"]] = None,
        x_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return conditionals along ``x_prev`` prefixes (or a greedy rollout).

        A joint AR transition has no single prefix-independent ``[B, D, S]``
        tensor.  Callers evaluating a known transition should therefore pass
        its next state as ``x_prev``.  Omitting it returns conditionals along a
        deterministic greedy rollout, which is useful for inspection only.
        """
        fb = fb or self.fb
        was_training = self.models[fb].training
        self.models[fb].eval()
        with self.emas[fb].average_parameters():
            logits = self._conditional_logits(
                self.models[fb], x_t, t, x_prev=x_prev
            )
        if was_training:
            self.models[fb].train()
        return logits

    def markov_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        fb: Literal["forward", "backward"],
        return_transitions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        samples, logits = self._conditional_logits(
            self.models[fb], x_t, t, sample=True
        )
        if return_transitions:
            return samples, logits
        return samples
