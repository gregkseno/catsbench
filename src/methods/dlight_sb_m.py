from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .dlight_sb import DLightSB
from ..data.batch import Batch
from ..data.prior import Prior
from ..utils import optimize_coupling, gumbel_sample


M_HPARAMS = (
    'use_mini_batch', 'kl_loss_coeff', 'mse_loss_coeff',
)


class DLightSB_M(DLightSB):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_categories: int,
        num_potentials: int,
        num_timesteps: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        kl_loss_coeff: float = 1.0,
        mse_loss_coeff: float = 0.0,
        use_mini_batch: bool = False,
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian', 
        entropy_lambda: float = 1e-3,
        entropy_warmup_steps: int = 20_000,
        sample_prob: float = 0.9,
        tau: float = 1.0,
    ) -> None:
        super().__init__(
            prior=prior,
            dim=dim,
            num_categories=num_categories,
            num_potentials=num_potentials,
            num_timesteps=num_timesteps,
            optimizer=optimizer,
            scheduler=scheduler,
            distr_init=distr_init,
            entropy_lambda=entropy_lambda,
            entropy_warmup_steps=entropy_warmup_steps,
            sample_prob=sample_prob,
            tau=tau,
        )
        self.save_hyperparameters(*M_HPARAMS, logger=False)

    def kl_loss(
        self,
        true_logits: torch.Tensor,
        pred_marginal_logits: torch.Tensor,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
        log_u_t: torch.Tensor,
        log_u_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """Joint KL-divergence calculation."""
        true_log_probs = true_logits.log_softmax(dim=-1)
        pred_marginal_log_probs = pred_marginal_logits.log_softmax(dim=-1)
        true_probs = true_log_probs.exp()

        marginal_kl = (
            true_probs * (true_log_probs - pred_marginal_log_probs)
        ).sum(dim=-1).flatten(start_dim=1).sum(dim=-1)
        sampled_marginal_log_prob = torch.gather(
            pred_marginal_log_probs, dim=-1, index=x_tp1.unsqueeze(-1)
        ).squeeze(-1).flatten(start_dim=1).sum(dim=-1)

        x_t = x_t.flatten(start_dim=1)
        x_tp1 = x_tp1.flatten(start_dim=1)
        log_reference = self.prior.extract(
            "onestep", t + 1, row_id=x_t, column_id=x_tp1
        ).sum(dim=1)
        log_potential_tp1 = torch.logsumexp(
            self.log_alpha[None, :] + log_u_tp1.sum(dim=1), dim=-1
        )
        log_normalizer_t = torch.logsumexp(
            self.log_alpha[None, :] + log_u_t.sum(dim=1), dim=-1
        )
        sampled_joint_log_prob = (
            log_reference + log_potential_tp1 - log_normalizer_t
        )

        return (
            marginal_kl
            + sampled_marginal_log_prob
            - sampled_joint_log_prob
        ).mean()
        
    def mse_loss(
        self,
        true_logits: torch.Tensor,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
        log_u_t: torch.Tensor,
        log_u_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """Joint MSE calculation."""
        true_log_probs = true_logits.log_softmax(dim=-1)
        sampled_true_log_prob = torch.gather(
            true_log_probs, dim=-1, index=x_tp1.unsqueeze(-1)
        ).squeeze(-1).flatten(start_dim=1).sum(dim=-1)
        x_t = x_t.flatten(start_dim=1)
        x_tp1 = x_tp1.flatten(start_dim=1)
        log_reference = self.prior.extract(
            "onestep", t + 1, row_id=x_t, column_id=x_tp1
        ).sum(dim=1)
        log_potential_tp1 = torch.logsumexp(
            self.log_alpha[None, :] + log_u_tp1.sum(dim=1), dim=-1
        )
        log_normalizer_t = torch.logsumexp(
            self.log_alpha[None, :] + log_u_t.sum(dim=1), dim=-1
        )
        sampled_model_log_prob = (
            log_reference + log_potential_tp1 - log_normalizer_t
        )

        true_joint_prob = sampled_true_log_prob.exp()
        model_joint_prob = sampled_model_log_prob.exp()
        loss = (
            (true_joint_prob - model_joint_prob).square()
            / true_joint_prob.clamp_min(torch.finfo(true_joint_prob.dtype).tiny)
        )
        return loss.mean()
    
    def _log_phi_tp1_all(
        self, 
        log_u_t: torch.Tensor,
        tp1: torch.Tensor,
        log_u_table: torch.Tensor,
    ) -> torch.Tensor:
        if tp1.numel() == 1:
            tp1 = tp1.expand(log_u_t.shape[0])
        log_z_t = log_u_t.sum(dim=1) # [B, K]
        log_w = (
            self.log_alpha.view(1, 1, -1)
            + log_z_t[:, None, :]
            - log_u_t
        ) # [B, D, K]
        log_u_tp1 = log_u_table.index_select(0, tp1.long()) # [B, D, S, K]
        return torch.logsumexp(
            log_u_tp1 + log_w.unsqueeze(-2), dim=-1
        ) # [B, D, S]

    def _log_u_tp1_star_all(
        self, 
        log_u_table: torch.Tensor,
        tp1: torch.Tensor,
        k_star: torch.Tensor,
    ) -> torch.Tensor:
        if tp1.numel() == 1:
            tp1 = tp1.expand(k_star.shape[0])
        log_u_tp1 = log_u_table.index_select(0, tp1.long()) # [B, D, S, K]
        k_index = k_star.view(-1, 1, 1, 1).expand(
            -1, self.hparams.dim, self.hparams.num_categories, 1
        )
        return torch.gather(log_u_tp1, dim=-1, index=k_index).squeeze(-1)

    def get_transition_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        log_u_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_u_table is None:
            log_u_table = self.precompute_log_u_table(self.prior.log_p_cum)
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t, log_u_table) # [B, D, K]
        log_phi_tp1 = self._log_phi_tp1_all(
            log_u_t, tp1, log_u_table
        ) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        transition_logits = log_phi_tp1 + onestep  
        return transition_logits.reshape(*input_shape, self.hparams.num_categories) # [B, ..., S]

    def optimal_projection(
        self,
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=0, high=self.hparams.num_timesteps + 1,
            size=(batch_size,), device=self.device
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)

        true_q_posterior_logits = self.prior.posterior_logits_reverse(true_x_end, x_t, t, logits=False)
        time_to_end = self.hparams.num_timesteps + 1 - t
        log_u_table = self.precompute_log_u_table(
            self.prior.log_p_cum
        ) # [T, D, S, K]
        log_u_t = self._log_u_t(
            x_t.flatten(start_dim=1), time_to_end, log_u_table
        ) # [B, D, K]

        loss = true_q_posterior_logits.new_zeros(())
        kl = true_q_posterior_logits.new_zeros(())
        mse = true_q_posterior_logits.new_zeros(())
        if self.hparams.kl_loss_coeff > 0 or self.hparams.mse_loss_coeff > 0:
            x_tp1 = gumbel_sample(true_q_posterior_logits, tau=1.0, dim=-1)
            log_u_tp1 = self._log_u_t(
                x_tp1.flatten(start_dim=1), time_to_end - 1, log_u_table
            ) # [B, D, K]
        if self.hparams.kl_loss_coeff > 0:
            pred_p_transition_logits = self.get_transition_logits(
                x_t, t=t, log_u_table=log_u_table
            )
            kl = self.kl_loss(
                true_q_posterior_logits,
                pred_p_transition_logits,
                x_t,
                x_tp1,
                t,
                log_u_t,
                log_u_tp1,
            )
            loss = loss + self.hparams.kl_loss_coeff * kl
        if self.hparams.mse_loss_coeff > 0:
            mse = self.mse_loss(
                true_q_posterior_logits,
                x_t,
                x_tp1,
                t,
                log_u_t,
                log_u_tp1,
            )
            loss = loss + self.hparams.mse_loss_coeff * mse

        ent_lambda = self._ent_lambda()
        track_entropy_grad = torch.is_grad_enabled() and ent_lambda > 0
        with torch.set_grad_enabled(track_entropy_grad):
            log_z_t = log_u_t.sum(dim=1)
            ent = self.entropy_loss(self.log_alpha[None, :] + log_z_t, dim=-1)

        if ent_lambda > 0:
            loss = loss - ent_lambda * ent.mean()

        posterior_eff_k = ent.detach().exp().mean()
        prior_eff_k = self.entropy_loss(self.log_alpha.detach(), dim=0).exp()

        info = {
            'kl_loss': kl, 'mse_loss': mse, 
            'posterior_eff_k': posterior_eff_k,
            'prior_eff_k': prior_eff_k
        }
        return loss, info

    def loss(
        self,
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.hparams.use_mini_batch:
            true_x_start, true_x_end = optimize_coupling(
                true_x_start, true_x_end
            )
        return self.optimal_projection(true_x_start, true_x_end)

    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Dict[str, Any]:
        x_start, x_end = batch

        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
            output_batch = Batch(encoded=(x_start, x_end))
        else:
            output_batch = batch
        loss, info = self.optimal_projection(x_start, x_end)

        info = {f"train/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return {'loss': loss, 'batch': output_batch}

    @torch.no_grad()
    def markov_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        return_transitions: bool = False,
        log_u_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_u_table is None:
            log_u_table = self.precompute_log_u_table(self.prior.log_p_cum)
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)

        t_orig = t  # keep original for onestep
        t = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = self._log_u_t(x_t, t, log_u_table) # [B, D, K]
        log_z_t = log_u_t.sum(dim=1) # [B, K]
        log_w_k = self.log_alpha[None, :] + log_z_t # [B, K]
        k_star = gumbel_sample(log_w_k, tau=self.hparams.tau, dim=-1) # [B]

        log_u_tp1_star = self._log_u_tp1_star_all(
            log_u_table, tp1, k_star
        ) # [B, D, S]
        onestep = self.prior.extract("onestep", t_orig + 1, row_id=x_t) # [B, D, S]
        logits = onestep + log_u_tp1_star # [B, D, S]
        
        x_tp1 = gumbel_sample(logits, tau=self.hparams.tau, dim=-1).reshape(input_shape) # [B, ...]
        if return_transitions:
            log_phi_tp1 = self._log_phi_tp1_all(
                log_u_t, tp1, log_u_table
            ) # [B, D, S]
            transition_logits = log_phi_tp1 + onestep # [B, D, S]
            return x_tp1, transition_logits
        return x_tp1

    @torch.no_grad()
    def sample(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = True
    ) -> torch.Tensor:
        '''Sample from the model starting from `x` returning the final sample.'''
        log_u_table = self.precompute_log_u_table(self.prior.log_p_cum)
        if use_onestep_sampling:
            input_shape = x_start.shape
            x_start = x_start.flatten(start_dim=1) # (B, D)

            last_timestep = torch.full(
                size=(x_start.shape[0],), 
                fill_value=self.hparams.num_timesteps + 1, 
                device=self.device 
            )
            log_u = self._log_u_t(
                x_start, last_timestep, log_u_table
            ) # [B, D, K]
            log_z = log_u.sum(dim=1) # [B, K]

            log_w_k = self.log_alpha[None, :] + log_z # [B, K]
            k_star = gumbel_sample(log_w_k, dim=-1, tau=self.hparams.tau) # [B]
            k_idx = k_star.view(
                x_start.shape[0], 1, 1, 1 # [B, 1, 1, 1]
            ).expand(-1, self.hparams.dim, self.hparams.num_categories, 1) # [B, D, S, 1]
            log_cp_cores_star = torch.gather(
                self.log_cp_cores.unsqueeze(0).expand(
                    x_start.shape[0], -1, -1, -1 # [B, D, S, K]
                ), dim=-1, index=k_idx
            ).squeeze(-1) # [B, D, S]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_start) # [B, D, S]
            logits = log_pi_ref + log_cp_cores_star # [B, D, S]
            x_end = gumbel_sample(logits, dim=-1, tau=self.hparams.tau).reshape(input_shape) # [B, ...]
        else:
            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
                t = torch.full([x_start.shape[0]], t, device=self.device)
                x_t = self.markov_sample(
                    x_t, t, return_transitions=False, log_u_table=log_u_table
                )
            x_end = x_t
        return x_end
    
    @torch.no_grad()
    def sample_trajectory(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = False,
        return_transitions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_onestep_sampling:
            assert not return_transitions, \
                'Returning transitions is not supported when using bridge samples!'
        
        trajectory, transitions = [x_start], []
        if use_onestep_sampling:
            x_end = self.sample(x_start, use_onestep_sampling=True)
            
            for t in range(1, self.hparams.num_timesteps + 1):
                t = torch.full((x_start.shape[0],), t, device=x_start.device)
                x_t = self.prior.sample_bridge(x_start, x_end, t)
                trajectory.append(x_t)
            trajectory.append(x_end)
            
        else:
            log_u_table = self.precompute_log_u_table(self.prior.log_p_cum)
            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
                t = torch.full([x_t.shape[0]], t, device=self.device)
                out = self.markov_sample(
                    x_t, t,
                    return_transitions=return_transitions,
                    log_u_table=log_u_table,
                )
                if return_transitions:
                    x_t, logits = out
                    transitions.append(logits)
                else:
                    x_t = out
                trajectory.append(x_t)

        trajectory = torch.stack(trajectory, dim=0)
        if return_transitions:
            transitions = torch.stack(transitions, dim=0)
            return trajectory, transitions
        return trajectory
