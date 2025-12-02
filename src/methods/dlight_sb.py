from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule

from src.utils import gumbel_sample
from src.utils.ranked_logger import RankedLogger
from src.data.prior import Prior


log = RankedLogger(__name__, rank_zero_only=True)

HPARAMS = (
    'dim', 'num_categories', 'num_potentials', 'num_timesteps', 
    'distr_init', 'tau', 'sample_prob', 'optimizer', 'scheduler'
)

class DLightSB(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_categories: int,
        num_potentials: int,
        num_timesteps: int,
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian',
        sample_prob: float = 0.9,
        tau: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.ParameterList([
            nn.Parameter(torch.empty(
                num_potentials, num_categories,
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            for _ in range(dim)
        ])

        self.bidirectional = False  
        self.iteration = 1
        self._did_weight_init = False
        self._loaded_from_ckpt = False

    @torch.no_grad()
    def init_weights(self, init_samples: Optional[torch.Tensor] = None):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        if self.hparams.distr_init == 'gaussian':
            cur = (-1.0 + (0.5**2) * torch.randn(
                self.hparams.dim, self.hparams.num_potentials, self.hparams.num_categories, 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype)
            ) / (self.hparams.num_categories * self.hparams.num_potentials)
            cur = torch.log((cur ** 2).clamp_min(1e-12))  # (D, K, S)
            for d in range(self.hparams.dim):
                self.log_cp_cores[d].data.copy_(cur[d])

        elif self.hparams.distr_init == 'uniform':
            val = torch.log(torch.tensor(
                1.0 / (self.hparams.num_categories * self.hparams.num_potentials), 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            for d in range(self.hparams.dim):
                self.log_cp_cores[d].data.fill_(val)

        elif self.hparams.distr_init == 'samples':
            assert init_samples is not None, "init_samples should not be None when using benchmark samples"
            init_samples = torch.as_tensor(init_samples, device=self.log_alpha.device)
            assert init_samples.dim() == 2 and init_samples.shape == (self.hparams.num_potentials, self.hparams.dim), \
                f"init_samples must be (num_potentials, dim), got {tuple(init_samples.shape)}"

            base_val = torch.log(torch.tensor(
                (1 - self.hparams.sample_prob) / (self.hparams.num_categories - 1),
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            hot_val = torch.tensor(math.log(
                self.hparams.sample_prob), device=self.log_alpha.device, dtype=self.log_alpha.dtype
            )

            for d in range(self.hparams.dim):
                core = torch.full((
                    self.hparams.num_potentials, self.hparams.num_categories), base_val, 
                    device=self.log_alpha.device, dtype=self.log_alpha.dtype
                )
                col_idx = init_samples[:, d].long().view(self.hparams.num_potentials, 1)                  # (K, 1)
                core.scatter_(dim=1, index=col_idx, src=hot_val.expand(self.hparams.num_potentials, 1))   # (K, S)
                self.log_cp_cores[d].data.copy_(core)

        else:
            raise ValueError(f"Invalid distr_init: {self.hparams.distr_init}")

        self._did_weight_init = True

    def load_state_dict(self, state_dict, strict: bool = True):
        ignored = {"c2st.weight", "c2st.bias", "cond_c2st.weight", "cond_c2st.bias"}
        filtered = {k: v for k, v in state_dict.items() if k not in ignored}
        missing, unexpected = LightningModule.load_state_dict(self, filtered, strict=False)

        filtered_out = [k for k in state_dict if k in ignored]
        if filtered_out:
            log.info(f"Ignored keys during load_state_dict: {filtered_out}")
        if missing:
            log.info(f"Missing keys after load (expected): {missing}")
        if unexpected:
            log.info(f"Unexpected keys (ignored by strict=False): {unexpected}")
        return missing, unexpected
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['iteration'] = self.iteration

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._loaded_from_ckpt = True
        if 'iteration' in checkpoint:
            self.iteration = checkpoint['iteration']

    def setup(self, stage: Literal['fit', 'validate', 'test']):
        if stage in (None, "fit") and not self._did_weight_init and not self._loaded_from_ckpt:
            init_samples = None
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'benchmark'):
                benchmark = self.trainer.datamodule.benchmark
                init_samples: torch.Tensor = benchmark.sample_input(self.hparams.num_potentials)
                init_samples = init_samples.flatten(start_dim=1) # (num_potentials, dim)
            self.init_weights(init_samples)

    def get_transition_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = torch.empty(
            x_t.shape[0], self.hparams.num_potentials, self.hparams.dim, device=self.device
        ) # [B, K, D]
        for d in range(self.hparams.dim):
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_t[:, d]) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], dim=-1 # [B, K, S] 
            ) # [B, K]
        sum_log_u_t = log_u_t.sum(dim=-1)  # [B, K]

        transition_logits = torch.empty(
            x_t.shape[0], self.hparams.dim, self.hparams.num_categories, device=self.device
        ) # [B, D, S]
        x_tp1_d = torch.arange(self.hparams.num_categories, device=self.device) # [S]
        x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
        tp1_repeated = tp1.repeat_interleave(self.hparams.num_categories) # [B*S]
        for d in range(self.hparams.dim):
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], dim=-1 # [B*S, K, S]
            )  # [B*S, K]
            log_u_tp1_d = log_u_tp1_d.reshape(x_t.shape[0], self.hparams.num_categories, self.hparams.num_potentials) # [B, S, K]
            log_u_tp1_d = log_u_tp1_d.permute(0, 2, 1) # [B, K, S]      
            log_phi_tp1_d = torch.logsumexp(
                self.log_alpha[None, :, None] + log_u_tp1_d + (sum_log_u_t - log_u_t[:, :, d])[:, :, None], dim=1 # [B, K, S]
            ) # [B, S]
            transition_logits[:, d, :] = log_phi_tp1_d + self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) # [B, S]
        return transition_logits.reshape(*input_shape, self.hparams.num_categories) # [B, ..., S]

    def get_log_v(self, x_end: torch.Tensor) -> torch.Tensor:
        x_end = x_end.flatten(start_dim=1)[:, :, None, None].expand(-1, -1, self.hparams.num_potentials, -1) # (B, D, K, 1)
        log_cp_cores = torch.stack(list(self.log_cp_cores), dim=0) # (D, K, S)
        log_cp_cores = log_cp_cores[None, :, :, :].expand(x_end.shape[0], -1, -1, -1) # (B, D, K, S)
        log_r = torch.gather(log_cp_cores, dim=-1, index=x_end) # (B, D, K, 1)
        log_r = log_r.squeeze(-1).sum(dim=1) # (B, K)
        return torch.logsumexp(self.log_alpha[None, :] + log_r, dim=1) # (B)

    def get_log_c(self, x_start: torch.Tensor) -> torch.Tensor:
        x_start = x_start.flatten(start_dim=1)
        log_z = torch.zeros(x_start.shape[0], self.hparams.num_potentials, device=self.device)
        for d in range(self.hparams.dim):
            log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
            log_z = log_z + torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :], dim=2
            ) # (1, K, S) + (B, 1, S) -> (B, K)
        return torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(1, K) + (B, K) -> (B)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        outputs['loss'] = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'train/loss': outputs['loss'], 
            f'train/log_v': log_v.mean(), 
            f'train/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return outputs

    def on_train_epoch_end(self) -> None:
        self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'val/loss': loss, 
            f'val/log_v': log_v.mean(), 
            f'val/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'test/loss': loss, 
            f'test/log_v': log_v.mean(), 
            f'test/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return outputs

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer = self.hparams.optimizer(
            params=[self.log_alpha] + list(self.log_cp_cores)
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    @torch.no_grad()
    def markov_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        return_transitions: bool = False
    ) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)

        t_orig = t  # keep original for onestep
        t   = self.hparams.num_timesteps + 1 - t_orig
        tp1 = self.hparams.num_timesteps - t_orig

        log_u_t = torch.empty(x_t.shape[0], self.hparams.num_potentials, self.hparams.dim, device=self.device)
        for d in range(self.hparams.dim):
            x_d = x_t[:, d] # [B]
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_d) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], dim=-1
            ) # [B, K]

        log_w_k = self.log_alpha[None, :] + log_u_t.sum(dim=-1) # [B, K]
        k_star = gumbel_sample(log_w_k, tau=self.hparams.tau, dim=-1)  # [B]

        logits = torch.empty(x_t.shape[0], self.hparams.dim, self.hparams.num_categories, device=self.device)
        for d in range(self.hparams.dim):
            x_tp1_d = torch.arange(self.hparams.num_categories, device=self.device) # [S]
            x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
            tp1_repeated = tp1.repeat_interleave(self.hparams.num_categories) # [B*S]
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], # [B*S, K, S]
                dim=-1
            )  # [B*S, K]
            log_u_tp1_d = (log_u_tp1_d
                .reshape(x_t.shape[0], self.hparams.num_categories, self.hparams.num_potentials)
                .permute(0, 2, 1) # [B, K, S]
            )

            batch_idx = torch.arange(x_t.shape[0], device=self.device)
            log_u_tp1_star = log_u_tp1_d[batch_idx, k_star, :] # [B, S]
            logits_x_tp1_d = self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) + log_u_tp1_star # [B, S]
            logits[:, d, :] = logits_x_tp1_d
            
        x_tp1 = gumbel_sample(logits, tau=self.hparams.tau, dim=-1)
        if return_transitions:
            # TODO: Optimize logits computation
            return x_tp1.reshape(input_shape), self.get_transition_logits(x_t, t_orig)
        return x_tp1.reshape(input_shape)

    @torch.no_grad()
    def sample(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = True
    ) -> torch.Tensor:
        """Sample from the model starting from `x` returning the final sample."""
        if use_onestep_sampling:
            input_shape = x_start.shape
            x_start = x_start.flatten(start_dim=1) # (B, D)

            log_z = torch.zeros(x_start.shape[0], self.hparams.num_potentials, device=self.device)
            for d in range(self.hparams.dim):
                log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
                log_z = log_z + torch.logsumexp(
                    self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :], dim=2
                ) # (1, K, S) + (B, 1, S) -> (B, K)

            log_w_k = self.log_alpha[None, :] + log_z # (B, K)
            k_star = gumbel_sample(log_w_k, dim=-1, tau=self.hparams.tau) # (B,)

            logits = torch.empty(x_start.shape[0], self.hparams.dim, self.hparams.num_categories, device=self.device)
            for d in range(self.hparams.dim):
                log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
                log_cp_cores_d = self.log_cp_cores[d][None, :, :].expand(x_start.shape[0], -1, -1) # (B, K, S)
                log_cp_cores_d_selected = torch.gather(
                    log_cp_cores_d, dim=1, index=k_star[:, None, None].expand(-1, -1, self.hparams.num_categories)
                ).squeeze(1) # (B, 1, S) -> (B, S)
                log_p_d_selected = log_cp_cores_d_selected + log_pi_ref[:, :] # (B, S)
                logits[:, d, :] = log_p_d_selected
            x_end = gumbel_sample(
                logits, dim=-1, tau=self.hparams.tau
            ).reshape(input_shape)
        else:

            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
                t = torch.full([x_start.shape[0]], t, device=self.device)
                x_t = self.markov_sample(x_t, t, return_transitions=False)
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
            x_t = x_start
            for t in range(0, self.hparams.num_timesteps + 1):
                t = torch.full([x_t.shape[0]], t, device=self.device)
                out = self.markov_sample(x_t, t, return_transitions=return_transitions)
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
