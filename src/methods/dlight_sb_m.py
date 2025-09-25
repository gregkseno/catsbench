from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule

from src.data.prior import Prior
from src.utils import optimize_coupling
from src.utils.logging.console import RankedLogger


HPARAMS = (
    'dim', 'num_potentials', 'sample_prob',
    'use_mini_batch', 'distr_init',
    'kl_loss_coeff', 'mse_loss_coeff', 
    'optimizer', 'scheduler'
)
log = RankedLogger(__name__, rank_zero_only=True)


class DLightSB_M(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        kl_loss_coeff: float = 1.0,
        mse_loss_coeff: float = 0.0,
        use_mini_batch: bool = False,
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian', 
        sample_prob: float = 0.9
    ) -> None:
        super().__init__()
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.ParameterList([
            nn.Parameter(torch.empty(
                num_potentials, prior.num_categories,
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
                self.hparams.dim, self.hparams.num_potentials, self.prior.num_categories, 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype)
            ) / (self.prior.num_categories * self.hparams.num_potentials)
            cur = torch.log((cur ** 2).clamp_min(1e-12))  # (D, K, S)
            for d in range(self.hparams.dim):
                self.log_cp_cores[d].data.copy_(cur[d])

        elif self.hparams.distr_init == 'uniform':
            val = torch.log(torch.tensor(
                1.0 / (self.prior.num_categories * self.hparams.num_potentials), 
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
                (1 - self.hparams.sample_prob) / (self.prior.num_categories - 1),
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            hot_val = torch.tensor(math.log(
                self.hparams.sample_prob), device=self.log_alpha.device, dtype=self.log_alpha.dtype
            )

            for d in range(self.hparams.dim):
                core = torch.full((
                    self.hparams.num_potentials, self.prior.num_categories), base_val, 
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._loaded_from_ckpt = True

    def setup(self, stage: Literal['fit', 'validate', 'test']):
        if stage in (None, "fit") and not self._did_weight_init and not self._loaded_from_ckpt:
            assert hasattr(self.trainer, 'datamodule'), "Trainer has no datamodule attribute"
            assert hasattr(self.trainer.datamodule, 'benchmark'), "Datamodule has no benchmark attribute"
            init_samples: torch.Tesnor = self.trainer.datamodule.benchmark.sample_input(self.hparams.num_potentials)
            init_samples = init_samples.flatten(start_dim=1) # (num_potentials, dim)
            self.init_weights(init_samples)

    def kl_loss(
        self,
        true_q_posterior_logits: torch.Tensor, 
        pred_q_posterior_logits: torch.Tensor,
    ) -> torch.Tensor:        
        '''KL-divergence calculation.'''
        kl_loss = torch.softmax(true_q_posterior_logits, dim=-1) * (
            torch.log_softmax(true_q_posterior_logits, dim=-1)
            - torch.log_softmax(pred_q_posterior_logits, dim=-1)
        )
        kl_loss = kl_loss.sum(dim=-1).mean()
        return kl_loss
        
    def mse_loss(
        self,
        true_q_posterior_logits: torch.Tensor, 
        pred_q_posterior_logits: torch.Tensor,
    ) -> torch.Tensor:        
        '''MSE calculation.'''
        mse_loss = F.mse_loss(
            torch.softmax(true_q_posterior_logits, dim=-1), 
            torch.softmax(pred_q_posterior_logits, dim=-1)
        )
        return mse_loss 

    def get_sb_transition_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.prior.num_timesteps + 1 - t_orig
        tp1 = self.prior.num_timesteps - t_orig

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
            x_t.shape[0], self.hparams.dim, self.prior.num_categories, device=self.device
        ) # [B, D, S]
        x_tp1_d = torch.arange(self.prior.num_categories, device=self.device) # [S]
        x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
        tp1_repeated = tp1.repeat_interleave(self.prior.num_categories) # [B*S]
        for d in range(self.hparams.dim):
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], dim=-1 # [B*S, K, S]
            )  # [B*S, K]
            log_u_tp1_d = log_u_tp1_d.reshape(x_t.shape[0], self.prior.num_categories, self.hparams.num_potentials) # [B, S, K]
            log_u_tp1_d = log_u_tp1_d.permute(0, 2, 1) # [B, K, S]      
            log_phi_tp1_d = torch.logsumexp(
                self.log_alpha[None, :, None] + log_u_tp1_d + (sum_log_u_t - log_u_t[:, :, d])[:, :, None], dim=1 # [B, K, S]
            ) # [B, S]
            transition_logits[:, d, :] = log_phi_tp1_d + self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) # [B, S]
        return transition_logits.reshape(*input_shape, self.prior.num_categories) # [B, ..., S]

    def optimal_projection(
        self,
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=0, high=self.prior.num_timesteps + 1,
            size=(batch_size,), device=self.device
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)

        true_q_posterior_logits = self.prior.posterior_logits_reverse(true_x_end, x_t, t, logits=False)
        pred_q_transition_logits = self.get_sb_transition_logits(x_t, t=t)
        pred_q_transition_logits = pred_q_transition_logits.log_softmax(dim=-1)

        kl = self.kl_loss(true_q_posterior_logits, pred_q_transition_logits)
        mse = self.mse_loss(true_q_posterior_logits, pred_q_transition_logits)

        loss = self.hparams.kl_loss_coeff * kl + \
               self.hparams.mse_loss_coeff * mse
        
        info = {f'kl_loss': kl, f'mse_loss': mse}
        return loss, info

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch

        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        loss, info = self.optimal_projection(x_start, x_end)
        outputs['loss'] = loss

        info = {f"train/{k}": v for k, v in info.items()}
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

        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)

        _, info = self.optimal_projection(x_start, x_end)
        info = {f"val/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        # if first iteration apply optional mini-batch sampling
        if self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)

        _, info = self.optimal_projection(x_start, x_end)
        info = {f"test/{k}": v for k, v in info.items()}
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
    def markov_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)

        t_orig = t  # keep original for onestep
        t   = self.prior.num_timesteps + 1 - t_orig
        tp1 = self.prior.num_timesteps - t_orig

        log_u_t = torch.empty(x_t.shape[0], self.hparams.num_potentials, self.hparams.dim, device=self.device)
        for d in range(self.hparams.dim):
            x_d = x_t[:, d] # [B]
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_d) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], dim=-1
            ) # [B, K]

        log_w_k = self.log_alpha[None, :] + log_u_t.sum(dim=-1) # [B, K]
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1, keepdim=True) # [B, K]
        k_star = torch.multinomial(log_p_k.exp(), num_samples=1).squeeze(-1) # [B]

        x_tp1 = torch.empty(x_t.shape[0], self.hparams.dim, dtype=torch.long, device=self.device)
        for d in range(self.hparams.dim):
            x_tp1_d = torch.arange(self.prior.num_categories, device=self.device) # [S]
            x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
            tp1_repeated = tp1.repeat_interleave(self.prior.num_categories) # [B*S]
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], # [B*S, K, S]
                dim=-1
            )  # [B*S, K]
            log_u_tp1_d = (log_u_tp1_d
                .reshape(x_t.shape[0], self.prior.num_categories, self.hparams.num_potentials)
                .permute(0, 2, 1) # [B, K, S]
            )

            batch_idx = torch.arange(x_t.shape[0], device=self.device)
            log_u_tp1_star = log_u_tp1_d[batch_idx, k_star, :] # [B, S]
            logits_x_tp1_d = self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) + log_u_tp1_star # [B, S]

            x_tp1[:, d] = torch.multinomial(
                torch.softmax(logits_x_tp1_d, dim=-1), num_samples=1
            ).squeeze(-1)
        return x_tp1.reshape(input_shape)

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample from the model starting from `x` returning the final sample."""
        for t in range(0, self.prior.num_timesteps + 1):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t)
        return x
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Sample from the model starting from `x` returning the full trajectory."""
        trajectory = [x]
        for t in range(0, self.prior.num_timesteps + 1):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory
