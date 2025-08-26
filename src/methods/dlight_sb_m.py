from typing import Any, Dict, List, Literal, Optional, Tuple

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
    'dim', 'num_potentials', 
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
        distr_init: Literal['uniform', 'gaussian'] = 'gaussian', 
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.bidirectional = False  
        self.iteration = 1
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.ParameterList()

        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        for _ in range(dim):
            if distr_init == 'gaussian':
                cur_core = (-1.0 + 0.5**2 * torch.randn(num_potentials, prior.num_categories)) \
                   / (prior.num_categories * num_potentials)
                cur_log_core = torch.log(cur_core ** 2)
            elif distr_init == 'uniform':
                cur_log_core = torch.log(
                    torch.ones(num_potentials, prior.num_categories) /
                    (prior.num_categories * num_potentials)
                )
            else:
                raise ValueError(f'Invalid distr_init: {distr_init}')
            self.log_cp_cores.append(nn.Parameter(cur_log_core))

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
        t_orig = t  # keep original for onestep
        t = self.prior.num_timesteps + 1 - t_orig
        tp1 = self.prior.num_timesteps - t_orig

        log_u_t = torch.empty(
            x_t.shape[0], 
            self.hparams.num_potentials, 
            self.hparams.dim, 
            device=self.device
        ) # [B, K, D]
        for d in range(self.hparams.dim):
            x_t_d = x_t[:, d]  # [B]
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_t_d) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], # [B, K, S]
                dim=-1
            ) # fill with [B, K]
        sum_log_u_t = log_u_t.sum(dim=-1)  # [B, K]

        transition_logits = torch.empty(
            x_t.shape[0], 
            self.hparams.dim, 
            self.prior.num_categories, 
            device=self.device
        ) # [B, D, S]
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
            
            log_phi_tp1_d = torch.logsumexp(
                self.log_alpha[None, :, None] + log_u_tp1_d + (sum_log_u_t - log_u_t[:, :, d])[:, :, None], # [B, K, S]
                dim=1
            ) # [B, S]
            transition_logits[:, d, :] = log_phi_tp1_d \
                + self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) # fill with [B, S]
        return transition_logits # [B, D, S]

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
        optimizer  = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler},
        return {'optimizer': optimizer}

    @torch.no_grad()
    def markov_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        return x_tp1

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
