from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule

from src.utils.logging.console import RankedLogger
from src.data.prior import Prior


log = RankedLogger(__name__, rank_zero_only=True)

HPARAMS = (
    'dim', 'num_potentials', 'distr_init', 
    'optimizer', 'scheduler'
)

class DLightSB(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian',
        sample_prob: float = 0.9
    ):
        super().__init__()

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

            # NOTE: @Ark-130994 I think it is better to implement using loop for clarity and match the previous initializations
            elif distr_init == 'samples':
                init_samples = benchamrk.sample_target(dim * prior.num_categories)
                cp_cores = torch.full((dim, prior.num_categories), (1 - sample_prob) / (prior.num_categories - 1))
                cp_cores[torch.arange(dim), init_samples] = sample_prob
                self.log_cp_cores = torch.log(cp_cores)

            else:
                raise ValueError(f'Invalid distr_init: {distr_init}')
            self.log_cp_cores.append(nn.Parameter(cur_log_core))

    def get_log_v(self, x_end: torch.Tensor) -> torch.Tensor:
        x_end = x_end.flatten(start_dim=1)
        log_terms = self.log_alpha[None, :]  # (1, K)
        
        for d in range(x_end.shape[1]):
            y_d = x_end[:, d]  # (batch_size,)
            log_r_d = self.log_cp_cores[d][:, y_d].T  # (batch_size, K)
            log_terms = log_terms + log_r_d
            
        log_v = torch.logsumexp(log_terms, dim=1)  # (batch_size,)
        return log_v

    def get_log_c(self, x_start: torch.Tensor) -> torch.Tensor:
        x_start = x_start.flatten(start_dim=1)
        log_z = torch.zeros(x_start.shape[0], self.hparams.num_potentials, device=self.device)
        
        for d in range(self.hparams.dim):
            x_d = x_start[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d)
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c

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
        optimizer  = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler},
        return {'optimizer': optimizer}

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.flatten(start_dim=1)

        log_z = torch.zeros(x.shape[0], self.hparams.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.hparams.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d)
            
            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(x.shape[0], self.hparams.dim, dtype=torch.long, device=self.device)
    
        for d in range(self.hparams.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(x.shape[0], device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        return y_samples.reshape(input_shape)
    
    @torch.no_grad()
    def sample_trajectory(self, x_start: torch.Tensor) -> torch.Tensor:
        x_end = self.sample(x_start)

        trajectory = [x_start]
        for t in range(1, self.prior.num_timesteps + 1):
            t = torch.full((x_start.shape[0],), t, device=x_start.device)
            x_t = self.prior.sample_bridge(x_start, x_end, t)
            trajectory.append(x_t)
        trajectory.append(x_end)
        return torch.stack(trajectory, dim=0)