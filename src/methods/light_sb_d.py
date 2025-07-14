from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule

from src.data.prior import Prior


HPARAMS = (
    'dim', 'num_potentials', 'distr_init', 
    'optimizer', 'scheduler'
)

def create_dimensional_points(
    d: int , min_val: float = 10.0, max_val: float = 40.0, device: str = 'cpu'
):
    base_points = torch.tensor([
        [min_val, min_val, max_val, max_val],  
        [min_val, max_val, max_val, min_val]   
    ], device=device)
    
    full_points = torch.full((d, 4), min_val, device=device)
    full_points[:2] = base_points
    
    if d >= 3:
        full_points[2, 1] = max_val  
        full_points[2, 2] = max_val  
    
    return full_points

class LightSB_D(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'poisson'] = 'uniform', 
    ):
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)        

        self.bidirectional = True
        self.first_iteration = True
        self.ipf_iteration = 1
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self._initialize_parameters(distr_init)
            
    def _initialize_parameters_old(
        self, distr_init: Literal['uniform', 'gaussian', 'poisson']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        for core in self.log_cp_cores:
            if distr_init == 'gaussian':
                nn.init.normal_(core, mean=-1.0, std=0.5)
            elif distr_init == 'uniform':
                nn.init.constant_(core, -torch.log(torch.tensor(self.prior.num_categories * 1.0)))
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")

    def _initialize_parameters(
        self, distr_init: Literal['uniform', 'gaussian', 'poisson']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        self.log_cp_cores = []
        
        if distr_init == 'poisson':
            rates = create_dimensional_points(self.hparams.dim, 10, self.prior.num_categories-10).to(self.device)
            y_d   = torch.arange(self.prior.num_categories, device=self.device)  #(D, S)
            
        for d in range(self.hparams.dim):
            if distr_init == 'gaussian':
                cur_core = (-1.0 + 0.5**2*torch.randn(self.hparams.num_potentials, self.prior.num_categories)) \
                           / (self.prior.num_categories * self.hparams.num_potentials)
                cur_log_core = torch.log(cur_core**2)
            elif distr_init == 'uniform':
                cur_log_core = torch.log(torch.ones(self.hparams.num_potentials, self.prior.num_categories) \
                                        / (self.prior.num_categories * self.hparams.num_potentials))
            elif distr_init == 'poisson':
                rate = rates[d] # (K,)
                cur_log_core = y_d[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(y_d[None, :] + 1) #(K, S)
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")
            
            cur_log_core = cur_log_core.to(self.device)
            self.log_cp_cores.append(nn.Parameter(cur_log_core))
        self._make_model_parameters()

    def _make_model_parameters(self) -> None:
        parameters = []
        for core in self.log_cp_cores:
            parameters.append(core)
        self.parameters = nn.ParameterList(parameters)
                
    def get_log_v(self, y: torch.Tensor) -> torch.Tensor:
        log_terms = self.log_alpha[None, :]  # (1, K)
        
        for d in range(y.shape[1]):
            y_d = y[:, d]  # (batch_size,)
            log_r_d = self.log_cp_cores[d][:, y_d].T  # (batch_size, K)
            log_terms = log_terms + log_r_d
            
        log_v = torch.logsumexp(log_terms, dim=1)  # (batch_size,)
        return log_v

    def get_log_c(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0], x.shape
        log_z = torch.zeros(batch_size, self.hparams.num_potentials, device=self.device)
        
        for d in range(self.hparams.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(batch_size,), 
                fill_value=self.prior.num_timesteps+1, 
                device=self.device
            )
            log_pi_ref = self.prior.extract('cumulative', last_timestep, row_id=x_d)
            
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f"train/loss": loss, 
            f"train/log_v": log_v.mean(), 
            f"train/log_c": log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/ipf_iteration', self.ipf_iteration, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # if self.current_epoch >= self.hparams.num_first_iterations:
        self.ipf_iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f"val/loss": loss, 
            f"val/log_v": log_v.mean(), 
            f"val/log_c": log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/ipf_iteration', self.ipf_iteration, prog_bar=True)

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer  = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler},
        return {'optimizer': optimizer}

    @torch.no_grad()
    def sample(self, x: torch.Tensor, **kwargs) -> torch.Tensor: # kwargs for Logger
        batch_size = x.shape[0]
        log_z = torch.zeros(batch_size, self.hparams.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.hparams.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(batch_size,), 
                fill_value=self.prior.num_timesteps+1, 
                device=self.device
            )
            log_pi_ref = self.prior.extract('cumulative', last_timestep, row_id=x_d)

            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(batch_size, self.hparams.dim, dtype=torch.long, device=self.device)
    
        for d in range(self.hparams.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(batch_size, device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        return y_samples

    def get_log_probs(self) -> None:
        if self.dist_type == 'categorical':
            probs = F.softmax(self.cp_cores, dim=-1)
        
        elif self.hparams.dist_type == 'poisson_old':
            rates = torch.tensor() 
            y = torch.arange(self.num_categories, device=rates.device)
            log_probs = y * torch.log(rates.unsqueeze(-1)) - rates.unsqueeze(-1)
            log_probs -= torch.lgamma(y + 1)
            probs = torch.exp(log_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        elif self.dist_type == 'poisson':
            
            rates = create_dimensional_points(self.hparams.dim, 10, self.num_categories-10).to(self.device)
            
            y     = [torch.arange(self.num_categories, device=self.device) for _ in range(self.hparams.dim)]
            
            for d in range(self.hparams.dim):
                y_d  = y[d]     
                rate = rates[d] 
                log_prob_d = y_d[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(y_d[None, :] + 1) 
                self.log_cp_cores.append(log_prob_d)

        elif self.dist_type == 'negbinomial':
            r = 1 + 9 * torch.sigmoid(self.r_r)  
            p = torch.sigmoid(self.r_p)          
            y = torch.arange(self.num_categories, device=r.device)

            log_binom = torch.lgamma(y + r.unsqueeze(-1)) - torch.lgamma(r.unsqueeze(-1))
            log_binom -= torch.lgamma(y + 1)
            log_probs = log_binom + r.unsqueeze(-1) * torch.log(p.unsqueeze(-1))
            log_probs += y * torch.log(1 - p.unsqueeze(-1))
            probs = torch.exp(log_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        elif self.dist_type == 'bernoulli':
            p = torch.sigmoid(self.cp_cores)  
            probs = torch.stack([1 - p, p], dim=-1)
