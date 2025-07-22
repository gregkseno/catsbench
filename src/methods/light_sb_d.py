from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule
import numpy as np
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from src.utils import RankedLogger
from src.data.prior import Prior


log = RankedLogger(__name__, rank_zero_only=True)


def get_means(dim: int, num_clusters: int = 5, min_separation: float = 8, seed: int = 43):
    torch.manual_seed(seed)
    means_hd = torch.zeros((num_clusters, dim))
#
    #for k in range(1, num_clusters):
    #    candidate = torch.empty(dim)
    #    candidate.uniform_(-2, 2)
    #    means_hd[k] = candidate
#
    #return means_hd

    for k in range(1, num_clusters):
        candidate = torch.empty(dim)
        valid = False
        for _ in range(1000):  # Max 1000 trials
            candidate.uniform_(-5, 5) #(-15, 15) for 2 (-10, 10) for 16 (-5, 5) for 64
            # Calculate distances to existing means
            dists = torch.norm(means_hd[:k] - candidate, dim=1)
            if torch.all(dists >= min_separation):
                means_hd[k] = candidate
                valid = True
                break
        if not valid:
            raise RuntimeError(f"Couldn't place cluster {k}")
    
    return means_hd

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
        distr_init: Literal['uniform', 'gaussian', 'benchmark'] = 'gaussian', 
    ):
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.bidirectional = False  
        self.iteration = 1
        self.prior = prior
        
        # TODO: Add names to parameters
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        #self.log_alpha = nn.Parameter(torch.log(torch.ones(self.hparams.num_potentials)/self.hparams.num_potentials))
        self._initialize_parameters_old(distr_init)
            
    def _initialize_parameters(
        self, distr_init: Literal['uniform', 'gaussian', 'benchmark']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        if distr_init == 'gaussian':
            spread = 0.6
            rb, lb = 5, -5
            means = torch.rand((self.hparams.num_potentials, self.hparams.dim))* (rb - lb) + lb
            #get_means(self.dim, self.num_potentials) #(K, D)
            stds = [spread * torch.ones(self.hparams.dim) for _ in range(self.hparams.num_potentials)] # (K, D)
            
            dists = [Normal(loc=means[k], scale=stds[k]) for k in range(self.hparams.num_potentials)]
            cp_cores = [torch.abs(dists[k].sample((self.prior.num_categories,))) for k in range(self.hparams.num_potentials)]
            log_cp_cores = [torch.log(core) for core in cp_cores]

        elif distr_init == 'uniform':
            log_cp_cores = [torch.log(torch.ones((self.prior.num_categories, self.hparams.dim))/(self.prior.num_categories * self.hparams.dim))]*self.hparams.num_potentials

        #elif distr_init == 'dirichlet':
        #    concentrations = torch.ones((self.num_categories, self.dim))/(self.num_categories * self.dim)
        #    log_cp_cores = [Dirichlet(concentrations).sample(self.num) for ]

        #self.log_cp_cores = [log_cp_core.to(self.device) for log_cp_core in log_cp_cores]
        #print(self.device)
        self.log_cp_cores = torch.stack(log_cp_cores, dim=1).permute(2, 1, 0).cuda()
        #print(self.log_cp_cores.device)
        self._make_model_parameters()

    def _initialize_parameters_old_old(
        self, distr_init: Literal['uniform', 'gaussian', 'benchmark']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        self.log_cp_cores = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hparams.num_potentials, self.prior.num_categories))
            for _ in range(self.hparams.dim)
        ])

        for core in self.log_cp_cores:
            if distr_init == 'gaussian':
                nn.init.normal_(core, mean=-1.0, std=0.5)
            elif distr_init == 'uniform':
                nn.init.constant_(core, -torch.log(torch.tensor(self.prior.num_categories * 1.0)))
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")

    def _initialize_parameters_old(
        self, distr_init: Literal['uniform', 'gaussian', 'benchmark']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        self.log_cp_cores = []
        
        if distr_init == 'benchmark':
            rates = create_dimensional_points(self.hparams.dim, 10, self.prior.num_categories-10).to(self.device)
            y_d   = torch.arange(self.prior.num_categories, device=self.device)  #(D, S)
        if distr_init == 'poisson':
            rates = torch.randint(5, 50, (self.hparams.dim, self.hparams.num_potentials))
            y_d   = torch.arange(self.prior.num_categories, device=self.device)  #(D, S)

        for d in range(self.hparams.dim):
            
            if distr_init == 'gaussian':
                cur_core = (-1.0 + 0.5**2*torch.randn(self.hparams.num_potentials, self.prior.num_categories)) \
            / (self.prior.num_categories * self.hparams.num_potentials)
                cur_log_core = torch.log(cur_core**2)
                #self.log_cp_cores.append(nn.Parameter(cur_log_core))

            elif distr_init == 'uniform':
                cur_log_core = torch.log(torch.ones(self.hparams.num_potentials, self.prior.num_categories) \
                                        / (self.prior.num_categories * self.hparams.num_potentials))
            
            elif distr_init in ['benchmark', 'poisson']:
                rate = rates[d] # (K,)
                cur_log_core = y_d[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(y_d[None, :] + 1) #(K, S)
            #elif distr_init == 'dirichlet':
            #    cur_log_core = 
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
        
        log_z = torch.zeros(x.shape[0], self.hparams.num_potentials, device=self.device)
        
        for d in range(self.hparams.dim):
            x_d = x[:, d]

            last_timestep = torch.full(
                size=(x.shape[0],), 
                fill_value=self.prior.num_timesteps, 
                device=self.device
            )
            log_pi_ref = self.prior.extract('cumulative', last_timestep, row_id=x_d)
            #log_pi_ref = torch.log(pi_ref)
            
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c
    
    def init_by_samples(self, samples, device):
        marginals = []
        num_samples = samples.shape[0]

        for dim in range(self.hparams.dim):
            dim_samples = samples[:, dim]
            
            counts = torch.histc(dim_samples.float(),  # Ensure float type for histc
                                bins=self.prior.num_categories,
                                min=0,
                                max=self.prior.num_categories-1)
            
            probs = (counts + 1e-8) / (num_samples + 1e-8 * self.prior.num_categories)
            marginals.append(probs.log())  # Store log probabilities

        self.log_cp_cores = nn.ParameterList()
        for dim in range(self.hparams.dim):
            base = torch.zeros(self.hparams.num_potentials, self.prior.num_categories)
            
            base[0] = marginals[dim]
            
            base[1:] = torch.randn(self.hparams.num_potentials - 1, 
                                self.prior.num_categories) * 0.01
            
            self.log_cp_cores.append(nn.Parameter(base.to(device)))

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
            f"train/loss": outputs['loss'], 
            f"train/log_v": log_v.mean(), 
            f"train/log_c": log_c.mean()
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
            f"val/loss": loss, 
            f"val/log_v": log_v.mean(), 
            f"val/log_c": log_c.mean()
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
            f"test/loss": loss, 
            f"test/log_v": log_v.mean(), 
            f"test/log_c": log_c.mean()
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
        log_z = torch.zeros(x.shape[0], self.hparams.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.hparams.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(x.shape[0],), 
                fill_value=self.prior.num_timesteps, 
                device=self.device
            )
            log_pi_ref = self.prior.extract('cumulative', last_timestep, row_id=x_d)
            #log_pi_ref = torch.log(pi_ref)
            
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
        
        return y_samples
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor, pca=None) -> torch.Tensor:
        if pca is None:
            out = torch.stack([x, self.sample(x).cpu()], dim=0)
        else:
            out = np.stack([pca.transform(x), pca.transform(self.sample(x).cpu())], axis=0)
        return out

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule
import numpy as np

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


HPARAMS = (
    'dim', 'num_potentials', 'distr_init', 
    'optimizer', 'scheduler'
)

class AutoregressiveLightSB_D(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'benchmark'] = 'gaussian', 
    ):
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.iteration = 1
        self.prior = prior
        
        # TODO: Add names to parameters
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self._initialize_parameters(distr_init)
            
        self.autoregressive_cores = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hparams.num_potentials * d, 128),
                nn.ReLU(),
                nn.Linear(128, self.hparams.num_potentials * self.prior.num_categories)
            ) for d in range(1, self.hparams.dim)
        ])
                
    def get_log_v(self, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        log_terms = self.log_alpha[None, :]  # (1, K)
        
        # First dimension (no autoregressive dependency)
        y0 = y[:, 0]
        log_r0 = self.log_cp_cores[0][:, y0].T  # (batch_size, K)
        log_terms = log_terms + log_r0
        
        # Autoregressive processing for subsequent dimensions
        for d in range(1, self.hparams.dim):
            # Gather previous dimensions' outputs
            prev_dims = y[:, :d]
            
            # Encode previous dimensions
            prev_flat = prev_dims.view(batch_size, -1).float()
            ar_out = self.autoregressive_cores[d-1](prev_flat)
            ar_out = ar_out.view(batch_size, self.hparams.num_potentials, self.prior.num_categories)
            
            # Combine with current dimension
            yd = y[:, d]
            log_rd = self.log_cp_cores[d][:, yd].T  # Base term
            log_ar = ar_out[torch.arange(batch_size), :, yd]  # Autoregressive term
            
            log_terms = log_terms + log_rd + log_ar
            
        return torch.logsumexp(log_terms, dim=1)
    
    def get_log_c(self, x: torch.Tensor) -> torch.Tensor:
        # Requires sequential integration - use ancestral sampling
        num_samples = 1000  # Number of samples for Monte Carlo integration
        device = x.device
        
        # Initialize with first dimension
        log_probs = torch.zeros(x.shape[0], device=device)
        y_samples = torch.zeros(x.shape[0], self.hparams.dim, device=device, dtype=torch.long)
        
        # Sample first dimension
        with torch.no_grad():
            timestep = torch.full((x.shape[0],), self.prior.num_timesteps, device=device)
            pi_ref = self.prior.extract('cumulative', timestep, row_id=x[:, 0])
            y_samples[:, 0] = Categorical(pi_ref).sample()
        
        # Autoregressive sampling for subsequent dimensions
        for d in range(1, self.hparams.dim):
            # Build conditioning input
            prev_dims = y_samples[:, :d]
            prev_flat = prev_dims.view(x.shape[0], -1).float()
            
            # Get autoregressive output
            ar_out = self.autoregressive_cores[d-1](prev_flat)
            ar_out = ar_out.view(x.shape[0], self.hparams.num_potentials, self.prior.num_categories)
            
            # Combine with base term
            combined = (self.log_cp_cores[d][None, :, :] + 
                       ar_out).logsumexp(dim=1)  # (batch_size, num_categories)
            
            # Sample next dimension
            with torch.no_grad():
                # Get prior transition probabilities
                timestep = torch.full((x.shape[0],), self.prior.num_timesteps, device=device)
                pi_ref = self.prior.extract('cumulative', timestep, row_id=x[:, d])
                
                # Combine model and prior
                log_prob = combined + torch.log(pi_ref)
                y_samples[:, d] = Categorical(torch.exp(log_prob)).sample()
        
        # Compute log_v for all sampled paths
        log_v_samples = self.get_log_v(y_samples)
        
        # Monte Carlo estimate of log_C
        log_C = torch.logsumexp(log_v_samples, dim=0) - torch.log(torch.tensor(num_samples))
        return log_C
    