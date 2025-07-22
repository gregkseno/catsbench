from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import numpy as np
from torch.distributions.log_normal import LogNormal

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.dirichlet import Dirichlet
from utils import get_means
from prior import Prior


HPARAMS = (
    'dim', 'num_potentials', 'distr_init', 
    'optimizer', 'scheduler'
)

class LightSB_D(nn.Module):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        distr_init: Literal['benchmark_gaussian'] = 'benchmark_gaussian', 
        device = 'cpu'
    ):
        super().__init__()

        self.prior = prior
        self.dim   = dim  
        self.num_potentials = num_potentials
        self.num_categories = prior.num_categories 
        self.device         = device   
        
        self.log_alpha = nn.Parameter(torch.log(torch.ones(self.num_potentials)/self.num_potentials)
)
        self._initialize_cores(distr_init)

    def _initialize_cores(
        self, distr_init: Literal['benchmark_gaussian']
    ) -> None:
        
        if distr_init == 'benchmark_gaussian_old':
            spread = 1
            
            means = torch.randint(2, self.num_categories - 2, (self.num_potentials, self.dim))#.unsqueeze(-1).repeat(1, self.dim) #get_means(self.dim, self.num_potentials) #(K, D)
            stds = [spread * torch.ones(self.dim) for _ in range(self.num_potentials)] # (K, D)
            stds = torch.stack(stds, dim=0)

            dists = [Normal(loc=means[k], scale=stds[k]) for k in range(self.num_potentials)]
            y_d = [torch.arange(0, self.num_categories )]*self.dim 
            y_d = torch.stack(y_d, dim=1)

            log_cp_cores = [dist.log_prob(y_d) for dist in dists] #(K, D, S)
            #print(log_cp_cores[0].shape)

        if distr_init == 'benchmark_gaussian':
            spread = 1
        
            means = torch.randint(2, self.num_categories - 2, (self.num_potentials, self.dim))  # (K, D)
            stds = torch.full((self.num_potentials, self.dim), spread)                          # (K, D)

            # y_d: all category values across D dims, shape (num_categories, dim)
            y_d = torch.arange(self.num_categories).view(self.num_categories, 1).repeat(1, self.dim)  # (S, D)

            # Reshape for broadcasting
            # y_d: (1, S, D), means: (K, 1, D), stds: (K, 1, D)
            y_d = y_d.unsqueeze(0)          # (1, S, D)
            means = means.unsqueeze(1)      # (K, 1, D)
            stds = stds.unsqueeze(1)        # (K, 1, D)

            # Manual Gaussian log-prob: (K, S, D)
            log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi)) \
                        - torch.log(stds) \
                        - 0.5 * ((y_d - means) / stds) ** 2

            # Permute to (D, K, S) if needed for CP cores
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1)

        elif distr_init == 'categorical':
            spread = 1
            
            means = torch.randint(2, self.num_categories - 2, (self.num_potentials, self.dim))#.unsqueeze(-1).repeat(1, self.dim) #get_means(self.dim, self.num_potentials) #(K, D)
            stds = [spread * torch.ones(self.dim) for _ in range(self.num_potentials)] # (K, D)
            stds = torch.stack(stds, dim=0)
            #print(stds.shape)
            #print(means[0].shape)
            dists = [Normal(loc=means[k], scale=stds[k]) for k in range(self.num_potentials)]
            #cp_cores = [torch.abs(dists[k].sample((self.num_categories,))) for k in range(self.num_potentials)]
            y_d = [torch.arange(0, self.num_categories )]*self.dim 
            y_d = torch.stack(y_d, dim=1)
            #y = torch.cartesian_prod(y_d, y_d)
            #print(y.shape)
            log_cp_cores = [dist.log_prob(y_d) for dist in dists] #(K, D, S)
            print(log_cp_cores[0].shape)
            #log_cp_cores = [torch.log(core) for core in cp_cores]

        elif distr_init == 'uniform':
            log_cp_cores = [torch.log(torch.ones((self.num_categories, self.dim))/(self.num_categories * self.dim))]*self.num_potentials

        #elif distr_init == 'dirichlet':
        #    concentrations = torch.ones((self.num_categories, self.dim))/(self.num_categories * self.dim)
        #    log_cp_cores = [Dirichlet(concentrations).sample(self.num) for ]

        #self.log_cp_cores = torch.stack(log_cp_cores, dim=1).permute(2, 1, 0)
        #print(self.log_cp_cores.shape)
        #self._make_model_parameters()


    def _initialize_parameters_old(
        self, distr_init: Literal['uniform', 'gaussian', 'benchmark']
    ) -> None:
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        self.log_cp_cores = []

        means = get_means
        for d in range(self.dim):
            
            if distr_init == 'gaussian':
                cur_core = (-1.0 + 0.5**2*torch.randn(self.num_potentials, self.num_categories)) \
            / (self.num_categories * self.num_potentials)
                cur_log_core = torch.log(cur_core**2)
                #self.log_cp_cores.append(nn.Parameter(cur_log_core))

            elif distr_init == 'uniform':
                cur_log_core = torch.log(torch.ones(self.num_potentials, self.num_categories) \
                                        / (self.num_categories * self.num_potentials))
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")
            
            cur_log_core = cur_log_core.to(self.device)
            self.log_cp_cores.append(nn.Parameter(cur_log_core))
        self._make_model_parameters()

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(x.shape[0],), 
                fill_value=self.prior.num_timesteps, 
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
    
        y_samples = torch.zeros(x.shape[0], self.dim, dtype=torch.long, device=self.device)
    
        for d in range(self.dim):
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
            out = torch.stack([x, self.sample(x)], dim=0)
        else:
            out = np.stack([pca.transform(x), pca.transform(self.sample(x).cpu())], axis=0)
        return out


            #spread = 0.6
            #means = get_means(self.dim, self.num_potentials)
            #covs = torch.eye(self.dim).repeat(self.num_potentials, 1, 1) * (spread ** 2)
##
            #probs = torch.ones(self.num_potentials) / self.num_potentials
            #mix = Categorical(probs=probs)
            #comp = MultivariateNormal(loc=means, covariance_matrix=covs)
            #gmm = MixtureSameFamily(mix, comp)
#           #
#
            #self.log_cp_cores = gmm.sample((self.num_potentials, self.num_categories))
            #self.log_cp_cores = self.log_cp_cores.permute((2, 0, 1))
#
            #self.log_cp_cores = gmm.sample((self.num_potentials, self.num_categories))
            ##plt.scatter(self.log_cp_cores[0, 0], self.log_cp_cores[0, 1])
            #plt.scatter(self.log_cp_cores[:, 0], self.log_cp_cores[:, 1])
#
            #plt.show()
            #print(self.log_cp_cores.shape)
