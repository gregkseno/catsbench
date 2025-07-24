from typing import Literal
from .samplers import (
    DiscreteGaussianDataset, 
    DiscreteUniformDataset,
)
import torch
from torch.utils.data import DataLoader
from .utils import LoaderSampler, sample_separated_means

import os
from .prior import Prior
from pathlib import Path

class BenchmarkDiscreteEOT:
    def __init__(
        self, 
        alpha: float,
        num_samples: int,
        dim: int,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        num_potentials: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'gaussian_log',
        ] = 'uniform',
        input_dist: Literal['gaussian', 'uniform'] = 'gaussian',
        save_path: str = '../data/benchmark',
        benchmark_type: Literal['gaussian_mixture'] = 'gaussian_mixture',
        device='cpu',
        save_bench=True
    ):
        self.dim = dim
        self.num_potentials = num_potentials
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)
        self.save_path = save_path
        self.device    = device
        
        self.folder_name = f"{save_path}/dim_{dim}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        self.solver_path = self.folder_name + f'D_P0_{input_dist}.pth'
        self.source_path = self.folder_name + f'X0_P0_{input_dist}.pt'
        self.target_path = self.folder_name + f'X1_P0_{input_dist}.pt'
        
        if os.path.exists(self.source_path) and os.path.exists(self.target_path) and os.path.exists(self.solver_path):
            print('Loading saved solver and benchmark pairs...')

            self.log_params   = torch.load(self.solver_path, map_location=device)
            self.log_alpha    = self.log_params['log_alpha']
            self.log_cp_cores = self.log_params['log_cp_cores']

            self.input_dataset  = torch.load(self.source_path)
            self.target_dataset = torch.load(self.target_path) 

        else:
            print('Computing benchmark...')
            if input_dist == 'gaussian':
                self.input_dataset = DiscreteGaussianDataset(num_samples=num_samples, dim=dim, num_categories=num_categories, train=True).dataset
            elif input_dist == 'uniform':
                self.input_dataset = DiscreteUniformDataset(num_samples=num_samples, dim=dim, num_categories=num_categories, train=True).dataset
            else:
                raise ValueError(f'Unknown input distribution: {input_dist}')
            
            self.input_dataset = self.input_dataset.to(device)
            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)

            if benchmark_type == 'gaussian_mixture':
                
                #spreads = {2:2, 16:4, 64:8}
                spreads = {2:4, 16:8, 64:16}

                means = sample_separated_means(num_potentials, dim, num_categories, min_dist=10)#
                #means = torch.randint(5, num_categories-7, (num_potentials, dim))
                stds = torch.full((num_potentials, dim), spreads[dim])                          # (K, D)
                y_d = torch.arange(num_categories).view(num_categories, 1).repeat(1, dim)  # (S, D)
                y_d = y_d.unsqueeze(0)          
                means = means.unsqueeze(1)      
                stds = stds.unsqueeze(1)        
                log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi) )- torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2
            
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device)

            print('Sampling target points...')

            self.target_dataset = self.sample_target_given_input(self.input_dataset, return_trajectories=False)

            self.log_params = {
                               'log_alpha': self.log_alpha,
                               'log_cp_cores': self.log_cp_cores
                              }
            if save_bench:
                self.save()

                    
        # NOTE: what is this?
        random_indices      = torch.randperm(len(self.input_dataset))
        self.input_dataset  = self.input_dataset[random_indices]
        self.target_dataset = self.target_dataset[random_indices]

    #########################################################################################
    # TODO: @Ark-130994, please, make the samplers work without batch size.
    # the batch_size must be specified in sample methods
        input_dataloader  = DataLoader(self.input_dataset, batch_size=128, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=128, shuffle=False)
        self.input_sampler  = LoaderSampler(input_dataloader)
        self.target_sampler = LoaderSampler(target_dataloader)

    def sample_input(self, n_samples: int) -> torch.Tensor:
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples: int) -> torch.Tensor:
        return self.target_sampler.sample(n_samples)
    #########################################################################################
    
    def sample_target_given_input(self, x: torch.Tensor, return_trajectories: bool = False) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(x.shape[0],), 
                fill_value=self.prior.num_timesteps, 
                device=self.device
            )
            log_pi_ref = self.prior.extract('cumulative', last_timestep, row_id=x_d).to(self.device)
            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(x.shape[0], self.dim, dtype=torch.long)
    
        for d in range(self.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(x.shape[0], device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        if return_trajectories is False:
            return y_samples
        else:
            return torch.stack([x, y_samples], dim=0)
    
    def save(self):
        print(f'Saving benchmark to {self.folder_name}...')
        #Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

        torch.save(self.log_params, self.solver_path)
        torch.save(self.input_dataset, self.source_path)
        torch.save(self.target_dataset, self.target_path)