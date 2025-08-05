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
                spreads = {2:4, 16:8, 64:16}
                means = sample_separated_means(num_potentials, dim, num_categories, min_dist=10)#
                stds = torch.full((num_potentials, dim), spreads[dim])                          # (K, D)
                y_d = torch.arange(num_categories).view(num_categories, 1).repeat(1, dim)  # (S, D)
                y_d = y_d.unsqueeze(0)          
                means = means.unsqueeze(1)      
                stds = stds.unsqueeze(1)        
                log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi) )- torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2
            
            if benchmark_type == 'uniform':
                log_cp_cores = torch.rand(size=(num_potentials, num_categories, dim))   

            if benchmark_type == 'mnist':
                pass     
            
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device) #(D, K, S)

            print('Sampling target points...')
            self.target_dataset = self.sample_target_given_input(self.input_dataset, return_trajectories=False)

            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }
            # random_indices      = torch.randperm(len(self.input_dataset))
            # self.input_dataset  = self.input_dataset[random_indices]
            random_indices      = torch.randperm(len(self.target_dataset))
            self.target_dataset = self.target_dataset[random_indices]
            if save_bench:
                self.save()

    #########################################################################################
    # TODO: @Ark-130994, please, make the samplers work without batch size.
    # the batch_size must be specified in sample methods
        input_dataloader  = DataLoader(self.input_dataset, batch_size=512, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=512, shuffle=False)
        self.input_sampler  = LoaderSampler(input_dataloader)
        self.target_sampler = LoaderSampler(target_dataloader)

    def sample_input(self, n_samples: int) -> torch.Tensor:
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples: int) -> torch.Tensor:
        return self.target_sampler.sample(n_samples)
    #########################################################################################
    
    @torch.no_grad()
    def sample_target_given_input(self, x: torch.Tensor, return_trajectories: bool = False) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d).to(self.device)
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
        
        if not return_trajectories:
            return y_samples
        else:
            return torch.stack([x.cpu(), y_samples.cpu()], dim=0)
        
    def to(self, device: str) -> None:
        self.device = device
        self.input_dataset = self.input_dataset.to(device)
        self.target_dataset = self.target_dataset.to(device)
        self.log_alpha = self.log_alpha.to(device)
        self.log_cp_cores = self.log_cp_cores.to(device)
        self.prior = self.prior.to(device)
    
    def save(self):
        print(f'Saving benchmark to {self.folder_name}...')
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

        torch.save(self.log_params, self.solver_path)
        torch.save(self.input_dataset, self.source_path)
        torch.save(self.target_dataset, self.target_path)

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
from src.benchmark.samplers import DiscreteColoredMNISTDataset


class BenchmarkDiscreteEOTImages(BenchmarkDiscreteEOT):
    def __init__(
        self, 
        alpha: float,
        dim: int, 
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        num_potentials: int,
        num_samples: int = 10000,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'gaussian_log',
        ] = 'uniform',
        save_path: str = '../data/cmnist',
        benchmark_type: Literal['cmnist'] = 'cmnist',
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
        
        self.folder_name = f"{save_path}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        
        self.solver_path = self.folder_name + f'D_c0_{benchmark_type}.pth'
        self.source_path = self.folder_name + f'X0_c0_{benchmark_type}.pt'
        self.target_path = self.folder_name + f'X1_c0_{benchmark_type}.pt'

        if os.path.exists(self.source_path) and os.path.exists(self.target_path) and os.path.exists(self.solver_path):
            print('Loading saved solver and benchmark pairs...')

            self.log_params   = torch.load(self.solver_path, map_location=device)
            self.log_alpha    = self.log_params['log_alpha']
            self.log_cp_cores = self.log_params['log_cp_cores']

            self.input_dataset  = torch.load(self.source_path)
            self.target_dataset = torch.load(self.target_path) 

        else:
            print('Computing benchmark...')
            
            self.input_dataset = DiscreteColoredMNISTDataset(3, data_dir='../data',  img_size=32).dataset[:num_samples]
            self.input_dataset = self.input_dataset.view(len(self.input_dataset), -1)
            print(self.input_dataset.shape)

            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)

            if benchmark_type == 'gaussian_mixture':
                dim = 32*32*3
                means = sample_separated_means(num_potentials, dim, num_categories, min_dist=10)#
                stds = torch.full((num_potentials, dim), 8)                          # (K, D)
                y_d = torch.arange(num_categories).view(num_categories, 1).repeat(1, dim)  # (S, D)
                y_d = y_d.unsqueeze(0)          
                means = means.unsqueeze(1)      
                stds = stds.unsqueeze(1)        
                log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi) )- torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2
            
            elif benchmark_type == 'log_gaussian':
                mu = torch.zeros(dim)          
                sigma = torch.ones(dim) * 0.5  

                log_normal = torch.distributions.LogNormal(mu, sigma)

                log_cp_cores = log_normal.sample((num_potentials, num_categories,))

            elif benchmark_type == 'uniform':
                log_cp_cores = torch.rand(size=(num_potentials, num_categories, dim))       
            
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device) #(K, S, D) -> (D, K, S)

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
        input_dataloader  = DataLoader(self.input_dataset, batch_size=512, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=512, shuffle=False)
        self.input_sampler  = LoaderSampler(input_dataloader)
        self.target_sampler = LoaderSampler(target_dataloader)

    def sample_input(self, n_samples: int) -> torch.Tensor:
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples: int) -> torch.Tensor:
        return self.target_sampler.sample(n_samples)
    #########################################################################################
    
    @torch.no_grad()
    def sample_target_given_input(self, x: torch.Tensor, return_trajectories: bool = False) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d).to(self.device)
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
        
        if not return_trajectories:
            return y_samples
        else:
            return torch.stack([x.cpu(), y_samples.cpu()], dim=0)
        
    def to(self, device: str) -> None:
        self.device = device
        self.input_dataset = self.input_dataset.to(device)
        self.target_dataset = self.target_dataset.to(device)
        self.log_alpha = self.log_alpha.to(device)
        self.log_cp_cores = self.log_cp_cores.to(device)
        self.prior = self.prior.to(device)
    
    def save(self):
        print(f'Saving benchmark to {self.folder_name}...')
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

        torch.save(self.log_params, self.solver_path)
        torch.save(self.input_dataset, self.source_path)
        torch.save(self.target_dataset, self.target_path)