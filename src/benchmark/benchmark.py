
from typing import Literal
from samplers import (
    DiscreteGaussianDataset, 
    DiscreteUniformDataset,
)
import torch
from torch.utils.data import DataLoader
from utils import LoaderSampler

import os
from benchmark.prior import Prior


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
        save_path: str = './data',
    ):
        self.dim = dim
        self.num_potentials = num_potentials
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        )
        
        # NOTE: @Ark-130994, can we make the benchmark files with some tree directory structure
        # e.g. `./data/benchmark/dim_{dim}/P0_{input_dist}/prior_{prior_type}/num_categories_{num_categories}/alpha_{alpha}`?
        config_name_solver = f'D_benchmark_dim_{dim}_P0_{input_dist}_prior_{prior_type}_num_categories_{num_categories}_alpha_{alpha}.pth'
        config_name_source = f'X0_benchmark_dim_{dim}_P0_{input_dist}_prior_{prior_type}_num_categories_{num_categories}_alpha_{alpha}.pt'
        config_name_target = f'X1_benchmark_dim_{dim}_P0_{input_dist}_prior_{prior_type}_num_categories_{num_categories}_alpha_{alpha}.pt'
        self.solver_path = os.path.join(save_path, config_name_solver)
        self.source_path = os.path.join(save_path, config_name_source)
        self.target_path = os.path.join(save_path, config_name_target)
        
        if not os.path.exists(self.source_path) or not os.path.exists(self.target_path) or not os.path.exists(self.solver_path):
            print('Computing benchmark...')
            if input_dist == 'gaussian':
                self.input_dataset = DiscreteGaussianDataset(num_samples=num_samples, dim=dim, num_categories=num_categories, train=True)
            elif input_dist == 'uniform':
                self.input_dataset = DiscreteUniformDataset(num_samples=num_samples, dim=dim, num_categories=num_categories, train=True)
            else:
                raise ValueError(f'Unknown input distribution: {input_dist}')
            
            print('Sampling target points...')
            self.target_dataset = self.D.sample(self.input_dataset)

            #########################################################################################
            # TODO: @Ark-130994, please, change the saving of cores, to work without LightSB_D class
            print('Saving benchmark...')
            torch.save(self.D.state_dict(), self.solver_path)
            torch.save(self.input_dataset, self.source_path)
            torch.save(self.target_dataset, self.target_path)
            #########################################################################################
        else:
            print('Loading saved solver and benchmark pairs...')
            #########################################################################################
            # TODO: @Ark-130994, please, rework the loading of cores, to work without LightSB_D class
            state_dict = torch.load(self.solver_path)
            self.log_cp_cores = state_dict['log_cp_cores']
            self.log_alpha = state_dict['log_alpha']
            #########################################################################################

            self.input_dataset  = torch.load(self.source_path)
            self.target_dataset = torch.load(self.target_path) 
                    
        # NOTE: what is this?
        random_indices      = torch.randperm(len(self.input_dataset))
        self.input_dataset  = self.input_dataset[random_indices]
        self.target_dataset = self.target_dataset[random_indices]

    #########################################################################################
    # TODO: @Ark-130994, please, make the samplers work without batch size.
    # the batch_size must be specified in sample methods
        input_dataloader  = DataLoader(self.input_dataset, batch_size=128, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=128, shuffle=False)
        self.input_sampler  = LoaderSampler(input_dataloader, self.device)
        self.target_sampler = LoaderSampler(target_dataloader, self.device)

    def sample_input(self, n_samples: int) -> torch.Tensor:
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples: int) -> torch.Tensor:
        return self.target_sampler.sample(n_samples)
    #########################################################################################
    
    def sample_target_given_input(self, x: torch.Tensor) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            last_timestep = torch.full(
                size=(x.shape[0],), 
                fill_value=self.prior.num_timesteps, 
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
    
        y_samples = torch.zeros(x.shape[0], self.dim, dtype=torch.long)
    
        for d in range(self.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(x.shape[0], device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        return y_samples