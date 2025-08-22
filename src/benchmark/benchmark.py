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

from typing import Literal, Optional
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
from transformers import PreTrainedTokenizerFast

class BenchmarkDiscreteEOTBase:
        
    def get_log_cp_cores(self, benchmark_type, spread=15):
        if benchmark_type == 'gaussian_mixture':
            means = sample_separated_means(self.num_potentials, self.dim, self.prior.num_categories, min_dist=10)
            stds = torch.full((self.num_potentials, self.dim), spread)                          # (K, D)
            y_d = torch.arange(self.prior.num_categories).view(self.prior.num_categories, 1).repeat(1, self.dim)  # (S, D)
            y_d = y_d.unsqueeze(0)          
            means = means.unsqueeze(1)      
            stds = stds.unsqueeze(1)        
            log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi) )- torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2
        
        elif benchmark_type == 'log_gaussian':
            mu = torch.zeros(self.dim)          
            sigma = torch.ones(self.dim) * 0.5  

            log_normal = torch.distributions.LogNormal(mu, sigma)
            log_cp_cores = log_normal.sample((self.num_potentials, self.prior.num_categories,))

        elif benchmark_type == 'uniform':
            log_cp_cores = torch.rand(size=(self.num_potentials, self.prior.num_categories, self.dim))     

        return log_cp_cores
    
    def sample_target(self, num_samples: int) -> torch.Tensor:
        '''Sample independent target data'''
        input_data = self.sample_input(num_samples)
        target_data = self.sample_target_given_input(input_data)
        return target_data
    
    def sample_input_target(self, num_samples: int) -> torch.Tensor:
        '''Sample paired input and target data'''
        input_data = self.sample_input(num_samples)
        target_data = self.sample_target_given_input(input_data)
        return input_data, target_data
    
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

class BenchmarkDiscreteEOT(BenchmarkDiscreteEOTBase):
    def __init__(
        self, 
        alpha: float,
        num_val_samples: int,
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
        save_path: str = ',../data/benchmark',
        benchmark_type: Literal['gaussian_mixture'] = 'gaussian_mixture',
        device = 'cpu',
        save_bench = True
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

        self.input_dist = input_dist
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
            print('Computing validation benchmark pairs...')
            
            self.input_dataset = self.sample_input(num_val_samples).to(device)
            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)
            
            spreads = {50:{2:1.5, 16:1.5, 64:2.5}, 200:{2:4, 16:8, 64:16}}
            log_cp_cores = self.get_log_cp_cores(benchmark_type, spread=spreads[self.prior.num_categories][dim])
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device) #(D, K, S)

            print('Sampling validation target points...')
            self.target_dataset = self.sample_target_given_input(self.input_dataset, return_trajectories=False)

            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }

            random_indices      = torch.randperm(len(self.target_dataset))
            self.input_dataset  = self.input_dataset[random_indices]
            self.target_dataset = self.target_dataset[random_indices]
            if save_bench:
                self.save()

    def sample_input(self, num_samples: int) -> torch.tensor:
        '''Sample independent source data'''
        if self.input_dist == 'gaussian':
            input_samples = DiscreteGaussianDataset(num_samples=num_samples, dim=self.dim, num_categories=self.prior.num_categories, train=True).dataset
        elif self.input_dist == 'uniform':
            input_samples = DiscreteUniformDataset(num_samples=num_samples, dim=self.dim, num_categories=self.prior.num_categories, train=True).dataset
        else:
            raise ValueError(f'Unknown input distribution: {self.input_dist}')
        return input_samples

class BenchmarkDiscreteEOTImages(BenchmarkDiscreteEOT):
    def __init__(
        self, 
        alpha: float,
        dim: int, 
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        num_potentials: int,
        samples_per_digit: int = 5000,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'gaussian_log',
        ] = 'uniform',
        save_path: str = '../data/cmnist',
        benchmark_type: Literal['gaussian_mixture'] = 'gaussian_mixture',
        device='cpu',
        save_bench=True,
        batch_size = 128  # Useful if we want to train iterating over all batches.
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

        elif save_bench:
            print('Computing benchmark...')

            input_samples = []
            
            for ix in range(10):
                samples_digit = DiscreteColoredMNISTDataset(ix, data_dir='../data',  img_size=32).dataset[:samples_per_digit]
                input_samples.append(samples_digit.view(samples_per_digit, -1))
            
            self.input_dataset = torch.cat(input_samples, dim=0)
            print(self.input_dataset.shape)

            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)

            if benchmark_type == 'gaussian_mixture':
                dim = 32*32*3
                means = sample_separated_means(num_potentials, dim, num_categories, min_dist=10)
                stds = torch.full((num_potentials, dim), 15)                          # (K, D)
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
            target_samples = []
            for ix in range(10):
                target_batch = self.sample_target_given_input(self.input_dataset[ix*samples_per_digit:(ix+1)*samples_per_digit], return_trajectories=False)
                target_samples.append(target_batch)

            self.target_dataset = torch.cat(target_samples)
            
            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }

        random_indices      = torch.randperm(len(self.input_dataset))
        self.input_dataset  = self.input_dataset[random_indices]
        self.target_dataset = self.target_dataset[random_indices]

        input_dataloader  = DataLoader(self.input_dataset, batch_size=batch_size, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=False)
        target_dataloader_shuffled = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=True)

        self.input_sampler  = LoaderSampler(input_dataloader)
        self.target_sampler = LoaderSampler(target_dataloader)
        self.target_sampler_shuffled = LoaderSampler(target_dataloader_shuffled)

class BenchmarkDiscreteEOTImagesGenerated(BenchmarkDiscreteEOT):
    def __init__(
        self, 
        generator,
        alpha: float,
        dim: int, 
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        num_potentials: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'gaussian_log',
        ] = 'gaussian',
        save_path: str = '../data/cmnist',
        benchmark_type: Literal['gaussian_mixture'] = 'gaussian_mixture',
        device='cpu',
        precompute: bool = False,
        save_bench: bool = True,
        batch_size = 128  # Useful if we want to train iterating over all batches.
    ):
        self.generator = generator
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
        print('Path: ', self.solver_path)

        if os.path.exists(self.source_path) and os.path.exists(self.target_path) and os.path.exists(self.solver_path):
            print('Loading saved solver and benchmark pairs...')

            self.log_params   = torch.load(self.solver_path, map_location=device)
            self.log_alpha    = self.log_params['log_alpha']
            self.log_cp_cores = self.log_params['log_cp_cores']

            self.input_dataset  = torch.load(self.source_path, map_location=device)
            self.target_dataset = torch.load(self.target_path, map_location=device) 

        elif precompute:
            print('Computing benchmark...')

            noise = torch.randn((100_000, 512)).to(device)
            input_samples = generator(noise)*0.5 + 0.5
            self.input_dataset = (input_samples * 255).to(torch.int32).reshape(-1, self.dim)

            print(self.input_dataset.shape)

            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)
            log_cp_cores = self.get_log_cp_cores(benchmark_type, spread=15)
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device) #(K, S, D) -> (D, K, S)

            print('Sampling target points...')
            target_samples = []
            n_batches = 100000//5000
            for ix in range(n_batches):
                target_batch = self.sample_target_given_input(self.input_dataset[ix*5000:(ix+1)*5000], return_trajectories=False)
                target_samples.append(target_batch)

            self.target_dataset = torch.cat(target_samples, dim=0)
            
            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }
            if save_bench:
                self.save()

        else:
            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)
            log_cp_cores = self.get_log_cp_cores(benchmark_type, spread=15)
            self.log_cp_cores = log_cp_cores.permute(2, 0, 1).to(device) #(K, S, D) -> (D, K, S)
            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }

        try:
            random_indices      = torch.randperm(len(self.input_dataset))
            self.input_dataset  = self.input_dataset[random_indices]
            self.target_dataset = self.target_dataset[random_indices]

            input_dataloader  = DataLoader(self.input_dataset, batch_size=batch_size, shuffle=False)
            target_dataloader = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=False)
            target_dataloader_shuffled = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=True)

            self.input_sampler  = LoaderSampler(input_dataloader)
            self.target_sampler = LoaderSampler(target_dataloader)
            self.target_sampler_shuffled = LoaderSampler(target_dataloader_shuffled)
        except:
            print('No precomputed samples, only generation is possible...')

    def generate_input(self, num_samples: int) -> torch.tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        output = self.generator(noise).reshape(-1, self.dim)
        return output
    
    def generate_target(self, num_samples: int) -> torch.tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        output = self.generator(noise).reshape(-1, self.dim)
        noised_output = self.sample_target_given_input(output)
        return noised_output
    
    def generate_input_target(self, num_samples: int)-> torch.tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        output = self.generator(noise).reshape(-1, self.dim)
        noised_output = self.sample_target_given_input(output)
        return output, noised_output

class BenchmarkDiscreteEOTImagesGenerated_old(BenchmarkDiscreteEOT):
    def __init__(
        self, 
        alpha: float,
        dim: int, 
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        num_potentials: int,
        samples_per_batch: int = 5000,
        gen_idx: bool = 0,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'gaussian_log',
        ] = 'uniform',
        save_path: str = '../data/cmnist',
        benchmark_type: Literal['gaussian_mixture'] = 'gaussian_mixture',
        device='cpu',
        save_bench=True,
        batch_size = 128  # Useful if we want to train iterating over all batches.
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
        
        self.solver_path = self.folder_name + f'D_c0_{benchmark_type}_gen_{gen_idx}.pth'
        self.source_path = self.folder_name + f'X0_c0_{benchmark_type}_gen_{gen_idx}.pt'
        self.target_path = self.folder_name + f'X1_c0_{benchmark_type}_gen_{gen_idx}.pt'
        print('Path: ', self.solver_path)
        if os.path.exists(self.source_path) and os.path.exists(self.target_path) and os.path.exists(self.solver_path):
            print('Loading saved solver and benchmark pairs...')

            self.log_params   = torch.load(self.solver_path, map_location=device)
            self.log_alpha    = self.log_params['log_alpha']
            self.log_cp_cores = self.log_params['log_cp_cores']

            self.input_dataset  = torch.load(self.source_path, map_location=device)
            self.target_dataset = torch.load(self.target_path, map_location=device) 

        else:
            print('Computing benchmark...')

            input_samples = torch.load(f'../ckpts/ddpm_100K_generated_samples_{gen_idx}.pth')
            self.input_dataset = (input_samples * 255).to(torch.int32).reshape(-1, 32*32*3)

            print(self.input_dataset.shape)

            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device)/self.num_potentials)

            if benchmark_type == 'gaussian_mixture':
                dim = 32*32*3
                means = sample_separated_means(num_potentials, dim, num_categories, min_dist=10)
                stds = torch.full((num_potentials, dim), 15)                          # (K, D)
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
            target_samples = []
            n_batches = 100000//samples_per_batch
            for ix in range(n_batches):
                target_batch = self.sample_target_given_input(self.input_dataset[ix*samples_per_batch:(ix+1)*samples_per_batch], return_trajectories=False)
                target_samples.append(target_batch)

            self.target_dataset = torch.cat(target_samples, dim=0)
            
            self.log_params = {
                'log_alpha': self.log_alpha,
                'log_cp_cores': self.log_cp_cores
            }
            if save_bench:
                self.save()

        random_indices      = torch.randperm(len(self.input_dataset))
        self.input_dataset  = self.input_dataset[random_indices]
        self.target_dataset = self.target_dataset[random_indices]

        input_dataloader  = DataLoader(self.input_dataset, batch_size=batch_size, shuffle=False)
        target_dataloader = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=False)
        target_dataloader_shuffled = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=True)

        self.input_sampler  = LoaderSampler(input_dataloader)
        self.target_sampler = LoaderSampler(target_dataloader)
        self.target_sampler_shuffled = LoaderSampler(target_dataloader_shuffled)