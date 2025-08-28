from typing import Literal, Optional, Tuple
import os

import torch
from torch import nn

from benchmark.prior import Prior
from benchmark.stylegan2 import legacy, dnnlib
from benchmark.utils import  continuous_to_discrete, sample_separated_means, Logger


log = Logger(__name__, rank_zero_only=True)
SPREADS = {50:{2:1.5, 16:1.5, 64:2.5}, 200:{2:4, 16:8, 64:16}}

class BenchmarkBase:
    dim: int
    num_potentials: int
    prior: Prior
    log_alpha: torch.Tensor
    log_cp_cores: torch.Tensor
    input_dataset: torch.Tensor
    target_dataset: torch.Tensor

    def _get_log_cp_cores(
        self, 
        benchmark_type: Literal['gaussian_mixture', 'log_gaussian', 'uniform'], 
        spread: float = 15.0
    ) -> torch.Tensor:
        if benchmark_type == 'gaussian_mixture':
            means = sample_separated_means(
                self.num_potentials, self.dim, self.prior.num_categories, min_dist=10
            ).unsqueeze(1)
            stds = torch.full((self.num_potentials, self.dim), spread).unsqueeze(1) # (K, D)
            y_d = torch.arange(
                self.prior.num_categories
            ).view(self.prior.num_categories, 1).repeat(1, self.dim).unsqueeze(0)  # (S, D)
            log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi)) - torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2
        
        elif benchmark_type == 'log_gaussian':
            mu = torch.zeros(self.dim)          
            sigma = torch.ones(self.dim) * 0.5  

            log_normal = torch.distributions.LogNormal(mu, sigma)
            log_cp_cores: torch.Tensor = log_normal.sample((self.num_potentials, self.prior.num_categories,)) # type: ignore

        elif benchmark_type == 'uniform':
            log_cp_cores = torch.rand(size=(self.num_potentials, self.prior.num_categories, self.dim))     

        else:
            raise ValueError(f'Unknown benchmark type: {benchmark_type}')

        return log_cp_cores
    
    @torch.no_grad()
    def sample_target_given_input(self, x: torch.Tensor, return_trajectories: bool = False) -> torch.Tensor:
        log_z = torch.zeros(x.shape[0], self.num_potentials, device=x.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d).to(x.device)
            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(x.shape[0], self.dim, dtype=torch.long, device=x.device)
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
            return torch.stack([x, y_samples], dim=0)
    
    def save(
        self, solver_path: str, source_path: str, target_path: str, dir: str
    ):
        log.info(f'Saving benchmark to {dir}...')
        os.makedirs(dir, exist_ok=True)
        torch.save({'log_alpha': self.log_alpha.cpu(), 'log_cp_cores': self.log_cp_cores.cpu()}, solver_path)
        torch.save(self.input_dataset.cpu(), source_path)
        torch.save(self.target_dataset.cpu(), target_path)

    def load(
        self, solver_path: str, source_path: str, target_path: str, dir: str
    ):
        log.info(f'Loading saved solver and benchmark pairs from {dir}...')
        log_params = torch.load(solver_path, map_location=torch.device('cpu'))
        self.log_alpha = log_params['log_alpha']
        self.log_cp_cores = log_params['log_cp_cores']

        self.input_dataset  = torch.load(source_path, map_location=torch.device('cpu'))
        self.target_dataset = torch.load(target_path, map_location=torch.device('cpu')) 

    @property
    def device(self) -> torch.device:
        return self.log_alpha.device
    
    def to(self, device: torch.device):
        self.log_alpha = self.log_alpha.to(device)
        self.log_cp_cores = self.log_cp_cores.to(device)
        if hasattr(self, 'generator'):
            self.generator = self.generator.to(device)

class Benchmark(BenchmarkBase):
    def __init__(
        self, 
        dim: int,
        num_potentials: int,
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
        ] = 'uniform',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        input_dist: Literal['gaussian', 'uniform'] = 'gaussian',
        save_path: str = '../data/benchmark',
    ):
        super().__init__()
        self.dim = dim
        self.num_potentials = num_potentials
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        )
        self.input_dist = input_dist

        benchmark_dir = f"{save_path}/dim_{dim}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        solver_path = os.path.join(benchmark_dir, f'D_P0_{input_dist}.pth') 
        source_path = os.path.join(benchmark_dir, f'X0_P0_{input_dist}.pt')
        target_path = os.path.join(benchmark_dir, f'X1_P0_{input_dist}.pt')
        if os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(solver_path):
            self.load(solver_path, source_path, target_path, benchmark_dir)

        else:
            log.info('Loading parameters...')
            self.log_alpha = torch.log(torch.ones(self.num_potentials) / self.num_potentials)
            self.log_cp_cores = self._get_log_cp_cores(
                benchmark_type, spread=SPREADS[self.prior.num_categories][dim]
            ).permute(2, 0, 1) # (D, K, S)

            log.info('Sampling validation dataset...')
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            self.input_dataset = self.sample_input(num_val_samples)
            self.target_dataset = self.sample_target_given_input(self.input_dataset)

            random_indices = torch.randperm(len(self.target_dataset))
            self.input_dataset  = self.input_dataset[random_indices]
            self.target_dataset = self.target_dataset[random_indices]

            self.save(solver_path, source_path, target_path, benchmark_dir)

    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        '''Sample independent source data'''
        if self.input_dist == 'gaussian':
            samples = continuous_to_discrete(
                torch.randn(size=[num_samples, self.dim], device=self.device), 
                self.prior.num_categories
            )
        elif self.input_dist == 'uniform':
            samples = continuous_to_discrete(
                6 * torch.rand(size=(num_samples, self.dim), device=self.device) - 3,
                self.prior.num_categories
            )
        else:
            raise ValueError(f'Unknown input distribution: {self.input_dist}')
        return samples
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        '''Sample independent target data'''
        input_samples = self.sample_input(num_samples)
        target_samples = self.sample_target_given_input(input_samples)
        return target_samples
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Sample paired input and target data'''
        input_samples = self.sample_input(num_samples)
        target_samples = self.sample_target_given_input(input_samples)
        return input_samples, target_samples

class BenchmarkImages(BenchmarkBase):
    generator: nn.Module

    def __init__(
        self, 
        num_potentials: int,
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
        ] = 'gaussian',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        generator_path: str = '../checkpoints/cmnist_stylegan2.pkl',
        save_path: str = '../data/benchmark_images',
    ):
        super().__init__()
        self.dim = 3 * 32 * 32
        self.num_potentials = num_potentials
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        )
        self._load_generator(generator_path)
        
        benchmark_dir = f"{save_path}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        solver_path = os.path.join(benchmark_dir, f'D_c0_{benchmark_type}.pth') 
        source_path = os.path.join(benchmark_dir, f'X0_c0_{benchmark_type}.pt')
        target_path = os.path.join(benchmark_dir, f'X1_c0_{benchmark_type}.pt')
        if os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(solver_path):
            self.load(solver_path, source_path, target_path, benchmark_dir)
           
        else:
            log.info('Loading parameters...')
            self.log_alpha = torch.log(torch.ones(self.num_potentials) / self.num_potentials)
            self.log_cp_cores = self._get_log_cp_cores(benchmark_type, spread=15).permute(2, 0, 1) # (D, K, S)

            log.info('Sampling validation dataset...')
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            samples_per_batch = 2000
            num_batches = num_val_samples // samples_per_batch
            self.input_dataset = torch.empty((num_batches * samples_per_batch, 3, 32, 32), dtype=torch.int)
            self.target_dataset = torch.empty((num_batches * samples_per_batch, 3, 32, 32), dtype=torch.int)
            for i in range(num_batches):
                noise = torch.randn((samples_per_batch, 512), device=self.device)
                start, end = samples_per_batch * i, samples_per_batch * (i + 1)
                self.input_dataset[start:end] = self._postporcess(self.generator(noise, None))
                self.target_dataset[start:end] = self.sample_target_given_input(
                    torch.flatten(self.input_dataset[start:end], start_dim=1)
                ).reshape_as(self.input_dataset[start:end])

            self.save(solver_path, source_path, target_path, benchmark_dir)

    @staticmethod
    def _postporcess(outputs: torch.Tensor) -> torch.Tensor:
        return ((outputs *0.5 + 0.5).clamp(0, 1) * 255).to(torch.int)

    def _load_generator(self, generator_path: str):
        log.info('Loading StyleGAN2 generator checkpoint...')
        with dnnlib.util.open_url(generator_path) as f:
            self.generator: nn.Module = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    # NOTE: Here we have reversed setup:
    #       - Input: CMNIST images;
    #       - Target: noised CMNIST images.
    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        return input_samples
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        target_samples = self.sample_target_given_input(torch.flatten(input_samples, start_dim=1))
        return target_samples.reshape_as(input_samples)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int)-> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        target_samples = self.sample_target_given_input(torch.flatten(input_samples, start_dim=1))
        return input_samples, target_samples.reshape_as(input_samples)
