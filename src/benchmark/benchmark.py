
from samplers import (
    DiscreteGaussianDataset, 
    DiscreteUniformDataset,
)
import torch
from torch.utils.data import DataLoader
from utils import LoaderSampler
from light_sb_d_sampler import LightSB_D_Sampler

from types import SimpleNamespace 
import os
from prior import Prior


class BenchmarkDiscreteEOT():
    def __init__(self, prior_params, solver_params, dataset_params, save_path, device='cpu'):

        super().__init__()
        prior_space     = SimpleNamespace(**prior_params)
        solver_space    = SimpleNamespace(**solver_params)
        dataset_space   = SimpleNamespace(**dataset_params)
        self.device = device

        config_name_solver = f'D_benchmark_dim_{dataset_space.dim}_P0_{dataset_space.source_dist}_prior_{prior_space.prior_type}_num_categories_{prior_space.num_categories}_alpha_{prior_space.alpha}.pth'
        config_name_source = f'X0_benchmark_dim_{dataset_space.dim}_P0_{dataset_space.source_dist}_prior_{prior_space.prior_type}_num_categories_{prior_space.num_categories}_alpha_{prior_space.alpha}.pt'
        config_name_target = f'X1_benchmark_dim_{dataset_space.dim}_P0_{dataset_space.source_dist}_prior_{prior_space.prior_type}_num_categories_{prior_space.num_categories}_alpha_{prior_space.alpha}.pt'

        self.solver_path = os.path.join(save_path, config_name_solver)
        self.source_path = os.path.join(save_path, config_name_source)
        self.target_path = os.path.join(save_path, config_name_target)
        
        if os.path.exists(self.source_path) is False or os.path.exists(self.target_path) is False or os.path.exists(self.solver_path) is False:
            compute_benchmark = True
        else:
            compute_benchmark = False

        prior  = Prior(**prior_params).to(device)
        self.D = LightSB_D_Sampler(prior=prior, **solver_params, device=device)
        
        if compute_benchmark is False:
            print('Loading saved solver and benchmark pairs...')
            self.D.load_state_dict(torch.load(self.solver_path, weights_only=True))
            self.input_dataset  = torch.load(self.source_path)
            self.target_dataset = torch.load(self.target_path)
            
        else:
            print('Computing benchmark...')
            if dataset_space.source_dist == 'gaussian':
                self.input_dataset = DiscreteGaussianDataset(num_samples=dataset_space.full_size, dim=dataset_space.dim, num_categories=prior_space.num_categories, train=True).dataset
            
            if dataset_space.source_dist == 'uniform':
                self.input_dataset = DiscreteUniformDataset(num_samples=dataset_space.full_size, dim=dataset_space.dim, num_categories=prior_space.num_categories, train=True).dataset
    
            print('Sampling target points...')
            self.target_dataset = self.D.sample(self.input_dataset.to(device)).cpu()                

        random_indices      = torch.randperm(len(self.input_dataset))
        self.input_dataset  = self.input_dataset[random_indices].cpu()
        self.target_dataset = self.target_dataset[random_indices].cpu()
        
        input_dataloader   = DataLoader(self.input_dataset, batch_size=dataset_space.batch_size, shuffle=False)
        self.input_sampler = LoaderSampler(input_dataloader, self.device)
            
        target_dataloader   = DataLoader(self.target_dataset, batch_size=dataset_space.batch_size, shuffle=False)
        self.target_sampler = LoaderSampler(target_dataloader, self.device)

    def sample_input(self, n_samples):
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples):
        return self.target_sampler.sample(n_samples)
    
    def sample_target_given_input(self, x):
        X1 = self.D.sample_trajectory(x).cpu()
        return X1
    
    def save(self):
        print('Saving benchmark...')
        torch.save(self.D.state_dict(), self.solver_path)
        torch.save(self.input_dataset, self.source_path)
        torch.save(self.target_dataset, self.target_path)