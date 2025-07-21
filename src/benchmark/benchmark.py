import torch
from torch.utils.data import DataLoader
from src.utils.data import LoaderSampler
import os
from src.data.prior import Prior
from src.methods.light_sb_d import LightSB_D

class BenchmarkDiscreteEOT():
    def __init__(self, source_dist, dim, prior, num_potentials=4, compute_benchmark=True, input_dataset=None,
                 batch_size=512, save_path=None, device='cpu'):

        super().__init__()
        self.input_dataset     = input_dataset
        self.dim               = dim
        self.source_dist       = source_dist
        self.num_potentials    = num_potentials
        self.batch_size        = batch_size
        self.prior             = prior
        self.compute_benchmark = compute_benchmark
        self.device            = device

        if compute_benchmark is False:
            self.input_dataset = torch.load(f'benchmark_data/X0_benchmark_dim_{dim}_P0_{source_dist}.pt')
            self.target_dataset = torch.load(f'benchmark_data/X1_benchmark_dim_{dim}_P0_{source_dist}_prior_{prior.prior_type}_beta_{prior.beta}.pt')
            
        else:
            D = LightSB_D(prior=prior, dim=dim, num_potentials=num_potentials, distr_init='benchmark', optimizer=None).to(device)
    
            self.target_dataset = D.sample(self.input_dataset).cpu()

            if save_path is not None:
                torch.save(self.target_dataset, os.path.join(save_path, f'X1_benchmark_dim_{dim}_P0_{source_dist}_prior_{prior.prior_type}_beta_{prior.alpha}.pt'))

        input_dataloader   = DataLoader(self.input_dataset, batch_size=batch_size, shuffle=False)
        self.input_sampler = LoaderSampler(input_dataloader, self.device)
            
        target_dataloader   = DataLoader(self.target_dataset, batch_size=batch_size, shuffle=False)
        self.target_sampler = LoaderSampler(target_dataloader, self.device)

    def sample_input(self, n_samples):
        return self.input_sampler.sample(n_samples)

    def sample_target(self, n_samples):
        return self.target_sampler.sample(n_samples)
    
    def sample_target_given_input(self, x):
        D = LightSB_D(prior=self.prior, dim=self.dim, num_potentials=self.num_potentials, distr_init='benchmark', optimizer=None).to(self.device)
        
        X1 = D.sample(x).cpu()
        return X1