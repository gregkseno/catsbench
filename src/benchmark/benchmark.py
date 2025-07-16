import torch
from torch.utils.data import DataLoader
from src.utils.data import LoaderSampler
import os
from src.data.prior import Prior
from src.methods.light_sb_d import LightSB_D

class BenchmarkDiscreteEOT():
    def __init__(self, source_dist, dim, prior, num_potentials=4, compute_benchmark=True, X0_full=None,
                 batch_size=512, save_path=None, device='cpu'):

        super().__init__()
        self.X0_full           = X0_full
        self.dim               = dim
        self.source_dist       = source_dist
        self.num_potentials    = num_potentials
        self.batch_size        = batch_size
        self.prior             = prior
        self.compute_benchmark = compute_benchmark
        self.device            = device

        if compute_benchmark is False:
            self.X0_full = torch.load(f'benchmark_data/X0_benchmark_dim_{dim}_P0_{source_dist}.pt')
            self.X1_full = torch.load(f'benchmark_data/X1_benchmark_dim_{dim}_P0_{source_dist}_prior_{prior.prior_type}_beta_{prior.beta}.pt')
            
        else:
            #print(X0_full)
            #assert X0_full is None, 'X0_full must be provided to get new benchmarks...'
            D     = LightSB_D(prior=prior, dim=dim, num_potentials=num_potentials, distr_init='benchmark', optimizer=None).to(device)
    
            self.X1_full = D.sample(self.X0_full).cpu()

            if save_path is not None:
                torch.save(self.X1_full, os.path.join(save_path, f'X1_benchmark_dim_{dim}_P0_{source_dist}_prior_{prior.prior_type}_beta_{prior.alpha}.pt'))

        X0_dataloader   = DataLoader(self.X0_full, batch_size=batch_size, shuffle=False)
        self.X0_sampler = LoaderSampler(X0_dataloader, self.device)
            
        X1_dataloader   = DataLoader(self.X1_full, batch_size=batch_size, shuffle=False)
        self.X1_sampler = LoaderSampler(X1_dataloader, self.device)

    def sample_X0(self, n_samples):
        return self.X0_sampler.sample(n_samples)

    def sample_X1(self, n_samples):
        return self.X1_sampler.sample(n_samples)
    
    def sample_X1_given_X0(self, X0):
        D = LightSB_D(prior=self.prior, dim=self.dim, num_potentials=self.num_potentials, distr_init='benchmark', optimizer=None).to(self.device)
        
        X1 = D.sample(X0).cpu()
        return X1