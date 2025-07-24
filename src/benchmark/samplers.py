from typing import Any, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
from torch.utils.data import Dataset, DataLoader

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal



def _continuous_to_discrete(
    batch: Union[torch.Tensor, np.ndarray], 
    num_categories: int,
    quantize_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
):
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).contiguous()
    if quantize_range is None:
        quantize_range = (-3, 3)
    bin_edges = torch.linspace(
        quantize_range[0], 
        quantize_range[1], 
        num_categories - 1
    )
    discrete_batch = torch.bucketize(batch, bin_edges)
    return discrete_batch

class DiscreteUniformDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):
        dataset = 6 * torch.rand(size=(num_samples, dim)) - 3
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset  

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteGaussianDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):          
        dataset = torch.randn(size=[num_samples, dim])
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteMixtureGaussianDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_potentials: int = 4, num_categories: int = 100, spread: float = 0.8
    ):                  
        means = get_means(dim, num_potentials)
        covs = torch.eye(dim).repeat(num_potentials, 1, 1) * (spread ** 2)

        probs = torch.ones(num_potentials) / num_potentials
        mix = Categorical(probs=probs)
        comp = MultivariateNormal(loc=means, covariance_matrix=covs)
        gmm = MixtureSameFamily(mix, comp)

        continuous_samples = gmm.sample((num_samples,))

        samples = _continuous_to_discrete(continuous_samples, num_categories=num_categories, quantize_range=(-15, 15))
        self.dataset = samples

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
class DiscreteSwissRollDataset(Dataset):
    def __init__(
            self, num_samples: int, noise: float = 0.8, num_categories: int = 100, train: bool = True
        ):
        dataset = make_swiss_roll(
            n_samples=num_samples,
            noise=noise
        )[0][:, [0, 2]]  / 7.5
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset   

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

#class Sampler:
#    def __init__(
#        self, device='cuda',
#    ):
#        self.device = device
#    
#    def sample(self, size=5):
#        pass
#        
#class LoaderSampler(Sampler):
#    def __init__(self, loader, device='cpu'):
#        super(LoaderSampler, self).__init__(device)
#        self.loader = loader
#        self.it = iter(self.loader)
#        
#    def __len__(self):
#        return len(self.loader)
#    
#    def reset_sampler(self):
#        self.it = iter(self.loader)
#        
#    def sample(self, size=5):
#        if size <= self.loader.batch_size:
#            try:
#                batch = next(self.it)
#            except StopIteration:
#                self.it = iter(self.loader)
#                return self.sample(size)
#            if len(batch) < size:
#                return self.sample(size)
#                
#            return batch[:size].to(self.device)
#            
#        elif size > self.loader.batch_size:
#            samples = []
#            cur_size = 0
#            
#            while cur_size < size:
#                try:
#                    batch = next(self.it)
#                    samples.append(batch)
#                    cur_size += batch.shape[0]
#                except StopIteration:
#                    self.it = iter(self.loader)
#                    print(f'Maximum size allowed exceeded, returning {cur_size} samples...')
#                    samples = torch.cat(samples, dim=0)
#                    return samples[:cur_size].to(self.device)
#                
#            samples = torch.cat(samples, dim=0)
#            return samples[:size].to(self.device)
