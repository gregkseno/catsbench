from typing import Any
import numpy as np
import ot
import torch
from torch.utils.data import Dataset, DataLoader


def broadcast(tensor: torch.Tensor, num_add_dims: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += tensor.dim() + 1
    shape = [*tensor.shape[:dim], *([1] * num_add_dims), *tensor.shape[dim:]]
    return tensor.reshape(*shape)

def convert_to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x
    
def convert_to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return x

def make_infinite_dataloader(dataloader: DataLoader[Any]) -> Any:
    while True:
        yield from dataloader

class CoupleDataset(Dataset):
    """A dataset that couples two datasets together, allowing for paired sampling."""
    def __init__(self, input_dataset: Dataset, target_dataset: Dataset):
        self.input_dataset, self.target_dataset = input_dataset, target_dataset
        self.len_input, self.len_target = len(input_dataset), len(target_dataset)
        self.length = max(self.len_input, self.len_target)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.input_dataset[idx % self.len_input],
                self.target_dataset[idx % self.len_target])


def optimize_coupling(x: torch.Tensor, y: torch.Tensor):
    """Permutes batches of data to optimize the coupling between them using Euclidian distance."""
    # get optimal transport coupling between two batches
    x, y = x.float(), y.float()
    a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    M = torch.cdist(x, y) ** 2
    pi = ot.emd(a, b, M.detach().cpu().numpy())
    
    # sample mapping
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    return x[i], y[j]

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
        
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        if size <= self.loader.batch_size:
            try:
                batch = next(self.it)
            except StopIteration:
                self.it = iter(self.loader)
                return self.sample(size)
            if len(batch) < size:
                return self.sample(size)
                
            return batch[:size].to(self.device)
            
        elif size > self.loader.batch_size:
            samples = []
            cur_size = 0
            
            while cur_size < size:
                try:
                    batch = next(self.it)
                    samples.append(batch)
                    cur_size += batch.shape[0]
                except StopIteration:
                    self.it = iter(self.loader)
                    print(f'Maximum size allowed exceeded, returning {cur_size} samples...')
                    samples = torch.cat(samples, dim=0)
                    return samples[:cur_size].to(self.device)
                
            samples = torch.cat(samples, dim=0)
            return samples[:size].to(self.device)
