from typing import Any
import numpy as np
import ot
import torch
from torch.utils.data import Dataset, DataLoader


def broadcast(t: torch.Tensor, num_add_dims: int) -> torch.Tensor:
    shape = [t.shape[0]] + [1] * num_add_dims
    return t.reshape(shape)

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
    def __init__(self, dataset_0, dataset_1):
        self.dataset_0, self.dataset_1 = dataset_0, dataset_1
        self.len_0, self.len_1 = len(dataset_0), len(dataset_1)
        self.length = max(self.len_0, self.len_1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.dataset_0[idx % self.len_0],
                self.dataset_1[idx % self.len_1])


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
