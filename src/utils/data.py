from typing import Any, Callable, Literal, Optional, Tuple, Union
import math
import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader


def broadcast(tensor: torch.Tensor, num_add_dims: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += tensor.dim() + 1
    shape = [*tensor.shape[:dim], *([1] * num_add_dims), *tensor.shape[dim:]]
    return tensor.reshape(*shape)

def log_space_product(log_matrix1: torch.Tensor, log_matrix2: torch.Tensor) -> torch.Tensor: 
    # if log_matrix1.is_cuda and log_matrix2.is_cuda:
    #     import genbmm
    #     return genbmm.logbmm(log_matrix1, log_matrix2)
    log_matrix1 = log_matrix1[..., :, None]
    log_matrix2 = log_matrix2[..., None, :, :]
    return torch.logsumexp(log_matrix1 + log_matrix2, dim=-2)

def logits_prod(log_matrix1: torch.Tensor, log_matrix2: torch.Tensor) -> torch.Tensor: 
    # if log_matrix1.is_cuda and log_matrix2.is_cuda:
    #     import genbmm
    #     orig_shape = log_matrix1.shape
    #     log_matrix1 = log_matrix1.reshape(orig_shape[0], -1, orig_shape[-1])
    #     if log_matrix2.dim() == 2:
    #         log_matrix2 = log_matrix2.unsqueeze(0).expand(log_matrix1.shape[0], -1, -1)
    #     else:
    #         if log_matrix2.shape[0] == 1 and log_matrix1.shape[0] != 1:
    #             log_matrix2 = log_matrix2.expand(log_matrix1.shape[0], -1, -1)
    #     out = genbmm.logbmm(log_matrix1, log_matrix2)
    #     return out.reshape(*orig_shape)
    # else:
    log_matrix1 = log_matrix1.unsqueeze(-1) # [batchsize, ..., num_categories, 1]
    insert_nones = [None] * (log_matrix1.ndim - 3)
    idx = (slice(None), *insert_nones, slice(None), slice(None))
    log_matrix2 = log_matrix2[idx] # [batchsize, ..., num_categories, num_categories]
    return torch.logsumexp(log_matrix1 + log_matrix2, dim=-2)

def gumbel_sample(logits: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    finfo = torch.finfo(logits.dtype)
    noise = torch.rand_like(logits)
    noise = torch.clamp(noise, min=finfo.tiny, max=1. - finfo.eps)
    gumbel_noise = -torch.log(-torch.log(noise))
    return torch.argmax((logits / tau) + gumbel_noise, dim=dim)

def convert_to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x
    
def convert_to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return x

def continuous_to_discrete(
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

def make_infinite_dataloader(dataloader: DataLoader[Any]) -> Any:
    while True:
        yield from dataloader

class CoupleDataset(Dataset):
    """A dataset that couples two datasets together, allowing for paired sampling."""
    def __init__(self, input_dataset: torch.Tensor, target_dataset: torch.Tensor):
        self.input_dataset, self.target_dataset = input_dataset, target_dataset
        self.len_input, self.len_target = len(input_dataset), len(target_dataset)
        self.length = max(self.len_input, self.len_target)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.input_dataset[idx % self.len_input],
                self.target_dataset[idx % self.len_target])

class InfiniteCoupleDataset(IterableDataset):
    """A dataset that couples two datasets together, allowing for infinite paired sampling."""
    def __init__(
        self, 
        batch_size: int, 
        sample_input: Callable, 
        sample_target: Callable
    ):
        self.batch_size = batch_size
        self.sample_input = sample_input
        self.sample_target = sample_target

    def __iter__(self):
        while True:
            yield self.sample_input(self.batch_size), self.sample_target(self.batch_size)

def optimize_coupling(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutes batches of data to optimize the coupling between them using Euclidian distance."""
    # get optimal transport coupling between two batches
    input_shape, input_dtype = x.shape[1:], x.dtype
    x = x.flatten(start_dim=1).float()
    y = y.flatten(start_dim=1).float()
    a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    M = torch.cdist(x, y) ** 2
    pi: np.ndarray = ot.emd(a, b, M.detach().cpu().numpy()) # type: ignore
    
    # sample mapping
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    x = x[i].reshape(-1, *input_shape).to(dtype=input_dtype)
    y = y[j].reshape(-1, *input_shape).to(dtype=input_dtype)
    return x, y
