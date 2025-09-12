from typing import Mapping, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.distributed as dist


def broadcast(tensor: torch.Tensor, num_add_dims: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += tensor.dim() + 1
    shape = [*tensor.shape[:dim], *([1] * num_add_dims), *tensor.shape[dim:]]
    return tensor.reshape(*shape)

def log_space_product(log_matrix1: torch.Tensor, log_matrix2: torch.Tensor) -> torch.Tensor: 
    log_matrix1 = log_matrix1[..., :, None]
    log_matrix2 = log_matrix2[..., None, :, :]
    return torch.logsumexp(log_matrix1 + log_matrix2, dim=-2)    

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
        num_categories - 1,
        device=batch.device
    )
    discrete_batch = torch.bucketize(batch, bin_edges)
    return discrete_batch

def sample_separated_means(
    num_potentials: int, 
    dim: int, 
    num_categories: int, 
    min_dist: float = 5,
    max_attempts: int = 5000,
    device: str ='cpu'
):
    means = []
    attempts = 0
    low, high = 5, num_categories - 5

    while len(means) < num_potentials and attempts < max_attempts:
        candidate = torch.randint(low, high, (dim,), device=device)
        if all(torch.norm(candidate - m.float()) >= min_dist for m in means):
            means.append(candidate)
        attempts += 1

    if len(means) < num_potentials:
        raise RuntimeError(f"Could only generate {len(means)} points with min_dist={min_dist}")
    
    return torch.stack(means)

class Logger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger using torch.distributed."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """
        Initializes a multi-GPU-friendly python logger.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to log only from rank zero process.
        :param extra: Optional dict with contextual info (passed to LoggerAdapter).
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    @staticmethod
    def _get_rank() -> int:
        """Get current process rank. Defaults to 0 if torch.distributed is not initialized."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _rank_prefixed_message(msg: str, rank: int) -> str:
        """Prefix log message with rank for clarity in multi-GPU setup."""
        return f"[Rank {rank}] {msg}"

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """
        Delegate a log call to the underlying logger, with rank prefix.

        :param level: Logging level.
        :param msg: Message to log.
        :param rank: Specific rank to log from. If None, logs from all ranks (or rank 0 if `rank_zero_only=True`).
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = self._get_rank()
            msg = self._rank_prefixed_message(msg, current_rank)

            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None or current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
