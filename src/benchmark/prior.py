from typing import Literal, Optional, Tuple, Union

from math import log
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
from benchmark.utils import broadcast, log_space_product


def get_cum_matrices(num_timesteps: int, log_onestep_matrix: torch.Tensor) -> torch.Tensor:
    num_categories = log_onestep_matrix.shape[0]
    log_cum_matrices = torch.empty(size=(num_timesteps, num_categories, num_categories), dtype=log_onestep_matrix.dtype)
    
    # Add identity matrix for the zero timestep
    log_identity = torch.full((num_categories, num_categories), float('-inf'), dtype=torch.float64)  # log(0) = -inf
    rows = torch.arange(num_categories)
    log_identity[rows, rows] = 0.0
    log_cum_matrices[0] = log_identity[:]

    for timestep in tqdm(range(1, num_timesteps)):
        log_cum_matrices[timestep] = log_space_product(log_cum_matrices[timestep-1], log_onestep_matrix)
    assert log_onestep_matrix.shape == log_cum_matrices[0].shape, f'Wrong shape!'
    return log_cum_matrices

def uniform_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    log_equal = log(1 - alpha)                  # log((1 - alpha)/(num_categories - 1) + 1e-20)
    log_diff  = log(alpha/(num_categories - 1)) # log(alpha + (1 - alpha)/(num_categories - 1) + 1e-20)

    log_p_onestep_mat = torch.full((num_categories, num_categories), log_diff)  
    rows = torch.arange(num_categories).to(torch.int32)
    log_p_onestep_mat[rows, rows] = log_equal

    log_p_onestep_mat = get_cum_matrices(num_skip_steps + 1, log_p_onestep_mat)[-1]
    log_p_cum_mats = get_cum_matrices(num_timesteps + 2, log_p_onestep_mat)

    return log_p_onestep_mat.transpose(0, 1), log_p_cum_mats

def uniform_power(num_timesteps, S, gamma, dtype=torch.float32):
    
    alpha = 1 - (gamma * S) / (S - 1)
    alpha_N = alpha ** num_timesteps
    
    diag_value = torch.tensor(alpha_N + (1 - alpha_N) / S, dtype=dtype)
    off_diag_value = torch.tensor((1 - alpha_N) / S, dtype=dtype)
    
    log_matrix = torch.full((S, S), torch.log(off_diag_value), dtype=dtype)

    indices = torch.arange(S)
    log_matrix[indices, indices] = torch.log(diag_value)
    
    return log_matrix

def get_cum_matrices_uniform(num_timesteps: int, num_categories, gamma, num_skip_steps=1) -> torch.Tensor:
    
    log_cum_matrices = torch.empty(size=(num_timesteps, num_categories, num_categories), dtype=torch.float32)
    
    log_identity = torch.full((num_categories, num_categories), float('-inf'), dtype=torch.float64)  # log(0) = -inf
    rows = torch.arange(num_categories)
    log_identity[rows, rows] = 0.0
    log_cum_matrices[0] = log_identity[:]

    for timestep in range(1, num_timesteps):
        log_cum_matrices[timestep] = uniform_power(timestep*num_skip_steps, num_categories, gamma)
    return log_cum_matrices
    
def uniform_closed_form_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
):

    log_p_onestep_mat = get_cum_matrices_uniform(num_skip_steps + 1, num_categories, alpha)[-1]
    log_p_cum_mats = get_cum_matrices_uniform(num_timesteps+2, num_categories, alpha, num_skip_steps)

    return log_p_onestep_mat.transpose(0, 1), log_p_cum_mats

def gaussian_prior(
    alpha: float,
    num_categories: int, 
    num_timesteps: int, 
    num_skip_steps: int,
    use_doubly_stochastic: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:

    max_distance = num_categories - 1
    if not use_doubly_stochastic:
        indices = torch.arange(num_categories, dtype=torch.float64)[None, ...]
        values = (-4 * (indices - indices.T)**2) / ((alpha *max_distance)**2)
        log_p_onestep_mat = F.log_softmax(values, dim=1)

    else: 
        sum_aux = -4 * torch.arange(-num_categories + 1, num_categories+1, dtype=torch.float64)**2 / (alpha * max_distance)**2
        log_Z   = torch.logsumexp(sum_aux, dim=0)

        indices = torch.arange(num_categories, dtype=torch.float64)
        diff = indices[:, None] - indices[None, :]
        exp_argument = -4*(diff)**2/(alpha * max_distance)**2
        log_pi_vals = exp_argument - log_Z

        off_diag_mask = (diff != 0)

        off_diag_probs = torch.exp(log_pi_vals) * off_diag_mask.float()
        diag_log = torch.log1p(-off_diag_probs.sum(dim=1, keepdim=True))
        
        log_p_onestep_mat = torch.full((num_categories, num_categories), -1e8, dtype=torch.float64)  # -inf
        
        log_p_onestep_mat[off_diag_mask] = log_pi_vals[off_diag_mask]
        
        rows = torch.arange(num_categories)
        log_p_onestep_mat[rows, rows] = diag_log.squeeze(1)

    log_p_onestep_mat = get_cum_matrices(num_skip_steps + 1, log_p_onestep_mat)[-1]
    log_p_cum_mats = get_cum_matrices(num_timesteps + 2, log_p_onestep_mat)

    return log_p_onestep_mat.transpose(0, 1), log_p_cum_mats

# Cumulative returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        0->2        ...         0->N        0->N+1       

# Inherit from nn.Module to do device casting automatically
class Prior(nn.Module):
    log_p_onestep: torch.Tensor
    log_p_cum: torch.Tensor
    
    def __init__(
        self, 
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'uniform_closed_form'
        ] = 'uniform',
        eps: float = 1e-20,
        dtype: Union[str, torch.dtype] = torch.float32
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.num_skip_steps = num_skip_steps
        self.eps = eps
        self.prior_type = prior_type

        if isinstance(dtype, str):
            self.dtype: torch.dtype = getattr(torch, dtype)
        else:
            self.dtype: torch.dtype = dtype

        if prior_type == 'gaussian':
            log_p_onestep, log_p_cum = gaussian_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'uniform':
            log_p_onestep, log_p_cum = uniform_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        elif prior_type == 'uniform_closed_form':
            log_p_onestep, log_p_cum = uniform_closed_form_prior(alpha, num_categories, num_timesteps, num_skip_steps)
        else:
            raise NotImplementedError(f'Got unknown prior: {prior_type} or centroids is None!')
        
        self.register_buffer("log_p_onestep", log_p_onestep)
        self.register_buffer("log_p_cum", log_p_cum)
        self.to(dtype=self.dtype)
        
    def extract(
        self, 
        mat_type: Literal['onestep', 'cumulative'], 
        t: torch.Tensor, 
        *,
        row_id: Optional[torch.Tensor] = None,
        column_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts row/column/element from transition matrix."""     
        if row_id is not None and column_id is not None:
            t = broadcast(t, row_id.dim() - 1)
            if mat_type  == 'onestep':
                return self.log_p_onestep[row_id, column_id]
            else: 
                return self.log_p_cum[t, row_id, column_id]
            
        elif row_id is not None and column_id is None:
            t = broadcast(t, row_id.dim() - 1)
            if mat_type  == 'onestep':
                return self.log_p_onestep[row_id]
            else: 
                return self.log_p_cum[t, row_id, :]
        
        elif row_id is None and column_id is not None:
            t = broadcast(t, column_id.dim() - 1)
            if mat_type == 'onestep':
                result = torch.index_select(
                    self.log_p_onestep, dim=1, 
                    index=column_id.reshape(-1)
                )
                return result.reshape(*column_id.shape, self.num_categories)
            else:
                return self.log_p_cum[t, :, column_id]
        else:   
            raise ValueError('row_id and column_id cannot be None both!')
        
    def extract_last_cum_matrix(self, x: torch.Tensor) -> torch.Tensor:
        last_timestep = torch.full(
            size=(x.shape[0],), 
            fill_value=self.num_timesteps, 
            device=x.device 
        )
        return self.extract('cumulative', last_timestep, row_id=x)
    
    def sample_bridge(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Samples from bridge $p(x_{t} | x_{0}, x_{1})$."""
        log_p_start_t = self.extract('cumulative', t, row_id=x_start)
        log_p_t_end = self.extract('cumulative', self.num_timesteps + 1 - t, column_id=x_end)
        log_probs = log_p_start_t + log_p_t_end
        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        
        noise = torch.rand_like(log_probs)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(log_probs + gumbel_noise, dim=-1)

        is_final_step = broadcast(t, x_start.dim() - 1) == self.num_timesteps + 1
        x_t = torch.where(is_final_step, x_end, x_t)

        is_first_step = broadcast(t, x_start.dim() - 1) == 1
        x_t = torch.where(is_first_step, x_start, x_t)

        return x_t
