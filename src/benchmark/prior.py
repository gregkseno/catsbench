from typing import Literal, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from utils import broadcast
from math import log


def log_space_product(A, B):
    A_exp = A.unsqueeze(2)  
    B_exp = B.unsqueeze(0)  

    return torch.logsumexp(A_exp + B_exp, dim=1)

def get_cum_matrices(num_timesteps: int, onestep_matrix: torch.Tensor) -> torch.Tensor:
    num_categories = onestep_matrix.shape[0]
    cum_matrices = torch.empty(size=(num_timesteps, num_categories, num_categories), dtype=onestep_matrix.dtype)

    cum_matrices[0] = torch.eye(num_categories, dtype=onestep_matrix.dtype)
    
    for timestep in range(1, num_timesteps):
        cum_matrices[timestep] = cum_matrices[timestep-1] @ onestep_matrix
    
    assert onestep_matrix.shape == cum_matrices[0].shape, f'Wrong shape!'
    return cum_matrices

def get_log_cum_matrices(num_timesteps: int, log_onestep_matrix: torch.Tensor) -> torch.Tensor:
    num_categories = log_onestep_matrix.shape[0]
    log_cum_matrices = torch.empty(size=(num_timesteps, num_categories, num_categories), dtype=log_onestep_matrix.dtype)
    log_identity = torch.full((num_categories, num_categories), float('-inf'), dtype=torch.float64)  # log(0) = -inf
    rows = torch.arange(num_categories)
    log_identity[rows, rows] = 0.0

    log_cum_matrices[0] = log_identity[:]#torch.clone(log_onestep_matrix)
    
    for timestep in range(1, num_timesteps):
        log_cum_matrices[timestep] = log_space_product(log_cum_matrices[timestep-1], log_onestep_matrix)

    assert log_onestep_matrix.shape == log_cum_matrices[0].shape, f'Wrong shape!'
    return log_cum_matrices

def uniform_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_onestep_mat = torch.tensor([alpha] * num_categories**2, dtype=torch.float64)
    p_onestep_mat = p_onestep_mat / (num_categories - 1)
    p_onestep_mat = p_onestep_mat.view(num_categories, num_categories)
    p_onestep_mat -= torch.diag(torch.diag(p_onestep_mat))
    p_onestep_mat += torch.diag(torch.ones(num_categories, dtype=torch.float64) - alpha)
    p_onestep_mat = torch.matrix_power(p_onestep_mat, n=num_skip_steps)

    p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)

    return p_onestep_mat.transpose(0, 1), p_cum_mats

def uniform_log_prior(
    alpha: float, 
    num_categories: int, 
    num_timesteps: int,
    num_skip_steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    log_equal  = log(1 - alpha)                 #log((1 - alpha)/(num_categories - 1) + 1e-20)
    log_diff = log(alpha/(num_categories - 1))#log(alpha + (1 - alpha)/(num_categories - 1) + 1e-20)

    log_p_onestep_mat = torch.full((num_categories, num_categories), log_diff)  #(batch_size, S)
    rows = torch.arange(num_categories).to(torch.int32)
    log_p_onestep_mat[rows, rows] = log_equal
    #print(torch.exp(log_p_onestep_mat))

    p_onestep_mat = torch.tensor([alpha] * num_categories**2, dtype=torch.float64)
    p_onestep_mat = p_onestep_mat / (num_categories - 1)
    p_onestep_mat = p_onestep_mat.view(num_categories, num_categories)
    p_onestep_mat -= torch.diag(torch.diag(p_onestep_mat))
    p_onestep_mat += torch.diag(torch.ones(num_categories, dtype=torch.float64) - alpha)
    #print(p_onestep_mat)

    #if num_skip_steps > 1:
    #p_onestep_mat = torch.matrix_power(p_onestep_mat, n=num_skip_steps)
    #print(num_timesteps-2)
    p_onestep_mat_1 = torch.matrix_power(p_onestep_mat, n=num_timesteps+1)
    print(p_onestep_mat_1)

    log_p_cum_mats = get_log_cum_matrices(num_timesteps + 2, log_p_onestep_mat)
    print(torch.exp(log_p_cum_mats)[-1])

    #p_cum_mats = get_cum_matrices(num_timesteps + 2, p_onestep_mat)
    #print(torch.exp(log_p_cum_mats))
    #print(p_cum_mats)

    assert False

    return log_p_onestep_mat.transpose(0, 1), log_p_cum_mats


def gaussian_prior_log(
    alpha: float,
    num_categories: int, 
    num_timesteps: int, 
    num_skip_steps: int,
    use_doubly_stochastic: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:

    max_distance = num_categories - 1
    if not use_doubly_stochastic:
        indices = np.arange(num_categories)[None, ...]
        values = (-4 * (indices - indices.T)**2) / ((alpha *max_distance)**2)
        p_onestep_mat = softmax(values, axis=1)
    else: # this logic mathcing D3PM article

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


    if num_skip_steps > 1:
        p_onestep_mat = torch.exp(log_p_onestep_mat)

        p_onestep_mat = torch.linalg.matrix_power(p_onestep_mat, n=num_skip_steps)
        log_p_onestep_mat = torch.log(p_onestep_mat)

    log_p_cum_mats = get_log_cum_matrices(num_timesteps + 2, log_p_onestep_mat)

    return log_p_onestep_mat.transpose(0, 1), log_p_cum_mats

# Cumulative returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        0->2        ...         0->N        0->N+1       

# Onestep returns with following pattern
# 0         1           2           ...         N           N+1
# 0->0      0->1        1->2        ...         N-1->N      N->N+1     

# Inherit from nn.Module to do device casting automatically
class Prior(nn.Module):
    def __init__(
        self, 
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
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
            if mat_type  == 'onestep':
                return self.log_p_onestep[:, column_id]
            else:
                return self.log_p_cum[t, :, column_id]
        else:   
            raise ValueError('row_id and column_id cannot be None both!')

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
    
    def bridge_logits(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Calculates log probability of $p(x_{t} | x_{0}, x_{1})$."""
        log_p_start_t = self.extract('cumulative', t, row_id=x_start)
        log_p_t_end = self.extract('cumulative', self.num_timesteps + 1 - t, column_id=x_end)
        log_probs = log_p_start_t + log_p_t_end

        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        return log_probs
    
    def posterior_logits(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        logits: bool = False
    ) -> torch.Tensor:
        r"""Calculates logits of $p(x_{t-1} | x_{t}, x_{0})$.
        If logits is True, the output is summed over x_0 and transition matrix returned.""" 
        if not logits:
            x_start_logits = torch.log(torch.nn.functional.one_hot(x_start, self.num_categories) + self.eps)
        else:
            x_start_logits = x_start.clone()
        assert x_start_logits.shape == x_t.shape + (self.num_categories,), f"x_start_logits.shape: {x_start_logits.shape}, x_t.shape: {x_t.shape}"
        x_start_logits = x_start_logits.to(self.dtype)
        # fact1 is "guess of x_{t}" from x_{t-1}
        log_fact1 = self.extract('onestep', t, row_id=x_t)

        # fact2 is "guess of x_{t-1}" from x_{0}
        x_start_logits = x_start_logits.log_softmax(dim=-1)  # bs, ..., num_categories
        log_fact2 = torch.logsumexp(x_start_logits.unsqueeze(-1) + self.log_p_cum[t-1], dim=-2) 

        p_posterior_logits = log_fact1 + log_fact2
        p_posterior_logits = p_posterior_logits - p_posterior_logits.logsumexp(dim=-1, keepdim=True) # Normalize
        
        # Use `torch.where` because when `t == 1` x_start_logits are actually x_0 already
        is_first_step = broadcast(t, x_t.dim()) == 1
        p_posterior_logits = torch.where(is_first_step, x_start_logits, p_posterior_logits)
        return p_posterior_logits

