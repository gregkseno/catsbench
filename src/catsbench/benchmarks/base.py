from dataclasses import dataclass
from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .hub_mixin import BenchmarkModelHubMixin
from ..prior import Prior
from ..utils import (
    continuous_to_discrete, 
    gumbel_sample, 
    Logger
)

log = Logger('catsbench', rank_zero_only=True)
STDS = {2:2, 16:2, 64:2}

@dataclass
class BenchmarkBaseConfig:
    dim: int
    num_potentials: int
    num_categories: int
    alpha: float
    num_timesteps: int
    num_skip_steps: int
    prior_type: Literal['gaussian', 'uniform']
    benchmark_type: Literal['gaussian', 'uniform']
    num_val_samples: int
    reverse: bool
    tau: float
    params_dtype: str = 'float32'

class BenchmarkBase(nn.Module, BenchmarkModelHubMixin):

    def __init__(
        self,
        config: BenchmarkBaseConfig,
        init_params: bool = True,
        device: Union[str, torch.device] = 'cpu',
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_potentials = config.num_potentials
        self.num_categories = config.num_categories
        self.alpha = config.alpha
        self.num_timesteps = config.num_timesteps
        self.num_skip_steps = config.num_skip_steps
        self.prior_type = config.prior_type
        self.benchmark_type = config.benchmark_type
        self.num_val_samples = config.num_val_samples
        self.reverse = config.reverse
        self.tau = config.tau
        self.params_dtype = getattr(torch, config.params_dtype)

        if init_params:
            log.info('Initializing parameters...')
            log_alpha, log_cp_cores = self._init_parameters(
                config.benchmark_type, self.params_dtype, device
            )
        else:
            log.info('Skipping parameters initialization!')
            log_alpha = torch.empty(
                (self.num_potentials,), dtype=self.params_dtype, device=device
            )
            log_cp_cores = torch.empty(
                (self.dim, self.num_potentials, self.num_categories), 
                dtype=self.params_dtype, 
                device=device
            )
        self.register_buffer('log_alpha', log_alpha)
        self.register_buffer('log_cp_cores', log_cp_cores)
        
        log.info('Initializing prior...')
        self.prior = Prior(
            alpha=config.alpha,
            num_categories=config.num_categories,
            num_timesteps=config.num_timesteps,
            num_skip_steps=config.num_skip_steps,
            tau=config.tau,
            prior_type=config.prior_type,
            dtype=self.params_dtype,
            device=device
        )
        
    def _init_parameters(
        self, 
        benchmark_type: Literal['gaussian', 'uniform'], 
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize weights uniformly
        log_alpha = torch.log(
            torch.ones(
                self.num_potentials, 
                dtype=dtype, 
                device=device
            ) / self.num_potentials
        )

        # initialize CP cores
        if benchmark_type == 'gaussian':
            # means are sampled from uniformly from sphere of radius 5
            # use rejection sampling to ensure means are well separated
            means_list = []
            min_dist = 2.5 * STDS[self.dim]
            for _ in range(self.num_potentials):
                for _ in range(100):
                    candidate = torch.randn(self.dim, 1, dtype=dtype, device=device)
                    candidate = 5 * candidate / candidate.norm()
                    
                    if not means_list:
                        means_list.append(candidate)
                        break
                        
                    dists = torch.norm(torch.cat(means_list, dim=1) - candidate, dim=0)
                    if (dists > min_dist).all():
                        means_list.append(candidate)
                        break
                else:
                    means_list.append(candidate)
            
            means = torch.cat(means_list, dim=1)
            stds = torch.full(
                (self.dim, self.num_potentials), 
                fill_value=STDS[self.dim], 
                dtype=dtype,
                device=device
            ).unsqueeze(-1)

            # autmatically scale to the num_categories
            means = continuous_to_discrete(
                means, self.num_categories, quantize_range=(-5,5)
            ).unsqueeze(-1).to(dtype=dtype)
            log.info(f'Sampled means (indices):\n{means.squeeze(-1).T.cpu().numpy()}')
            
            # compute log probs
            distribution = Normal(means, stds)
            values = torch.arange(
                self.num_categories, device=device
            ).view(1, 1, self.num_categories)
            log_cp_cores = distribution.log_prob(values) # (D, K, S)

        elif benchmark_type == 'uniform':
            log_cp_cores = torch.rand(
                (self.dim, self.num_potentials, self.num_categories), dtype=dtype, device=device
            ) # (D, K, S)

        else:
            raise ValueError(f'Unknown benchmark type: {benchmark_type}')
        
        return log_alpha, log_cp_cores
    
    @property
    def name(self) -> str:
        return f'd{self.dim}_s{self.num_categories}_{self.prior.prior_type}_a{self.prior.alpha}_{self.benchmark_type}'

    @property
    def device(self) -> str:
        return self.log_cp_cores.device

    def get_transition_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)
        t_orig = t  # keep original for onestep
        t = self.num_timesteps + 1 - t_orig
        tp1 = self.num_timesteps - t_orig

        log_u_t = torch.empty(
            x_t.shape[0], self.num_potentials, self.dim, device=self.device
        ) # [B, K, D]
        for d in range(self.dim):
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_t[:, d]) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], dim=-1 # [B, K, S] 
            ) # [B, K]
        sum_log_u_t = log_u_t.sum(dim=-1)  # [B, K]

        transition_logits = torch.empty(
            x_t.shape[0], self.dim, self.num_categories, device=self.device
        ) # [B, D, S]
        x_tp1_d = torch.arange(self.num_categories, device=self.device) # [S]
        x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
        tp1_repeated = tp1.repeat_interleave(self.num_categories) # [B*S]
        for d in range(self.dim):
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], dim=-1 # [B*S, K, S]
            )  # [B*S, K]
            log_u_tp1_d = log_u_tp1_d.reshape(x_t.shape[0], self.num_categories, self.num_potentials) # [B, S, K]
            log_u_tp1_d = log_u_tp1_d.permute(0, 2, 1) # [B, K, S]      
            log_phi_tp1_d = torch.logsumexp(
                self.log_alpha[None, :, None] + log_u_tp1_d + (sum_log_u_t - log_u_t[:, :, d])[:, :, None], dim=1 # [B, K, S]
            ) # [B, S]
            transition_logits[:, d, :] = log_phi_tp1_d + self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) # [B, S]
        return transition_logits.reshape(*input_shape, self.num_categories) # [B, ..., S]

    @torch.no_grad()
    def markov_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        return_transitions: bool = False
    ) -> torch.Tensor:
        input_shape = x_t.shape
        x_t = x_t.flatten(start_dim=1)

        t_orig = t  # keep original for onestep
        t   = self.num_timesteps + 1 - t_orig
        tp1 = self.num_timesteps - t_orig

        log_u_t = torch.empty(x_t.shape[0], self.num_potentials, self.dim, device=self.device)
        for d in range(self.dim):
            x_d = x_t[:, d] # [B]
            log_pi_ref_t = self.prior.extract('cumulative', t, row_id=x_d) # [B, S]
            log_u_t[:, :, d] = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_t[:, None, :], dim=-1
            ) # [B, K]

        log_w_k = self.log_alpha[None, :] + log_u_t.sum(dim=-1) # [B, K]
        k_star = gumbel_sample(log_w_k, tau=self.tau, dim=-1)  # [B]

        logits = torch.empty(x_t.shape[0], self.dim, self.num_categories, device=self.device)
        for d in range(self.dim):
            x_tp1_d = torch.arange(self.num_categories, device=self.device) # [S]
            x_tp1_d = x_tp1_d.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1) # [B*S]
            tp1_repeated = tp1.repeat_interleave(self.num_categories) # [B*S]
            log_pi_ref_tp1 = self.prior.extract('cumulative', tp1_repeated, row_id=x_tp1_d) # [B*S, S]
            log_u_tp1_d = torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref_tp1[:, None, :], # [B*S, K, S]
                dim=-1
            )  # [B*S, K]
            log_u_tp1_d = (log_u_tp1_d
                .reshape(x_t.shape[0], self.num_categories, self.num_potentials)
                .permute(0, 2, 1) # [B, K, S]
            )

            batch_idx = torch.arange(x_t.shape[0], device=self.device)
            log_u_tp1_star = log_u_tp1_d[batch_idx, k_star, :] # [B, S]
            logits_x_tp1_d = self.prior.extract('onestep', t_orig+1, row_id=x_t[:, d]) + log_u_tp1_star # [B, S]
            logits[:, d, :] = logits_x_tp1_d
            
        x_tp1 = gumbel_sample(logits, tau=self.tau, dim=-1)
        if return_transitions:
            # TODO: Optimize logits computation
            return x_tp1.reshape(input_shape), self.get_transition_logits(x_t, t_orig)
        return x_tp1.reshape(input_shape)

    @torch.no_grad()
    def sample(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = True
    ) -> torch.Tensor:
        '''Sample from the model starting from `x` returning the final sample.'''
        if use_onestep_sampling:
            input_shape = x_start.shape
            x_start = x_start.flatten(start_dim=1) # (B, D)

            log_z = torch.zeros(x_start.shape[0], self.num_potentials, device=self.device)
            for d in range(self.dim):
                log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
                log_z = log_z + torch.logsumexp(
                    self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :], dim=2
                ) # (1, K, S) + (B, 1, S) -> (B, K)

            log_w_k = self.log_alpha[None, :] + log_z # (B, K)
            k_star = gumbel_sample(log_w_k, dim=-1, tau=self.tau) # (B,)

            logits = torch.empty(x_start.shape[0], self.dim, self.num_categories, device=self.device)
            for d in range(self.dim):
                log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
                log_cp_cores_d = self.log_cp_cores[d][None, :, :].expand(x_start.shape[0], -1, -1) # (B, K, S)
                log_cp_cores_d_selected = torch.gather(
                    log_cp_cores_d, dim=1, index=k_star[:, None, None].expand(-1, -1, self.num_categories)
                ).squeeze(1) # (B, 1, S) -> (B, S)
                log_p_d_selected = log_cp_cores_d_selected + log_pi_ref[:, :] # (B, S)
                logits[:, d, :] = log_p_d_selected
            x_end = gumbel_sample(
                logits, dim=-1, tau=self.tau
            ).reshape(input_shape)
        else:

            x_t = x_start
            for t in range(0, self.num_timesteps + 1):
                t = torch.full([x_start.shape[0]], t, device=self.device)
                x_t = self.markov_sample(x_t, t, return_transitions=False)
            x_end = x_t
        return x_end
    
    @torch.no_grad()
    def sample_trajectory(
        self, 
        x_start: torch.Tensor, 
        use_onestep_sampling: bool = False,
        return_transitions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_onestep_sampling:
            assert not return_transitions, \
                'Returning transitions is not supported when using bridge samples!'
        
        trajectory, transitions = [x_start], []
        if use_onestep_sampling:
            x_end = self.sample(x_start, use_onestep_sampling=True)
            
            for t in range(1, self.num_timesteps + 1):
                t = torch.full((x_start.shape[0],), t, device=x_start.device)
                x_t = self.prior.sample_bridge(x_start, x_end, t)
                trajectory.append(x_t)
            trajectory.append(x_end)
            
        else:
            x_t = x_start
            for t in range(0, self.num_timesteps + 1):
                t = torch.full([x_t.shape[0]], t, device=self.device)
                out = self.markov_sample(x_t, t, return_transitions=return_transitions)
                if return_transitions:
                    x_t, logits = out
                    transitions.append(logits)
                else:
                    x_t = out
                trajectory.append(x_t)

        trajectory = torch.stack(trajectory, dim=0)
        if return_transitions:
            transitions = torch.stack(transitions, dim=0)
            return trajectory, transitions
        return trajectory

    def _sample_input(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError

    def _sample_target(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        if self.reverse:
            return self._sample_target(num_samples)
        return self._sample_input(num_samples)
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        if self.reverse:
            return self._sample_input(num_samples)
        return self._sample_target(num_samples)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Sample paired input and target data'''
        input_samples = self._sample_input(num_samples)
        target_samples = self.sample(input_samples)
        if self.reversed:
            return target_samples, input_samples
        return input_samples, target_samples

    def _plot_distribution_samples(self, num_samples: int):
        raise NotImplementedError
    
    def _plot_trajectory_samples(
        self, 
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int
    ):
        raise NotImplementedError

    def plot(
        self, 
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int
    ):
        self._plot_distribution_samples(num_samples)
        self._plot_trajectory_samples(num_samples, num_trajectories, num_translations)
