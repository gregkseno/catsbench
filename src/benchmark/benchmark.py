from typing import Literal, Optional, Tuple, Union
import os

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from tqdm import tqdm

from benchmark.prior import Prior
from benchmark.stylegan2 import legacy, dnnlib
from benchmark.stylegan2.training.networks import Generator
from benchmark.utils import  continuous_to_discrete, sample_separated_means, Logger, gumbel_sample

log = Logger(__name__, rank_zero_only=True)
SPREADS = {50:{2:1.5, 16:1.5, 64:2.5}, 200:{2:4, 16:8, 64:16}}

class BenchmarkBase:
    dim: int
    num_categories: int
    num_potentials: int
    num_timesteps: int
    reversed: bool
    tau: float
    prior: Prior
    log_alpha: torch.Tensor
    log_cp_cores: torch.Tensor
    input_dataset: torch.Tensor
    target_dataset: torch.Tensor
    device: str

    def _get_log_cp_cores(
        self, 
        benchmark_type: Literal['gaussian_mixture', 'log_gaussian', 'uniform'], 
        spread: float = 15.0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        if benchmark_type == 'gaussian_mixture':
            means = sample_separated_means(
                self.num_potentials, self.dim, self.num_categories, min_dist=10, device=device
            ).unsqueeze(1)
            stds = torch.full((self.num_potentials, self.dim), spread, device=device).unsqueeze(1) # (K, D)
            y_d = torch.arange(
                self.num_categories, device=device
            ).view(self.num_categories, 1).repeat(1, self.dim).unsqueeze(0)  # (S, D)
            log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi, device=device)) - torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2

        elif benchmark_type == 'log_gaussian':
            mu = torch.zeros(self.dim, device=device)          
            sigma = torch.ones(self.dim, device=device) * 0.5  

            log_normal = torch.distributions.LogNormal(mu, sigma)
            log_cp_cores: torch.Tensor = log_normal.sample((self.num_potentials, self.num_categories,)) # type: ignore

        elif benchmark_type == 'uniform':
            log_cp_cores = torch.rand(
                (self.num_potentials, self.num_categories, self.dim), device=device
            )     

        else:
            raise ValueError(f'Unknown benchmark type: {benchmark_type}')

        return log_cp_cores

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
        """Sample from the model starting from `x` returning the final sample."""
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

    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        if self.reversed:
            return self._sample_target(num_samples)
        return self._sample_input(num_samples)
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        if self.reversed:
            return self._sample_input(num_samples)
        return self._sample_target(num_samples)
    
    def save(
        self, solver_path: str, source_path: str, target_path: str, dir: str
    ):
        log.info(f'Saving benchmark to {dir}...')
        os.makedirs(dir, exist_ok=True)
        torch.save({'log_alpha': self.log_alpha.cpu(), 'log_cp_cores': self.log_cp_cores.cpu()}, solver_path)
        torch.save(self.input_dataset.cpu(), source_path)
        torch.save(self.target_dataset.cpu(), target_path)

    def load(
        self, solver_path: str, source_path: str, target_path: str, dir: str
    ):
        log.info(f'Loading saved solver and benchmark pairs from {dir}...')
        log_params = torch.load(solver_path, map_location=torch.device('cpu'))
        self.log_alpha = log_params['log_alpha']
        self.log_cp_cores = log_params['log_cp_cores']

        self.input_dataset  = torch.load(source_path, map_location=torch.device('cpu')).long()
        self.target_dataset = torch.load(target_path, map_location=torch.device('cpu')).long()
    
    def to(self, device: torch.device):
        self.prior.to(device)
        self.log_alpha = self.log_alpha.to(device)
        self.log_cp_cores = self.log_cp_cores.to(device)
        if hasattr(self, 'generator'):
            self.generator = self.generator.to(device)
        self.device = device

class Benchmark(BenchmarkBase):
    def __init__(
        self, 
        dim: int,
        num_potentials: int,
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
        ] = 'uniform',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        input_dist: Literal['gaussian', 'uniform'] = 'gaussian',
        reversed: bool = False,
        tau: float = 1.0,
        save_path: str = '../data/benchmark',
        device: str = 'cpu'
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.num_potentials = num_potentials
        self.num_timesteps = num_timesteps
        self.input_dist = input_dist
        self.reversed = reversed
        self.tau = tau
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)
        self.device = device

        benchmark_dir = f"{save_path}/dim_{dim}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        solver_path = os.path.join(benchmark_dir, f'D_P0_{input_dist}.pth') 
        source_path = os.path.join(benchmark_dir, f'X0_P0_{input_dist}.pt')
        target_path = os.path.join(benchmark_dir, f'X1_P0_{input_dist}.pt')
        if os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(solver_path):
            self.load(solver_path, source_path, target_path, benchmark_dir)
            self.to(device)

        else:
            log.info('Initializing parameters...')
            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device) / self.num_potentials)
            self.log_cp_cores = self._get_log_cp_cores(
                benchmark_type, spread=SPREADS[self.num_categories][dim], device=device
            ).permute(2, 0, 1).contiguous() # (D, K, S)

            log.info('Sampling validation dataset...')
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            self.input_dataset = self.sample_input(num_val_samples)
            self.target_dataset = self.sample(self.input_dataset)

            random_indices = torch.randperm(len(self.target_dataset))
            self.input_dataset  = self.input_dataset[random_indices]
            self.target_dataset = self.target_dataset[random_indices]
            
            self.save(solver_path, source_path, target_path, benchmark_dir)

    @torch.no_grad()
    def _sample_input(self, num_samples: int) -> torch.Tensor:
        '''Sample independent source data'''
        if self.input_dist == 'gaussian':
            samples = continuous_to_discrete(
                torch.randn(size=[num_samples, self.dim], device=self.device), 
                self.num_categories
            )
        elif self.input_dist == 'uniform':
            samples = continuous_to_discrete(
                6 * torch.rand(size=(num_samples, self.dim), device=self.device) - 3,
                self.num_categories
            )
        else:
            raise ValueError(f'Unknown input distribution: {self.input_dist}')
        return samples
    
    @torch.no_grad()
    def _sample_target(self, num_samples: int) -> torch.Tensor:
        '''Sample independent target data'''
        input_samples = self.sample_input(num_samples)
        target_samples = self.sample(input_samples)
        return target_samples
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Sample paired input and target data'''
        input_samples = self._sample_input(num_samples)
        target_samples = self.sample(input_samples)
        if self.reversed:
            return target_samples, input_samples
        return input_samples, target_samples

class BenchmarkImage(BenchmarkBase):
    generator: nn.Module

    def __init__(
        self, 
        dim: int,
        input_shape: Tuple[int, int, int],
        num_potentials: int,
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
        ] = 'gaussian',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        reversed: bool = True,
        tau: float = 1.0,
        generator_path: str = '../checkpoints/cmnist_stylegan2.pkl',
        save_path: str = '../data/benchmark_images',
        device: str = 'cpu'
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.num_potentials = num_potentials
        self.num_timesteps = num_timesteps
        self.reversed = reversed
        self.tau = tau
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)
        self._load_generator(generator_path, device=device)
        self.device = device
        
        benchmark_dir = f"{save_path}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        solver_path = os.path.join(benchmark_dir, f'D_c0_{benchmark_type}.pth') 
        source_path = os.path.join(benchmark_dir, f'X0_c0_{benchmark_type}.pt')
        target_path = os.path.join(benchmark_dir, f'X1_c0_{benchmark_type}.pt')
        if os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(solver_path):
            self.load(solver_path, source_path, target_path, benchmark_dir)
            self.to(device)

        else:
            log.info('Loading parameters...')
            self.log_alpha = torch.log(torch.ones(self.num_potentials, device=device) / self.num_potentials)
            self.log_cp_cores = self._get_log_cp_cores(benchmark_type, spread=15, device=device).permute(2, 0, 1).contiguous() # (D, K, S)

            log.info('Sampling validation dataset...')
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            samples_per_batch = 2000
            num_batches = num_val_samples // samples_per_batch
            self.input_dataset = torch.empty((num_batches * samples_per_batch, *self.input_shape), dtype=torch.int)
            self.target_dataset = torch.empty((num_batches * samples_per_batch, *self.input_shape), dtype=torch.int)
            for i in range(num_batches):
                noise = torch.randn((samples_per_batch, 512), device=self.device)
                start, end = samples_per_batch * i, samples_per_batch * (i + 1)
                self.input_dataset[start:end] = self._postprocess(self.generator(noise, None)).cpu()
                self.target_dataset[start:end] = self.sample(
                    self.input_dataset[start:end].to(device)
                ).reshape_as(self.input_dataset[start:end])

            self.save(solver_path, source_path, target_path, benchmark_dir)

    @staticmethod
    def _postprocess(outputs: torch.Tensor) -> torch.Tensor:
        return ((outputs * 0.5 + 0.5).clamp(0, 1) * 255).long()

    def _load_generator(self, generator_path: str, device: str = 'cpu'):
        log.info('Loading StyleGAN2 generator checkpoint...')
        with dnnlib.util.open_url(generator_path) as f:
            self.generator: Generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        log.info(f'Generator loaded on {next(iter(self.generator.parameters())).device}!')

    # NOTE: Here we have reversed setup:
    #       - Input: CMNIST images;
    #       - Target: noised CMNIST images.
    @torch.no_grad()
    def _sample_input(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postprocess(self.generator(noise, None))
        return input_samples
    
    @torch.no_grad()
    def _sample_target(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postprocess(self.generator(noise, None))
        target_samples = self.sample(input_samples)
        return target_samples.reshape_as(input_samples)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int)-> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postprocess(self.generator(noise, None))
        target_samples = self.sample(input_samples).reshape_as(input_samples)
        if self.reversed:
            return target_samples, input_samples
        return input_samples, target_samples

class SentenceGenerator:
    def __init__(self, model_path):
        self.model     = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path, local_files_only=True, padding_side='left')
        self.model.eval()
        
        self.vocab_size = self.tokenizer.vocab_size

    @torch.no_grad()
    def generate_tokens(self, batch_size=10, n_tokens=64, temperature=0.8):

        input_ids = torch.randint(0, self.vocab_size, (batch_size, 1), device=self.model.device)
        outputs = self.model.generate(
                                      input_ids,
                                      max_new_tokens       = n_tokens - 1,
                                      num_return_sequences = 1,
                                      temperature          = temperature,
                                      do_sample            = True,
                                      pad_token_id         = self.tokenizer.eos_token_id,
                                     )

        return outputs
    
    def to(self, device):
        self.model =self.model.to(device)
        return self

    @property
    def device(self):
        return self.model.device
        
    @torch.no_grad()
    def decode(self, batch: torch.tensor) -> list:
        full_text_list = []
        
        for i in range(len(batch)):
            sequence_ids = batch[i].tolist()
            full_text = self.tokenizer.decode(sequence_ids)
            full_text_list.append(full_text)

        return full_text_list

class BenchmarkText(BenchmarkBase):
    def __init__(
        self, 
        dim: int,
        num_potentials: int,
        alpha: float,
        num_categories: int,
        num_timesteps: int,
        num_skip_steps: int,
        prior_type: Literal[
            'uniform', 
            'gaussian',
            'uniform_closed_form'
        ] = 'gaussian',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        reversed: bool = True,
        tau: float = 1.0,
        generator_path: str = '../checkpoints/gpt2-tinystories-final',
        save_path: str = '../data/benchmark_texts',
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.dim = dim
        self.num_categories = num_categories
        self.num_potentials = num_potentials
        self.num_timesteps = num_timesteps
        self.reversed = reversed
        self.tau = tau

        log.info('Initializing prior...')
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)
        self.generator = SentenceGenerator(generator_path)
        self.device = device

        benchmark_dir = f"{save_path}/num_categories_{num_categories}/prior_{prior_type}/alpha_{alpha}/"
        solver_path = os.path.join(benchmark_dir, f'D_c0_{benchmark_type}.pth') 
        source_path = os.path.join(benchmark_dir, f'X0_c0_{benchmark_type}.pt')
        target_path = os.path.join(benchmark_dir, f'X1_c0_{benchmark_type}.pt')

        if os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(solver_path):
            self.load(solver_path, source_path, target_path, benchmark_dir)

            self.to(device)

        else:
            log.info('Loading parameters...')
            self.log_alpha = torch.log(torch.ones(self.num_potentials) / self.num_potentials).to(device)
            self.log_cp_cores = self._get_log_cp_cores(benchmark_type, device=device).contiguous().permute(2, 0, 1)

            log.info('Sampling validation dataset...')
            self.generator = self.generator.to(device)
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            samples_per_batch = 2000
            num_batches = num_val_samples // samples_per_batch
            self.input_dataset = torch.empty((num_batches * samples_per_batch, self.dim), dtype=torch.int)
            self.target_dataset = torch.empty((num_batches * samples_per_batch, self.dim), dtype=torch.int)
            for i in tqdm(range(num_batches)):
                start, end = samples_per_batch * i, samples_per_batch * (i + 1)
                self.input_dataset[start:end] = self.generator.generate_tokens(batch_size=samples_per_batch, n_tokens=self.dim).cpu()
                self.target_dataset[start:end] = self.sample(
                    self.input_dataset[start:end].to(device)
                ).reshape_as(self.input_dataset[start:end])

            self.save(solver_path, source_path, target_path, benchmark_dir)

    
    @torch.no_grad()
    def _sample_input(self, num_samples: int) -> torch.Tensor:
        tokens = self.generator.generate_tokens(batch_size=num_samples, n_tokens=self.dim)#.to('cuda')
        return tokens
    
    @torch.no_grad()
    def _sample_target(self, num_samples: int) -> torch.Tensor:
        tokens = self.generator.generate_tokens(batch_size=num_samples, n_tokens=self.dim)#.to('cuda')
        noised_tokens = self.sample(tokens)
        return noised_tokens.reshape_as(tokens)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int)-> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = self.generator.generate_tokens(batch_size=num_samples, n_tokens=self.dim)#.to('cuda')
        target_tokens = self.sample(input_tokens).reshape_as(input_tokens)
        if self.reversed:
            return target_tokens, input_tokens
        return input_tokens, target_tokens
    
