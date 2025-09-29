from typing import Literal, Optional, Tuple
import os

import torch
from torch import nn

from benchmark.prior import Prior
from benchmark.stylegan2 import legacy, dnnlib
from benchmark.stylegan2.training.networks import Generator
from benchmark.utils import  continuous_to_discrete, sample_separated_means, Logger
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence

log = Logger(__name__, rank_zero_only=True)
SPREADS = {50:{2:1.5, 16:1.5, 64:2.5}, 200:{2:4, 16:8, 64:16}}

class BenchmarkBase:
    dim: int
    num_potentials: int
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
                self.num_potentials, self.dim, self.prior.num_categories, min_dist=10, device=device
            ).unsqueeze(1)
            stds = torch.full((self.num_potentials, self.dim), spread, device=device).unsqueeze(1) # (K, D)
            y_d = torch.arange(
                self.prior.num_categories, device=device
            ).view(self.prior.num_categories, 1).repeat(1, self.dim).unsqueeze(0)  # (S, D)
            log_cp_cores = -0.5 * torch.log(torch.tensor(2 * torch.pi, device=device)) - torch.log(stds) - 0.5 * ((y_d - means) / stds) ** 2

        elif benchmark_type == 'log_gaussian':
            mu = torch.zeros(self.dim, device=device)          
            sigma = torch.ones(self.dim, device=device) * 0.5  

            log_normal = torch.distributions.LogNormal(mu, sigma)
            log_cp_cores: torch.Tensor = log_normal.sample((self.num_potentials, self.prior.num_categories,)) # type: ignore

        elif benchmark_type == 'uniform':
            log_cp_cores = torch.rand(
                (self.num_potentials, self.prior.num_categories, self.dim), device=device
            )     

        else:
            raise ValueError(f'Unknown benchmark type: {benchmark_type}')

        return log_cp_cores
    
    @torch.no_grad()
    def sample_target_given_input(self, x: torch.Tensor, return_trajectories: bool = False) -> torch.Tensor:
        input_shape = x.shape
        x = x.flatten(start_dim=1)

        log_z = torch.zeros(x.shape[0], self.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d)
            
            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (B, S) -> (B, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (B, K)
            log_z = log_z + log_inner # (B, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (B, K) -> (B, K)
        noise = torch.rand_like(log_w_k)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        k_stars = torch.argmax(log_w_k + gumbel_noise, dim=-1) # (B,)

        y_samples = torch.empty(x.shape[0], self.dim, dtype=torch.long, device=self.device)
        batch_idx = torch.arange(x.shape[0], device=self.device)

        for d in range(self.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            noise = torch.rand_like(log_p_d_selected)
            noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
            gumbel_noise = -torch.log(-torch.log(noise))
            y_d = torch.argmax(log_p_d_selected + gumbel_noise, dim=-1)
            y_samples[:, d] = y_d
               
        if not return_trajectories:
            return y_samples.reshape(input_shape)
        else:
            trajectory = [x]
            for t in range(1, self.prior.num_timesteps + 1):
                t = torch.full((x.shape[0],), t, device=x.device)
                x_t = self.prior.sample_bridge(x, y_samples, t)
                trajectory.append(x_t)
            trajectory.append(y_samples)
            return torch.stack(trajectory, dim=0)
    
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
        save_path: str = '../data/benchmark',
        device: str = 'cpu'
    ):
        super().__init__()
        self.dim = dim
        self.num_potentials = num_potentials
        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)
        self.input_dist = input_dist
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
                benchmark_type, spread=SPREADS[self.prior.num_categories][dim], device=device
            ).permute(2, 0, 1).contiguous() # (D, K, S)

            log.info('Sampling validation dataset...')
            assert num_val_samples is not None, 'For benchmark computation the `num_val_samples` must be provided!'
            self.input_dataset = self.sample_input(num_val_samples)
            self.target_dataset = self.sample_target_given_input(self.input_dataset)

            random_indices = torch.randperm(len(self.target_dataset))
            self.input_dataset  = self.input_dataset[random_indices]
            self.target_dataset = self.target_dataset[random_indices]
            
            self.save(solver_path, source_path, target_path, benchmark_dir)

    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        '''Sample independent source data'''
        if self.input_dist == 'gaussian':
            samples = continuous_to_discrete(
                torch.randn(size=[num_samples, self.dim], device=self.device), 
                self.prior.num_categories
            )
        elif self.input_dist == 'uniform':
            samples = continuous_to_discrete(
                6 * torch.rand(size=(num_samples, self.dim), device=self.device) - 3,
                self.prior.num_categories
            )
        else:
            raise ValueError(f'Unknown input distribution: {self.input_dist}')
        return samples
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        '''Sample independent target data'''
        input_samples = self.sample_input(num_samples)
        target_samples = self.sample_target_given_input(input_samples)
        return target_samples
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Sample paired input and target data'''
        input_samples = self.sample_input(num_samples)
        target_samples = self.sample_target_given_input(input_samples)
        return input_samples, target_samples

class BenchmarkImages(BenchmarkBase):
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
        generator_path: str = '../checkpoints/cmnist_stylegan2.pkl',
        save_path: str = '../data/benchmark_images',
        device: str = 'cpu'
    ):
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.num_potentials = num_potentials
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
                self.input_dataset[start:end] = self._postporcess(self.generator(noise, None)).cpu()
                self.target_dataset[start:end] = self.sample_target_given_input(
                    self.input_dataset[start:end].to(device)
                ).reshape_as(self.input_dataset[start:end])

            self.save(solver_path, source_path, target_path, benchmark_dir)

    @staticmethod
    def _postporcess(outputs: torch.Tensor) -> torch.Tensor:
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
    def sample_input(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        return input_samples
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        target_samples = self.sample_target_given_input(input_samples)
        return target_samples.reshape_as(input_samples)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int)-> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postporcess(self.generator(noise, None))
        target_samples = self.sample_target_given_input(input_samples)
        return input_samples, target_samples.reshape_as(input_samples)

class SentenceGenerator:
    def __init__(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.valid_prompts = [' ']
        self.model.eval()
        
        self.tokenized_prompts = self._preprocess_prompts()
        
    def _preprocess_prompts(self):
        tokenized_prompts = []
        for prompt in self.valid_prompts:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            tokenized_prompts.append(tokens)
        
        padded_prompts = pad_sequence(tokenized_prompts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded_prompts
    
    def generate_tokens(self, n_sentences=10, d_tokens=64, temperature=0.8, batch_size=8):

        all_tokens = []
        
        for i in range(0, n_sentences, batch_size):
            current_batch_size = min(batch_size, n_sentences - i)
            batch_sentences = self._generate_batch(current_batch_size, d_tokens, temperature)
            all_tokens.extend(batch_sentences)
        
        all_tokens = torch.cat(all_tokens, dim=0).reshape(n_sentences, d_tokens)
        print(all_tokens.shape)
        return all_tokens
    
    def _generate_batch(self, batch_size, d_tokens, temperature):
        prompt_indices = torch.randint(0, len(self.tokenized_prompts), (batch_size,))
        input_ids = self.tokenized_prompts[prompt_indices]
        
        prompt_length = input_ids.shape[1]
        max_new_tokens = d_tokens - prompt_length
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=None,
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                top_p=0.9,
            )
        
        batch_tokens = []
        for i in range(batch_size):
            original_prompt = input_ids[i]
            non_padding_mask = original_prompt != self.tokenizer.pad_token_id
            original_length = non_padding_mask.sum().item()
            
            generated_tokens = output[i][original_length:]
            
            current_length =  len(generated_tokens)
            if current_length < d_tokens:
                padding_needed = d_tokens - current_length
                padding_tokens = torch.full((padding_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                generated_tokens = torch.cat([generated_tokens, padding_tokens])

            assert len(generated_tokens) == d_tokens, f'{len(generated_tokens)}'
            batch_tokens.append(generated_tokens)
            #sentence = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            #batch_sentences.append(sentence)
        
        return batch_tokens

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
        ] = 'gaussian',
        benchmark_type: Literal[
            'gaussian_mixture', 
            'log_gaussian', 
            'uniform'
        ]  = 'gaussian_mixture',
        num_val_samples: Optional[int] = None,
        tokenizer_path: str = '../data/gutenberg_fine_tuned',
        save_path: str = '../data/benchmark_text',
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.dim = dim
        self.num_potentials = num_potentials
        self.device = device

        self.prior  = Prior(
            alpha=alpha, 
            num_categories=num_categories, 
            num_timesteps=num_timesteps, 
            num_skip_steps=num_skip_steps, 
            prior_type=prior_type
        ).to(device)

        # Load tokenizer and model
        print('Initializing token generator...')
        self.generator = SentenceGenerator(tokenizer_path)
        
        self.log_alpha = torch.log(torch.ones(self.num_potentials) / self.num_potentials).to(device)

        self.log_cp_cores = self._get_log_cp_cores(benchmark_type, device=device).contiguous().permute(2, 0, 1)
    
    @torch.no_grad()
    def sample_input(self, num_samples: int) -> torch.Tensor:
        tokens = self.generator.generate_tokens(n_sentences=num_samples, d_tokens=self.dim)#.to('cuda')
        return tokens
    
    @torch.no_grad()
    def sample_target(self, num_samples: int) -> torch.Tensor:
        tokens = self.generator.generate_tokens(n_sentences=num_samples, d_tokens=self.dim)#.to('cuda')
        noised_tokens = self.sample_target_given_input(tokens)
        return noised_tokens.reshape_as(tokens)
    
    @torch.no_grad()
    def sample_input_target(self, num_samples: int)-> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = self.generator.generate_tokens(n_sentences=num_samples, d_tokens=self.dim)#.to('cuda')
        target_tokens = self.sample_target_given_input(input_tokens)
        return input_tokens, target_tokens.reshape_as(input_tokens)
