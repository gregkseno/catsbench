from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from .base import BenchmarkBase, BenchmarkBaseConfig
from ..utils import  convert_to_numpy
from ..utils import Logger


log = Logger(__name__, rank_zero_only=True)

class SentenceGenerator(nn.Module):
    def __init__(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path, local_files_only=True, padding_side='left')
        self.model.eval()
        
        self.vocab_size = self.tokenizer.vocab_size

    @torch.no_grad()
    def generate_tokens(self, batch_size=10, n_tokens=64, temperature=0.8):

        input_ids = torch.randint(0, self.vocab_size, (batch_size, 1), device=self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=n_tokens - 1,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
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

@dataclass 
class BenchmarkTextConfig(BenchmarkBaseConfig):
    generator_kwargs: Dict[str, Any]

class BenchmarkText(BenchmarkBase):

    def __init__(
        self, 
        config: BenchmarkTextConfig,
        *,
        init_benchmark: bool = True,
        generator_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        if not config.reverse:
            raise ValueError('Only reverse benchmarks are supported for text data.')
        
        super().__init__(config)
        if init_benchmark:
            log.info('Loading GPT2 from new checkpoint...')
            if generator_path is None:
                raise ValueError('generator_path must be provided when init_benchmark is True')
            generator = self._load_generator(
                generator_path, device, **config.generator_kwargs
            )
        else:
            log.info('Skipping GPT2 initialization!')
            generator = pipe
            
        self.register_buffer('generator', generator)
        self.register_buffers(init_benchmark, device)

    @property
    def name(self) -> str:
        return f'text_' + super().name

    @torch.no_grad()
    def _sample_input(self, num_samples: int) -> torch.LongTensor:
        noise = torch.randn((num_samples, 512), device=self.device)
        input_samples = self._postprocess(self.generator(noise, None))
        return input_samples
    
    def _plot_samples(self, num_samples: int, **kwargs):
        # prepare samples
        nrow = int(num_samples**0.5)
        x_start = self.input_dataset[:num_samples]
        x_end = self.target_dataset[:num_samples]
        if self.reverse:
            pred_x_end = self.sample(x_end, use_onestep_sampling=True)
        else:
            pred_x_end = self.sample(x_start, use_onestep_sampling=True)
        x_start = convert_to_numpy(make_grid(x_start, nrow=nrow))
        x_end = convert_to_numpy(make_grid(x_end, nrow=nrow))
        pred_x_end = convert_to_numpy(make_grid(pred_x_end, nrow=nrow))
        
        # plot samples
        fig, axs = plt.subplots(
            1, 3, dpi=kwargs.get('dpi', 200),
            figsize=kwargs.get('fig_size', (12, 4))
        )
        axs[0].imshow(x_start.transpose(1, 2, 0), label=r'p_{start}') 
        axs[1].imshow(x_end.transpose(1, 2, 0), label=r'p_{end}')
        axs[2].imshow(pred_x_end.transpose(1, 2, 0), label=r'p_{pred}')

        fig.suptitle('Benchmark samples', fontsize=16)
        axs[0].set_title(r'$p_{start}$', fontsize=12)
        axs[1].set_title(r'$p_{end}$', fontsize=12)
        axs[2].set_title(r'$p_{pred}$', fontsize=12)
        for ax in axs:
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_axis_off()
        fig.tight_layout(pad=0.5)
        plt.show()        
        plt.close()

    def _plot_trajectories(
        self, 
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int,
        **kwargs
    ):
        # prepare samples
        traj_start = self.input_dataset[:num_trajectories]
        repeats = [num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *self.input_dataset.shape[1:])

        trajectories = self.sample_trajectory(
            traj_start, use_onestep_sampling=True
        )
        num_timesteps, nrow = trajectories.shape[:2]
        trajectories = torch.stack([
                trajectories[0], 
                trajectories[num_timesteps // 8], 
                trajectories[num_timesteps // 2], 
                trajectories[(num_timesteps * 7) // 8], 
                trajectories[-1]
            ], dim=0
        )
        trajectories = convert_to_numpy(make_grid(
            trajectories.reshape(-1, *self.input_shape),
            nrow=nrow
        ))

        # plot samples
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(trajectories.transpose(1, 2, 0))  

        fig.suptitle('Benchmark trajectories', fontsize=16)         
        ax.get_xaxis().set_ticklabels([])
        fig.tight_layout(pad=0.5)
        plt.show()
        plt.close()
