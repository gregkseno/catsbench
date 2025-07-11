from typing import Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only

from src.utils import convert_to_numpy, fig2img, optimize_coupling


class ToyLogger(Callback):
    def __init__(
        self,
        num_trajectories: int, 
        num_translations: int,
        axlim: Optional[Tuple[float, float]] = None,
        samples_figsize: Optional[Tuple[int, int]] = None,
        trajectories_figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 100
    ):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        self.axlim = [7, 43] if axlim is None else axlim
        self.samples_fig_config = {
            'figsize': (12, 4) if samples_figsize is None else samples_figsize,
            'dpi': dpi,
        }
        self.samples_start_config = {
            'label': r'$p_{start}$', 'c': 'g', 's': 35, 'edgecolor': 'black'
        }
        self.samples_end_config = {
            'label': r'$p_{end}$', 'c': 'orange', 's': 35, 'edgecolor': 'black'
        }
        self.samples_pred_config = {
            'label': r'$p_{\theta}$', 'c': 'yellow', 's': 35, 'edgecolor': 'black'
        }
        self.trajectories_fig_config = {
            'figsize': (8, 8) if trajectories_figsize is None else trajectories_figsize,
            'dpi': dpi,
        }
        self.trajectories_pred_config = {
            'c': 'salmon', 's': 120, 'edgecolors': 'black', 
            'label': 'Fitted distribution', 'zorder': 1, 'linewidth': 0.8
        }
        self.trajectories_start_config = {
            'c': 'lime', 's': 180, 'edgecolors': 'black', 
            'label': r'Trajectory start ($x \sim p_0$)', 'zorder': 3
        }
        self.trajectories_end_config = {
            'c': 'yellow', 's': 100, 'edgecolors': 'black', 
            'label': r'Trajectory end (fitted)', 'zorder': 3
        }
        self.trajectory_lines_config = {
            'back': {'c': 'black', 'markeredgecolor': 'black', 'linewidth': 2, 'zorder': 2},
            'front': {'c': 'grey', 'markeredgecolor': 'black', 'linewidth': 1, 'zorder': 2}
        }

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        fb, bf = 'forward', 'backward'
        if pl_module.bidirectional and pl_module.current_epoch % 2 != 0:
            fb, bf = 'backward', 'forward'
        pl_module.eval()

        if fb == 'forward': x_start, x_end = batch
        else: x_end, x_start = batch

        # if first iteration apply optional mini-batch sampling
        if pl_module.first_iteration and pl_module.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if not pl_module.first_iteration and pl_module.bidirectional:
            x_start, x_end = pl_module.sample(x_end, fb=bf), x_end 

        if batch_idx == 0:
            self._log_smaples(x_start, x_end, fb, pl_module, 'train')
            if getattr(pl_module, 'sample_trajectory', None) is not None:
                self._log_trajectories(x_start, fb, pl_module, stage='train')

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        fb = 'forward'
        if pl_module.bidirectional and pl_module.current_epoch % 2 != 0:
            fb = 'backward'
        pl_module.eval()

        if fb == 'forward': x_start, x_end = batch
        else: x_end, x_start = batch

        if batch_idx == 0:
            self._log_smaples(x_start, x_end, fb, pl_module, 'val')
            if getattr(pl_module, 'sample_trajectory', None) is not None:
                self._log_trajectories(x_start, fb, pl_module, stage='val')

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        fb = 'forward'
        if pl_module.bidirectional and pl_module.current_epoch % 2 != 0:
            fb = 'backward'
        pl_module.eval()

        if fb == 'forward': x_start, x_end = batch
        else: x_end, x_start = batch

        if batch_idx == 0:
            self._log_smaples(x_start, x_end, fb, pl_module, 'test')
            if getattr(pl_module, 'sample_trajectory', None) is not None:
                self._log_trajectories(x_start, fb, pl_module, stage='test')

    @rank_zero_only
    def _log_smaples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        fb: Literal['forward', 'backward'],
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        pred_x_end = convert_to_numpy(pl_module.sample(x_start, fb=fb))
        x_start = convert_to_numpy(x_start)
        x_end = convert_to_numpy(x_end)

        fig, axes = plt.subplots(1, 3, **self.samples_fig_config)
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.imf_iteration}')

        axes[0].scatter(x_start[:, 0], x_start[:, 1], **self.samples_start_config) 
        axes[1].scatter(x_end[:, 0], x_end[:, 1], **self.samples_end_config)
        axes[2].scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.samples_pred_config) 
        
        for i in range(3):
            axes[i].grid()
            axes[i].set_xlim(self.axlim)
            axes[i].set_ylim(self.axlim)
            axes[i].legend(loc="lower left")
        fig.tight_layout(pad=0.5)
        img = fig2img(fig)
        pl_module.logger.log_image(
            key=f'{stage}/samples_{fb}', images=[img], step=pl_module.global_step
        )
        plt.close()

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray, 
        fb: Literal['forward', 'backward'],
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        fig, ax = plt.subplots(1, 1, **self.trajectories_fig_config)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.imf_iteration}')
        
        pred_x_end = convert_to_numpy(pl_module.sample(x_start, fb=fb))
        traj_start = x_start[:self.num_trajectories]
        repeats = [self.num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *x_start.shape[1:])
        trajectories = pl_module.sample_trajectory(traj_start, fb)

        # Reduce number of timesteps for visualization
        num_timesteps = trajectories.shape[0]
        trajectories = convert_to_numpy(torch.stack([
                trajectories[0], 
                trajectories[num_timesteps // 8], 
                trajectories[num_timesteps // 2], 
                trajectories[(num_timesteps * 7) // 8], 
                trajectories[-1]
            ], dim=0
        ))

        ax.scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.trajectories_pred_config)
        ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], **self.trajectories_start_config)
        ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], **self.trajectories_end_config)
        for i in range(self.num_trajectories):
            ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['back'])
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['front'], 
                label='Intermediate predictions' if i == 0 else None
            )
        ax.legend(loc='lower left')
        ax.set_xlim(self.axlim)
        ax.set_ylim(self.axlim)
        fig.tight_layout(pad=0.5)
        img = fig2img(fig)
        pl_module.logger.log_image(
            key=f'{stage}/trajectories_{fb}', images=[img], step=pl_module.global_step
        )
        plt.close()
