import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from torchvision.utils import make_grid

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only

from src.metrics import FID, CMMD, GenerativeNLL, ClassifierAccuracy
from src.metrics import MSE, LPIPS, HammingDistance, EditDistance, BLEUScore
from src.utils import convert_to_numpy, convert_to_torch, fig2img


class ImageLogger(Callback):
    """
    Logs samples + trajectories once per epoch.
    Sub-class for image / text specifics.
    """
    def __init__(self):
        super().__init__()

        # metric objects for calculating and averaging accuracy across batches
        # TODO: Aggregate metrics depend on the exp_type
        self.metrics_val = ...
        self.metrics_test = ...


    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert dataloader_idx in {0, 1}, (
            'Number of dataloaders larger than 2 '
            f'is not supported, got {dataloader_idx}.'
        )
        fb = 'forward' if dataloader_idx == 0 else 'backward'
        if fb == 'forward':
            x_start, x_end = batch
        else:
            x_end, x_start = batch

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert dataloader_idx in {0, 1}, (
            'Number of dataloaders larger than 2 '
            f'is not supported, got {dataloader_idx}.'
        )
        fb = 'forward' if dataloader_idx == 0 else 'backward'
        if fb == 'forward':
            x_start, x_end = batch
        else:
            x_end, x_start = batch

    def _log_samples(
        self,
        x_end: torch.Tensor | np.ndarray, 
        x_start: torch.Tensor | np.ndarray, 
        pred_x_start: torch.Tensor | np.ndarray, 
        fb: Literal['forward', 'backward'],
        labels: Optional[List[str]] = [r'$p_0$', r'$p_1$', r'$q_{\theta}$'],
        step: Optional[int] = None,
        iteration: Optional[int] = None
    ):
        nrow = int(x_start.shape[0]**0.5)
        x_start = convert_to_numpy(make_grid(convert_to_torch(x_start), nrow=nrow))
        x_end = convert_to_numpy(make_grid(convert_to_torch(x_end), nrow=nrow))
        pred_x_start = convert_to_numpy(make_grid(convert_to_torch(pred_x_start), nrow=nrow))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True)
        if iteration is not None:
            fig.suptitle(f'Iteration {iteration}')

        axes[0].imshow(x_end.transpose(1, 2, 0)) 
        axes[1].imshow(x_start.transpose(1, 2, 0))
        axes[2].imshow(pred_x_start.transpose(1, 2, 0))
        
        for i in range(3):
            axes[i].get_xaxis().set_ticklabels([])
            axes[i].get_yaxis().set_ticklabels([])
            axes[i].set_axis_off()
            if labels is not None:
                axes[i].set_title(labels[i])
                
        plt.show()
        fig.tight_layout(pad=0.5)
        im = fig2img(fig)
        
        logger = self.trainer.logger
        if hasattr(logger, "log_image"):    # W&B, Neptune, Comet 2.0
            logger.log_image(
                key=f"{stage}/samples",
                images=[grid], step=step
            )
        else:                               # TensorBoard / CSV
            logger.experiment.add_image(
                f"{stage}/samples", grid, step
            )
        tracker.log({f'samples_{fb}': [wandb.Image(im)]}, step=step)

        plt.close()

    def visualize_trajectory_image(
        pred_x_start: torch.Tensor | np.ndarray, 
        trajectories: torch.Tensor | np.ndarray, 
        fb: Literal['forward', 'backward'],
        figsize: Tuple[float, float] | None = None,
        iteration: Optional[int] = None, 
        exp_path: Optional[str] = None,
        tracker: Optional[GeneralTracker] = None,
        step: Optional[int] = None
    ):
        if figsize is None:
            figsize = (8, 8)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Sampling
        batch_size = trajectories.shape[1]
        trajectories = trajectories.reshape(-1, *trajectories.shape[2:])
        trajectories = convert_to_numpy(make_grid(convert_to_torch(trajectories), nrow=batch_size))

        ax.imshow(trajectories.transpose(1, 2, 0)) 
        if iteration is not None:
            ax.set_title(f'Iteration {iteration}')
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_axis_off()
        plt.show()
        
        fig.tight_layout(pad=0.5)
        im = fig2img(fig)

        if exp_path is not None:
            fig_path = os.path.join(exp_path, 'trajectories', f'trajectories_{fb}_{iteration}_step_{step}.png')
            if not os.path.isfile(fig_path):
                im.save(fig_path)
        
        if tracker:
            tracker.log({f'trajectories_{fb}': [wandb.Image(im)]}, step=step)

        plt.close()