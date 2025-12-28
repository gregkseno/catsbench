from typing import Any, Dict, Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger, CometLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from .base import BasePlotterCallback
from ..methods import DLightSB, DLightSB_M, CSBM, AlphaCSBM
from ..utils import convert_to_numpy, fig2img

from catsbench import BenchmarkImage


class BenchmarkImagePlotterCallback(BasePlotterCallback):
    benchmark: Optional[BenchmarkImage] = None

    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, int, int],
        num_categories: int,
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int,
    ):
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.num_categories = num_categories

        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations

    def setup(
        self,
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM], 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if self.benchmark is not None:
            return
        assert hasattr(trainer.datamodule, 'benchmark'), \
            'Wrong datamodule! It should have `benchmark` attribute'
        self.benchmark = trainer.datamodule.benchmark

    @rank_zero_only
    def _log_smaples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        nrow = int(x_start.shape[0]**0.5)
        
        pred_x_end = convert_to_numpy(make_grid(pl_module.sample(x_start), nrow=nrow))
        x_start = convert_to_numpy(make_grid(x_start, nrow=nrow))
        x_end = convert_to_numpy(make_grid(x_end, nrow=nrow))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True)
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"

        axes[0].imshow(x_start.transpose(1, 2, 0), label=r'p_{start}') 
        axes[1].imshow(x_end.transpose(1, 2, 0), label=r'p_{end}')
        axes[2].imshow(pred_x_end.transpose(1, 2, 0), label=r'p_{\theta}')
        for i in range(3):
            axes[i].get_xaxis().set_ticklabels([])
            axes[i].get_yaxis().set_ticklabels([])
            axes[i].set_axis_off()
            axes[i].legend()
                
        plt.show()
        fig.tight_layout(pad=0.5)
        img = fig2img(fig)
        
        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/samples_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            pl_module.logger.experiment.log_image(
                image_data=img, name=f'{stage}/samples_{fb}', step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, TensorBoardLogger): # can be optimized with add_fig 
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = img_np[:, :, None]            
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/samples_{fb}', img_tensor=img_np, global_step=pl_module.global_step,
                dataformats='HWC'
            )            
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, TensorBoardLogger or CometLogger.'
            )
        
        plt.close()

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray,
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        
        traj_start = x_start[:self.num_trajectories]
        repeats = [self.num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *x_start.shape[1:])

        pred_trajectories = pl_module.sample_trajectory(traj_start)
        num_timesteps, nrow = pred_trajectories.shape[:2]
        pred_trajectories = torch.stack([
                pred_trajectories[0], 
                pred_trajectories[num_timesteps // 8], 
                pred_trajectories[num_timesteps // 2], 
                pred_trajectories[(num_timesteps * 7) // 8], 
                pred_trajectories[-1]
            ], dim=0
        )
        pred_trajectories = convert_to_numpy(make_grid(pred_trajectories.reshape(-1, *self.input_shape), nrow=nrow))

        ax.imshow(pred_trajectories.transpose(1, 2, 0))           
        ax.get_xaxis().set_ticklabels([])

        plt.show()
        fig.tight_layout(pad=0.5)
        img = fig2img(fig)
        
        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/trajectories_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            pl_module.logger.experiment.log_image(
                image_data=img, name=f'{stage}/trajectories_{fb}', step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, TensorBoardLogger):
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = img_np[:, :, None]
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/trajectories_{fb}', img_tensor=img_np, global_step=pl_module.global_step,
                dataformats='HWC'
            )            
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, TensorBoardLogger or CometLogger.'
            )
        
        plt.close()
