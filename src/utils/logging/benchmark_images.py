from typing import Any, Dict, Literal, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger, CometLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from src.utils import convert_to_numpy, fig2img
from src.metrics.c2st import ClassifierTwoSampleTest
from src.utils.logging.console import RankedLogger
from benchmark import BenchmarkImages


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkImagesLogger(Callback):
    benchmark: BenchmarkImages

    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, int, int],
        num_categories: int,
        train_test_split: float,
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int,
    ):
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.num_categories = num_categories

        self.train_test_split = train_test_split
        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        self._buffers = {
            stage: {'x_start': [], 'x_end': []} \
                for stage in ('train', 'val', 'test')
        }

    def _reset_buf(self, stage: Literal['train', 'val', 'test']) -> None:
        self._buffers[stage]['x_start'].clear()
        self._buffers[stage]['x_end'].clear()

    def _accumulate_buf(
        self, 
        stage: Literal['train', 'val', 'test'],
        x_start: torch.Tensor, 
        x_end: torch.Tensor
    ) -> None:
        buf = self._buffers[stage]
        have = sum(t.shape[0] for t in buf['x_start'])
        remain = self.num_samples - have
        if remain <= 0:
            return
        take = min(remain, x_start.shape[0])
        buf['x_start'].append(x_start[:take].detach())
        buf['x_end'].append(x_end[:take].detach())

    def _log_buf(self, stage: Literal['train', 'val', 'test'], pl_module: LightningModule) -> None:
        buf = self._buffers[stage]
        if not buf['x_start']:
            return
        x_start = torch.cat(buf['x_start'], dim=0)[:self.num_samples]
        x_end = torch.cat(buf['x_end'], dim=0)[:self.num_samples]
        self._log_smaples(x_start, x_end, pl_module, stage)
        self._log_trajectories(x_start, x_end, pl_module, stage=stage)
        self._reset_buf(stage)

    def setup(
        self,
        trainer: Trainer, 
        pl_module: LightningModule, 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if pl_module.current_epoch != 0 and stage != 'fit':
            return

        # get benchmark class
        assert hasattr(trainer.datamodule, 'benchmark'), \
            'Wrong datamodule! It should have `benchmark` attribute'
        self.benchmark = trainer.datamodule.benchmark
        self.benchmark.to(pl_module.device)

        # initialize metrics
        pl_module.fid = FrechetInceptionDistance(normalize=True)
        pl_module.c2st = ClassifierTwoSampleTest()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_buf('train')

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pl_module.eval()
        x_start, x_end = outputs['x_start'], outputs['x_end']
        self._accumulate_buf('train', x_start, x_end)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_buf('train', pl_module)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_buf('val')

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pl_module.eval()
        x_start, x_end = outputs['x_start'], outputs['x_end']
        self._accumulate_buf('val', x_start, x_end)

        pred_x_end = pl_module.sample(x_start)
        pl_module.fid.update(x_end, real=True)
        pl_module.fid.update(pred_x_end, real=False)
        pl_module.c2st.update(
            torch.cat([x_start.flatten(start_dim=1), x_end.flatten(start_dim=1)], dim=-1), 
            torch.cat([x_start.flatten(start_dim=1), pred_x_end.flatten(start_dim=1)], dim=-1),
            train=batch_idx < int(len(trainer.train_dataloader) * self.train_test_split)
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        
        fid = pl_module.fid.compute()
        pl_module.log(f'val/fid_{fb}', fid)
        pl_module.fid.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'val/c2st_{fb}', c2st)
        pl_module.c2st.reset()

        self._log_buf('val', pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_buf('test')

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        pl_module.eval()
        x_start, x_end = outputs['x_start'], outputs['x_end']
        self._accumulate_buf('test', x_start, x_end)

        pred_x_end = pl_module.sample(x_start)
        pl_module.fid.update(x_end, real=True)
        pl_module.fid.update(pred_x_end, real=False)
        pl_module.c2st.update(
            torch.cat([x_start.flatten(start_dim=1), x_end.flatten(start_dim=1)], dim=-1), 
            torch.cat([x_start.flatten(start_dim=1), pred_x_end.flatten(start_dim=1)], dim=-1),
            train=batch_idx < int(len(trainer.train_dataloader) * self.train_test_split)
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        
        fid = pl_module.fid.compute()
        pl_module.log(f'val/fid_{fb}', fid)
        pl_module.fid.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'test/c2st_{fb}', c2st)
        pl_module.c2st.reset()

        self._log_buf('test', pl_module)

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
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.iteration}')

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
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.iteration}')
        
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
