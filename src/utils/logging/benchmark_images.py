from typing import Any, Dict, Literal, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid
from torchmetrics import MetricCollection
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger, CometLogger
from lightning.pytorch.utilities import rank_zero_only

from src.utils import convert_to_numpy, fig2img
from src.metrics.c2st import ClassifierTwoSampleTest
from src.metrics.contingency_similarity import ContingencySimilarity
from src.metrics.tv_complement import TVComplement
from src.utils.logging.console import RankedLogger
from src.benchmark import BenchmarkImages


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkImagesLogger(Callback):
    benchmark: BenchmarkImages

    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
    
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
        pl_module.metrics = MetricCollection(
            {
                'tv_complement': TVComplement(self.dim, self.num_categories),
                'contingency_similarity': ContingencySimilarity(self.dim, self.num_categories),
            },
        )
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
        pl_module.metrics.update(x_end, pred_x_end)
        pl_module.c2st.update(
            torch.cat([x_start, x_end], dim=-1), 
            torch.cat([x_start, pred_x_end], dim=-1)
        )

        repeated_x_end = x_end[0].unsqueeze(0).expand(self.num_cond_samples, -1)
        cond_x_end = self.benchmark.sample_target_given_input(repeated_x_end)
        cond_pred_x_end = pl_module.sample(repeated_x_end)
        pl_module.cond_metrics.update(cond_x_end, cond_pred_x_end)
        pl_module.cond_c2st.update(
            torch.cat([repeated_x_end, cond_x_end], dim=-1), 
            torch.cat([repeated_x_end, cond_pred_x_end], dim=-1)
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        
        metrics = pl_module.metrics.compute()
        metrics = {f'val/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        pl_module.metrics.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'val/c2st_{fb}', c2st)
        pl_module.c2st.reset()

        cond_c2st = pl_module.cond_c2st.compute()
        pl_module.log(f'val/cond_c2st_{fb}', cond_c2st)
        pl_module.cond_c2st.reset()
        
        cond_metrics = pl_module.cond_metrics.compute()
        cond_metrics = {f'val/{k}_{fb}': v for k, v in cond_metrics.items()}
        pl_module.log_dict(cond_metrics)
        pl_module.cond_metrics.reset()

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
        pl_module.metrics.update(x_end, pred_x_end)
        pl_module.c2st.update(
            torch.cat([x_start, x_end], dim=-1), 
            torch.cat([x_start, pred_x_end], dim=-1)
        )
        
        repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
        cond_x_end = self.benchmark.sample_target_given_input(repeated_x_start)
        cond_pred_x_end = pl_module.sample(repeated_x_start)
        pl_module.cond_metrics.update(cond_x_end, cond_pred_x_end)
        pl_module.cond_c2st.update(
            torch.cat([repeated_x_start, cond_x_end], dim=-1), 
            torch.cat([repeated_x_start, cond_pred_x_end], dim=-1)
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        
        metrics = pl_module.metrics.compute()
        metrics = {f'test/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        pl_module.metrics.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'test/c2st_{fb}', c2st)
        pl_module.c2st.reset()

        cond_c2st = pl_module.cond_c2st.compute()
        pl_module.log(f'test/cond_c2st_{fb}', cond_c2st)
        pl_module.cond_c2st.reset()

        cond_metrics = pl_module.cond_metrics.compute()
        cond_metrics = {f'test/{k}_{fb}': v for k, v in cond_metrics.items()}
        pl_module.log_dict(cond_metrics)
        pl_module.cond_metrics.reset()

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
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger or CometLogger.'
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
        pred_trajectories = convert_to_numpy(make_grid(pred_trajectories.reshape(-1, self.dim), nrow=nrow))

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
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger or CometLogger.'
            )
        
        plt.close()
