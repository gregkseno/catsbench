from typing import Any, Dict, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from torchmetrics import MetricCollection
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger, CometLogger

from src.utils import convert_to_numpy, fig2img
from src.metrics.contingency_similarity import ContingencySimilarity
from src.metrics.tv_complement import TVComplement
from src.metrics.pqmass import PQMass
from src.utils.logging.console import RankedLogger
from src.benchmark import BenchmarkDiscreteEOT


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkLogger(Callback):
    benchmark: BenchmarkDiscreteEOT

    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_cond_samples: int,
        num_refs: int,
        kernel: str,
        num_trajectories: int, 
        num_translations: int,
        re_tessellation: Optional[int] = None,
        permute_tests: Optional[int] = None,
        axlim: Optional[Tuple[float, float]] = None,
        samples_figsize: Optional[Tuple[int, int]] = None,
        trajectories_figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 100
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.num_cond_samples = num_cond_samples
        self.num_refs = num_refs
        self.re_tessellation = re_tessellation
        self.permute_tests = permute_tests
        self.kernel = kernel
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        if dim > 2:
            self.pca = PCA(n_components=2)

        self.tv_complement = TVComplement(dim, num_categories)
        self.contingency_similarity = ContingencySimilarity(dim, num_categories)
        self.pqmass = PQMass(dim, num_refs, re_tessellation, permute_tests, kernel)

        self.axlim = [7, 93] if axlim is None else axlim
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

        # initialize PCA
        self.pca.fit(convert_to_numpy(
            torch.cat([self.benchmark.input_dataset, self.benchmark.target_dataset], dim=0)
        ))

        # initialize metrics
        pl_module.metrics = MetricCollection(
            {
                'tv_complement': TVComplement(self.dim, self.num_categories),
                'contingency_similarity': ContingencySimilarity(self.dim, self.num_categories),
                'pqmass': PQMass(self.dim, self.num_refs, self.re_tessellation, self.permute_tests, self.kernel)
            },
        )
        pl_module.cond_metrics = pl_module.metrics.clone(prefix='cond_')

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

        if batch_idx == 0:
            self._log_smaples(x_start, x_end, pl_module, 'train')   
            self._log_trajectories(x_start, pl_module, stage='train')

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

        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        pred_x_end = pl_module.sample(x_start)
        metrics = pl_module.metrics(x_end, pred_x_end)
        metrics = {f'val/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        
        repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
        cond_x_end = self.benchmark.sample_target_given_input(repeated_x_start)
        cond_pred_x_end = pl_module.sample(repeated_x_start)
        cond_metrics = pl_module.cond_metrics(cond_x_end, cond_pred_x_end)
        cond_metrics = {f'val/{k}_{fb}': v for k, v in cond_metrics.items()}
        pl_module.log_dict(cond_metrics)

        if batch_idx == len(trainer.val_dataloaders) - 1:
            self._log_smaples(x_start, x_end, pl_module, 'val')
            self._log_trajectories(x_start, pl_module, stage='val')

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

        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        pred_x_end = pl_module.sample(x_start)
        metrics = pl_module.metrics(x_end, pred_x_end)
        metrics = {f'test/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        
        repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
        cond_x_end = self.benchmark.sample_target_given_input(repeated_x_start)
        cond_pred_x_end = pl_module.sample(repeated_x_start)
        cond_metrics = pl_module.cond_metrics(cond_x_end, cond_pred_x_end)
        cond_metrics = {f'test/{k}_{fb}': v for k, v in cond_metrics.items()}
        pl_module.log_dict(cond_metrics)

        if batch_idx == len(trainer.test_dataloaders) - 1:
            self._log_smaples(x_start, x_end, pl_module, 'test')
            self._log_trajectories(x_start, pl_module, stage='test')

    @rank_zero_only
    def _log_smaples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        fb = 'forward' if not pl_module.bidirectional or pl_module.current_epoch % 2 == 0 else 'backward'
        pred_x_end = self.pca.transform(convert_to_numpy(pl_module.sample(x_start)))
        x_start = self.pca.transform(convert_to_numpy(x_start))
        x_end = self.pca.transform(convert_to_numpy(x_end))

        fig, axes = plt.subplots(1, 3, **self.samples_fig_config)
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.iteration}')

        axes[0].scatter(x_start[:, 0], x_start[:, 1], **self.samples_start_config) 
        axes[1].scatter(x_end[:, 0], x_end[:, 1], **self.samples_end_config)
        axes[2].scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.samples_pred_config) 
        
        for i in range(3):
            axes[i].grid()
            axes[i].set_xlim(self.axlim)
            axes[i].set_ylim(self.axlim)
            axes[i].legend(loc='lower left')
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
        fig, axs = plt.subplots(1, 2, **self.trajectories_fig_config)
        for i in range(2):
            axs[i].get_xaxis().set_ticklabels([])
            axs[i].get_yaxis().set_ticklabels([])
        fig.suptitle(f'Epoch {pl_module.current_epoch}, Iteration {pl_module.iteration}')
        
        pred_x_end = self.pca.transform(convert_to_numpy(pl_module.sample(x_start)))
        traj_start = x_start[:self.num_trajectories]
        repeats = [self.num_translations] + [1] * traj_start.dim()
        traj_start = traj_start.unsqueeze(0).repeat(*repeats)
        traj_start = traj_start.reshape(-1, *x_start.shape[1:])

        # ground truth trajectories
        trajectories = self.benchmark.sample_target_given_input(
            traj_start, return_trajectories=True
        )
        trajectories = convert_to_numpy(trajectories.reshape(-1, self.dim))
        trajectories = self.pca.transform(trajectories)
        trajectories = trajectories.reshape(-1, self.num_trajectories * self.num_translations, 2)

        axs[0].scatter(x_end[:, 0], x_end[:, 1], **self.trajectories_pred_config)
        axs[0].scatter(trajectories[0, :, 0], trajectories[0, :, 1], **self.trajectories_start_config)
        axs[0].scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], **self.trajectories_end_config)
        for i in range(self.num_trajectories * self.num_translations):
            axs[0].plot(trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['back'])
            axs[0].plot(
                trajectories[:, i, 0], trajectories[:, i, 1], **self.trajectory_lines_config['front'], 
                label='Trajectory (fitted)' if i == 0 else ''
            )

        # model's trajectories
        pred_trajectories = pl_module.sample_trajectory(traj_start)
        num_timesteps = pred_trajectories.shape[0]
        if num_timesteps > 10:
            pred_trajectories = torch.stack([
                    pred_trajectories[0], 
                    pred_trajectories[num_timesteps // 8], 
                    pred_trajectories[num_timesteps // 2], 
                    pred_trajectories[(num_timesteps * 7) // 8], 
                    pred_trajectories[-1]
                ], dim=0
            )
        pred_trajectories = convert_to_numpy(pred_trajectories.reshape(-1, self.dim))
        pred_trajectories = self.pca.transform(pred_trajectories)
        pred_trajectories = pred_trajectories.reshape(-1, self.num_trajectories * self.num_translations, 2)

        axs[1].scatter(pred_x_end[:, 0], pred_x_end[:, 1], **self.trajectories_pred_config)
        axs[1].scatter(pred_trajectories[0, :, 0], pred_trajectories[0, :, 1], **self.trajectories_start_config)
        axs[1].scatter(pred_trajectories[-1, :, 0], pred_trajectories[-1, :, 1], **self.trajectories_end_config)
        for i in range(self.num_trajectories * self.num_translations):
            axs[1].plot(pred_trajectories[:, i, 0], pred_trajectories[:, i, 1], **self.trajectory_lines_config['back'])
            axs[1].plot(
                pred_trajectories[:, i, 0], pred_trajectories[:, i, 1], **self.trajectory_lines_config['front'], 
                label='Trajectory (ground truth)' if i == 0 else ''
            )
        
        for i in range(2):
            axs[i].legend(loc='lower left')
            axs[i].set_xlim(self.axlim)
            axs[i].set_ylim(self.axlim)
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
