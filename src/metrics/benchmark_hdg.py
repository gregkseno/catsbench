from typing import Any, Dict, Literal, Optional, Tuple, Union
import torch

from torchmetrics import MetricCollection
from lightning.pytorch import Callback, Trainer

from catsbench import BenchmarkHDG
from catsbench.metrics import (
    ClassifierTwoSampleTest,
    ShapeScore,
    TrendScore,
    TrajectoryKLDivergence
)

from ..methods import DLightSB, DLightSB_M, CSBM, AlphaCSBM
from ..utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkHDGMetricsCallback(Callback):
    benchmark: Optional[BenchmarkHDG] = None

    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_cond_samples: int,
        train_test_split: Optional[float] = 0.8,
        classifier_lr: Optional[float] = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.num_cond_samples = num_cond_samples
        self.train_test_split = train_test_split
        self.classifier_lr = classifier_lr

    def setup(
        self,
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM], 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if self.benchmark is not None and hasattr(pl_module, 'metrics'):
            return
        
        assert hasattr(trainer.datamodule, 'benchmark'), \
            'Wrong datamodule! It should have `benchmark` attribute'
        self.benchmark = trainer.datamodule.benchmark
        
        # initialize unconditional metrics
        pl_module.metrics = MetricCollection(
            {'shape_score': ShapeScore(self.dim, self.num_categories, conditional=False),
             'trend_score': TrendScore(self.dim, self.num_categories, conditional=False)},
        )
        # initialize conditional metrics
        if self.benchmark.reverse: 
            pl_module.c2st = ClassifierTwoSampleTest(
                dim=2*self.dim, num_categories=self.num_categories, lr=self.classifier_lr
            )
        else:
            pl_module.cond_metrics = MetricCollection(
                {'cond_shape_score': ShapeScore(self.dim, self.num_categories, conditional=True),
                 'cond_trend_score': TrendScore(self.dim, self.num_categories, conditional=True)},
            )
            pl_module.forward_kl_div = TrajectoryKLDivergence(logits=True)
            pl_module.reverse_kl_div = TrajectoryKLDivergence(logits=True)

    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        outputs: Dict[str, Any],
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        pl_module.eval()
        x_start, x_end = outputs['x_start'], outputs['x_end']

        # update unconditional metrics
        pred_x_end = pl_module.sample(x_start)
        pl_module.metrics.update(x_end, pred_x_end)

        # update conditional metrics
        if self.benchmark.reverse:
            loader_attr = "train_dataloader" if stage == "train" else f"{stage}_dataloaders"
            limit = getattr(trainer, f"limit_{stage}_batches")
            loader = getattr(trainer, loader_attr)
            num_batches = limit if limit is not None else len(loader)
            train_mode = batch_idx < int(num_batches * self.train_test_split)

            pl_module.c2st.update(
                real_data=torch.cat([x_start, x_end], dim=-1),
                pred_data=torch.cat([x_start, pred_x_end], dim=-1),
                train=train_mode
            )
        else:
            repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
            cond_x_end = self.benchmark.sample(repeated_x_start)
            cond_pred_x_end = pl_module.sample(repeated_x_start)
            pl_module.cond_metrics.update(cond_x_end, cond_pred_x_end)

            true_trajectory, true_transition_logits = self.benchmark.sample_trajectory(x_start, return_transitions=True)
            pred_trajectory, pred_transition_logits = pl_module.sample_trajectory(x_start, return_transitions=True)
            
            # we need only num_steps + 1 points to compute transitions
            true_trajectory = true_trajectory[:-1]
            pred_trajectory = pred_trajectory[:-1]
            
            timesteps = torch.arange(true_trajectory.shape[0], device=pl_module.device)
            timesteps = timesteps.repeat_interleave(true_trajectory.shape[1])
            
            true_trajectory = true_trajectory.flatten(end_dim=1)
            pred_trajectory = pred_trajectory.flatten(end_dim=1)
            true_transition_logits = true_transition_logits.flatten(end_dim=1)
            pred_transition_logits = pred_transition_logits.flatten(end_dim=1)

            # the KL div must be computed in cross fashion:
            # forward KL is KL with respect to true trajectory
            # reverse KL is KL with respect to predicted trajectory
            pl_module.reverse_kl_div.update(
                p=pred_transition_logits, 
                q=self.benchmark.get_transition_logits(pred_trajectory, timesteps)
            )
            if isinstance(pl_module, (CSBM, AlphaCSBM)):
                timesteps = (timesteps + 1).flip(dims=[0])
            pl_module.forward_kl_div.update(
                p=true_transition_logits, 
                q=pl_module.get_transition_logits(true_trajectory, timesteps)
            )
            

    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        # compute and log unconditional metrics
        metrics = pl_module.metrics.compute()
        metrics = {f'{stage}/{k}_{fb}': v for k, v in metrics.items()}
        pl_module.log_dict(metrics)
        pl_module.metrics.reset()

        # compute and log conditional metrics
        if self.benchmark.reverse:
            c2st = pl_module.c2st.compute()
            pl_module.log(f'{stage}/c2st_{fb}', c2st)
            pl_module.c2st.reset()
        else:
            cond_metrics = pl_module.cond_metrics.compute()
            cond_metrics = {f'{stage}/{k}_{fb}': v for k, v in cond_metrics.items()}
            pl_module.log_dict(cond_metrics)
            pl_module.cond_metrics.reset()

            forward_kl_div = pl_module.forward_kl_div.compute()
            pl_module.log(f'{stage}/forward_kl_div_{fb}', forward_kl_div)
            pl_module.forward_kl_div.reset()

            reverse_kl_div = pl_module.reverse_kl_div.compute()
            pl_module.log(f'{stage}/reverse_kl_div_{fb}', reverse_kl_div)
            pl_module.reverse_kl_div.reset()
            
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._update_metrics(
            trainer, pl_module, outputs, batch_idx, stage='val'
        )
       
    def on_validation_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM]
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='val'
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._update_metrics(
            trainer, pl_module, outputs, batch_idx, stage='test'
        )

    def on_test_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM]
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='test'
        )
