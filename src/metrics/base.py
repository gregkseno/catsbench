from typing import Any, Dict, Literal, Tuple
import torch

from lightning.pytorch import Callback, LightningModule, Trainer

MetricModule = LightningModule


class BaseMetricsCallback(Callback):

    def __init__(self,):
        super().__init__()

    def _setup_callback(
        self,
        trainer: Trainer,
        pl_module: MetricModule,
        stage: Literal['fit', 'validate', 'test'],
    ) -> None:
        raise NotImplementedError

    def setup(
        self,
        trainer: Trainer, 
        pl_module: MetricModule, 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        self._setup_callback(trainer, pl_module, stage)

    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: MetricModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        raise NotImplementedError
            
    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: MetricModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        raise NotImplementedError
            
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MetricModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        self._update_metrics(
            trainer, pl_module, outputs, batch, batch_idx, stage='val'
        )
        if was_training:
            pl_module.train()
       
    def on_validation_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: MetricModule
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='val'
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: MetricModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        self._update_metrics(
            trainer, pl_module, outputs, batch, batch_idx, stage='test'
        )
        if was_training:
            pl_module.train()

    def on_test_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: MetricModule
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='test'
        )
