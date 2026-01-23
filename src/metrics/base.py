from typing import Any, Dict, Literal, Tuple, Union
import torch

from lightning.pytorch import Callback, Trainer

from ..methods import DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT


class BaseMetricsCallback(Callback):

    def __init__(self,):
        super().__init__()

    def _init_metrics(
        self,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT],
    ) -> None:
        raise NotImplementedError

    def setup(
        self,
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT], 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if self.benchmark is not None and hasattr(pl_module, 'metrics'):
            return
        
        assert hasattr(trainer.datamodule, 'benchmark'), \
            'Wrong datamodule! It should have `benchmark` attribute'
        self.benchmark = trainer.datamodule.benchmark
        
        # initialize unconditional metrics
        self._init_metrics(pl_module)

    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT],
        outputs: Dict[str, Any],
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        raise NotImplementedError
            
    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT],
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        raise NotImplementedError
            
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT],
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        self._update_metrics(
            trainer, pl_module, outputs, batch_idx, stage='val'
        )
        if was_training:
            pl_module.train()
       
    def on_validation_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT]
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='val'
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT],
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        self._update_metrics(
            trainer, pl_module, outputs, batch_idx, stage='test'
        )
        if was_training:
            pl_module.train()

    def on_test_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM, CNOT]
    ):
        self._compute_and_log_metrics(
            trainer, pl_module, stage='test'
        )
