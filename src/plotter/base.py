from typing import Any, Dict, Literal, Tuple
import numpy as np
import torch

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only


class BasePlotterCallback(Callback):
    dim: int
    num_categories: int
    num_samples: int
    num_trajectories: int
    num_translations: int

    def __init__(
        self,
    ):
        """Initializes buffers for storing samples and trajectories."""
        super().__init__()
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
        x_start, x_end = outputs['x_start'], outputs['x_end']
        self._accumulate_buf('val', x_start, x_end)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
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
        x_start, x_end = outputs['x_start'], outputs['x_end']
        self._accumulate_buf('test', x_start, x_end)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self._log_buf('test', pl_module)

    @rank_zero_only
    def _log_smaples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        raise NotImplementedError

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray,
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        raise NotImplementedError