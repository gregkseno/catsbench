from typing import Any, Literal
import numpy as np
import torch

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only

from ..data.batch import Batch
from ..methods import BaseMethod

class BasePlotterCallback(Callback):

    def __init__(
        self,
    ):
        """Initializes buffers for storing samples and trajectories."""
        super().__init__()
        self._buffers = {
            stage: {
                'x_start': [], 'x_end': [],
                'raw_x_start': [], 'raw_x_end': [],
            }
            for stage in ('train', 'val', 'test')
        }

    def _reset_buf(self, stage: Literal['train', 'val', 'test']) -> None:
        for values in self._buffers[stage].values():
            values.clear()

    def _accumulate_buf(
        self, 
        stage: Literal['train', 'val', 'test'],
        x_start: torch.Tensor, 
        x_end: torch.Tensor,
        raw_x_start: torch.Tensor,
        raw_x_end: torch.Tensor,
    ) -> None:
        buf = self._buffers[stage]
        have = sum(t.shape[0] for t in buf['x_start'])
        remain = self.num_samples - have
        if remain <= 0:
            return
        take = min(remain, x_start.shape[0])
        buf['x_start'].append(x_start[:take].detach())
        buf['x_end'].append(x_end[:take].detach())
        buf['raw_x_start'].append(raw_x_start[:take].detach())
        buf['raw_x_end'].append(raw_x_end[:take].detach())

    def _log_buf(self, stage: Literal['train', 'val', 'test'], pl_module: BaseMethod) -> None:
        buf = self._buffers[stage]
        if not buf['x_start']:
            return
        x_start = torch.cat(buf['x_start'], dim=0)[:self.num_samples]
        x_end = torch.cat(buf['x_end'], dim=0)[:self.num_samples]
        raw_x_start = torch.cat(buf['raw_x_start'], dim=0)[:self.num_samples]
        raw_x_end = torch.cat(buf['raw_x_end'], dim=0)[:self.num_samples]
        self._log_samples(
            x_start, x_end, pl_module, stage,
            raw_x_start=raw_x_start,
            raw_x_end=raw_x_end,
        )
        self._log_trajectories(
            x_start, x_end, pl_module, stage=stage,
            raw_x_start=raw_x_start,
            raw_x_end=raw_x_end,
        )
        self._reset_buf(stage)

    def _accumulate_batch(
        self,
        stage: Literal['train', 'val', 'test'],
        batch: Batch,
        pl_module: BaseMethod,
    ) -> None:
        x_start, x_end = batch.encoded
        raw_x_start, raw_x_end = batch.raw
        if getattr(pl_module, 'fb', None) == 'backward':
            x_start, x_end = x_end, x_start
            raw_x_start, raw_x_end = raw_x_end, raw_x_start
        self._accumulate_buf(
            stage, x_start, x_end, raw_x_start, raw_x_end
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        self._reset_buf('train')

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        self._accumulate_batch('train', batch, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        self._log_buf('train', pl_module)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        self._reset_buf('val')

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        self._accumulate_batch('val', batch, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseMethod):
        self._log_buf('val', pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: BaseMethod) -> None:
        self._reset_buf('test')

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        self._accumulate_batch('test', batch, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseMethod):
        self._log_buf('test', pl_module)

    @rank_zero_only
    def _log_samples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        raw_x_start: torch.Tensor | np.ndarray | None = None,
        raw_x_end: torch.Tensor | np.ndarray | None = None,
        fb: Literal['forward', 'backward'] = 'forward',
    ):
        raise NotImplementedError

    @rank_zero_only
    def _log_trajectories(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray,
        pl_module: BaseMethod,
        stage: Literal['train', 'val', 'test'] = 'train',
        raw_x_start: torch.Tensor | np.ndarray | None = None,
        raw_x_end: torch.Tensor | np.ndarray | None = None,
        fb: Literal['forward', 'backward'] = 'forward',
    ):
        raise NotImplementedError
