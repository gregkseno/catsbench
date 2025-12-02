from typing import Any, Dict, Literal, Tuple
import torch

from torchmetrics.image import FrechetInceptionDistance
from lightning.pytorch import Callback, Trainer, LightningModule

from src.metrics.c2st import ClassifierTwoSampleTest
from benchmark import BenchmarkImage


class BenchmarkImageMetricsCallback(Callback):
    benchmark: BenchmarkImage

    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, int, int],
        num_categories: int,
        train_test_split: float,
    ):
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.train_test_split = train_test_split

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
        pl_module.c2st = ClassifierTwoSampleTest(dim=2*self.dim)

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
        len_data = len(trainer.val_dataloaders) if trainer.limit_val_batches is None else trainer.limit_val_batches
        train_mode = batch_idx < int(len_data * self.train_test_split)
        with torch.inference_mode(not train_mode):
            pl_module.c2st.update(
                torch.cat([x_start, x_end], dim=1),
                torch.cat([x_start, pred_x_end], dim=1),
                train=train_mode
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        fid = pl_module.fid.compute()
        pl_module.log(f'val/fid_{fb}', fid)
        pl_module.fid.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'val/c2st_{fb}', c2st)
        pl_module.c2st.reset()

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
        len_data = len(trainer.test_dataloaders) if trainer.limit_test_batches is None else trainer.limit_test_batches
        train_mode = batch_idx < int(len_data * self.train_test_split)
        with torch.inference_mode(not train_mode):
            pl_module.c2st.update(
                torch.cat([x_start, x_end], dim=1),
                torch.cat([x_start, pred_x_end], dim=1),
                train=train_mode
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        # fast metric first
        c2st = pl_module.c2st.compute()
        pl_module.log(f'test/c2st_{fb}', c2st)
        pl_module.c2st.reset()

        fid = pl_module.fid.compute()
        pl_module.log(f'val/fid_{fb}', fid)
        pl_module.fid.reset()
