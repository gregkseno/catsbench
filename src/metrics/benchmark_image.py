from typing import Any, Dict, Literal, Tuple, Union
import torch

from torchmetrics.image import FrechetInceptionDistance
from lightning.pytorch import Trainer

from catsbench import BenchmarkImage
from catsbench.metrics import ClassifierTwoSampleTest

from .base import BaseMetricsCallback
from ..methods import DLightSB, DLightSB_M, CSBM, AlphaCSBM


class BenchmarkImageMetricsCallback(BaseMetricsCallback):
    benchmark: BenchmarkImage

    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, int, int],
        num_categories: int,
        train_test_split: float = 0.8,
        classifier_lr: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.num_categories = num_categories

        self.classifier_lr = classifier_lr
        self.train_test_split = train_test_split

    def _init_metrics(
        self,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM], 
    ) -> None:
        # initialize metrics
        pl_module.fid = FrechetInceptionDistance(normalize=True)
        pl_module.c2st = ClassifierTwoSampleTest(
            dim=2*self.dim, num_categories=self.num_categories, lr=self.classifier_lr
        )

    def _update_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        outputs: Dict[str, Any],
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        x_start, x_end = outputs['x_start'], outputs['x_end']

        # update unconditional metrics
        pred_x_end = pl_module.sample(x_start)
        pl_module.fid.update(x_end, real=True)
        pl_module.fid.update(pred_x_end, real=False)

        # update conditional metrics
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
    
    def _compute_and_log_metrics(
        self,
        trainer: Trainer,
        pl_module: Union[DLightSB, DLightSB_M, CSBM, AlphaCSBM],
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        fid = pl_module.fid.compute()
        pl_module.log(f'{stage}/fid_{fb}', fid)
        pl_module.fid.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'{stage}/c2st_{fb}', c2st)
        pl_module.c2st.reset()
