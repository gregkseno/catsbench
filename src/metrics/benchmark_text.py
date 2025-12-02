from typing import Any, Dict, Literal, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger, CometLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from transformers import AutoModelWithLMHead, AutoTokenizer
from torchmetrics.text import Perplexity
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


from src.utils import convert_to_numpy, fig2img
from src.utils.ranked_logger import RankedLogger
from benchmark import BenchmarkText
from benchmark.metrics import ClassifierTwoSampleTest


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkTextMetricsCallback(Callback):
    benchmark: BenchmarkText

    def __init__(
        self,
        dim: int,
        num_categories: int,
        train_test_split: float,
        num_samples: int, 
        num_trajectories: int, 
        num_translations: int,
        model_path: str
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.train_test_split = train_test_split
        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_translations = num_translations
        self._buffers = {
            stage: {'x_start': [], 'x_end': []} \
                for stage in ('train', 'val', 'test')
        }

        self.model     = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path, local_files_only=True, padding_side='left')

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

        self.model = self.model.to(pl_module.device)

        # initialize metrics
        pl_module.perplexity = Perplexity()
        pl_module.c2st = ClassifierTwoSampleTest(2*self.dim, model='linear')


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
        pred_x_end = pl_module.sample(x_start)

        with torch.no_grad():

            pred_x_end_out = self.model(pred_x_end, labels=pred_x_end)
            pred_x_end_logits = pred_x_end_out.logits

        pl_module.perplexity.update(pred_x_end_logits[:, :-1], x_end[:, 1:])
        pl_module.c2st.update(
            torch.cat([x_start, x_end], dim=1), 
            torch.cat([x_start, pred_x_end], dim=1),
            train=batch_idx < int(len(trainer.val_dataloaders) * self.train_test_split)
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        perplexity = pl_module.perplexity.compute()
        pl_module.log(f'val/perplexity_{fb}', perplexity)
        pl_module.perplexity.reset()

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
        pred_x_end = pl_module.sample(x_start)

        with torch.no_grad():

            pred_x_end_out = self.model(pred_x_end, labels=pred_x_end)
            pred_x_end_logits = pred_x_end_out.logits

        pl_module.perplexity.update(pred_x_end_logits[:, :-1], x_end[:, 1:])
        pl_module.c2st.update(
            torch.cat([x_start, x_end], dim=1),
            torch.cat([x_start, pred_x_end], dim=1),
            train=batch_idx < int(len(trainer.test_dataloaders) * self.train_test_split)
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        perplexity = pl_module.perplexity.compute()
        pl_module.log(f'val/perplexity_{fb}', perplexity)
        pl_module.perplexity.reset()

        c2st = pl_module.c2st.compute()
        pl_module.log(f'test/c2st_{fb}', c2st)
        pl_module.c2st.reset()
