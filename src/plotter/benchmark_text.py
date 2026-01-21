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
from src.metrics.c2st import ClassifierTwoSampleTest
from src.utils.ranked_logger import RankedLogger
from src.metrics.callbacks.benchmark_hdg import BenchmarkTexts


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkTextPlotterCallback(Callback):
    benchmark: BenchmarkTexts

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

        self._log_buf('test', pl_module)

    @rank_zero_only
    def _log_smaples(
        self,
        x_start: torch.Tensor | np.ndarray, 
        x_end: torch.Tensor | np.ndarray, 
        pl_module: LightningModule,
        stage: Literal['train', 'val', 'test'] = 'train',
    ):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        pred_x_end = self.tokenizer.decode(pl_module.sample(x_start)[0].tolist())
        x_start = self.tokenizer.decode(x_start[0].tolist())
        print(x_start)
        x_end = self.tokenizer.decode(x_end[0].tolist())
        print(x_end)

        
        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/samples_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            pl_module.logger.experiment.log_text(
                        x_start, 
                        metadata={
                                    'name': f'{stage}/input_{fb}',
                                    'step': pl_module.global_step,
                                }
                    )

            pl_module.logger.experiment.log_text(
                x_end, 
                metadata={
                                    'name': f'{stage}/target_{fb}',
                                    'step': pl_module.global_step,
                                }
            )

            pl_module.logger.experiment.log_text(
                pred_x_end, 
                metadata={
                                    'name': f'{stage}/prediction_{fb}',
                                    'step': pl_module.global_step,
                                }
            )
        elif isinstance(pl_module.logger, TensorBoardLogger): # can be optimized with add_fig 
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = img_np[:, :, None]            
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/samples_{fb}', img_tensor=img_np, global_step=pl_module.global_step,
                dataformats='HWC'
            )            
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, TensorBoardLogger or CometLogger.'
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
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        iteration = getattr(pl_module, "iteration", None)
        if iteration is not None:
            suptitle += f", Iteration {iteration}"
        
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

        pred_trajectories = pred_trajectories.reshape(-1, self.dim)
        pred_trajectories_text = []
        for ix in range(len(pred_trajectories)):
            pred_trajectories_text.append(self.tokenizer.decode(pred_trajectories[ix].tolist()))
        
        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.log_image(
                key=f'{stage}/trajectories_{fb}', images=[img], step=pl_module.global_step
            )
        elif isinstance(pl_module.logger, CometLogger):
            for i, text in enumerate(pred_trajectories_text):
                pl_module.logger.experiment.log_text(text, metadata={
                                    'name': f'{stage}/trajectories_{fb}_{i}',
                                    'step': pl_module.global_step,
                                })
            
        elif isinstance(pl_module.logger, TensorBoardLogger):
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = img_np[:, :, None]
            pl_module.logger.experiment.add_image(
                tag=f'{stage}/trajectories_{fb}', img_tensor=img_np, global_step=pl_module.global_step,
                dataformats='HWC'
            )            
        else:
            raise ValueError(
                f'Unsupported logger type: {type(pl_module.logger)}. Expected WandbLogger, TensorBoardLogger or CometLogger.'
            )
        
        plt.close()
