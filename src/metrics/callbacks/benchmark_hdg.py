from typing import Any, Dict, Literal, Optional, Tuple
import torch

from torchmetrics import MetricCollection
from lightning.pytorch import Callback, Trainer, LightningModule

from src.metrics.c2st import ClassifierTwoSampleTest
from src.metrics.contingency_similarity import ContingencySimilarity
from src.metrics.tv_complement import TVComplement
from src.metrics.trajectory_kl_divergence import TrajectoryKLDivergence
from benchmark import Benchmark


class BenchmarkHDGMetricsCallback(Callback):
    benchmark: Benchmark

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
        pl_module: LightningModule, 
        stage: Literal['fit', 'validate', 'test']
    ) -> None:
        if hasattr(pl_module, 'metrics'):
            return

        # get benchmark class
        assert hasattr(trainer.datamodule, 'benchmark'), \
            'Wrong datamodule! It should have `benchmark` attribute'
        self.benchmark = trainer.datamodule.benchmark
        
        # initialize unconditional metrics
        pl_module.metrics = MetricCollection(
            {
                'tv_complement': TVComplement(self.dim, self.num_categories),
                'contingency_similarity': ContingencySimilarity(self.dim, self.num_categories),
            },
        )
        # initialize conditional metrics
        if self.benchmark.reversed: 
            pl_module.c2st = ClassifierTwoSampleTest(
                dim=2*self.dim, num_categories=self.num_categories, lr=self.classifier_lr
            )
        else:
            pl_module.cond_metrics = pl_module.metrics.clone(prefix='cond_')
            pl_module.forward_kl_div = TrajectoryKLDivergence(log_prob=True)
            pl_module.reverse_kl_div = TrajectoryKLDivergence(log_prob=True)

    def _compute_metrics(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch_idx: int,
        stage: Literal['train', 'val', 'test'] = 'train',
    ) -> None:
        pl_module.eval()
        x_start, x_end = outputs['x_start'], outputs['x_end']
        pred_x_end = pl_module.sample(x_start)

        # update unconditional metrics
        pl_module.metrics.update(x_end, pred_x_end)

        # update conditional metrics
        if self.benchmark.reversed:
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
            cond_x_end = self.benchmark.sample_target_given_input(repeated_x_start)
            cond_pred_x_end = pl_module.sample(repeated_x_start)
            pl_module.cond_metrics.update(cond_x_end, cond_pred_x_end)
            

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._compute_metrics(
            trainer, pl_module, outputs, batch_idx, stage='val'
        )
       

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        if not trainer.datamodule.hparams.reversed:
            metrics = pl_module.metrics.compute()
            metrics = {f'val/{k}_{fb}': v for k, v in metrics.items()}
            pl_module.log_dict(metrics)
            pl_module.metrics.reset()
            
            cond_metrics = pl_module.cond_metrics.compute()
            cond_metrics = {f'val/{k}_{fb}': v for k, v in cond_metrics.items()}
            pl_module.log_dict(cond_metrics)
            pl_module.cond_metrics.reset()

            kl_div = pl_module.kl_div.compute()
            pl_module.log(f'val/kl_div_{fb}', kl_div)
            pl_module.kl_div.reset()
        else:
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
        if not trainer.datamodule.hparams.reversed:
            pl_module.metrics.update(x_end, pred_x_end)
            
            repeated_x_start = x_start[0].unsqueeze(0).expand(self.num_cond_samples, -1)
            cond_x_end = self.benchmark.sample_target_given_input(repeated_x_start)
            cond_pred_x_end = pl_module.sample(repeated_x_start)
            pl_module.cond_metrics.update(cond_x_end, cond_pred_x_end)

            x = x_start
            for t in range(0, pl_module.prior.num_timesteps + 1):
                t = torch.tensor([t] * x.shape[0], device=pl_module.device)
                # there is must be a better naming for identifying sampling direction
                # but for now all bidirectional methods use reversed time steps during sampling
                if pl_module.bidirectional:
                    pred_transition_logits = pl_module.get_transition_logits(
                        x, pl_module.prior.num_timesteps + 1 - t
                    )
                else:
                    pred_transition_logits = pl_module.get_transition_logits(x, t)
                true_transition_logits = self.benchmark.get_transition_logits(x, t).clamp(min=1e-20)
                pl_module.kl_div.update(
                    true_transition_logits.reshape(-1, self.num_categories).log_softmax(dim=-1),
                    pred_transition_logits.reshape(-1, self.num_categories).log_softmax(dim=-1)
                )

                noise = torch.rand_like(pred_transition_logits)
                noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
                gumbel_noise = -torch.log(-torch.log(noise))
                x = torch.argmax(pred_transition_logits + gumbel_noise, dim=-1)
        else:
            len_data = len(trainer.test_dataloaders) if trainer.limit_test_batches is None else trainer.limit_test_batches
            train_mode = batch_idx < int(len_data * self.train_test_split)
            pl_module.c2st.update(
                real_data=torch.cat([x_start, x_end], dim=-1),
                pred_data=torch.cat([x_start, pred_x_end], dim=-1),
                train=train_mode
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        fb = getattr(pl_module, 'fb', None) or 'forward' 
        
        if not trainer.datamodule.hparams.reversed:
            metrics = pl_module.metrics.compute()
            metrics = {f'test/{k}_{fb}': v for k, v in metrics.items()}
            pl_module.log_dict(metrics)
            pl_module.metrics.reset()

            cond_metrics = pl_module.cond_metrics.compute()
            cond_metrics = {f'test/{k}_{fb}': v for k, v in cond_metrics.items()}
            pl_module.log_dict(cond_metrics)
            pl_module.cond_metrics.reset()

            kl_div = pl_module.kl_div.compute()
            pl_module.log(f'test/kl_div_{fb}', kl_div)
            pl_module.kl_div.reset()
        else:
            c2st = pl_module.c2st.compute()
            pl_module.log(f'test/c2st_{fb}', c2st)
            pl_module.c2st.reset()
