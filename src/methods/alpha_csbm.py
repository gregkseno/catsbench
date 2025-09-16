from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch_ema import ExponentialMovingAverage 
from lightning import LightningModule

from src.data.prior import Prior
from src.utils import optimize_coupling
from src.utils.logging.console import RankedLogger


HPARAMS = (
    'kl_loss_coeff', 'ce_loss_coeff', 'mse_loss_coeff', 
    'use_mini_batch', 'ignore_index', 'num_first_iterations',
    'optimizer', 'scheduler', 'argmax_mode'
)
log = RankedLogger(__name__, rank_zero_only=True)

# NOTE: start and end is swapped because alpha-CSBM uses 
# reverse diffusion notation
class AlphaCSBM(LightningModule):
    def __init__(
        self,
        prior: Prior,
        model: nn.Module,
        ema: ExponentialMovingAverage, # partially initialized
        optimizer: Optimizer, # partially initialized 
        num_first_iterations: int,
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        kl_loss_coeff: float = 1.0,
        ce_loss_coeff: float = 0.001,
        mse_loss_coeff: float = 0.0,
        use_mini_batch: bool = False,
        ignore_index: int = -100,
        argmax_mode: bool = True
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)
        self.bidirectional = False
        self.first_iteration = True
        self.iteration = 1
        self.prior = prior
        
        # models explicitly stated to be able log parameters
        self.model_forward = model
        self.model_backward = deepcopy(model)
        self.models = {
            'forward': self.model_forward,
            'backward': self.model_backward,
        }
        self.emas: Dict[str, ExponentialMovingAverage] = {
            'forward': ema(self.models['forward'].parameters()),
            'backward': ema(self.models['backward'].parameters())
        }
    
    def kl_loss(
        self,
        true_q_posterior_logits: torch.Tensor, 
        pred_q_posterior_logits: torch.Tensor,
    ) -> torch.Tensor:        
        '''KL-divergence calculation.'''    
        kl_loss = torch.softmax(true_q_posterior_logits, dim=-1) * (
            torch.log_softmax(true_q_posterior_logits, dim=-1)
            - torch.log_softmax(pred_q_posterior_logits, dim=-1)
        )
        kl_loss = kl_loss.sum(dim=-1).mean()
        return kl_loss
        
    def mse_loss(
        self,
        true_q_posterior_logits: torch.Tensor, 
        pred_q_posterior_logits: torch.Tensor,
    ) -> torch.Tensor:        
        '''MSE calculation.'''    
        mse_loss = F.mse_loss(
            torch.softmax(true_q_posterior_logits, dim=-1), 
            torch.softmax(pred_q_posterior_logits, dim=-1)
        )
        return mse_loss

    def ce_loss(
        self,
        true_x_start: torch.Tensor, 
        pred_x_start_logits: torch.Tensor, 
    ) -> torch.Tensor:   
        '''Cross-Entropy calculation.'''         
        pred_x_start_logits = pred_x_start_logits.flatten(start_dim=0, end_dim=-2)
        true_x_start = true_x_start.flatten(start_dim=0, end_dim=-1).long()
        ce_loss = F.cross_entropy(pred_x_start_logits, true_x_start, ignore_index=self.hparams.ignore_index)
        return ce_loss

    # EMA does not inherit nn.Module so it must be 
    # putted to device manually 
    def setup(self, stage: Literal['fit', 'validate', 'test']) -> None:
        self.emas['forward'].to(self.device)
        self.emas['backward'].to(self.device)

    def on_train_epoch_start(self) -> None:
        self.first_iteration = self.current_epoch + 1 < self.hparams.num_first_iterations
    
    def markovian_projection(
        self,
        fb: Literal['forward', 'backward'],
        true_x_start: torch.Tensor,
        true_x_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  
        batch_size = true_x_start.shape[0]
        t = torch.randint(
            low=1, high=self.prior.num_timesteps + 2,
            size=(batch_size,), device=self.device
        )
        x_t = self.prior.sample_bridge(true_x_start, true_x_end, t)

        pred_x_start_logits = self.models[fb](x_t, t)
        true_q_posterior_logits = self.prior.posterior_logits(true_x_start, x_t, t, logits=False)
        pred_q_posterior_logits = self.prior.posterior_logits(pred_x_start_logits, x_t, t, logits=True)

        kl = self.kl_loss(true_q_posterior_logits, pred_q_posterior_logits)
        ce = 0.0 if self.hparams.ce_loss_coeff == 0 else self.ce_loss(true_x_start, pred_x_start_logits)
        mse = 0.0 if self.hparams.mse_loss_coeff == 0 else self.mse_loss(true_q_posterior_logits, pred_q_posterior_logits)

        loss = self.hparams.kl_loss_coeff * kl + \
               self.hparams.ce_loss_coeff * ce + \
               self.hparams.mse_loss_coeff * mse
        
        info = {f'kl_loss_{fb}': kl, f'ce_loss_{fb}': ce, f'mse_loss_{fb}': mse}
        return loss, info

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        b = batch[0].shape[0] // 2
        x_end, x_start = batch
        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        outputs = {'x_start': x_end, 'x_end': x_start} # For logger

        if self.first_iteration:
            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], x_end[:b])
            loss_backward, info_backward = self.markovian_projection('backward', x_end[b:], x_start[b:])
        else:
            pred_x_end = self.sample(x_start[:b], fb='backward')
            pred_x_start = self.sample(x_end[:b], fb='forward')
            outputs['x_start'] = pred_x_end # for visualization forward training paris

            loss_forward, info_forward = self.markovian_projection('forward', x_start[:b], pred_x_end)
            loss_backward, info_backward = self.markovian_projection('backward', x_end[:b], pred_x_start)
        outputs['loss'] = (loss_forward + loss_backward) / 2
        
        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"train/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)

        return outputs
    
    def on_before_zero_grad(self, optimizer) -> None:
        self.emas['forward'].update()
        self.emas['backward'].update()

    def on_train_epoch_end(self) -> None:
        if not self.first_iteration: 
            self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        b = batch[0].shape[0] // 2
        x_end, x_start = batch
        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        outputs = {'x_start': x_end, 'x_end': x_start} # For logger

        if self.first_iteration:
            _, info_forward = self.markovian_projection('forward', x_start[:b], x_end[:b])
            _, info_backward = self.markovian_projection('backward', x_end[b:], x_start[b:])
        else:
            pred_x_end = self.sample(x_start[:b], fb='backward')
            pred_x_start = self.sample(x_end[:b], fb='forward')

            _, info_forward = self.markovian_projection('forward', x_start[:b], pred_x_end)
            _, info_backward = self.markovian_projection('backward', x_end[:b], pred_x_start)
        
        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"val/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        b = batch[0].shape[0] // 2
        x_end, x_start = batch
        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        outputs = {'x_start': x_end, 'x_end': x_start} # For logger

        if self.first_iteration:
            _, info_forward = self.markovian_projection('forward', x_start[:b], x_end[:b])
            _, info_backward = self.markovian_projection('backward', x_end[b:], x_start[b:])
        else:
            pred_x_end = self.sample(x_start[:b], fb='backward')
            pred_x_start = self.sample(x_end[:b], fb='forward')

            _, info_forward = self.markovian_projection('forward', x_start[:b], pred_x_end)
            _, info_backward = self.markovian_projection('backward', x_end[:b], pred_x_start)
        
        # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
        info = {f"test/{k}": v for k, v in {**info_forward, **info_backward}.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return outputs

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer  = self.hparams.optimizer(
            params=[
                {"params": self.model_forward.parameters()},
                {"params": self.model_backward.parameters()},
            ]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['ema_forward'] = self.emas['forward'].state_dict()
        checkpoint['ema_backward'] = self.emas['backward'].state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'ema_forward' in checkpoint:
            self.emas['forward'].load_state_dict(checkpoint['ema_forward'])
            self.emas['forward'].to(self.device)
        if 'ema_backward' in checkpoint:
            self.emas['backward'].load_state_dict(checkpoint['ema_backward'])
            self.emas['backward'].to(self.device)

    def markov_sample(
        self, x: torch.Tensor, t: torch.Tensor, fb: Literal['forward', 'backward']
    ) -> torch.Tensor:
        with self.emas[fb].average_parameters():
            pred_x_start_logits = self.models[fb](x, t)
        pred_q_posterior_logits = self.prior.posterior_logits(pred_x_start_logits, x, t, logits=True)
        noise = torch.rand_like(pred_q_posterior_logits)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        random_samples = torch.argmax(pred_q_posterior_logits + gumbel_noise, dim=-1)

        if self.hparams.argmax_mode:
            first_step = (t == 1).long().view((x.shape[0], *[1] * (x.dim() - 1)))        
            argmax_samples = pred_q_posterior_logits.argmax(dim=-1)
            samples = first_step * argmax_samples + (1 - first_step) * random_samples
            return samples
        else:
            return random_samples
        
    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, fb: Optional[Literal['forward', 'backward']] = None
    ) -> torch.Tensor:
        """Sample from the model starting from `x` returning the final sample."""
        fb = 'forward' if fb is None else fb
        was_training = self.models[fb].training  
        self.models[fb].eval()
        for t in reversed(range(1, self.prior.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, fb)
        if was_training: self.models[fb].train()
        return x
    
    @torch.no_grad()
    def sample_trajectory(
        self, x: torch.Tensor, fb: Optional[Literal['forward', 'backward']] = None
    ) -> torch.Tensor:
        """Sample from the model starting from `x` returning the full trajectory."""
        fb = 'forward' if fb is None else fb
        was_training = self.models[fb].training  
        self.models[fb].eval()
        trajectory = [x]
        for t in reversed(range(1, self.prior.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, fb)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        if was_training: self.models[fb].train()
        return trajectory
