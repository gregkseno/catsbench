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
    'use_mini_batch', 'ignore_index', 'num_first_iterations', 'accumulate_grad_batches',
    'optimizer', 'scheduler', 'argmax_mode'
)
log = RankedLogger(__name__, rank_zero_only=True)

# NOTE: start and end is swapped because CSBM uses 
# reverse diffusion notation
class CSBM(LightningModule):
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
        accumulate_grad_batches: int = 1,
        argmax_mode: bool = True
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        # save only `HPARAMS` for memory efficiency (probably :))
        self.save_hyperparameters(*HPARAMS, logger=False)
        self.bidirectional = True
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
    
        self.automatic_optimization = False

    def load_state_dict(self, state_dict, strict: bool = True):
        ignored = {"c2st.weight", "c2st.bias", "cond_c2st.weight", "cond_c2st.bias"}
        filtered = {k: v for k, v in state_dict.items() if k not in ignored}
        missing, unexpected = LightningModule.load_state_dict(self, filtered, strict=False)

        filtered_out = [k for k in state_dict if k in ignored]
        if filtered_out:
            log.info(f"Ignored keys during load_state_dict: {filtered_out}")
        if missing:
            log.info(f"Missing keys after load (expected): {missing}")
        if unexpected:
            log.info(f"Unexpected keys (ignored by strict=False): {unexpected}")
        return missing, unexpected

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
        pred_x_start_logits = pred_x_start_logits.flatten(start_dim=0, end_dim=-2).float()
        true_x_start = true_x_start.flatten(start_dim=0, end_dim=-1).long()
        ce_loss = F.cross_entropy(pred_x_start_logits, true_x_start, ignore_index=self.hparams.ignore_index)
        return ce_loss

    # EMA does not inherit nn.Module so it must be 
    # putted to device manually 
    def setup(self, stage: Literal['fit', 'validate', 'test']) -> None:
        self.emas['forward'].to(self.device)
        self.emas['backward'].to(self.device)

    def on_train_epoch_start(self) -> None:
        # twice, for each direction
        self.first_iteration = self.current_epoch + 1 < (2 * self.hparams.num_first_iterations)

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
        fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
        bf = 'backward' if fb == 'forward' else 'forward'
        self.models[fb].train()
        self.models[bf].eval()
        
        # DataLoader have the x_0, x_1 order so we need to swap
        if fb == 'forward': x_end, x_start = batch
        else: x_start, x_end = batch

        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if not self.first_iteration:
            x_start, x_end = x_start, self.sample(x_start, fb=bf) 
        outputs = {'x_start': x_end, 'x_end': x_start} # For Logger

        optimizers = {
            'forward': self.optimizers()[0],
            'backward': self.optimizers()[1],
        }
        if self.lr_schedulers() is not None:
            schedulers = {
                'forward': self.lr_schedulers()[0],
                'backward': self.lr_schedulers()[1],
            }
        with self.toggled_optimizer(optimizers[fb]):
            loss, info = self.markovian_projection(fb, x_start, x_end)
            # logs step-wise loss, `add_dataloader_idx=False` is used to have custom fb prefix
            info = {f"train/{k}": v for k, v in info.items()}
            self.log_dict(info, prog_bar=True, sync_dist=True) 
            self.log('train/iteration', self.iteration, prog_bar=True)

            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            # do gradient accumulation and clipping
            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
                self.clip_gradients(
                    optimizers[fb], gradient_clip_val=self.trainer.gradient_clip_val
                )
                optimizers[fb].step()
                if self.lr_schedulers() is not None:
                    # pass loss in case scheduler requires metrics like ReduceLROnPlateau
                    schedulers[fb].step(loss.detach()) 
                self.emas[fb].update()
                optimizers[fb].zero_grad()
        return outputs

    def on_train_epoch_end(self) -> None:
        fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
        if fb == 'backward' and not self.first_iteration: 
            # if the ended epoch corresponds to `backward`` 
            # then in next itreation we will be a new iteration
            self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
        bf = 'backward' if fb == 'forward' else 'forward'
        
        # DataLoader have the x_0, x_1 order so we need to swap
        if fb == 'forward': x_end, x_start = batch
        else: x_start, x_end = batch
        outputs = {'x_start': x_end, 'x_end': x_start} # For logger

        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if not self.first_iteration:
            x_start, x_end = x_start, self.sample(x_start, fb=bf)

        _, info = self.markovian_projection(fb, x_start, x_end)
        info = {f"val/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
        bf = 'backward' if fb == 'forward' else 'forward'
        
        # DataLoader have the x_0, x_1 order so we need to swap
        if fb == 'forward': x_end, x_start = batch
        else: x_start, x_end = batch
        outputs = {'x_start': x_end, 'x_end': x_start} # For logger

        # if first iteration apply optional mini-batch sampling
        if self.first_iteration and self.hparams.use_mini_batch:
            x_start, x_end = optimize_coupling(x_start, x_end)
        # otherwise generate samples from reverse model
        if not self.first_iteration:
            x_start, x_end = x_start, self.sample(x_start, fb=bf)

        _, info = self.markovian_projection(fb, x_start, x_end)
        info = {f"test/{k}": v for k, v in info.items()}
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return outputs

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer_forward  = self.hparams.optimizer(params=self.models['forward'].parameters())
        optimizer_backward  = self.hparams.optimizer(params=self.models['backward'].parameters())

        if self.hparams.scheduler is not None:
            scheduler_forward = self.hparams.scheduler(optimizer=optimizer_forward)
            scheduler_backward = self.hparams.scheduler(optimizer=optimizer_backward)
            return [
                {'optimizer': optimizer_forward, 'lr_scheduler': scheduler_forward},
                {'optimizer': optimizer_backward, 'lr_scheduler': scheduler_backward}
            ]
        return [{'optimizer': optimizer_forward}, {'optimizer': optimizer_backward}]

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
        if fb is None:
            fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
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
        if fb is None:
            fb = 'forward' if self.current_epoch % 2 == 0 else 'backward'
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

