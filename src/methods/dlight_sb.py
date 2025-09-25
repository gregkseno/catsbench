from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule

from src.utils.logging.console import RankedLogger
from src.data.prior import Prior


log = RankedLogger(__name__, rank_zero_only=True)

HPARAMS = (
    'dim', 'num_potentials', 'distr_init', 
    'sample_prob', 'optimizer', 'scheduler'
)

class DLightSB(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian',
        sample_prob: float = 0.9
    ):
        super().__init__()
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.ParameterList([
            nn.Parameter(torch.empty(
                num_potentials, prior.num_categories,
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            for _ in range(dim)
        ])

        self.bidirectional = False  
        self.iteration = 1
        self._did_weight_init = False
        self._loaded_from_ckpt = False

    @torch.no_grad()
    def init_weights(self, init_samples: Optional[torch.Tensor] = None):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        if self.hparams.distr_init == 'gaussian':
            cur = (-1.0 + (0.5**2) * torch.randn(
                self.hparams.dim, self.hparams.num_potentials, self.prior.num_categories, 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype)
            ) / (self.prior.num_categories * self.hparams.num_potentials)
            cur = torch.log((cur ** 2).clamp_min(1e-12))  # (D, K, S)
            for d in range(self.hparams.dim):
                self.log_cp_cores[d].data.copy_(cur[d])

        elif self.hparams.distr_init == 'uniform':
            val = torch.log(torch.tensor(
                1.0 / (self.prior.num_categories * self.hparams.num_potentials), 
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            for d in range(self.hparams.dim):
                self.log_cp_cores[d].data.fill_(val)

        elif self.hparams.distr_init == 'samples':
            assert init_samples is not None, "init_samples should not be None when using benchmark samples"
            init_samples = torch.as_tensor(init_samples, device=self.log_alpha.device)
            assert init_samples.dim() == 2 and init_samples.shape == (self.hparams.num_potentials, self.hparams.dim), \
                f"init_samples must be (num_potentials, dim), got {tuple(init_samples.shape)}"

            base_val = torch.log(torch.tensor(
                (1 - self.hparams.sample_prob) / (self.prior.num_categories - 1),
                device=self.log_alpha.device, dtype=self.log_alpha.dtype
            ))
            hot_val = torch.tensor(math.log(
                self.hparams.sample_prob), device=self.log_alpha.device, dtype=self.log_alpha.dtype
            )

            for d in range(self.hparams.dim):
                core = torch.full((
                    self.hparams.num_potentials, self.prior.num_categories), base_val, 
                    device=self.log_alpha.device, dtype=self.log_alpha.dtype
                )
                col_idx = init_samples[:, d].long().view(self.hparams.num_potentials, 1)                  # (K, 1)
                core.scatter_(dim=1, index=col_idx, src=hot_val.expand(self.hparams.num_potentials, 1))   # (K, S)
                self.log_cp_cores[d].data.copy_(core)

        else:
            raise ValueError(f"Invalid distr_init: {self.hparams.distr_init}")

        self._did_weight_init = True

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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._loaded_from_ckpt = True

    def setup(self, stage: Literal['fit', 'validate', 'test']):
        if stage in (None, "fit") and not self._did_weight_init and not self._loaded_from_ckpt:
            assert hasattr(self.trainer, 'datamodule'), "Trainer has no datamodule attribute"
            assert hasattr(self.trainer.datamodule, 'benchmark'), "Datamodule has no benchmark attribute"
            init_samples: torch.Tensor = self.trainer.datamodule.benchmark.sample_input(self.hparams.num_potentials)
            init_samples = init_samples.flatten(start_dim=1) # (num_potentials, dim)
            self.init_weights(init_samples)

    def get_log_v(self, x_end: torch.Tensor) -> torch.Tensor:
        x_end = x_end.flatten(start_dim=1)[:, :, None, None].expand(-1, -1, self.hparams.num_potentials, -1) # (B, D, K, 1)
        log_cp_cores = torch.stack(list(self.log_cp_cores), dim=0) # (D, K, S)
        log_cp_cores = log_cp_cores[None, :, :, :].expand(x_end.shape[0], -1, -1, -1) # (B, D, K, S)
        log_r = torch.gather(log_cp_cores, dim=-1, index=x_end) # (B, D, K, 1)
        log_r = log_r.squeeze(-1).sum(dim=1) # (B, K)
        return torch.logsumexp(self.log_alpha[None, :] + log_r, dim=1) # (B)

    def get_log_c(self, x_start: torch.Tensor) -> torch.Tensor:
        x_start = x_start.flatten(start_dim=1)
        log_z = torch.zeros(x_start.shape[0], self.hparams.num_potentials, device=self.device)
        for d in range(self.hparams.dim):
            log_pi_ref = self.prior.extract_last_cum_matrix(x_start[:, d]) # (B, S)
            log_z = log_z + torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :], dim=2
            ) # (1, K, S) + (B, 1, S) -> (B, K)
        return torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(1, K) + (B, K) -> (B)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        outputs['loss'] = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'train/loss': outputs['loss'], 
            f'train/log_v': log_v.mean(), 
            f'train/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return outputs

    def on_train_epoch_end(self) -> None:
        self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'val/loss': loss, 
            f'val/log_v': log_v.mean(), 
            f'val/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'test/loss': loss, 
            f'test/log_v': log_v.mean(), 
            f'test/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return outputs

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer = self.hparams.optimizer(
            params=[self.log_alpha] + list(self.log_cp_cores)
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.flatten(start_dim=1) # (B, D)

        log_z = torch.zeros(x.shape[0], self.hparams.num_potentials, device=self.device)
        for d in range(self.hparams.dim):
            log_pi_ref = self.prior.extract_last_cum_matrix(x[:, d]) # (B, S)
            log_z = log_z + torch.logsumexp(
                self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :], dim=2
            ) # (1, K, S) + (B, 1, S) -> (B, K)

        log_w_k = self.log_alpha[None, :] + log_z # (B, K)
        noise = torch.rand_like(log_w_k)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        k_star = torch.argmax(log_w_k + gumbel_noise, dim=-1) # (B)

        y_samples = torch.empty(x.shape[0], self.hparams.dim, dtype=torch.long, device=self.device)
        for d in range(self.hparams.dim):
            log_pi_ref = self.prior.extract_last_cum_matrix(x[:, d]) # (B, S)
            log_cp_cores_d = self.log_cp_cores[d][None, :, :].expand(x.shape[0], -1, -1) # (B, K, S)
            log_cp_cores_d_selected = torch.gather(
                log_cp_cores_d, dim=1, index=k_star[:, None, None].expand(-1, -1, self.prior.num_categories)
            ).squeeze(1) # (B, 1, S) -> (B, S)
            log_p_d_selected = log_cp_cores_d_selected + log_pi_ref[:, :] # (B, S)
            noise = torch.rand_like(log_p_d_selected)
            noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
            gumbel_noise = -torch.log(-torch.log(noise))
            y_d = torch.argmax(log_p_d_selected + gumbel_noise, dim=-1)
            y_samples[:, d] = y_d
        
        return y_samples.reshape(input_shape)
    
    @torch.no_grad()
    def sample_trajectory(self, x_start: torch.Tensor) -> torch.Tensor:
        x_end = self.sample(x_start)

        trajectory = [x_start]
        for t in range(1, self.prior.num_timesteps + 1):
            t = torch.full((x_start.shape[0],), t, device=x_start.device)
            x_t = self.prior.sample_bridge(x_start, x_end, t)
            trajectory.append(x_t)
        trajectory.append(x_end)
        return torch.stack(trajectory, dim=0)
    
class DLightSB_old(LightningModule):
    def __init__(
        self, 
        prior: Prior,
        dim: int,
        num_potentials: int, 
        optimizer: Optimizer, # partially initialized 
        scheduler: Optional[LRScheduler] = None, # partially initialized 
        distr_init: Literal['uniform', 'gaussian', 'samples'] = 'gaussian',
        sample_prob: float = 0.9
    ):
        super().__init__()
        self.save_hyperparameters(*HPARAMS, logger=False)        
        self.prior = prior
        
        self.log_alpha = nn.Parameter(torch.zeros(num_potentials))
        self.log_cp_cores = nn.Parameter(torch.empty(
            dim, num_potentials, prior.num_categories,
            device=self.log_alpha.device, dtype=self.log_alpha.dtype
        ))

        self.bidirectional = False  
        self.iteration = 1
        self._did_weight_init = False
        self._loaded_from_ckpt = False

    @torch.no_grad()
    def init_weights(self, init_samples: Optional[torch.Tensor] = None):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        if self.hparams.distr_init == 'gaussian':
            cur = (-1.0 + (0.5**2) * torch.randn(
                self.hparams.dim, self.hparams.num_potentials, self.prior.num_categories,
                device=self.log_cp_cores.device, dtype=self.log_cp_cores.dtype
            )) / (self.prior.num_categories * self.hparams.num_potentials)
            self.log_cp_cores.copy_(torch.log((cur ** 2).clamp_min(1e-12)))

        elif self.hparams.distr_init == 'uniform':
            val = torch.log(torch.tensor(
                1.0 / (self.prior.num_categories * self.hparams.num_potentials),
                device=self.log_cp_cores.device, dtype=self.log_cp_cores.dtype)
            )
            self.log_cp_cores.fill_(val)

        elif self.hparams.distr_init == 'samples':
            assert init_samples is not None, "init_samples should not be None when using benchmark samples"
            init_samples = torch.as_tensor(init_samples, device=self.log_cp_cores.device)
            base_val = torch.log(torch.tensor(
                (1 - self.hparams.sample_prob) / (self.prior.num_categories - 1),
                device=self.log_cp_cores.device, dtype=self.log_cp_cores.dtype)
            )
            self.log_cp_cores.fill_(base_val)

            idx = init_samples.t().unsqueeze(-1).long()  # (dim, num_potentials, 1)
            src = torch.full(
                idx.shape, math.log(self.hparams.sample_prob),
                device=self.log_cp_cores.device, dtype=self.log_cp_cores.dtype
            )
            self.log_cp_cores.scatter_(2, idx, src)

        else:
            raise ValueError(f"Invalid distr_init: {self.hparams.distr_init}")
        
        self._did_weight_init = True

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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._loaded_from_ckpt = True

    def setup(self, stage: Literal['fit', 'validate', 'test']):
        if stage in (None, "fit") and not self._did_weight_init and not self._loaded_from_ckpt:
            assert hasattr(self.trainer, 'datamodule'), "Trainer has no datamodule attribute"
            assert hasattr(self.trainer.datamodule, 'benchmark'), "Datamodule has no benchmark attribute"
            init_samples: torch.Tensor = self.trainer.datamodule.benchmark.sample_input(self.hparams.num_potentials)
            init_samples = init_samples.flatten(start_dim=1) # (num_potentials, dim)
            self.init_weights(init_samples)

    def get_log_v(self, x_end: torch.Tensor) -> torch.Tensor:
        x_end = x_end.flatten(start_dim=1)
        log_terms = self.log_alpha[None, :]  # (1, K)
        
        for d in range(x_end.shape[1]):
            y_d = x_end[:, d]  # (batch_size,)
            log_r_d = self.log_cp_cores[d][:, y_d].T  # (batch_size, K)
            log_terms = log_terms + log_r_d
            
        log_v = torch.logsumexp(log_terms, dim=1)  # (batch_size,)
        return log_v

    def get_log_c(self, x_start: torch.Tensor) -> torch.Tensor:
        x_start = x_start.flatten(start_dim=1)
        log_z = torch.zeros(x_start.shape[0], self.hparams.num_potentials, device=self.device)
        
        for d in range(self.hparams.dim):
            x_d = x_start[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d)
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        outputs['loss'] = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'train/loss': outputs['loss'], 
            f'train/log_v': log_v.mean(), 
            f'train/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return outputs

    def on_train_epoch_end(self) -> None:
        self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'val/loss': loss, 
            f'val/log_v': log_v.mean(), 
            f'val/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('val/iteration', self.iteration, prog_bar=True)
        return outputs

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        log_v = self.get_log_v(x_end)
        log_c = self.get_log_c(x_start)
        loss = (-log_v + log_c).mean()

        # logs step-wise loss
        info = {
            f'test/loss': loss, 
            f'test/log_v': log_v.mean(), 
            f'test/log_c': log_c.mean()
        }
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('test/iteration', self.iteration, prog_bar=True)
        return outputs

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        optimizer = self.hparams.optimizer(
            params=[self.log_alpha, self.log_cp_cores]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.flatten(start_dim=1)

        log_z = torch.zeros(x.shape[0], self.hparams.num_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.hparams.dim):
            x_d = x[:, d]
            log_pi_ref = self.prior.extract_last_cum_matrix(x_d)
            
            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(x.shape[0], self.hparams.dim, dtype=torch.long, device=self.device)
    
        for d in range(self.hparams.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(x.shape[0], device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        return y_samples.reshape(input_shape)
    
    @torch.no_grad()
    def sample_trajectory(self, x_start: torch.Tensor) -> torch.Tensor:
        x_end = self.sample(x_start)

        trajectory = [x_start]
        for t in range(1, self.prior.num_timesteps + 1):
            t = torch.full((x_start.shape[0],), t, device=x_start.device)
            x_t = self.prior.sample_bridge(x_start, x_end, t)
            trajectory.append(x_t)
        trajectory.append(x_end)
        return torch.stack(trajectory, dim=0)