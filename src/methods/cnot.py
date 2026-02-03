import torch
import torch.nn as nn
from torch.distributions import Categorical
from lightning import LightningModule
from src.utils.ranked_logger import RankedLogger
from typing import Any, Dict, Optional, Tuple
from src.data.prior import Prior
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler



def compute_gradient_penalty(f_model, real_samples, fake_samples, device):
    """Compute gradient penalty in embedding space"""
    batch_size = real_samples.size(0)
    
    # Get embeddings for real and fake samples
    real_emb = f_model.emb(real_samples)  # (B, D, emb_dim)
    fake_emb = f_model.emb(fake_samples)  # (B, D, emb_dim)
    
    # Random interpolation coefficients
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate in embedding space
    interpolated_emb = alpha * real_emb + (1 - alpha) * fake_emb
    interpolated_emb.requires_grad_(True)
    
    # Forward pass through critic (using embedding)
    d_interpolated = f_model.forward_from_embedding(interpolated_emb)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_emb,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient penalty (WGAN-GP style)
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

class PotentialMLP(nn.Module):
    def __init__(self, dim, num_categories, layers=[512]*4, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(num_categories, emb_dim)
        self.emb_dim = emb_dim
        self.dim = dim
        input_dim = dim * emb_dim
    
        layers_list = []
        in_dim = input_dim
        for h in layers:
            layers_list += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
            in_dim = h
        layers_list.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers_list)
    
    def forward(self, x):
        """Forward pass with integer input"""
        x = self.emb(x)              # (B, D, emb_dim)
        x = x.view(x.size(0), -1)    # (B, D*emb_dim)
        return self.mlp(x).squeeze(-1)
    
    def forward_from_embedding(self, x_emb):
        """Forward pass from embedding tensor (for gradient penalty)"""
        x = x_emb.view(x_emb.size(0), -1)  # (B, D*emb_dim)
        return self.mlp(x).squeeze(-1)

class PIModel(nn.Module):
    def __init__(self, prior, dim, num_categories, layers=[128, 128, 128], dropout=0.1):

        super().__init__()
        self.prior = prior
        self.dim = dim
        self.num_categories = num_categories
        self.dropout = dropout
        
        ch_prev = dim
        self.logits_net = []
        for ch_next in layers:
            self.logits_net.append(nn.Linear(ch_prev, ch_next))
            self.logits_net.append(nn.ReLU())
            if dropout > 0:
                self.logits_net.append(nn.Dropout(dropout))
            ch_prev = ch_next
        
        self.logits_net.append(nn.Linear(ch_prev, num_categories * dim))
        self.logits_net = nn.Sequential(*self.logits_net)

    
    def forward(self, x0, training=False):
        
        logits = self.logits_net(x0.to(torch.float32)).view(-1, self.dim, self.num_categories)
        
        samples = []
        log_probs = []
        entropies = []
        dist = Categorical(logits=logits)  # batch_shape = (B, dim)

        samples = dist.sample()  # (B, dim)
        
        if training:
            log_probs = dist.log_prob(samples).sum(dim=1)
            entropies = dist.entropy().sum(dim=1)
            
            return samples, log_probs, entropies
        return samples

    
log = RankedLogger(__name__, rank_zero_only=True)

HPARAMS = (
    'dim', 'num_categories', 'epsilon', 'pi_layers', 'f_layers', 'f_emb_dim', 'num_timesteps',
    'lambda_reg', 'pi_iters', 'f_reg_until', 'pi_optimizer', 'f_optimizer', 'pi_scheduler', 'f_scheduler'
)

class CNOT(LightningModule):
    def __init__(self, 
                 prior: Prior, 
                 dim: int, 
                 num_categories: int, 
                 epsilon: float, 
                 pi_layers: list, 
                 f_layers: list, 
                 f_emb_dim: int, 
                 num_timesteps: int, 
                 lambda_reg: float, 
                 pi_iters: int, 
                 f_reg_until: int,
                 pi_optimizer: Optimizer, # partially initialized 
                 f_optimizer: Optimizer, # partially initialized 
                 pi_scheduler: Optional[LRScheduler] = None, # partially initialized 
                 f_scheduler: Optional[LRScheduler] = None, # partially initialized 
                 ):
        
        super().__init__()

        self.save_hyperparameters(*HPARAMS, logger=False) 
        self.iteration = 0

        self.prior = prior

        self.pi_model = PIModel(prior, dim, num_categories, layers=pi_layers)
        self.f_model = PotentialMLP(dim, num_categories, layers=f_layers, emb_dim=f_emb_dim)
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
    
    def configure_optimizers(self):
        pi_opt = self.hparams.pi_optimizer(self.pi_model.parameters())
        f_opt  = self.hparams.f_optimizer(self.f_model.parameters())

        #if self.hparams.scheduler is not None:
        #    pi_scheduler = self.hparams.pi_scheduler(optimizer=pi_opt)
        #    f_scheduler = self.hparams.f_scheduler(optimizer=f_opt)
        #    return [
        #        {'optimizer': pi_opt, 'lr_scheduler': pi_scheduler},
        #        {'optimizer': f_opt, 'lr_scheduler': f_scheduler}
        #    ]
        return [{'optimizer': pi_opt}, {'optimizer': f_opt}]

    
    def training_step(self, batch, batch_idx):
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end}
        optimizers = {
            'pi_model': self.optimizers()[0],
            'f_model': self.optimizers()[1],
        }
        if self.lr_schedulers():
            schedulers = {
                'pi_model': self.lr_schedulers()[0],
                'f_model': self.lr_schedulers()[1],
            }

        pi_opt = optimizers['pi_model']
        f_opt = optimizers['f_model']

        info = {}

        cycle = self.hparams.pi_iters + 1
        if batch_idx % cycle == cycle - 1:
            f_opt.zero_grad()

            with torch.no_grad():
                X1_given_X0 = self.pi_model(x_start, training=False)

            if self.hparams.lambda_reg > 0 and self.iteration < self.hparams.f_reg_until:
                grad_penalty = compute_gradient_penalty(self.f_model, x_end, X1_given_X0, device=x_start.device)
                standard_loss = (self.f_model(X1_given_X0).mean() - self.f_model(x_end).mean())
                f_loss = standard_loss + self.hparams.lambda_reg * grad_penalty
                info['train/grad_penalty'] = grad_penalty.mean()
            else:
                f_loss = (self.f_model(X1_given_X0).mean() - self.f_model(x_end).mean())


            self.manual_backward(f_loss)
            info[f'train/f_loss'] = f_loss.mean()

            self.iteration += 1

            torch.nn.utils.clip_grad_norm_(self.f_model.parameters(), 1.0)
            f_opt.step()
        
        else:
            pi_opt.zero_grad()

            X1_given_X0, log_probs, entropies = self.pi_model(x_start, training=True)

            with torch.no_grad():
                f_val = self.f_model(X1_given_X0).squeeze()

            last_timestep = torch.full((x_start.shape[0],), self.prior.num_timesteps + 1, device=x_start.device, dtype=torch.int32,)
            cost = -self.prior.extract("cumulative", last_timestep, row_id=x_start, column_id=X1_given_X0,).sum(dim=1)
            advantage = (cost - f_val).detach()
            pi_loss = ((advantage * log_probs).mean() - self.hparams.epsilon * entropies.mean())

            self.manual_backward(pi_loss)

            info['train/cost'] = cost.mean()
            info['train/pi_loss'] = pi_loss.mean()
            info['train/entropy'] = entropies.mean()

            torch.nn.utils.clip_grad_norm_(self.pi_model.parameters(), max(0.1, self.hparams.epsilon),)
            pi_opt.step()
        
        self.log_dict(info, prog_bar=True, sync_dist=True) 
        self.log('train/iteration', self.iteration, prog_bar=True)
        return outputs

    #def on_train_epoch_end(self) -> None:
    #    self.iteration += 1

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        return outputs

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end} # For logger

        return outputs

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['iteration'] = self.iteration

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'iteration' in checkpoint:
            self.iteration = checkpoint['iteration']

    @torch.no_grad()
    def sample(self, x: torch.Tensor = None) -> torch.Tensor:
        return self.pi_model(x)
    
    @torch.no_grad()
    def sample_trajectory(self, 
        x_start: torch.Tensor, 
    ) -> torch.Tensor:
        x_end = self.pi_model(x_start)

        trajectory = [x_start]
        for t in range(1, self.prior.num_timesteps + 1):
            t = torch.full((x_start.shape[0],), t, device=x_start.device)
            x_t = self.prior.sample_bridge(x_start, x_end, t)
            trajectory.append(x_t)
        trajectory.append(x_end)
        return torch.stack(trajectory, dim=0)