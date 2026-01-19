import torch
import torch.nn as nn
from torch.distributions import Categorical
from lightning import LightningModule
from src.utils.ranked_logger import RankedLogger
from typing import Any, Dict, List, Literal, Optional, Tuple, Union



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
    'dim', 'num_categories'
)

class CNOT(LightningModule):
    def __init__(self, prior, dim, num_categories, parameters_pi, parameters_f):
        super().__init__()

        self.save_hyperparameters(*HPARAMS, logger=False) 

        layers_pi = parameters_pi['layers']
        layers_f = parameters_f['layers']
        emb_dim = parameters_f['emb_dim']

        self.pi_model = PIModel(prior, dim, num_categories, layers=layers_pi)
        self.f_model = PotentialMLP(dim, num_categories, layers=layers_f, emb_dim=emb_dim)
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

        if self.hparams.scheduler is not None:
            scheduler_pi = self.hparams.scheduler(optimizer=pi_opt)
            scheduler_f = self.hparams.scheduler(optimizer=f_opt)
            return [
                {'optimizer': pi_opt, 'lr_scheduler': scheduler_pi},
                {'optimizer': f_opt, 'lr_scheduler': scheduler_f}
            ]
        return [{'optimizer': pi_opt}, {'optimizer': f_opt}]

    
    def training_step(self, batch, batch_idx):
        x_start, x_end = batch
        outputs = {'x_start': x_start, 'x_end': x_end}
        pi_opt, f_opt = self.optimizers()

        for _ in range(self.hparams.PI_ITERS):
            pi_opt.zero_grad()

            X0 = self.bench.sample_input(self.hparams.BATCH_SIZE)
            X1 = self.bench.sample_target(self.hparams.BATCH_SIZE)

            X1_given_X0, log_probs, entropies = self.pi_model(X0, training=True)

            with torch.no_grad():
                f_val = self.f_model(X1_given_X0).squeeze()

            last_timestep = torch.full((X0.shape[0],), self.hparams.NUM_TIMESTEPS + 1, device=X0.device, dtype=torch.int32,)
            cost = -self.prior.extract("cumulative", last_timestep, row_id=X0, column_id=X1_given_X0,).sum(dim=1)
            advantage = (cost - f_val).detach()
            pi_loss = ((advantage * log_probs).mean() - self.hparams.EPSILON * entropies.mean())

            self.manual_backward(pi_loss)

            torch.nn.utils.clip_grad_norm_(self.pi_model.parameters(), max(0.1, self.hparams.EPSILON),)
            pi_opt.step()

        # ------------------
        # Critic update
        # ------------------
        f_opt.zero_grad()

        X0 = self.bench.sample_input(self.hparams.BATCH_SIZE)
        X1 = self.bench.sample_target(self.hparams.BATCH_SIZE)

        with torch.no_grad():
            X1_given_X0 = self.pi_model(X0, training=False)

        if self.hparams.LAMBDA_REG > 0 and self.current_epoch < self.hparams.REG_UNTIL:
            grad_penalty = compute_gradient_penalty(self.f_model, X1, X1_given_X0)
            standard_loss = (self.f_model(X1_given_X0).mean() - self.f_model(X1).mean())
            f_loss = standard_loss + self.hparams.LAMBDA_REG * grad_penalty
        else:
            f_loss = (self.f_model(X1_given_X0).mean() - self.f_model(X1).mean())

        self.manual_backward(f_loss)
        torch.nn.utils.clip_grad_norm_(self.f_model.parameters(), 1.0)
        f_opt.step()
        
        self.log("train/f_loss", f_loss, prog_bar=True)

        info = {
            f'train/pi_loss': pi_loss, 
            f'train/entropy': entropies.mean().mean(), 
            f'train/f_loss': f_loss.mean(),
            f'train/grad_penalty': grad_penalty.mean()
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