from typing import List

import torch
from torch import nn
from torch.nn import functional as F

class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int, 
        num_timesteps: int,
        timestep_dim: int = 2, 
        layers: List[int] = [128, 128, 128],
     ) -> None: 
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        net = []
        ch_prev = input_dim + timestep_dim
        for ch_next in layers:
            net.extend([nn.Linear(ch_prev, ch_next), nn.ReLU()])
            ch_prev = ch_next
        net.append(nn.Linear(ch_prev, num_categories * input_dim))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, timestep_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ##############################################
        # Additional parametrization from D3PM article
        # x_one_hot = F.one_hot(x, self.num_categories) 
        # mean = (self.num_categories - 1) / 2
        # x = x / mean - 1
        ##############################################

        x_start_logits = self.net(torch.cat([x.float(), self.timestep_embedding(t)], dim=1))
        x_start_logits = x_start_logits.view(-1, self.input_dim, self.num_categories)

        ##############################################
        # Additional parametrization from D3PM article
        # x_start_logits = x_start_logits + x_one_hot
        ##############################################
        return x_start_logits