import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(
        self, 
        input_dim: int,       
        num_categories: int, 
        num_timesteps: int,  
        timestep_dim: int = 2,  
        layers=[128, 128],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.timestep_dim = timestep_dim
        self.layers = layers
        
        net = []
        ch_prev = input_dim + timestep_dim
        for ch_next in layers:
            net.extend([MaskedLinear(ch_prev, ch_next), nn.ReLU()])
            ch_prev = ch_next
        net.append(MaskedLinear(ch_prev, num_categories * input_dim))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, timestep_dim)
        
        self.m = {}
        self.masks_built = False
        self.update_masks()
        
    def update_masks(self):
        """Construct MADE masks once; natural ordering; no randomness."""
        if self.masks_built:
            return
        self.masks_built = True

        L = len(self.layers)
        D = self.input_dim
        C = self.num_categories
        K = self.timestep_dim

        # ----- degree assignment ------------------------------------------
        # input layer (K timestep cols degree 0, then data degrees 1…D)
        deg_in = np.concatenate((np.zeros(K, dtype=int),
                                 np.arange(1, D + 1, dtype=int)))
        self.m = {-1: deg_in}

        # deterministic hidden degrees: cycle through [deg_min … D-1]
        prev_min = 0
        for l, h in enumerate(self.layers):
            span = max(1, D - prev_min)               # ≥1 to avoid div/0
            self.m[l] = (np.arange(h) % span) + prev_min
            prev_min = int(self.m[l].min())           # always 0 after first

        # ----- build binary masks -----------------------------------------
        masks = []
        # input → hidden[0]
        m0 = self.m[-1][:, None] <= self.m[0][None, :]
        m0[:K, :] = 1                                 # **unmask timestep cols**
        masks.append(m0)
        # hidden[i] → hidden[i+1]
        for l in range(1, L):
            masks.append(self.m[l - 1][:, None] <= self.m[l][None, :])
        # last hidden → output
        out_deg = np.repeat(np.arange(1, D + 1, dtype=int), C)
        masks.append(self.m[L - 1][:, None] < out_deg[None, :])

        # ----- install masks into every MaskedLinear ----------------------
        mlayers = [m for m in self.net.modules() if isinstance(m, MaskedLinear)]
        assert len(mlayers) == len(masks), "mask/layer count mismatch"
        for layer, mask in zip(mlayers, masks):
            layer.set_mask(mask.astype(np.uint8))

    
    def forward(self, x, t):
        x_start_logits = self.net(torch.cat([x.float(), self.timestep_embedding(t)], dim=1))
        x_start_logits = x_start_logits.view(-1, self.input_dim, self.num_categories)
        return x_start_logits
