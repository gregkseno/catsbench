import torch
from torch import nn

from ...vq_diffusion.modeling.transformers.transformer_utils import UnCondition2ImageTransformer


class LatentTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
        hidden_dim: int = 256,
        num_channels: int = 4,
        num_layers: int = 18,
        num_att_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        content_emb_config = {
            "num_embed": num_categories,
            "spatial_size": input_dim,
            "embed_dim": hidden_dim,
            "trainable": True,
            "pos_emb_type": "embedding",
        }
        self.model = UnCondition2ImageTransformer(
            n_layer=num_layers,
            n_embd=hidden_dim,
            n_head=num_att_heads,
            content_seq_len=input_dim * input_dim,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            mlp_hidden_times=num_channels,
            block_activate="GELU2",
            attn_type="self",
            content_spatial_size=[input_dim, input_dim],
            diffusion_step=num_timesteps + 2,
            timestep_type="adalayernorm",
            content_emb_config=content_emb_config,
            mlp_type="conv_mlp",
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)
