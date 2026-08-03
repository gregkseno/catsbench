"""A small autoregressive denoiser for vector-valued categorical states."""

from typing import Sequence

import torch
from torch import nn


class AutoregressiveDenoiserMLP(nn.Module):
    """MLP used by :class:`src.methods.ar_csbm.ARCSBM`.

    The model sees the complete noisy state and a partially filled reverse-state
    prefix.  Unknown prefix coordinates use the extra embedding id
    ``mask_token_id == num_categories``.  ARCSBM reveals one coordinate at a
    time, preventing target leakage even though the MLP itself is not masked.
    """

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        num_timesteps: int,
        category_dim: int = 16,
        timestep_dim: int = 8,
        layers: Sequence[int] = (256, 256, 256),
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if num_categories < 2:
            raise ValueError("num_categories must be at least 2")

        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        self.mask_token_id = num_categories

        self.state_embedding = nn.Embedding(num_categories, category_dim)
        self.prefix_embedding = nn.Embedding(num_categories + 1, category_dim)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, timestep_dim)

        in_features = 2 * input_dim * category_dim + timestep_dim
        net = []
        for out_features in layers:
            net.extend((nn.Linear(in_features, out_features), nn.SiLU()))
            in_features = out_features
        net.append(nn.Linear(in_features, input_dim * num_categories))
        self.net = nn.Sequential(*net)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.shape != x_prefix.shape:
            raise ValueError(
                f"x_t and x_prefix must have equal shapes, got "
                f"{tuple(x_t.shape)} and {tuple(x_prefix.shape)}"
            )
        if x_t.ndim < 2 or x_t[0].numel() != self.input_dim:
            raise ValueError(
                f"Expected each state to contain {self.input_dim} coordinates, "
                f"got shape {tuple(x_t.shape)}"
            )

        batch_size = x_t.shape[0]
        event_shape = x_t.shape[1:]
        x_t = x_t.reshape(batch_size, self.input_dim).long()
        x_prefix = x_prefix.reshape(batch_size, self.input_dim).long()
        features = torch.cat(
            (
                self.state_embedding(x_t).flatten(start_dim=1),
                self.prefix_embedding(x_prefix).flatten(start_dim=1),
                self.timestep_embedding(t),
            ),
            dim=-1,
        )
        logits = self.net(features)
        return logits.reshape(
            batch_size, *event_shape, self.num_categories
        )

    def teacher_forced_logits(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all teacher-forced conditionals in one vectorized call."""
        if x_t.shape != x_prev.shape:
            raise ValueError(
                f"x_t and x_prev must have equal shapes, got "
                f"{tuple(x_t.shape)} and {tuple(x_prev.shape)}"
            )

        batch_size = x_t.shape[0]
        event_shape = x_t.shape[1:]
        event_size = x_t[0].numel()
        if event_size != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} coordinates, got {event_size}"
            )

        x_flat = x_t.reshape(batch_size, event_size)
        prev_flat = x_prev.reshape(batch_size, event_size)
        # contexts[b, i] contains exactly z_<i and masks z_>=i.
        contexts = prev_flat[:, None, :].expand(-1, event_size, -1).clone()
        positions = torch.arange(event_size, device=x_t.device)
        hidden = positions[None, :] >= positions[:, None]
        contexts.masked_fill_(hidden[None], self.mask_token_id)

        repeated_x = x_flat[:, None, :].expand(-1, event_size, -1)
        repeated_t = t[:, None].expand(-1, event_size)
        all_logits = self.forward(
            repeated_x.reshape(batch_size * event_size, event_size),
            repeated_t.reshape(batch_size * event_size),
            contexts.reshape(batch_size * event_size, event_size),
        ).reshape(
            batch_size,
            event_size,
            event_size,
            self.num_categories,
        )
        logits = all_logits[:, positions, positions]
        return logits.reshape(batch_size, *event_shape, self.num_categories)
