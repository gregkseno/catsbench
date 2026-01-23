from typing import Optional
import torch
from torchmetrics import Metric


class Entropy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, dim: int, num_categories: int):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.add_state(
            "counts",
            default=torch.zeros(dim, num_categories, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "totals",
            default=torch.zeros(dim, dtype=torch.int),
            dist_reduce_fx="sum",
        )

        self._dim_offsets: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("Expect a 2-D tensor (batch_size, dim).")
        if x.max() >= self.num_categories or x.min() < 0:
            raise ValueError(f"x contains values outside [0, num_categories-1] with num_categories={self.num_categories}.")

        batch_size = x.shape[0]

        if (self._dim_offsets is None) or (self._dim_offsets.device != x.device):
            self._dim_offsets = torch.arange(
                self.dim, device=x.device, dtype=torch.long
            ) * self.num_categories

        offsets = self._dim_offsets.repeat(batch_size)
        code = offsets + x.reshape(-1).to(torch.long)
        flat = torch.bincount(code, minlength=self.dim * self.num_categories)
        cnt = flat.reshape(self.dim, self.num_categories).int()

        self.counts += cnt
        self.totals += batch_size

    def compute(self) -> torch.Tensor:
        totals = self.totals.to(torch.float32)
        entropy = torch.zeros(self.dim, device=self.counts.device, dtype=torch.float32)

        valid = totals > 0
        if valid.any():
            probs = self.counts[valid].to(torch.float32) / totals[valid].unsqueeze(1)
            non_zero = probs > 0
            plogp = torch.zeros_like(probs)
            plogp[non_zero] = probs[non_zero] * torch.log(probs[non_zero])
            entropy[valid] = -plogp.sum(dim=1)

        return entropy
