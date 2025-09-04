from typing import Literal
import torch
from torchmetrics import Metric


class TVComplement(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    real_counts: torch.Tensor
    pred_counts: torch.Tensor
    
    def __init__(
        self,
        dim: int,
        num_categories: int,
        reduction: Literal['mean', 'none'] = 'mean',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.reduction = reduction

        self.add_state(
            "real_counts",
            default=torch.zeros(dim, num_categories, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_counts",
            default=torch.zeros(dim, num_categories, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        
    @torch.no_grad()
    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        if real_data.shape != pred_data.shape or real_data.ndim != 2:
            raise ValueError("Expect two equal-shaped 2-D tensors (batch_size, dim).")

        batch_size = real_data.shape[0]
        # лениво кэшируем оффсеты d * num_categories
        if not hasattr(self, "_dim_offsets") or self._dim_offsets.device != real_data.device \
           or self._dim_offsets.numel() != self.dim or getattr(self, "_cached_S", None) != self.num_categories:
            self._dim_offsets = torch.arange(self.dim, device=real_data.device, dtype=torch.long) * self.num_categories

        offsets = self._dim_offsets.repeat(batch_size) # (B*dim,)
        r_code = offsets + real_data.reshape(-1)
        p_code = offsets + pred_data.reshape(-1)

        r_cnt = torch.bincount(r_code, minlength=self.dim * self.num_categories).int()
        p_cnt = torch.bincount(p_code, minlength=self.dim * self.num_categories).int()

        self.real_counts += r_cnt.reshape(self.dim, self.num_categories)
        self.pred_counts += p_cnt.reshape(self.dim, self.num_categories)

    def compute(self) -> torch.Tensor:
        real_totals = self.real_counts.sum(dim=1, keepdim=True) # (D, 1)
        pred_totals = self.pred_counts.sum(dim=1, keepdim=True) # (D, 1)

        reals = torch.where(
            real_totals > 0,
            self.real_counts.float() / real_totals.float(),
            torch.zeros_like(self.real_counts, dtype=torch.float),
        )
        preds = torch.where(
            pred_totals > 0,
            self.pred_counts / pred_totals.float(),
            torch.zeros_like(self.pred_counts, dtype=torch.float),
        )
        tvd = 0.5 * torch.abs(reals - preds).sum(dim=1) # (D)
        scores = 1.0 - tvd
        
        if self.reduction == "mean":
            return scores.mean()
        return scores


if __name__ == "__main__":
    metric = TVComplement(dim=2, num_categories=3)
    real  = torch.tensor([[0, 0], [1, 1], [2, 2]])
    pred = real.clone()
    metric.update(real, pred)
    score = metric.compute()
    assert torch.allclose(score, torch.ones(1)), f"got {score}"

    metric = TVComplement(dim=2, num_categories=3, reduction='none')
    real  = torch.tensor([[0, 0], [1, 1], [2, 2]])
    pred = torch.tensor([[0, 0], [0, 0], [0, 0]])
    metric.update(real, pred)
    score = metric.compute()
    expected = torch.tensor([1.0 - 2.0 / 3.0, 1.0 - 2.0 / 3.0])
    assert torch.allclose(score, expected, atol=1e-6), f"got {score}"
