from typing import Literal
import torch
from torch.nn import functional as F
from torchmetrics import Metric


class TVComplement(Metric):
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
            default=torch.zeros(dim, num_categories, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_counts",
            default=torch.zeros(dim, num_categories, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        if real_data.shape != pred_data.shape or real_data.ndim != 2:
            raise ValueError("Expect two equal‑shaped 2‑D tensors (batch_size, dim).")

        real_batch_counts = F.one_hot(real_data, self.num_categories).sum(dim=0)
        pred_batch_counts = F.one_hot(pred_data, self.num_categories).sum(dim=0)

        self.real_counts += real_batch_counts
        self.pred_counts += pred_batch_counts

    def compute(self) -> torch.Tensor:
        real_totals = self.real_counts.sum(dim=1, keepdim=True) # (D, 1)
        pred_totals = self.pred_counts.sum(dim=1, keepdim=True) # (D, 1)

        reals = self.real_counts.float() / real_totals
        preds = self.pred_counts.float() / pred_totals
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
