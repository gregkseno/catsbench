from typing import Literal
import torch
from torch.nn import functional as F
from torchmetrics import Metric


class ContingencySimilarity(Metric):
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
            default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_counts",
            default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        if real_data.shape != pred_data.shape or real_data.ndim != 2:
            raise ValueError("Expect real_data/pred_data of shape (batch_size, dim).")
        if real_data.shape[1] != self.dim:
            raise ValueError(f"Expected second dim = {self.dim}, got {real_data.shape[1]}")
        
        real_batch_one_hot = F.one_hot(real_data, self.num_categories) # (B, D, S)
        pred_batch_one_hot = F.one_hot(pred_data, self.num_categories) # (B, D, S)

        real_batch_counts = torch.einsum('bis,bjc->bijsc', real_batch_one_hot, real_batch_one_hot) # (B, D, D, S, S)
        pred_batch_counts = torch.einsum('bis,bjc->bijsc', pred_batch_one_hot, pred_batch_one_hot) # (B, D, D, S, S)

        self.real_counts += real_batch_counts.sum(dim=0)
        self.pred_counts += pred_batch_counts.sum(dim=0)

    def compute(self) -> torch.Tensor:
        real_total = self.real_counts.sum(dim=[2, 3], keepdim=True) # (D, D, 1, 1)
        pred_total = self.pred_counts.sum(dim=[2, 3], keepdim=True) # (D, D, 1, 1)

        reals = self.real_counts.float() / real_total
        preds = self.pred_counts.float() / pred_total

        tvd = 0.5 * torch.abs(reals - preds).sum(dim=[2, 3]) # (D, D)
        scores = 1.0 - tvd

        if self.reduction == "mean":
            return scores.mean()
        return scores
    

if __name__ == "__main__":
    metric = ContingencySimilarity(dim=2, num_categories=3)
    real  = torch.tensor([[0, 0], [1, 1], [2, 2], [2, 2]])
    pred = real.clone()
    metric.update(real, pred)
    score = metric.compute()
    assert torch.allclose(score, torch.ones(1)), f"got {score}"