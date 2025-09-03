from typing import Literal
import torch
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
            default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pred_counts",
            default=torch.zeros(dim, dim, num_categories, num_categories, dtype=torch.int),
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
        
        for d in range(self.dim):
            r_d = real_data[:, d]  # (B,)
            p_d = pred_data[:, d]  # (B,)
            for j in range(self.dim):
                r_idx = r_d * self.num_categories + real_data[:, j]
                p_idx = p_d * self.num_categories + pred_data[:, j]

                r_cnt = torch.bincount(r_idx, minlength=self.num_categories**2).to(torch.int)
                p_cnt = torch.bincount(p_idx, minlength=self.num_categories**2).to(torch.int)

                self.real_counts[d, j] += r_cnt.reshape(self.num_categories, self.num_categories)
                self.pred_counts[d, j] += p_cnt.reshape(self.num_categories, self.num_categories)

    def compute(self) -> torch.Tensor:
        real_total = self.real_counts.sum(dim=[2, 3], keepdim=True) # (D, D, 1, 1)
        pred_total = self.pred_counts.sum(dim=[2, 3], keepdim=True) # (D, D, 1, 1)

        reals = torch.where(
            real_total > 0, self.real_counts.float() / real_total.float(), 
            torch.zeros_like(self.real_counts, dtype=torch.float)
        )
        preds = torch.where(
            pred_total > 0, self.pred_counts.float() / pred_total.float(), 
            torch.zeros_like(self.pred_counts, dtype=torch.float)
        )

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