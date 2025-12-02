from typing import List, Literal
import torch

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class ShapeScore(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    scores: List[torch.Tensor]
    real_counts: torch.Tensor
    pred_counts: torch.Tensor
    
    def __init__(
        self,
        dim: int,
        num_categories: int,
        conditional: bool = False,
        reduction: Literal['mean', 'none'] = 'mean',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories
        self.conditional = conditional
        self.reduction = reduction

        if conditional:
            self.add_state("scores", default=[], dist_reduce_fx='cat')
        else:
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

    def _compute_score(self, real_counts: torch.Tensor, pred_counts: torch.Tensor) -> torch.Tensor:
        real_totals = real_counts.sum(dim=1, keepdim=True).float()
        pred_totals = pred_counts.sum(dim=1, keepdim=True).float()

        reals = real_counts.float() / (real_totals + torch.finfo(real_totals.dtype).eps)
        preds = pred_counts.float() / (pred_totals + torch.finfo(pred_totals.dtype).eps)
        
        tvd = 0.5 * torch.abs(reals - preds).sum(dim=1) 
        return 1.0 - tvd
        
    @torch.no_grad()
    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        if real_data.shape != pred_data.shape or real_data.ndim != 2:
            raise ValueError("Expect two equal-shaped 2-D tensors (batch_size, dim).")
        if real_data.max() >= self.num_categories or pred_data.max() >= self.num_categories:
            raise ValueError(f"Data contains values >= num_categories ({self.num_categories})")

        batch_size = real_data.shape[0]
        if not hasattr(self, "_dim_offsets") or self._dim_offsets.device != real_data.device:
            self._dim_offsets = torch.arange(
                self.dim, device=real_data.device, dtype=torch.long
            ) * self.num_categories

        offsets = self._dim_offsets.repeat(batch_size) # (B*dim,)
        r_code = offsets + real_data.reshape(-1)
        p_code = offsets + pred_data.reshape(-1)

        r_cnt = torch.bincount(
            r_code, minlength=self.dim * self.num_categories
        ).reshape(self.dim, self.num_categories).int()
        p_cnt = torch.bincount(
            p_code, minlength=self.dim * self.num_categories
        ).reshape(self.dim, self.num_categories).int()

        # for conditional, compute and store score immediately 
        if self.conditional:
            score = self._compute_score(r_cnt, p_cnt)
            self.scores.append(score.unsqueeze(0)) # to enable stacking using cat
            
        # for unconditional, accumulate counts
        else:
            self.real_counts += r_cnt
            self.pred_counts += p_cnt

    def compute(self) -> torch.Tensor:
        if self.conditional:
            all_scores = dim_zero_cat(self.scores) # (N, D)
            if self.reduction == "mean":
                return all_scores.mean()
            return all_scores
        
        scores = self._compute_score(self.real_counts, self.pred_counts)
        if self.reduction == "mean":
            return scores.mean()
        return scores
