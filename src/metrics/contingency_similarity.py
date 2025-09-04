from typing import Literal
import torch
from torchmetrics import Metric


class ContingencySimilarity(Metric):
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

        if not hasattr(self, "_tri_i") or self._tri_i.numel() != (self.dim * (self.dim + 1)) // 2:
            ii, jj = torch.triu_indices(self.dim, self.dim, offset=0, device=real_data.device)
            self._tri_i = ii # (K,)
            self._tri_j = jj # (K,)
            K = ii.numel()
            self._tri_block_offsets = torch.arange(K, device=real_data.device, dtype=torch.long) \
                * (self.num_categories * self.num_categories)

        r_pair_tri = (real_data.index_select(1, self._tri_i) * self.num_categories + real_data.index_select(1, self._tri_j))  # (B, K)
        p_pair_tri = (pred_data.index_select(1, self._tri_i) * self.num_categories + pred_data.index_select(1, self._tri_j))  # (B, K)

        r_code = (self._tri_block_offsets.view(1, -1) + r_pair_tri).reshape(-1)  # (B*K,)
        p_code = (self._tri_block_offsets.view(1, -1) + p_pair_tri).reshape(-1)  # (B*K,)

        K = self._tri_i.numel()
        bins = K * self.num_categories * self.num_categories
        r_cnt_tri = torch.bincount(r_code, minlength=bins).reshape(K, self.num_categories, self.num_categories).int()
        p_cnt_tri = torch.bincount(p_code, minlength=bins).reshape(K, self.num_categories, self.num_categories).int()

        diag_mask = (self._tri_i == self._tri_j)
        if diag_mask.any():
            i_d = self._tri_i[diag_mask]
            j_d = self._tri_j[diag_mask]
            self.real_counts.index_put_((i_d, j_d), r_cnt_tri[diag_mask], accumulate=True)
            self.pred_counts.index_put_((i_d, j_d), p_cnt_tri[diag_mask], accumulate=True)

        off_mask = ~diag_mask
        if off_mask.any():
            i_o = self._tri_i[off_mask]
            j_o = self._tri_j[off_mask]
            r_o = r_cnt_tri[off_mask]
            p_o = p_cnt_tri[off_mask]

            self.real_counts.index_put_((i_o, j_o), r_o, accumulate=True)
            self.pred_counts.index_put_((i_o, j_o), p_o, accumulate=True)

            self.real_counts.index_put_((j_o, i_o), r_o.transpose(-2, -1), accumulate=True)
            self.pred_counts.index_put_((j_o, i_o), p_o.transpose(-2, -1), accumulate=True)

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