from typing import Literal, Optional
import torch
from torchmetrics.regression import KLDivergence

from benchmark.utils import stable_clamp


class TrajectoryKLDivergence(KLDivergence):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self, 
        logits: bool = True,
        reduction: Optional[Literal['mean', 'sum', 'none']] = 'mean',
    ):
        super().__init__(log_prob=logits, reduction=reduction)
        self.logits = logits

    def update(self, p: torch.Tensor, q: torch.Tensor) -> None:
        if len(p.shape) < 3 or p.shape != q.shape:
            raise ValueError(
                "Inputs must be trajectories with shape [batch_size, ..., num_categories]!"
            )
        if self.logits:
            p = p.log_softmax(dim=-1)
            q = q.log_softmax(dim=-1)

        mode = 'logs' if self.logits else 'probs'
        p = stable_clamp(p, mode=mode)
        q = stable_clamp(q, mode=mode)
        
        super().update(
            p=p.flatten(end_dim=-2), 
            q=q.flatten(end_dim=-2)
        )
