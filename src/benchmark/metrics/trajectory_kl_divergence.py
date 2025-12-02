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

    def update(self, real: torch.Tensor, pred: torch.Tensor) -> None:
        assert len(real.shape) > 3, \
            "Inputs must be trajectories with shape [num_trajectories, batch_size, ..., num_categories]!"
        if self.logits:
            real = real.log_softmax(dim=-1)
            pred = pred.log_softmax(dim=-1)
        mode = 'logs' if self.logits else 'probs'
        real = stable_clamp(real, mode=mode)
        pred = stable_clamp(pred, mode=mode)
        super().update(
            p=real.flatten(end_dim=-2), 
            q=pred.flatten(end_dim=-2)
        )
