from typing import Optional
import torch
from torchmetrics.text import Perplexity

from catsbench.benchmarks.base import BenchmarkBase 


class SBPerplexity(Perplexity):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        benchmark: BenchmarkBase,
        ignore_index: Optional[int] = None
    ):
        super().__init__(
            ignore_index=ignore_index, 
            reduction='mean'
        )
        self.benchmark = benchmark

    def update(
        self, 
        x_start: torch.Tensor, 
        pred_x_end: torch.Tensor
    ):
        log_probs = self.benchmark.log_prob(
            x_start, pred_x_end
        )
        return super().update(
            preds=log_probs, 
            target=pred_x_end
        )