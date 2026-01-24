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
        super().__init__(ignore_index=ignore_index)
        self.benchmark = benchmark

    def update(
        self, 
        x_start: torch.Tensor, 
        pred_x_end: torch.Tensor
    ):
        logits = self.benchmark.get_cum_transition_logits(x_start)
        return super().update(
            preds=logits, 
            target=pred_x_end
        )