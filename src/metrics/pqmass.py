from typing import Literal
import torch
from torch.nn import functional as F
from torchmetrics import Metric

from pqm import pqm_chi2


class PQMass(Metric):
    real_data: torch.Tensor
    pred_data: torch.Tensor
    
    def __init__(
        self,
        dim: int,
        num_refs: int = 100,
        re_tessellation: int = 1000,
        permute_tests: int = 10,
        kernel: str = 'euclidean',
        reduction: Literal['mean', 'none'] = 'mean',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_refs = num_refs
        self.re_tessellation = re_tessellation
        self.permute_tests = permute_tests
        self.kernel = kernel
        self.reduction = reduction

        self.add_state("real_data", default=torch.zeros(0))
        self.add_state("pred_data", default=torch.zeros(0))

    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        self.real_data = torch.cat([self.real_data, real_data], dim=0)
        self.pred_data = torch.cat([self.pred_data, pred_data], dim=0)

    def compute(self) -> torch.Tensor:
        chi2 = pqm_chi2(
            self.real_data, 
            self.pred_data, 
            num_refs=self.num_refs,
            re_tessellation=self.re_tessellation,
            permute_tests=self.permute_tests,
            kernel=self.kernel
        )
        if self.reduction == "mean":
            return chi2.mean()
        return chi2
