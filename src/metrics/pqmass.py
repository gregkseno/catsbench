from typing import Literal
import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from pqm import pqm_chi2

from src.utils import convert_to_numpy


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

        self.add_state("real_data", default=[], dist_reduce_fx='cat')
        self.add_state("pred_data", default=[], dist_reduce_fx='cat')

    def update(
        self,
        real_data: torch.Tensor,
        pred_data: torch.Tensor,
    ) -> None:
        self.real_data.append(real_data) 
        self.pred_data.append(pred_data)

    def compute(self) -> torch.Tensor:
        chi2 = pqm_chi2(
            convert_to_numpy(dim_zero_cat(self.real_data)), # convertation only because pqm_chi2 kernel
            convert_to_numpy(dim_zero_cat(self.pred_data)), # parameter works with numpy arrays 
            num_refs=self.num_refs,
            re_tessellation=self.re_tessellation,
            permute_tests=self.permute_tests,
            kernel=self.kernel
        )
        if self.reduction == "mean":
            if isinstance(chi2, tuple):
                chi2 = torch.tensor(chi2[0])
            elif isinstance(chi2, list):
                if len(chi2) > 0:
                    if isinstance(chi2[0], tuple):
                        chi2 = torch.tensor([c[0] for c in chi2])
                    else:
                        chi2 = torch.tensor(chi2)
                else:
                    chi2 = torch.tensor(0.0)
            elif isinstance(chi2, np.ndarray):
                chi2 = torch.tensor(chi2)
            elif not isinstance(chi2, torch.Tensor):
                chi2 = torch.tensor(chi2)
            chi2 = torch.mean(chi2)
        
        return chi2
