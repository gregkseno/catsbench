from typing import Any, Dict, Optional
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from src.utils import convert_to_numpy


class ClassifierTwoSampleTest(Metric):
    def __init__(
        self,
        train_fraction: float = 0.7,
        logistic_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        self.train_fraction = train_fraction
        self.logistic_kwargs = logistic_kwargs or {}

        # Local storage of samples; not aggregated across processes.
        self.add_state("real_data", default=[], dist_reduce_fx='cat')
        self.add_state("pred_data", default=[], dist_reduce_fx='cat')

    def update(self, real_data: torch.Tensor, pred_data: torch.Tensor) -> None:
        if real_data.ndim != 2 or pred_data.ndim != 2:
            raise ValueError("real_data and pred_data must be 2-D tensors shaped (N, D)")
        self.real_data.append(real_data.detach().float())
        self.pred_data.append(pred_data.detach().float())

    def compute(self):
        real_data = convert_to_numpy(dim_zero_cat(self.real_data))
        pred_data = convert_to_numpy(dim_zero_cat(self.pred_data))
        pred_target = np.ones(pred_data.shape[0], dtype=np.int64)
        real_target = np.zeors(real_data.shape[0], dtype=np.int64)

        X = np.vstack([real_data, pred_data])
        y = np.concatenate([pred_target, real_target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_fraction, stratify=y, shuffle=True,
        )

        model = LogisticRegression(**self.logistic_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        correct = (y_pred == y_test).sum()
        n_test = y_test.size
        acc = correct / n_test

        return torch.tensor(acc, dtype=torch.float32)



if __name__ == "__main__":
    metric = ClassifierTwoSampleTest()
    metric.update(torch.randn(512, 5), torch.randn(512, 5) + 0.5)
    result = metric.compute()
    print(result)
