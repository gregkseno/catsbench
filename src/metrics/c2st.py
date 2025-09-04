import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional import auroc


class ClassifierTwoSampleTest(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self, 
        dim: int, 
        lr: float = 1e-2, 
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize an inivisible linear layer parameters of which won't be accesible from outside
        w = torch.nn.Parameter(torch.empty(1, self.dim))
        b = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.kaiming_uniform_(w, a=5**0.5)
        torch.nn.init.zeros_(b)

        self.register_buffer("weight", w.detach().clone().requires_grad_(True))
        self.register_buffer("bias", b.detach().clone().requires_grad_(True))
        self.optimizer = torch.optim.SGD([self.weight, self.bias], lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def reset(self) -> None:
        super().reset()
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.bias)
        self.optimizer = torch.optim.SGD(
            [self.weight, self.bias], lr=self.lr, weight_decay=self.weight_decay
        )

    def update(self, real_data: torch.Tensor, pred_data: torch.Tensor, train: bool) -> None:
        assert real_data.shape == pred_data.shape, "real_data and pred_data must have the same shape!"
        x_real = real_data.detach().float()
        x_pred = pred_data.detach().float()

        x = torch.cat([x_real, x_pred], dim=0)
        y = torch.cat(
            [
                torch.zeros(x_real.size(0), device=x.device, dtype=torch.float),
                torch.ones(x_pred.size(0), device=x.device, dtype=torch.float)
            ],
            dim=0
        )

        if train:
            logits = F.linear(x, self.weight, self.bias).squeeze(-1)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.SUM)
                self.weight /= world_size
                self.bias /= world_size
        else:
            with torch.no_grad():
                logits = F.linear(x, self.weight, self.bias).squeeze(-1)
                probs = torch.sigmoid(logits).detach()
            targets = y.detach().long()

            self.probs.append(probs)
            self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        if len(self.probs) == 0:
            return torch.tensor(0.0, dtype=torch.float)
        
        probs = dim_zero_cat(self.probs)
        targets = dim_zero_cat(self.targets)
        auroc_value = auroc(probs, targets, task="binary")
        score = 1.0 - torch.abs(auroc_value - 0.5) * 2.0
        return score


if __name__ == "__main__":
    torch.manual_seed(0)
    metric = ClassifierTwoSampleTest(dim=5, lr=1e-1)
    metric.update(torch.randn(512, 5), torch.randn(512, 5) + 0.5, train=True)
    metric.update(torch.randn(512, 5), torch.randn(512, 5) + 0.5, train=True)
    metric.update(torch.randn(256, 5), torch.randn(256, 5) + 0.5, train=False)
    print("Score:", float(metric.compute()))
