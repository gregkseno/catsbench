from typing import Literal, Optional, Tuple, Iterator
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.functional import auroc


class DSConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class TinyDSCNN(nn.Module):
    def __init__(self, in_channels: int = 3, w: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, w, 3, padding=1, bias=False)
        self.b1 = DSConv(w, 2*w, stride=2)
        self.b2 = DSConv(2*w, 4*w, stride=2)
        self.b3 = DSConv(4*w, 4*w, stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4*w, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.gap(x).flatten(1)
        return self.head(x).squeeze(1)

class ClassifierTwoSampleTest(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(
        self, 
        dim: int, 
        input_shape: Optional[Tuple[int, int, int]] = None,
        model: Literal['linear', 'cnn'] = 'linear',
        seal_params: bool = True,
        lr: float = 1e-2, 
        weight_decay: float = 0.0
    ):
        # if model == 'cnn':
        #     assert input_shape is not None, \
        #         '`input_shape` must be provided for CNN model!'

        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.model_type = model
        self.seal_params = seal_params
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize an inivisible model parameters of which won't be accesible from outside
        if model == 'linear':
            self.model = nn.Linear(dim, 1)
        elif model == 'cnn':
            self.model = nn.Linear(dim, 1) # TinyDSCNN(in_channels=input_shape[0])
        else:
            raise ValueError(f'Unknown model type: {model}! Supported types are: linear, cnn.')
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.add_state('probs', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:  # type: ignore[override]
        if self.seal_params:
            return iter(())
        return super().parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):  # type: ignore[override]
        if self.seal_params:
            return iter(())
        return super().named_parameters(prefix=prefix, recurse=recurse)

    def reset(self) -> None:
        super().reset()
        self.model.reset_parameters()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    @torch.no_grad()
    def _ddp_average_grads_before_step(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size()
        for p in self.model.parameters():
            if p.grad is None:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world_size)

    def update(self, real_data: torch.Tensor, pred_data: torch.Tensor, train: bool) -> None:
        # if self.model_type == 'linear':
        #     assert real_data.dim() == 2 and pred_data.dim() == 2, '`linear` expects [B, D]!'
        #     assert real_data.shape[1] == self.dim and pred_data.shape[1] == self.dim, 'Wrong dim!'
        # else:
        #     assert real_data.dim() == 4 and pred_data.dim() == 4, '`cnn` expects [B, C, H, W]!'
        #     C, H, W = self.input_shape
        #     assert real_data.shape[1] == C and real_data.shape[2] == H and real_data.shape[3] == W, 'Wrong shape!'
        #     assert pred_data.shape[1] == C and pred_data.shape[2] == H and pred_data.shape[3] == W, 'Wrong shape!'
        
        x_real = real_data.detach().float().flatten(start_dim=1)
        x_pred = pred_data.detach().float().flatten(start_dim=1)

        x = torch.cat([x_real, x_pred], dim=0)
        y = torch.cat(
            [
                torch.zeros(x_real.shape[0], device=x.device, dtype=torch.float),
                torch.ones(x_pred.shape[0], device=x.device, dtype=torch.float)
            ],
            dim=0
        )

        if train:
            self.model.train()
            with torch.enable_grad():
                loss = self.criterion(self.model(x).squeeze(-1), y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._ddp_average_grads_before_step()
                self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(self.model(x).squeeze(-1)).detach()
            self.probs.append(probs)
            self.targets.append(y.detach().long())

    def compute(self) -> torch.Tensor:
        if len(self.probs) == 0:
            return torch.tensor(0.0, dtype=torch.float)
        
        probs = dim_zero_cat(self.probs)
        targets = dim_zero_cat(self.targets)
        auroc_value = auroc(probs, targets, task='binary')
        score = 1.0 - torch.abs(auroc_value - 0.5) * 2.0
        return score


if __name__ == '__main__':
    torch.manual_seed(0)
    metric = ClassifierTwoSampleTest(dim=5, lr=1e-1)
    metric.update(torch.randn(512, 5), torch.randn(512, 5) + 0.5, train=True)
    metric.update(torch.randn(512, 5), torch.randn(512, 5) + 0.5, train=True)
    metric.update(torch.randn(256, 5), torch.randn(256, 5) + 0.5, train=False)
    print('Score:', float(metric.compute()))
