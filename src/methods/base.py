from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from lightning import LightningModule


class BaseMethod(LightningModule, ABC):
    """Common base class for all benchmark methods."""

    @abstractmethod
    def sample(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_trajectory(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError
