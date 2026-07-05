from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCodec(nn.Module, ABC):
    """Base interface for categorical image codecs."""

    @abstractmethod
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        """Encode floating-point images in [0, 1] into categorical states."""
        raise NotImplementedError

    @abstractmethod
    def decode_to_image(self, cats: torch.Tensor) -> torch.Tensor:
        """Decode categorical states into floating-point images in [0, 1]."""
        raise NotImplementedError
