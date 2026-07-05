from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCodec(nn.Module, ABC):
    """Base interface for categorical image codecs."""

    @abstractmethod
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into categorical model states."""
        raise NotImplementedError

    @abstractmethod
    def decode_to_image(self, cats: torch.Tensor) -> torch.Tensor:
        """Decode categorical model states into displayable images."""
        raise NotImplementedError
