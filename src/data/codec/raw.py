import torch
from torch import nn


class RawImageCodec(nn.Module):
    """Identity categorical codec for integer-valued raw images."""

    def __init__(self, num_categories: int = 256) -> None:
        super().__init__()
        self.num_categories = num_categories

    @torch.no_grad()
    def encode_to_cats(self, images: torch.Tensor) -> torch.Tensor:
        if images.is_floating_point() and images.numel() and images.max() <= 1:
            images = images * (self.num_categories - 1)
        return images.to(dtype=torch.int64).clamp_(0, self.num_categories - 1)

    @torch.no_grad()
    def decode_to_image(self, cats: torch.Tensor) -> torch.Tensor:
        return cats.float().div(self.num_categories - 1).clamp_(0, 1)
