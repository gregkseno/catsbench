from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch


@dataclass
class Batch(Sequence[torch.Tensor]):
    encoded: tuple[torch.Tensor, torch.Tensor]
    raw: tuple[torch.Tensor | None, torch.Tensor | None]

    def __getitem__(self, index):
        return self.encoded[index]

    def __len__(self) -> int:
        return len(self.encoded)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.encoded)
