from typing import Any, Callable, Literal, Optional

import torch
from lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..utils import CoupleDataset, RepeatedDataset


class DiscreteColoredMNISTDataset(Dataset):
    def __init__(self, target_digit: int, data_dir: str, train: bool = True, img_size: int = 32) -> None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(self._get_random_colored_images),
        ])
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
        self.dataset = (255 * torch.stack(
            [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == target_digit],
            dim=0,
        )).to(dtype=torch.int64)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _get_random_colored_images(image: torch.Tensor) -> torch.Tensor:
        hue = 360 * torch.rand(1)
        image_min = 0
        image_diff = (image - image_min) * (hue % 60) / 60
        image_inc, image_dec = image_diff, image - image_diff
        colored_image = torch.zeros((3, image.shape[1], image.shape[2]))
        hue_sector = torch.round(hue / 60) % 6
        if hue_sector == 0:
            colored_image[0], colored_image[1], colored_image[2] = image, image_inc, image_min
        elif hue_sector == 1:
            colored_image[0], colored_image[1], colored_image[2] = image_dec, image, image_min
        elif hue_sector == 2:
            colored_image[0], colored_image[1], colored_image[2] = image_min, image, image_inc
        elif hue_sector == 3:
            colored_image[0], colored_image[1], colored_image[2] = image_min, image_dec, image
        elif hue_sector == 4:
            colored_image[0], colored_image[1], colored_image[2] = image_inc, image_min, image
        else:
            colored_image[0], colored_image[1], colored_image[2] = image, image_min, image_dec
        return colored_image


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


class ImageDataModule(LightningDataModule):
    """Generic paired image marginals with a shared categorical image codec."""

    def __init__(
        self,
        input_dataset: Callable[..., Dataset],
        target_dataset: Callable[..., Dataset],
        codec: nn.Module,
        batch_size: int,
        val_batch_size: int,
        num_train_batches: int,
        train_data_mode: Literal["raw", "encoded"] = "raw",
        eval_data_mode: Literal["raw", "encoded"] = "raw",
        num_workers: int = 0,
        pin_memory: bool = False,
        num_categories: int = 256,
        dim: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("codec",), logger=False)
        self.codec = codec
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data by setting `self.data_train`, `self.data_val`, and `self.data_test`."""
        # setup is called multiple times for fit, validate, and test.
        if not self.data_train and not self.data_val and not self.data_test:
            coupled_train = CoupleDataset(
                input_dataset=self.hparams.input_dataset(),
                target_dataset=self.hparams.target_dataset(),
            )
            self.data_train = RepeatedDataset(
                coupled_train,
                length=self.hparams.num_train_batches * self.hparams.batch_size,
            )
            self.data_val = CoupleDataset(
                input_dataset=self.hparams.input_dataset(train=False),
                target_dataset=self.hparams.target_dataset(train=False),
            )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        mode = self.hparams.train_data_mode if self.trainer.training else self.hparams.eval_data_mode
        if mode == "encoded":
            return batch
        x, y = batch
        self.codec.to(x.device)
        return self.codec.encode_to_cats(x), self.codec.encode_to_cats(y)
    
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
        )
