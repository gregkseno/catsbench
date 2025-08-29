from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from src.utils import CoupleDataset, InfiniteCoupleDataset
from benchmark import BenchmarkImages

class BenchmarkImagesDataModule(LightningDataModule):
    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, int, int],
        num_categories: int,
        num_potentials: int,
        batch_size: int,
        val_batch_size: int,
        benchmark_config: Dict[str, Any],
        num_workers: int = 0,
        pin_memory: bool = False,
        dir: str = './data/benchmark_images',
        generator_path: str = './checkpoints/cmnist_stylegan.pkl'
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.benchmark: Optional[BenchmarkImages] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # will be divided by the number of devices in `setup`
        self.batch_size_per_device = batch_size 

    def prepare_data(self) -> None:
        pass
        # BenchmarkImages.download(...)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data by seting variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        assert self.trainer.limit_train_batches > 1, '`self.trainer.limit_train_batches` must be set since the dataloaders are infinite!'
        # dividing here because the `trainer` is not available in the constructor
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            self.val_batch_size_per_device = self.hparams.val_batch_size // self.trainer.world_size

        # here is an `if` because the `setup` method is called multiple times 
        # for trainer.fit, trainer.validate, trainer.test, etc.
        if not self.benchmark and not self.data_train and not self.data_val and not self.data_test:
            self.benchmark = BenchmarkImages(**self.hparams.benchmark_config)
            self.benchmark.to(device=self.trainer.device)

            # Permute the target dataset to ensure unpaired setup
            random_indices = torch.randperm(len(self.benchmark.target_dataset))
            self.benchmark.target_dataset = self.benchmark.target_dataset[random_indices]

            ###################### TRAINING DATASET ######################
            # NOTE: The desired generation direction is:
            # target (noiced digit) -> input (clean digit) 
            self.data_train = InfiniteCoupleDataset(
                self.batch_size_per_device,
                sample_input=self.benchmark.sample_target,
                sample_target=self.benchmark.sample_input,
            )

            ####################### VALIDATION DATASET ######################
            self.data_val = CoupleDataset(
                input_dataset=self.benchmark.target_dataset,
                target_dataset=self.benchmark.input_dataset,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )