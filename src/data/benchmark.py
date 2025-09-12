from typing import Any, Dict, Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from src.utils.logging.console import RankedLogger
from src.utils import CoupleDataset, InfiniteCoupleDataset
from benchmark import Benchmark


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkDataModule(LightningDataModule):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_potentials: int,
        batch_size: int,
        val_batch_size: int,
        input_dist: Literal['gaussian', 'uniform'],
        benchmark_config: Dict[str, Any],
        num_workers: int = 0,
        pin_memory: bool = False,
        dir: str = './data/benchmark',
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.benchmark: Optional[Benchmark] = None # type: ignore
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass
        # Benchmark.download(...)

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
            log.info(f"batch_size per device: {self.batch_size_per_device}")
            log.info(f"val_batch_size per device: {self.val_batch_size_per_device}")

        # here is an `if` because the `setup` method is called multiple times 
        # for trainer.fit, trainer.validate, trainer.test, etc.
        if not self.benchmark and not self.data_train and not self.data_val and not self.data_test:
            device = self.trainer.strategy.root_device if self.trainer is not None else 'cpu'
            self.benchmark = Benchmark(**self.hparams.benchmark_config, device=device)
            log.info(f"Loading Benchmark datasets to {device}...")

            # Permute the target dataset to ensure unpaired setup
            random_indices = torch.randperm(len(self.benchmark.target_dataset))
            self.benchmark.target_dataset = self.benchmark.target_dataset[random_indices]

            ###################### TRAINING DATASET ######################
            self.data_train = InfiniteCoupleDataset(
                self.batch_size_per_device,
                sample_input=self.benchmark.sample_input,
                sample_target=self.benchmark.sample_target,
            )

            ####################### VALIDATION DATASET ######################
            self.data_val = CoupleDataset(
                input_dataset=self.benchmark.input_dataset,
                target_dataset=self.benchmark.target_dataset,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )