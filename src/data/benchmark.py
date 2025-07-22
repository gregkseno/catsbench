from typing import Any, Dict, Literal, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from src.utils import CoupleDataset, make_infinite_dataloader
from src.benchmark import BenchmarkDiscreteEOT


class BenchmarkDataModule(LightningDataModule):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        num_potentials: int,
        num_samples: int,
        train_val_test_split: Tuple[float, float, float],
        batch_size: int,
        input_dist: Literal['gaussian', 'uniform'],
        benchmark_config: Dict[str, Any],
        num_workers: int = 0,
        pin_memory: bool = False,
        dir: str = './data',
    ) -> None:
        assert len(train_val_test_split) == 3, ( 
            "train_val_test_split must be a tuple of three floats "
            "representing the proportions for train, val, and test sets."
        )
        assert sum(train_val_test_split) == 1.0, \
            "The sum of train_val_test_split must be equal to 1.0."

        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.benchmark: Optional[BenchmarkDiscreteEOT] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # will be divided by the number of devices in `setup`
        self.batch_size_per_device = batch_size 

    def prepare_data(self) -> None:
        pass
        # BenchmarkDiscreteEOT.download(...)

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

        # here is an `if` because the `setup` method is called multiple times 
        # for trainer.fit, trainer.validate, trainer.test, etc.
        if not self.benchmark and not self.data_train and not self.data_val and not self.data_test:
            self.benchmark = BenchmarkDiscreteEOT(self.hparams.benchmark_config)

            ###################### TRAINING DATASET ######################
            size_train = int(self.hparams.num_samples * self.hparams.train_val_test_split[0])
            self.data_train = CoupleDataset(
                input_dataset=self.benchmark.input_dataset[:size_train],
                target_dataset=self.benchmark.target_dataset[:size_train],
            )

            ####################### VALIDATION DATASET ######################
            size_val = int(self.hparams.num_samples * self.hparams.train_val_test_split[0])
            self.data_val = CoupleDataset(
                input_dataset=self.benchmark.input_dataset[:size_val],
                target_dataset=self.benchmark.target_dataset[:size_val],
            )
            ######################### TEST DATASET #########################
            size_test = int(self.hparams.num_samples * self.hparams.train_val_test_split[0])
            self.data_test = CoupleDataset(
                input_dataset=self.benchmark.input_dataset[:size_test],
                target_dataset=self.benchmark.target_dataset[:size_test]
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return make_infinite_dataloader(DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        ))

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )