from typing import Any, Callable, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from ..utils.ranked_logger import RankedLogger
from ..utils import CoupleDataset, SampledCoupleDataset
from catsbench import BenchmarkHD


log = RankedLogger(__name__, rank_zero_only=True)

class BenchmarkDataModule(LightningDataModule):
    def __init__(
        self,
        dim: int,
        input_shape: Tuple[int, ...],
        num_categories: int,
        batch_size: int,
        val_batch_size: int,
        num_train_batches: int,
        benchmark: Callable, # from_pretrained method of Benchmark classes
        num_timesteps: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.benchmark: Optional[Union[BenchmarkHD]] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # cache the benchmark initialization
        self.hparams.benchmark(init_benchmark=False, device='cpu')

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data by seting variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # here is an `if` because the `setup` method is called multiple times 
        # for trainer.fit, trainer.validate, trainer.test, etc.
        if not self.benchmark and not self.data_train and not self.data_val and not self.data_test:
            device = self.trainer.strategy.root_device if self.trainer is not None else 'cpu'
            log.info(f"Loading Benchmark datasets to {device}...")
            self.benchmark = self.hparams.benchmark(
                num_timesteps=self.hparams.num_timesteps, 
                init_benchmark=False, 
                device=device
            )

            ###################### TRAINING DATASET ######################
            self.data_train = SampledCoupleDataset(
                length=self.hparams.num_train_batches * self.hparams.batch_size,
                sample_input=self.benchmark.sample_input,
                sample_target=self.benchmark.sample_target,
            )

            ####################### VALIDATION/TEST DATASET ######################
            self.data_val = CoupleDataset(
                input_dataset=self.benchmark.input_dataset,
                target_dataset=self.benchmark.target_dataset,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
