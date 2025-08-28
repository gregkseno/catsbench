from typing import Any, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
from torch.utils.data import Dataset, DataLoader

from lightning import LightningDataModule
from src.utils import CoupleDataset, make_infinite_dataloader
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal


def _continuous_to_discrete(
    batch: Union[torch.Tensor, np.ndarray], 
    num_categories: int,
    quantize_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
):
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).contiguous()
    if quantize_range is None:
        quantize_range = (-3, 3)
    bin_edges = torch.linspace(
        quantize_range[0], 
        quantize_range[1], 
        num_categories - 1
    )
    discrete_batch = torch.bucketize(batch, bin_edges)
    return discrete_batch

class DiscreteUniformDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):
        dataset = 6 * torch.rand(size=(num_samples, dim)) - 3
        if not train and dim == 2:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset  

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class DiscreteGaussianDataset(Dataset):
    def __init__(
        self, num_samples: int, dim: int, num_categories: int = 100, train: bool = True
    ):          
        dataset = torch.randn(size=[num_samples, dim])
        if not train and dim == 2:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
            
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
class DiscreteSwissRollDataset(Dataset):
    def __init__(
            self, num_samples: int, noise: float = 0.8, num_categories: int = 100, train: bool = True
        ):
        dataset = make_swiss_roll(
            n_samples=num_samples,
            noise=noise
        )[0][:, [0, 2]]  / 7.5
        if not train:
            dataset[:4] = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        dataset = _continuous_to_discrete(dataset, num_categories)
        self.dataset = dataset   

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

class ToyDataModule(LightningDataModule):
    def __init__(
        self,
        input_dataset: Union[DiscreteUniformDataset, DiscreteGaussianDataset, DiscreteSwissRollDataset],
        target_dataset: Union[DiscreteUniformDataset, DiscreteGaussianDataset, DiscreteSwissRollDataset],
        dim: int,
        num_categories: int,
        num_samples: int,
        train_test_split: Tuple[float, float, float],
        batch_size: int,
        val_batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        assert dim == 2, "This datamodule is designed for 2D data only."
        assert len(train_test_split) == 2, ( 
            "train_test_split must be a tuple of three floats "
            "representing the proportions for train and val sets."
        )
        assert sum(train_test_split) == 1.0, \
            "The sum of train_test_split must be equal to 1.0."

        super().__init__()
        # somehow this function is able to load all 
        # the method arguments and put to `self.hparams`
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # will be divided by the number of devices in `setup`
        self.batch_size_per_device = batch_size 


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
        if not self.data_train and not self.data_val and not self.data_test:
            ###################### TRAINING DATASET ######################
            size_train = int(self.hparams.num_samples * self.hparams.train_test_split[0])
            self.data_train = CoupleDataset(
                input_dataset=self.hparams.input_dataset(num_samples=size_train), 
                target_dataset=self.hparams.target_dataset(num_samples=size_train)
            )

            ####################### VALIDATION DATASET ######################
            size_val = int(self.hparams.num_samples * self.hparams.train_test_split[1])
            self.data_val = CoupleDataset(
                input_dataset=self.hparams.input_dataset(num_samples=size_val, train=False), 
                target_dataset=self.hparams.target_dataset(num_samples=size_val, train=False)
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