import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig
from torch.utils.data import Subset

from src.data.benchmark_datamodule import BenchmarkDataModule
from src.methods.base import BaseMethod
from src.utils import CoupleDataset, NumpyDataset, RepeatedDataset

from .fit import BaseWorkflow


@dataclass(frozen=True)
class IMFPhase:
    direction: Literal["forward", "backward"]
    iteration: int

    def __post_init__(self) -> None:
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"Unknown IMF direction: {self.direction}")
        if self.iteration < 1:
            raise ValueError("The IMF iteration must be positive.")

    @property
    def opposite_direction(self) -> Literal["forward", "backward"]:
        return "backward" if self.direction == "forward" else "forward"


class IMFCacheWriter(BasePredictionWriter):
    """Write IMF targets one predict batch at a time."""

    def __init__(
        self,
        cache_dir,
        expected_size,
        storage_dtype="uint16",
    ):
        super().__init__(write_interval="batch")
        self.cache_dir = Path(cache_dir)
        self.targets_dir = self.cache_dir / "targets"
        self.success_path = self.cache_dir / "_SUCCESS"
        self.expected_size = expected_size
        self.storage_dtype = getattr(torch, storage_dtype)
        self.completed_ids = set(self._target_paths())
        self.pending_ids = [
            index
            for index in range(expected_size)
            if index not in self.completed_ids
        ]
        self.is_complete = self.success_path.is_file()

    def _target_paths(self):
        paths = {}
        for path in sorted(self.targets_dir.glob("*.npy")):
            try:
                input_id = int(path.stem.split("-rank-")[0])
            except ValueError:
                continue
            paths.setdefault(input_id, path)
        return paths

    def on_predict_start(self, trainer, pl_module):
        """Prepare the cache directory across all ranks."""
        if trainer.is_global_zero:
            self.targets_dir.mkdir(parents=True, exist_ok=True)
        trainer.strategy.barrier("imf-cache-start")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Save the current prediction batch to the cache."""
        if batch_indices is None:
            raise RuntimeError("Lightning did not provide predict batch indices.")

        input_ids = [self.pending_ids[index] for index in batch_indices]
        targets = prediction.detach().cpu().to(self.storage_dtype)
        for input_id, target in zip(input_ids, targets):
            path = self.targets_dir / (
                f"{input_id:08d}-rank-{trainer.global_rank}.npy"
            )
            if path.exists():
                continue
            temporary = path.with_name(f".{path.name}.tmp")
            with temporary.open("wb") as file:
                np.save(file, target.numpy())
            os.replace(temporary, path)
            self.completed_ids.add(input_id)

    def on_predict_end(self, trainer, pl_module):
        """Finalize caching"""
        trainer.strategy.barrier("imf-cache-targets")
        self.completed_ids = set(self._target_paths())
        if (
            trainer.is_global_zero
            and self.completed_ids == set(range(self.expected_size))
        ):
            temporary = self.success_path.with_suffix(".tmp")
            temporary.touch()
            os.replace(temporary, self.success_path)
        trainer.strategy.barrier("imf-cache-finish")
        self.is_complete = self.success_path.is_file()

    def get_target_dataset(self):
        if not self.is_complete:
            raise RuntimeError(f"IMF cache is incomplete: {self.cache_dir}")

        paths = self._target_paths()
        if set(paths) != set(range(self.expected_size)):
            raise RuntimeError(f"IMF cache is incomplete: {self.cache_dir}")
        return NumpyDataset([paths[index] for index in range(self.expected_size)])


class IMFWorkflow(BaseWorkflow):
    """Implement the IMF training procedure."""

    def __init__(
        self,
        trainer_config: DictConfig,
        callbacks_config: Optional[DictConfig],
        logger_config: Optional[DictConfig],
        num_first_iterations: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            trainer_config=trainer_config,
            callbacks_config=callbacks_config,
            logger_config=logger_config,
        )
        if num_first_iterations < 1:
            raise ValueError("num_first_iterations must be positive.")
        self.num_first_iterations = num_first_iterations
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def make_trainer(
        self,
        hparams: Mapping[str, Any],
        **overrides,
    ) -> Trainer:
        """Override to ensure that ModelCheckpoint(save_last=True) is used."""
        trainer = super().make_trainer(hparams, **overrides)
        checkpoint_callback = trainer.checkpoint_callback
        if (
            not isinstance(checkpoint_callback, ModelCheckpoint)
            or not checkpoint_callback.save_last
        ):
            raise ValueError(
                "IMFWorkflow requires ModelCheckpoint(save_last=True)."
            )
        return trainer

    def get_phase_by_epoch(self, epoch: int) -> IMFPhase:
        """Return the phase by the epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        if epoch < self.num_first_iterations:
            return IMFPhase(direction="forward", iteration=1)
        if epoch < 2 * self.num_first_iterations:
            return IMFPhase(direction="backward", iteration=1)

        offset = epoch - 2 * self.num_first_iterations
        iteration = 2 + offset // 2
        direction = "forward" if offset % 2 == 0 else "backward"
        return IMFPhase(direction=direction, iteration=iteration)

    def get_phase_max_epoch(self, epoch: int) -> int:
        """Return the last epoch of the current phase."""
        if epoch < self.num_first_iterations:
            return self.num_first_iterations
        if epoch < 2 * self.num_first_iterations:
            return 2 * self.num_first_iterations
        return epoch + 1

    def build_cache(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        phase: IMFPhase,
        ckpt_path: str,
    ) -> None:
        if self.cache_dir is None:
            raise ValueError(
                "IMFWorkflow requires cache_dir after the first iteration."
            )

        # since each phase the trainer objet is new
        # we need to obtain datasets that ONLY initialized in the setup method
        datamodule.setup("predict")
        input_dataset = (
            datamodule.initial_coupling.target_dataset
            if phase.direction == "forward"
            else datamodule.initial_coupling.input_dataset
        )

        # the writer loads completed ids and computes pending ids
        writer = IMFCacheWriter(
            self.cache_dir
            / f"iteration_{phase.iteration:03d}"
            / phase.direction,
            len(input_dataset),
        )

        if not writer.is_complete:
            # the subset therefore is used to predict only the pending ids
            datamodule.predict_dataset = Subset(
                input_dataset,
                writer.pending_ids,
            )
            predictor = instantiate(
                self.trainer_config,
                callbacks=[writer, TQDMProgressBar()],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )
            predictor.predict(
                method,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                return_predictions=False,
            )
            datamodule.predict_dataset = None

        targets = writer.get_target_dataset()
        pairs = (
            CoupleDataset(targets, input_dataset, cached=True)
            if phase.direction == "forward"
            else CoupleDataset(input_dataset, targets, cached=True)
        )
        datamodule.data_train = RepeatedDataset(
            pairs, length=len(datamodule.data_train),
        )

    def test(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: str,
        hparams: Mapping[str, Any],
    ) -> Trainer:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        method.set_training_phase(
            self.get_phase_by_epoch(int(checkpoint["epoch"]))
        )
        del checkpoint
        return super().test(method, datamodule, ckpt_path, hparams)

    def train(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        ckpt_path: Optional[str],
        hparams: Mapping[str, Any],
    ) -> Optional[Trainer]:
        if self.cache_dir is not None and isinstance(
            datamodule,
            BenchmarkDataModule,
        ):
            raise ValueError(
                "IMF cache is not supported for BenchmarkDataModule."
            )

        # get max and current (from checkpoint) epoch
        max_epochs = int(self.trainer_config.max_epochs)
        current_epoch = 0
        if ckpt_path is not None:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            current_epoch = int(checkpoint["epoch"]) + 1
            del checkpoint

        if current_epoch >= max_epochs:
            return None

        while current_epoch < max_epochs:
            phase = self.get_phase_by_epoch(current_epoch)
            method.set_training_phase(phase)

            if self.cache_dir is not None and phase.iteration > 1:
                self.build_cache(method, datamodule, phase, ckpt_path)

            # FIT STAGE
            trainer = self.make_trainer(
                hparams,
                max_epochs=min(
                    self.get_phase_max_epoch(current_epoch),
                    max_epochs
                )
            )
            trainer.fit(method, datamodule=datamodule, ckpt_path=ckpt_path)

            # check that ModelCheckpoint saved a last checkpoint for the next phase
            ckpt_path = trainer.checkpoint_callback.last_model_path
            if not ckpt_path:
                raise RuntimeError("ModelCheckpoint did not save a last checkpoint.")

            current_epoch = trainer.current_epoch

        return trainer
