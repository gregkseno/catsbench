from dataclasses import dataclass, field
from typing import Literal

from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import Subset

from src.methods.base import BaseMethod
from src.utils import CoupleDataset, RepeatedDataset

from .imf import IMFCacheWriter, IMFWorkflow


@dataclass(frozen=True)
class AlphaIMFPhase:
    iteration: int
    direction: Literal["both"] = field(default="both", init=False)

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("The IMF iteration must be positive.")


class AlphaIMFWorkflow(IMFWorkflow):
    """Advance Alpha-CSBM while both directions train in every epoch."""

    # it is enought to override get_phase_by_epoch and get_phase_max_epoch 
    # to change the behavior of the workflow from IMF to Alpha-IMF.
    def get_phase_by_epoch(self, epoch: int) -> AlphaIMFPhase:
        """Return the phase by the epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        iteration = max(1, epoch - self.num_first_iterations + 2)
        return AlphaIMFPhase(iteration=iteration)

    def get_phase_max_epoch(self, epoch: int) -> int:
        """Return the last epoch of the current phase."""
        if epoch < self.num_first_iterations:
            return self.num_first_iterations
        return epoch + 1

    def build_cache(
        self,
        method: BaseMethod,
        datamodule: LightningDataModule,
        phase: AlphaIMFPhase,
        ckpt_path: str,
    ) -> None:
        if self.cache_dir is None:
            raise ValueError(
                "AlphaIMFWorkflow requires cache_dir after the first iteration."
            )

        datamodule.setup("predict")
        initial_coupling = datamodule.initial_coupling
        x_end_dataset = initial_coupling.input_dataset
        x_start_dataset = initial_coupling.target_dataset
        train_length = len(datamodule.data_train)

        cached_pairs = {}
        cache_specs = (
            ("forward", x_start_dataset, "backward"),
            ("backward", x_end_dataset, "forward"),
        )
        for direction, input_dataset, sampling_direction in cache_specs:
            writer = IMFCacheWriter(
                self.cache_dir
                / f"iteration_{phase.iteration:03d}"
                / direction,
                len(input_dataset),
            )

            if not writer.is_complete:
                method.set_predict_direction(sampling_direction)
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
            cached_pairs[direction] = (
                CoupleDataset(targets, input_dataset, cached=True)
                if direction == "forward"
                else CoupleDataset(input_dataset, targets, cached=True)
            )

        datamodule.data_train = RepeatedDataset(
            CoupleDataset(
                cached_pairs["forward"],
                cached_pairs["backward"],
                cached=True,
            ),
            length=train_length,
        )
