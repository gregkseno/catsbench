from typing import Any, Mapping, Optional

from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig

from src.methods.base import BaseMethod

from .fit import FitWorkflow
from .phase import TrainingPhase


class IMFWorkflow(FitWorkflow):
    """Alternate one CSBM direction per epoch according to the IMF schedule."""

    def __init__(
        self,
        trainer_config: DictConfig,
        callbacks_config: Optional[DictConfig],
        logger_config: Optional[DictConfig],
        output_dir: str,
        num_first_iterations: int,
    ) -> None:
        super().__init__(
            trainer_config=trainer_config,
            callbacks_config=callbacks_config,
            logger_config=logger_config,
            output_dir=output_dir,
        )
        if num_first_iterations < 1:
            raise ValueError("num_first_iterations must be positive.")
        self.num_first_iterations = num_first_iterations

    def phase_for_epoch(self, epoch: int) -> TrainingPhase:
        """Return the phase used by the legacy CSBM schedule at ``epoch``."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        cycle = epoch // 2
        iteration = max(1, cycle - self.num_first_iterations + 2)
        direction = "forward" if epoch % 2 == 0 else "backward"
        return TrainingPhase(direction=direction, iteration=iteration)

    def _activate_phase(self, pl_module: BaseMethod, epoch: int) -> None:
        setter = getattr(pl_module, "set_training_phase", None)
        if setter is None:
            raise TypeError(
                f"{type(self).__name__} requires a phase-aware method, got "
                f"{type(pl_module).__name__}."
            )
        setter(self.phase_for_epoch(epoch))

    def setup(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        stage: str,
    ) -> None:
        if stage == "fit":
            self._activate_phase(pl_module, trainer.current_epoch)

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
    ) -> None:
        self._activate_phase(pl_module, trainer.current_epoch)

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
    ) -> None:
        self._activate_phase(pl_module, trainer.current_epoch)

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
    ) -> None:
        # Keep the in-memory method at the same resume phase as legacy CSBM.
        self._activate_phase(pl_module, trainer.current_epoch)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
    ) -> None:
        # This also covers the sanity check, which runs before on_train_start.
        self._activate_phase(pl_module, trainer.current_epoch)

    @staticmethod
    def _resume_epoch(checkpoint: Mapping[str, Any]) -> int:
        try:
            progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]
            return int(progress["current"]["processed"])
        except (KeyError, TypeError):
            return int(checkpoint.get("epoch", 0))

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        checkpoint: dict[str, Any],
    ) -> None:
        # Lightning's processed epoch count points at the phase to resume. This
        # also reproduces the legacy `iteration` value at epoch-end checkpoints.
        phase = self.phase_for_epoch(self._resume_epoch(checkpoint))
        checkpoint["training_phase"] = phase.state_dict()
        checkpoint["iteration"] = phase.iteration

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: BaseMethod,
        checkpoint: dict[str, Any],
    ) -> None:
        state = checkpoint.get("training_phase")
        phase = (
            TrainingPhase.from_state_dict(state)
            if state is not None
            else self.phase_for_epoch(self._resume_epoch(checkpoint))
        )
        pl_module.set_training_phase(phase)
