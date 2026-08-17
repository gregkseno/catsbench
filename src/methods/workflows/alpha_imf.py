from .imf import IMFWorkflow
from .phase import TrainingPhase


class AlphaIMFWorkflow(IMFWorkflow):
    """Advance Alpha-CSBM while both directions train in every epoch."""

    def phase_for_epoch(self, epoch: int) -> TrainingPhase:
        """Return the phase used by the legacy Alpha-CSBM schedule."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        iteration = max(1, epoch - self.num_first_iterations + 2)
        return TrainingPhase(direction="both", iteration=iteration)
