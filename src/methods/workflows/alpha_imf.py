from dataclasses import dataclass, field
from typing import Literal

from .imf import IMFWorkflow


@dataclass(frozen=True)
class AlphaIMFPhase:
    iteration: int
    direction: Literal["both"] = field(default="both", init=False)

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("The IMF iteration must be positive.")

    @property
    def opposite_direction(self) -> Literal["both"]:
        return "both"


class AlphaIMFWorkflow(IMFWorkflow):
    """Advance Alpha-CSBM while both directions train in every epoch."""

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
