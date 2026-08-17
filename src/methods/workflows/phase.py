from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


Direction = Literal["forward", "backward"]


@dataclass(frozen=True)
class TrainingPhase:
    """The method-facing state of one IMF training phase."""

    direction: Direction
    iteration: int

    def __post_init__(self) -> None:
        if self.direction not in ("forward", "backward"):
            raise ValueError(f"Unknown training direction: {self.direction}")
        if self.iteration < 1:
            raise ValueError("The IMF iteration must be positive.")

    @property
    def opposite_direction(self) -> Direction:
        return "backward" if self.direction == "forward" else "forward"

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> "TrainingPhase":
        return cls(
            direction=state_dict["direction"],
            iteration=int(state_dict["iteration"]),
        )
