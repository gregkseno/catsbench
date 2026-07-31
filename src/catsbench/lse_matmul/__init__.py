"""Public API and backend selection for log-semiring matrix multiplication."""

import os
from typing import Literal, cast

import torch

from .pytorch import (
    any_logits_lse_matmul,
    cpu_lse_matmul,
    normalized_lse_matmul,
)


LSEImplementation = Literal["cpu", "normalized", "any", "triton"]
DEFAULT_IMPLEMENTATION: LSEImplementation = "triton"
IMPLEMENTATIONS: tuple[LSEImplementation, ...] = (
    "cpu",
    "normalized",
    "any",
    "triton",
)


def lse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    use_exp2: bool = True,
    implementation: LSEImplementation | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """Compute ``logsumexp_k(a[..., i, k] + b[..., k, j])``.

    With ``eps > 0``, compute ``log(sum_k(exp(a + b)) + eps)``. The epsilon
    participates inside every backend's reduction so exact-zero outputs and
    their gradients remain finite.

    The default ``triton`` backend provides strict log-domain CUDA arithmetic
    and automatically falls back to the exact reference reduction on CPU.
    Use ``normalized`` only for normalized log-probabilities, ``any`` for
    stabilized arbitrary logits, or ``cpu`` as an exact CPU reference.
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Inner dimension mismatch: {a.shape[-1]} != {b.shape[-2]}"
        )

    selected = implementation or os.environ.get(
        "LSE_BACKEND", DEFAULT_IMPLEMENTATION
    )
    if selected not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown LSE backend {selected!r}; expected one of {IMPLEMENTATIONS}."
        )
    backend = cast(LSEImplementation, selected)

    if backend == "cpu":
        return cpu_lse_matmul(a, b, eps=eps)
    if backend == "normalized":
        return normalized_lse_matmul(a, b, eps=eps)
    if backend == "any":
        return any_logits_lse_matmul(a, b, eps=eps)
    if not a.is_cuda and not b.is_cuda:
        return cpu_lse_matmul(a, b, eps=eps)
    if not (a.is_cuda and b.is_cuda):
        raise ValueError(
            "LSE operands must be on the same device; got one CPU and one CUDA tensor."
        )

    # Triton autotuning requires an active CUDA driver, so import it lazily.
    from .triton import triton_lse_matmul

    return triton_lse_matmul(a, b, use_exp2=use_exp2, eps=eps)

__all__ = [
    "DEFAULT_IMPLEMENTATION",
    "IMPLEMENTATIONS",
    "LSEImplementation",
    "lse_matmul",
]
