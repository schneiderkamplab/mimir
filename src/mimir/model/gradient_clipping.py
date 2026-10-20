from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GradientClipMetrics:
    """Gradient-clipping diagnostics expressed in mean-gradient units."""

    total_norm: Tensor
    clip_coefficient: Tensor
    clipped: Tensor


def _local_scalar(value: Tensor) -> Tensor:
    # FSDP2 gradients are DTensors. Their global norm is replicated, so the
    # local scalar is safe for logging after clip_grad_norm_ has dispatched.
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    return value.detach()


def clip_grad_norm_mean_units(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
    summed_gradient_scale: float = 1.0,
) -> GradientClipMetrics:
    """Clip a global L2 norm while reporting mean-gradient-scale metrics.

    Some distributed paths retain summed gradients because their optimizer is
    scale-invariant. ``summed_gradient_scale`` maps that representation back to
    the user-facing mean-gradient convention used by ``max_norm``.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if summed_gradient_scale <= 0:
        raise ValueError("summed_gradient_scale must be positive")

    raw_max_norm = max_norm * summed_gradient_scale
    raw_total_norm = torch.nn.utils.clip_grad_norm_(
        list(parameters),
        max_norm=raw_max_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
    )
    raw_total_norm = _local_scalar(raw_total_norm)
    total_norm = raw_total_norm / summed_gradient_scale
    max_norm_tensor = total_norm.new_tensor(max_norm)
    raw_max_norm_tensor = raw_total_norm.new_tensor(raw_max_norm)
    clip_coefficient = torch.clamp(
        raw_max_norm_tensor / (raw_total_norm + 1e-6), max=1.0
    )
    clipped = (total_norm > max_norm_tensor).to(total_norm.dtype)
    return GradientClipMetrics(
        total_norm=total_norm,
        clip_coefficient=clip_coefficient,
        clipped=clipped,
    )
