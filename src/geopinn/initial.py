"""Initial displacement profiles, in a form autograd can differentiate."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

Profile = Callable[[Tensor], Tensor]
ICName = Literal["sine", "gaussian"]


def sine_torch(
    *,
    amplitude: float = 1.0,
    mode: int = 1,
    length: float = 1.0,
) -> Profile:
    """``u0(x) = A sin(n π x / L)``, which vanishes at ``x = 0`` and ``x = L``."""
    if mode < 1:
        raise ValueError("mode must be a positive integer")
    if length <= 0.0:
        raise ValueError("length must be positive")
    wavenumber = mode * math.pi / length

    def profile(x: Tensor) -> Tensor:
        return amplitude * torch.sin(wavenumber * x)

    return profile


def gaussian_torch(
    *,
    center: float,
    width: float,
    length: float = 1.0,
    amplitude: float = 1.0,
) -> Profile:
    """Gaussian bump with a linear correction so ``u0(0) = u0(L) = 0``.

    The raw profile ``exp(-(x - μ)² / (2 σ²))`` does not satisfy homogeneous
    Dirichlet conditions. Subtracting the linear interpolant of its endpoint
    values restores ``u0(0) = u0(L) = 0`` without changing the smooth interior.
    """
    if width <= 0.0 or length <= 0.0:
        raise ValueError("width and length must be positive")
    left = math.exp(-((0.0 - center) ** 2) / (2.0 * width**2))
    right = math.exp(-((length - center) ** 2) / (2.0 * width**2))

    def profile(x: Tensor) -> Tensor:
        raw = torch.exp(-((x - center) ** 2) / (2.0 * width**2))
        linear = left * (1.0 - x / length) + right * (x / length)
        return amplitude * (raw - linear)

    return profile


def displacement_torch(
    kind: ICName,
    *,
    amplitude: float = 1.0,
    mode: int = 1,
    length: float = 1.0,
    center: float = 0.5,
    width: float = 0.1,
) -> Profile:
    """Build the nondimensional initial displacement used by the trial function."""
    if kind == "sine":
        return sine_torch(amplitude=amplitude, mode=mode, length=length)
    if kind == "gaussian":
        return gaussian_torch(center=center, width=width, length=length, amplitude=amplitude)
    raise ValueError(f"unknown initial condition {kind!r}")
