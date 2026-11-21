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


def sine_2d_torch(
    *,
    amplitude: float = 1.0,
    mode_x: int = 1,
    mode_z: int = 1,
    length_x: float = 1.0,
    length_z: float = 1.0,
) -> Profile:
    """``u0(x, z) = A sin(m π x / Lx) sin(n π z / Lz)``.

    ``x`` has columns ``(x, z)``. The product vanishes on the boundary of the
    rectangle.
    """
    if mode_x < 1 or mode_z < 1:
        raise ValueError("modes must be positive integers")
    if length_x <= 0.0 or length_z <= 0.0:
        raise ValueError("lengths must be positive")
    kx = mode_x * math.pi / length_x
    kz = mode_z * math.pi / length_z

    def profile(x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("sine_2d expects coordinates of shape (N, 2)")
        return amplitude * torch.sin(kx * x[:, 0:1]) * torch.sin(kz * x[:, 1:2])

    return profile


def gaussian_2d_torch(
    *,
    center_x: float,
    center_z: float,
    width_x: float,
    width_z: float,
    length_x: float = 1.0,
    length_z: float = 1.0,
    amplitude: float = 1.0,
) -> Profile:
    """Gaussian bump multiplied by ``sin(π x / Lx) sin(π z / Lz)``.

    The window is what enforces homogeneous Dirichlet data. The formula matches
    :func:`geopinn.reference.analytic.gaussian_pulse_2d`. ``x`` has columns
    ``(x, z)``.
    """
    if min(width_x, width_z, length_x, length_z) <= 0.0:
        raise ValueError("widths and lengths must be positive")

    def profile(x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("gaussian_2d expects coordinates of shape (N, 2)")
        xs = x[:, 0:1]
        zs = x[:, 1:2]
        raw = torch.exp(
            -((xs - center_x) ** 2 / (2.0 * width_x**2) + (zs - center_z) ** 2 / (2.0 * width_z**2))
        )
        window = torch.sin(math.pi * xs / length_x) * torch.sin(math.pi * zs / length_z)
        return amplitude * raw * window

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
