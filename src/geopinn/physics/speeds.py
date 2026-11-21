"""Built-in wave-speed profiles.

A trainable ``c(x, z)`` or a rock-physics map is still a tensor or a callable
passed straight to :class:`geopinn.physics.acoustic.AcousticWave`. The profiles
here are fixed heterogeneous media the 2D forward benchmark can name from
YAML. They use the constant-density strong form

    u_tt = c(x, z)^2 (u_xx + u_zz) + s,

not the divergence form ``∇·(c² ∇u)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor


def layered_profile(
    z: np.ndarray,
    *,
    c_top: float,
    c_bottom: float,
    interface: float,
    transition: float,
) -> np.ndarray:
    """Smooth two-layer speed, fast or slow depending on which side is larger.

    .. math::

        c(z) = \\frac{c_\\mathrm{top} + c_\\mathrm{bottom}}{2}
             + \\frac{c_\\mathrm{bottom} - c_\\mathrm{top}}{2}
               \\tanh\\frac{z - z_\\mathrm{int}}{w}.

    ``z < z_int`` tends to ``c_top`` and ``z > z_int`` tends to ``c_bottom``.
    ``w`` is the thickness of the ramp. A sharp jump is the limit ``w → 0``,
    which this helper does not take: the strong form used by the PINN assumes
    a classical ``c² ∇²u``.
    """
    if transition <= 0.0:
        raise ValueError("transition must be positive")
    if c_top <= 0.0 or c_bottom <= 0.0:
        raise ValueError("layer speeds must be positive")
    depth = np.asarray(z, dtype=np.float64)
    mid = 0.5 * (c_top + c_bottom)
    half = 0.5 * (c_bottom - c_top)
    return mid + half * np.tanh((depth - interface) / transition)


def layered_speed_hat(
    *,
    c_top: float,
    c_bottom: float,
    interface: float,
    transition: float,
    length: float,
    time: float,
) -> Callable[[Tensor], Tensor]:
    """Nondimensional ``c_hat(x', z') = c(z) T / Lx``.

    The training coordinates are ``x' = x / Lx`` and ``z' = z / Lx``, so the
    wave equation the residual sees is

        u'_t't' = c_hat² (u'_x'x' + u'_z'z').

    The callable takes the spatial batch ``(N, 2)`` and returns ``(N, 1)``.
    That is the same signature as a one-argument ``c(x)`` coefficient.
    """
    if length <= 0.0 or time <= 0.0:
        raise ValueError("length and time must be positive")
    if transition <= 0.0 or c_top <= 0.0 or c_bottom <= 0.0:
        raise ValueError("layer speeds and the transition width must be positive")
    scale = time / length
    mid = 0.5 * (c_top + c_bottom)
    half = 0.5 * (c_bottom - c_top)

    def speed(x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("layered c(x, z) expects coordinates of shape (N, 2)")
        z_phys = x[:, 1:2] * length
        c_phys = mid + half * torch.tanh((z_phys - interface) / transition)
        return c_phys * scale

    return speed
