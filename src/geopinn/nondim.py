"""Nondimensional scaling so the learned wave speed is O(1)."""

from __future__ import annotations

import warnings

from torch import Tensor


class ScaleWarning(UserWarning):
    """``c T / L`` is far from order one."""


class Scales:
    """Map a dimensional acoustic IBVP onto O(1) variables.

    The change of variables is

        x = L x',   t = T t',   u = U u'.

    Substituting the constant-coefficient wave equation ``u_tt = c² u_xx``
    produces

        u'_t't' = (c T / L)² u'_x'x'.

    In 2D the same reference length scales both coordinates,

        x = L x',   z = L z',   t = T t',   u = U u',

    with ``z' ∈ [0, Lz / L]``. The residual the network sees is

        u'_t't' = c_hat(x', z')² (u'_x'x' + u'_z'z'),

    and ``c_hat = c T / L`` should be O(1). A rectangle that is not a square
    is a training box whose second side is ``z_extent``, not a second wave
    speed. Training always happens in the primed coordinates; physical fields
    are recovered by multiplying by ``U``.
    """

    def __init__(
        self,
        *,
        length: float,
        time: float,
        amplitude: float,
        wave_speed: float,
        depth: float | None = None,
    ) -> None:
        if length <= 0.0 or time <= 0.0 or wave_speed <= 0.0 or amplitude <= 0.0:
            raise ValueError("length, time, wave_speed, and amplitude must be positive")
        if depth is not None and depth <= 0.0:
            raise ValueError("depth must be positive")
        self.length = float(length)
        self.time = float(time)
        self.amplitude = float(amplitude)
        self.wave_speed = float(wave_speed)
        # ``depth`` is the physical Lz. It defaults to Lx, a square.
        self.depth = float(length if depth is None else depth)

    @property
    def c_hat(self) -> float:
        """Nondimensional wave speed ``c T / L`` along the reference length."""
        return self.wave_speed * self.time / self.length

    @property
    def z_extent(self) -> float:
        """Nondimensional depth of the training box, ``Lz / Lx``."""
        return self.depth / self.length

    def x_hat(self, x: Tensor) -> Tensor:
        return x / self.length

    def t_hat(self, t: Tensor) -> Tensor:
        return t / self.time

    def u_physical(self, u_hat: Tensor) -> Tensor:
        return u_hat * self.amplitude

    def u_hat_from_physical(self, u: Tensor) -> Tensor:
        return u / self.amplitude


def warn_if_poorly_scaled(scales: Scales) -> None:
    """Warn when ``|c T / L|`` is outside ``[1e-2, 1e2]``."""
    c_hat = scales.c_hat
    if not 1.0e-2 <= abs(c_hat) <= 1.0e2:
        warnings.warn(
            f"nondimensional wave speed c*T/L = {c_hat:.3e} is far from O(1); "
            "rescale the time horizon or the length before training",
            ScaleWarning,
            stacklevel=2,
        )
