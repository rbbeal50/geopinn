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

    ``c_hat = c T / L`` should be O(1). Otherwise the PDE residual is dominated
    by the choice of units rather than by the approximation error. Training
    always happens in the primed coordinates on the unit interval; physical
    fields are recovered by multiplying by ``U``.
    """

    def __init__(
        self,
        *,
        length: float,
        time: float,
        amplitude: float,
        wave_speed: float,
    ) -> None:
        if length <= 0.0 or time <= 0.0 or wave_speed <= 0.0 or amplitude <= 0.0:
            raise ValueError("length, time, wave_speed, and amplitude must be positive")
        self.length = float(length)
        self.time = float(time)
        self.amplitude = float(amplitude)
        self.wave_speed = float(wave_speed)

    @property
    def c_hat(self) -> float:
        """Nondimensional wave speed ``c T / L``."""
        return self.wave_speed * self.time / self.length

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
