"""Exact 1D wave solutions: standing wave and d'Alembert with images."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

Profile = Callable[[np.ndarray], np.ndarray]


def sine_profile(
    *,
    amplitude: float = 1.0,
    mode: int = 1,
    length: float = 1.0,
) -> Profile:
    """``u0(x) = A sin(n π x / L)`` on the numpy side of the verification."""
    if mode < 1 or length <= 0.0:
        raise ValueError("mode must be positive and length must be positive")
    wavenumber = mode * math.pi / length

    def profile(x: np.ndarray) -> np.ndarray:
        return amplitude * np.sin(wavenumber * np.asarray(x, dtype=np.float64))

    return profile


def gaussian_profile(
    *,
    center: float,
    width: float,
    length: float = 1.0,
    amplitude: float = 1.0,
) -> Profile:
    """Gaussian bump corrected to vanish at ``x = 0`` and ``x = L``.

    Matches :func:`geopinn.initial.gaussian_torch`. The endpoint values of the
    raw Gaussian are removed with the linear interpolant between them.
    """
    if width <= 0.0 or length <= 0.0:
        raise ValueError("width and length must be positive")
    left = math.exp(-((0.0 - center) ** 2) / (2.0 * width**2))
    right = math.exp(-((length - center) ** 2) / (2.0 * width**2))

    def profile(x: np.ndarray) -> np.ndarray:
        coord = np.asarray(x, dtype=np.float64)
        raw = np.exp(-((coord - center) ** 2) / (2.0 * width**2))
        linear = left * (1.0 - coord / length) + right * (coord / length)
        return amplitude * (raw - linear)

    return profile


def standing_wave_1d(
    x: ArrayLike,
    t: ArrayLike,
    *,
    wave_speed: float,
    length: float = 1.0,
    amplitude: float = 1.0,
    mode: int = 1,
) -> np.ndarray:
    """Exact separated solution of the homogeneous 1D wave equation.

    .. math::

        u(x, t) = A \\sin(n \\pi x / L) \\cos(n \\pi c t / L)

    satisfies ``u_tt = c² u_xx``, ``u(0, t) = u(L, t) = 0``,
    ``u(x, 0) = A sin(n π x / L)`` and ``u_t(x, 0) = 0``. ``x`` and ``t`` must
    broadcast.
    """
    if mode < 1 or length <= 0.0 or wave_speed < 0.0:
        raise ValueError("mode and length must be positive and wave_speed non-negative")
    x_arr = np.asarray(x, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    wavenumber = mode * math.pi / length
    frequency = wave_speed * wavenumber
    return amplitude * np.sin(wavenumber * x_arr) * np.cos(frequency * t_arr)


def dalembert_dirichlet(
    x: ArrayLike,
    t: ArrayLike,
    *,
    wave_speed: float,
    length: float,
    displacement: Profile,
) -> np.ndarray:
    """d'Alembert's formula on ``[0, L]`` with homogeneous Dirichlet ends.

    With zero initial velocity,

        u(x, t) = 1/2 (f_ext(x - c t) + f_ext(x + c t)),

    where ``f_ext`` is the odd periodic extension of ``displacement`` with
    period ``2L`` (method of images). ``displacement`` is defined on ``[0, L]``
    and must vanish at the endpoints, otherwise the extension is discontinuous
    and the boundary condition is not satisfied.

    For ``f(x) = sin(π x / L)`` the extension is ``sin`` itself and the formula
    reduces to :func:`standing_wave_1d`.
    """
    if length <= 0.0 or wave_speed < 0.0:
        raise ValueError("length must be positive and wave_speed non-negative")
    x_b, t_b = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
    )
    left = _odd_periodic(x_b - wave_speed * t_b, length, displacement)
    right = _odd_periodic(x_b + wave_speed * t_b, length, displacement)
    # ndarray arithmetic is typed as Any in the installed NumPy stubs.
    return cast(np.ndarray, 0.5 * (left + right))


def _odd_periodic(y: np.ndarray, length: float, profile: Profile) -> np.ndarray:
    """Odd extension about 0 and about ``L``, extended with period ``2L``."""
    period = 2.0 * length
    y_mod = np.mod(y, period)
    direct = y_mod <= length
    out = np.empty(y.shape, dtype=np.float64)
    if np.any(direct):
        out[direct] = np.asarray(profile(y_mod[direct]), dtype=np.float64)
    reflected = ~direct
    if np.any(reflected):
        mirror = period - y_mod[reflected]
        out[reflected] = -np.asarray(profile(mirror), dtype=np.float64)
    return out
