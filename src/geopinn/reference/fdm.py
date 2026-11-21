"""Second-order explicit finite differences for the acoustic wave equation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]
BoundaryFn = Callable[[np.ndarray], None]


class CFLViolation(ValueError):
    """The explicit scheme's CFL ratio is above the stability limit."""


@dataclass(frozen=True)
class Wave1DSolution:
    """Full space-time history of a 1D explicit solve.

    ``u[k, i]`` approximates the field at ``t[k]`` and ``x[i]``.
    """

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    dx: float
    dt: float
    cfl: float
    wave_speed: float


@dataclass(frozen=True)
class Wave2DSolution:
    """Final-time field of a 2D explicit solve, plus the step that produced it.

    ``u`` has shape ``(ny, nx)`` with ``y`` along axis 0. The full history is
    not retained; 2D support is here so the same leapfrog integrator and CFL
    check extend off the 1D benchmark.
    """

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    dx: float
    dy: float
    dt: float
    cfl: float
    t_final: float
    wave_speed: float


def cfl_ratio(wave_speed: float, dt: float, spacings: tuple[float, ...]) -> float:
    """CFL ratio ``c Δt √(Σ 1/Δx_i²)``.

    The standard second-order Cartesian stencil is stable for a ratio at most 1.
    In 1D this is ``c Δt / Δx``. In 2D with ``Δx = Δy`` it is ``c Δt √2 / Δx``.
    """
    if wave_speed < 0.0 or dt <= 0.0:
        raise ValueError("wave_speed must be non-negative and dt must be positive")
    if not spacings:
        raise ValueError("at least one spacing is required")
    accum = 0.0
    for spacing in spacings:
        if spacing <= 0.0:
            raise ValueError("spacings must be positive")
        accum += 1.0 / (spacing * spacing)
    return wave_speed * dt * math.sqrt(accum)


def check_cfl(
    wave_speed: float,
    dt: float,
    spacings: tuple[float, ...],
    *,
    limit: float = 1.0,
) -> float:
    """Return the CFL ratio, or raise :class:`CFLViolation` if it exceeds ``limit``."""
    ratio = cfl_ratio(wave_speed, dt, spacings)
    if ratio > limit + 1.0e-12:
        raise CFLViolation(
            f"CFL ratio {ratio:.6f} exceeds {limit:.6f} "
            f"(c={wave_speed}, dt={dt}, spacings={spacings})"
        )
    return ratio


def second_difference_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central ``∂²/∂x²``. Endpoints are left at 0.

    The interior stencil is ``(u_{i+1} - 2 u_i + u_{i-1}) / Δx²``. Dirichlet
    updates overwrite the endpoints, so the boundary values of this operator
    are never used as a one-sided derivative.
    """
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    out = np.zeros_like(field, dtype=np.float64)
    out[1:-1] = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (dx * dx)
    return out


def second_difference_2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order Cartesian Laplacian. Boundary nodes are left at 0."""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    out = np.zeros_like(field, dtype=np.float64)
    out[1:-1, 1:-1] = (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx) + (
        field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]
    ) / (dy * dy)
    return out


def solve_wave_1d(
    u0: np.ndarray,
    *,
    wave_speed: float,
    dx: float,
    t_final: float,
    cfl: float = 0.5,
    velocity0: np.ndarray | None = None,
) -> Wave1DSolution:
    """Integrate ``u_tt = c² u_xx`` with homogeneous Dirichlet ends.

    The scheme is the second-order leapfrog method

        (u^{n+1} - 2 u^n + u^{n-1}) / Δt² = c² (u^n_{i+1} - 2 u^n_i + u^n_{i-1}) / Δx²,

    which is stable when :func:`cfl_ratio` is at most 1. The first step uses the
    Taylor expansion

        u(Δt) = u(0) + Δt v0 + (Δt² / 2) u_tt(0),

    with ``u_tt(0) = c² u_xx(0)``, so a zero initial velocity stays second-order.
    ``Δt`` is then shortened so the last step lands on ``t_final``.
    """
    initial = np.asarray(u0, dtype=np.float64)
    if initial.ndim != 1 or initial.shape[0] < 3:
        raise ValueError("u0 must be a 1D array with at least 3 nodes")
    dt, n_steps, ratio = _time_step(t_final, (dx,), wave_speed, cfl)
    c2 = wave_speed**2

    def acceleration(field: np.ndarray) -> np.ndarray:
        return c2 * second_difference_1d(field, dx)

    def apply_bc(field: np.ndarray) -> None:
        field[0] = 0.0
        field[-1] = 0.0

    history = integrate_leapfrog(
        initial,
        acceleration,
        dt,
        n_steps,
        apply_bc,
        velocity0=None if velocity0 is None else np.asarray(velocity0, dtype=np.float64),
    )
    n_nodes = initial.shape[0]
    return Wave1DSolution(
        x=np.linspace(0.0, dx * (n_nodes - 1), n_nodes),
        t=np.linspace(0.0, t_final, n_steps + 1),
        u=history,
        dx=dx,
        dt=dt,
        cfl=ratio,
        wave_speed=wave_speed,
    )


def solve_wave_2d(
    u0: np.ndarray,
    *,
    wave_speed: float,
    dx: float,
    dy: float,
    t_final: float,
    cfl: float = 0.5,
    velocity0: np.ndarray | None = None,
) -> Wave2DSolution:
    """Integrate ``u_tt = c² (u_xx + u_yy)`` on a rectangle with zero boundary trace.

    ``u0`` has shape ``(ny, nx)``. Only the final field is returned. The CFL
    ratio uses both spacings, so the 2D limit is stricter than the 1D one.
    """
    initial = np.asarray(u0, dtype=np.float64)
    if initial.ndim != 2 or min(initial.shape) < 3:
        raise ValueError("u0 must have shape (ny, nx) with at least 3 nodes each")
    dt, n_steps, ratio = _time_step(t_final, (dx, dy), wave_speed, cfl)
    c2 = wave_speed**2

    def acceleration(field: np.ndarray) -> np.ndarray:
        return c2 * second_difference_2d(field, dx, dy)

    def apply_bc(field: np.ndarray) -> None:
        field[0, :] = 0.0
        field[-1, :] = 0.0
        field[:, 0] = 0.0
        field[:, -1] = 0.0

    history = integrate_leapfrog(
        initial,
        acceleration,
        dt,
        n_steps,
        apply_bc,
        velocity0=None if velocity0 is None else np.asarray(velocity0, dtype=np.float64),
    )
    ny, nx = initial.shape
    return Wave2DSolution(
        x=np.linspace(0.0, dx * (nx - 1), nx),
        y=np.linspace(0.0, dy * (ny - 1), ny),
        u=history[-1],
        dx=dx,
        dy=dy,
        dt=dt,
        cfl=ratio,
        t_final=t_final,
        wave_speed=wave_speed,
    )


def integrate_leapfrog(
    u0: np.ndarray,
    acceleration: ArrayFn,
    dt: float,
    n_steps: int,
    apply_bc: BoundaryFn,
    velocity0: np.ndarray | None = None,
) -> np.ndarray:
    """Leapfrog integrator. Returns the history including the initial field.

    ``acceleration(u)`` must return ``c² ∇²u`` on the same shape as ``u``.
    The history has shape ``(n_steps + 1, *u.shape)``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    previous = np.array(u0, dtype=np.float64, copy=True)
    apply_bc(previous)
    if velocity0 is None:
        velocity = np.zeros_like(previous)
    else:
        velocity = np.array(velocity0, dtype=np.float64, copy=True)
    # u(Δt) = u(0) + Δt v0 + (Δt² / 2) u_tt(0), and u_tt = c² ∇²u.
    current = previous + dt * velocity + 0.5 * (dt**2) * acceleration(previous)
    apply_bc(current)
    frames = [previous.copy(), current.copy()]
    for _ in range(1, n_steps):
        nxt = 2.0 * current - previous + (dt**2) * acceleration(current)
        apply_bc(nxt)
        previous = current
        current = nxt
        frames.append(current.copy())
    return np.stack(frames, axis=0)


def _time_step(
    t_final: float,
    spacings: tuple[float, ...],
    wave_speed: float,
    cfl: float,
) -> tuple[float, int, float]:
    if t_final <= 0.0:
        raise ValueError("t_final must be positive")
    if not 0.0 < cfl <= 1.0:
        raise CFLViolation(f"requested CFL {cfl} must lie in (0, 1]")
    metric = math.sqrt(sum(1.0 / (spacing * spacing) for spacing in spacings))
    dt_target = cfl / (wave_speed * metric)
    n_steps = max(1, math.ceil(t_final / dt_target - 1.0e-12))
    dt = t_final / n_steps
    ratio = check_cfl(wave_speed, dt, spacings, limit=1.0)
    return dt, n_steps, ratio
