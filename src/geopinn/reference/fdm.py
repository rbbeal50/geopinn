"""Explicit finite differences for the acoustic wave equation.

The 1D scheme is the second-order leapfrog method. The 2D scheme is the same
integrator on a rectangle: second-order in time, second- or fourth-order in
space, with a CFL check. The boundary treatment today is homogeneous
Dirichlet. ``boundary="absorbing"`` and ``boundary="pml"`` are the hook for a
later sponge or split PML; they raise until that update is filled in.

The 2D strong form this discretizes is

    u_tt = c(x, z)^2 (u_xx + u_zz) + s,

including a heterogeneous speed. It is the constant-density form, not
``∇·(c² ∇u)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]
BoundaryFn = Callable[[np.ndarray], None]
SpeedFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
SpeedLike = float | np.ndarray | SpeedFn
SourceFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
SpatialOrder = Literal[2, 4]
BoundaryScheme = Literal["dirichlet", "absorbing", "pml"]


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

    ``u`` has shape ``(nz, nx)``. Axis 0 is the second Cartesian coordinate,
    called ``y`` here and ``z`` by the 2D PINN (depth). Axis 1 is ``x``.
    ``wave_speed`` is the fastest speed on the grid, which is the value the
    CFL check used. ``snapshots`` stores ``(time, field)`` nearest to each
    requested sample time.
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
    spatial_order: int = 2
    snapshots: tuple[tuple[float, np.ndarray], ...] = ()

    @property
    def z(self) -> np.ndarray:
        """Second Cartesian axis. Geophysical depth. Same array as ``y``."""
        return self.y


def spatial_cfl_limit(order: int) -> float:
    """Stability limit of ``c Δt √(Σ 1/Δx_i²)`` for this spatial stencil.

    Leapfrog is stable when ``ω Δt ≤ 2``. The second-order stencil reaches
    ``|λ| = 4/h²`` at the Nyquist mode, which gives the limit 1. The
    fourth-order stencil ``(-1, 16, -30, 16, -1) / (12 h²)`` reaches
    ``|λ| = 16/(3 h²)``, which gives ``√3 / 2``. Summing the symbols over
    Cartesian axes does not change that bound on the combined ratio.
    """
    if order == 2:
        return 1.0
    if order == 4:
        return math.sqrt(3.0) / 2.0
    raise ValueError("spatial order must be 2 or 4")


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
    """Second-order Cartesian Laplacian. Boundary nodes are left at 0.

    Axis 1 is differenced with ``dx`` and axis 0 with ``dy``. In the 2D
    acoustic problem axis 0 is ``z`` and axis 1 is ``x``, so ``dy`` is ``Δz``.
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    out = np.zeros_like(field, dtype=np.float64)
    out[1:-1, 1:-1] = (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx) + (
        field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]
    ) / (dy * dy)
    return out


def laplacian_2d(field: np.ndarray, dx: float, dy: float, order: SpatialOrder) -> np.ndarray:
    """Cartesian ``u_xx + u_zz`` of order 2 or 4. Boundary nodes are left at 0.

    Fourth order uses the centered stencil

        (-u_{i-2} + 16 u_{i-1} - 30 u_i + 16 u_{i+1} - u_{i+2}) / (12 h²)

    with the odd extension through a homogeneous Dirichlet boundary,
    ``u(-s) = -u(s)``. That extension is exact for every sine mode that
    vanishes on the rectangle, so the standing-wave test sees the interior
    order rather than a one-sided closure.
    """
    if order == 2:
        return second_difference_2d(field, dx, dy)
    if order == 4:
        # Axis 0 is z (spacing dy); axis 1 is x (spacing dx).
        # ndarray arithmetic is typed as Any in the installed NumPy stubs.
        return cast(
            np.ndarray,
            _second_derivative_order4(field, dy, axis=0)
            + _second_derivative_order4(field, dx, axis=1),
        )
    raise ValueError("spatial order must be 2 or 4")


def _odd_reflect_axis(field: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Pad one axis by odd reflection through a zero boundary trace.

    Only that axis grows. Padding every axis at once would make the centered
    stencil slices disagree with the unpadded output.
    """
    n = field.shape[axis]
    if n < width + 2:
        raise ValueError("grid is too short for the fourth-order stencil")
    left = np.flip(np.take(field, range(1, width + 1), axis=axis), axis=axis)
    right = np.flip(np.take(field, range(n - width - 1, n - 1), axis=axis), axis=axis)
    return np.concatenate((-left, np.asarray(field, dtype=np.float64), -right), axis=axis)


def _second_derivative_order4(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Fourth-order pure second derivative. Endpoints stay 0."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    width = 2
    padded = _odd_reflect_axis(field, width, axis)
    out = np.zeros_like(field, dtype=np.float64)
    n = field.shape[axis]
    inv = 1.0 / (12.0 * spacing * spacing)
    center = width + 1
    stop = width + n - 1

    def shifted(start: int, end: int) -> tuple[slice, ...]:
        index = [slice(None)] * field.ndim
        index[axis] = slice(start, end)
        return tuple(index)

    out[shifted(1, n - 1)] = (
        -padded[shifted(center + 2, stop + 2)]
        + 16.0 * padded[shifted(center + 1, stop + 1)]
        - 30.0 * padded[shifted(center, stop)]
        + 16.0 * padded[shifted(center - 1, stop - 1)]
        - padded[shifted(center - 2, stop - 2)]
    ) * inv
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
    wave_speed: SpeedLike,
    dx: float,
    dy: float,
    t_final: float,
    cfl: float = 0.5,
    velocity0: np.ndarray | None = None,
    spatial_order: SpatialOrder = 2,
    source: SourceFn | None = None,
    boundary: BoundaryScheme = "dirichlet",
    snapshot_times: Sequence[float] | None = None,
) -> Wave2DSolution:
    """Integrate ``u_tt = c(x, z)² (u_xx + u_zz) + s`` on a rectangle.

    ``u0`` has shape ``(nz, nx)``: axis 0 is ``z`` (stored as ``y`` on the
    result) and axis 1 is ``x``. ``dy`` is ``Δz``. ``wave_speed`` is a positive
    float, an array of that shape, or ``c(x, z)`` evaluated on the meshgrids.
    The CFL ratio uses the fastest speed. ``spatial_order`` is 2 or 4.
    ``boundary`` is ``dirichlet`` (homogeneous). ``absorbing`` and ``pml``
    raise; that branch is where a sponge or a split PML will replace the
    near-edge update.

    ``snapshot_times`` keeps the step nearest each requested time. The final
    field is always ``u``.
    """
    initial = np.asarray(u0, dtype=np.float64)
    if initial.ndim != 2 or min(initial.shape) < 3:
        raise ValueError("u0 must have shape (nz, nx) with at least 3 nodes each")
    if spatial_order not in (2, 4):
        raise ValueError("spatial order must be 2 or 4")
    ny, nx = initial.shape
    x = np.linspace(0.0, dx * (nx - 1), nx)
    y = np.linspace(0.0, dy * (ny - 1), ny)
    zz, xx = np.meshgrid(y, x, indexing="ij")
    speed, c_max = _speed_grid(wave_speed, xx, zz)
    limit = spatial_cfl_limit(spatial_order)
    dt, n_steps, ratio = _time_step(t_final, (dx, dy), c_max, cfl, limit=limit)
    c2 = speed**2
    # The source closure reads this. integrate_leapfrog writes the time of
    # the field it is about to accelerate.
    clock = [0.0]

    def acceleration(field: np.ndarray) -> np.ndarray:
        # u_tt = c(x, z)^2 (u_xx + u_zz) + s(x, z, t).
        acc = cast(np.ndarray, c2 * laplacian_2d(field, dx, dy, spatial_order))
        if source is None:
            return acc
        forcing = np.asarray(source(xx, zz, clock[0]), dtype=np.float64)
        if forcing.shape != field.shape:
            raise ValueError(f"source shape {forcing.shape} does not match the grid {field.shape}")
        summed = acc + forcing
        return summed

    history = integrate_leapfrog(
        initial,
        acceleration,
        dt,
        n_steps,
        _boundary_operator(boundary),
        velocity0=None if velocity0 is None else np.asarray(velocity0, dtype=np.float64),
        clock=clock,
    )
    times = np.linspace(0.0, t_final, n_steps + 1)
    snapshots: list[tuple[float, np.ndarray]] = []
    if snapshot_times is not None:
        for requested in snapshot_times:
            index = int(np.argmin(np.abs(times - float(requested))))
            snapshots.append((float(times[index]), history[index].copy()))
    return Wave2DSolution(
        x=x,
        y=y,
        u=history[-1],
        dx=dx,
        dy=dy,
        dt=dt,
        cfl=ratio,
        t_final=t_final,
        wave_speed=c_max,
        spatial_order=spatial_order,
        snapshots=tuple(snapshots),
    )


def _speed_grid(
    wave_speed: SpeedLike,
    xx: np.ndarray,
    zz: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Broadcast ``c(x, z)`` onto the grid and return ``(c, max c)``."""
    if isinstance(wave_speed, bool) or isinstance(wave_speed, (int, float)):
        if isinstance(wave_speed, bool) or float(wave_speed) <= 0.0:
            raise ValueError("wave_speed must be positive")
        value = float(wave_speed)
        return np.full(xx.shape, value, dtype=np.float64), value
    if isinstance(wave_speed, np.ndarray):
        grid = np.asarray(wave_speed, dtype=np.float64)
    else:
        grid = np.asarray(wave_speed(xx, zz), dtype=np.float64)
    if grid.shape != xx.shape:
        raise ValueError(f"wave_speed shape {grid.shape} does not match the grid {xx.shape}")
    if np.any(grid <= 0.0):
        raise ValueError("wave_speed must be positive everywhere")
    return grid, float(np.max(grid))


def _boundary_operator(boundary: BoundaryScheme) -> BoundaryFn:
    """Select the lateral update. Only Dirichlet is implemented."""
    if boundary == "dirichlet":

        def apply_dirichlet(field: np.ndarray) -> None:
            # Homogeneous Dirichlet: u = 0 on x = 0, x = Lx, z = 0, z = Lz.
            field[0, :] = 0.0
            field[-1, :] = 0.0
            field[:, 0] = 0.0
            field[:, -1] = 0.0

        return apply_dirichlet
    if boundary == "absorbing":
        raise NotImplementedError(
            "Absorbing boundaries are not implemented. The hook is "
            "solve_wave_2d(boundary='absorbing'): replace the outgoing "
            "characteristic on the edge nodes inside the leapfrog update. "
            "Only Dirichlet is implemented."
        )
    if boundary == "pml":
        raise NotImplementedError(
            "A PML is not implemented. The hook is solve_wave_2d(boundary='pml'): "
            "the split-field or convolutional PML updates the sponge nodes in "
            "the same place as the Dirichlet trace. Only Dirichlet is implemented."
        )
    raise ValueError("boundary must be 'dirichlet', 'absorbing', or 'pml'")


def integrate_leapfrog(
    u0: np.ndarray,
    acceleration: ArrayFn,
    dt: float,
    n_steps: int,
    apply_bc: BoundaryFn,
    velocity0: np.ndarray | None = None,
    clock: list[float] | None = None,
) -> np.ndarray:
    """Leapfrog integrator. Returns the history including the initial field.

    ``acceleration(u)`` must return ``u_tt`` on the same shape as ``u``.
    The history has shape ``(n_steps + 1, *u.shape)``. When ``clock`` is a
    one-element list, it is set to the time of the field about to be
    accelerated so a time-dependent source can close over it.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    def stamp(instant: float) -> None:
        if clock is None:
            return
        if not clock:
            clock.append(instant)
        else:
            clock[0] = instant

    previous = np.array(u0, dtype=np.float64, copy=True)
    apply_bc(previous)
    if velocity0 is None:
        velocity = np.zeros_like(previous)
    else:
        velocity = np.array(velocity0, dtype=np.float64, copy=True)
    # u(Δt) = u(0) + Δt v0 + (Δt² / 2) u_tt(0).
    stamp(0.0)
    current = previous + dt * velocity + 0.5 * (dt**2) * acceleration(previous)
    apply_bc(current)
    frames = [previous.copy(), current.copy()]
    for step in range(1, n_steps):
        # ``current`` is the field at time ``step * dt``.
        stamp(step * dt)
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
    limit: float = 1.0,
) -> tuple[float, int, float]:
    if t_final <= 0.0:
        raise ValueError("t_final must be positive")
    if limit <= 0.0:
        raise ValueError("CFL limit must be positive")
    if not 0.0 < cfl <= limit + 1.0e-15:
        raise CFLViolation(f"requested CFL {cfl} must lie in (0, {limit}] for this spatial stencil")
    metric = math.sqrt(sum(1.0 / (spacing * spacing) for spacing in spacings))
    dt_target = cfl / (wave_speed * metric)
    n_steps = max(1, math.ceil(t_final / dt_target - 1.0e-12))
    dt = t_final / n_steps
    ratio = check_cfl(wave_speed, dt, spacings, limit=limit)
    return dt, n_steps, ratio
