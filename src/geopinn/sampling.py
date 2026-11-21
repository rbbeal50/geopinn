"""Collocation points for the interior, the boundary, and the initial line."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import qmc
from torch import Tensor

SamplerName = Literal["lhs", "sobol"]
Extents = float | Sequence[float]


@dataclass(frozen=True)
class PointBatch:
    """Collocation coordinates.

    ``x`` has shape ``(N, spatial_dim)`` and ``t`` has shape ``(N, 1)``.
    """

    x: Tensor
    t: Tensor


def sample_interior(
    n: int,
    *,
    spatial_dim: int = 1,
    x_max: Extents = 1.0,
    t_max: float = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample the open space-time box.

    A float ``x_max`` is the same upper bound on every spatial axis, which is
    the 1D interval and the unit square. A sequence gives one extent per axis,
    so a rectangle can be ``(1, Lz / Lx)`` after the single-length scaling.
    The last coordinate is time, in ``[0, t_max]``.

    Sobol' and Latin hypercube points both live in the unit cube and are then
    scaled. Sobol' draws are low-discrepancy; ``n`` must be a power of two so
    the sequence stays balanced.
    """
    if spatial_dim < 1:
        raise ValueError("spatial_dim must be positive")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    highs_x = _extents(spatial_dim, x_max, name="x_max")
    points = _sample_box(
        n,
        lows=np.zeros(spatial_dim + 1),
        highs=np.concatenate((highs_x, np.array([t_max]))),
        method=method,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    return PointBatch(x=points[:, :spatial_dim], t=points[:, spatial_dim:])


def sample_boundary_1d(
    n_per_side: int,
    *,
    length: float = 1.0,
    t_max: float = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample both Dirichlet ends ``x = 0`` and ``x = length``.

    ``n_per_side`` points are drawn on each end, with independent streams, and
    stacked so the batch has ``2 n_per_side`` rows.
    """
    if length <= 0.0 or t_max <= 0.0:
        raise ValueError("length and t_max must be positive")
    target = device or torch.device("cpu")
    t_left = _sample_box(
        n_per_side,
        lows=np.array([0.0]),
        highs=np.array([t_max]),
        method=method,
        seed=seed,
        dtype=dtype,
        device=target,
    )
    t_right = _sample_box(
        n_per_side,
        lows=np.array([0.0]),
        highs=np.array([t_max]),
        method=method,
        seed=seed + 1,
        dtype=dtype,
        device=target,
    )
    n = n_per_side
    x = torch.empty((2 * n, 1), dtype=dtype, device=target)
    x[:n, 0] = 0.0
    x[n:, 0] = length
    t = torch.cat((t_left, t_right), dim=0)
    return PointBatch(x=x, t=t)


def sample_boundary(
    n_per_edge: int,
    *,
    lengths: tuple[float, ...] = (1.0,),
    t_max: float = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample the boundary of a 1D interval or a 2D rectangle, for ``t ∈ [0, t_max]``.

    One length delegates to :func:`sample_boundary_1d`. Two lengths ``(Lx, Lz)``
    sample the four edges ``x = 0``, ``x = Lx``, ``z = 0``, and ``z = Lz``,
    with ``n_per_edge`` points on each. ``x`` then has shape ``(4 n, 2)``.
    """
    if len(lengths) == 1:
        return sample_boundary_1d(
            n_per_edge,
            length=lengths[0],
            t_max=t_max,
            method=method,
            seed=seed,
            dtype=dtype,
            device=device,
        )
    if len(lengths) == 2:
        return _sample_boundary_rectangle(
            n_per_edge,
            length_x=lengths[0],
            length_z=lengths[1],
            t_max=t_max,
            method=method,
            seed=seed,
            dtype=dtype,
            device=device,
        )
    raise ValueError("boundary sampling supports 1 or 2 spatial dimensions")


def _sample_boundary_rectangle(
    n_per_edge: int,
    *,
    length_x: float,
    length_z: float,
    t_max: float,
    method: SamplerName,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> PointBatch:
    if length_x <= 0.0 or length_z <= 0.0 or t_max <= 0.0:
        raise ValueError("lengths and t_max must be positive")
    target = device or torch.device("cpu")
    # (fixed axis, fixed value, free-axis upper bound). Axis 0 is x, axis 1 is z.
    edges = (
        (0, 0.0, length_z),
        (0, length_x, length_z),
        (1, 0.0, length_x),
        (1, length_z, length_x),
    )
    coordinates: list[Tensor] = []
    times: list[Tensor] = []
    for index, (axis, fixed, free_max) in enumerate(edges):
        free = _sample_box(
            n_per_edge,
            lows=np.zeros(2),
            highs=np.array([free_max, t_max]),
            method=method,
            seed=seed + index,
            dtype=dtype,
            device=target,
        )
        coord = torch.zeros((n_per_edge, 2), dtype=dtype, device=target)
        if axis == 0:
            coord[:, 0] = fixed
            coord[:, 1] = free[:, 0]
        else:
            coord[:, 0] = free[:, 0]
            coord[:, 1] = fixed
        coordinates.append(coord)
        times.append(free[:, 1:2])
    return PointBatch(x=torch.cat(coordinates, dim=0), t=torch.cat(times, dim=0))


def sample_initial(
    n: int,
    *,
    spatial_dim: int = 1,
    length: Extents = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample the initial slice ``t = 0``.

    ``length`` is one upper bound, or one per spatial axis. In 2D the columns
    of ``x`` are ``(x, z)``.
    """
    if spatial_dim < 1:
        raise ValueError("spatial_dim must be positive")
    highs = _extents(spatial_dim, length, name="length")
    x = _sample_box(
        n,
        lows=np.zeros(spatial_dim),
        highs=highs,
        method=method,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    t = torch.zeros((n, 1), dtype=dtype, device=x.device)
    return PointBatch(x=x, t=t)


def _extents(spatial_dim: int, value: Extents, *, name: str) -> np.ndarray:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        if isinstance(value, bool) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")
        return np.full(spatial_dim, float(value), dtype=np.float64)
    highs = np.asarray(list(value), dtype=np.float64)
    if highs.shape != (spatial_dim,):
        raise ValueError(f"{name} must be a float or {spatial_dim} positive extents")
    if np.any(highs <= 0.0):
        raise ValueError(f"{name} extents must be positive")
    return highs


def _sample_box(
    n: int,
    *,
    lows: np.ndarray,
    highs: np.ndarray,
    method: SamplerName,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | None,
) -> Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    unit = _unit_sample(n, int(lows.shape[0]), method, seed)
    values = lows + unit * (highs - lows)
    target = device or torch.device("cpu")
    array = np.ascontiguousarray(values, dtype=np.float64)
    return torch.tensor(array, dtype=dtype, device=target)


def _unit_sample(n: int, dim: int, method: SamplerName, seed: int) -> np.ndarray:
    if method == "lhs":
        engine = qmc.LatinHypercube(d=dim, seed=seed)
        return np.asarray(engine.random(n), dtype=np.float64)
    if method == "sobol":
        if not _is_power_of_two(n):
            raise ValueError(
                "Sobol sample count must be a power of two so the sequence stays balanced"
            )
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return np.asarray(engine.random(n), dtype=np.float64)
    raise ValueError(f"unknown sampler {method!r}; expected 'lhs' or 'sobol'")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
