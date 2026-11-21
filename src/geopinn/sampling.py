"""Collocation points for the interior, the boundary, and the initial line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import qmc
from torch import Tensor

SamplerName = Literal["lhs", "sobol"]


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
    x_max: float = 1.0,
    t_max: float = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample the open space-time box ``[0, x_max]^d x [0, t_max]``.

    Sobol' and Latin hypercube points both live in the unit cube and are then
    scaled. Sobol' draws are low-discrepancy; ``n`` must be a power of two so
    the sequence stays balanced.
    """
    if spatial_dim < 1:
        raise ValueError("spatial_dim must be positive")
    points = _sample_box(
        n,
        lows=np.zeros(spatial_dim + 1),
        highs=np.concatenate((np.full(spatial_dim, x_max), np.array([t_max]))),
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


def sample_initial(
    n: int,
    *,
    spatial_dim: int = 1,
    length: float = 1.0,
    method: SamplerName = "sobol",
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> PointBatch:
    """Sample the initial line ``t = 0``, ``x ∈ [0, length]^d``."""
    if spatial_dim < 1 or length <= 0.0:
        raise ValueError("spatial_dim and length must be positive")
    x = _sample_box(
        n,
        lows=np.zeros(spatial_dim),
        highs=np.full(spatial_dim, length),
        method=method,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    t = torch.zeros((n, 1), dtype=dtype, device=x.device)
    return PointBatch(x=x, t=t)


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
