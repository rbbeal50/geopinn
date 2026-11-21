"""Collocation shapes, bounds, and deterministic samplers."""

from __future__ import annotations

import pytest
import torch

from geopinn.sampling import SamplerName, sample_boundary_1d, sample_initial, sample_interior


def test_interior_shapes_and_bounds_for_both_samplers() -> None:
    cases: tuple[tuple[SamplerName, int], ...] = (("sobol", 32), ("lhs", 20))
    for method, count in cases:
        batch = sample_interior(count, spatial_dim=2, x_max=1.5, t_max=0.4, method=method, seed=7)
        assert batch.x.shape == (count, 2)
        assert batch.t.shape == (count, 1)
        assert torch.all(batch.x >= 0.0) and torch.all(batch.x < 1.5 + 1.0e-12)
        assert torch.all(batch.t >= 0.0) and torch.all(batch.t < 0.4 + 1.0e-12)
        assert batch.x.dtype == torch.float64


def test_boundary_and_initial_geometry() -> None:
    boundary = sample_boundary_1d(16, length=2.0, t_max=0.5, method="sobol", seed=1)
    assert boundary.x.shape == (32, 1)
    assert torch.all(boundary.x[:16] == 0.0)
    assert torch.all(boundary.x[16:] == 2.0)
    assert torch.all(boundary.t >= 0.0) and torch.all(boundary.t < 0.5 + 1.0e-12)

    initial = sample_initial(16, spatial_dim=1, length=2.0, method="sobol", seed=2)
    assert initial.x.shape == (16, 1)
    assert torch.all(initial.t == 0.0)
    assert torch.all(initial.x >= 0.0) and torch.all(initial.x <= 2.0)


def test_sampling_is_deterministic_and_seed_dependent() -> None:
    first = sample_interior(16, method="sobol", seed=4)
    second = sample_interior(16, method="sobol", seed=4)
    other = sample_interior(16, method="sobol", seed=5)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.t, second.t)
    assert not torch.equal(first.x, other.x)


def test_sobol_rejects_a_count_that_is_not_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        sample_interior(12, method="sobol", seed=0)
