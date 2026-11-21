"""Soft constraints: loss terms on the exact solution, and generic BC/IC targets."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from geopinn.initial import sine_torch
from geopinn.losses import pinn_loss
from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.boundary import BoundaryCondition, BoundaryKind
from geopinn.sampling import sample_boundary_1d, sample_initial, sample_interior

WEIGHTS = {"pde": 1.0, "bc": 10.0, "ic_u": 10.0, "ic_ut": 10.0}


def _standing(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.cos(math.pi * t)


def test_boundary_and_initial_samplers_hit_their_sets() -> None:
    boundary = sample_boundary_1d(16, length=1.0, t_max=1.0, method="sobol", seed=1)
    initial = sample_initial(16, length=1.0, method="sobol", seed=2)
    assert torch.all((boundary.x == 0.0) | (boundary.x == 1.0))
    assert torch.all((boundary.t >= 0.0) & (boundary.t <= 1.0))
    assert torch.all(initial.t == 0.0)
    assert torch.all((initial.x >= 0.0) & (initial.x <= 1.0))
    assert boundary.x.shape == (32, 1)
    assert initial.x.shape == (16, 1)


def test_soft_terms_vanish_on_the_exact_solution() -> None:
    pde = AcousticWave(
        wave_speed=1.0,
        boundary=BoundaryCondition(BoundaryKind.DIRICHLET, 0.0),
        initial_displacement=sine_torch(),
        initial_velocity=0.0,
    )
    interior = sample_interior(32, method="sobol", seed=0)
    boundary = sample_boundary_1d(16, method="sobol", seed=1)
    initial = sample_initial(16, method="sobol", seed=2)
    terms = pinn_loss(
        pde,
        _standing,
        interior,
        boundary,
        initial,
        WEIGHTS,
        monitor_constraints=True,
    )
    for name in ("pde", "bc", "ic_u", "ic_ut"):
        value = terms.get(name)
        assert value is not None
        assert float(value.detach()) < 1.0e-8


def test_dirichlet_target_comes_from_the_problem() -> None:
    target = 0.3
    pde = AcousticWave(
        wave_speed=1.0,
        boundary=BoundaryCondition(BoundaryKind.DIRICHLET, target),
        initial_displacement=sine_torch(),
        initial_velocity=0.0,
    )
    boundary = sample_boundary_1d(8, method="lhs", seed=3)

    def predict_target(x: Tensor, t: Tensor) -> Tensor:
        del t
        return torch.full_like(x, target)

    matched = pde.bc(boundary.x, boundary.t, predict_target)
    assert torch.allclose(matched, torch.zeros_like(matched), atol=1.0e-12)

    def predict_zero(x: Tensor, t: Tensor) -> Tensor:
        del t
        return torch.zeros_like(x)

    missed = pde.bc(boundary.x, boundary.t, predict_zero)
    assert torch.allclose(missed, torch.full_like(missed, -target), atol=1.0e-12)


def test_initial_velocity_target_is_the_supplied_value() -> None:
    pde = AcousticWave(
        wave_speed=1.0,
        initial_displacement=sine_torch(),
        initial_velocity=0.5,
    )
    x = torch.linspace(0.1, 0.9, 9).reshape(-1, 1)
    t = torch.zeros_like(x).requires_grad_(True)
    # Standing wave has u_t(x, 0) = 0, so the residual is -v0.
    _, velocity = pde.ic(x, t, _standing)
    assert torch.allclose(velocity, torch.full_like(velocity, -0.5), atol=1.0e-8)


def test_neumann_and_absorbing_are_explicit_hooks() -> None:
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)

    def predict(coord: Tensor, time: Tensor) -> Tensor:
        del time
        return torch.zeros_like(coord)

    neumann = AcousticWave(
        wave_speed=1.0,
        boundary=BoundaryCondition(BoundaryKind.NEUMANN, 0.0),
    )
    absorbing = AcousticWave(
        wave_speed=1.0,
        boundary=BoundaryCondition(BoundaryKind.ABSORBING, 0.0),
    )
    with pytest.raises(NotImplementedError, match="Neumann"):
        neumann.bc(x, t, predict)
    with pytest.raises(NotImplementedError, match="Absorbing"):
        absorbing.bc(x, t, predict)
