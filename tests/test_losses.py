"""Loss weights, the optional data term, and exact enforcement of the 1D constraints."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from geopinn.ansatz import HardConstraintWave1D
from geopinn.initial import gaussian_torch, sine_torch
from geopinn.losses import pinn_loss
from geopinn.nets.mlp import MLP
from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.base import PDE, ResidualTerm, Site
from geopinn.physics.fields import Fields
from geopinn.sampling import PointBatch


class _FakePDE(PDE):
    def residual_terms(self) -> tuple[ResidualTerm, ...]:
        return (
            ResidualTerm("pde", Site.INTERIOR, _ones(1.0)),
            ResidualTerm("bc", Site.BOUNDARY, _ones(2.0)),
            ResidualTerm("ic_u", Site.INITIAL, _ones(3.0)),
            ResidualTerm("ic_ut", Site.INITIAL, _ones(4.0)),
        )


def _ones(value: float) -> Callable[[Tensor, Tensor, Fields], Tensor]:
    def evaluate(x: Tensor, t: Tensor, fields: Fields) -> Tensor:
        del t, fields
        return value * torch.ones_like(x)

    return evaluate


def _batch(n: int, value: float) -> PointBatch:
    x = torch.full((n, 1), value)
    t = torch.full((n, 1), value)
    return PointBatch(x=x, t=t)


def test_weights_and_named_observations_add_as_mean_squares() -> None:
    weights = {
        "pde": 1.0,
        "bc": 10.0,
        "ic_u": 100.0,
        "ic_ut": 1000.0,
        "data": 0.5,
        "data_vp": 2.0,
        # A name nothing emits is ignored, so a future term can sit in the YAML.
        "regularization_c": 50.0,
    }
    prediction = torch.full((4, 1), 3.0)
    target = torch.zeros(4, 1)
    vp = torch.ones(4, 1)
    terms = pinn_loss(
        _FakePDE(),
        lambda x, t: x,
        _batch(5, 0.2),
        _batch(4, 0.0),
        _batch(6, 0.0),
        weights,
        observations={"data": (prediction, target), "data_vp": (vp, target)},
    )
    # mse(1) = 1, mse(2) = 4, mse(3) = 9, mse(4) = 16, mse(3) = 9, mse(1) = 1.
    expected = 1.0 * 1.0 + 10.0 * 4.0 + 100.0 * 9.0 + 1000.0 * 16.0 + 0.5 * 9.0 + 2.0 * 1.0
    assert torch.allclose(terms.total, torch.tensor(expected))
    residual = terms.get("pde")
    data = terms.get("data")
    assert residual is not None and data is not None
    assert torch.allclose(residual, torch.tensor(1.0))
    assert torch.allclose(data, torch.tensor(9.0))
    assert "regularization_c" not in terms.terms


def test_hard_constraints_match_dirichlet_and_both_initial_conditions() -> None:
    net = MLP(2, 1, hidden_layers=2, hidden_features=8, activation="tanh", seed=11)
    modal = HardConstraintWave1D(net, form="modal_sine")
    _assert_constraints(modal, sine_torch())

    polynomial = HardConstraintWave1D(
        MLP(2, 1, hidden_layers=2, hidden_features=8, activation="sin", seed=12),
        form="polynomial",
        initial_displacement=sine_torch(),
    )
    _assert_constraints(polynomial, sine_torch())

    gaussian = gaussian_torch(center=0.45, width=0.1, length=1.0)
    gaussian_field = HardConstraintWave1D(
        MLP(2, 1, hidden_layers=2, hidden_features=8, seed=13),
        form="polynomial",
        initial_displacement=gaussian,
    )
    _assert_constraints(gaussian_field, gaussian)


def _assert_constraints(field: nn.Module, displacement: Callable[[Tensor], Tensor]) -> None:
    pde = AcousticWave(wave_speed=1.0, initial_displacement=displacement)
    t = torch.linspace(0.0, 1.0, 21).reshape(-1, 1)
    left = pde.bc(torch.zeros_like(t), t, field)
    right = pde.bc(torch.ones_like(t), t, field)
    assert torch.allclose(left, torch.zeros_like(left), atol=1.0e-12)
    assert torch.allclose(right, torch.zeros_like(right), atol=1.0e-12)

    x = torch.linspace(0.0, 1.0, 31).reshape(-1, 1)
    t0 = torch.zeros_like(x, requires_grad=False).detach().requires_grad_(True)
    displacement_residual, velocity_residual = pde.ic(x, t0, field)
    assert torch.allclose(
        displacement_residual, torch.zeros_like(displacement_residual), atol=1.0e-10
    )
    assert torch.allclose(velocity_residual, torch.zeros_like(velocity_residual), atol=1.0e-10)
