"""MLP shapes, activation choices, Fourier features, and seeded initialisation."""

from __future__ import annotations

import torch
from torch import nn

from geopinn.nets.mlp import MLP


def _first_linear(net: MLP) -> nn.Linear:
    layer = net.net[0]
    assert isinstance(layer, nn.Linear)
    return layer


def test_seeded_tanh_and_sine_networks_have_the_right_shape() -> None:
    coords = torch.linspace(0.0, 1.0, 12).reshape(6, 2)
    first = MLP(2, 1, hidden_layers=3, hidden_features=16, activation="tanh", seed=1)
    second = MLP(2, 1, hidden_layers=3, hidden_features=16, activation="tanh", seed=1)
    other = MLP(2, 1, hidden_layers=3, hidden_features=16, activation="tanh", seed=2)
    assert torch.equal(_first_linear(first).weight, _first_linear(second).weight)
    assert not torch.equal(_first_linear(first).weight, _first_linear(other).weight)
    assert first(coords).shape == (6, 1)

    sine = MLP(2, 1, hidden_layers=2, hidden_features=8, activation="sin", sine_omega=2.0, seed=1)
    assert torch.isfinite(sine(coords)).all()


def test_fourier_features_keep_the_output_dimension() -> None:
    coords = torch.rand(9, 2)
    net = MLP(
        2,
        1,
        hidden_layers=2,
        hidden_features=8,
        fourier_features=6,
        fourier_sigma=0.5,
        seed=4,
    )
    assert net(coords).shape == (9, 1)
    assert net.fourier is not None
    basis = net.fourier.B
    assert isinstance(basis, torch.Tensor)
    assert basis.shape == (2, 6)
    assert not basis.requires_grad
