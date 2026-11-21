"""Automatic derivatives against finite differences and closed forms."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from geopinn.nets.mlp import MLP
from geopinn.physics.differentiate import laplacian, second_partial


def _fd_second(function: Callable[[Tensor], Tensor], center: Tensor, step: float) -> Tensor:
    """Fourth-order central second derivative.

    f''(x) = [-f(x+2h) + 16 f(x+h) - 30 f(x) + 16 f(x-h) - f(x-2h)] / (12 h²).
    The stencil is O(h^4), so the comparison is not limited by the usual O(h²)
    truncation of a three-point difference.
    """
    acc = torch.zeros_like(center)
    for weight, offset in ((-1.0, 2.0), (16.0, 1.0), (-30.0, 0.0), (16.0, -1.0), (-1.0, -2.0)):
        acc = acc + weight * function(center + offset * step)
    return acc / (12.0 * step * step)


def test_standing_wave_second_derivatives_match_finite_differences() -> None:
    c = 1.25
    x = torch.linspace(0.2, 0.8, 7).reshape(-1, 1)
    t = torch.linspace(0.1, 0.7, 5).reshape(-1, 1)
    xx, tt = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing="ij")
    x_flat = xx.reshape(-1, 1).detach().requires_grad_(True)
    t_flat = tt.reshape(-1, 1).detach().requires_grad_(True)

    def field_of(x_var: Tensor, t_var: Tensor) -> Tensor:
        return torch.sin(math.pi * x_var) * torch.cos(math.pi * c * t_var)

    def along_x(shifted: Tensor) -> Tensor:
        return field_of(shifted, t_flat.detach())

    def along_t(shifted: Tensor) -> Tensor:
        return field_of(x_flat.detach(), shifted)

    u = field_of(x_flat, t_flat)
    u_xx = second_partial(u, x_flat)
    u_tt = second_partial(u, t_flat)
    assert torch.allclose(
        u_xx, _fd_second(along_x, x_flat.detach(), 1.0e-3), atol=1.0e-6, rtol=1.0e-6
    )
    assert torch.allclose(
        u_tt, _fd_second(along_t, t_flat.detach(), 1.0e-3), atol=1.0e-6, rtol=1.0e-6
    )
    # Closed form: u_xx = -π² u, u_tt = -(π c)² u.
    assert torch.allclose(u_xx, -(math.pi**2) * u, atol=1.0e-10)
    assert torch.allclose(u_tt, -((math.pi * c) ** 2) * u, atol=1.0e-10)


def test_mlp_second_derivative_matches_finite_differences() -> None:
    net = MLP(2, 1, hidden_layers=2, hidden_features=8, activation="tanh", seed=3)
    x = torch.linspace(0.3, 0.7, 5).reshape(-1, 1)
    t_value = 0.4

    def field(shifted_x: Tensor) -> Tensor:
        coords = torch.cat((shifted_x, torch.full_like(shifted_x, t_value)), dim=1)
        return cast(Tensor, net(coords))

    x_leaf = x.detach().requires_grad_(True)
    got = second_partial(field(x_leaf), x_leaf)
    ref = _fd_second(field, x.detach(), 1.0e-4)
    assert torch.allclose(got, ref, atol=1.0e-5, rtol=1.0e-5)


def test_laplacian_sums_cartesian_second_derivatives() -> None:
    # ∇²[sin(πx) sin(πy)] = -2 π² sin(πx) sin(πy).
    coords_1d = torch.linspace(0.1, 0.9, 6)
    xx, yy = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).detach().requires_grad_(True)
    field = torch.sin(math.pi * points[:, 0:1]) * torch.sin(math.pi * points[:, 1:2])
    lap = laplacian(field, points)
    expected = -2.0 * (math.pi**2) * field
    assert torch.allclose(lap, expected, atol=1.0e-8)
