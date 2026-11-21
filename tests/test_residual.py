"""The strong form is zero on the exact d'Alembert solution."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from geopinn.ansatz import HardConstraintWave1D
from geopinn.initial import sine_torch
from geopinn.physics.acoustic import AcousticWave


def _grid(
    n_x: int,
    n_t: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[Tensor, Tensor]:
    x = torch.linspace(x_min, x_max, n_x)
    t = torch.linspace(t_min, t_max, n_t)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return (
        xx.reshape(-1, 1).detach().requires_grad_(True),
        tt.reshape(-1, 1).detach().requires_grad_(True),
    )


def test_residual_vanishes_on_standing_wave() -> None:
    c = 1.3
    pde = AcousticWave(wave_speed=c)
    x, t = _grid(19, 13, x_min=0.0, x_max=1.0)

    def predict(x_in: Tensor, t_in: Tensor) -> Tensor:
        # u = sin(π x) cos(π c t), the d'Alembert / standing-wave solution.
        return torch.sin(math.pi * x_in) * torch.cos(math.pi * c * t_in)

    residual = pde.residual(x, t, predict)
    field = predict(x, t)
    # Guard against a residual that is zero only because every derivative is dropped.
    assert torch.max(torch.abs(field)).item() > 0.5
    assert torch.max(torch.abs(residual)).item() < 1.0e-8


def test_residual_vanishes_with_a_manufactured_source() -> None:
    # u = sin(π x) exp(-t) gives u_tt - c² u_xx = (1 + c² π²) u, so s = (1 + c² π²) u.
    c = 0.8
    factor = 1.0 + (c**2) * (math.pi**2)

    def predict(x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(math.pi * x) * torch.exp(-t)

    def source(x: Tensor, t: Tensor) -> Tensor:
        return factor * predict(x, t)

    pde = AcousticWave(wave_speed=c, source=source)
    x, t = _grid(11, 9, x_min=0.05, x_max=0.95)
    residual = pde.residual(x, t, predict)
    assert torch.max(torch.abs(residual)).item() < 1.0e-8


def test_two_dimensional_mode_has_zero_residual() -> None:
    # u = sin(π x) sin(π y) cos(π c √2 t) solves u_tt = c² (u_xx + u_yy).
    c = 1.1
    omega = math.pi * c * math.sqrt(2.0)
    axis = torch.linspace(0.1, 0.9, 5)
    xx, yy, tt = torch.meshgrid(axis, axis, axis, indexing="ij")
    x = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).detach().requires_grad_(True)
    t = tt.reshape(-1, 1).detach().requires_grad_(True)

    def predict(x_in: Tensor, t_in: Tensor) -> Tensor:
        return (
            torch.sin(math.pi * x_in[:, 0:1])
            * torch.sin(math.pi * x_in[:, 1:2])
            * torch.cos(omega * t_in)
        )

    residual = AcousticWave(wave_speed=c).residual(x, t, predict)
    assert torch.max(torch.abs(residual)).item() < 1.0e-8


def _time_factor(t: Tensor, wave_speed: float) -> Tensor:
    """Stable value of (cos(ω t) - 1) / t², including the t = 0 limit -ω²/2."""
    omega = math.pi * wave_speed
    half = 0.5 * omega * t
    sinc = cast(Tensor, torch.special.sinc(half / math.pi))
    return -(omega**2) / 2.0 * sinc**2


class _ExactModal(nn.Module):
    def __init__(self, wave_speed: float) -> None:
        super().__init__()
        self.wave_speed = wave_speed

    def forward(self, coords: Tensor) -> Tensor:
        # N such that sin(π x) (1 + t² N) = sin(π x) cos(π c t).
        return _time_factor(coords[:, 1:2], self.wave_speed)


class _ExactPolynomial(nn.Module):
    def __init__(self, wave_speed: float) -> None:
        super().__init__()
        self.wave_speed = wave_speed

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        t = coords[:, 1:2]
        # N = sin(π x) (cos(ω t) - 1) / (t² x (1 - x)), away from the endpoints.
        return torch.sin(math.pi * x) * _time_factor(t, self.wave_speed) / (x * (1.0 - x))


def test_exact_network_inside_the_ansatz_has_zero_residual() -> None:
    c = 1.0
    pde = AcousticWave(wave_speed=c, initial_displacement=sine_torch())
    modal = HardConstraintWave1D(_ExactModal(c), form="modal_sine")
    x, t = _grid(15, 11)
    u = modal(x, t)
    exact = torch.sin(math.pi * x) * torch.cos(math.pi * c * t)
    assert torch.allclose(u, exact, atol=1.0e-10)
    # torch.special.sinc is the right pointwise limit at t = 0, but its
    # reverse-mode derivative is undefined there. The PDE residual is checked
    # off that single instant; the values above already include it.
    x_r, t_r = _grid(15, 11, t_min=0.05)
    assert torch.max(torch.abs(pde.residual(x_r, t_r, modal))).item() < 1.0e-8

    polynomial = HardConstraintWave1D(
        _ExactPolynomial(c),
        form="polynomial",
        initial_displacement=sine_torch(),
    )
    x_in, t_in = _grid(12, 8, x_min=0.15, x_max=0.85, t_min=0.05)
    u_poly = polynomial(x_in, t_in)
    exact_poly = torch.sin(math.pi * x_in) * torch.cos(math.pi * c * t_in)
    assert torch.allclose(u_poly, exact_poly, atol=1.0e-10)
    assert torch.max(torch.abs(pde.residual(x_in, t_in, polynomial))).item() < 1.0e-7
