"""Coefficients, named fields, and the hook for a later c(x) inversion."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.coefficients import evaluate_coefficient
from geopinn.physics.fields import Fields


def _batch() -> tuple[Tensor, Tensor]:
    x = torch.linspace(0.2, 0.8, 6).reshape(-1, 1).detach().requires_grad_(True)
    t = torch.full((6, 1), 0.3).detach().requires_grad_(True)
    return x, t


def _standing(x: Tensor, t: Tensor) -> Tensor:
    return torch.sin(math.pi * x) * torch.cos(math.pi * t)


def test_float_tensor_and_callables_are_the_same_constant_speed() -> None:
    x, t = _batch()
    speed = 1.3

    def spatial(coord: Tensor) -> Tensor:
        return torch.full_like(coord, speed)

    def spacetime(coord: Tensor, time: Tensor) -> Tensor:
        return torch.full_like(coord, speed) + 0.0 * time

    class Speed(nn.Module):
        def forward(self, coord: Tensor) -> Tensor:
            return torch.full_like(coord, speed)

    reference = AcousticWave(wave_speed=speed).residual(x, t, _standing)
    for coeff in (torch.tensor(speed), spatial, spacetime, Speed()):
        got = AcousticWave(wave_speed=coeff).residual(x, t, _standing)
        assert torch.allclose(got, reference, atol=1.0e-12)


def test_a_trainable_speed_receives_a_gradient() -> None:
    class Speed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw = nn.Parameter(torch.tensor(0.7))

        def forward(self, coord: Tensor) -> Tensor:
            return self.raw.reshape(1, 1).expand(coord.shape[0], 1)

    speed = Speed()
    x, t = _batch()
    residual = AcousticWave(wave_speed=speed).residual(x, t, _standing)
    (grad,) = torch.autograd.grad(residual.square().mean(), speed.raw)
    assert float(grad.abs()) > 0.0


def test_spatially_varying_speed_matches_the_pointwise_strong_form() -> None:
    # u = sin(pi x) cos(pi t) solves u_tt = u_xx, so
    # u_tt - c(x)^2 u_xx = pi^2 (c(x)^2 - 1) u.
    def speed(coord: Tensor) -> Tensor:
        return 1.0 + 0.25 * coord

    x, t = _batch()
    residual = AcousticWave(wave_speed=speed).residual(x, t, _standing)
    field = _standing(x, t)
    c = evaluate_coefficient(speed, x, t)
    expected = (math.pi**2) * (c**2 - 1.0) * field
    assert torch.allclose(residual, expected, atol=1.0e-10)


def test_extra_output_columns_are_named_and_ignored_by_the_wave_equation() -> None:
    x, t = _batch()
    scalar = _standing(x, t)
    porosity = torch.full_like(scalar, 0.2)
    stacked = torch.cat((scalar, porosity), dim=1)
    fields = Fields.from_columns(stacked, ("u", "porosity"))
    assert fields["porosity"].shape == scalar.shape

    def predict(coord: Tensor, time: Tensor) -> Fields:
        del coord, time
        return fields

    reference = AcousticWave(wave_speed=1.0).residual(x, t, _standing)
    got = AcousticWave(wave_speed=1.0).residual(x, t, predict)
    assert torch.allclose(got, reference, atol=1.0e-12)
