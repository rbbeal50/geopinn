"""2D sampling, trial functions, soft loss, and a heterogeneous speed."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from geopinn.ansatz import HardConstraintWave2D
from geopinn.config import TrainConfig, load_config
from geopinn.initial import gaussian_2d_torch, sine_2d_torch
from geopinn.losses import _causal_mse, pinn_loss
from geopinn.nets.mlp import MLP
from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.boundary import BoundaryCondition, BoundaryKind
from geopinn.physics.coefficients import evaluate_coefficient
from geopinn.physics.speeds import layered_profile, layered_speed_hat
from geopinn.reference.analytic import gaussian_pulse_2d, standing_wave_2d
from geopinn.sampling import sample_boundary, sample_initial, sample_interior
from geopinn.train import predict_physical, train

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"pde": 1.0, "bc": 10.0, "ic_u": 10.0, "ic_ut": 10.0}


def _standing(x: Tensor, t: Tensor) -> Tensor:
    # u = sin(π x) sin(π z) cos(π √2 t), the unit-square first mode at c = 1.
    omega = math.pi * math.sqrt(2.0)
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2]) * torch.cos(omega * t)


def test_rectangle_boundary_lies_on_the_four_edges() -> None:
    batch = sample_boundary(16, lengths=(1.0, 0.8), t_max=1.0, method="sobol", seed=3)
    assert batch.x.shape == (64, 2)
    assert batch.t.shape == (64, 1)
    x = batch.x[:, 0]
    z = batch.x[:, 1]
    on_x = (x == 0.0) | (x == 1.0)
    on_z = (z == 0.0) | (z == 0.8)
    assert torch.all(on_x | on_z)
    assert torch.all((batch.t >= 0.0) & (batch.t <= 1.0))
    # Each edge is present.
    assert torch.any((x == 0.0) & (z > 0.0) & (z < 0.8))
    assert torch.any((z == 0.0) & (x > 0.0) & (x < 1.0))


def test_interior_accepts_a_rectangular_box() -> None:
    batch = sample_interior(32, spatial_dim=2, x_max=(1.0, 1.5), t_max=0.4, method="sobol", seed=1)
    assert batch.x.shape == (32, 2)
    assert torch.all(batch.x[:, 0] <= 1.0)
    assert torch.all(batch.x[:, 1] <= 1.5)
    initial = sample_initial(16, spatial_dim=2, length=(1.0, 1.5), method="sobol", seed=2)
    assert initial.x.shape == (16, 2)
    assert torch.all(initial.t == 0.0)
    assert torch.all(initial.x[:, 1] <= 1.5)


def test_numpy_and_torch_pulses_agree() -> None:
    x = np.linspace(0.0, 1.0, 21)
    z = np.linspace(0.0, 1.0, 17)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    got = gaussian_pulse_2d(
        xx,
        zz,
        center_x=0.35,
        center_z=0.3,
        width_x=0.08,
        width_z=0.07,
    )
    coords = torch.tensor(np.stack((xx, zz), axis=-1).reshape(-1, 2), dtype=torch.float64)
    torch_pulse = gaussian_2d_torch(center_x=0.35, center_z=0.3, width_x=0.08, width_z=0.07)
    expect = torch_pulse(coords).reshape(xx.shape).detach().numpy()
    assert np.allclose(got, expect, atol=1.0e-12)
    assert np.all(got[0, :] == 0.0)
    assert np.all(got[:, 0] == 0.0)
    closed = standing_wave_2d(xx, zz, 0.0, wave_speed=1.0, mode_x=2, mode_z=3)
    assert np.allclose(closed, np.sin(2 * np.pi * xx) * np.sin(3 * np.pi * zz))


def test_soft_terms_vanish_on_the_2d_standing_wave() -> None:
    pde = AcousticWave(
        wave_speed=1.0,
        boundary=BoundaryCondition(BoundaryKind.DIRICHLET, 0.0),
        initial_displacement=sine_2d_torch(),
        initial_velocity=0.0,
    )
    interior = sample_interior(64, spatial_dim=2, method="sobol", seed=0)
    boundary = sample_boundary(16, lengths=(1.0, 1.0), method="sobol", seed=1)
    initial = sample_initial(32, spatial_dim=2, method="sobol", seed=2)
    terms = pinn_loss(pde, _standing, interior, boundary, initial, WEIGHTS)
    for name in ("pde", "bc", "ic_u", "ic_ut"):
        value = terms.get(name)
        assert value is not None
        assert float(value.detach()) < 1.0e-8


def test_hard_2d_ansatz_matches_dirichlet_and_both_initial_conditions() -> None:
    torch.manual_seed(0)
    net = MLP(in_features=3, out_features=1, hidden_layers=2, hidden_features=8, seed=1)
    modal = HardConstraintWave2D(net, form="modal_sine", mode_x=1, mode_z=1)
    n = 12
    edge_t = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    zeros = torch.zeros_like(edge_t)
    ones = torch.ones_like(edge_t)
    for x, z in (
        (zeros, edge_t),
        (ones, edge_t),
        (edge_t, zeros),
        (edge_t, ones),
    ):
        field = modal(torch.cat((x, z), dim=1), edge_t)
        assert torch.allclose(field, torch.zeros_like(field), atol=1.0e-12)

    xs = torch.linspace(0.05, 0.95, n)
    xx, zz = torch.meshgrid(xs, xs, indexing="ij")
    points = torch.stack((xx.reshape(-1), zz.reshape(-1)), dim=1)
    at_zero = modal(points, torch.zeros(points.shape[0], 1))
    expect = sine_2d_torch()(points)
    assert torch.allclose(at_zero, expect, atol=1.0e-12)
    leaf_x = points.detach().requires_grad_(True)
    leaf_t = torch.zeros(points.shape[0], 1).requires_grad_(True)
    velocity = torch.autograd.grad(modal(leaf_x, leaf_t).sum(), leaf_t, create_graph=False)[0]
    assert velocity is not None
    assert torch.allclose(velocity, torch.zeros_like(velocity), atol=1.0e-10)

    pulse = gaussian_2d_torch(center_x=0.4, center_z=0.45, width_x=0.1, width_z=0.12)
    polynomial = HardConstraintWave2D(net, form="polynomial", initial_displacement=pulse)
    assert torch.allclose(polynomial(points, torch.zeros(points.shape[0], 1)), pulse(points))


def test_layered_speed_matches_between_numpy_and_the_nondimensional_callable() -> None:
    z = np.linspace(0.0, 1.0, 9)
    physical = layered_profile(z, c_top=1.0, c_bottom=1.6, interface=0.5, transition=0.08)
    assert physical[0] == pytest.approx(1.0, abs=0.02)
    assert physical[-1] == pytest.approx(1.6, abs=0.02)
    speed = layered_speed_hat(
        c_top=1.0,
        c_bottom=1.6,
        interface=0.5,
        transition=0.08,
        length=1.0,
        time=1.0,
    )
    coords = torch.zeros((z.shape[0], 2), dtype=torch.float64)
    coords[:, 1] = torch.tensor(z)
    got = speed(coords).detach().numpy().reshape(-1)
    assert np.allclose(got, physical, atol=1.0e-12)


def test_variable_speed_residual_is_the_constant_density_strong_form() -> None:
    # u = sin(π x) sin(π z) cos(π t)
    # u_tt = -π² u,  u_xx + u_zz = -2 π² u
    # u_tt - c² ∇²u = π² (2 c² - 1) u
    def speed(coord: Tensor) -> Tensor:
        return 1.0 + 0.3 * coord[:, 1:2]

    axis = torch.linspace(0.15, 0.85, 5)
    xx, zz, tt = torch.meshgrid(axis, axis, axis, indexing="ij")
    x = torch.stack((xx.reshape(-1), zz.reshape(-1)), dim=1).detach().requires_grad_(True)
    t = tt.reshape(-1, 1).detach().requires_grad_(True)

    def predict(x_in: Tensor, t_in: Tensor) -> Tensor:
        return (
            torch.sin(math.pi * x_in[:, 0:1])
            * torch.sin(math.pi * x_in[:, 1:2])
            * torch.cos(math.pi * t_in)
        )

    residual = AcousticWave(wave_speed=speed).residual(x, t, predict)
    field = predict(x, t)
    c = evaluate_coefficient(speed, x, t)
    expected = (math.pi**2) * (2.0 * c**2 - 1.0) * field
    assert torch.allclose(residual, expected, atol=1.0e-8)


def test_causal_weight_keeps_an_early_residual() -> None:
    # Four bins. The first half of the samples is wrong and the second half is exact.
    # Wang, Sankaran, Perdikaris (2022): w_i = exp(-ε Σ_{k<i} L_k). Early losses
    # then dominate, so the weighted mean stays above the plain mean, which is
    # diluted by the quiet late bins.
    residual = torch.zeros(40, 1, dtype=torch.float64)
    residual[:20] = 1.0
    t = torch.linspace(0.0, 1.0, 40, dtype=torch.float64).reshape(-1, 1)
    plain = float((residual**2).mean())
    causal = float(_causal_mse(residual, t, epsilon=20.0, n_bins=4).detach())
    assert causal > plain
    zeros = _causal_mse(torch.zeros_like(residual), t, epsilon=20.0, n_bins=4)
    assert float(zeros.detach()) == pytest.approx(0.0)


def test_2d_configs_round_trip() -> None:
    for name in ("wave2d_standing_soft.yaml", "wave2d_standing_hard.yaml", "wave2d_layered.yaml"):
        config = load_config(ROOT / "configs" / name)
        assert config.problem.spatial_dim == 2
        assert TrainConfig.from_dict(config.to_dict()) == config
    soft = load_config(ROOT / "configs" / "wave2d_standing_soft.yaml")
    assert soft.constraints == "soft"
    assert soft.network.activation == "sin"
    hard = load_config(ROOT / "configs" / "wave2d_standing_hard.yaml")
    assert hard.constraints == "hard"
    assert hard.constraint_form == "modal_sine"
    layered = load_config(ROOT / "configs" / "wave2d_layered.yaml")
    assert layered.coefficients.wave_speed.form == "layered"
    assert layered.problem.ic == "gaussian"
    assert layered.loss.causal is True


def _tiny(config: TrainConfig, tmp_path: Path, tag: str) -> TrainConfig:
    return replace(
        config,
        checkpoint=str(tmp_path / f"{tag}.pt"),
        log_every=1,
        network=replace(
            config.network,
            hidden_layers=2,
            hidden_dim=8,
            fourier_features=0,
        ),
        sampling=replace(
            config.sampling,
            n_interior=32,
            n_boundary_per_side=8,
            n_initial=8,
            method="lhs",
        ),
        optim=replace(config.optim, adam_steps=2, lbfgs_steps=1, adam_lr_step_frac=1.0),
        loss=replace(config.loss, causal=False),
    )


def test_short_2d_run_reloads(tmp_path: Path) -> None:
    config = _tiny(load_config(ROOT / "configs" / "wave2d_standing_soft.yaml"), tmp_path, "soft")
    result = train(config, verbose=False)
    assert result.history[-1]["stage"] == "lbfgs"
    x = np.linspace(0.0, 1.0, 4)
    z = np.linspace(0.0, 1.0, 3)
    t = np.linspace(0.0, 1.0, 2)
    xx, zz, tt = np.meshgrid(x, z, t, indexing="ij")
    values = predict_physical(result.model, result.scales, xx, tt, z=zz)
    assert values.shape == xx.shape
    assert np.isfinite(values).all()
