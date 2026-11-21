"""Finite-difference reference: CFL, stencil order, and second-order convergence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geopinn.reference.analytic import (
    dalembert_dirichlet,
    gaussian_profile,
    standing_wave_1d,
    standing_wave_2d,
)
from geopinn.reference.fdm import (
    CFLViolation,
    cfl_ratio,
    check_cfl,
    laplacian_2d,
    second_difference_1d,
    second_difference_2d,
    solve_wave_1d,
    solve_wave_2d,
    spatial_cfl_limit,
)


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.ravel()))
    return float(np.linalg.norm(error.ravel()) / denom)


def test_cfl_ratio_reduces_to_one_dimension_and_rejects_unstable_steps() -> None:
    assert cfl_ratio(2.0, 0.1, (0.2,)) == pytest.approx(1.0)
    # Equal spacing in 2D: c dt √2 / dx.
    assert cfl_ratio(1.0, 0.1, (0.2, 0.2)) == pytest.approx(0.1 * math.sqrt(2.0) / 0.2)
    assert check_cfl(1.0, 1.0, (1.0,)) == pytest.approx(1.0)
    with pytest.raises(CFLViolation):
        check_cfl(1.0, 1.1, (1.0,))
    with pytest.raises(CFLViolation):
        solve_wave_1d(
            np.sin(np.linspace(0.0, math.pi, 8)), wave_speed=1.0, dx=0.1, t_final=0.2, cfl=1.5
        )


def test_second_difference_is_second_order_on_a_sine() -> None:
    def interior_error(intervals: int) -> float:
        x = np.linspace(0.0, 1.0, intervals + 1)
        dx = float(x[1] - x[0])
        field = np.sin(math.pi * x)
        got = second_difference_1d(field, dx)
        exact = -(math.pi**2) * field
        return float(np.max(np.abs(got[1:-1] - exact[1:-1])))

    coarse = interior_error(40)
    fine = interior_error(80)
    order = math.log(coarse / fine) / math.log(2.0)
    assert order == pytest.approx(2.0, abs=0.05)


def _laplacian_error(intervals: int) -> float:
    axis = np.linspace(0.0, 1.0, intervals + 1)
    dx = float(axis[1] - axis[0])
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    # Wavenumbers π and 2π. The Cartesian Laplacian is -(1² + 2²) π² u.
    field = np.sin(math.pi * xx) * np.sin(2.0 * math.pi * yy)
    got = second_difference_2d(field, dx, dx)
    exact = -(1.0 + 4.0) * (math.pi**2) * field
    return float(np.max(np.abs(got[1:-1, 1:-1] - exact[1:-1, 1:-1])))


def test_second_difference_2d_is_second_order() -> None:
    coarse = _laplacian_error(40)
    fine = _laplacian_error(80)
    order = math.log(coarse / fine) / math.log(2.0)
    assert order == pytest.approx(2.0, abs=0.05)


def test_wave_1d_matches_the_standing_wave() -> None:
    intervals = 80
    dx = 1.0 / intervals
    x = np.linspace(0.0, 1.0, intervals + 1)
    solution = solve_wave_1d(np.sin(math.pi * x), wave_speed=1.0, dx=dx, t_final=1.0, cfl=0.5)
    assert solution.t[-1] == pytest.approx(1.0)
    assert solution.u.shape == (solution.t.shape[0], x.shape[0])
    exact = standing_wave_1d(x, 1.0, wave_speed=1.0)
    # A single Fourier mode is unusually accurate for this scheme (the leading
    # truncation terms partly cancel). Agreement, not the design order, is the
    # check here.
    assert _relative_l2(solution.u[-1] - exact, exact) < 1.0e-6


def test_wave_1d_broadband_pulse_converges_at_second_order() -> None:
    """Design order on a Gaussian pulse, referenced by d'Alembert with images.

    Fixed CFL keeps Δt proportional to Δx, so the global error is O(Δx² + Δt²).
    """
    t_final = 0.5
    profile = gaussian_profile(center=0.45, width=0.1)
    errors: list[float] = []
    spacings: list[float] = []
    for intervals in (80, 160, 320):
        dx = 1.0 / intervals
        x = np.linspace(0.0, 1.0, intervals + 1)
        solution = solve_wave_1d(profile(x), wave_speed=1.0, dx=dx, t_final=t_final, cfl=0.5)
        assert solution.cfl <= 0.5 + 1.0e-12
        exact = dalembert_dirichlet(
            x,
            t_final,
            wave_speed=1.0,
            length=1.0,
            displacement=profile,
        )
        errors.append(_relative_l2(solution.u[-1] - exact, exact))
        spacings.append(dx)

    assert errors[-1] < 1.0e-3
    assert errors[0] > errors[1] > errors[2]
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert orders[0] == pytest.approx(2.0, abs=0.1)
    assert orders[1] == pytest.approx(2.0, abs=0.1)


def _mode_grid(
    intervals: int, mode_x: int, mode_z: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes with axis 0 = z and axis 1 = x, matching ``solve_wave_2d``."""
    x = np.linspace(0.0, 1.0, intervals + 1)
    z = np.linspace(0.0, 1.0, intervals + 1)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    initial = np.sin(mode_x * math.pi * xx) * np.sin(mode_z * math.pi * zz)
    return xx, zz, initial


def _laplacian_error_order4(intervals: int) -> float:
    _xx, _zz, field = _mode_grid(intervals, mode_x=1, mode_z=2)
    dx = 1.0 / intervals
    # ∇² of sin(π x) sin(2 π z) is -(1² + 2²) π² u.
    got = laplacian_2d(field, dx, dx, 4)
    exact = -(1.0 + 4.0) * (math.pi**2) * field
    return float(np.max(np.abs(got[2:-2, 2:-2] - exact[2:-2, 2:-2])))


def test_laplacian_2d_fourth_order_on_a_sine_mode() -> None:
    coarse = _laplacian_error_order4(16)
    fine = _laplacian_error_order4(32)
    order = math.log(coarse / fine) / math.log(2.0)
    assert order == pytest.approx(4.0, abs=0.05)
    assert fine < coarse


def test_fourth_order_cfl_limit_rejects_a_second_order_step() -> None:
    assert spatial_cfl_limit(4) == pytest.approx(math.sqrt(3.0) / 2.0)
    intervals = 16
    dx = 1.0 / intervals
    _, _, initial = _mode_grid(intervals, 1, 1)
    with pytest.raises(CFLViolation):
        solve_wave_2d(
            initial,
            wave_speed=1.0,
            dx=dx,
            dy=dx,
            t_final=0.1,
            cfl=0.95,
            spatial_order=4,
        )


def test_absorbing_and_pml_are_named_hooks() -> None:
    initial = np.zeros((8, 8))
    initial[3, 3] = 1.0
    with pytest.raises(NotImplementedError, match="absorbing"):
        solve_wave_2d(
            initial,
            wave_speed=1.0,
            dx=0.1,
            dy=0.1,
            t_final=0.05,
            boundary="absorbing",
        )
    with pytest.raises(NotImplementedError, match="pml"):
        solve_wave_2d(
            initial,
            wave_speed=1.0,
            dx=0.1,
            dy=0.1,
            t_final=0.05,
            boundary="pml",
        )


def test_wave_2d_standing_wave_converges_at_second_order() -> None:
    """Fixed CFL, so Δt ∝ Δx and the global error is O(Δx² + Δt²).

    Modes (2, 3) keep the truncation visible. A single low mode can look
    better than the design order.
    """
    t_final = 0.3
    mode_x, mode_z = 2, 3
    omega = math.pi * math.sqrt(mode_x**2 + mode_z**2)
    errors: list[float] = []
    spacings: list[float] = []
    for intervals in (20, 40, 80):
        dx = 1.0 / intervals
        _xx, zz, initial = _mode_grid(intervals, mode_x, mode_z)
        solution = solve_wave_2d(
            initial,
            wave_speed=1.0,
            dx=dx,
            dy=dx,
            t_final=t_final,
            cfl=0.5,
            spatial_order=2,
        )
        exact = initial * math.cos(omega * t_final)
        errors.append(_relative_l2(solution.u - exact, exact))
        spacings.append(dx)
        assert solution.u.shape == initial.shape
        assert solution.z.shape == (zz.shape[0],)
    assert errors[-1] < 5.0e-3
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert orders[0] == pytest.approx(2.0, abs=0.15)
    assert orders[1] == pytest.approx(2.0, abs=0.15)


def test_wave_2d_fourth_order_scheme_on_the_standing_wave() -> None:
    """Δt ∝ Δx² so the second-order time error scales like the spatial error.

    The analytic field is ``sin(2 π x) sin(π z) cos(ω t)``.
    """
    t_final = 0.2
    mode_x, mode_z = 2, 1
    omega = math.pi * math.sqrt(float(mode_x**2 + mode_z**2))
    errors: list[float] = []
    spacings: list[float] = []
    for intervals in (16, 24, 32):
        dx = 1.0 / intervals
        # c Δt √2 / Δx = cfl, and we want Δt = 0.5 Δx², hence cfl = 0.5 √2 Δx.
        cfl = 0.5 * math.sqrt(2.0) * dx
        _, _, initial = _mode_grid(intervals, mode_x, mode_z)
        solution = solve_wave_2d(
            initial,
            wave_speed=1.0,
            dx=dx,
            dy=dx,
            t_final=t_final,
            cfl=cfl,
            spatial_order=4,
        )
        exact = initial * math.cos(omega * t_final)
        errors.append(_relative_l2(solution.u - exact, exact))
        spacings.append(dx)
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert errors[-1] < errors[0] * 0.2
    assert min(orders) > 3.2


def test_heterogeneous_cfl_uses_the_fastest_speed_and_a_callable() -> None:
    intervals = 16
    dx = 1.0 / intervals
    xx, zz, initial = _mode_grid(intervals, 1, 1)
    fast = np.full(initial.shape, 1.0)
    fast[4, 5] = 2.5

    def speed(x_mesh: np.ndarray, z_mesh: np.ndarray) -> np.ndarray:
        return 1.0 + 0.0 * x_mesh + 0.25 * z_mesh

    array_run = solve_wave_2d(initial, wave_speed=fast, dx=dx, dy=dx, t_final=0.05, cfl=0.45)
    assert array_run.wave_speed == pytest.approx(2.5)
    assert array_run.cfl <= 0.45 + 1.0e-12
    callable_run = solve_wave_2d(initial, wave_speed=speed, dx=dx, dy=dx, t_final=0.05, cfl=0.45)
    assert callable_run.wave_speed == pytest.approx(float(np.max(speed(xx, zz))))
    assert np.isfinite(callable_run.u).all()


def test_heterogeneous_manufactured_solution_converges() -> None:
    """u = sin(π x) sin(π z) cos(ω t) with ω² = 2 π² and c = 1 + 0.4 z.

    Then u_tt - c² (u_xx + u_zz) = 2 π² (c² - 1) u, which is the source.
    """
    t_final = 0.2
    omega2 = 2.0 * math.pi**2

    def speed(xx: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return 1.0 + 0.4 * zz

    def source(xx: np.ndarray, zz: np.ndarray, time: float) -> np.ndarray:
        field = np.sin(math.pi * xx) * np.sin(math.pi * zz) * math.cos(math.sqrt(omega2) * time)
        c = speed(xx, zz)
        return omega2 * (c**2 - 1.0) * field

    errors: list[float] = []
    spacings: list[float] = []
    for intervals in (16, 32, 48):
        dx = 1.0 / intervals
        _xx, _zz, initial = _mode_grid(intervals, 1, 1)
        solution = solve_wave_2d(
            initial,
            wave_speed=speed,
            dx=dx,
            dy=dx,
            t_final=t_final,
            cfl=0.45,
            source=source,
            snapshot_times=(0.0, t_final / 2.0, t_final),
        )
        exact = initial * math.cos(math.sqrt(omega2) * t_final)
        errors.append(_relative_l2(solution.u - exact, exact))
        spacings.append(dx)
        assert len(solution.snapshots) == 3
        assert solution.snapshots[-1][0] == pytest.approx(t_final)
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert errors[-1] < 2.0e-3
    assert orders[0] == pytest.approx(2.0, abs=0.25)
    assert orders[1] == pytest.approx(2.0, abs=0.25)


def test_standing_wave_2d_matches_the_separated_solution_at_the_final_time() -> None:
    intervals = 40
    dx = 1.0 / intervals
    _, _, initial = _mode_grid(intervals, 1, 1)
    t_final = 0.2
    solution = solve_wave_2d(initial, wave_speed=1.0, dx=dx, dy=dx, t_final=t_final, cfl=0.45)
    exact = standing_wave_2d(solution.x[None, :], solution.z[:, None], t_final, wave_speed=1.0)
    assert _relative_l2(solution.u - exact, exact) < 2.0e-3


def test_wave_2d_dirichlet_mode_tracks_the_exact_solution() -> None:
    c = 1.0
    t_final = 0.25
    intervals = 48
    dx = 1.0 / intervals
    axis = np.linspace(0.0, 1.0, intervals + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    initial = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    solution = solve_wave_2d(initial, wave_speed=c, dx=dx, dy=dx, t_final=t_final, cfl=0.45)
    omega = math.pi * c * math.sqrt(2.0)
    exact = initial * math.cos(omega * t_final)
    assert _relative_l2(solution.u - exact, exact) < 5.0e-3
    assert solution.cfl <= 0.45 + 1.0e-12
