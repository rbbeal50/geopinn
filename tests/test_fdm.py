"""Finite-difference reference: CFL, stencil order, and second-order convergence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geopinn.reference.analytic import dalembert_dirichlet, gaussian_profile, standing_wave_1d
from geopinn.reference.fdm import (
    CFLViolation,
    cfl_ratio,
    check_cfl,
    second_difference_1d,
    second_difference_2d,
    solve_wave_1d,
    solve_wave_2d,
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
