"""Nondimensional wave speed and the standing-wave scaling identity."""

from __future__ import annotations

import numpy as np
import pytest

from geopinn.nondim import Scales, ScaleWarning, warn_if_poorly_scaled
from geopinn.reference.analytic import standing_wave_1d


def test_c_hat_and_round_trip_of_the_standing_wave() -> None:
    scales = Scales(length=2.0, time=0.5, amplitude=3.0, wave_speed=4.0)
    # c T / L = 4 * 0.5 / 2 = 1.
    assert scales.c_hat == pytest.approx(1.0)
    x = np.linspace(0.0, scales.length, 25)
    t = np.linspace(0.0, scales.time, 17)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    physical = standing_wave_1d(
        xx,
        tt,
        wave_speed=scales.wave_speed,
        length=scales.length,
        amplitude=scales.amplitude,
    )
    hat = standing_wave_1d(
        xx / scales.length,
        tt / scales.time,
        wave_speed=scales.c_hat,
        length=1.0,
        amplitude=1.0,
    )
    assert np.allclose(physical, scales.amplitude * hat, atol=1.0e-12)


def test_poor_scaling_warns_and_bad_scales_are_rejected() -> None:
    with pytest.warns(ScaleWarning):
        warn_if_poorly_scaled(Scales(length=1.0, time=1.0, amplitude=1.0, wave_speed=1.0e-4))
    with pytest.raises(ValueError):
        Scales(length=0.0, time=1.0, amplitude=1.0, wave_speed=1.0)
