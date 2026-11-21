"""d'Alembert with images agrees with the standing wave, including a Gaussian profile."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geopinn.initial import gaussian_torch, sine_torch
from geopinn.reference.analytic import (
    dalembert_dirichlet,
    gaussian_profile,
    sine_profile,
    standing_wave_1d,
)


def test_images_reproduce_the_standing_wave() -> None:
    length = 1.0
    c = 1.7
    x = np.linspace(0.0, length, 81)
    t = np.linspace(0.0, 1.2, 41)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    images = dalembert_dirichlet(
        xx,
        tt,
        wave_speed=c,
        length=length,
        displacement=sine_profile(length=length),
    )
    closed = standing_wave_1d(xx, tt, wave_speed=c, length=length)
    assert np.allclose(images, closed, atol=1.0e-12)
    # At t = 0 the formula collapses to the initial displacement.
    assert np.allclose(images[:, 0], np.sin(np.pi * x), atol=1.0e-12)


def test_numpy_and_torch_profiles_agree_and_vanish_at_the_ends() -> None:
    length = 1.0
    center = 0.4
    width = 0.08
    x = np.linspace(0.0, length, 201)
    numpy_sine = sine_profile(length=length)(x)
    torch_sine = sine_torch(length=length)(torch.tensor(x).reshape(-1, 1)).detach().numpy().ravel()
    assert np.allclose(numpy_sine, torch_sine, atol=1.0e-12)

    numpy_gauss = gaussian_profile(center=center, width=width, length=length)(x)
    torch_gauss = (
        gaussian_torch(center=center, width=width, length=length)(torch.tensor(x).reshape(-1, 1))
        .detach()
        .numpy()
        .ravel()
    )
    assert np.allclose(numpy_gauss, torch_gauss, atol=1.0e-12)
    assert numpy_gauss[0] == pytest.approx(0.0, abs=1.0e-14)
    assert numpy_gauss[-1] == pytest.approx(0.0, abs=1.0e-14)
    assert abs(float(numpy_sine[0])) < 1.0e-14
    assert abs(float(numpy_sine[-1])) < 1.0e-14
