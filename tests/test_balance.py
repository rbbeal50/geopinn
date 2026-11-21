"""Wang 2021 gradient-norm weights upweight the term with the smaller gradient."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from geopinn.balance import wang2021_weights


def test_wang_ratio_uses_max_reference_over_mean_other() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    # L = (a p)^2 has derivative 2 a^2 p.
    # pde: a = 10 -> |grad| = 2 * 100 * 2 = 400
    # bc:  a = 0.1 -> |grad| = 2 * 0.01 * 2 = 0.04
    # ratio = 400 / 0.04 = 10000
    pde = (10.0 * parameter) ** 2
    bc = (0.1 * parameter) ** 2
    running: dict[str, float] = {}
    updated = wang2021_weights(
        [parameter],
        {"pde": pde, "bc": bc},
        {"pde": 1.0, "bc": 1.0},
        running,
        alpha=1.0,
    )
    assert updated["pde"] == pytest.approx(1.0)
    assert updated["bc"] == pytest.approx(10000.0)
    assert running["bc"] == pytest.approx(10000.0)


def test_wang_mixes_with_the_previous_weight() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    pde = (10.0 * parameter) ** 2
    bc = (0.1 * parameter) ** 2
    running = {"bc": 1.0}
    updated = wang2021_weights(
        [parameter],
        {"pde": pde, "bc": bc},
        {"pde": 1.0, "bc": 1.0},
        running,
        alpha=0.9,
    )
    assert updated["bc"] == pytest.approx(0.1 * 1.0 + 0.9 * 10000.0)
