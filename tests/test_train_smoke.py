"""A tiny training run: the loop, the checkpoint, and repeatability."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from geopinn.config import TrainConfig, load_config
from geopinn.train import build_trial, load_trained_model, predict_physical, train

ROOT = Path(__file__).resolve().parents[1]


def _tiny(tmp_path: Path, *, tag: str) -> TrainConfig:
    config = load_config(ROOT / "configs" / "wave1d.yaml")
    return replace(
        config,
        checkpoint=str(tmp_path / f"{tag}.pt"),
        log_every=1,
        network=replace(config.network, hidden_layers=2, hidden_dim=8),
        sampling=replace(
            config.sampling,
            n_interior=32,
            n_boundary_per_side=8,
            n_initial=8,
            method="lhs",
        ),
        optim=replace(config.optim, adam_steps=2, lbfgs_steps=1, adam_lr_step_frac=1.0),
    )


def test_short_run_changes_weights_and_reloads(tmp_path: Path) -> None:
    config = _tiny(tmp_path, tag="run")
    fresh = build_trial(config)
    initial = {key: value.detach().clone() for key, value in fresh.state_dict().items()}
    result = train(config, verbose=False)
    trained = result.model.state_dict()
    assert any(not torch.allclose(initial[key], trained[key]) for key in initial)
    assert result.checkpoint_path is not None
    assert result.history
    assert result.history[0]["stage"] == "adam"
    assert result.history[-1]["stage"] == "lbfgs"

    loaded, loaded_config, scales = load_trained_model(result.checkpoint_path)
    assert loaded_config.seed == config.seed
    x = np.linspace(0.0, config.problem.length, 6)
    t = np.linspace(0.0, config.problem.time_final, 4)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    original = predict_physical(result.model, result.scales, xx, tt)
    restored = predict_physical(loaded, scales, xx, tt)
    np.testing.assert_allclose(original, restored, atol=1.0e-12)


def test_same_seed_reproduces_the_short_run(tmp_path: Path) -> None:
    first = train(_tiny(tmp_path, tag="a"), verbose=False)
    second = train(_tiny(tmp_path, tag="b"), verbose=False)
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.allclose(left, right)
