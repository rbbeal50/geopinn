"""The shipped YAML loads, round-trips, and rejects inconsistent problems."""

from __future__ import annotations

from pathlib import Path

import pytest

from geopinn.config import ConfigError, TrainConfig, load_config
from geopinn.nondim import Scales
from geopinn.physics.coefficients import CoefficientNotBuilt, resolve_wave_speed

ROOT = Path(__file__).resolve().parents[1]


def test_wave1d_config_is_an_o1_sine_problem() -> None:
    config = load_config(ROOT / "configs" / "wave1d.yaml")
    assert config.problem.ic == "sine"
    assert config.loss.hard_constraints is True
    assert config.loss.constraint_form == "modal_sine"
    assert config.loss.weights["residual"] == pytest.approx(1.0)
    assert config.network.field_names == ("u",)
    assert config.coefficients.wave_speed.form == "constant"
    assert config.seed == 0
    scales = Scales(
        length=config.problem.length,
        time=config.problem.time_final,
        amplitude=config.problem.amplitude,
        wave_speed=config.problem.wave_speed,
    )
    assert scales.c_hat == pytest.approx(1.0)
    assert TrainConfig.from_dict(config.to_dict()) == config


def test_unknown_keys_and_inconsistent_constraints_are_rejected() -> None:
    config = load_config(ROOT / "configs" / "wave1d.yaml")
    extra = config.to_dict()
    extra["not_a_field"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        TrainConfig.from_dict(extra)

    gaussian = config.to_dict()
    gaussian["problem"]["ic"] = "gaussian"
    with pytest.raises(ConfigError, match="modal_sine"):
        TrainConfig.from_dict(gaussian)

    soft = config.to_dict()
    soft["loss"]["hard_constraints"] = False
    with pytest.raises(ConfigError, match="none"):
        TrainConfig.from_dict(soft)


def test_schema_keeps_a_slot_for_a_trainable_speed_and_extra_loss_names() -> None:
    config = load_config(ROOT / "configs" / "wave1d.yaml")
    data = config.to_dict()
    data["loss"]["weights"]["regularization_c"] = 1.0e-4
    data["coefficients"]["wave_speed"] = {"form": "field", "source": "c_of_rock"}
    parsed = TrainConfig.from_dict(data)
    assert parsed.loss.weights["regularization_c"] == pytest.approx(1.0e-4)
    assert parsed.coefficients.wave_speed.source == "c_of_rock"
    with pytest.raises(CoefficientNotBuilt, match="tensor or a callable"):
        resolve_wave_speed(parsed.coefficients.wave_speed, parsed.problem.wave_speed)

    multi = config.to_dict()
    multi["network"]["field_names"] = ["u", "porosity"]
    with pytest.raises(ConfigError, match="field_names"):
        TrainConfig.from_dict(multi)
