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
    assert config.problem.boundary == "dirichlet"
    assert config.problem.dirichlet_value == pytest.approx(0.0)
    assert config.constraints == "hard"
    assert config.constraint_form == "modal_sine"
    assert config.loss.balance == "none"
    assert config.loss.weights["pde"] == pytest.approx(1.0)
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
    soft["constraints"] = "soft"
    with pytest.raises(ConfigError, match="none"):
        TrainConfig.from_dict(soft)


def test_constraints_default_to_soft_and_both_modes_parse() -> None:
    hard = load_config(ROOT / "configs" / "wave1d.yaml")
    soft = load_config(ROOT / "configs" / "wave1d_soft.yaml")
    assert hard.constraints == "hard"
    assert soft.constraints == "soft"
    assert soft.constraint_form == "none"
    assert soft.loss.balance == "none"
    assert soft.network.activation == "sin"
    assert soft.problem.boundary == "dirichlet"
    wang = soft.to_dict()
    wang["loss"]["balance"] = "wang2021"
    assert TrainConfig.from_dict(wang).loss.balance == "wang2021"

    omitted = hard.to_dict()
    del omitted["constraints"]
    del omitted["constraint_form"]
    parsed = TrainConfig.from_dict(omitted)
    assert parsed.constraints == "soft"
    assert parsed.constraint_form == "none"


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
