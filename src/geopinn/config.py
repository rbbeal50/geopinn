"""Typed YAML configuration for a training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

ICName = Literal["sine", "gaussian"]
ActivationName = Literal["tanh", "sin"]
SamplerName = Literal["lhs", "sobol"]
ConstraintMode = Literal["soft", "hard"]
ConstraintForm = Literal["modal_sine", "polynomial", "none"]
CoefficientForm = Literal["constant", "field", "layered"]
BoundaryName = Literal["dirichlet", "neumann", "absorbing"]
BalanceMode = Literal["none", "wang2021"]


class ConfigError(ValueError):
    """The YAML document is missing a field or has the wrong type."""


@dataclass(frozen=True)
class ProblemConfig:
    """The IBVP the trainer hands to the PDE.

    ``boundary`` selects the trace. ``dirichlet_value`` is the Dirichlet
    target ``g`` (a constant in this schema; a field target is a coefficient
    passed to ``AcousticWave`` in Python). ``ic`` selects the initial
    displacement profile. ``initial_velocity`` is ``v0``. Neither profile is
    hard-coded inside the residual.
    """

    length: float
    time_final: float
    wave_speed: float
    amplitude: float
    boundary: BoundaryName
    dirichlet_value: float
    ic: ICName
    initial_velocity: float
    gaussian_center: float
    gaussian_width: float
    spatial_dim: int = 1
    depth: float | None = None
    mode_x: int = 1
    mode_z: int = 1
    gaussian_center_z: float | None = None
    gaussian_width_z: float | None = None


@dataclass(frozen=True)
class NetworkConfig:
    """The trial network.

    ``field_names`` names each output column. ``["u"]`` is the scalar acoustic
    wavefield. A later inversion can use ``["u", "porosity", "saturation"]``
    or keep the rock properties in a separate coefficient network; either way
    the loss addresses fields by these names.
    """

    field_names: tuple[str, ...]
    hidden_layers: int
    hidden_dim: int
    activation: ActivationName
    sine_omega: float
    fourier_features: int
    fourier_sigma: float


@dataclass(frozen=True)
class SamplingConfig:
    n_interior: int
    n_boundary_per_side: int
    n_initial: int
    method: SamplerName


@dataclass(frozen=True)
class WaveSpeedSpec:
    """Where the acoustic speed comes from.

    ``constant`` uses ``problem.wave_speed`` (then nondimensionalized).
    ``layered`` is a smooth two-layer ``c(z)`` (``c_top``, ``c_bottom``,
    ``interface``, ``transition``). ``field`` reserves ``source`` as the name
    of a trainable ``c(x, z)`` or a rock-physics map. The trainer does not
    build ``field`` yet; the residual already accepts a tensor or a callable.
    """

    form: CoefficientForm
    source: str | None = None
    c_top: float | None = None
    c_bottom: float | None = None
    interface: float | None = None
    transition: float | None = None


@dataclass(frozen=True)
class CoefficientConfig:
    """Named PDE coefficients. Keys match names the equation reads."""

    wave_speed: WaveSpeedSpec


@dataclass(frozen=True)
class LossConfig:
    """Named weights and optional gradient-norm balancing.

    ``weights`` maps a residual or observation name to a non-negative
    multiplier. The acoustic names are ``pde``, ``bc``, ``ic_u``, and
    ``ic_ut``. Names the active PDE does not emit are kept, so the file can
    already list ``data_vp`` or ``regularization_c``.

    ``balance`` is ``none`` or ``wang2021``. The latter is the learning-rate
    annealing of Wang, Teng, and Perdikaris (2021): each non-PDE weight tracks
    the ratio of gradient magnitudes. It is off by default. ``balance_alpha``
    is the paper's mixing rate and ``balance_every`` is how many Adam steps
    pass between updates.

    ``causal`` multiplies the PDE residual by the time-causal weights of Wang,
    Sankaran, and Perdikaris (2022) during Adam. It is off by default, and
    L-BFGS always sees the plain mean square.
    """

    weights: dict[str, float]
    balance: BalanceMode = "none"
    balance_alpha: float = 0.9
    balance_every: int = 10
    causal: bool = False
    causal_epsilon: float = 1.0
    causal_bins: int = 16


@dataclass(frozen=True)
class OptimConfig:
    adam_steps: int
    adam_lr: float
    adam_lr_gamma: float
    adam_lr_step_frac: float
    lbfgs_steps: int
    lbfgs_lr: float
    lbfgs_history: int
    lbfgs_line_search_evals: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    log_every: int
    checkpoint: str
    constraints: ConstraintMode
    constraint_form: ConstraintForm
    problem: ProblemConfig
    coefficients: CoefficientConfig
    network: NetworkConfig
    sampling: SamplingConfig
    loss: LossConfig
    optim: OptimConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainConfig:
        _check_keys(
            data,
            {
                "seed",
                "device",
                "log_every",
                "checkpoint",
                "constraints",
                "constraint_form",
                "problem",
                "coefficients",
                "network",
                "sampling",
                "loss",
                "optim",
            },
            "config",
            optional={"constraints", "constraint_form"},
        )
        constraints = cast(
            ConstraintMode,
            _optional_choice(data, "constraints", ("soft", "hard"), default="soft"),
        )
        constraint_form = cast(
            ConstraintForm,
            _optional_choice(
                data,
                "constraint_form",
                ("modal_sine", "polynomial", "none"),
                default="none",
            ),
        )
        problem = _problem(_section(data, "problem"))
        coefficients = _coefficients(_section(data, "coefficients"))
        network = _network(_section(data, "network"))
        sampling = _sampling(_section(data, "sampling"))
        loss = _loss(_section(data, "loss"))
        optim = _optim(_section(data, "optim"))
        _validate_consistency(
            constraints,
            constraint_form,
            problem,
            coefficients,
            network,
            loss,
            sampling,
            optim,
        )
        return TrainConfig(
            seed=_req_int(data, "seed", minimum=0),
            device=_req_str(data, "device"),
            log_every=_req_int(data, "log_every", minimum=1),
            checkpoint=_req_str(data, "checkpoint"),
            constraints=constraints,
            constraint_form=constraint_form,
            problem=problem,
            coefficients=coefficients,
            network=network,
            sampling=sampling,
            loss=loss,
            optim=optim,
        )


def load_config(path: Path) -> TrainConfig:
    """Read a YAML training config."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not contain a mapping")
    return TrainConfig.from_dict(cast(dict[str, Any], raw))


def _problem(data: dict[str, Any]) -> ProblemConfig:
    _check_keys(
        data,
        {
            "length",
            "time_final",
            "wave_speed",
            "amplitude",
            "boundary",
            "dirichlet_value",
            "ic",
            "initial_velocity",
            "gaussian_center",
            "gaussian_width",
            "spatial_dim",
            "depth",
            "mode_x",
            "mode_z",
            "gaussian_center_z",
            "gaussian_width_z",
        },
        "problem",
        optional={
            "spatial_dim",
            "depth",
            "mode_x",
            "mode_z",
            "gaussian_center_z",
            "gaussian_width_z",
        },
    )
    ic = _req_choice(data, "ic", ("sine", "gaussian"))
    boundary = _req_choice(data, "boundary", ("dirichlet", "neumann", "absorbing"))
    spatial_dim = _optional_int(data, "spatial_dim", default=1)
    if spatial_dim not in (1, 2):
        raise ConfigError("problem.spatial_dim must be 1 or 2")
    return ProblemConfig(
        length=_req_float(data, "length", positive=True),
        time_final=_req_float(data, "time_final", positive=True),
        wave_speed=_req_float(data, "wave_speed", positive=True),
        amplitude=_req_float(data, "amplitude", positive=True),
        boundary=cast(BoundaryName, boundary),
        dirichlet_value=_req_float(data, "dirichlet_value"),
        ic=cast(ICName, ic),
        initial_velocity=_req_float(data, "initial_velocity"),
        gaussian_center=_req_float(data, "gaussian_center"),
        gaussian_width=_req_float(data, "gaussian_width", positive=True),
        spatial_dim=spatial_dim,
        depth=_optional_float_or_none(data, "depth", positive=True),
        mode_x=_optional_int(data, "mode_x", default=1),
        mode_z=_optional_int(data, "mode_z", default=1),
        gaussian_center_z=_optional_float_or_none(data, "gaussian_center_z"),
        gaussian_width_z=_optional_float_or_none(data, "gaussian_width_z", positive=True),
    )


def _network(data: dict[str, Any]) -> NetworkConfig:
    _check_keys(
        data,
        {
            "field_names",
            "hidden_layers",
            "hidden_dim",
            "activation",
            "sine_omega",
            "fourier_features",
            "fourier_sigma",
        },
        "network",
    )
    activation = _req_choice(data, "activation", ("tanh", "sin"))
    return NetworkConfig(
        field_names=_req_str_tuple(data, "field_names"),
        hidden_layers=_req_int(data, "hidden_layers", minimum=1),
        hidden_dim=_req_int(data, "hidden_dim", minimum=1),
        activation=cast(ActivationName, activation),
        sine_omega=_req_float(data, "sine_omega", positive=True),
        fourier_features=_req_int(data, "fourier_features", minimum=0),
        fourier_sigma=_req_float(data, "fourier_sigma", positive=True),
    )


def _coefficients(data: dict[str, Any]) -> CoefficientConfig:
    _check_keys(data, {"wave_speed"}, "coefficients")
    return CoefficientConfig(wave_speed=_wave_speed(_section(data, "wave_speed")))


def _wave_speed(data: dict[str, Any]) -> WaveSpeedSpec:
    _check_keys(
        data,
        {"form", "source", "c_top", "c_bottom", "interface", "transition"},
        "coefficients.wave_speed",
        optional={"source", "c_top", "c_bottom", "interface", "transition"},
    )
    form = _req_choice(data, "form", ("constant", "field", "layered"))
    source = _optional_str(data, "source")
    c_top = _optional_float_or_none(data, "c_top", positive=True)
    c_bottom = _optional_float_or_none(data, "c_bottom", positive=True)
    interface = _optional_float_or_none(data, "interface")
    transition = _optional_float_or_none(data, "transition", positive=True)
    if form == "constant" and source is not None:
        raise ConfigError("coefficients.wave_speed.source is only used when form is field")
    if form == "field" and source is None:
        raise ConfigError("coefficients.wave_speed.form field requires source")
    if form == "layered" and source is not None:
        raise ConfigError(
            "coefficients.wave_speed.source is the field-form slot, not a layered profile"
        )
    layered_keys = {
        "c_top": c_top,
        "c_bottom": c_bottom,
        "interface": interface,
        "transition": transition,
    }
    if form == "layered":
        missing = sorted(name for name, value in layered_keys.items() if value is None)
        if missing:
            names = ", ".join(missing)
            raise ConfigError(f"coefficients.wave_speed.form layered requires {names}")
    elif any(value is not None for value in layered_keys.values()):
        raise ConfigError("c_top, c_bottom, interface, and transition require form: layered")
    return WaveSpeedSpec(
        form=cast(CoefficientForm, form),
        source=source,
        c_top=c_top,
        c_bottom=c_bottom,
        interface=interface,
        transition=transition,
    )


def _sampling(data: dict[str, Any]) -> SamplingConfig:
    _check_keys(
        data,
        {"n_interior", "n_boundary_per_side", "n_initial", "method"},
        "sampling",
    )
    method = _req_choice(data, "method", ("lhs", "sobol"))
    return SamplingConfig(
        n_interior=_req_int(data, "n_interior", minimum=1),
        n_boundary_per_side=_req_int(data, "n_boundary_per_side", minimum=1),
        n_initial=_req_int(data, "n_initial", minimum=1),
        method=cast(SamplerName, method),
    )


def _loss(data: dict[str, Any]) -> LossConfig:
    _check_keys(
        data,
        {
            "weights",
            "balance",
            "balance_alpha",
            "balance_every",
            "causal",
            "causal_epsilon",
            "causal_bins",
        },
        "loss",
        optional={
            "balance",
            "balance_alpha",
            "balance_every",
            "causal",
            "causal_epsilon",
            "causal_bins",
        },
    )
    balance = cast(
        BalanceMode,
        _optional_choice(data, "balance", ("none", "wang2021"), default="none"),
    )
    alpha = _optional_float(data, "balance_alpha", default=0.9)
    every = _optional_int(data, "balance_every", default=10)
    return LossConfig(
        weights=_req_weights(data, "weights"),
        balance=balance,
        balance_alpha=alpha,
        balance_every=every,
        causal=_optional_bool(data, "causal", default=False),
        causal_epsilon=_optional_float(data, "causal_epsilon", default=1.0),
        causal_bins=_optional_int(data, "causal_bins", default=16),
    )


def _optim(data: dict[str, Any]) -> OptimConfig:
    _check_keys(
        data,
        {
            "adam_steps",
            "adam_lr",
            "adam_lr_gamma",
            "adam_lr_step_frac",
            "lbfgs_steps",
            "lbfgs_lr",
            "lbfgs_history",
            "lbfgs_line_search_evals",
        },
        "optim",
    )
    return OptimConfig(
        adam_steps=_req_int(data, "adam_steps", minimum=0),
        adam_lr=_req_float(data, "adam_lr", positive=True),
        adam_lr_gamma=_req_float(data, "adam_lr_gamma", positive=True),
        adam_lr_step_frac=_req_float(data, "adam_lr_step_frac", positive=True),
        lbfgs_steps=_req_int(data, "lbfgs_steps", minimum=0),
        lbfgs_lr=_req_float(data, "lbfgs_lr", positive=True),
        lbfgs_history=_req_int(data, "lbfgs_history", minimum=1),
        lbfgs_line_search_evals=_req_int(data, "lbfgs_line_search_evals", minimum=1),
    )


def _validate_consistency(
    constraints: ConstraintMode,
    constraint_form: ConstraintForm,
    problem: ProblemConfig,
    coefficients: CoefficientConfig,
    network: NetworkConfig,
    loss: LossConfig,
    sampling: SamplingConfig,
    optim: OptimConfig,
) -> None:
    if coefficients.wave_speed.form == "constant" and problem.wave_speed <= 0.0:
        raise ConfigError("problem.wave_speed must be positive")
    if not 0.0 < optim.adam_lr_step_frac <= 1.0:
        raise ConfigError("optim.adam_lr_step_frac must lie in (0, 1]")
    if not 0.0 < loss.balance_alpha <= 1.0:
        raise ConfigError("loss.balance_alpha must lie in (0, 1]")
    if loss.balance_every < 1:
        raise ConfigError("loss.balance_every must be >= 1")
    for name in ("pde", "bc", "ic_u", "ic_ut"):
        if name not in loss.weights:
            raise ConfigError(f"loss.weights must include {name}")
    if problem.boundary != "dirichlet":
        raise ConfigError(
            "only Dirichlet targets are implemented; Neumann and absorbing "
            "conditions are selected in AcousticWave._boundary_residual"
        )
    if constraints == "soft" and constraint_form != "none":
        raise ConfigError("constraint_form must be 'none' when constraints is soft")
    if constraints == "hard" and constraint_form == "none":
        raise ConfigError("constraints hard requires constraint_form modal_sine or polynomial")
    if constraint_form == "modal_sine" and problem.ic != "sine":
        raise ConfigError("modal_sine hard constraint requires problem.ic = sine")
    if constraints == "hard" and network.field_names != ("u",):
        raise ConfigError("hard constraints in this increment require network.field_names: [u]")
    if constraints == "hard" and (
        problem.boundary != "dirichlet"
        or problem.dirichlet_value != 0.0
        or problem.initial_velocity != 0.0
    ):
        raise ConfigError(
            "the hard trial function enforces homogeneous Dirichlet data and "
            "zero initial velocity; use constraints: soft for other targets"
        )
    if problem.spatial_dim == 1 and problem.depth is not None:
        raise ConfigError("problem.depth is the 2D vertical extent; omit it in 1D")
    if problem.spatial_dim == 1 and (problem.mode_x != 1 or problem.mode_z != 1):
        raise ConfigError("mode_x and mode_z apply when spatial_dim is 2")
    if problem.spatial_dim == 1 and (
        problem.gaussian_center_z is not None or problem.gaussian_width_z is not None
    ):
        raise ConfigError("gaussian_center_z and gaussian_width_z apply when spatial_dim is 2")
    if problem.mode_x < 1 or problem.mode_z < 1:
        raise ConfigError("mode_x and mode_z must be >= 1")
    if coefficients.wave_speed.form == "layered" and problem.spatial_dim != 2:
        raise ConfigError("layered wave speed is a c(z) profile; set problem.spatial_dim: 2")
    if loss.causal_epsilon <= 0.0:
        raise ConfigError("loss.causal_epsilon must be positive")
    if loss.causal_bins < 2:
        raise ConfigError("loss.causal_bins must be >= 2")
    if sampling.method == "sobol":
        for name, count in (
            ("n_interior", sampling.n_interior),
            ("n_boundary_per_side", sampling.n_boundary_per_side),
            ("n_initial", sampling.n_initial),
        ):
            if count & (count - 1) != 0:
                raise ConfigError(f"sampling.{name} must be a power of two when method is sobol")


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return cast(dict[str, Any], value)


def _check_keys(
    data: dict[str, Any],
    allowed: set[str],
    where: str,
    *,
    optional: set[str] | None = None,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"unknown {where} keys: {names}")
    required = allowed if optional is None else allowed - optional
    missing = sorted(required - set(data))
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"missing {where} keys: {names}")


def _req_str(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_choice(
    data: dict[str, Any],
    key: str,
    options: tuple[str, ...],
    *,
    default: str,
) -> str:
    if key not in data or data[key] is None:
        return default
    return _req_choice(data, key, options)


def _optional_float(data: dict[str, Any], key: str, *, default: float) -> float:
    if key not in data or data[key] is None:
        return default
    return _req_float(data, key)


def _optional_float_or_none(
    data: dict[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> float | None:
    if key not in data or data[key] is None:
        return None
    return _req_float(data, key, positive=positive)


def _optional_bool(data: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _optional_int(data: dict[str, Any], key: str, *, default: int) -> int:
    if key not in data or data[key] is None:
        return default
    return _req_int(data, key, minimum=0)


def _req_int(data: dict[str, Any], key: str, *, minimum: int) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}")
    return value


def _req_float(data: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    number = float(value)
    if positive and number <= 0.0:
        raise ConfigError(f"{key} must be positive")
    return number


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    return _req_str(data, key)


def _req_str_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{key} must be a non-empty list of strings")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise ConfigError(f"{key} must be a non-empty list of strings")
        names.append(item)
    if len(set(names)) != len(names):
        raise ConfigError(f"{key} contains a duplicate name")
    return tuple(names)


def _req_weights(data: dict[str, Any], key: str) -> dict[str, float]:
    value = data[key]
    if not isinstance(value, dict) or not value:
        raise ConfigError(f"{key} must be a non-empty mapping of names to weights")
    weights: dict[str, float] = {}
    for name, raw in cast(dict[object, object], value).items():
        if not isinstance(name, str) or name == "":
            raise ConfigError(f"{key} keys must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ConfigError(f"{key}.{name} must be a number")
        number = float(raw)
        if number < 0.0:
            raise ConfigError(f"{key}.{name} must be non-negative")
        weights[name] = number
    return weights


def _req_choice(data: dict[str, Any], key: str, options: tuple[str, ...]) -> str:
    value = data[key]
    if not isinstance(value, str) or value not in options:
        choices = ", ".join(options)
        raise ConfigError(f"{key} must be one of: {choices}")
    return value
