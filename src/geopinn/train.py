"""Adam followed by L-BFGS, with seeding, nondimensionalization, and checkpoints."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from torch import Tensor, nn

from geopinn.ansatz import HardConstraintWave1D, UnconstrainedField
from geopinn.config import TrainConfig, load_config
from geopinn.initial import displacement_torch
from geopinn.losses import LossTerms, pinn_loss
from geopinn.nets.mlp import MLP
from geopinn.nondim import Scales, warn_if_poorly_scaled
from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.coefficients import resolve_wave_speed
from geopinn.sampling import PointBatch, sample_boundary_1d, sample_initial, sample_interior
from geopinn.seed import set_seed

DTYPE = torch.float64


class HistoryRow(TypedDict):
    step: int
    stage: str
    loss: float
    residual: float
    bc: float | None
    ic: float | None
    ic_velocity: float | None


class TrainingError(RuntimeError):
    """The optimizer produced a non-finite loss or could not be checkpointed."""


@dataclass
class TrainResult:
    """Trained trial function and the record of the optimization."""

    model: nn.Module
    scales: Scales
    history: list[HistoryRow]
    config: TrainConfig
    seconds: float
    checkpoint_path: Path | None


def scales_from_config(config: TrainConfig) -> Scales:
    problem = config.problem
    return Scales(
        length=problem.length,
        time=problem.time_final,
        amplitude=problem.amplitude,
        wave_speed=problem.wave_speed,
    )


def build_trial(config: TrainConfig) -> nn.Module:
    """Build the coordinate network and, when requested, the hard constraint."""
    network = config.network
    net = MLP(
        in_features=2,
        out_features=len(network.field_names),
        hidden_layers=network.hidden_layers,
        hidden_features=network.hidden_dim,
        activation=network.activation,
        sine_omega=network.sine_omega,
        fourier_features=network.fourier_features,
        fourier_sigma=network.fourier_sigma,
        map_unit_interval=network.fourier_features == 0,
        seed=config.seed,
    )
    if not config.loss.hard_constraints:
        return UnconstrainedField(net)
    if config.loss.constraint_form == "modal_sine":
        return HardConstraintWave1D(net, form="modal_sine")
    return HardConstraintWave1D(
        net,
        form="polynomial",
        initial_displacement=hat_displacement(config),
    )


def hat_displacement(config: TrainConfig) -> Callable[[Tensor], Tensor]:
    """Initial displacement in nondimensional variables, where the domain is ``[0, 1]``.

    Physical ``u0`` is divided by the amplitude scale ``U``, and ``x`` is
    divided by ``L``. For the sine problem this is exactly ``sin(π x')``.
    """
    problem = config.problem
    if problem.ic == "sine":
        return displacement_torch("sine", amplitude=1.0, mode=1, length=1.0)
    return displacement_torch(
        "gaussian",
        amplitude=1.0,
        length=1.0,
        center=problem.gaussian_center / problem.length,
        width=problem.gaussian_width / problem.length,
    )


def train(config: TrainConfig, *, verbose: bool = True) -> TrainResult:
    """Train on the nondimensional acoustic IBVP described by ``config``.

    Collocation is fixed for the whole run (required for a meaningful L-BFGS
    step on the strong-form residual). When hard constraints are on, the
    boundary and initial weights are set to zero because those conditions are
    satisfied by the trial function; they are still evaluated on log steps.
    """
    started = time.perf_counter()
    set_seed(config.seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        result = _train_with_dtype(config, verbose=verbose)
    finally:
        torch.set_default_dtype(previous_dtype)
    result.seconds = time.perf_counter() - started
    return result


def predict_physical(
    model: nn.Module,
    scales: Scales,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate the trained field at physical coordinates. ``x`` and ``t`` broadcast."""
    x_arr = np.asarray(x, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    x_b, t_b = np.broadcast_arrays(x_arr, t_arr)
    shape = x_b.shape
    parameter = next(model.parameters())
    x_hat = torch.tensor(
        np.ascontiguousarray(x_b).reshape(-1, 1) / scales.length,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    t_hat = torch.tensor(
        np.ascontiguousarray(t_b).reshape(-1, 1) / scales.time,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        u_hat = model(x_hat, t_hat)
    if was_training:
        model.train()
    values = cast(np.ndarray, (scales.amplitude * u_hat).detach().cpu().numpy())
    return values.reshape(shape)


def load_trained_model(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, TrainConfig, Scales]:
    """Rebuild a trial function from a checkpoint written by :func:`train`."""
    try:
        raw = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise TrainingError(f"{path} is not a geopinn checkpoint")
    blob = cast(dict[str, object], raw)
    config_obj = blob.get("config")
    state = blob.get("model")
    if not isinstance(config_obj, dict) or not isinstance(state, dict):
        raise TrainingError(f"{path} is missing config or model weights")
    config = TrainConfig.from_dict(cast(dict[str, object], config_obj))
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        model = build_trial(config).to(device)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(cast(dict[str, Tensor], state))
    model.eval()
    return model, config, scales_from_config(config)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``geopinn-train --config configs/wave1d.yaml``."""
    parser = argparse.ArgumentParser(description="Train a geopinn model from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.checkpoint is not None:
        config = replace(config, checkpoint=str(args.checkpoint))
    result = train(config, verbose=True)
    final = result.history[-1]["loss"] if result.history else float("nan")
    print(
        f"training finished in {result.seconds:.1f}s  final_loss={final:.3e}  "
        f"checkpoint={result.checkpoint_path}",
        flush=True,
    )


def _train_with_dtype(config: TrainConfig, *, verbose: bool) -> TrainResult:
    device = torch.device(config.device)
    scales = scales_from_config(config)
    warn_if_poorly_scaled(scales)
    model = build_trial(config).to(device)
    pde = AcousticWave(
        wave_speed=resolve_wave_speed(config.coefficients.wave_speed, scales.c_hat),
        initial_displacement=hat_displacement(config),
        initial_velocity=lambda x: torch.zeros_like(x),
    )
    interior, boundary, initial = _collocation(config, device)
    weights = _weights(config)
    history: list[HistoryRow] = []
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None

    def objective(*, monitor: bool) -> LossTerms:
        return pinn_loss(
            pde,
            model,
            interior,
            boundary,
            initial,
            weights,
            monitor_constraints=monitor,
            field_names=config.network.field_names,
        )

    def consider(terms: LossTerms) -> float:
        nonlocal best_loss, best_state
        value = float(terms.total.detach())
        if not math.isfinite(value):
            raise TrainingError("non-finite loss")
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone() for key, tensor in model.state_dict().items()
            }
        return value

    if config.optim.adam_steps > 0:
        _run_adam(config, model, objective, consider, history, verbose)
        # Score the weights produced by the last Adam update, not only the
        # pre-update state recorded inside the loop.
        consider(objective(monitor=False))
    if config.optim.lbfgs_steps > 0:
        _run_lbfgs(config, model, objective, consider, history, verbose)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = _save_checkpoint(config, model, history)
    return TrainResult(
        model=model,
        scales=scales,
        history=history,
        config=config,
        seconds=0.0,
        checkpoint_path=checkpoint_path,
    )


def _run_adam(
    config: TrainConfig,
    model: nn.Module,
    objective: Callable[..., LossTerms],
    consider: Callable[[LossTerms], float],
    history: list[HistoryRow],
    verbose: bool,
) -> None:
    optim = config.optim
    optimizer = torch.optim.Adam(model.parameters(), lr=optim.adam_lr)
    step_size = max(1, round(optim.adam_steps * optim.adam_lr_step_frac))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=optim.adam_lr_gamma,
    )
    model.train()
    for step in range(1, optim.adam_steps + 1):
        monitor = step % config.log_every == 0 or step == optim.adam_steps
        optimizer.zero_grad(set_to_none=True)
        terms = objective(monitor=monitor)
        consider(terms)
        _backward(terms.total)
        optimizer.step()
        scheduler.step()
        row = _row("adam", step, terms, monitored=monitor)
        history.append(row)
        if monitor:
            _print_row(verbose, row)


def _run_lbfgs(
    config: TrainConfig,
    model: nn.Module,
    objective: Callable[..., LossTerms],
    consider: Callable[[LossTerms], float],
    history: list[HistoryRow],
    verbose: bool,
) -> None:
    optim_cfg = config.optim
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=optim_cfg.lbfgs_lr,
        max_iter=1,
        max_eval=optim_cfg.lbfgs_line_search_evals,
        history_size=optim_cfg.lbfgs_history,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        line_search_fn="strong_wolfe",
    )
    model.train()
    for step in range(1, optim_cfg.lbfgs_steps + 1):

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            terms = objective(monitor=False)
            if not torch.isfinite(terms.total):
                raise TrainingError("non-finite loss during L-BFGS")
            _backward(terms.total)
            return terms.total

        try:
            # LBFGS.step is untyped in the torch 2.15 stubs.
            optimizer.step(closure)  # type: ignore[no-untyped-call]
        except RuntimeError as exc:
            if "line search" not in str(exc).lower():
                raise
            if verbose:
                print(f"lbfgs line search stopped at step {step}: {exc}", flush=True)
            break
        monitor = step % config.log_every == 0 or step == optim_cfg.lbfgs_steps
        terms = objective(monitor=monitor)
        consider(terms)
        row = _row("lbfgs", step, terms, monitored=monitor)
        history.append(row)
        if monitor:
            _print_row(verbose, row)


def _collocation(
    config: TrainConfig,
    device: torch.device,
) -> tuple[PointBatch, PointBatch, PointBatch]:
    # Sampling is done in nondimensional coordinates. The physical box
    # [0, L] x [0, T] is the unit square after x' = x/L, t' = t/T.
    sampling = config.sampling
    interior = sample_interior(
        sampling.n_interior,
        spatial_dim=1,
        x_max=1.0,
        t_max=1.0,
        method=sampling.method,
        seed=config.seed,
        dtype=DTYPE,
        device=device,
    )
    boundary = sample_boundary_1d(
        sampling.n_boundary_per_side,
        length=1.0,
        t_max=1.0,
        method=sampling.method,
        seed=config.seed + 1,
        dtype=DTYPE,
        device=device,
    )
    initial = sample_initial(
        sampling.n_initial,
        spatial_dim=1,
        length=1.0,
        method=sampling.method,
        seed=config.seed + 2,
        dtype=DTYPE,
        device=device,
    )
    return interior, boundary, initial


def _weights(config: TrainConfig) -> dict[str, float]:
    weights = dict(config.loss.weights)
    if config.loss.hard_constraints:
        # The trial function already satisfies these, so they must not compete
        # with the PDE residual. They are still evaluated on log steps.
        for name in ("bc", "ic", "ic_velocity"):
            weights[name] = 0.0
    return weights


def _backward(loss: Tensor) -> None:
    # Tensor.backward is untyped in the torch 2.15 stubs.
    loss.backward()  # type: ignore[no-untyped-call]


def _row(stage: str, step: int, terms: LossTerms, *, monitored: bool) -> HistoryRow:
    def _piece(name: str) -> float | None:
        value = terms.get(name)
        if not monitored or value is None:
            return None
        return float(value.detach())

    residual = terms.get("residual")
    return HistoryRow(
        step=step,
        stage=stage,
        loss=float(terms.total.detach()),
        residual=0.0 if residual is None else float(residual.detach()),
        bc=_piece("bc"),
        ic=_piece("ic"),
        ic_velocity=_piece("ic_velocity"),
    )


def _print_row(verbose: bool, row: HistoryRow) -> None:
    if not verbose:
        return
    bc = row["bc"]
    ic = row["ic"]
    velocity = row["ic_velocity"]
    bc_text = "n/a" if bc is None else f"{bc:.3e}"
    ic_text = "n/a" if ic is None else f"{ic:.3e}"
    velocity_text = "n/a" if velocity is None else f"{velocity:.3e}"
    print(
        f"{row['stage']:5s} step {row['step']:5d}  loss {row['loss']:.3e}  "
        f"residual {row['residual']:.3e}  bc {bc_text}  ic {ic_text}  ic_t {velocity_text}",
        flush=True,
    )


def _save_checkpoint(
    config: TrainConfig,
    model: nn.Module,
    history: list[HistoryRow],
) -> Path | None:
    if config.checkpoint == "":
        return None
    path = Path(config.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "history": history,
            "seed": config.seed,
        },
        path,
    )
    return path


if __name__ == "__main__":
    main()
