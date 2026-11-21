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

from geopinn.ansatz import HardConstraintWave1D, HardConstraintWave2D, UnconstrainedField
from geopinn.balance import wang2021_weights
from geopinn.config import TrainConfig, load_config
from geopinn.initial import displacement_torch, gaussian_2d_torch, sine_2d_torch
from geopinn.losses import LossTerms, pinn_loss
from geopinn.nets.mlp import MLP
from geopinn.nondim import Scales, warn_if_poorly_scaled
from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.boundary import BoundaryCondition, BoundaryKind
from geopinn.physics.coefficients import resolve_wave_speed
from geopinn.sampling import PointBatch, sample_boundary, sample_initial, sample_interior
from geopinn.seed import set_seed

DTYPE = torch.float64


class HistoryRow(TypedDict):
    step: int
    stage: str
    loss: float
    pde: float
    bc: float | None
    ic_u: float | None
    ic_ut: float | None


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
    loss_weights: dict[str, float]


def scales_from_config(config: TrainConfig) -> Scales:
    problem = config.problem
    speed = problem.wave_speed
    spec = config.coefficients.wave_speed
    if spec.form == "layered" and spec.c_top is not None and spec.c_bottom is not None:
        # The O(1) check follows the fastest layer. The residual uses the profile.
        speed = max(spec.c_top, spec.c_bottom)
    depth = None
    if problem.spatial_dim == 2:
        depth = problem.length if problem.depth is None else problem.depth
    return Scales(
        length=problem.length,
        time=problem.time_final,
        amplitude=problem.amplitude,
        wave_speed=speed,
        depth=depth,
    )


def build_trial(config: TrainConfig) -> nn.Module:
    """Build the coordinate network and, when requested, the hard constraint."""
    network = config.network
    spatial_dim = config.problem.spatial_dim
    net = MLP(
        in_features=spatial_dim + 1,
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
    if config.constraints == "soft":
        return UnconstrainedField(net)
    if spatial_dim == 1:
        if config.constraint_form == "modal_sine":
            return HardConstraintWave1D(net, form="modal_sine")
        return HardConstraintWave1D(
            net,
            form="polynomial",
            initial_displacement=hat_displacement(config),
        )
    z_extent = scales_from_config(config).z_extent
    if config.constraint_form == "modal_sine":
        return HardConstraintWave2D(
            net,
            form="modal_sine",
            z_extent=z_extent,
            mode_x=config.problem.mode_x,
            mode_z=config.problem.mode_z,
        )
    return HardConstraintWave2D(
        net,
        form="polynomial",
        z_extent=z_extent,
        initial_displacement=hat_displacement(config),
    )


def hat_displacement(config: TrainConfig) -> Callable[[Tensor], Tensor]:
    """Initial displacement in nondimensional variables.

    Physical ``u0`` is divided by the amplitude scale ``U``. In 1D the
    coordinate is ``x' = x / L`` on ``[0, 1]``, and the sine problem is
    ``sin(π x')``. In 2D the columns are ``(x / Lx, z / Lx)``.
    """
    problem = config.problem
    if problem.spatial_dim == 1:
        if problem.ic == "sine":
            return displacement_torch("sine", amplitude=1.0, mode=1, length=1.0)
        return displacement_torch(
            "gaussian",
            amplitude=1.0,
            length=1.0,
            center=problem.gaussian_center / problem.length,
            width=problem.gaussian_width / problem.length,
        )
    depth = problem.length if problem.depth is None else problem.depth
    z_extent = depth / problem.length
    if problem.ic == "sine":
        return sine_2d_torch(
            amplitude=1.0,
            mode_x=problem.mode_x,
            mode_z=problem.mode_z,
            length_x=1.0,
            length_z=z_extent,
        )
    if problem.gaussian_center_z is None:
        center_z = problem.gaussian_center
    else:
        center_z = problem.gaussian_center_z
    if problem.gaussian_width_z is None:
        width_z = problem.gaussian_width
    else:
        width_z = problem.gaussian_width_z
    return gaussian_2d_torch(
        amplitude=1.0,
        center_x=problem.gaussian_center / problem.length,
        center_z=center_z / problem.length,
        width_x=problem.gaussian_width / problem.length,
        width_z=width_z / problem.length,
        length_x=1.0,
        length_z=z_extent,
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
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the trained field at physical coordinates.

    In 1D, ``x`` and ``t`` broadcast. In 2D, pass ``z`` as well. The network
    is queried at ``x' = x / Lx`` and, in 2D, ``z' = z / Lx`` (the same
    reference length that keeps ``c_hat²`` in front of the Laplacian).
    """
    parameter = next(model.parameters())
    if z is None:
        x_arr = np.asarray(x, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        x_b, t_b = np.broadcast_arrays(x_arr, t_arr)
        shape = x_b.shape
        x_hat = torch.tensor(
            np.ascontiguousarray(x_b).reshape(-1, 1) / scales.length,
            dtype=parameter.dtype,
            device=parameter.device,
        )
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        x_b, z_b, t_b = np.broadcast_arrays(x_arr, z_arr, t_arr)
        shape = x_b.shape
        coords = np.stack(
            (
                np.ascontiguousarray(x_b).reshape(-1) / scales.length,
                np.ascontiguousarray(z_b).reshape(-1) / scales.length,
            ),
            axis=1,
        )
        x_hat = torch.tensor(coords, dtype=parameter.dtype, device=parameter.device)
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
        wave_speed=resolve_wave_speed(
            config.coefficients.wave_speed,
            scales.c_hat,
            length=scales.length,
            time=scales.time,
        ),
        boundary=BoundaryCondition(
            BoundaryKind(config.problem.boundary),
            config.problem.dirichlet_value,
        ),
        initial_displacement=hat_displacement(config),
        initial_velocity=config.problem.initial_velocity,
    )
    interior, boundary, initial = _collocation(config, device)
    active_weights = _weights(config)
    running_weights: dict[str, float] = dict(active_weights)
    history: list[HistoryRow] = []
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None

    def objective(*, monitor: bool, causal: bool = False) -> LossTerms:
        # Causal weights are an Adam device. L-BFGS minimizes the plain mean
        # square, which is the residual the checkpoint is scored on.
        epsilon = config.loss.causal_epsilon if causal and config.loss.causal else None
        return pinn_loss(
            pde,
            model,
            interior,
            boundary,
            initial,
            active_weights,
            monitor_constraints=monitor,
            field_names=config.network.field_names,
            causal_epsilon=epsilon,
            causal_bins=config.loss.causal_bins,
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
        _run_adam(
            config,
            model,
            objective,
            consider,
            history,
            verbose,
            active_weights,
            running_weights,
        )
        # Score the weights produced by the last Adam update, not only the
        # pre-update state recorded inside the loop. Pass causal=True so this
        # score stays on the same scale as the Adam objective. When causal
        # training is off the flag does nothing.
        consider(objective(monitor=False, causal=True))
    adam_state = best_state
    if config.loss.causal and config.optim.lbfgs_steps > 0 and adam_state is not None:
        # The causal objective is not the plain residual, so L-BFGS competes
        # with itself. The two candidates are compared on the plain loss below.
        best_loss = math.inf
        best_state = None
    if config.optim.lbfgs_steps > 0:
        _run_lbfgs(config, model, objective, consider, history, verbose)
    if config.loss.causal and config.optim.lbfgs_steps > 0 and adam_state is not None:
        best_state = _prefer_plain_loss(model, objective, adam_state, best_state)

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
        loss_weights=dict(active_weights),
    )


def _run_adam(
    config: TrainConfig,
    model: nn.Module,
    objective: Callable[..., LossTerms],
    consider: Callable[[LossTerms], float],
    history: list[HistoryRow],
    verbose: bool,
    active_weights: dict[str, float],
    running_weights: dict[str, float],
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
        terms = objective(monitor=monitor, causal=True)
        consider(terms)
        if config.loss.balance == "wang2021" and step % config.loss.balance_every == 0:
            # Update multipliers for the next step. This step's total loss
            # already used the previous multipliers.
            annealed = wang2021_weights(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                terms.terms,
                _weights(config),
                running_weights,
                alpha=config.loss.balance_alpha,
            )
            active_weights.clear()
            active_weights.update(annealed)
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


def _training_extents(config: TrainConfig) -> tuple[float, ...]:
    """Spatial side lengths of the nondimensional training box.

    ``x' = x / Lx`` lives in ``[0, 1]``. In 2D, ``z' = z / Lx`` lives in
    ``[0, Lz / Lx]``. One reference length keeps the residual
    ``u_tt - c_hat(x, z)² (u_xx + u_zz)``.
    """
    if config.problem.spatial_dim == 1:
        return (1.0,)
    depth = config.problem.length if config.problem.depth is None else config.problem.depth
    return (1.0, depth / config.problem.length)


def _collocation(
    config: TrainConfig,
    device: torch.device,
) -> tuple[PointBatch, PointBatch, PointBatch]:
    # Sampling is done in nondimensional coordinates. The physical box
    # [0, L] x [0, T] is the unit interval after x' = x/L, t' = t/T.
    # A 2D rectangle uses the extents from _training_extents.
    sampling = config.sampling
    extents = _training_extents(config)
    interior = sample_interior(
        sampling.n_interior,
        spatial_dim=len(extents),
        x_max=extents,
        t_max=1.0,
        method=sampling.method,
        seed=config.seed,
        dtype=DTYPE,
        device=device,
    )
    boundary = sample_boundary(
        sampling.n_boundary_per_side,
        lengths=extents,
        t_max=1.0,
        method=sampling.method,
        seed=config.seed + 1,
        dtype=DTYPE,
        device=device,
    )
    initial = sample_initial(
        sampling.n_initial,
        spatial_dim=len(extents),
        length=extents,
        method=sampling.method,
        seed=config.seed + 2,
        dtype=DTYPE,
        device=device,
    )
    return interior, boundary, initial


def _prefer_plain_loss(
    model: nn.Module,
    objective: Callable[..., LossTerms],
    adam_state: dict[str, Tensor],
    lbfgs_state: dict[str, Tensor] | None,
) -> dict[str, Tensor]:
    """Keep the causal-Adam or plain-LBFGS weights with the smaller plain loss."""

    def score(state: dict[str, Tensor]) -> float:
        model.load_state_dict(state)
        return float(objective(monitor=False, causal=False).total.detach())

    adam_loss = score(adam_state)
    if lbfgs_state is None:
        return adam_state
    lbfgs_loss = score(lbfgs_state)
    if lbfgs_loss <= adam_loss:
        return lbfgs_state
    return adam_state


def _weights(config: TrainConfig) -> dict[str, float]:
    weights = dict(config.loss.weights)
    if config.constraints == "hard":
        # The trial function already satisfies these, so they must not compete
        # with the PDE residual. They are still evaluated on log steps.
        for name in ("bc", "ic_u", "ic_ut"):
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

    pde = terms.get("pde")
    return HistoryRow(
        step=step,
        stage=stage,
        loss=float(terms.total.detach()),
        pde=0.0 if pde is None else float(pde.detach()),
        bc=_piece("bc"),
        ic_u=_piece("ic_u"),
        ic_ut=_piece("ic_ut"),
    )


def _print_row(verbose: bool, row: HistoryRow) -> None:
    if not verbose:
        return
    bc = row["bc"]
    ic_u = row["ic_u"]
    ic_ut = row["ic_ut"]
    bc_text = "n/a" if bc is None else f"{bc:.3e}"
    ic_text = "n/a" if ic_u is None else f"{ic_u:.3e}"
    velocity_text = "n/a" if ic_ut is None else f"{ic_ut:.3e}"
    print(
        f"{row['stage']:5s} step {row['step']:5d}  loss {row['loss']:.3e}  "
        f"pde {row['pde']:.3e}  bc {bc_text}  ic_u {ic_text}  ic_ut {velocity_text}",
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
