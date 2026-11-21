"""Train hard and soft 1D acoustic PINNs and compare them with d'Alembert.

Run from the repository root:

    python benchmarks/wave1d_dalembert.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from geopinn.config import TrainConfig, load_config
from geopinn.initial import displacement_torch
from geopinn.physics.differentiate import partial
from geopinn.reference.analytic import standing_wave_1d
from geopinn.reference.fdm import solve_wave_1d
from geopinn.train import TrainResult, predict_physical, train

FDM_INTERVALS = 400
FDM_CFL = 0.5
EVAL_NX = 257
EVAL_NT = 129


@dataclass
class ModeEval:
    """One trained mode on the shared evaluation grid."""

    name: str
    result: TrainResult
    exact: np.ndarray
    pinn: np.ndarray
    metrics: dict[str, float]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-config", type=Path, default=Path("configs/wave1d.yaml"))
    parser.add_argument("--soft-config", type=Path, default=Path("configs/wave1d_soft.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wave1d"))
    parser.add_argument("--modes", default="hard,soft", help="Comma-separated: hard, soft")
    parser.add_argument("--adam-steps", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/wave1d_comparison.png"),
    )
    args = parser.parse_args(argv)
    selected = {item.strip() for item in args.modes.split(",") if item.strip()}
    args.output.mkdir(parents=True, exist_ok=True)

    modes: list[ModeEval] = []
    if "hard" in selected:
        modes.append(
            _run_mode(
                "hard",
                args.hard_config,
                args.output / "hard",
                args.adam_steps,
                args.lbfgs_steps,
            )
        )
    if "soft" in selected:
        modes.append(
            _run_mode(
                "soft",
                args.soft_config,
                args.output / "soft",
                args.adam_steps,
                args.lbfgs_steps,
            )
        )
    if not modes:
        raise SystemExit("select at least one of hard, soft")

    payload = {mode.name: mode.metrics for mode in modes}
    text = json.dumps(payload, indent=2) + "\n"
    (args.output / "metrics.json").write_text(text, encoding="utf-8")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.with_name("wave1d_metrics.json").write_text(text, encoding="utf-8")
    if {mode.name for mode in modes} == {"hard", "soft"}:
        hard = next(mode for mode in modes if mode.name == "hard")
        soft = next(mode for mode in modes if mode.name == "soft")
        _save_comparison(hard, soft, args.output / "wave1d_comparison.png", args.artifact)
    print(text, flush=True)


def _run_mode(
    name: str,
    config_path: Path,
    output: Path,
    adam_steps: int | None,
    lbfgs_steps: int | None,
) -> ModeEval:
    output.mkdir(parents=True, exist_ok=True)
    config = _with_steps(load_config(config_path), adam_steps, lbfgs_steps)
    config = replace(config, checkpoint=str(output / "checkpoint.pt"))
    print(f"=== {name} constraints ({config.constraints}) ===", flush=True)
    result = train(config, verbose=True)
    return _evaluate(name, result)


def _with_steps(
    config: TrainConfig,
    adam_steps: int | None,
    lbfgs_steps: int | None,
) -> TrainConfig:
    optim = config.optim
    if adam_steps is not None:
        optim = replace(optim, adam_steps=adam_steps)
    if lbfgs_steps is not None:
        optim = replace(optim, lbfgs_steps=lbfgs_steps)
    return replace(config, optim=optim)


def _evaluate(name: str, result: TrainResult) -> ModeEval:
    problem = result.config.problem
    if problem.ic != "sine":
        raise ValueError("this benchmark verifies the sine standing wave; set problem.ic to sine")
    x = np.linspace(0.0, problem.length, EVAL_NX)
    t = np.linspace(0.0, problem.time_final, EVAL_NT)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    exact = standing_wave_1d(
        xx,
        tt,
        wave_speed=problem.wave_speed,
        length=problem.length,
        amplitude=problem.amplitude,
    )
    pinn = predict_physical(result.model, result.scales, xx, tt)
    rel_analytic = _relative_l2(pinn - exact, exact)
    rel_final = _relative_l2(pinn[:, -1] - exact[:, -1], exact[:, -1])

    dx = problem.length / FDM_INTERVALS
    x_fdm = np.linspace(0.0, problem.length, FDM_INTERVALS + 1)
    u0 = problem.amplitude * np.sin(math.pi * x_fdm / problem.length)
    fdm = solve_wave_1d(
        u0,
        wave_speed=problem.wave_speed,
        dx=dx,
        t_final=problem.time_final,
        cfl=FDM_CFL,
    )
    fdm_exact = standing_wave_1d(
        fdm.x[None, :],
        fdm.t[:, None],
        wave_speed=problem.wave_speed,
        length=problem.length,
        amplitude=problem.amplitude,
    )
    pinn_on_fdm = predict_physical(result.model, result.scales, fdm.x[None, :], fdm.t[:, None])
    constraints = _constraint_errors(result)
    n_params = sum(parameter.numel() for parameter in result.model.parameters())
    metrics = {
        "relative_l2_analytic": rel_analytic,
        "relative_l2_analytic_final_time": rel_final,
        "max_abs_analytic": float(np.max(np.abs(pinn - exact))),
        "relative_l2_fdm": _relative_l2(pinn_on_fdm - fdm.u, fdm.u),
        "fdm_relative_l2_analytic": _relative_l2(fdm.u - fdm_exact, fdm_exact),
        "bc_max_abs": constraints["bc_max_abs"],
        "ic_u_max_abs": constraints["ic_u_max_abs"],
        "ic_ut_max_abs": constraints["ic_ut_max_abs"],
        "c_hat": result.scales.c_hat,
        "n_parameters": float(n_params),
        "seconds": result.seconds,
        "final_loss": result.history[-1]["loss"] if result.history else float("nan"),
        "balance": 1.0 if result.config.loss.balance == "wang2021" else 0.0,
        "w_bc": result.loss_weights.get("bc", 0.0),
        "w_ic_u": result.loss_weights.get("ic_u", 0.0),
        "w_ic_ut": result.loss_weights.get("ic_ut", 0.0),
    }
    return ModeEval(
        name=name,
        result=result,
        exact=exact.T,
        pinn=pinn.T,
        metrics=metrics,
    )


def _constraint_errors(result: TrainResult) -> dict[str, float]:
    """Max |boundary|, |u - u0| and |u_t| on the nondimensional initial line."""
    model = result.model
    parameter = next(model.parameters())
    n = 501
    x = torch.linspace(0.0, 1.0, n, dtype=parameter.dtype, device=parameter.device).reshape(-1, 1)
    t_line = torch.linspace(0.0, 1.0, n, dtype=parameter.dtype, device=parameter.device).reshape(
        -1, 1
    )
    displacement = displacement_torch("sine", amplitude=1.0, length=1.0)
    model.eval()
    with torch.no_grad():
        bc_left = model(torch.zeros_like(t_line), t_line).abs().max().item()
        bc_right = model(torch.ones_like(t_line), t_line).abs().max().item()
        ic_u = (model(x, torch.zeros_like(x)) - displacement(x)).abs().max().item()
    x_leaf = x.detach().clone().requires_grad_(True)
    t_leaf = torch.zeros_like(x_leaf).requires_grad_(True)
    velocity = partial(model(x_leaf, t_leaf), t_leaf)
    return {
        "bc_max_abs": max(float(bc_left), float(bc_right)),
        "ic_u_max_abs": float(ic_u),
        "ic_ut_max_abs": float(velocity.abs().max().detach()),
    }


def _save_comparison(hard: ModeEval, soft: ModeEval, output: Path, artifact: Path) -> None:
    problem = hard.result.config.problem
    x = np.linspace(0.0, problem.length, EVAL_NX)
    t = np.linspace(0.0, problem.time_final, EVAL_NT)
    extent = (float(x[0]), float(x[-1]), float(t[0]), float(t[-1]))
    vmin = float(min(np.min(hard.exact), np.min(soft.exact)))
    vmax = float(max(np.max(hard.exact), np.max(soft.exact)))
    hard_error = hard.pinn - hard.exact
    soft_error = soft.pinn - soft.exact
    err_max = float(max(np.max(np.abs(hard_error)), np.max(np.abs(soft_error))))

    fig = plt.figure(figsize=(12.4, 9.6))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.05], hspace=0.42, wspace=0.28)
    ax_exact = fig.add_subplot(grid[0, 0])
    ax_hard = fig.add_subplot(grid[0, 1])
    ax_soft = fig.add_subplot(grid[0, 2])
    ax_hard_err = fig.add_subplot(grid[1, 0])
    ax_soft_err = fig.add_subplot(grid[1, 1])
    ax_loss = fig.add_subplot(grid[1, 2])
    ax_slice = fig.add_subplot(grid[2, :])

    image = _show(ax_exact, hard.exact, extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    _show(ax_hard, hard.pinn, extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    _show(ax_soft, soft.pinn, extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    _show(ax_hard_err, hard_error, extent=extent, vmin=-err_max, vmax=err_max, cmap="coolwarm")
    err_image = _show(
        ax_soft_err,
        soft_error,
        extent=extent,
        vmin=-err_max,
        vmax=err_max,
        cmap="coolwarm",
    )
    for axis, title in (
        (ax_exact, "Analytic"),
        (ax_hard, f"Hard  L2 {hard.metrics['relative_l2_analytic']:.2e}"),
        (ax_soft, f"Soft  L2 {soft.metrics['relative_l2_analytic']:.2e}"),
        (ax_hard_err, "Hard - analytic"),
        (ax_soft_err, "Soft - analytic"),
    ):
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
    fig.colorbar(image, ax=[ax_exact, ax_hard, ax_soft], fraction=0.046, pad=0.04)
    fig.colorbar(err_image, ax=[ax_hard_err, ax_soft_err], fraction=0.046, pad=0.04)

    _plot_loss(ax_loss, hard, "C0", "hard")
    _plot_loss(ax_loss, soft, "C1", "soft")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=8)

    trace_x = np.linspace(0.0, problem.length, 401)
    times = (0.0, 0.5 * problem.time_final, problem.time_final)
    for time, color in zip(times, ("C0", "C1", "C2"), strict=True):
        exact_line = standing_wave_1d(
            trace_x,
            time,
            wave_speed=problem.wave_speed,
            length=problem.length,
            amplitude=problem.amplitude,
        )
        hard_line = predict_physical(
            hard.result.model, hard.result.scales, trace_x, np.full_like(trace_x, time)
        )
        soft_line = predict_physical(
            soft.result.model, soft.result.scales, trace_x, np.full_like(trace_x, time)
        )
        ax_slice.plot(trace_x, exact_line, color=color, lw=2.0)
        ax_slice.plot(trace_x, hard_line, color=color, lw=1.2, ls="--")
        ax_slice.plot(trace_x, soft_line, color=color, lw=1.2, ls=":")
    ax_slice.legend(
        handles=[
            Line2D([0], [0], color="0.2", lw=2.0, label="analytic"),
            Line2D([0], [0], color="0.2", lw=1.2, ls="--", label="hard"),
            Line2D([0], [0], color="0.2", lw=1.2, ls=":", label="soft"),
            *(
                Line2D([0], [0], color=color, lw=2.0, label=label)
                for color, label in zip(
                    ("C0", "C1", "C2"),
                    ("t = 0", "t = T/2", "t = T"),
                    strict=True,
                )
            ),
        ],
        ncol=6,
        fontsize=8,
    )
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("u")
    ax_slice.set_title("Traces")
    ax_slice.set_xlim(0.0, problem.length)
    fig.suptitle(
        "1D acoustic PINN    "
        f"hard rel L2 {hard.metrics['relative_l2_analytic']:.3e} "
        f"({hard.metrics['seconds']:.0f}s)    "
        f"soft rel L2 {soft.metrics['relative_l2_analytic']:.3e} "
        f"({soft.metrics['seconds']:.0f}s)    "
        f"FDM rel L2 {hard.metrics['fdm_relative_l2_analytic']:.3e}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(artifact, dpi=160)
    plt.close(fig)


def _show(
    axis: Axes,
    data: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: str,
) -> Any:
    return axis.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )


def _plot_loss(axis: Axes, mode: ModeEval, color: str, label: str) -> None:
    losses = [row["loss"] for row in mode.result.history]
    if not losses:
        return
    axis.semilogy(
        np.arange(1, len(losses) + 1),
        np.clip(losses, 1.0e-16, None),
        color=color,
        label=label,
    )


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(error.ravel()))
    return float(np.linalg.norm(error.ravel()) / denom)


if __name__ == "__main__":
    main()
