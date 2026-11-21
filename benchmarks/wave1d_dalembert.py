"""Train the 1D acoustic PINN and compare it with d'Alembert and the FDM reference.

Run from the repository root:

    python benchmarks/wave1d_dalembert.py --config configs/wave1d.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from geopinn.config import load_config
from geopinn.initial import displacement_torch
from geopinn.physics.differentiate import partial
from geopinn.reference.analytic import standing_wave_1d
from geopinn.reference.fdm import solve_wave_1d
from geopinn.train import TrainResult, predict_physical, train

FDM_INTERVALS = 400
FDM_CFL = 0.5
EVAL_NX = 257
EVAL_NT = 129


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wave1d.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wave1d"))
    parser.add_argument("--adam-steps", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/wave1d_comparison.png"),
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    optim = config.optim
    if args.adam_steps is not None:
        optim = replace(optim, adam_steps=args.adam_steps)
    if args.lbfgs_steps is not None:
        optim = replace(optim, lbfgs_steps=args.lbfgs_steps)
    config = replace(config, optim=optim, checkpoint=str(args.output / "checkpoint.pt"))

    result = train(config, verbose=True)
    metrics = evaluate(result, args.output, args.artifact)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    artifact_metrics = args.artifact.with_name("wave1d_metrics.json")
    artifact_metrics.parent.mkdir(parents=True, exist_ok=True)
    artifact_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def evaluate(result: TrainResult, output: Path, artifact: Path) -> dict[str, float]:
    config = result.config
    problem = config.problem
    if problem.ic != "sine":
        raise ValueError("this benchmark verifies the sine standing wave; set problem.ic to sine")
    scales = result.scales
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
    pinn = predict_physical(result.model, scales, xx, tt)
    rel_analytic = _relative_l2(pinn - exact, exact)
    max_abs = float(np.max(np.abs(pinn - exact)))
    final = exact[:, -1]
    rel_final = _relative_l2(pinn[:, -1] - final, final)

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
    pinn_on_fdm = predict_physical(result.model, scales, fdm.x[None, :], fdm.t[:, None])
    # solve_wave_1d stores history as (time, space); the mesh above matches that.
    rel_fdm = _relative_l2(pinn_on_fdm - fdm.u, fdm.u)
    fdm_vs_exact = _relative_l2(fdm.u - fdm_exact, fdm_exact)

    constraints = _constraint_errors(result)
    _save_figure(
        x=x,
        t=t,
        exact=exact.T,
        pinn=pinn.T,
        x_fdm=fdm.x,
        fdm_final=fdm.u[-1],
        history_loss=[row["loss"] for row in result.history],
        history_stage=[row["stage"] for row in result.history],
        rel_analytic=rel_analytic,
        fdm_vs_exact=fdm_vs_exact,
        output=output / "wave1d_comparison.png",
        artifact=artifact,
        length=problem.length,
        time_final=problem.time_final,
        wave_speed=problem.wave_speed,
        amplitude=problem.amplitude,
        model=result,
    )
    n_params = sum(parameter.numel() for parameter in result.model.parameters())
    return {
        "relative_l2_analytic": rel_analytic,
        "relative_l2_analytic_final_time": rel_final,
        "max_abs_analytic": max_abs,
        "relative_l2_fdm": rel_fdm,
        "fdm_relative_l2_analytic": fdm_vs_exact,
        "bc_max_abs": constraints["bc_max_abs"],
        "ic_u_max_abs": constraints["ic_u_max_abs"],
        "ic_velocity_max_abs": constraints["ic_velocity_max_abs"],
        "c_hat": scales.c_hat,
        "n_parameters": float(n_params),
        "seconds": result.seconds,
        "final_loss": result.history[-1]["loss"] if result.history else float("nan"),
        "fdm_dx": fdm.dx,
        "fdm_dt": fdm.dt,
        "fdm_cfl": fdm.cfl,
    }


def _constraint_errors(result: TrainResult) -> dict[str, float]:
    """Max |boundary|, |u - u0| and |u_t| on the nondimensional initial line."""
    model = result.model
    parameter = next(model.parameters())
    n = 501
    x = torch.linspace(0.0, 1.0, n, dtype=parameter.dtype, device=parameter.device).reshape(-1, 1)
    t_line = torch.linspace(0.0, 1.0, n, dtype=parameter.dtype, device=parameter.device).reshape(
        -1, 1
    )
    displacement = displacement_torch(
        "sine",
        amplitude=1.0,
        length=1.0,
    )
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
        "ic_velocity_max_abs": float(velocity.abs().max().detach()),
    }


def _save_figure(
    *,
    x: np.ndarray,
    t: np.ndarray,
    exact: np.ndarray,
    pinn: np.ndarray,
    x_fdm: np.ndarray,
    fdm_final: np.ndarray,
    history_loss: list[float],
    history_stage: list[str],
    rel_analytic: float,
    fdm_vs_exact: float,
    output: Path,
    artifact: Path,
    length: float,
    time_final: float,
    wave_speed: float,
    amplitude: float,
    model: TrainResult,
) -> None:
    # exact and pinn are passed as (time, space) for imshow.
    error = pinn - exact
    fig = plt.figure(figsize=(12.2, 7.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.32)
    ax_exact = fig.add_subplot(grid[0, 0])
    ax_pinn = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[0, 2])
    ax_slice = fig.add_subplot(grid[1, 0:2])
    ax_loss = fig.add_subplot(grid[1, 2])

    extent = (float(x[0]), float(x[-1]), float(t[0]), float(t[-1]))
    vmin = float(np.min(exact))
    vmax = float(np.max(exact))
    image = ax_exact.imshow(
        exact,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax_pinn.imshow(
        pinn,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    err_max = float(np.max(np.abs(error)))
    err_image = ax_err.imshow(
        error,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-err_max,
        vmax=err_max,
    )
    for axis, title in (
        (ax_exact, "Analytic"),
        (ax_pinn, "PINN"),
        (ax_err, "PINN - analytic"),
    ):
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
    fig.colorbar(image, ax=[ax_exact, ax_pinn], fraction=0.046, pad=0.04)
    fig.colorbar(err_image, ax=ax_err, fraction=0.046, pad=0.04)

    colors = ("C0", "C1", "C2")
    for time, color in zip((0.0, 0.5 * time_final, time_final), colors, strict=True):
        trace_x = np.linspace(0.0, length, 401)
        exact_line = standing_wave_1d(
            trace_x,
            time,
            wave_speed=wave_speed,
            length=length,
            amplitude=amplitude,
        )
        pinn_line = predict_physical(
            model.model, model.scales, trace_x, np.full_like(trace_x, time)
        )
        ax_slice.plot(trace_x, exact_line, color=color, lw=2.0)
        ax_slice.plot(trace_x, pinn_line, color=color, lw=1.3, ls="--")
    ax_slice.plot(
        x_fdm,
        fdm_final,
        color="k",
        ls="none",
        marker="o",
        ms=3.0,
        markevery=16,
    )
    legend_handles = [
        Line2D([0], [0], color="0.2", lw=2.0, label="analytic"),
        Line2D([0], [0], color="0.2", lw=1.3, ls="--", label="PINN"),
        Line2D([0], [0], color="k", marker="o", ls="none", label="FDM at t = T"),
        *(
            Line2D([0], [0], color=color, lw=2.0, label=label)
            for color, label in zip(colors, ("t = 0", "t = T/2", "t = T"), strict=True)
        ),
    ]
    ax_slice.legend(handles=legend_handles, ncol=3, fontsize=8)
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("u")
    ax_slice.set_title("Traces")
    ax_slice.set_xlim(0.0, length)

    adam = [
        loss for loss, stage in zip(history_loss, history_stage, strict=True) if stage == "adam"
    ]
    lbfgs = [
        loss for loss, stage in zip(history_loss, history_stage, strict=True) if stage == "lbfgs"
    ]
    if adam:
        ax_loss.semilogy(np.arange(1, len(adam) + 1), np.clip(adam, 1.0e-16, None), label="Adam")
    if lbfgs:
        ax_loss.semilogy(
            np.arange(len(adam) + 1, len(adam) + len(lbfgs) + 1),
            np.clip(lbfgs, 1.0e-16, None),
            label="L-BFGS",
        )
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=8)
    fig.suptitle(
        f"1D acoustic PINN    rel L2 vs analytic = {rel_analytic:.3e}"
        f"    FDM vs analytic = {fdm_vs_exact:.3e}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(artifact, dpi=160)
    plt.close(fig)


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(error.ravel()))
    return float(np.linalg.norm(error.ravel()) / denom)


if __name__ == "__main__":
    main()
