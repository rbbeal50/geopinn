"""Train 2D acoustic PINNs and compare them with the analytic mode and the FDM.

Run from the repository root:

    python benchmarks/wave2d_forward.py

The standing-wave runs are checked against ``sin(m π x) sin(n π z) cos(ω t)``.
The layered run is checked against the 2D leapfrog scheme.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from geopinn.config import TrainConfig, load_config
from geopinn.physics.speeds import layered_profile
from geopinn.reference.analytic import gaussian_pulse_2d, standing_wave_2d
from geopinn.reference.fdm import solve_wave_2d
from geopinn.train import TrainResult, predict_physical, train

FDM_CFL = 0.4
FDM_ORDER: Literal[4] = 4
STANDING_NX = 97
STANDING_NT = 17


@dataclass
class CaseEval:
    """One trained case and the numbers the report quotes."""

    name: str
    result: TrainResult
    metrics: dict[str, float | str]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/wave2d"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("/opt/cursor/artifacts"))
    parser.add_argument(
        "--cases",
        default="standing-soft,standing-hard,layered",
        help="Comma-separated: standing-soft, standing-hard, layered",
    )
    parser.add_argument("--adam-steps", type=int, default=None)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    parser.add_argument("--fdm-intervals", type=int, default=160)
    args = parser.parse_args(argv)
    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    cases: list[CaseEval] = []
    for name in selected:
        cases.append(
            _run_case(
                name,
                args.output / name,
                args.adam_steps,
                args.lbfgs_steps,
                args.fdm_intervals,
                args.artifact_dir,
            )
        )
    payload: dict[str, Any] = {
        "hardware": _hardware(),
        "cases": {case.name: case.metrics for case in cases},
    }
    text = json.dumps(payload, indent=2) + "\n"
    (args.output / "metrics.json").write_text(text, encoding="utf-8")
    (args.artifact_dir / "wave2d_metrics.json").write_text(text, encoding="utf-8")
    print(text, flush=True)


def _run_case(
    name: str,
    output: Path,
    adam_steps: int | None,
    lbfgs_steps: int | None,
    fdm_intervals: int,
    artifact_dir: Path,
) -> CaseEval:
    output.mkdir(parents=True, exist_ok=True)
    config_name = {
        "standing-soft": "wave2d_standing_soft.yaml",
        "standing-hard": "wave2d_standing_hard.yaml",
        "layered": "wave2d_layered.yaml",
    }.get(name)
    if config_name is None:
        raise SystemExit(f"unknown case {name!r}")
    config = _with_steps(load_config(Path("configs") / config_name), adam_steps, lbfgs_steps)
    config = replace(config, checkpoint=str(output / "checkpoint.pt"))
    print(f"=== {name} ({config.constraints}) ===", flush=True)
    result = train(config, verbose=True)
    if name == "layered":
        metrics = _evaluate_layered(result, fdm_intervals, output, artifact_dir)
    else:
        metrics = _evaluate_standing(result, output, artifact_dir, name)
    metrics["seconds"] = result.seconds
    metrics["hardware"] = _hardware()
    return CaseEval(name=name, result=result, metrics=metrics)


def _with_steps(
    config: TrainConfig, adam_steps: int | None, lbfgs_steps: int | None
) -> TrainConfig:
    optim = config.optim
    if adam_steps is not None:
        optim = replace(optim, adam_steps=adam_steps)
    if lbfgs_steps is not None:
        optim = replace(optim, lbfgs_steps=lbfgs_steps)
    return replace(config, optim=optim)


def _evaluate_standing(
    result: TrainResult,
    output: Path,
    artifact_dir: Path,
    name: str,
) -> dict[str, float | str]:
    problem = result.config.problem
    depth = problem.length if problem.depth is None else problem.depth
    x = np.linspace(0.0, problem.length, STANDING_NX)
    z = np.linspace(0.0, depth, STANDING_NX)
    times = np.linspace(0.0, problem.time_final, STANDING_NT)
    xx, zz, tt = np.meshgrid(x, z, times, indexing="ij")
    exact = standing_wave_2d(
        xx,
        zz,
        tt,
        wave_speed=problem.wave_speed,
        length_x=problem.length,
        length_z=depth,
        amplitude=problem.amplitude,
        mode_x=problem.mode_x,
        mode_z=problem.mode_z,
    )
    pinn = predict_physical(result.model, result.scales, xx, tt, z=zz)
    rel = _relative_l2(pinn - exact, exact)
    final = _relative_l2(pinn[:, :, -1] - exact[:, :, -1], exact[:, :, -1])
    metrics: dict[str, float | str] = {
        "relative_l2_analytic": rel,
        "relative_l2_analytic_final_time": final,
        "max_abs_analytic": float(np.max(np.abs(pinn - exact))),
        "constraints": result.config.constraints,
        "activation": result.config.network.activation,
        "fourier_features": float(result.config.network.fourier_features),
        "causal": 1.0 if result.config.loss.causal else 0.0,
        "n_parameters": float(sum(p.numel() for p in result.model.parameters())),
        "final_loss": result.history[-1]["loss"] if result.history else float("nan"),
    }
    _save_standing_plot(name, x, z, times, exact, pinn, rel, output, artifact_dir)
    return metrics


def _evaluate_layered(
    result: TrainResult,
    intervals: int,
    output: Path,
    artifact_dir: Path,
) -> dict[str, float | str]:
    problem = result.config.problem
    spec = result.config.coefficients.wave_speed
    if (
        spec.c_top is None
        or spec.c_bottom is None
        or spec.interface is None
        or spec.transition is None
    ):
        raise ValueError("layered benchmark requires the layered speed parameters")
    c_top = spec.c_top
    c_bottom = spec.c_bottom
    interface = spec.interface
    transition = spec.transition
    depth = problem.length if problem.depth is None else problem.depth
    center_z = (
        problem.gaussian_center if problem.gaussian_center_z is None else problem.gaussian_center_z
    )
    width_z = (
        problem.gaussian_width if problem.gaussian_width_z is None else problem.gaussian_width_z
    )
    dx = problem.length / intervals
    dz = depth / intervals
    x = np.linspace(0.0, problem.length, intervals + 1)
    z = np.linspace(0.0, depth, intervals + 1)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    initial = gaussian_pulse_2d(
        xx,
        zz,
        center_x=problem.gaussian_center,
        center_z=center_z,
        width_x=problem.gaussian_width,
        width_z=width_z,
        length_x=problem.length,
        length_z=depth,
        amplitude=problem.amplitude,
    )

    def speed(x_mesh: np.ndarray, z_mesh: np.ndarray) -> np.ndarray:
        del x_mesh
        return layered_profile(
            z_mesh,
            c_top=c_top,
            c_bottom=c_bottom,
            interface=interface,
            transition=transition,
        )

    times = np.linspace(0.0, problem.time_final, 5)
    started = time.perf_counter()
    reference = solve_wave_2d(
        initial,
        wave_speed=speed,
        dx=dx,
        dy=dz,
        t_final=problem.time_final,
        cfl=FDM_CFL,
        spatial_order=FDM_ORDER,
        snapshot_times=tuple(float(value) for value in times),
    )
    coarse_initial = _subsample(initial, 2)
    coarse = solve_wave_2d(
        coarse_initial,
        wave_speed=speed,
        dx=dx * 2.0,
        dy=dz * 2.0,
        t_final=problem.time_final,
        cfl=FDM_CFL,
        spatial_order=FDM_ORDER,
    )
    fdm_seconds = time.perf_counter() - started
    # Compare at the coarse nodes. If the global error is C h^p, the fine-grid
    # error is about |u_coarse - u_fine| / (2^p - 1). p = 4 is the design order;
    # the odd extension is exact only for sine modes, so this is an estimate.
    fine_on_coarse = reference.u[::2, ::2]
    if fine_on_coarse.shape != coarse.u.shape:
        fine_on_coarse = fine_on_coarse[: coarse.u.shape[0], : coarse.u.shape[1]]
    fdm_gap = _relative_l2(coarse.u - fine_on_coarse, fine_on_coarse)
    richardson = fdm_gap / (2.0**FDM_ORDER - 1.0)

    frames_fdm: list[np.ndarray] = []
    frames_pinn: list[np.ndarray] = []
    sample_times: list[float] = []
    for instant, frame in reference.snapshots:
        sample_times.append(instant)
        frames_fdm.append(frame)
        tt = np.full_like(xx, instant)
        frames_pinn.append(predict_physical(result.model, result.scales, xx, tt, z=zz))
    fdm_stack = np.stack(frames_fdm, axis=0)
    pinn_stack = np.stack(frames_pinn, axis=0)
    rel = _relative_l2(pinn_stack - fdm_stack, fdm_stack)
    final = _relative_l2(frames_pinn[-1] - frames_fdm[-1], frames_fdm[-1])
    metrics: dict[str, float | str] = {
        "relative_l2_fdm": rel,
        "relative_l2_fdm_final_time": final,
        "max_abs_fdm": float(np.max(np.abs(pinn_stack - fdm_stack))),
        "fdm_coarse_gap": fdm_gap,
        "fdm_richardson_l2": richardson,
        "fdm_seconds": fdm_seconds,
        "fdm_intervals": float(intervals),
        "fdm_order": float(FDM_ORDER),
        "fdm_cfl": reference.cfl,
        "constraints": result.config.constraints,
        "activation": result.config.network.activation,
        "fourier_features": float(result.config.network.fourier_features),
        "fourier_sigma": result.config.network.fourier_sigma,
        "causal": 1.0 if result.config.loss.causal else 0.0,
        "causal_epsilon": result.config.loss.causal_epsilon,
        "n_parameters": float(sum(p.numel() for p in result.model.parameters())),
        "final_loss": result.history[-1]["loss"] if result.history else float("nan"),
    }
    _save_layered_plot(
        x,
        z,
        sample_times,
        frames_fdm,
        frames_pinn,
        rel,
        output,
        artifact_dir,
    )
    return metrics


def _subsample(field: np.ndarray, stride: int) -> np.ndarray:
    return np.ascontiguousarray(field[::stride, ::stride])


def _save_standing_plot(
    name: str,
    x: np.ndarray,
    z: np.ndarray,
    times: np.ndarray,
    exact: np.ndarray,
    pinn: np.ndarray,
    rel: float,
    output: Path,
    artifact_dir: Path,
) -> None:
    # Show three times. exact/pinn have shape (nz, nx, nt) because meshgrid ij
    # was called as (x, z, t), so axis 0 is x and axis 1 is z.
    picks = (0, exact.shape[2] // 2, exact.shape[2] - 1)
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0), constrained_layout=True)
    vmax = float(np.max(np.abs(exact)))
    error = pinn - exact
    err_max = float(np.max(np.abs(error)))
    extent = (float(x[0]), float(x[-1]), float(z[0]), float(z[-1]))
    for column, index in enumerate(picks):
        _image(axes[0, column], exact[:, :, index].T, extent, -vmax, vmax, "RdBu_r")
        _image(axes[1, column], pinn[:, :, index].T, extent, -vmax, vmax, "RdBu_r")
        _image(axes[2, column], error[:, :, index].T, extent, -err_max, err_max, "coolwarm")
        axes[0, column].set_title(f"t = {times[index]:.2f}")
    axes[0, 0].set_ylabel("analytic  z")
    axes[1, 0].set_ylabel("PINN  z")
    axes[2, 0].set_ylabel("error  z")
    for axis in axes[2, :]:
        axis.set_xlabel("x")
    fig.suptitle(f"{name}    relative L2 {rel:.3e}")
    path = output / "standing_snapshots.png"
    fig.savefig(path, dpi=140)
    fig.savefig(artifact_dir / f"wave2d_{name}.png", dpi=140)
    plt.close(fig)


def _save_layered_plot(
    x: np.ndarray,
    z: np.ndarray,
    times: list[float],
    fdm: list[np.ndarray],
    pinn: list[np.ndarray],
    rel: float,
    output: Path,
    artifact_dir: Path,
) -> None:
    # FDM fields are (nz, nx) with axis 0 = z. imshow with origin lower and a
    # transpose puts x on the horizontal axis and z on the vertical axis.
    n = len(times)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n + 1.2, 8.2), constrained_layout=True)
    stacked = np.stack(fdm, axis=0)
    vmax = float(np.max(np.abs(stacked)))
    err_max = max(float(np.max(np.abs(pinn[i] - fdm[i]))) for i in range(n))
    extent = (float(x[0]), float(x[-1]), float(z[0]), float(z[-1]))
    for column, instant in enumerate(times):
        _image(axes[0, column], fdm[column], extent, -vmax, vmax, "RdBu_r")
        _image(axes[1, column], pinn[column], extent, -vmax, vmax, "RdBu_r")
        _image(axes[2, column], pinn[column] - fdm[column], extent, -err_max, err_max, "coolwarm")
        axes[0, column].set_title(f"t = {instant:.2f}")
    axes[0, 0].set_ylabel("FDM  z")
    axes[1, 0].set_ylabel("PINN  z")
    axes[2, 0].set_ylabel("PINN - FDM  z")
    for axis in axes[2, :]:
        axis.set_xlabel("x")
    fig.suptitle(f"layered Gaussian pulse    relative L2 vs FDM {rel:.3e}")
    fig.savefig(output / "layered_snapshots.png", dpi=140)
    fig.savefig(artifact_dir / "wave2d_layered_snapshots.png", dpi=140)
    plt.close(fig)


def _image(
    axis: Axes,
    field: np.ndarray,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    # field is (nz, nx). imshow rows are vertical, so this is z up, x across.
    axis.imshow(
        field, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto"
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(error.ravel()))
    return float(np.linalg.norm(error.ravel()) / denom)


def _hardware() -> str:
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0))
    machine = platform.processor() or platform.machine()
    return f"CPU {machine}, {os.cpu_count()} cores, torch {torch.__version__}"


if __name__ == "__main__":
    main()
