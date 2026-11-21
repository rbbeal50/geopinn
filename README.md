# geopinn

Physics-informed neural networks for geophysical forward and inverse problems, written in plain PyTorch. No DeepXDE or other PINN libraries.

The first increment solves the **1D acoustic wave equation** forward in time and checks the network against d'Alembert's formula and a second-order finite-difference scheme. Later increments add 2D acoustics, an inverse problem for \(c(x)\), and other geophysical operators (gravity, DC resistivity, Darcy flow, heat conduction).

## Governing equation

Constant-coefficient acoustics,

\[
\partial_{tt} u - c^{2} \nabla^{2} u - s = 0.
\]

In one space dimension with no source this is

\[
u_{tt} - c^{2} u_{xx} = 0, \qquad x \in [0, L], \quad t \in [0, T].
\]

The benchmark initial-boundary-value problem is homogeneous Dirichlet with a quiescent start from the first sine mode,

\[
u(0, t) = u(L, t) = 0, \qquad u(x, 0) = A \sin(\pi x / L), \qquad u_t(x, 0) = 0.
\]

The exact solution (standing wave, and the reduction of d'Alembert's formula under the odd periodic extension of the initial displacement) is

\[
u(x, t) = A \sin(\pi x / L) \, \cos(\pi c t / L).
\]

`geopinn.reference.analytic.dalembert_dirichlet` evaluates the method-of-images formula for a general initial displacement that vanishes at the ends. `standing_wave_1d` is the closed form above.

## Nondimensionalization

Training is done in scaled variables

\[
x = L x', \qquad t = T t', \qquad u = U u',
\]

so the wave equation becomes

\[
u'_{t't'} - \left(\frac{c T}{L}\right)^{2} u'_{x'x'} = 0.
\]

The nondimensional speed \(c T / L\) is kept \(\mathcal{O}(1)\). In `configs/wave1d.yaml` the problem is already scaled: \(L = T = U = c = 1\), hence \(c T / L = 1\), and the exact field is \(\sin(\pi x')\cos(\pi t')\). `Scales` in `geopinn.nondim` is what the trainer uses, and it warns if \(|c T / L|\) falls outside \([10^{-2}, 10^{2}]\).

## Trial function and loss

The strong form is evaluated with reverse-mode automatic differentiation (`geopinn.physics.differentiate`). The Laplacian is summed over whatever spatial dimension `x` has; the 1D problem is the case of one spatial column.

The composite loss is a sum of named mean-square terms. The names and the weights both come from the run, not from a fixed argument list:

\[
\mathcal{L} = \sum_k w_k \|R_k\|^2.
\]

For this wave problem the names are `residual` (\(u_{tt} - c^2\nabla^2 u - s\)), `bc`, `ic` (the displacement), `ic_velocity`, and `data`. \(w_k\) is `loss.weights` in the YAML. A weight whose name nothing emits is kept and ignored, so a later term can be written down before the module that produces it exists.

Both initial conditions are encoded: the value of \(u\) and the value of \(u_t\).

For the sine benchmark the trial function builds those conditions in, so the optimizer only sees the PDE residual (Lagaris, Likas, Fotiadis, 1998):

\[
u(x', t') = \sin(\pi x') \, \bigl(1 + t'^{2} N_\theta(x', t')\bigr).
\]

The factor \(t'^{2}\) is what enforces \(u_t(x', 0) = 0\). A polynomial factor \(x'(1 - x')\) is available for a general initial displacement that vanishes at the ends (a boundary-corrected Gaussian is implemented). Soft constraints remain in the loss and are selected with `hard_constraints: false`.

\(N_\theta\) is a fully connected network: tanh or sine activations, Xavier (tanh) or SIREN (sine) initialisation, and optional random Fourier features. Inputs in the unit square are mapped to \([-1, 1]\) when Fourier features are off. Weights are float64. The run is seeded (Python, NumPy, PyTorch, plus dedicated generators for the network and for the sampler).

Collocation points are a fixed Latin-hypercube or Sobol' set. Optimization is Adam, with one learning-rate drop, then L-BFGS with a strong-Wolfe line search. Checkpoints store the weights, the config, and the loss history.

The classical reference in `geopinn.reference.fdm` is the second-order leapfrog scheme

\[
\frac{u^{n+1}_i - 2 u^n_i + u^{n-1}_i}{\Delta t^{2}}
  = c^{2} \frac{u^n_{i+1} - 2 u^n_i + u^n_{i-1}}{\Delta x^{2}},
\]

started from the second-order Taylor step so that a prescribed initial velocity (zero in this problem) does not drop the global order. The same integrator accepts a 2D Cartesian Laplacian. The CFL ratio \(c \Delta t \sqrt{\sum_i \Delta x_i^{-2}}\) must be at most 1; `check_cfl` raises otherwise.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

A CPU wheel of PyTorch is enough. If pip selects a large CUDA build, install the CPU wheel first and then the project:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Quickstart

From the repository root:

```bash
geopinn-train --config configs/wave1d.yaml
python benchmarks/wave1d_dalembert.py --config configs/wave1d.yaml
```

The benchmark trains the network, writes `outputs/wave1d/metrics.json` and `outputs/wave1d/wave1d_comparison.png`, and prints the relative \(L^2\) errors. The headline number is the space-time relative \(L^2\) error against the standing-wave solution,

\[
\frac{\|u_\theta - u\|_2}{\|u\|_2}.
\]

## Measured accuracy

`configs/wave1d.yaml` (seed 0, 4×64 tanh network, 2048 Sobol points, 1000 Adam steps then 100 L-BFGS steps, float64) was run on CPU. Space-time relative \(L^2\) against the standing wave was **1.45×10⁻⁴**, under the 10⁻³ target. The same run:

| comparison | relative \(L^2\) |
| --- | --- |
| PINN vs standing wave, full space-time | 1.45×10⁻⁴ |
| PINN vs standing wave, at \(t = T\) | 1.60×10⁻⁴ |
| PINN vs finite differences | 1.46×10⁻⁴ |
| finite differences vs standing wave | 3.22×10⁻⁶ |

The max absolute error against the standing wave was 2.12×10⁻⁴. The hard constraint matches the Dirichlet data and both initial conditions to roundoff: boundary residual 1.2×10⁻¹⁶, initial displacement residual 0, initial velocity residual 0. Training took about 16 seconds. A rerun with the same seed on the same machine reproduces these figures; the last digits can move on different hardware.

## Tests

```bash
pytest
ruff check src tests benchmarks
mypy
```

What the tests lock in:

- autograd second derivatives against a fourth-order finite-difference stencil, including the Cartesian Laplacian in 2D
- PDE residual at roundoff on the exact standing wave, on a manufactured source, and on a 2D separated mode
- the hard-constraint ansatz reproduces that exact field when \(N\) is the closed-form correction, and it matches Dirichlet data plus both initial conditions for sine and Gaussian data
- d'Alembert-with-images agrees with the standing wave
- the 1D finite-difference scheme agrees with the analytic solution and converges at second order under grid refinement; the 2D stencil is the same order
- sampler shapes, bounds, and seed determinism
- a two-step training run reloads from the checkpoint and repeats under the same seed

`mypy` is run in strict mode. The only ignore is `ignore_missing_imports` for SciPy, Matplotlib, PyYAML, and pytest, which do not ship complete stubs. There are no in-code `# type: ignore` comments.

## Layout

```
configs/wave1d.yaml          training config
src/geopinn/nets/mlp.py      tanh / sine MLP, optional Fourier features
src/geopinn/physics/         PDE interface, coefficients, named fields, acoustics
src/geopinn/sampling.py      interior, boundary, and initial collocation
src/geopinn/losses.py        weighted residual + BC + IC + optional data
src/geopinn/nondim.py        x/L, t/T, u/U
src/geopinn/train.py         Adam then L-BFGS, CLI, checkpoints
src/geopinn/reference/       d'Alembert solution and explicit FDM
benchmarks/wave1d_dalembert.py
tests/
notebooks/                   placeholder
```

## Extension API

The 1D benchmark is a constant \(c\). The interfaces underneath it are the ones a rock-physics module and a 2D inversion should plug into. Nothing in those interfaces assumes \(c\) is a Python float, that the network has a single output, or that the loss has a fixed list of arguments.

**Physics base class.** `PDE.residual_terms()` returns `ResidualTerm(name, where, evaluate)`. `evaluate(x, t, fields)` is the pointwise residual. `where` is the interior, the boundary, or the initial line. `AcousticWave` emits `residual`, `bc`, `ic`, and `ic_velocity`. The scalar unknown is the field named `u`. A coefficient is a float, a tensor (gradients included), or a callable. A one-argument callable receives \(x\); a two-argument callable receives \((x, t)\). That callable is where a Gassmann-type map or a trainable \(c(x)\) will sit. For a constant float the residual is still the plain multiply \(u_{tt} - c^2\nabla^2 u\), which is the form the benchmark verified. For a spatially varying \(c\) the same line is the constant-density strong form; the divergence form \(\nabla\cdot(c^2\nabla u)\) agrees with it when \(c\) is constant.

**Loss composition.** `pinn_loss` groups terms that share a collocation site, runs one forward pass per site, and adds \(w_k\|R_k\|^2\). Observations are an extra mapping of name to `(prediction, target)`, so `data_vp` is the same mechanism as `data`. A missing weight is 0. Hard constraints zero `bc`, `ic`, and `ic_velocity` in the trainer because the trial function already satisfies them; the YAML values stay as documentation for the soft-constraint path.

**Config schema.** `TrainConfig` is a frozen dataclass and `TrainConfig.from_dict` is what the YAML loader fills. The blocks are `problem`, `coefficients`, `network`, `sampling`, `loss`, and `optim`.

```yaml
coefficients:
  wave_speed:
    form: constant          # problem.wave_speed, then c*T/L
    # form: field           # not built yet
    # source: c_of_rock     # trainable c(x) or a rock-physics map
network:
  field_names: [u]          # later: [u, porosity, saturation]
loss:
  weights:
    residual: 1.0
    bc: 100.0
    ic: 100.0
    ic_velocity: 100.0
    data: 1.0
    # data_vp: 1.0
    # regularization_c: 1.0e-6
  hard_constraints: true
  constraint_form: modal_sine
```

`form: field` parses and names its `source`, and the trainer raises `CoefficientNotBuilt` until a builder exists. Passing the tensor or callable into `AcousticWave(wave_speed=...)` already works, including backprop into a parameter speed.

## Roadmap

1. **2D acoustic forward problem** on a rectangle, same hard or soft constraints, verified against the 2D leapfrog scheme already in `reference/fdm.py`.
2. **Inverse problem for \(c(x)\)**: add the wave speed as an unknown field, a data term on seismograms or on \(u\), and a check against a manufactured \(c(x)\).
3. **Other physics**, each as a new `PDE` with the same residual / boundary / initial interface and the same nondimensional scaling: gravity, DC resistivity, Darcy pressure diffusion, and heat conduction.

## References

- d'Alembert's formula and the method of images for the finite string.
- Lagaris, Likas, Fotiadis, "Artificial neural networks for solving ordinary and partial differential equations" (1998), for the trial function that satisfies the initial and boundary data.
- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks" (2019), for the strong-form residual loss.
- Sitzmann et al., "Implicit neural representations with periodic activation functions" (2020), for the sine-network initialisation.
