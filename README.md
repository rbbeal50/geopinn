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

For this wave problem the names are `pde` (\(u_{tt} - c^2\nabla^2 u - s\)), `bc` (Dirichlet \(u - g\)), `ic_u` (\(u - u_0\)), `ic_ut` (\(u_t - v_0\), with \(u_t\) from autograd), and `data`. \(w_k\) is `loss.weights` in the YAML. A weight whose name nothing emits is kept and ignored, so a later term can be written down before the module that produces it exists. The targets \(g\), \(u_0\), and \(v_0\) come from the problem (`boundary`, `dirichlet_value`, `ic`, `initial_velocity`). Dirichlet is implemented. Neumann and absorbing conditions are named on `BoundaryKind` and raised in `AcousticWave._boundary_residual` until those residuals are filled in.

`constraints` selects how those conditions are enforced. The default, when the key is omitted, is `soft`: the network output is the field, \(u = N_\theta(x, t)\), and the four terms above are all in the loss. `hard` keeps the trial function that satisfies homogeneous Dirichlet data and both initial conditions exactly, and the trainer zeros `bc`, `ic_u`, and `ic_ut` so only the PDE residual is minimized (Lagaris, Likas, Fotiadis, 1998):

\[
u(x', t') = \sin(\pi x') \, \bigl(1 + t'^{2} N_\theta(x', t')\bigr).
\]

The factor \(t'^{2}\) is what enforces \(u_t(x', 0) = 0\). A polynomial factor \(x'(1 - x')\) is the hard option for a general initial displacement that vanishes at the ends (a boundary-corrected Gaussian is implemented). Set `constraint_form: modal_sine` or `polynomial` only when `constraints` is `hard`.

`loss.balance: wang2021` turns on gradient-norm annealing (Wang, Teng, Perdikaris, 2021). For every non-PDE term it sets the weight from \(\max|\nabla L_{\mathrm{pde}}| / \mathrm{mean}|\nabla L_k|\), smoothed with mixing rate `balance_alpha` (0.9 in the paper) every `balance_every` Adam steps, then freezes the weights for L-BFGS. This is the right balancer when constraint terms and the PDE fight over the Adam step. It is not residual-based point reweighting, which moves collocation points inside one term. The switch defaults to `none`. On this particular standing wave the constraint residuals get small quickly and the annealer then pins their weights at the cap, which slows the PDE, so the soft benchmark leaves it off and uses fixed weights of 100.

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
geopinn-train --config configs/wave1d_soft.yaml
python benchmarks/wave1d_dalembert.py
```

`configs/wave1d_soft.yaml` is the soft run. `configs/wave1d.yaml` is the hard run. The benchmark trains both.

The benchmark trains the network, writes `outputs/wave1d/metrics.json` and `outputs/wave1d/wave1d_comparison.png`, and prints the relative \(L^2\) errors. The headline number is the space-time relative \(L^2\) error against the standing-wave solution,

\[
\frac{\|u_\theta - u\|_2}{\|u\|_2}.
\]

## Measured accuracy

Both modes were trained on CPU from the checked-in configs (seed 0, float64). Space-time relative \(L^2\) against the standing wave:

| mode | network and budget | relative \(L^2\) | at \(t = T\) | time |
| --- | --- | --- | --- | --- |
| hard (`configs/wave1d.yaml`) | 4×64 tanh, 2048 Sobol points, 1000 Adam + 100 L-BFGS | **1.45×10⁻⁴** | 1.60×10⁻⁴ | 16 s |
| soft (`configs/wave1d_soft.yaml`) | 4×64 sine, 4096 interior / 256 per side / 512 initial, weights 100, 1000 Adam + 600 L-BFGS | **9.89×10⁻⁴** | 1.16×10⁻³ | 74 s |

The finite-difference reference is 3.22×10⁻⁶ from the analytic solution. Hard-mode boundary and initial residuals are at roundoff (boundary 1.2×10⁻¹⁶, both initial residuals 0). Soft-mode max absolute error is 1.69×10⁻³. The soft run needed the sine network and the longer L-BFGS polish; tanh on a shorter budget stayed near 4×10⁻³, and gradient-norm annealing was left off for the reason above. A rerun with the same seed on the same machine reproduces these figures; the last digits can move on different hardware.

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
- soft-mode `pde`, `bc`, `ic_u`, and `ic_ut` are at roundoff when the network is replaced by the exact standing wave; Dirichlet and initial-velocity targets are the values supplied by the problem; boundary and initial samplers land on those sets
- both `constraints: hard` and `constraints: soft` parse, and omitting `constraints` defaults to soft
- d'Alembert-with-images agrees with the standing wave
- the 1D finite-difference scheme agrees with the analytic solution and converges at second order under grid refinement; the 2D stencil is the same order
- sampler shapes, bounds, and seed determinism
- a two-step training run reloads from the checkpoint and repeats under the same seed

`mypy` is run in strict mode. `ignore_missing_imports` covers SciPy, Matplotlib, PyYAML, and pytest. Two `# type: ignore[no-untyped-call]` comments mark `Tensor.backward` and `LBFGS.step`, which the installed torch stubs leave untyped.

## Layout

```
configs/wave1d.yaml          hard-constraint benchmark
configs/wave1d_soft.yaml     soft-constraint benchmark
src/geopinn/nets/mlp.py      tanh / sine MLP, optional Fourier features
src/geopinn/physics/         PDE interface, boundary targets, coefficients, acoustics
src/geopinn/balance.py       optional Wang 2021 gradient-norm weights
src/geopinn/sampling.py      interior, boundary, and initial collocation
src/geopinn/losses.py        named weighted PDE + BC + IC + optional data
src/geopinn/nondim.py        x/L, t/T, u/U
src/geopinn/train.py         Adam then L-BFGS, CLI, checkpoints
src/geopinn/reference/       d'Alembert solution and explicit FDM
benchmarks/wave1d_dalembert.py
tests/
notebooks/                   placeholder
```

## Extension API

The 1D benchmark is a constant \(c\). The interfaces underneath it are the ones a rock-physics module and a 2D inversion should plug into. Nothing in those interfaces assumes \(c\) is a Python float, that the network has a single output, or that the loss has a fixed list of arguments.

**Physics base class.** `PDE.residual_terms()` returns `ResidualTerm(name, where, evaluate)`. `evaluate(x, t, fields)` is the pointwise residual. `where` is the interior, the boundary, or the initial line. `AcousticWave` emits `pde`, `bc`, `ic_u`, and `ic_ut`. The scalar unknown is the field named `u`. Boundary and initial targets are coefficients supplied with the problem, not constants inside the residual. `BoundaryKind` is `dirichlet`, `neumann`, or `absorbing`; only Dirichlet (`u - g`) is implemented, and the other two raise in `_boundary_residual` where their formulas belong. A coefficient is a float, a tensor (gradients included), or a callable. A one-argument callable receives \(x\); a two-argument callable receives \((x, t)\). That callable is where a Gassmann-type map or a trainable \(c(x)\) will sit. For a constant float the PDE residual is still the plain multiply \(u_{tt} - c^2\nabla^2 u\). For a spatially varying \(c\) the same line is the constant-density strong form; the divergence form \(\nabla\cdot(c^2\nabla u)\) agrees with it when \(c\) is constant.

**Loss composition.** `pinn_loss` groups terms that share a collocation site, runs one forward pass per site, and adds \(w_k\|R_k\|^2\). In soft mode that is \(w_{\mathrm{pde}} L_{\mathrm{pde}} + w_{\mathrm{bc}} L_{\mathrm{bc}} + w_{\mathrm{ic\_u}} L_{\mathrm{ic\_u}} + w_{\mathrm{ic\_ut}} L_{\mathrm{ic\_ut}}\), with \(u_t\) from autograd and the batches coming from the boundary and initial samplers. Observations are an extra mapping of name to `(prediction, target)`, so `data_vp` is the same mechanism as `data`. A missing weight is 0. Hard mode zeros `bc`, `ic_u`, and `ic_ut` in the trainer because the trial function already satisfies them; the YAML values stay as the soft-constraint weights. `loss.balance: wang2021` optionally replaces those non-PDE weights by gradient-norm ratios. It is off unless requested.

**Config schema.** `TrainConfig` is a frozen dataclass and `TrainConfig.from_dict` is what the YAML loader fills. Top-level `constraints` is `soft` (the default if omitted) or `hard`. `constraint_form` is `none` for soft and `modal_sine` or `polynomial` for hard. The other blocks are `problem`, `coefficients`, `network`, `sampling`, `loss`, and `optim`.

```yaml
constraints: soft             # hard | soft; omitted means soft
constraint_form: none         # modal_sine or polynomial only when hard
problem:
  boundary: dirichlet         # neumann and absorbing are named, not implemented
  dirichlet_value: 0.0        # g in u - g
  ic: sine                    # u0; gaussian is the other built-in profile
  initial_velocity: 0.0       # v0
coefficients:
  wave_speed:
    form: constant            # problem.wave_speed, then c*T/L
    # form: field             # not built yet
    # source: c_of_rock       # trainable c(x) or a rock-physics map
network:
  field_names: [u]            # later: [u, porosity, saturation]
loss:
  weights:
    pde: 1.0
    bc: 100.0
    ic_u: 100.0
    ic_ut: 100.0
    data: 1.0
    # data_vp: 1.0
    # regularization_c: 1.0e-6
  balance: none               # or wang2021
  balance_alpha: 0.9
  balance_every: 10
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
- Wang, Teng, Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks" (2021), for the optional loss-term weights.
