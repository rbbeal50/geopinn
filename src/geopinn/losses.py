"""Named, weighted strong-form loss.

Each PDE term and each observation contributes one mean-square piece. The
YAML map ``loss.weights`` selects the pieces by name:

    L = sum_k  w_k ‖R_k‖²

Names are not fixed by this module. The acoustic equation emits ``residual``,
``bc``, ``ic_u``, and ``ic_ut``. Observations passed in by the caller use
whatever names they were given (``data``, later ``data_vp``, and so on). A
weight whose name is not emitted is ignored, so a config can mention a term
that a future physics module will produce. A missing weight defaults to 0,
so a new term stays off until the YAML opts in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from geopinn.physics.base import PDE, ResidualTerm, Site
from geopinn.physics.fields import Fields, coerce_fields
from geopinn.sampling import PointBatch

PredictFn = Callable[[Tensor, Tensor], Tensor | Fields]


@dataclass
class LossTerms:
    """Scalar loss pieces. ``total`` keeps its graph. ``terms`` are unweighted MSEs."""

    total: Tensor
    terms: dict[str, Tensor]

    def get(self, name: str) -> Tensor | None:
        return self.terms.get(name)


def pinn_loss(
    pde: PDE,
    predict: PredictFn,
    interior: PointBatch,
    boundary: PointBatch,
    initial: PointBatch,
    weights: Mapping[str, float],
    *,
    observations: Mapping[str, tuple[Tensor, Tensor]] | None = None,
    monitor_constraints: bool = True,
    field_names: Sequence[str] = ("u",),
    causal_epsilon: float | None = None,
    causal_bins: int = 16,
) -> LossTerms:
    """Evaluate the composite loss at a fixed collocation set.

    Interior coordinates are promoted to differentiable leaves inside this
    function, because the strong form needs derivatives of the fields. A term
    whose weight is zero is skipped unless ``monitor_constraints`` is set, in
    which case it is still evaluated so training logs can report it.

    ``causal_epsilon`` turns on the causal residual weights of Wang, Sankaran,
    and Perdikaris (2022) for the ``pde`` term only. Time is split into
    ``causal_bins``. Bin ``i`` is multiplied by ``exp(-ε Σ_{k<i} L_k)`` with
    the earlier bin losses detached, so the optimizer fits the past before the
    future. ``None`` is the plain mean square used by the 1D benchmarks.
    """
    batches = {
        Site.INTERIOR: interior,
        Site.BOUNDARY: boundary,
        Site.INITIAL: initial,
    }
    grouped: dict[Site, list[ResidualTerm]] = {}
    seen: set[str] = set()
    for term in pde.residual_terms():
        if term.name in seen:
            raise ValueError(f"duplicate residual name {term.name!r}")
        seen.add(term.name)
        grouped.setdefault(term.where, []).append(term)

    parts: dict[str, Tensor] = {}
    for site, terms in grouped.items():
        active = [
            term for term in terms if weights.get(term.name, 0.0) != 0.0 or monitor_constraints
        ]
        if not active:
            continue
        x, t = _leaves(batches[site])
        fields = coerce_fields(predict(x, t), field_names)
        for term in active:
            pointwise = term.evaluate(x, t, fields)
            if term.name == "pde" and causal_epsilon is not None:
                parts[term.name] = _causal_mse(pointwise, t, causal_epsilon, causal_bins)
            else:
                parts[term.name] = _mse(pointwise)

    if observations is not None:
        for name, (prediction, target) in observations.items():
            parts[name] = _mse(prediction - target)

    total = _weighted_sum(parts, weights)
    return LossTerms(total=total, terms=parts)


def _weighted_sum(parts: Mapping[str, Tensor], weights: Mapping[str, float]) -> Tensor:
    total: Tensor | None = None
    for name, value in parts.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        scaled = weight * value
        total = scaled if total is None else total + scaled
    if total is not None:
        return total
    any_part = next(iter(parts.values()), None)
    if any_part is None:
        raise ValueError("the loss has no PDE terms and no observations")
    return any_part.detach().new_zeros(())


def _mse(residual: Tensor) -> Tensor:
    return (residual**2).mean()


def _causal_mse(residual: Tensor, t: Tensor, epsilon: float, n_bins: int) -> Tensor:
    """Time-weighted PDE loss. Earlier bins keep weight 1 until they are small.

    Empty bins are dropped so a bin with no samples is not treated as a solved
    residual. A single occupied bin, or a zero time span, falls back to the
    plain mean square.
    """
    if epsilon <= 0.0:
        raise ValueError("causal_epsilon must be positive")
    if n_bins < 1:
        raise ValueError("causal_bins must be positive")
    values = residual.reshape(-1)
    times = t.detach().reshape(-1)
    if values.shape != times.shape:
        raise ValueError("residual and t must have the same number of points")
    t_min = times.min()
    t_max = times.max()
    span = t_max - t_min
    if n_bins == 1 or float(span.detach()) <= 0.0:
        return (values**2).mean()
    scaled = (times - t_min) / span
    index = torch.clamp((scaled * n_bins).to(dtype=torch.long), min=0, max=n_bins - 1)
    losses: list[Tensor] = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        if int(mask.sum()) == 0:
            continue
        losses.append((values[mask] ** 2).mean())
    if len(losses) == 1:
        return losses[0]
    stacked = torch.stack(losses)
    # Detach the history. The current bin still sends a gradient.
    previous = torch.cumsum(stacked.detach(), dim=0) - stacked.detach()
    weights = torch.exp(-epsilon * previous)
    return (weights * stacked).sum() / weights.sum()


def _leaves(batch: PointBatch) -> tuple[Tensor, Tensor]:
    x = batch.x.detach().requires_grad_(True)
    t = batch.t.detach().requires_grad_(True)
    return x, t
