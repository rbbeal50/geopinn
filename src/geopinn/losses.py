"""Named, weighted strong-form loss.

Each PDE term and each observation contributes one mean-square piece. The
YAML map ``loss.weights`` selects the pieces by name:

    L = sum_k  w_k ‖R_k‖²

Names are not fixed by this module. The acoustic equation emits ``residual``,
``bc``, ``ic``, and ``ic_velocity``. Observations passed in by the caller use
whatever names they were given (``data``, later ``data_vp``, and so on). A
weight whose name is not emitted is ignored, so a config can mention a term
that a future physics module will produce. A missing weight defaults to 0,
so a new term stays off until the YAML opts in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
) -> LossTerms:
    """Evaluate the composite loss at a fixed collocation set.

    Interior coordinates are promoted to differentiable leaves inside this
    function, because the strong form needs derivatives of the fields. A term
    whose weight is zero is skipped unless ``monitor_constraints`` is set, in
    which case it is still evaluated so training logs can report it.
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
            parts[term.name] = _mse(term.evaluate(x, t, fields))

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


def _leaves(batch: PointBatch) -> tuple[Tensor, Tensor]:
    x = batch.x.detach().requires_grad_(True)
    t = batch.t.detach().requires_grad_(True)
    return x, t
