"""Gradient-norm loss balancing (Wang, Teng, Perdikaris 2021).

Soft constraints fail when the PDE residual and the boundary or initial
terms have gradient magnitudes that differ by orders of magnitude, so one
term owns the Adam step. Wang et al. recompute a multiplier for each
non-reference term from

    lambda_hat = max |grad L_ref| / mean |grad L_k|

and smooth it with lambda <- (1 - alpha) lambda + alpha lambda_hat.
The reference term (the PDE residual) keeps its YAML weight. This
rebalances terms against each other.

It is not residual-based adaptive sampling, which reweights collocation
points inside a single term. That is a different lever and is not what the
competing BC/IC losses need.

The update is optional and off unless the config asks for it. Each update
is an extra reverse-mode pass per term, so the trainer runs it every
``balance_every`` Adam steps and then freezes the multipliers for L-BFGS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

# Practical guard. The ratio of gradient components can spike for a step or
# two early in training; an unbounded multiplier makes the next Adam step
# ignore the PDE residual.
_MIN_WEIGHT = 1.0e-8
_MAX_WEIGHT = 1.0e5


def wang2021_weights(
    parameters: Sequence[Tensor],
    parts: Mapping[str, Tensor],
    base: Mapping[str, float],
    running: dict[str, float],
    *,
    alpha: float,
    reference: str = "pde",
) -> dict[str, float]:
    """Return term weights after one learning-rate-annealing update.

    ``parts`` are unweighted mean squares that still have a graph.
    ``base`` is the YAML weight (after hard constraints have zeroed the
    terms the trial function already satisfies). ``running`` stores the
    smoothed multiplier and is updated in place. ``alpha`` is the paper's
    mixing rate: 0.9 puts most of the step on the new ratio.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("balance alpha must lie in (0, 1]")
    updated = dict(base)
    reference_loss = parts.get(reference)
    if reference_loss is None or base.get(reference, 0.0) == 0.0:
        return updated
    reference_max = _max_abs_grad(reference_loss, parameters)
    if reference_max == 0.0:
        return updated
    for name, loss in parts.items():
        if name == reference or base.get(name, 0.0) == 0.0:
            continue
        mean_abs = _mean_abs_grad(loss, parameters)
        if mean_abs == 0.0:
            continue
        proposed = reference_max / mean_abs
        previous = running.get(name, base[name])
        blended = (1.0 - alpha) * previous + alpha * proposed
        blended = min(max(blended, _MIN_WEIGHT), _MAX_WEIGHT)
        running[name] = blended
        updated[name] = blended
    return updated


def _max_abs_grad(loss: Tensor, parameters: Sequence[Tensor]) -> float:
    components = _abs_components(loss, parameters)
    if components is None:
        return 0.0
    return float(components.max())


def _mean_abs_grad(loss: Tensor, parameters: Sequence[Tensor]) -> float:
    components = _abs_components(loss, parameters)
    if components is None:
        return 0.0
    return float(components.mean())


def _abs_components(loss: Tensor, parameters: Sequence[Tensor]) -> Tensor | None:
    if not parameters:
        return None
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    pieces = [grad.detach().abs().reshape(-1) for grad in grads if grad is not None]
    if not pieces:
        return None
    return torch.cat(pieces)
