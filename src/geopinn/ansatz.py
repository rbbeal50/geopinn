"""Trial functions. Hard constraints build the IBVP data into the architecture."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, cast

import torch
from torch import Tensor, nn

ConstraintForm = Literal["modal_sine", "polynomial"]
Profile = Callable[[Tensor], Tensor]


class UnconstrainedField(nn.Module):
    """Network output used directly as the field ``u(x, t)``.

    Boundary and initial conditions are then enforced only through the loss.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _check_columns(x, t)
        return _as_tensor(self.net(torch.cat((x, t), dim=1)))


class HardConstraintWave1D(nn.Module):
    """1D trial function with homogeneous Dirichlet data and a quiescent start.

    Coordinates are nondimensional, with ``x ∈ [0, 1]`` and ``t ∈ [0, 1]``.
    Two forms are available.

    Polynomial (any initial displacement that vanishes at the ends):

        u(x, t) = u0(x) + t² x (1 - x) N_θ(x, t).

    First-mode sine:

        u(x, t) = sin(π x) (1 + t² N_θ(x, t)).

    Both satisfy, by construction (Lagaris et al., 1998),

        u(0, t) = u(1, t) = 0,
        u(x, 0) = u0(x),
        u_t(x, 0) = 0,

    so the loss only has to drive the PDE residual to zero. The factor ``t²``
    is what enforces the velocity condition: differentiating in time brings
    down a factor of ``t``, which is zero on the initial line.
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        form: ConstraintForm,
        initial_displacement: Profile | None = None,
    ) -> None:
        super().__init__()
        if form not in ("modal_sine", "polynomial"):
            raise ValueError("form must be 'modal_sine' or 'polynomial'")
        if form == "polynomial" and initial_displacement is None:
            raise ValueError("polynomial form requires initial_displacement")
        if form == "modal_sine" and initial_displacement is not None:
            raise ValueError("modal_sine encodes u0(x) = sin(pi x) itself")
        self.net = net
        self.form: ConstraintForm = form
        self.initial_displacement = initial_displacement

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _check_columns(x, t)
        network = _as_tensor(self.net(torch.cat((x, t), dim=1)))
        if self.form == "modal_sine":
            return torch.sin(math.pi * x) * (1.0 + t * t * network)
        if self.initial_displacement is None:
            raise RuntimeError("polynomial form is missing its initial displacement")
        return self.initial_displacement(x) + (t * t) * x * (1.0 - x) * network


def _as_tensor(value: object) -> Tensor:
    # nn.Module.__call__ is typed as returning Any in the installed torch stubs.
    return cast(Tensor, value)


def _check_columns(x: Tensor, t: Tensor) -> None:
    if x.ndim != 2 or t.ndim != 2 or x.shape[1] != 1 or t.shape[1] != 1:
        raise ValueError("x and t must have shape (N, 1)")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must contain the same number of points")
