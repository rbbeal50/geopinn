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
    """Network output used directly as the field.

    In 1D, ``x`` has shape ``(N, 1)``. In 2D it has shape ``(N, 2)`` with
    columns ``(x, z)``, and ``t`` is still ``(N, 1)``. Boundary and initial
    conditions are enforced only through the loss. This is the default.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _check_spacetime(x, t)
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
        _check_columns(x, t, spatial_dim=1)
        network = _as_tensor(self.net(torch.cat((x, t), dim=1)))
        if self.form == "modal_sine":
            return torch.sin(math.pi * x) * (1.0 + t * t * network)
        if self.initial_displacement is None:
            raise RuntimeError("polynomial form is missing its initial displacement")
        return self.initial_displacement(x) + (t * t) * x * (1.0 - x) * network


class HardConstraintWave2D(nn.Module):
    """2D trial function with homogeneous Dirichlet data and a quiescent start.

    Coordinates are nondimensional. ``x`` has columns ``(x, z)`` with
    ``x ∈ [0, x_extent]`` and ``z ∈ [0, z_extent]``, and ``t ∈ [0, 1]``.
    On the square training box both extents are 1.

    First-mode product (any positive integer modes):

        u = sin(m π x / x_extent) sin(n π z / z_extent) (1 + t² N_θ).

    Polynomial, for an initial displacement that already vanishes on the
    boundary of the rectangle:

        u = u0(x, z) + t² x (x_extent - x) z (z_extent - z) N_θ.

    Both satisfy ``u = 0`` on the boundary, ``u(·, 0) = u0``, and ``u_t(·, 0) = 0``
    (Lagaris et al., 1998). The loss only drives the PDE residual to zero.
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        form: ConstraintForm,
        x_extent: float = 1.0,
        z_extent: float = 1.0,
        mode_x: int = 1,
        mode_z: int = 1,
        initial_displacement: Profile | None = None,
    ) -> None:
        super().__init__()
        if form not in ("modal_sine", "polynomial"):
            raise ValueError("form must be 'modal_sine' or 'polynomial'")
        if x_extent <= 0.0 or z_extent <= 0.0:
            raise ValueError("extents must be positive")
        if mode_x < 1 or mode_z < 1:
            raise ValueError("modes must be positive integers")
        if form == "polynomial" and initial_displacement is None:
            raise ValueError("polynomial form requires initial_displacement")
        if form == "modal_sine" and initial_displacement is not None:
            raise ValueError("modal_sine encodes the product of sines itself")
        self.net = net
        self.form: ConstraintForm = form
        self.x_extent = float(x_extent)
        self.z_extent = float(z_extent)
        self.mode_x = int(mode_x)
        self.mode_z = int(mode_z)
        self.initial_displacement = initial_displacement

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _check_columns(x, t, spatial_dim=2)
        network = _as_tensor(self.net(torch.cat((x, t), dim=1)))
        xs = x[:, 0:1]
        zs = x[:, 1:2]
        if self.form == "modal_sine":
            # u = sin(m π x / Lx) sin(n π z / Lz) (1 + t² N).
            shape = torch.sin(self.mode_x * math.pi * xs / self.x_extent) * torch.sin(
                self.mode_z * math.pi * zs / self.z_extent
            )
            return shape * (1.0 + t * t * network)
        if self.initial_displacement is None:
            raise RuntimeError("polynomial form is missing its initial displacement")
        window = xs * (self.x_extent - xs) * zs * (self.z_extent - zs)
        return self.initial_displacement(x) + (t * t) * window * network


def _as_tensor(value: object) -> Tensor:
    # nn.Module.__call__ is typed as returning Any in the installed torch stubs.
    return cast(Tensor, value)


def _check_spacetime(x: Tensor, t: Tensor) -> None:
    if x.ndim != 2 or t.ndim != 2 or x.shape[1] < 1 or t.shape[1] != 1:
        raise ValueError("x must have shape (N, d) and t shape (N, 1)")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must contain the same number of points")


def _check_columns(x: Tensor, t: Tensor, *, spatial_dim: int) -> None:
    if x.ndim != 2 or t.ndim != 2 or x.shape[1] != spatial_dim or t.shape[1] != 1:
        raise ValueError(
            f"x must have shape (N, {spatial_dim}) and t shape (N, 1); "
            f"got {tuple(x.shape)} and {tuple(t.shape)}"
        )
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must contain the same number of points")
