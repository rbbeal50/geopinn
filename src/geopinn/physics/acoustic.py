"""Acoustic wave equation ``u_tt - c² ∇²u - s = 0``."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from geopinn.physics.base import PDE, ResidualTerm, Site
from geopinn.physics.boundary import BoundaryCondition, BoundaryKind
from geopinn.physics.coefficients import CoeffLike, evaluate_coefficient
from geopinn.physics.differentiate import laplacian, partial, second_partial
from geopinn.physics.fields import Fields, coerce_fields

PredictFn = Callable[[Tensor, Tensor], Tensor | Fields]


class AcousticWave(PDE):
    """Acoustic wave equation with a coefficient wave speed.

    The strong form encoded by the ``pde`` term is

        u_tt - c(x, t)² ∇²u - s = 0.

    In one space dimension and with a constant ``c`` this is ``u_tt - c² u_xx = 0``,
    which is the forward problem verified in this increment. ``c`` is not baked
    in as a Python float only: pass a float, a tensor, or a callable. A float
    takes a plain multiply so the verified constant-coefficient graph is
    unchanged. A tensor (with grad) or a callable of ``x`` or ``(x, t)`` is the
    hook for a trainable ``c(x)`` and for a rock-physics map from porosity and
    saturation. For a spatially varying ``c`` the residual above is the
    constant-density form. The divergence form ``∇·(c² ∇u)`` is the same
    residual when ``c`` is constant and can replace this term later.

    The Laplacian is formed by automatic differentiation and is valid in any
    spatial dimension. The scalar unknown is the field named ``u``. Other
    named fields in the same batch are ignored, so a multi-output network can
    carry auxiliary parameters beside the wavefield.

    The boundary target comes from ``boundary``. Dirichlet is
    ``u - g(x, t)``. Neumann (outward ``u_n - h``) and absorbing
    (outgoing characteristic) are selected by ``BoundaryKind`` and raised as
    not-implemented until those residuals are filled in. Initial targets
    ``u0`` and ``v0`` are coefficients supplied by the problem. Nothing here
    assumes the initial displacement is ``sin(π x)``.

    Omitted boundary data mean homogeneous Dirichlet. Omitted initial data
    mean a medium at rest.
    """

    def __init__(
        self,
        wave_speed: CoeffLike,
        *,
        source: CoeffLike | None = None,
        boundary: BoundaryCondition | None = None,
        initial_displacement: CoeffLike | None = None,
        initial_velocity: CoeffLike | None = None,
    ) -> None:
        if isinstance(wave_speed, bool) or (
            isinstance(wave_speed, (int, float)) and wave_speed <= 0.0
        ):
            raise ValueError("a constant wave_speed must be positive")
        self.wave_speed = wave_speed
        self.source = source
        self.boundary = (
            boundary if boundary is not None else BoundaryCondition(BoundaryKind.DIRICHLET, 0.0)
        )
        self.initial_displacement = initial_displacement
        self.initial_velocity = initial_velocity

    def residual_terms(self) -> tuple[ResidualTerm, ...]:
        # Names match the soft-constraint loss:
        #   w_pde L_pde + w_bc L_bc + w_ic_u L_ic_u + w_ic_ut L_ic_ut.
        return (
            ResidualTerm("pde", Site.INTERIOR, self._pde_residual),
            ResidualTerm("bc", Site.BOUNDARY, self._boundary_residual),
            ResidualTerm("ic_u", Site.INITIAL, self._displacement_residual),
            ResidualTerm("ic_ut", Site.INITIAL, self._velocity_residual),
        )

    def residual(self, x: Tensor, t: Tensor, predict: PredictFn) -> Tensor:
        """Pointwise PDE residual, shape ``(N, 1)``."""
        return self._pde_residual(x, t, _as_fields(predict, x, t))

    def bc(self, x: Tensor, t: Tensor, predict: PredictFn) -> Tensor:
        """Boundary residual, shape ``(N, 1)``. Dirichlet is ``u - g``."""
        return self._boundary_residual(x, t, _as_fields(predict, x, t))

    def ic(self, x: Tensor, t: Tensor, predict: PredictFn) -> tuple[Tensor, Tensor]:
        """Initial residuals ``(u - u0, u_t - v0)``, each of shape ``(N, 1)``."""
        fields = _as_fields(predict, x, t)
        return self._displacement_residual(x, t, fields), self._velocity_residual(x, t, fields)

    def _pde_residual(self, x: Tensor, t: Tensor, fields: Fields) -> Tensor:
        # Acoustic wave equation, strong form:
        #   ∂²u/∂t² - c(x, t)² ∇²u - s = 0.
        # Homogeneous 1D problem used in this increment, constant c, s = 0:
        #   u_tt - c² u_xx = 0.
        field = fields["u"]
        u_tt = second_partial(field, t)
        lap = laplacian(field, x)
        source = _zero_or(self.source, field, x, t)
        if isinstance(self.wave_speed, (int, float)) and not isinstance(self.wave_speed, bool):
            return u_tt - (float(self.wave_speed) ** 2) * lap - source
        speed = evaluate_coefficient(self.wave_speed, x, t)
        return u_tt - speed**2 * lap - source

    def _boundary_residual(self, x: Tensor, t: Tensor, fields: Fields) -> Tensor:
        field = fields["u"]
        kind = self.boundary.kind
        if kind is BoundaryKind.DIRICHLET:
            # u - g(x, t) = 0. g is the problem's Dirichlet target.
            target = evaluate_coefficient(self.boundary.value, x, t)
            return field - target
        if kind is BoundaryKind.NEUMANN:
            raise NotImplementedError(
                "Neumann residual is the outward normal derivative minus the "
                "supplied flux, evaluated in AcousticWave._boundary_residual. "
                "Only Dirichlet is implemented."
            )
        raise NotImplementedError(
            "Absorbing residual is the outgoing characteristic "
            "(u_t + c u_n on a right-going end), evaluated in "
            "AcousticWave._boundary_residual. Only Dirichlet is implemented."
        )

    def _displacement_residual(self, x: Tensor, t: Tensor, fields: Fields) -> Tensor:
        # u(x, 0) - u0(x) = 0.
        field = fields["u"]
        displacement = _zero_or(self.initial_displacement, field, x, t)
        return field - displacement

    def _velocity_residual(self, x: Tensor, t: Tensor, fields: Fields) -> Tensor:
        # ∂u/∂t(x, 0) - v0(x) = 0.
        field = fields["u"]
        velocity = partial(field, t)
        initial_velocity = _zero_or(self.initial_velocity, field, x, t)
        return velocity - initial_velocity


def _as_fields(predict: PredictFn, x: Tensor, t: Tensor) -> Fields:
    return coerce_fields(predict(x, t))


def _zero_or(coeff: CoeffLike | None, like: Tensor, x: Tensor, t: Tensor) -> Tensor:
    if coeff is None:
        return torch.zeros_like(like)
    return evaluate_coefficient(coeff, x, t)
