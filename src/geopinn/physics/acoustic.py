"""Acoustic wave equation ``u_tt - c² ∇²u - s = 0``."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from geopinn.physics.base import PDE, ResidualTerm, Site
from geopinn.physics.coefficients import CoeffLike, evaluate_coefficient
from geopinn.physics.differentiate import laplacian, partial, second_partial
from geopinn.physics.fields import Fields, coerce_fields

PredictFn = Callable[[Tensor, Tensor], Tensor | Fields]


class AcousticWave(PDE):
    """Acoustic wave equation with a coefficient wave speed.

    The strong form encoded by the ``residual`` term is

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

    Dirichlet data default to ``g = 0``. Initial data default to a medium at
    rest (``u0 = 0``, ``v0 = 0``). Those data are coefficients too, so they
    may be callables or tensors.
    """

    def __init__(
        self,
        wave_speed: CoeffLike,
        *,
        source: CoeffLike | None = None,
        dirichlet: CoeffLike | None = None,
        initial_displacement: CoeffLike | None = None,
        initial_velocity: CoeffLike | None = None,
    ) -> None:
        if isinstance(wave_speed, bool) or (
            isinstance(wave_speed, (int, float)) and wave_speed <= 0.0
        ):
            raise ValueError("a constant wave_speed must be positive")
        self.wave_speed = wave_speed
        self.source = source
        self.dirichlet = dirichlet
        self.initial_displacement = initial_displacement
        self.initial_velocity = initial_velocity

    def residual_terms(self) -> tuple[ResidualTerm, ...]:
        return (
            ResidualTerm("residual", Site.INTERIOR, self._pde_residual),
            ResidualTerm("bc", Site.BOUNDARY, self._boundary_residual),
            ResidualTerm("ic", Site.INITIAL, self._displacement_residual),
            ResidualTerm("ic_velocity", Site.INITIAL, self._velocity_residual),
        )

    def residual(self, x: Tensor, t: Tensor, predict: PredictFn) -> Tensor:
        """Pointwise PDE residual, shape ``(N, 1)``."""
        return self._pde_residual(x, t, _as_fields(predict, x, t))

    def bc(self, x: Tensor, t: Tensor, predict: PredictFn) -> Tensor:
        """Dirichlet residual ``u - g``, shape ``(N, 1)``."""
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
        # Dirichlet residual u - g. Both ends of the 1D interval use g = 0.
        field = fields["u"]
        prescribed = _zero_or(self.dirichlet, field, x, t)
        return field - prescribed

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
