"""Boundary condition supplied by the problem, not by the trial function.

Dirichlet data are implemented. Neumann and absorbing ends are named here so a
later residual can branch on ``kind`` without inventing a new config key. The
target itself is a coefficient: a float, a tensor, or a callable of the
collocation points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from geopinn.physics.coefficients import CoeffLike


class BoundaryKind(Enum):
    """Which trace the boundary residual constrains."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ABSORBING = "absorbing"


@dataclass(frozen=True)
class BoundaryCondition:
    """Target for one boundary residual.

    ``value`` is the Dirichlet value ``g`` today. The same field will carry
    the Neumann flux or the absorbing impedance when those residuals exist.
    """

    kind: BoundaryKind
    value: CoeffLike
