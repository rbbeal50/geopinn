"""Strong-form PDE interface: named residuals and named fields.

Equations plug into the loss by listing :class:`ResidualTerm` objects. Each
term has a name, the collocation site it is evaluated on, and a function of
``(x, t, fields)``. The loss multiplies the mean square of each name by the
weight of that name in the YAML. Adding a data misfit or a regularization
term is a new name, not a new loss function.

Coefficients inside a term are tensors or callables (see
:mod:`geopinn.physics.coefficients`), so a later rock-physics module can hand
the wave equation a ``c(x)`` built from porosity and saturation, and a 2D
inversion can hand it a trainable field, without changing this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from torch import Tensor

from geopinn.physics.fields import Fields


class Site(Enum):
    """Which collocation set a residual is evaluated on."""

    INTERIOR = "interior"
    BOUNDARY = "boundary"
    INITIAL = "initial"


@dataclass(frozen=True)
class ResidualTerm:
    """One named contribution to the composite loss.

    ``evaluate(x, t, fields)`` returns a pointwise residual. Zero means that
    condition holds. ``where`` selects the interior, boundary, or initial
    batch. Several terms may share a site; the loss does one forward pass
    per site and then calls each term.
    """

    name: str
    where: Site
    evaluate: Callable[[Tensor, Tensor, Fields], Tensor]


class PDE(ABC):
    """Governing equation as a list of named residuals.

    Subclasses implement :meth:`residual_terms`. The loss only requires the
    names it is given weights for, so a new equation can emit a different set
    (for example a steady Darcy residual and a data misfit) without a new
    trainer. Acoustic problems also offer :meth:`geopinn.physics.acoustic.AcousticWave.residual`,
    :meth:`geopinn.physics.acoustic.AcousticWave.bc`, and
    :meth:`geopinn.physics.acoustic.AcousticWave.ic` for the conventional names.
    """

    @abstractmethod
    def residual_terms(self) -> tuple[ResidualTerm, ...]:
        """Named residuals this equation contributes."""

    @property
    def residual_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.residual_terms())

    def term(self, name: str) -> ResidualTerm:
        for term in self.residual_terms():
            if term.name == name:
                return term
        known = ", ".join(self.residual_names)
        raise KeyError(f"no residual named {name!r} (have {known})")
