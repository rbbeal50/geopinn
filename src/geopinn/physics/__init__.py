"""Governing equations and automatic differentiation for strong-form residuals."""

from geopinn.physics.acoustic import AcousticWave
from geopinn.physics.base import PDE, ResidualTerm, Site
from geopinn.physics.coefficients import evaluate_coefficient
from geopinn.physics.fields import Fields

__all__ = [
    "PDE",
    "AcousticWave",
    "Fields",
    "ResidualTerm",
    "Site",
    "evaluate_coefficient",
]
