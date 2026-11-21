"""Classical reference solutions used to verify the network."""

from geopinn.reference.analytic import dalembert_dirichlet, standing_wave_1d
from geopinn.reference.fdm import CFLViolation, check_cfl, solve_wave_1d

__all__ = [
    "CFLViolation",
    "check_cfl",
    "dalembert_dirichlet",
    "solve_wave_1d",
    "standing_wave_1d",
]
