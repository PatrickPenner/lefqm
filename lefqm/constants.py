"""Constants"""
from pathlib import Path

# import and re-export all lefshift constants
from lefshift.constants import *  # noqa

SHIELDING_COLUMN = "Shielding"
HARTREE_TO_KCALMOL = 627.5094740631
GEOMETRY_PRECISION = 1e-07

ENERGY_SD_PROPERTY = "energy in water (kcal/mol)"
BOLTZMANN_WEIGHT_SD_PROPERTY = "boltzmann weight in water"
SHIELDING_SD_PROPERTY = "isotropic shielding in water"

DEFAULT_CONFIG = Path(__file__).absolute().parent / "config.ini"
