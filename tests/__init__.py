"""Test registration"""
import logging

from .conformers_tests import ConformersTests
from .utils_tests import UtilsTests
from .main_tests import MainTests
from .commandline_calculation_tests import (
    CommandlineCalculationTests,
    ConformerGenerationTests,
    ShieldingCalculationTests,
)

# logging.disable(logging.CRITICAL)  # suppress noise from failure tests
