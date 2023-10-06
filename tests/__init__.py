"""Test registration"""
import logging

from .conformers_tests import ConformersTests
from .shieldings_tests import ShieldingCalculationTests
from .utils_tests import UtilsTests
from .main_tests import MainTests

logging.disable(logging.CRITICAL)  # suppress noise form failure tests
