"""Shielding generation tests"""
import unittest
from pathlib import Path

from rdkit.Chem import SDMolSupplier

from lefqm import constants, utils
from lefqm.shieldings import ShieldingCalculation


class ShieldingCalculationTests(unittest.TestCase):
    """Shielding generation tests"""

    def test_run(self):
        """Test shielidng generation"""
        mol = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))[0]
        config_path = Path(__file__).absolute().parent.parent / "lefqm" / "config.ini"
        config = utils.config_to_dict(config_path)
        config["cores"] = 1
        shieldings = ShieldingCalculation(config).run(mol)
        for index, atom in enumerate(mol.GetAtoms()):
            self.assertAlmostEqual(
                shieldings[index], atom.GetDoubleProp(constants.SHIELDING_SD_PROPERTY)
            )

    def test_turbomole_calculate_shieldings(self):
        """Test shielidng generation with turbomole"""
        mol = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))[0]
        shieldings = ShieldingCalculation.turbomole_calculate_shieldings(mol, cores=1)
        for index, atom in enumerate(mol.GetAtoms()):
            self.assertAlmostEqual(
                shieldings[index], atom.GetDoubleProp(constants.SHIELDING_SD_PROPERTY)
            )
