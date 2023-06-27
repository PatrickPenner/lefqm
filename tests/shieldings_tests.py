"""Shielding generation tests"""
import unittest

from rdkit.Chem import SDMolSupplier

from lefqm import constants
from lefqm.shieldings import calculate_shieldings


class ShieldingsTests(unittest.TestCase):
    """Shielding generation tests"""

    def test_shieldings(self):
        """Test shielidng generation"""
        mol = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))[0]
        shieldings = calculate_shieldings(mol, cores=1)
        for index, atom in enumerate(mol.GetAtoms()):
            self.assertAlmostEqual(
                shieldings[index], atom.GetDoubleProp(constants.SHIELDING_SD_PROPERTY)
            )
