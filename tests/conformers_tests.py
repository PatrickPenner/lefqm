"""Conformer generation tests"""
import unittest

from rdkit import Chem

from lefqm.conformers import normalize, rmsd_cluster
from lefqm.commandline_calculation import conformator_generate


class ConformersTests(unittest.TestCase):
    """Conformers tests"""

    def test_normalize(self):
        """Test molecule normalization"""
        mol = Chem.MolFromSmiles("C(C([O-])=O)c1nc(cs1)C(F)(F)F.[Na+] Z2489380833")
        mol = normalize(mol)
        self.assertEqual(
            Chem.MolToSmiles(mol), "O=C(O)Cc1nc(C(F)(F)F)cs1"
        )  # Z2489380833 without sodium

        mol = Chem.MolFromSmiles("C1CN(CC1O)c1nc2c(cccc2o1)F Z1900496514")
        mol = normalize(mol)
        self.assertEqual(
            Chem.MolToSmiles(mol), "O[C@@H]1CCN(c2nc3c(F)cccc3o2)C1"
        )  # Z1900496514 with stereo

        mol = Chem.MolFromSmiles("O=S(=O)(C1CC1)N1CCC[C@@H]1C(F)F Z2070069886")
        self.assertRaises(RuntimeError, normalize, mol)  # unassigned diastereomer

    def test_rmsd_cluster(self):
        """Test rmsd clustering"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = conformator_generate(mol)
        nof_conformations = mol.GetNumConformers()
        # fake energies
        conformation_energies = range(nof_conformations)
        conformation_indexes = rmsd_cluster(mol, conformation_energies, prune_threshold=0.5)
        self.assertEqual(len(conformation_indexes), 15)
        # ensure we are not changing the original conformers
        self.assertEqual(mol.GetNumConformers(), nof_conformations)
