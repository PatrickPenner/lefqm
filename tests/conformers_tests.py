"""Conformer generation tests"""
import math
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdmolops

from lefqm import utils
from lefqm.conformers import (
    ConformerGeneration,
    normalize,
    optimize,
    optimize_conformation,
    protonate,
    rmsd_cluster,
)


class ConformerGenerationTests(unittest.TestCase):
    """Conformer generation tests"""

    def test_run(self):
        """Test running conformer generation"""
        config_path = Path(__file__).absolute().parent.parent / "lefqm" / "config.ini"
        config = utils.config_to_dict(config_path)
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = ConformerGeneration(config).run(mol)
        self.assertEqual(mol.GetNumConformers(), 159)

    def test_conformator_generate(self):
        """Test conformer generation"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = ConformerGeneration.conformator_generate(mol)
        self.assertEqual(mol.GetNumConformers(), 159)


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

    def test_protonate(self):
        """Test tautomer/protomer generation"""
        mol = Chem.MolFromSmiles("O=S(=O)(C1CC1)N1CCC[C@@H]1C(F)F Z2070069886")
        mol = protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), 0)

        mol = Chem.MolFromSmiles("C1CN(C[C@H]1N)S(c1ccc(cc1)C(F)(F)F)(=O)=O Z1480871269")
        mol = protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), 1)

        mol = Chem.MolFromSmiles("Cc1cc(C(NC2(CC2)C(O)=O)=O)c(cc1F)[Cl] Z1603608423")
        mol = protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), -1)

    def test_rmsd_cluster(self):
        """Test rmsd clustering"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = ConformerGeneration.conformator_generate(mol)
        nof_conformations = mol.GetNumConformers()
        # fake energies
        conformation_energies = range(nof_conformations)
        conformation_indexes = rmsd_cluster(mol, conformation_energies, prune_threshold=0.5)
        self.assertEqual(len(conformation_indexes), 15)
        # ensure we are not changing the original conformers
        self.assertEqual(mol.GetNumConformers(), nof_conformations)

    def test_optimize_conformation(self):
        """Test optimizing a single conformation"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = ConformerGeneration.conformator_generate(mol)
        for i in range(1, mol.GetNumConformers()):
            mol.RemoveConformer(i)
        optimized_conformation, energy = optimize_conformation(mol, 0, cores=1)
        mol.AddConformer(optimized_conformation, assignId=True)

        # direct atom index mapping rmsd
        self.assertGreater(
            rdMolAlign.CalcRMS(mol, mol, prbId=0, refId=1),
            0.5,
        )
        self.assertAlmostEqual(energy, -45.455410995927, places=3)  # energy is still in hartrees

    def test_optimize(self):
        """Test optimization by XTB"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = ConformerGeneration.conformator_generate(mol)
        conformation_indexes = [0, 1]
        optimized_mol, energies = optimize(mol, conformation_indexes, cores=1)

        self.assertEqual(optimized_mol.GetNumConformers(), len(conformation_indexes))
        for index, conformation_index in enumerate(conformation_indexes):
            coordinates = mol.GetConformer(conformation_index).GetPositions()
            optimized_coordinates = optimized_mol.GetConformer(index).GetPositions()
            # direct atom index mapping in-place rmsd
            rmsd = math.sqrt(
                np.sum(np.power(coordinates - optimized_coordinates, 2)) / len(coordinates)
            )
            self.assertGreater(rmsd, 0)

        expected_energies = [-28523.70106556896, -28523.70080880777]
        self.assertTrue(
            all(
                math.isclose(energy, expected, abs_tol=1e-3)
                for energy, expected in zip(energies, expected_energies)
            )
        )
