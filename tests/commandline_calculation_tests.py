"""Test commandline calculations"""
import math
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolAlign, SDMolSupplier
import numpy as np

from lefqm import constants, utils
from lefqm.commandline_calculation import (
    moka_protonate,
    conformator_generate,
    omega_generate,
    xtb_optimize_conformation,
    xtb_optimize,
    turbomole_calculate_shieldings,
    ConformerGeneration,
    ShieldingCalculation,
)


class CommandlineCalculationTests(unittest.TestCase):
    """Test commandline calculation functions"""

    def test_moka_protonate(self):
        """Test tautomer/protomer generation"""
        mol = Chem.MolFromSmiles("O=S(=O)(C1CC1)N1CCC[C@@H]1C(F)F Z2070069886")
        mol = moka_protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), 0)

        mol = Chem.MolFromSmiles("C1CN(C[C@H]1N)S(c1ccc(cc1)C(F)(F)F)(=O)=O Z1480871269")
        mol = moka_protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), 1)

        mol = Chem.MolFromSmiles("Cc1cc(C(NC2(CC2)C(O)=O)=O)c(cc1F)[Cl] Z1603608423")
        mol = moka_protonate(mol)
        self.assertEqual(rdmolops.GetFormalCharge(mol), -1)

    def test_conformator_generate(self):
        """Test conformer generation"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = conformator_generate(mol)
        self.assertEqual(mol.GetNumConformers(), 159)

    def test_omega_generate(self):
        """Test omega conformer generation"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = omega_generate(mol)
        self.assertEqual(mol.GetNumConformers(), 250)

    def test_xtb_optimize_conformation(self):
        """Test optimizing a single conformation"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = conformator_generate(mol)
        for i in range(1, mol.GetNumConformers()):
            mol.RemoveConformer(i)
        optimized_conformation, energy = xtb_optimize_conformation(mol, 0, cores=1)
        mol.AddConformer(optimized_conformation, assignId=True)

        # direct atom index mapping rmsd
        self.assertGreater(
            rdMolAlign.CalcRMS(mol, mol, prbId=0, refId=1),
            0.5,
        )
        self.assertAlmostEqual(energy, -45.455410995927, places=3)  # energy is still in hartrees

    def test_xtb_optimize(self):
        """Test optimization by XTB"""
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")
        mol = conformator_generate(mol)
        conformation_indexes = [0, 1]
        optimized_mol, energies = xtb_optimize(
            mol, conformation_indexes=conformation_indexes, cores=1
        )

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

    def test_turbomole_calculate_shieldings(self):
        """Test shielidng generation with turbomole"""
        mol = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))[0]
        shieldings = turbomole_calculate_shieldings(mol, cores=1)
        for index, atom in enumerate(mol.GetAtoms()):
            self.assertAlmostEqual(
                shieldings[index], atom.GetDoubleProp(constants.SHIELDING_SD_PROPERTY)
            )


class ConformerGenerationTests(unittest.TestCase):
    """Conformer generation tests"""

    def test_run(self):
        """Test running conformer generation"""
        config_path = Path(__file__).absolute().parent.parent / "lefqm" / "config.ini"
        config = utils.config_to_dict(config_path)
        config["cores"] = 1
        mol = Chem.MolFromSmiles("C1CC1NC(c1ccc(CO)c(c1)F)=O Z1915979114")

        config["confgen_method"] = "conformator"
        mol_with_confs = ConformerGeneration(config).run(mol)
        self.assertEqual(mol_with_confs.GetNumConformers(), 159)

        config["confgen_method"] = "rdkit"
        mol_with_confs = ConformerGeneration(config).run(mol)
        self.assertEqual(mol_with_confs.GetNumConformers(), 250)

        config["confgen_method"] = "omega"
        mol_with_confs = ConformerGeneration(config).run(mol)
        self.assertEqual(mol_with_confs.GetNumConformers(), 250)


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
