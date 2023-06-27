"""Test QM utils"""
import math
import unittest

from rdkit.Chem import SDMolSupplier

from lefqm import utils


class UtilsTests(unittest.TestCase):
    """Test QM utils"""

    def test_get_fluorine_shieldings(self):
        """Test fluorine shielding extraction"""
        mols = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))
        fluorine_shieldings = utils.get_fluorine_shieldings(mols[0])
        self.assertEqual(len(fluorine_shieldings), 1)
        self.assertEqual(list(fluorine_shieldings.keys()), [12])

        fluorine_shieldings = utils.get_fluorine_shieldings(mols[1])
        self.assertEqual(len(fluorine_shieldings), 1)
        # keys for CF2 groups are tuples of length 2
        self.assertEqual(list(fluorine_shieldings.keys()), [(5, 6)])

        fluorine_shieldings = utils.get_fluorine_shieldings(mols[2])
        self.assertEqual(len(fluorine_shieldings), 1)
        # keys for CF3 groups are tuples of length 3
        self.assertEqual(list(fluorine_shieldings.keys()), [(5, 6, 7)])

    def test_get_boltzmann_weights(self):
        """Test Boltzmann weighting by energies"""
        mols = list(SDMolSupplier("tests/data/ensemble_with_shieldings.sdf", removeHs=False))
        energies = [float(mol.GetProp("energy in water (kcal/mol)")) for mol in mols]
        expected_weights = [
            0.14802857142009487,
            0.14759386890693613,
            0.14725465946782382,
            0.033812788349725974,
            0.0337470774054945,
            0.03370119038916379,
            0.03375554006894755,
            0.03347823734962448,
            0.03339913032858618,
            0.11782472728010801,
            0.1171065855838778,
            0.11669893970445261,
            0.003598683745164377,
        ]
        weights = utils.get_boltzmann_weights(energies)
        self.assertTrue(
            all(
                math.isclose(weight, expected)
                for weight, expected in zip(weights, expected_weights)
            )
        )

        # takes both lists and numpy arrays
        weights = utils.get_boltzmann_weights(energies)
        self.assertTrue(
            all(
                math.isclose(weight, expected)
                for weight, expected in zip(weights, expected_weights)
            )
        )

        # checks for empty energies
        with self.assertRaises(RuntimeError):
            weights = utils.get_boltzmann_weights([])

    def test_get_weighted_average_ensemble_shieldings(self):
        """Test weighted averaging of shieldings in an ensemble"""
        mols = list(SDMolSupplier("tests/data/ensemble_with_shieldings.sdf", removeHs=False))
        energies = [float(mol.GetProp("energy in water (kcal/mol)")) for mol in mols]
        weights = utils.get_boltzmann_weights(energies)
        ensemble_fluorine_shieldings = [utils.get_fluorine_shieldings(mol) for mol in mols]
        averaged_shieldings = utils.get_weighted_average_ensemble_shieldings(
            ensemble_fluorine_shieldings, weights
        )
        self.assertEqual(averaged_shieldings, {12: 291.0998977214819})

    def test_match_shieldings(self):
        """Test matching of shieldings in an ensemble"""
        mols = list(
            SDMolSupplier("tests/data/mismatched_ensemble_with_shieldings.sdf", removeHs=False)
        )
        shieldings = [utils.get_fluorine_shieldings(mol) for mol in mols]
        matched_shieldings = utils.match_shieldings(shieldings, mols)
        self.assertTrue(
            all(tuple(shielding.keys()) == ((5, 6, 7),) for shielding in matched_shieldings)
        )

    def test_get_ensemble_fluorine_shieldings(self):
        """Test getting the fluorine shieldings for an ensemble"""
        mols = list(SDMolSupplier("tests/data/ensemble_with_shieldings.sdf", removeHs=False))
        ensemble_fluorine_shieldings = utils.get_ensemble_fluorine_shieldings(mols)
        self.assertEqual(ensemble_fluorine_shieldings, {12: 291.0998977214819})

    def test_get_averaged_shielding(self):
        """Test get averaged shieldings"""
        shieldings = {(16, 17, 18): 233.67539121192218, (10, 11, 12): 233.55704916283617}
        averaged_shielding = {(10, 11, 12): 233.61622018737916}
        self.assertEqual(averaged_shielding, utils.get_averaged_shielding(shieldings))
