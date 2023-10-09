"""Test LEFQM main"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter, SmilesParserParams

from lefqm import constants
from lefqm.__main__ import get_args, call_subtool


class MainTests(unittest.TestCase):
    """Test LEFQM main"""

    def test_conformers_smiles(self):
        """Test conformers"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/mols.smi"
            output_path = tmp_path / "Z1188305542.sdf"
            args = get_args(
                [
                    "conformers",
                    str(input_path),
                    str(tmp_path),
                    # only do one molecule
                    "--chunk",
                    str(0),
                    "--chunk-size",
                    str(1),
                    "--cores",
                    str(1),
                ]
            )
            call_subtool(args)
            mols = list(SDMolSupplier(str(output_path)))
            self.assertGreater(len(mols), 0)
            self.assertEqual(len({mol.GetProp("_Name") for mol in mols}), 1)

    def test_conformers_csv(self):
        """Test conformers"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/mols.csv"
            output_path = tmp_path / "Z1188305542.sdf"
            args = get_args(
                [
                    "conformers",
                    str(input_path),
                    str(tmp_path),
                    # only do one molecule
                    "--chunk",
                    str(0),
                    "--chunk-size",
                    str(1),
                    "--cores",
                    str(1),
                ]
            )
            call_subtool(args)
            mols = list(SDMolSupplier(str(output_path)))
            self.assertGreater(len(mols), 0)
            self.assertEqual(len({mol.GetProp("_Name") for mol in mols}), 1)

    def test_shieldings(self):
        """Test shieldings"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mol = list(SDMolSupplier("tests/data/TFA_shieldings.sdf", removeHs=False))[0]
            input_path = tmp_path / "input.sdf"
            output_path = tmp_path / "output_shieldings.sdf"
            writer = SDWriter(str(input_path))
            writer.write(mol)
            writer.close()
            args = get_args(
                [
                    "shieldings",
                    str(input_path),
                    str(output_path),
                    "--cores",
                    str(1),
                ]
            )
            call_subtool(args)
            mol = list(SDMolSupplier(str(output_path), removeHs=False))[0]
            self.assertTrue(mol.HasProp("atom.dprop." + constants.SHIELDING_SD_PROPERTY))
            for atom in mol.GetAtoms():
                self.assertIsNotNone(atom.GetDoubleProp(constants.SHIELDING_SD_PROPERTY))

    def test_ensembles(self):
        """Test ensemble summarization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input"
            input_path.mkdir()
            mols = list(SDMolSupplier("tests/data/mols_with_shieldings.sdf", removeHs=False))
            for mol in mols:
                writer = SDWriter(str(input_path / (mol.GetProp("_Name") + ".sdf")))
                writer.write(mol)
                writer.close()
            output_path = tmp_path / "shieldings.csv"
            args = get_args(
                [
                    "ensembles",
                    str(input_path),
                    str(output_path),
                ]
            )
            call_subtool(args)
            shieldings = pd.read_csv(output_path)
            self.assertEqual(len(mols), len(shieldings))

            params = SmilesParserParams()
            params.removeHs = False
            for _, row in shieldings.iterrows():
                mol = Chem.MolFromSmiles(row["SMILES"], params)
                self.assertEqual(mol.GetAtomWithIdx(row["Atom Index"]).GetSymbol(), "F")

    def test_shifts(self):
        """Test shift conversion"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/shieldings.csv"
            calibration = "tests/data/shifts.csv"
            output_path = tmp_path / "shifts.csv"
            args = get_args(
                [
                    "shifts",
                    str(input_path),
                    "--calibration",
                    str(calibration),
                    str(output_path),
                ]
            )
            call_subtool(args)
            expected_shifts = pd.read_csv(calibration)
            shifts = pd.read_csv(output_path)
            for i in range(len(shifts)):
                self.assertAlmostEqual(
                    shifts[constants.SHIFT_COLUMN].values[i],
                    expected_shifts[constants.SHIFT_COLUMN].values[i],
                )
