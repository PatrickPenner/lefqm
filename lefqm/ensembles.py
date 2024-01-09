"""Process QM ensemble fluorine shielding calculations"""
import logging
from pathlib import Path

from lefshift.utils import map_atom_indexes
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SmilesParserParams

from lefqm import constants, utils


def add_ensembles_subparser(subparsers):
    """Add ensembles arguments as a subparser"""
    ensemble_parser = subparsers.add_parser("ensembles")
    ensemble_parser.add_argument(
        "input", help="Input directory of ensemble SDFs with QM properties", type=Path
    )
    ensemble_parser.add_argument("output", help="Output shielding CSV", type=Path)
    ensemble_parser.add_argument(
        "--lowest",
        action="store_true",
        help="Use only the lowest energy conformation instead of the ensemble",
    )
    ensemble_parser.add_argument(
        "--energy-property",
        type=str,
        help="Property that contains the energy of the conformer in water",
        default=constants.ENERGY_SD_PROPERTY,
    )
    ensemble_parser.add_argument(
        "--shielding-property",
        type=str,
        help="Property that contains the shieldings of the conformer",
        default=constants.SHIELDING_SD_PROPERTY,
    )
    ensemble_parser.add_argument(
        "-v", "--verbose", help="show verbose output", action="store_true"
    )


def make_output(calculated_shieldings):
    """Make output from ensemble shieldings

    :param calculated_shieldings: molecule and shielding pairs to write
    :type calculated_shieldings: list[rdkit.Chem.rdchem.Mol, dict]
    :return: output to write
    :rtype: pd.DataFrame
    """
    lines = []
    params = SmilesParserParams()
    params.removeHs = False
    for mol, shieldings in calculated_shieldings:
        name = mol.GetProp("_Name")
        # Prepare mapping atom indices to the way the molecule will be written
        # as smiles. The input was SDF so the atom order will change.
        smiles = Chem.MolToSmiles(mol)
        canon_mol = Chem.MolFromSmiles(smiles, params)
        atom_index_map = map_atom_indexes(mol, canon_mol)
        for key, shielding in shieldings.items():
            if isinstance(key, tuple):
                if len(key) == 2:
                    label = "CF2"
                elif len(key) == 3:
                    label = "CF3"
                else:
                    raise RuntimeError("Invalid fluorine group")
                # pick the first atom index as a representative
                key = key[0]
            else:
                label = "CF"

            lines.append([name, smiles, label, atom_index_map[key], shielding])
    return pd.DataFrame(
        lines,
        columns=[
            constants.ID_COLUMN,
            constants.SMILES_COLUMN,
            constants.LABEL_COLUMN,
            constants.ATOM_INDEX_COLUMN,
            constants.SHIELDING_COLUMN,
        ],
    )


def ensembles(args):
    """Process QM ensemble fluorine shielding calculations"""
    calculated_shieldings = []
    files = []
    if args.input.is_dir():
        files = sorted(args.input.glob("*.sdf"))
    else:
        files.append(args.input)
    logging.info("Combining ensembles from %s file(s)", len(files))

    for ensemble_path in files:
        ensemble = list(SDMolSupplier(str(ensemble_path), removeHs=False))
        if len(ensemble) == 0:
            raise RuntimeError(f"No molecules in {ensemble_path}")
        ensemble = [
            mol
            for mol in ensemble
            if mol.HasProp(args.energy_property)
            and mol.HasProp("atom.dprop." + args.shielding_property)
        ]
        if len(ensemble) == 0:
            raise RuntimeError(f"No molecules with shieldings in {ensemble_path}")

        if any(not mol.HasProp("_Name") for mol in ensemble):
            logging.warning(
                'Some molecules in "%s" did not have names. This ensemble will be output using the filename: "%s"',
                ensemble_path,
                ensemble_path.stem,
            )
            for mol in ensemble:
                mol.SetProp("_Name", ensemble_path.stem)
        if args.lowest:
            fluorine_shieldings = utils.get_lowest_energy_fluorine_shieldings(ensemble)
        else:
            fluorine_shieldings = utils.get_ensemble_fluorine_shieldings(
                ensemble,
                water_energy_property=args.energy_property,
                shielding_property=args.shielding_property,
            )
        if not fluorine_shieldings:
            raise RuntimeError("No shieldings")
        calculated_shieldings.append((ensemble[0], fluorine_shieldings))

    output = make_output(calculated_shieldings)
    output.to_csv(args.output, index=False)
