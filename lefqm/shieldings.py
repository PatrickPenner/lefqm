"""Shielding generation"""
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter

from lefqm import constants, utils
from lefqm.commandline_calculation import ShieldingCalculation


def add_shieldings_subparser(subparsers):
    """Add shieldings arguments as a subparser"""
    shieldings_parser = subparsers.add_parser("shieldings")
    shieldings_parser.add_argument("input", type=Path, help="Input SD file")
    shieldings_parser.add_argument("output", type=Path, help="Output SD file with shieldings")
    shieldings_parser.add_argument(
        "--cores",
        type=int,
        help="Maximum number of cores to use. Defaults to turbomole behavior",
    )
    shieldings_parser.add_argument(
        "-v", "--verbose", help="show verbose output", action="store_true"
    )
    shieldings_parser.add_argument(
        "--config",
        type=Path,
        help="Config file to read from",
        default=constants.DEFAULT_CONFIG,
    )


def shieldings(args):
    """Shielding generation"""
    config = utils.get_config(args.config)
    config["Parameters"]["cores"] = str(args.cores)

    writer = SDWriter(str(args.output))
    writer.SetForceV3000(True)
    for index, mol in enumerate(SDMolSupplier(str(args.input), removeHs=False)):
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(index)
        try:
            logging.info("Processing molecule %s", mol_name)
            shielding_constants = ShieldingCalculation(config).run(mol)
            for atom, shielding in zip(mol.GetAtoms(), shielding_constants):
                atom.SetDoubleProp(constants.SHIELDING_SD_PROPERTY, shielding)
            Chem.CreateAtomDoublePropertyList(mol, constants.SHIELDING_SD_PROPERTY)
            writer.write(mol)
        except Exception as exception:
            logging.warning("Molecule %s failed with error: %s", mol_name, exception)
    logging.info("done")
