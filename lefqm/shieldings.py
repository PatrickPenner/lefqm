"""Shielding generation"""
import configparser
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SDWriter, rdmolops

from lefqm import constants

X2T = "x2t"
RIDFT = "ridft"
MPSHIFT = "mpshift"

TURBOMOLE_CONTROL = """
$atoms
basis=def2-TZVP
$coord file=coord
$symmetry c1
$eht charge={charge}
$dft
functional xcfun set-gga
functional xcfun kt3 1.0
gridsize 3
$rij
$marij
$cosmo
  epsilon=78.39
$nmr dft shielding constants file=shielding
$end
"""


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
        default=Path(__file__).absolute().parent / "config.ini",
    )


class ShieldingCalculation:
    """NMR shielding constant calculation"""

    def __init__(self, config, run_dir_path=None):
        self.config = config
        self.run_dir_path = run_dir_path
        self.tmp_dir = None
        if self.run_dir_path is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.run_dir_path = Path(self.tmp_dir.name)

    def __del__(self):
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def run(self, mol):
        """Run shielding calculation

        :param mol: molecule to run the calcualtion on
        :type mol: rdkit.Chem.rdchem.Mol
        :return: shieldings for every atom
        :rtype: list[float]
        """
        if self.config["qm_method"] == "turbomole":
            return self.turbomole_calculate_shieldings(
                mol,
                x2t=self.config["x2t"],
                ridft=self.config["ridft"],
                mpshift=self.config["mpshift"],
                cores=self.config["cores"],
                run_dir_path=self.run_dir_path,
            )
        raise RuntimeError("Invalid QM method")

    @staticmethod
    def turbomole_calculate_shieldings(
        mol,
        x2t="x2t",
        ridft="ridft",
        mpshift="mpshift",
        cores=None,
        run_dir_path=None,
    ):
        """Calculate shieldings for a molecule

        :param mol: molecule with conformations to optimize
        :type mol: rdkit.Chem.rdchem.Mol
        :param x2t: path/call to x2t
        :type x2t: str
        :param ridft: path/call to ridft
        :type ridft: str
        :param mpshift: path/call to mpshift
        :type mpshift: str
        :param cores: maximum number of cores to use
        :type cores: int
        :return: shieldings for every atom
        :rtype: list[float]
        """
        if not shutil.which(x2t):
            raise RuntimeError(f"Cannot find {x2t}")
        if not shutil.which(ridft):
            raise RuntimeError(f"Cannot find {ridft}")
        if not shutil.which(mpshift):
            raise RuntimeError(f"Cannot find {mpshift}")

        tmp_dir = None
        if run_dir_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            run_dir_path = Path(tmp_dir.name)
        logging.debug("Calculating shieldings in %s", run_dir_path)

        xyz_path = run_dir_path / "input.xyz"
        with open(xyz_path, "w", encoding="utf8") as xyz_file:
            xyz_file.write(Chem.MolToXYZBlock(mol))

        command = [x2t, str(xyz_path)]
        logging.debug(" ".join(command))
        coord_output = subprocess.check_output(command, cwd=run_dir_path).decode("ascii")
        coord_path = run_dir_path / "coord"
        with open(coord_path, "w", encoding="utf8") as coord_file:
            coord_file.write(coord_output)

        control_path = run_dir_path / "control"
        with open(control_path, "w", encoding="utf8") as control_file:
            control_file.write(TURBOMOLE_CONTROL.format(charge=rdmolops.GetFormalCharge(mol)))

        ridft_log_path = run_dir_path / "ridft.log"
        with open(ridft_log_path, "w", encoding="utf8") as ridft_log_file:
            ridft_command = [ridft]
            if cores is not None:
                ridft_command.extend(["-nthreads", str(cores)])
            logging.debug(" ".join(ridft_command))
            subprocess.check_call(
                ridft_command,
                cwd=run_dir_path,
                stdout=ridft_log_file,
                stderr=subprocess.STDOUT,
            )

        with open(ridft_log_path, encoding="utf8") as ridft_log_file:
            lines = list(ridft_log_file.readlines())
            if ": all done  ****" not in lines[-6] or "did not converge!" in lines[-15]:
                logging.debug("\n".join(lines))
                raise RuntimeError("ridft calculation did not converge")

        mpshift_log_path = run_dir_path / "mpshift.log"
        with open(mpshift_log_path, "w", encoding="utf8") as mpshift_log_file:
            mpshift_command = [mpshift]
            if cores is not None:
                mpshift_command.extend(["-nthreads", str(cores)])
            logging.debug(" ".join(mpshift_command))
            subprocess.check_call(
                mpshift_command,
                cwd=run_dir_path,
                stdout=mpshift_log_file,
                stderr=subprocess.STDOUT,
            )

        with open(mpshift_log_path, encoding="utf8") as mpshift_log_file:
            lines = list(mpshift_log_file.readlines())
            if ": all done  ****" not in lines[-6]:
                logging.debug("\n".join(lines))
                raise RuntimeError("mpshift calculation did not converge")

        with open(run_dir_path / "shielding", encoding="utf8") as shieldings_file:
            lines = shieldings_file.readlines()

        shielding_constants = []
        # skip header line
        for line in lines[1:]:
            if "$end" in line:
                break
            if line.strip()[0] == "#":
                continue
            # [NO., TYPE, MULT., ISOTROPIC, ANISOTROPIC, dD/dB-CONTRIBUTION]
            tokens = [token.strip() for token in line.split(" ") if token.strip()]
            atom_name = tokens[1]
            if "*" in atom_name:
                atom_name = atom_name.replace("*", "")
            assert (
                mol.GetAtomWithIdx(int(tokens[0]) - 1).GetSymbol().lower() == atom_name
            ), "Order of atoms must be the same"
            shielding_constants.append(float(tokens[3]))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return shielding_constants


def shieldings(args):
    """Shielding generation"""
    config = configparser.ConfigParser()
    config.read(args.config)
    config = {**config._sections["Paths"], **config._sections["Workflow"]}
    config["cores"] = args.cores
    writer = SDWriter(str(args.output))
    for index, mol in enumerate(SDMolSupplier(str(args.input), removeHs=False)):
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(index)
        try:
            logging.info("Processing molecule %s", mol_name)
            shielding_constants = ShieldingCalculation.turbomole_calculate_shieldings(
                mol, cores=args.cores
            )
            shielding_constants = ShieldingCalculation(config).run(mol)
            for atom, shielding in zip(mol.GetAtoms(), shielding_constants):
                atom.SetDoubleProp(constants.SHIELDING_SD_PROPERTY, shielding)
            Chem.CreateAtomDoublePropertyList(mol, constants.SHIELDING_SD_PROPERTY)
            writer.write(mol)
        except Exception as exception:
            logging.warning("Molecule %s failed with error: %s", mol_name, exception)
    logging.info("done")
