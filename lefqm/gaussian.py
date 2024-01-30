"""Gaussian interface functions"""
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

GAUSSIAN_TEMPLATE = """
# B3LYP/cc-pVDZ NMR SCRF=SMD

Shieldings

{charge} 1
{xyz_string}

"""

SHIELDING_PATTERN = re.compile(r"\s*[0-9]+\s*[a-zA-Z]+\s*Isotropic\s*=\s*((\-|\+)?[0-9]+\.[0-9]+)")


def gaussian_read_isotropic_shieldings(log_path):
    """Read isotropic shielding from guassian log file

    :param log_path: path to the guassian log file
    :type log_apth: pathlib.Path
    :return: isotropic shielding constants
    :rtype: list[(int,str,float)]
    """
    isotropic_shielding_constants = []
    with open(log_path, encoding="utf8") as log_file:
        for line in log_file.readlines():
            if "Isotropic" not in line:
                continue

            shielding_match = SHIELDING_PATTERN.match(line)
            if shielding_match is None:
                continue

            value = shielding_match.group(1)
            try:
                value = float(value)
            except ValueError as error:
                logging.warning(error)
                value = None
            isotropic_shielding_constants.append(value)
    return isotropic_shielding_constants


def gaussian_calculate_shieldings(mol, gaussian="g16", run_dir_path=None):
    """Calculate gaussian shieldings for a molecule

    :param mol: molecule to calculate shieldings for
    :type mol: rdkit.Chem.rdchem.Mol
    :param gaussian: path/call to gaussian
    :type gaussian: str
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: shieldings for every atom
    :rtype: list[float]
    """
    if not shutil.which(gaussian):
        raise RuntimeError(f"Cannot find {gaussian}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)
    logging.debug("Calculating shieldings in %s", run_dir_path)

    xyz_string = "\n".join(Chem.MolToXYZBlock(mol).split("\n")[2:]).strip()
    input_string = GAUSSIAN_TEMPLATE.format(
        charge=rdmolops.GetFormalCharge(mol), xyz_string=xyz_string
    )
    input_name = "shielding.com"
    with open(run_dir_path / input_name, "w", encoding="utf8") as input_file:
        input_file.write(input_string)

    args = [gaussian, input_name]
    logging.debug(" ".join(args))
    log_path = run_dir_path / "shielding.log"
    try:
        subprocess.check_call(args, cwd=run_dir_path)
    except subprocess.CalledProcessError:
        logging.info(open(log_path, encoding="utf8").read())
        raise

    shielding_constants = gaussian_read_isotropic_shieldings(log_path)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return shielding_constants
