"""NWChem interface functions"""
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

NWCHEM_TEMPLATE = """
title "Shieldings"
memory global 8 gb
charge {charge}
geometry units angstroms
{xyz_string}
end
basis
 * library Def2-TZVP
end
property
 shielding
end
dft
 xc GGA_XC_KT3 1.0
end
cosmo
  solvent water
end
task dft property"""


SHIELDING_PATTERN = re.compile(r"\s*isotropic\s*=\s*((\-|\+)?[0-9]+\.[0-9]+)")


def nwchem_read_isotropic_shieldings(log_path):
    """Read isotropic shielding from nwchem log file

    :param log_path: path to the nwchem log file
    :type log_path: pathlib.Path
    :return: isotropic shielding constants
    :rtype: list[float]
    """
    isotropic_shielding_constants = []
    with open(log_path, encoding="utf8") as log_file:
        for line in log_file.readlines():
            if "isotropic" not in line:
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


def nwchem_calculate_shieldings(mol, nwchem="nwchem", run_dir_path=None):
    """Calculate nwchem shieldings for a molecule

    :param mol: molecule to calculate shieldings for
    :type mol: rdkit.Chem.rdchem.Mol
    :param nwchem: path/call to nwchem
    :type nwchem: str
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: shieldings for every atom
    :rtype: list[float]
    """
    if not shutil.which(nwchem):
        raise RuntimeError(f"Cannot find {nwchem}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)
    logging.debug("Calculating shieldings in %s", run_dir_path)

    xyz_string = "\n".join(Chem.MolToXYZBlock(mol).split("\n")[2:]).strip()
    input_string = NWCHEM_TEMPLATE.format(
        charge=rdmolops.GetFormalCharge(mol), xyz_string=xyz_string
    )
    input_name = "shielding.nw"
    with open(run_dir_path / input_name, "w", encoding="utf8") as input_file:
        input_file.write(input_string)

    args = [nwchem, input_name]
    logging.debug(" ".join(args))
    log_path = run_dir_path / "shielding.log"
    try:
        with open(log_path, "w", encoding="utf8") as log_file:
            subprocess.check_call(
                args, stderr=subprocess.STDOUT, stdout=log_file, cwd=run_dir_path
            )
    except subprocess.CalledProcessError:
        logging.info(open(log_path, encoding="utf8").read())
        raise

    shielding_constants = nwchem_read_isotropic_shieldings(log_path)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return shielding_constants
