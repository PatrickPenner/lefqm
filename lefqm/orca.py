"""Orca interface functions"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

ORCA_TEMPLATE = """
! DFT DEF2-TZVP CPCM(water) NMR
%method
   Functional gga_xc_kt3
end
{pal_block}
* xyz {charge} 1
{xyz_string}
*

"""

SHIELDING_PATTERN = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")

ORCA_SUCCESS_MARKER = "****ORCA TERMINATED NORMALLY****"


def orca_read_isotropic_shieldings(log_path):
    """Read isotropic shielding from orca log file

    :param log_path: path to the orca log file
    :type log_path: pathlib.Path
    :return: isotropic shielding constants
    :rtype: list[float]
    """
    isotropic_shielding_constants = []
    in_summary = False
    started = False
    with open(log_path, encoding="utf8") as log_file:
        for line in log_file.readlines():
            if "CHEMICAL SHIELDING SUMMARY (ppm)" in line:
                in_summary = True
                continue
            if not in_summary:
                continue

            shielding_match = SHIELDING_PATTERN.match(line)
            if shielding_match is None:
                if started:
                    break
                continue

            started = True
            isotropic_shielding_constants.append(float(shielding_match.group(3)))
    return isotropic_shielding_constants


def _orca_environment(orca):
    """Environment with the orca directory prepended to PATH

    The orca driver launches parallel executables from its own directory.
    """
    environment = os.environ.copy()
    orca_directory = os.path.dirname(os.path.abspath(orca))
    environment["PATH"] = orca_directory + os.pathsep + environment["PATH"]
    return environment


def orca_calculate_shieldings(mol, orca="orca", cores=None, run_dir_path=None, precision=10):
    """Calculate orca shieldings for a molecule

    :param mol: molecule to calculate shieldings for
    :type mol: rdkit.Chem.rdchem.Mol
    :param orca: path/call to orca
    :type orca: str
    :param cores: number of cores to use
    :type cores: int
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :param precision: coordinate precision for the input molecule
    :type precision: int
    :return: shieldings for every atom
    :rtype: list[float]
    """
    if not shutil.which(orca):
        raise RuntimeError(f"Cannot find {orca}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)
    logging.debug("Calculating shieldings in %s", run_dir_path)

    try:
        xyz_string = "\n".join(Chem.MolToXYZBlock(mol, precision=precision).split("\n")[2:]).strip()
        pal_block = f"%pal\n   nprocs {cores}\nend\n" if cores is not None else ""
        input_string = ORCA_TEMPLATE.format(
            charge=rdmolops.GetFormalCharge(mol),
            xyz_string=xyz_string,
            pal_block=pal_block,
        )
        input_name = "shielding.inp"
        with open(run_dir_path / input_name, "w", encoding="utf8") as input_file:
            input_file.write(input_string)

        args = [orca, input_name]
        logging.debug(" ".join(args))
        log_path = run_dir_path / "shielding.log"
        try:
            with open(log_path, "w", encoding="utf8") as log_file:
                subprocess.check_call(
                    args,
                    stderr=subprocess.STDOUT,
                    stdout=log_file,
                    cwd=run_dir_path,
                    env=_orca_environment(orca),
                )
        except subprocess.CalledProcessError as error:
            logging.info(open(log_path, encoding="utf8").read())
            raise RuntimeError(f"Orca calculation failed: {error}") from error

        with open(log_path, encoding="utf8") as log_file:
            log_content = log_file.read()
        if ORCA_SUCCESS_MARKER not in log_content:
            raise RuntimeError(
                f"Orca calculation did not terminate normally in {run_dir_path}\n"
                f"{log_content[-2000:]}"
            )

        shielding_constants = orca_read_isotropic_shieldings(log_path)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    return shielding_constants
