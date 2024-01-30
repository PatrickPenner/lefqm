"""Turbomole interface functions"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

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


def x2t_convert_mol(mol, x2t="x2t", run_dir_path=None):
    """Convert mol to turbomole coord input

    :param mol: molecule to convert
    :type mol: rdkit.Chem.rdchem.Mol
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: turbomole coord input
    :rtype: str
    """
    if not shutil.which(x2t):
        raise RuntimeError(f"Cannot find {x2t}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)
    logging.debug("Converting mol in %s", run_dir_path)

    xyz_path = run_dir_path / "input.xyz"
    with open(xyz_path, "w", encoding="utf8") as xyz_file:
        xyz_file.write(Chem.MolToXYZBlock(mol))

    command = [x2t, str(xyz_path)]
    logging.debug(" ".join(command))
    return subprocess.check_output(command, cwd=run_dir_path).decode("ascii")


def turbomole_read_isotropic_shieldings(shieldings_path):
    """Read isotropic shieldings from turbomole shielding file

    :param shieldings_path: path to the turbomole shielding file
    :type shieldings_path: pathlib.Path
    :return: isotropic shielding constants
    :rtype: list[float]
    """
    with open(shieldings_path, encoding="utf8") as shieldings_file:
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
        shielding_constants.append(float(tokens[3]))
    return shielding_constants


def turbomole_calculate_shieldings(
    mol,
    x2t="x2t",
    ridft="ridft",
    mpshift="mpshift",
    cores=None,
    run_dir_path=None,
):
    """Calculate shieldings for a molecule

    :param mol: molecule to calculate shieldings for
    :type mol: rdkit.Chem.rdchem.Mol
    :param x2t: path/call to x2t
    :type x2t: str
    :param ridft: path/call to ridft
    :type ridft: str
    :param mpshift: path/call to mpshift
    :type mpshift: str
    :param cores: maximum number of cores to use
    :type cores: int
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: shieldings for every atom
    :rtype: list[float]
    """
    if not shutil.which(ridft):
        raise RuntimeError(f"Cannot find {ridft}")
    if not shutil.which(mpshift):
        raise RuntimeError(f"Cannot find {mpshift}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)
    logging.debug("Calculating shieldings in %s", run_dir_path)

    coord_output = x2t_convert_mol(mol, x2t=x2t, run_dir_path=run_dir_path)
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

    shieldings_path = run_dir_path / "shielding"
    shielding_constants = turbomole_read_isotropic_shieldings(shieldings_path)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return shielding_constants
