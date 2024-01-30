"""Omega interface functions"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, SDWriter


def omega_generate(
    mol,
    omega="omega2",
    cores=1,
    max_confs=250,
    prune_threshold=0.15,
    run_dir_path=None,
):
    """Generate conformations with omega

    Omega commandline interface:
        Simple parameter list
            Execute Options
              -param : A parameter file

            File Options
              -in : Input filename
              -out : Output filename
              -prefix : Prefix to use to name output files
              -progress : Method of showing job progress. "none","dots","log","percent".
              -sdEnergy : Writes conformer energies to the SD tag field
              -verbose : Triggers copious logging output

            3D Construction Parameters
              -fromCT : Generate structures from connection-table only.

            Torsion Driving Parameters
              -ewindow : Energy window used for conformer selection [Defaults: dense:
                         15.0]
              -maxconfs : Maximum number of conformations to be saved. [Defaults: rocs:
                          50, fastrocs: 10, dense: 20000]
              -rms : RMS threshold used to determine duplicate conformations [Defaults:
                     dense: 0.3]
              -useGPU : Switch on GPU-accelerated torsion driving (Default: true on
                        supported Linux systems with GPU)

            Stereo Parameters
              -strictstereo : Requires that all chiral atoms and bonds have specified
                              stereo [Defaults: dense: false]

            General
              -strict : A convenience flag to set -strictstereo, -strictatomtyping, and
                        -strictfrags to true or false and override previous settings.

    :param mol: mol to generate conformations for
    :type mol: rdkit.Chem.rdchem.Mol
    :param omega: path/call to conformator
    :type omega: str
    :param cores: number of cores to run on
    :type cores: int
    :param max_confs: maximum number of conformations to generate
    :type max_confs: int
    :param prune_threshold: RMSD threshold of molecule to consider duplicates
    :type prune_threshold: float
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: mol with conformations
    :rtype: rdkit.Chem.rdchem.Mol
    """
    if not shutil.which(omega):
        raise RuntimeError(f"Cannot find {omega}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)

    logging.debug("Generating conformers in: %s", run_dir_path)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, clearConfs=True, randomSeed=63, useRandomCoords=True)

    input_file_path = run_dir_path / "input.sdf"
    writer = SDWriter(str(input_file_path))
    writer.SetForceV3000(True)
    writer.write(mol)
    writer.close()

    output_file_path = run_dir_path / "output.sdf"
    logging.debug("Output file is: %s", output_file_path)

    args = [
        omega,
        "-in",
        str(input_file_path),
        "-out",
        str(output_file_path),
        "-maxconfs",
        str(max_confs),
        "-rms",
        str(prune_threshold),
    ]
    if cores > 1:
        args.extend(["-mpi_np", str(cores)])

    logging.debug(" ".join(args))
    shell_output = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=run_dir_path)
    if shell_output:
        logging.debug(shell_output)

    if not output_file_path.exists():
        raise RuntimeError("Did not generate conformations")

    mols = list(SDMolSupplier(str(output_file_path), removeHs=False))
    mol = mols[0]
    for conformer_mol in mols[1:]:
        mol.AddConformer(conformer_mol.GetConformer(0), assignId=True)

    logging.info("Generated %s conformations", mol.GetNumConformers())

    if tmp_dir is not None:
        tmp_dir.cleanup()
    return mol
