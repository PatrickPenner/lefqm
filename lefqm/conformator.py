"""Conformator interface functions"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import SDMolSupplier


def conformator_generate(mol, conformator="conformator", max_confs=250, run_dir_path=None):
    """Generate conformations for a molecule using the conformator

    Conformator commandline interface:

        Calculates conformations for the molecules in the input file:

        Options:
         -h [ --help ]                  Prints help message
         -v [ --verbosity ] arg (=3)    Set verbosity level
                                        (0 = Quiet, 1 = One-line-summary, 2 = Errors,
                                        3 = Warnings, 4 = Info)
         -i [ --input ] arg             Input file, suffix is required.
         -o [ --output ] arg            Output file, suffix is required.
         -q [ --quality ] arg (=2)      Set quality level
                                        (1 = Fast, 2 = Best)
         -n [ --nOfConfs ] arg (=250)   Set maximum number of conformations to be
                                        generated.
         -f [ --from ] arg (=1)         Position of first entry in the calculation.
         -t [ --to ] arg (=4294967295)  Position of last entry in the calculation.
         --keep3d                       Keep initial 3D coordinates for molecule as
                                        starting point for conformation generation.
         --hydrogens                    Consider hydrogen clashes during conformation
                                        generation.
         --macrocycle_size arg (=10)    Define minimum size of macrocycles (<= 10)
         --rmsd_input                   Calculate the minimum RMSD of the closest
                                        ensemble member to the input structure.
         --rmsd_ensemble                Calculate the minimum RMSD of the closest
                                        ensemble members to each other.

        License:
         --license arg                  To reactivate the executable, please provide a
                                        new license key.

    :param mol: mol to generate conformations for
    :type mol: rdkit.Chem.rdchem.Mol
    :param conformator: path/call to conformator
    :type conformator: str
    :param max_confs: maximum number of conformations to generate
    :type max_confs: int
    :param run_dir_path: path to the directory to run in
    :type run_dir_path: pathlib.Path
    :return: mol with conformations
    :rtype: rdkit.Chem.rdchem.Mol
    """
    if not shutil.which(conformator):
        raise RuntimeError(f"Cannot find {conformator}")

    tmp_dir = None
    if run_dir_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir_path = Path(tmp_dir.name)

    logging.debug("Generating conformers in: %s", run_dir_path)

    input_file_path = run_dir_path / "input.smi"
    with open(input_file_path, "w", encoding="utf8") as input_file:
        input_file.write(Chem.MolToSmiles(mol) + " " + mol.GetProp("_Name"))

    output_file_path = run_dir_path / "output.sdf"
    logging.debug("Output file is: %s", output_file_path)

    args = [
        conformator,
        "-i",
        str(input_file_path),
        "-o",
        str(output_file_path),
        "--nOfConfs",
        str(max_confs),
    ]
    logging.debug(" ".join(args))
    shell_output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    if shell_output:
        logging.debug(shell_output)
    if (
        bytes("INPUT", encoding="ascii") not in shell_output
        and bytes("NONE", encoding="ascii") not in shell_output
        and bytes("PURE", encoding="ascii") not in shell_output
    ):
        logging.warning("Detected potential undefined stereo")
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
