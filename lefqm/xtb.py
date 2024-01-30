"""xTB interface functions"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

from lefqm import constants

XTB_DEFAULT = {
    "gfn": 2,
    "alpb": "water",
    "opt": None,
    "cycles": 100,
}

XTB_FAST = {
    "gfn": 0,
    "alpb": "water",
    "opt": None,
    "cycles": 200,
}

XTB_LAX = {
    "gfn": 2,
    "alpb": "water",
    "opt": "lax",
}


def xtb_optimize_conformation(mol, conformation_index, xtb="xtb", cores=None):
    """Optimize a conformation with XTB

    :param mol: molecule with conformations to optimize
    :type mol: rdkit.Chem.rdchem.Mol
    :param xtb: path/call to xtb
    :type xtb: str
    :param conformation_index: index of the conformation to optimize
    :type conformation_index: int
    :return: optimized conformation and XTB conformer energy in hartrees
    :rtype: (rdkit.Chem.rdchem.Conformer, float)
    """
    if not shutil.which(xtb):
        raise RuntimeError(f"Cannot find {xtb}")

    with tempfile.TemporaryDirectory() as xtb_dir:
        logging.debug("Optimizing conformers in: %s", xtb_dir)
        xtb_dir = Path(xtb_dir)
        charge = rdmolops.GetFormalCharge(mol)
        with open(xtb_dir / ".CHRG", "w", encoding="utf8") as charge_file:
            charge_file.write(str(charge))

        input_file_path = xtb_dir / "input.xyz"
        with open(input_file_path, "w", encoding="utf8") as input_file:
            input_file.write(Chem.MolToXYZBlock(mol, confId=conformation_index))

        command = [xtb, input_file_path.name]
        for option, value in XTB_DEFAULT.items():
            command.append(f"--{option}")
            if value is not None:
                command.append(str(value))

        if cores:
            command.extend(["--parallel", str(cores)])

        xtb_log_path = xtb_dir / "xtb.log"
        logging.debug(" ".join(command))
        try:
            with open(xtb_log_path, "w", encoding="utf8") as log_file:
                subprocess.check_call(
                    command, cwd=xtb_dir, stdout=log_file, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError:
            logging.info(open(xtb_log_path, encoding="utf8").read())
            raise

        with open(xtb_log_path, encoding="utf8") as log_file:
            lines = list(log_file.readlines())
            if "finished run on" not in lines[-16]:
                raise RuntimeError("XTB calculation did not converge")

        with open(xtb_dir / "xtbopt.xyz", encoding="utf8") as optimized_xyz_file:
            xyz_string = optimized_xyz_file.read()

        comment_line = [
            token.strip() for token in xyz_string.split("\n")[1].split(" ") if token.strip()
        ]
        assert "energy:" == comment_line[0], "Energy was not written with molecule"
        energy = float(comment_line[1])
        optimized_mol = Chem.MolFromXYZBlock(xyz_string)

    return optimized_mol.GetConformer(0), energy


def xtb_optimize(mol, xtb="xtb", conformation_indexes=None, cores=None):
    """Optimize the conformations of a molecule with XTB

    :param mol: molecule with conformations to optimize
    :type mol: rdkit.Chem.rdchem.Mol
    :return: optimized molecule and XTB conformer energies in kcal/mol
    :rtype: (rdkit.Chem.rdchem.Mol, list[float])
    """
    if conformation_indexes is None:
        conformation_indexes = range(mol.GetNumConformers())

    optimized_mol = Chem.Mol(mol, True)  # copy without conformers
    optimized_mol.SetProp("_Name", mol.GetProp("_Name"))
    energies = []
    for index, conformation_index in enumerate(conformation_indexes):
        try:
            optimized_conformer, energy = xtb_optimize_conformation(
                mol, conformation_index, xtb=xtb, cores=cores
            )
            energies.append(energy * constants.HARTREE_TO_KCALMOL)
            optimized_mol.AddConformer(optimized_conformer, assignId=True)
        except RuntimeError as error:
            logging.warning(
                "Skipped conformation in molecule %s that encountered error: %s",
                mol.GetProp("_Name"),
                error,
            )
        logging.info("Optimized %s / %s", index + 1, len(conformation_indexes))
    return optimized_mol, energies
