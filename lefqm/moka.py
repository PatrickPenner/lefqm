"""MoKa interface functions"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit.Chem import SDMolSupplier, SDWriter


def moka_protonate(mol, moka="blabber_sd", ph=7.6, min_abundance=90):
    """Protonate molecule

    Assign the most abundant tautomer/protomer. MoKa commandline interface:

        usage: /usr/prog/MoKa/4.0.1/blabber_sd.bin [options] <filename>

         <filename> can be a SD (.sd or .sdf) or MOL2 (.mol2) file

        options are:
         -h, --help                     display this help
             --version                  show version info
             --input-type=<sd|mol2|smi> input file type (autodetect)
         -p <pH list>, --pH=<pH list>   report most abundant species
                                        at given pH (7.4)
         -n, --neutralize               report only the neutral species
             --load-model=<database>    use a custom model database for predictions
         -o, --output=FILE              output file name (SD only)
         -m, --minimize                 perform full 3D minimization
         -t <abundance>                 report all species above the threshold
         -e, --equivalent-protomers     generate all equivalent protomers

        <pH list> is a comma separated list of the following:
            * single value
            * range ( min-max )

    :param mol: mol to generate tautomers/protomers for
    :type mol: rdkit.Chem.rdchem.Mol
    :param moka: path/call to moka
    :type moka: str
    :param ph: pH for protonation
    :type ph: float
    :param min_abundance: cutoff in percent for protomer abundance
    :type min_abundance: int
    :return: protonated mol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    if not shutil.which(moka):
        raise RuntimeError(f"Cannot find {moka}")

    with tempfile.TemporaryDirectory() as moka_dir:
        moka_dir = Path(moka_dir)
        logging.debug("Generating tautomers/protomers in: %s", moka_dir)

        input_file_path = moka_dir / "input.sdf"
        writer = SDWriter(str(input_file_path))
        writer.SetForceV3000(True)
        writer.write(mol)
        writer.close()

        output_file_path = moka_dir / "output.sdf"
        logging.debug("Output file is: %s", output_file_path)

        command = [
            moka,
            str(input_file_path),
            "-o",
            str(output_file_path),
            "-p",
            str(ph),
            "-t",
            str(min_abundance),
        ]
        logging.debug(" ".join(command))
        shell_output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if shell_output:
            logging.debug(shell_output)

        mols = SDMolSupplier(str(output_file_path), removeHs=False)
        if len(mols) == 1 and mols[0].GetProp("ABUNDANCE") == "No species found":
            logging.debug("MoKa found no protomers, returning original")
            return mol

        mols = sorted(
            mols, key=lambda mol: float(mol.GetProp("ABUNDANCE").split("%")[0]), reverse=True
        )
    return mols[0]
