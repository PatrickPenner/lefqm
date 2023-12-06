"""Commandline calculations"""
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, SDWriter, rdmolops

from lefqm import constants


class CommandlineCalculation:
    """Commandline calculation"""

    def __init__(self, config, run_dir_path=None):
        """Commandline calculation

        :param config: config for the calculation
        :type config: dict
        :param run_dir_path: path to the directory to run in
        :type run_dir_path: pathlib.Path
        """
        self.config = config
        self.run_dir_path = run_dir_path
        self.tmp_dir = None
        if self.run_dir_path is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.run_dir_path = Path(self.tmp_dir.name)

    def __del__(self):
        """Delete commandline calculation

        Remove the temporary directory if one was created.
        """
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()


class ConformerGeneration(CommandlineCalculation):
    """Conformer generation"""

    def run(self, mol):
        """Run conformer generation

        :param mol: molecule to run the generation for
        :type mol: rdkit.Chem.rdchem.Mol
        :return: mol with conformations
        :rtype: rdkit.Chem.rdchem.Mol
        """
        if self.config["confgen_method"] == "conformator":
            return conformator_generate(
                mol,
                conformator=self.config["conformator"],
                max_confs=int(self.config["max_confs"]),
                run_dir_path=self.run_dir_path,
            )
        if self.config["confgen_method"] == "rdkit":
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=int(self.config["max_confs"]))
            return mol
        if self.config["confgen_method"] == "omega":
            return omega_generate(
                mol,
                omega=self.config["omega"],
                max_confs=int(self.config["max_confs"]),
                cores=int(self.config["cores"]) if self.config["cores"] is not None else None,
                prune_threshold=float(self.config["conf_prune_threshold"]),
            )
        raise RuntimeError("Invalid QM method")


class ShieldingCalculation(CommandlineCalculation):
    """NMR shielding constant calculation"""

    def run(self, mol):
        """Run shielding calculation

        :param mol: molecule to run the calcualtion on
        :type mol: rdkit.Chem.rdchem.Mol
        :return: shieldings for every atom
        :rtype: list[float]
        """
        if self.config["qm_method"] == "turbomole":
            return turbomole_calculate_shieldings(
                mol,
                x2t=self.config["x2t"],
                ridft=self.config["ridft"],
                mpshift=self.config["mpshift"],
                cores=self.config["cores"],
                run_dir_path=self.run_dir_path,
            )
        if self.config["qm_method"] == "nwchem":
            return nwchem_calculate_shieldings(
                mol,
                nwchem=self.config["nwchem"],
                run_dir_path=self.run_dir_path,
            )
        if self.config["qm_method"] == "gaussian":
            return gaussian_calculate_shieldings(
                mol,
                gaussian=self.config["gaussian"],
                run_dir_path=self.run_dir_path,
            )
        raise RuntimeError("Invalid QM method")


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
        with open(xtb_log_path, "w", encoding="utf8") as log_file:
            logging.debug(" ".join(command))
            subprocess.check_call(command, cwd=xtb_dir, stdout=log_file, stderr=subprocess.STDOUT)

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


NWCHEM_TEMPLATE = """
title "Shieldings"
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


shielding_pattern = re.compile(r"\s*isotropic\s*=\s*((\-|\+)?[0-9]+\.[0-9]+)")


def nwchem_read_isotropic_shieldings(log_path):
    """Read isotropic shielding from nwchem log file

    :param log_path: path to the nwchem log file
    :type log_apth: pathlib.Path
    :return: isotropic shielding constants
    :rtype: list[float]
    """
    isotropic_shielding_constants = []
    with open(log_path, encoding="utf8") as log_file:
        for line in log_file.readlines():
            if "isotropic" not in line:
                continue

            shielding_match = shielding_pattern.match(line)
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
    with open(log_path, "w", encoding="utf8") as log_file:
        subprocess.check_call(args, stderr=subprocess.STDOUT, stdout=log_file, cwd=run_dir_path)

    shielding_constants = nwchem_read_isotropic_shieldings(log_path)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return shielding_constants


GAUSSIAN_TEMPLATE = """
# B3LYP/cc-pVDZ NMR SCRF=SMD

Shieldings

{charge} 1
{xyz_string}

"""

gaussian_shielding_pattern = re.compile(
    r"\s*[0-9]+\s*[a-zA-Z]+\s*Isotropic\s*=\s*((\-|\+)?[0-9]+\.[0-9]+)"
)


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

            shielding_match = gaussian_shielding_pattern.match(line)
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
    subprocess.check_call(args, cwd=run_dir_path)

    log_path = run_dir_path / "shielding.log"
    shielding_constants = gaussian_read_isotropic_shieldings(log_path)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return shielding_constants
