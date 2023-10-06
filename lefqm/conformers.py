"""Generate representative conformations"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lefshift.application_utils import validate_column
from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, SDWriter, StereoSpecified, rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.cluster.hierarchy import fcluster, linkage

from lefqm import constants, utils
from lefqm.commandline_calculation import CommandlineCalculation

CONFORMATOR = "conformator"
XTB = "xtb"
MOKA = "blabber_sd"

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


def add_conformers_subparser(subparsers):
    """Add conformers arguments as a subparser"""
    conformers_parser = subparsers.add_parser("conformers")
    conformers_parser.add_argument("input", type=Path, help="input SMILES or CSV file")
    conformers_parser.add_argument("output", type=Path, help="output directory for conformer SDFs")
    conformers_parser.add_argument(
        "--ensemble-size", type=int, help="size of the conformer ensemble", default=15
    )
    conformers_parser.add_argument(
        "--id-column",
        help="name of the column containing the ID",
        default=constants.ID_COLUMN,
        type=str,
    )
    conformers_parser.add_argument(
        "--smiles-column",
        help="name of the column containing the molecule SMILES",
        default=constants.SMILES_COLUMN,
        type=str,
    )
    conformers_parser.add_argument("--chunk", type=int, help="which chunk to process")
    conformers_parser.add_argument("--chunk-size", type=int, help="chunk size to use", default=10)
    conformers_parser.add_argument(
        "--cores", type=int, help="Maximum number of cores to use. Defaults to XTB behavior"
    )
    conformers_parser.add_argument(
        "-v", "--verbose", help="show verbose output", action="store_true"
    )
    conformers_parser.add_argument(
        "--config",
        type=Path,
        help="Config file to read from",
        default=Path(__file__).absolute().parent / "config.ini",
    )


def get_smiles_chunk(
    smiles_path,
    chunk=None,
    chunk_size=None,
    id_column=constants.ID_COLUMN,
    smiles_column=constants.SMILES_COLUMN,
):
    """Get SMILES chunk to process

    Get all if no chunking arguments were passed

    :param smiles_path: path to SMILES file
    :type smiles_path: pathlib.Path
    :param chunk: number of the chunk
    :type chunk: int
    :param chunk_size: size of the chunk
    :type chunk_size: int
    :return: list of SMILES making up the chunk
    :rtype: list[str]
    """
    start_index = None
    end_index = None
    if chunk is not None:
        start_index = chunk * chunk_size
        end_index = start_index + chunk_size

    smiles_chunk = []
    if smiles_path.suffix == ".smi":
        with open(smiles_path, encoding="utf8") as smiles_file:
            for index, line in enumerate(smiles_file.readlines()):
                if not line:
                    continue
                if end_index is not None and index >= end_index:
                    break
                if start_index is None or index >= start_index:
                    smiles_chunk.append(line.strip())
    elif smiles_path.suffix == ".csv":
        smiles_data = pd.read_csv(smiles_path, dtype={id_column: str, smiles_column: str})
        smiles_data = validate_column(
            smiles_data,
            smiles_column,
            str,
            f'Could not find structure column "{smiles_column}" in input. Specify structure column with name "--smiles-column" option.',
        )
        if start_index is not None and end_index is not None:
            smiles_data = smiles_data[start_index:end_index]
        smiles_chunk = list(smiles_data[smiles_column] + " " + smiles_data[id_column])

    if chunk is not None and len(smiles_chunk) == 0:
        raise RuntimeError(f"Invalid chunk {chunk} has no members. (chunks start with --chunk 0)")

    return smiles_chunk


def normalize(mol):
    """Normalize a mol

    Normalization does the following:
    1. Strip salts
    2. Uncharge
    3. Assign stereo centers

    More than one unassigned stereocenter will cause an error

    :param mol: mol to normalize
    :type mol: rdkit.Chem.rdchem.Mol
    :return: normalized mol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    stripped_mol = SaltRemover().StripMol(mol)
    if stripped_mol.GetNumAtoms() == 0:
        logging.warning(
            'Salt strippper removed entire molecule "%s", keeping original', mol.GetProp("_Name")
        )
    else:
        mol = stripped_mol
    if rdmolops.GetFormalCharge(mol) != 0:
        mol = rdMolStandardize.ChargeParent(mol)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    stereo_specified = [info.specified for info in rdmolops.FindPotentialStereo(mol)]
    unspecified_count = stereo_specified.count(StereoSpecified.Unspecified)
    if unspecified_count > 1 or (unspecified_count == 1 and len(stereo_specified) > 1):
        raise RuntimeError("Cannot process unassigned diastereomers")

    if unspecified_count == 1:
        stereo_options = StereoEnumerationOptions(
            tryEmbedding=True, onlyUnassigned=True, maxIsomers=1, rand=67
        )
        stereo_isomers = list(EnumerateStereoisomers(mol, options=stereo_options))
        assert len(stereo_isomers) >= 1, "RDKit should generate stereo isomers"
        mol = stereo_isomers[0]
    return mol


def protonate(mol, ph=7.6, min_abundance=90):
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
    :param ph: pH for protonation
    :type ph: float
    :param min_abundance: cutoff in percent for protomer abundance
    :type min_abundance: int
    :return: protonated mol
    :rtype: rdkit.Chem.rdchem.Mol
    """
    if not shutil.which(MOKA):
        raise RuntimeError(f"Cannot find {MOKA}")

    with tempfile.TemporaryDirectory() as moka_dir:
        moka_dir = Path(moka_dir)
        logging.debug("Generating tautomers/protomers in: %s", moka_dir)

        input_file_path = moka_dir / "input.sdf"
        writer = SDWriter(str(input_file_path))
        writer.write(mol)
        writer.close()

        output_file_path = moka_dir / "output.sdf"
        logging.debug("Output file is: %s", output_file_path)

        command = [
            MOKA,
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
            return self.conformator_generate(mol, run_dir_path=self.run_dir_path)
        raise RuntimeError("Invalid QM method")

    @staticmethod
    def conformator_generate(mol, conformator="conformator", run_dir_path=None):
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

        shell_output = subprocess.check_output(
            [
                conformator,
                "-i",
                str(input_file_path),
                "-o",
                str(output_file_path),
            ],
            stderr=subprocess.STDOUT,
        )
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


def prune_conformations(mol, prune_threshold):
    """Prune duplicated conformations

    :param mol: molecule with conformations
    :type mol: rdkit.Chem.rdchem.Mol
    :param prune_threshold: RMSD threshold of molecule to consider duplicates
    :type prune_threshold: float
    """
    nof_conformations = mol.GetNumConformers()
    rms_matrix = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
    full_rms_matrix = np.zeros((nof_conformations, nof_conformations))
    condensed_index = 0
    for i in range(1, nof_conformations):
        for j in range(i):
            full_rms_matrix[i, j] = rms_matrix[condensed_index]
            full_rms_matrix[j, i] = rms_matrix[condensed_index]
            condensed_index += 1

    # prune close conformations
    remove = set()
    for i in range(nof_conformations):
        if i in remove:
            continue

        for j in range(i + 1, nof_conformations):
            if full_rms_matrix[i, j] < prune_threshold:
                remove.add(j)

    for i in remove:
        mol.RemoveConformer(i)
    nof_conformations = mol.GetNumConformers()
    logging.debug("%s left after pruning by %s RMSD", nof_conformations, prune_threshold)


def rmsd_cluster(
    mol, conformation_energies, nof_clusters=15, nof_picks_cluster=1, prune_threshold=0.15
):
    """Cluster conformations by pairwise RMSD

    :param mol: molecule with conformations
    :type mol: rdkit.Chem.rdchem.Mol
    :param conformation_energies: energies of the conformations
    :type conformation_energies: list[float]
    :param nof_clusters: maximum number of clusters to generate
    :type nof_clusters: int
    :param nof_picks_cluster: number of conformations to pick per cluster
    :type nof_picks_cluster: int
    :param prune_threshold: RMSD threshold of molecule to consider duplicates
    :type prune_threshold: float
    :return: indexes of the picked conformations
    :rtype: list[int]
    """
    # copy mol to not modify the input mol
    mol = Chem.Mol(mol)
    assert (
        len(conformation_energies) == mol.GetNumConformers()
    ), "Must have energies for all conformations"

    # prune conformations
    prune_conformations(mol, prune_threshold)
    conformer_ids = [conformer.GetId() for conformer in mol.GetConformers()]
    if mol.GetNumConformers() <= nof_clusters:
        return conformer_ids

    # actual clustering
    rms_matrix = AllChem.GetConformerRMSMatrix(mol, prealigned=True)
    matrix = linkage(rms_matrix, method="ward", metric="euclidean")
    cluster_labels = fcluster(matrix, nof_clusters, criterion="maxclust")

    clusters = {}
    for index, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((index, conformation_energies[index]))

    cluster_picks = []
    for cluster in clusters.values():
        cluster_indexes, _ = zip(*sorted(cluster, key=lambda x: x[1]))
        cluster_picks.extend(cluster_indexes[:nof_picks_cluster])

    logging.debug("Clustered down to %s conformations", len(cluster_picks))
    # map cluster picks back to conformer IDs
    return [conformer_ids[pick] for pick in cluster_picks]


def optimize_conformation(mol, conformation_index, cores=None):
    """Optimize a conformation with XTB

    :param mol: molecule with conformations to optimize
    :type mol: rdkit.Chem.rdchem.Mol
    :param conformation_index: index of the conformation to optimize
    :type conformation_index: int
    :return: optimized conformation and XTB conformer energy in hartrees
    :rtype: (rdkit.Chem.rdchem.Conformer, float)
    """
    if not shutil.which(CONFORMATOR):
        raise RuntimeError(f"Cannot find {XTB}")

    with tempfile.TemporaryDirectory() as xtb_dir:
        logging.debug("Optimizing conformers in: %s", xtb_dir)
        xtb_dir = Path(xtb_dir)
        charge = rdmolops.GetFormalCharge(mol)
        with open(xtb_dir / ".CHRG", "w", encoding="utf8") as charge_file:
            charge_file.write(str(charge))

        input_file_path = xtb_dir / "input.xyz"
        with open(input_file_path, "w", encoding="utf8") as input_file:
            input_file.write(Chem.MolToXYZBlock(mol, confId=conformation_index))

        command = [XTB, input_file_path.name]
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


def optimize(mol, conformation_indexes=None, cores=None):
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
            optimized_conformer, energy = optimize_conformation(
                mol, conformation_index, cores=cores
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


def conformers(args):
    """Generate representative conformations"""
    config = utils.config_to_dict(args.config)
    config["cores"] = args.cores

    if args.input.suffix not in {".smi", ".csv"}:
        raise RuntimeError("Input may only be a SMILES (.smi) or a CSV (.csv)")

    args.output.mkdir(exist_ok=True, parents=True)
    for smiles in get_smiles_chunk(
        args.input, args.chunk, args.chunk_size, args.id_column, args.smiles_column
    ):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol.HasProp("_Name"):
                logging.warning(
                    "Molecule %s does not have a name. The SMILES will be used.", smiles
                )
                mol.SetProp("_Name", smiles)

            mol = normalize(mol)
            mol = protonate(mol)
            mol = ConformerGeneration(config).run(mol)

            # optimization
            mol, energies = optimize(mol, cores=args.cores)

            # RMSD clustering
            conformation_indexes = rmsd_cluster(
                mol, energies, nof_clusters=args.ensemble_size, nof_picks_cluster=1
            )

            writer = SDWriter(str(args.output / (mol.GetProp("_Name") + ".sdf")))
            conformation_energies = [energies[index] for index in conformation_indexes]
            boltzmann_weights = utils.get_boltzmann_weights(np.array(conformation_energies))
            for conformer_id, energy, weight in zip(
                conformation_indexes, conformation_energies, boltzmann_weights
            ):
                mol.SetProp(constants.ENERGY_SD_PROPERTY, str(energy))
                mol.SetProp(constants.BOLTZMANN_WEIGHT_SD_PROPERTY, str(weight))
                writer.write(mol, confId=conformer_id)
            writer.close()
        except Exception as exception:
            logging.warning("Molecule %s failed with error: %s", smiles, exception)
