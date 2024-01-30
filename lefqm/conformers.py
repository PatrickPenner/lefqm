"""Generate representative conformations"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lefshift.application_utils import validate_column
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, StereoSpecified, rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.cluster.hierarchy import fcluster, linkage

from lefqm import constants, utils
from lefqm.commandline_calculation import ConformerGeneration
from lefqm.moka import moka_protonate
from lefqm.xtb import xtb_optimize


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
        default=constants.DEFAULT_CONFIG,
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
        smiles_data = validate_column(
            smiles_data,
            id_column,
            str,
            f'Could not find ID column "{id_column}" in input. Specify structure column with name "--id-column" option.',
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


def conformers(args):
    """Generate representative conformations"""
    config = utils.get_config(args.config)
    config["Parameters"]["cores"] = str(args.cores)

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
            mol = moka_protonate(mol, moka=config["Paths"]["moka"])
            mol = ConformerGeneration(config).run(mol)

            # optimization
            mol, energies = xtb_optimize(mol, xtb=config["Paths"]["xtb"], cores=args.cores)

            # RMSD clustering
            conformation_indexes = rmsd_cluster(
                mol,
                energies,
                nof_clusters=args.ensemble_size,
                nof_picks_cluster=1,
                prune_threshold=float(config["Parameters"]["conf_prune_threshold"]),
            )

            writer = SDWriter(str(args.output / (mol.GetProp("_Name") + ".sdf")))
            writer.SetForceV3000(True)
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
