"""QM utils"""
import configparser
import copy
import io
import itertools
import logging
import math

import numpy as np
from lefshift import utils

from lefqm import constants


def get_fluorine_shieldings(mol, shielding_property=constants.SHIELDING_SD_PROPERTY):
    """Get fluorine shieldings for a molecule

    TODO make shieldings a type

    CF3 and CF2 fluorine shieldings are averaged. The keys for CF3 and CF2
    groups are tuples.

    :param mol: molecule to extract shieldings from
    :type mol: rdkit.Chem.rdchem.Mol
    :return: dict of atom indexes to shieldings
    :rtype: dict[int | tuple[int], float]
    """
    fluorine_atoms = {atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"}
    flourine_shieldings = {}
    processed_indexes = set()
    # CF3
    for cf3_group in itertools.combinations(fluorine_atoms, 3):
        if not utils.have_common_neighbor(cf3_group):
            continue
        cf3_group_indexes = tuple(sorted([atom.GetIdx() for atom in cf3_group]))
        if any(index in processed_indexes for index in cf3_group_indexes):
            continue
        flourine_shieldings[cf3_group_indexes] = sum(
            atom.GetDoubleProp(shielding_property) for atom in cf3_group
        ) / len(cf3_group)
        processed_indexes.update(cf3_group_indexes)
    # CF2
    for cf2_group in itertools.combinations(fluorine_atoms, 2):
        if not utils.have_common_neighbor(cf2_group):
            continue
        cf2_group_indexes = tuple(sorted([atom.GetIdx() for atom in cf2_group]))
        if any(index in processed_indexes for index in cf2_group_indexes):
            continue
        flourine_shieldings[cf2_group_indexes] = sum(
            atom.GetDoubleProp(shielding_property) for atom in cf2_group
        ) / len(cf2_group)
        processed_indexes.update(cf2_group_indexes)
    not_processed_indexes = [
        atom.GetIdx() for atom in fluorine_atoms if atom.GetIdx() not in processed_indexes
    ]
    # CF
    for atom_index in not_processed_indexes:
        flourine_shieldings[atom_index] = mol.GetAtomWithIdx(atom_index).GetDoubleProp(
            shielding_property
        )
    return flourine_shieldings


def get_boltzmann_weights(energies, temperature=298.15, boltzmann_constant=1.987204259e-3):
    """Calculate boltzmann weights

    :param energies: energies in kcal per Mol
    :type energies: list[float]
    :param temperature: temperature in Kelvin
    :type temperature: float
    :param boltzmann_constant: Boltzmann constant in kcal/(mol x K)
    :type boltzmann_constant: float
    :return: weights
    :rtype: np.ndarray[float]
    """
    if energies is None or len(energies) == 0:
        raise RuntimeError("Passed invalid energies")

    inverse_kt = 1.0 / (boltzmann_constant * temperature)
    energies = np.array(copy.copy(energies))
    energies -= energies.min()
    energies = np.exp(-energies * inverse_kt)
    energy_sum = np.sum(energies)
    return energies / energy_sum


def get_averaged_shielding(shieldings):
    """Get a single averaged shielding for multiple shieldings

    The lowest key will be associated with the new averaged shielding

    :param shieldings: shieldings to average
    :type shieldings: dict[int | tuple[int], float]
    :return: single averaged shielding
    :rtype: dict[int | tuple[int], float]
    """
    average = 0
    lowest_key = None
    lowest_key_value = None
    for key in shieldings:
        average += shieldings[key]
        key_value = key
        if isinstance(key, tuple):
            key_value = sum(key)
        if lowest_key_value is None or lowest_key_value > key_value:
            lowest_key_value = key_value
            lowest_key = key
    average /= len(shieldings)
    return {lowest_key: average}


def get_weighted_average_ensemble_shieldings(ensemble_shieldings, weights):
    """Calculate weighted averages of the ensemble shieldings

    :param ensemble_shieldings: ensemble shieldings to average
    :type ensemble_shieldings: list[dict[int | tuple[int], float]]
    :param weights: weights to average by
    :type weights: list[float]
    :return: shieldings average over the ensemble
    :rtype: dict[int | tuple[int], float]
    """
    normalized_weights = np.array(copy.copy(weights))
    if not math.isclose(np.sum(normalized_weights), 1.0):
        normalized_weights /= np.sum(normalized_weights)
    averaged_shieldings = {}
    for key in ensemble_shieldings[0].keys():
        averaged_shieldings[key] = sum(
            shieldings[key] * weight
            for shieldings, weight in zip(ensemble_shieldings, normalized_weights)
        )
    return averaged_shieldings


def match_shieldings(ensemble_shieldings, ensemble):
    """Match shieldings of an ensemble

    :param ensemble_shieldings: ensemble shieldings to match
    :type ensemble_shieldings: list[dict[int | tuple[int], float]]
    :param ensemble: ensemble of molecule isomers
    :type ensemble; list[rdkit.Chem.rdchem.Mol]
    :return: matched shieldings
    :rtype: list[dict[int | tuple[int], float]]
    """
    # first set of shieldings is used as reference
    reference_shieldings = ensemble_shieldings[0]
    reference_mol = ensemble[0]
    matched_ensemble_shieldings = []
    for shieldings, mol in zip(ensemble_shieldings, ensemble):
        if reference_shieldings.keys() == shieldings.keys():
            matched_ensemble_shieldings.append(shieldings)
            continue
        match_map = utils.map_atom_indexes(mol, reference_mol)
        matched_shieldings = {}
        for key in shieldings.keys():
            if isinstance(key, tuple):
                new_key = []
                for index in key:
                    new_key.append(match_map[index])
                new_key = tuple(new_key)
                matched_shieldings[new_key] = shieldings[key]
            else:
                matched_shieldings[match_map[key]] = shieldings[key]
        matched_ensemble_shieldings.append(matched_shieldings)
    return matched_ensemble_shieldings


def get_ensemble_fluorine_shieldings(
    ensemble,
    water_energy_property=constants.ENERGY_SD_PROPERTY,
    shielding_property=constants.SHIELDING_SD_PROPERTY,
):
    """Get fluorine shieldings for an ensemble of molecules

    :param ensemble: ensemble of molecule isomers
    :type ensemble; list[rdkit.Chem.rdchem.Mol]
    :return: dict of atom indexes to shieldings of the ensemble
    :rtype: dict[int | tuple[int], float]
    """
    ensemble_fluorine_shieldings = [
        get_fluorine_shieldings(mol, shielding_property=shielding_property) for mol in ensemble
    ]
    ensemble_fluorine_shieldings = match_shieldings(ensemble_fluorine_shieldings, ensemble)
    energies = [float(mol.GetProp(water_energy_property)) for mol in ensemble]
    conformer_weights = get_boltzmann_weights(energies)
    weighted_average_shieldings = get_weighted_average_ensemble_shieldings(
        ensemble_fluorine_shieldings, conformer_weights
    )
    return weighted_average_shieldings


def get_lowest_energy_fluorine_shieldings(
    ensemble,
    water_energy_property=constants.ENERGY_SD_PROPERTY,
    shielding_property=constants.SHIELDING_SD_PROPERTY,
):
    """Get fluorine shieldings for an ensemble of molecules

    :param ensemble: ensemble of molecule isomers
    :type ensemble; list[rdkit.Chem.rdchem.Mol]
    :return: dict of atom indexes to shieldings of the ensemble or None
    :rtype: dict[int | tuple[int], float] or None
    """
    energies = [float(mol.GetProp(water_energy_property)) for mol in ensemble]
    lowest_energy = min(energies)
    for energy, mol in zip(energies, ensemble):
        if math.isclose(energy, lowest_energy):
            return get_fluorine_shieldings(mol, shielding_property=shielding_property)

    return None


def get_config(config_file_path=constants.DEFAULT_CONFIG):
    """Read default and user-specified configuration"""
    config = configparser.ConfigParser()
    config.read(constants.DEFAULT_CONFIG)
    if config_file_path != constants.DEFAULT_CONFIG:
        config.read(config_file_path)
    with io.StringIO() as string_stream:
        string_stream.write("Config:\n")
        config.write(string_stream)
        string_stream.seek(0)
        logging.debug(string_stream.read())
    return config
