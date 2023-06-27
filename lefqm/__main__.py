"""Fluorine chemical shift prediction"""
import argparse
import logging

from lefqm.conformers import add_conformers_subparser, conformers
from lefqm.ensembles import add_ensembles_subparser, ensembles
from lefqm.shieldings import add_shieldings_subparser, shieldings
from lefqm.shifts import add_shifts_subparser, shifts

DESCRIPTION = """
LEFQM calculates fluorine chemical shifts using QM techniques

examples:

Generate conformers from SMILES file or a CSV containing SMILES into a
directory:

lefqm conformers input.smi conformer_ensembles/
lefqm conformers input.csv conformer_ensembles/ --smiles-column 'Name of SMILES column'  # specify which column contains the SMILES

SMILES and CSV files should contain IDs/names. These will be used to name the
output conformer ensembles. If these are not given the conformer ensembles will
be named by the input SMILES.

Calculate NMR shielding constants for an SDF containing molecules/conformers
using Turbomole:

lefqm shieldings conformers.sdf shieldings.sdf

Combine NMR shielding constants of conformer ensembles into a CSV:

lefqm ensembles shieldings.sdf shieldings.csv
lefqm ensembles shieldings/ shieldings.csv

Convert NMR shielding constants into chemical shifts with a linear regression
correction:

lefqm shifts shieldings.csv --calibration calibration.csv shifts.csv
"""


def get_args(args=None):
    """Get commandline arguments"""
    parser = argparse.ArgumentParser(
        prog="lefqm",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True

    add_conformers_subparser(subparsers)
    add_shieldings_subparser(subparsers)
    add_ensembles_subparser(subparsers)
    add_shifts_subparser(subparsers)

    return parser.parse_args(args=args)


def call_subtool(args):
    """Call a subtool based on the args"""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.action == "conformers":
        conformers(args)
    elif args.action == "ensembles":
        ensembles(args)
    elif args.action == "shieldings":
        shieldings(args)
    elif args.action == "shifts":
        shifts(args)
    else:
        raise RuntimeError("Invalid action")


def main():
    """Fluorine chemical shift prediction"""
    args = get_args()
    call_subtool(args)


if __name__ == "__main__":
    main()
