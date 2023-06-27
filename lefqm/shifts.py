"""Shift calculation"""
import logging
from pathlib import Path

import pandas as pd
from lefshift.application_utils import validate_column
from sklearn.linear_model import LinearRegression

from lefqm import constants


def add_shifts_subparser(subparsers):
    """Add shifts arguments as a subparser"""
    shifts_parser = subparsers.add_parser("shifts")
    shifts_parser.add_argument(
        "input", help="Input CSV with shieldings to convert to chemical shifts", type=Path
    )
    shifts_parser.add_argument(
        "--calibration", required=True, help="CSV calibration data for the conversion", type=Path
    )
    shifts_parser.add_argument("output", help="OUtput CSV with chemical shifts", type=Path)
    shifts_parser.add_argument(
        "--shielding-column",
        help="name of the column containing the shieldings constants",
        default=constants.SHIELDING_COLUMN,
        type=str,
    )
    shifts_parser.add_argument(
        "--shift-column",
        help="name of the column containing the chemical shift",
        default=constants.SHIFT_COLUMN,
        type=str,
    )
    shifts_parser.add_argument(
        "--id-column",
        help="name of the column containing the ID",
        default=constants.ID_COLUMN,
        type=str,
    )
    shifts_parser.add_argument("-v", "--verbose", help="show verbose output", action="store_true")


def shifts(args):
    """Convert shieldings in input to shifts"""
    data = pd.read_csv(args.input)
    id_column_error = f'Could not find ID column "{args.id_column}" in input. Specify ID column name with "--id-column" option.'
    shielding_column_error = f'Could not find shielding constants column "{args.shielding_column}" in input. Specify shielding column name with "--shielding-column" option.'
    shift_column_error = f'Could not find chemical shift column "{args.shift_column}" in input. Specify shift column name with "--shift-column" option.'
    data = validate_column(data, args.id_column, str, id_column_error)
    data = validate_column(data, args.shielding_column, float, shielding_column_error)

    calibration_data = pd.read_csv(args.calibration)
    calibration_data = validate_column(calibration_data, args.id_column, str, id_column_error)
    calibration_data = validate_column(
        calibration_data, args.shielding_column, float, shielding_column_error
    )
    calibration_data = validate_column(
        calibration_data, args.shift_column, float, shift_column_error
    )
    calibration_data = validate_column(calibration_data, constants.LABEL_COLUMN, str)

    all_shifts = []
    for cf_label in constants.CF_LABELS:
        logging.info('Calibrating conversion of "%s" shieldings', cf_label)
        current_calibration_data = calibration_data[
            calibration_data[constants.LABEL_COLUMN] == cf_label
        ]
        if len(current_calibration_data) == 0:
            logging.warning('No calibration data found for "%s", skipping', cf_label)
            continue

        calibration = LinearRegression().fit(
            current_calibration_data[args.shielding_column].values.reshape(-1, 1),
            current_calibration_data[args.shift_column],
        )

        logging.info('Converting "%s" shieldings', cf_label)
        current_data = data[data[constants.LABEL_COLUMN] == cf_label]
        if len(current_data) == 0:
            logging.warning('No input data found for "%s", skipping', cf_label)
            continue

        chemical_shifts = calibration.predict(
            current_data[args.shielding_column].values.reshape(-1, 1)
        )
        shifts_df = pd.DataFrame(
            chemical_shifts, columns=[args.shift_column], index=current_data.index
        )
        all_shifts.append(shifts_df)

    all_shifts_df = pd.concat(all_shifts)
    data_with_shifts = data.join(all_shifts_df, rsuffix=" Calculated")
    data_with_shifts.to_csv(args.output, index=False)
