"""
Step 2: extract pulse-response records and U1-U41 features from Step1 files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from config import DEFAULT_PT_VALUES, DEFAULT_SOC_VALUES, DEFAULT_U_INDEXES
from utils import (
    ensure_dir,
    filter_type_folders,
    is_train_file_by_name,
    list_excel_files,
    list_type_folders,
    parse_float_list,
    parse_int_list,
    parse_step_source_metadata,
)


RAW_PT_ORDER = list(DEFAULT_PT_VALUES)
PULSE_GROUP_COUNT = 5
STEPS_PER_PULSE_GROUP = 4
# Each SOC block in the Step1 workstep sheet contains all pulse-width groups
# plus two separator rows used by the original acquisition layout.
SOC_BLOCK_SIZE = len(RAW_PT_ORDER) * PULSE_GROUP_COUNT * STEPS_PER_PULSE_GROUP + 2
HEADER_ITEMS = (
    ["File_Name", "Mat", "No.", "ID", "Qn", "Q", "SOH", "Pt", "SOC", "SOCR"]
    + [f"U{i}" for i in DEFAULT_U_INDEXES]
)


def build_record_template(meta: dict, raw_values) -> List[object]:
    """Create the shared metadata portion of one Step2 output row."""
    record = [None] * len(HEADER_ITEMS)
    q = -float(raw_values[3][16])
    record[0] = meta["file_name"]
    record[1] = meta["mat"]
    record[2] = meta["no"]
    record[3] = meta["id"]
    record[4] = meta["qn"]
    record[5] = q
    record[6] = q / meta["qn"]
    return record


def extract_soc_row_number_by_index(soc_index: int, pt_value: float) -> int:
    """Map one SOC index and pulse width to the anchor row in the Step1 sheet."""
    pt_index = RAW_PT_ORDER.index(float(pt_value))
    return 4 + SOC_BLOCK_SIZE * soc_index + 2 + STEPS_PER_PULSE_GROUP * PULSE_GROUP_COUNT * pt_index


def write_u_features(record: List[object], raw_values, row_shift: int = 0) -> None:
    # The Step1 sheet stores alternating voltage columns around each pulse event.
    # U1 takes the first voltage point, then even/odd indexes switch columns.
    for u_num in DEFAULT_U_INDEXES:
        u_index = DEFAULT_U_INDEXES.index(u_num) + 1
        row_num = record[-1] + (u_num // 2) + row_shift
        if row_num >= raw_values.shape[0]:
            continue

        u_target = 9 + u_index

        if u_num == 1:
            record[u_target] = raw_values[row_num][12]
        elif u_num % 2 == 0:
            record[u_target] = raw_values[row_num][10]
        else:
            record[u_target] = raw_values[row_num][12]


def extract_records_for_file(input_file: Path, soc_values: List[int], pt_values: List[float], row_shift: int) -> List[List[object]]:
    """Extract all valid Step2 rows from one Step1 workbook."""
    raw_values = pd.read_excel(input_file, engine="openpyxl").values
    meta = parse_step_source_metadata(input_file.name)
    base_record = build_record_template(meta, raw_values)
    data: List[List[object]] = []

    if is_train_file_by_name(input_file.name):
        # Training files contain the full SOC ladder in one workbook.
        soc_iterable = [(float(soc_value), int(round(float(soc_value) / 5.0 - 1))) for soc_value in soc_values]
    else:
        # Non-training files encode a single SOC directly in the filename.
        soc_iterable = [(float(meta["soc_token"]), 0)]

    for soc_value, soc_index in soc_iterable:
        for pt_value in pt_values:
            record = list(base_record)
            soc_row_num = extract_soc_row_number_by_index(soc_index, float(pt_value))
            record[7] = float(pt_value)
            record[8] = float(soc_value)
            record[9] = float(raw_values[5 : (soc_row_num + 1), 18].sum()) / meta["qn"]
            record.append(soc_row_num)
            write_u_features(record, raw_values, row_shift=row_shift)
            record.pop()

            if record[10] is not None:
                data.append(record)

    return data


def process_type_folder(type_folder: Path, output_root: Path, soc_values: List[int], pt_values: List[float], row_shift: int) -> None:
    """Generate one Step2 workbook per Step1 workbook for a given material folder."""
    output_folder = output_root / type_folder.name
    ensure_dir(output_folder)
    input_files = list_excel_files(type_folder)
    print(f"[INFO] {type_folder.name}: {len(input_files)} files")

    for index, input_file in enumerate(input_files, start=1):
        records = extract_records_for_file(input_file, soc_values, pt_values, row_shift=row_shift)
        output_file = output_folder / input_file.name
        pd.DataFrame(records, columns=HEADER_ITEMS).to_excel(output_file, index=False, engine="openpyxl")
        print(f"[SAVE] ({index}/{len(input_files)}) {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step2: extract pulse turning-point features from workstep files.")
    parser.add_argument("--input-root", required=True, help="Root directory containing Step1 output type folders.")
    parser.add_argument("--output-root", required=True, help="Root directory for Step2 per-cell feature files.")
    parser.add_argument("--soc-values", default=None, help="Comma-separated SOC values to extract. Default: 5,10,...,90")
    parser.add_argument("--pt-values", default=None, help="Comma-separated pulse widths in seconds. Default: 0.03,0.05,...,5")
    parser.add_argument("--u-t-row-shift", type=int, default=0, help="Optional row shift for special files. Default: 0")
    parser.add_argument("--materials", nargs="*", default=None, help="Optional battery type folder names to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_dir(output_root)
    soc_values = parse_int_list(args.soc_values, DEFAULT_SOC_VALUES)
    pt_values = parse_float_list(args.pt_values, DEFAULT_PT_VALUES)

    invalid_pt_values = [value for value in pt_values if value not in RAW_PT_ORDER]
    if invalid_pt_values:
        raise ValueError(f"Unsupported pt values: {invalid_pt_values}. Supported values: {RAW_PT_ORDER}")

    type_folders = filter_type_folders(list_type_folders(input_root), args.materials)
    if not type_folders:
        raise RuntimeError(f"No battery type folders found under: {input_root}")

    for type_folder in type_folders:
        process_type_folder(type_folder, output_root, soc_values, pt_values, row_shift=args.u_t_row_shift)

    print("[FINISHED] Step2 completed.")


if __name__ == "__main__":
    main()
