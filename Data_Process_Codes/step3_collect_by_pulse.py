"""
Step 3: regroup Step2 cell-level outputs into one workbook per pulse width.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import DEFAULT_PT_VALUES, DEFAULT_SOC_VALUES, DEFAULT_U_INDEXES
from utils import (
    ensure_dir,
    filter_type_folders,
    infer_output_prefix_from_type_folder,
    is_train_file_by_name,
    list_excel_files,
    list_type_folders,
    parse_float_list,
    parse_int_list,
)


HEADER_ITEMS = (
    ["File_Name", "Mat", "No.", "ID", "Qn", "Q", "SOH", "Pt", "SOC", "SOCR"]
    + [f"U{i}" for i in DEFAULT_U_INDEXES]
)


def aggregate_one_type_folder(type_folder: Path, output_root: Path, soc_values: List[int], pt_values: List[float], overwrite: bool) -> None:
    """Build Step3 pulse-width workbooks for one material folder."""
    output_folder = output_root / type_folder.name
    ensure_dir(output_folder)

    pt_values_ms = [int(round(value * 1000)) for value in pt_values]
    # Bucket 0 stores the merged "SOC ALL" rows, while buckets 1..N store
    # the individual SOC sheets in the order of soc_values.
    per_pt_data: Dict[int, List[List[list]]] = {
        value_ms: [[] for _ in range(len(soc_values) + 1)] for value_ms in pt_values_ms
    }

    input_files = list_excel_files(type_folder)
    output_prefix = infer_output_prefix_from_type_folder(type_folder.name)
    print(f"[INFO] {type_folder.name}: {len(input_files)} files")

    for index, input_file in enumerate(input_files, start=1):
        data_frame = pd.read_excel(input_file, engine="openpyxl")
        is_train = is_train_file_by_name(input_file.name)

        for _, row in data_frame.iterrows():
            try:
                pt_ms = int(round(float(row["Pt"]) * 1000))
            except Exception:
                continue

            if pt_ms not in per_pt_data:
                continue

            row_list = row.reindex(HEADER_ITEMS).tolist()
            per_pt_data[pt_ms][0].append(row_list)

            if is_train:
                try:
                    soc_value = int(round(float(row["SOC"])))
                except Exception:
                    soc_value = None
                if soc_value in soc_values:
                    soc_index = soc_values.index(soc_value) + 1
                    per_pt_data[pt_ms][soc_index].append(row_list)

        print(f"[READ] ({index}/{len(input_files)}) {input_file.name}")

    for pt_value in pt_values:
        pt_ms = int(round(pt_value * 1000))
        output_file = output_folder / f"{output_prefix}{pt_ms}.xlsx"
        if output_file.exists() and not overwrite:
            print(f"[SKIP] {output_file}")
            continue

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            pd.DataFrame(per_pt_data[pt_ms][0], columns=HEADER_ITEMS).to_excel(writer, sheet_name="SOC ALL", index=False)

            for soc_value in soc_values:
                bucket_index = soc_values.index(soc_value) + 1
                rows = per_pt_data[pt_ms][bucket_index]
                if not rows:
                    rows = [[None] * len(HEADER_ITEMS)]
                pd.DataFrame(rows, columns=HEADER_ITEMS).to_excel(
                    writer,
                    sheet_name=f"SOC{soc_value}",
                    index=False,
                )

        print(f"[SAVE] {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step3: aggregate Step2 per-cell outputs into per-pulse-width workbooks.")
    parser.add_argument("--input-root", required=True, help="Root directory containing Step2 output type folders.")
    parser.add_argument("--output-root", required=True, help="Root directory for Step3 aggregated workbooks.")
    parser.add_argument("--soc-values", default=None, help="Comma-separated SOC values to keep as fixed sheets.")
    parser.add_argument("--pt-values", default=None, help="Comma-separated pulse widths in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Step3 files.")
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
    type_folders = filter_type_folders(list_type_folders(input_root), args.materials)
    if not type_folders:
        raise RuntimeError(f"No battery type folders found under: {input_root}")

    for type_folder in type_folders:
        aggregate_one_type_folder(type_folder, output_root, soc_values, pt_values, overwrite=args.overwrite)

    print("[FINISHED] Step3 completed.")


if __name__ == "__main__":
    main()
