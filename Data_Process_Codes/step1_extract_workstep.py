"""
Step 1: extract the workstep sheet from each raw workbook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config import WORKSTEP_SHEET_NAME_CANDIDATES
from utils import ensure_dir, filter_type_folders, is_first_part_file, list_excel_files, list_type_folders


def resolve_sheet_name(excel_path: Path, preferred_name: str | None, preferred_index: int | None) -> str | int:
    """Resolve the workstep sheet by preferred name, known aliases, or fallback index."""
    workbook = pd.ExcelFile(excel_path, engine="openpyxl")

    if preferred_name and preferred_name in workbook.sheet_names:
        return preferred_name

    for candidate in WORKSTEP_SHEET_NAME_CANDIDATES:
        if candidate in workbook.sheet_names:
            return candidate

    if preferred_index is not None and 0 <= preferred_index < len(workbook.sheet_names):
        return workbook.sheet_names[preferred_index]

    raise ValueError(f"Cannot resolve workstep sheet for {excel_path}")


def process_type_folder(type_folder: Path, output_root: Path, sheet_name: str | None, sheet_index: int | None) -> None:
    """Extract the workstep sheet for every first-part file in one material folder."""
    output_folder = output_root / type_folder.name
    ensure_dir(output_folder)

    input_files = [path for path in list_excel_files(type_folder) if is_first_part_file(path.name)]
    print(f"[INFO] {type_folder.name}: {len(input_files)} files")

    for index, input_file in enumerate(input_files, start=1):
        chosen_sheet = resolve_sheet_name(input_file, sheet_name, sheet_index)
        data = pd.read_excel(input_file, sheet_name=chosen_sheet, engine="openpyxl")
        output_file = output_folder / input_file.name
        data.to_excel(output_file, index=False, engine="openpyxl")
        print(f"[SAVE] ({index}/{len(input_files)}) {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step1: extract the workstep sheet from raw PulseBat files.")
    parser.add_argument("--input-root", required=True, help="Root directory containing raw battery type folders.")
    parser.add_argument("--output-root", required=True, help="Root directory for extracted workstep files.")
    parser.add_argument("--sheet-name", default=None, help="Preferred sheet name. If missing, fall back to known names and then sheet index.")
    parser.add_argument("--sheet-index", type=int, default=2, help="Fallback sheet index used when the workstep sheet name is unavailable.")
    parser.add_argument("--materials", nargs="*", default=None, help="Optional battery type folder names to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_dir(output_root)
    type_folders = filter_type_folders(list_type_folders(input_root), args.materials)
    if not type_folders:
        raise RuntimeError(f"No battery type folders found under: {input_root}")

    for type_folder in type_folders:
        process_type_folder(type_folder, output_root, args.sheet_name, args.sheet_index)

    print("[FINISHED] Step1 completed.")


if __name__ == "__main__":
    main()
