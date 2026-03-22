from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Try the most common extracted-sheet names first, then fall back to the first
# sheet if none of them are present.
PREFERRED_SHEETS = ["Sheet1", "工作表1", "工步层", "工作步层", "Workstep_Layer"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def pick_sheet_name(excel_path: Path) -> str:
    workbook = pd.ExcelFile(excel_path, engine="openpyxl")
    for candidate in PREFERRED_SHEETS:
        if candidate in workbook.sheet_names:
            return candidate
    return workbook.sheet_names[0]


def find_required_columns(columns: List[str]) -> Tuple[str, str, Optional[str]]:
    normalized = {normalize_colname(column): column for column in columns}

    def pick(candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if normalize_colname(candidate) in normalized:
                return normalized[normalize_colname(candidate)]
        return None

    no_column = pick(["No", "No.", "CellNo", "BatteryNo"])
    soh_column = pick(["SOH"])
    id_column = pick(["ID", "CellID"])

    if no_column is None or soh_column is None:
        raise KeyError(f"Required columns not found. Columns: {columns[:30]}")

    return no_column, soh_column, id_column


def list_type_folders(root: Path) -> List[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())


def list_excel_files(folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
        ],
        key=lambda path: path.name.lower(),
    )


def detect_type_summary(type_folder: Path, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    excel_files = list_excel_files(type_folder)
    cell_info: Dict[int, Dict[str, object]] = {}

    for excel_file in excel_files:
        sheet_name = pick_sheet_name(excel_file)
        header_frame = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0, engine="openpyxl")
        no_column, soh_column, id_column = find_required_columns(list(header_frame.columns))
        usecols = [column for column in [no_column, soh_column, id_column] if column is not None]

        data_frame = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=usecols, engine="openpyxl")
        data_frame[no_column] = pd.to_numeric(data_frame[no_column], errors="coerce")
        data_frame[soh_column] = pd.to_numeric(data_frame[soh_column], errors="coerce")
        data_frame = data_frame.dropna(subset=[no_column])
        data_frame[no_column] = data_frame[no_column].astype(int)

        soh_by_no = data_frame.groupby(no_column, dropna=True)[soh_column].max()
        if id_column is not None:
            id_by_no = data_frame.groupby(no_column, dropna=True)[id_column].first()
        else:
            id_by_no = pd.Series(dtype=str)

        for no_value, soh_value in soh_by_no.items():
            if pd.isna(soh_value):
                continue
            if no_value not in cell_info:
                cell_info[no_value] = {
                    "TypeFolder": type_folder.name,
                    "No": int(no_value),
                    "ID": str(id_by_no.get(no_value, "")).strip(),
                    "MaxSOH": float(soh_value),
                    "FilesOverThreshold": [],
                }

            cell_info[no_value]["MaxSOH"] = max(float(cell_info[no_value]["MaxSOH"]), float(soh_value))
            if float(soh_value) > threshold:
                cell_info[no_value]["FilesOverThreshold"].append(excel_file.name)

    per_cell = pd.DataFrame(sorted(cell_info.values(), key=lambda row: row["No"]))
    if per_cell.empty:
        return per_cell, per_cell

    per_cell["FilesOverThreshold"] = per_cell["FilesOverThreshold"].apply(lambda values: ",".join(sorted(set(values))))
    excluded = per_cell[per_cell["MaxSOH"] > threshold].copy()
    return per_cell, excluded


def rebuild_filtered_dataset(type_folder: Path, output_root: Path, excluded_numbers: set[int]) -> None:
    output_folder = output_root / type_folder.name
    ensure_dir(output_folder)

    for excel_file in list_excel_files(type_folder):
        sheet_name = pick_sheet_name(excel_file)
        data_frame = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")
        no_column, _, _ = find_required_columns(list(data_frame.columns))
        data_frame[no_column] = pd.to_numeric(data_frame[no_column], errors="coerce")
        filtered = data_frame[~data_frame[no_column].isin(excluded_numbers)].copy()
        filtered.to_excel(output_folder / excel_file.name, index=False, engine="openpyxl")


def plot_soh_distributions(type_to_soh: Dict[str, np.ndarray], output_file: Path) -> None:
    if not type_to_soh:
        return

    n_types = len(type_to_soh)
    ncols = 2
    nrows = int(np.ceil(n_types / ncols))
    plt.figure(figsize=(6 * ncols, 4 * nrows))

    for index, type_name in enumerate(sorted(type_to_soh), start=1):
        plt.subplot(nrows, ncols, index)
        values = type_to_soh[type_name]
        values = values[np.isfinite(values)]
        if values.size > 0:
            plt.hist(values, bins=np.arange(0.0, 1.21, 0.02))
        plt.title(type_name)
        plt.xlabel("SOH")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional utility: summarize SOH>1 cells by battery type.")
    parser.add_argument("--input-root", required=True, help="Root folder containing battery type folders.")
    parser.add_argument("--report-root", required=True, help="Directory for summary tables and plots.")
    parser.add_argument("--threshold", type=float, default=1.0, help="SOH threshold. Default: 1.0")
    parser.add_argument(
        "--rebuild-output-root",
        default=None,
        help="Optional output root. If given, rebuild a filtered copy with cells over the threshold removed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    report_root = Path(args.report_root)
    rebuild_output_root = Path(args.rebuild_output_root) if args.rebuild_output_root else None

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_dir(report_root)
    if rebuild_output_root is not None:
        ensure_dir(rebuild_output_root)

    summary_rows: List[dict] = []
    excluded_rows: List[pd.DataFrame] = []
    type_to_soh: Dict[str, np.ndarray] = {}

    for type_folder in list_type_folders(input_root):
        per_cell, excluded = detect_type_summary(type_folder, args.threshold)
        total_count = 0 if per_cell.empty else len(per_cell)
        excluded_count = 0 if excluded.empty else len(excluded)

        summary_rows.append(
            {
                "TypeFolder": type_folder.name,
                "QuantityTotal": total_count,
                "QuantitySOH_GT_Threshold": excluded_count,
                "Threshold": args.threshold,
            }
        )
        if not per_cell.empty:
            type_to_soh[type_folder.name] = per_cell["MaxSOH"].to_numpy(dtype=float)
        if not excluded.empty:
            excluded_rows.append(excluded)
        if rebuild_output_root is not None and not excluded.empty:
            rebuild_filtered_dataset(type_folder, rebuild_output_root, set(excluded["No"].tolist()))

        print(f"[INFO] {type_folder.name}: total={total_count}, soh_gt_threshold={excluded_count}")

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_excel(report_root / "summary_quantity_soh_gt1.xlsx", index=False, engine="openpyxl")
    summary_frame.to_csv(report_root / "summary_quantity_soh_gt1.csv", index=False, encoding="utf-8-sig")

    if excluded_rows:
        excluded_frame = pd.concat(excluded_rows, ignore_index=True)
    else:
        excluded_frame = pd.DataFrame(columns=["TypeFolder", "No", "ID", "MaxSOH", "FilesOverThreshold"])
    excluded_frame.to_excel(report_root / "excluded_cells.xlsx", index=False, engine="openpyxl")
    excluded_frame.to_csv(report_root / "excluded_cells.csv", index=False, encoding="utf-8-sig")

    plot_soh_distributions(type_to_soh, report_root / "soh_hist_grid_all_types.png")
    print("[FINISHED] SOH summary completed.")


if __name__ == "__main__":
    main()
