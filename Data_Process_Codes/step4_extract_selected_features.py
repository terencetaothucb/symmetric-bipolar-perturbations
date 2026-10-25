"""
Step 4: compute the selected modeling features from Step3 aggregated workbooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import (
    c_rate_label,
    ensure_dir,
    filter_type_folders,
    is_relevant_sheet_name,
    list_excel_files,
    list_type_folders,
    normalize_columns,
    parse_step3_file_metadata,
    resolve_project_path,
)


C_RATES_ABS = [0.5, 1.0, 1.5, 2.0, 2.5]
# Each C-rate and polarity maps to four Step3 voltage checkpoints:
# pre-pulse, step response, end-of-pulse, and relaxation voltage.
U_STARTS = {
    (0.5, +1): 1,
    (0.5, -1): 5,
    (1.0, +1): 9,
    (1.0, -1): 13,
    (1.5, +1): 17,
    (1.5, -1): 21,
    (2.0, +1): 25,
    (2.0, -1): 29,
    (2.5, +1): 33,
    (2.5, -1): 37,
}
NEXT_PRE_AFTER_NEG = {
    0.5: "U9",
    1.0: "U17",
    1.5: "U25",
    2.0: "U33",
    2.5: "U41",
}


def make_u_mapping() -> Dict[tuple, Dict[str, str]]:
    """Map each C-rate and polarity to the corresponding Step3 voltage columns."""
    mapping: Dict[tuple, Dict[str, str]] = {}
    for c_rate in C_RATES_ABS:
        for sign in (+1, -1):
            start = U_STARTS[(c_rate, sign)]
            mapping[(c_rate, sign)] = {
                "pre": f"U{start}",
                "step": f"U{start + 1}",
                "end": f"U{start + 2}",
                "rel": f"U{start + 3}",
            }
    return mapping


U_MAP = make_u_mapping()
FEATURE_PREFIXES_TO_DROP = ("Hyst_M3_", "fai_irrev_", "R0_", "Reff_", "E_loss_", "Eloss_proxy_")


def compute_selected_features(data_frame: pd.DataFrame, pulse_width_seconds: float) -> pd.DataFrame:
    """Append fai_irrev, resistance, and loss features to one Step3 sheet.

    For each pulse polarity, R0 uses the instantaneous step voltage and Reff
    uses the end-of-pulse voltage:

        R0   = abs(U_step - U_pre) / abs(I)
        Reff = abs(U_end  - U_pre) / abs(I)

    where abs(I) = C-rate * Qn.  The scalar manuscript comparisons use the
    arithmetic mean of the positive- and negative-pulse resistance values:

        R0_avg   = mean(R0_positive_pulse, R0_negative_pulse)
        Reff_avg = mean(Reff_positive_pulse, Reff_negative_pulse)

    Only the scalar ``*_avg_*`` resistance columns are stored in Step4. The
    polarity-specific resistance values are temporary calculation variables
    and are not written to the output workbook.
    """
    data_frame = data_frame.copy()
    data_frame.columns = normalize_columns(data_frame.columns)
    for u_index in range(1, 42):
        column = f"U{u_index}"
        if column not in data_frame.columns:
            raise KeyError(f"Missing required column: {column}")
    if "Qn" not in data_frame.columns:
        raise KeyError("Missing required column: Qn")

    qn = pd.to_numeric(data_frame["Qn"], errors="coerce").astype(float).values
    appended_columns: List[str] = []
    average_columns: List[str] = []

    for c_rate in C_RATES_ABS:
        label = c_rate_label(c_rate)

        pre_pos = pd.to_numeric(data_frame[U_MAP[(c_rate, +1)]["pre"]], errors="coerce").astype(float).values
        pre_neg = pd.to_numeric(data_frame[U_MAP[(c_rate, -1)]["pre"]], errors="coerce").astype(float).values
        next_pre = pd.to_numeric(data_frame[NEXT_PRE_AFTER_NEG[c_rate]], errors="coerce").astype(float).values

        fai_irrev = (np.abs(next_pre - pre_neg) + np.abs(pre_neg - pre_pos)) / 2.0
        fai_irrev_name = f"fai_irrev_{label}"
        data_frame[fai_irrev_name] = fai_irrev
        appended_columns.append(fai_irrev_name)

        # R0 uses the instantaneous step response; Reff uses the end-of-pulse
        # response. Only their two-polarity arithmetic means are persisted.
        # E_loss remains polarity-specific because it is a separate diagnostic
        # quantity rather than one of the scalar resistance comparisons.
        current_amperes = c_rate * qn
        r0_by_sign: Dict[int, np.ndarray] = {}
        reff_by_sign: Dict[int, np.ndarray] = {}
        for sign in (+1, -1):
            polarity = "p" if sign > 0 else "n"
            pre_column = U_MAP[(c_rate, sign)]["pre"]
            step_column = U_MAP[(c_rate, sign)]["step"]
            end_column = U_MAP[(c_rate, sign)]["end"]

            u_pre = pd.to_numeric(data_frame[pre_column], errors="coerce").astype(float).values
            u_step = pd.to_numeric(data_frame[step_column], errors="coerce").astype(float).values
            u_end = pd.to_numeric(data_frame[end_column], errors="coerce").astype(float).values
            delta_v_step = u_step - u_pre
            delta_v_end = u_end - u_pre

            r0 = np.where(current_amperes != 0, np.abs(delta_v_step) / current_amperes, np.nan)
            reff = np.where(current_amperes != 0, np.abs(delta_v_end) / current_amperes, np.nan)
            e_loss = np.where(np.isfinite(reff), (current_amperes ** 2) * reff * pulse_width_seconds, np.nan)

            e_loss_name = f"E_loss_{polarity}_{label}"
            data_frame[e_loss_name] = e_loss
            r0_by_sign[sign] = r0
            reff_by_sign[sign] = reff
            appended_columns.append(e_loss_name)

        r0_avg_name = f"R0_avg_{label}"
        reff_avg_name = f"Reff_avg_{label}"
        data_frame[r0_avg_name] = (r0_by_sign[+1] + r0_by_sign[-1]) / 2.0
        data_frame[reff_avg_name] = (reff_by_sign[+1] + reff_by_sign[-1]) / 2.0
        average_columns.extend([r0_avg_name, reff_avg_name])

    base_columns = [column for column in data_frame.columns if not column.startswith(FEATURE_PREFIXES_TO_DROP)]
    # Preserve the historical non-resistance columns and append only the
    # scalar-average R0/Reff inputs used by the manuscript configurations.
    return data_frame.loc[:, base_columns + appended_columns + average_columns]


def process_workbook(input_file: Path, output_file: Path) -> None:
    """Process one Step3 workbook and preserve non-target sheets unchanged."""
    metadata = parse_step3_file_metadata(input_file.name)
    pulse_width_seconds = metadata["pulse_width_ms"] / 1000.0
    workbook = pd.ExcelFile(input_file, engine="openpyxl")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name in workbook.sheet_names:
            data_frame = workbook.parse(sheet_name)
            if is_relevant_sheet_name(sheet_name):
                output_frame = compute_selected_features(data_frame, pulse_width_seconds)
            else:
                output_frame = data_frame
            output_frame.to_excel(writer, sheet_name=sheet_name, index=False)


def process_type_folder(type_folder: Path, output_root: Path, overwrite: bool) -> None:
    """Generate Step4 workbooks for one material folder."""
    output_folder = output_root / type_folder.name
    ensure_dir(output_folder)
    input_files = list_excel_files(type_folder)

    for index, input_file in enumerate(input_files, start=1):
        output_file = output_folder / input_file.name
        if output_file.exists() and not overwrite:
            print(f"[SKIP] {output_file}")
            continue
        process_workbook(input_file, output_file)
        print(f"[SAVE] ({index}/{len(input_files)}) {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step4: compute fai_irrev, scalar-average R0/Reff, and E_loss from Step3 workbooks."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing Step3 aggregated workbooks.")
    parser.add_argument("--output-root", required=True, help="Root directory for Step4 selected-feature outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Step4 files.")
    parser.add_argument("--materials", nargs="*", default=None, help="Optional battery type folder names to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = resolve_project_path(args.input_root)
    output_root = resolve_project_path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_dir(output_root)
    type_folders = filter_type_folders(list_type_folders(input_root), args.materials)
    if not type_folders:
        raise RuntimeError(f"No battery type folders found under: {input_root}")

    for type_folder in type_folders:
        process_type_folder(type_folder, output_root, overwrite=args.overwrite)

    print("[FINISHED] Step4 completed.")


if __name__ == "__main__":
    main()
