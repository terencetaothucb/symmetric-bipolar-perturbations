"""
Run the main PulseBat processing pipeline from Step1 through Step4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_PT_VALUES, DEFAULT_SOC_VALUES
from step1_extract_workstep import process_type_folder as run_step1_type
from step2_extract_pulse_features import process_type_folder as run_step2_type
from step3_collect_by_pulse import aggregate_one_type_folder as run_step3_type
from step4_extract_selected_features import process_type_folder as run_step4_type
from utils import ensure_dir, filter_type_folders, list_type_folders, parse_float_list, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the organized PulseBat main pipeline: step1 -> step4.")
    parser.add_argument("--raw-root", required=True, help="Raw data root containing battery type folders.")
    parser.add_argument("--step1-root", required=True, help="Output root for Step1.")
    parser.add_argument("--step2-root", required=True, help="Output root for Step2.")
    parser.add_argument("--step3-root", required=True, help="Output root for Step3.")
    parser.add_argument("--step4-root", required=True, help="Output root for Step4.")
    parser.add_argument("--sheet-name", default=None, help="Preferred Step1 sheet name.")
    parser.add_argument("--sheet-index", type=int, default=2, help="Fallback Step1 sheet index.")
    parser.add_argument("--soc-values", default=None, help="Comma-separated SOC values.")
    parser.add_argument("--pt-values", default=None, help="Comma-separated pulse widths in seconds.")
    parser.add_argument("--u-t-row-shift", type=int, default=0, help="Optional Step2 row shift for special files.")
    parser.add_argument("--materials", nargs="*", default=None, help="Optional battery type folder names to process.")
    parser.add_argument("--overwrite-step3", action="store_true", help="Overwrite existing Step3 files.")
    parser.add_argument("--overwrite-step4", action="store_true", help="Overwrite existing Step4 files.")
    return parser


def main() -> None:
    """Parse CLI arguments and run the four processing steps in sequence."""
    args = build_parser().parse_args()

    raw_root = Path(args.raw_root)
    step1_root = Path(args.step1_root)
    step2_root = Path(args.step2_root)
    step3_root = Path(args.step3_root)
    step4_root = Path(args.step4_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    soc_values = parse_int_list(args.soc_values, DEFAULT_SOC_VALUES)
    pt_values = parse_float_list(args.pt_values, DEFAULT_PT_VALUES)
    type_folders = filter_type_folders(list_type_folders(raw_root), args.materials)
    if not type_folders:
        raise RuntimeError(f"No battery type folders found under: {raw_root}")

    for root in [step1_root, step2_root, step3_root, step4_root]:
        ensure_dir(root)

    for type_folder in type_folders:
        run_step1_type(type_folder, step1_root, args.sheet_name, args.sheet_index)
        run_step2_type(step1_root / type_folder.name, step2_root, soc_values, pt_values, args.u_t_row_shift)
        run_step3_type(step2_root / type_folder.name, step3_root, soc_values, pt_values, args.overwrite_step3)
        run_step4_type(step3_root / type_folder.name, step4_root, args.overwrite_step4)

    print("[FINISHED] Full pipeline completed.")


if __name__ == "__main__":
    main()
