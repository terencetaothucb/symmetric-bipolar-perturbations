"""
Shared helper functions for directory handling, filename parsing, and sheet matching.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

from config import RELEVANT_SHEET_NAME_REGEXES


def ensure_dir(path: Path) -> None:
    """Create a directory and all missing parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str | None, default: Sequence[int]) -> List[int]:
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str | None, default: Sequence[float]) -> List[float]:
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def list_type_folders(root: Path) -> List[Path]:
    folders = [path for path in root.iterdir() if path.is_dir()]
    return sorted(folders, key=lambda path: path.name.lower())


def filter_type_folders(type_folders: Iterable[Path], selected_names: Sequence[str] | None) -> List[Path]:
    if not selected_names:
        return list(type_folders)
    selected = {name.strip().lower() for name in selected_names if name.strip()}
    return [folder for folder in type_folders if folder.name.lower() in selected]


def list_excel_files(folder: Path) -> List[Path]:
    """List regular .xlsx files while skipping temporary artifacts."""
    temp_markers = (".__notefix__.", ".__translate__.", ".__tmp__")
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".xlsx"
            and not path.name.startswith("~$")
            and not any(marker in path.name for marker in temp_markers)
        ],
        key=lambda path: path.name.lower(),
    )


def is_first_part_file(file_name: str) -> bool:
    tokens = file_name.split("_")
    if len(tokens) < 9:
        return True
    return tokens[8].split("-")[0] == "1"


def is_train_file_by_name(file_name: str) -> bool:
    tokens = file_name.split("_")
    if len(tokens) < 7:
        return True
    return len(tokens[6].split("-")) == 2


def infer_output_prefix_from_type_folder(type_folder_name: str) -> str:
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*Ah\s+([A-Za-z0-9]+)\s*$", type_folder_name.strip())
    if match:
        capacity = match.group(1) + "Ah"
        chemistry = match.group(2).upper()
        return f"{chemistry}_{capacity}_W_"
    return type_folder_name.replace(" ", "_") + "_W_"


def parse_step_source_metadata(file_name: str) -> dict:
    """Parse the metadata tokens encoded in a Step1/Step2 source filename."""
    tokens = file_name.split("_")
    if len(tokens) < 11:
        raise ValueError(f"Unexpected source filename format: {file_name}")

    return {
        "file_name": file_name,
        "mat": tokens[0],
        "qn": int(tokens[2]),
        "no": int(tokens[4]),
        "soc_token": tokens[6],
        "id": tokens[10].removesuffix(".xlsx"),
    }


def parse_step3_file_metadata(file_name: str) -> dict:
    """Parse chemistry, capacity, and pulse width from a Step3 workbook name."""
    match = re.match(r"^(?P<chem>[A-Za-z0-9]+)_(?P<cap>\d+(?:\.\d+)?)Ah_W_(?P<w>\d+)\.xlsx$", file_name)
    if not match:
        raise ValueError(f"Unexpected Step3 filename format: {file_name}")
    return {
        "chem": match.group("chem").upper(),
        "capacity_ah": float(match.group("cap")),
        "pulse_width_ms": int(match.group("w")),
    }


def c_rate_label(c_rate: float) -> str:
    if abs(c_rate - round(c_rate)) < 1e-9:
        return f"{int(round(c_rate))}C"
    return f"{c_rate:g}C"


def normalize_columns(columns: Sequence[object]) -> List[str]:
    return [str(column).strip() for column in columns]


def is_relevant_sheet_name(sheet_name: str) -> bool:
    stripped = sheet_name.strip()
    return any(pattern.match(stripped) for pattern in RELEVANT_SHEET_NAME_REGEXES)
