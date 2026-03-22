"""
Shared constants and sheet-matching rules for the data-processing pipeline.
"""

from __future__ import annotations

import re
from typing import List


DEFAULT_SOC_VALUES: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
DEFAULT_PT_VALUES: List[float] = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
DEFAULT_U_INDEXES: List[int] = list(range(1, 42))

WORKSTEP_SHEET_NAME_CANDIDATES: List[str] = [
    "\u5de5\u6b65\u5c42",
    "\u5de5\u4f5c\u6b65\u5c42",
    "Workstep_Layer",
]

RELEVANT_SHEET_NAME_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"^SOC\s*ALL$", flags=re.IGNORECASE),
    re.compile(r"^SOC\d+$", flags=re.IGNORECASE),
]
