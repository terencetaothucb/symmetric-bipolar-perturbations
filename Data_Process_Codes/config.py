"""
Shared constants and sheet-matching rules for the data-processing pipeline.
"""

from __future__ import annotations

import re
from typing import List


DEFAULT_SOC_VALUES: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
DEFAULT_PT_VALUES: List[float] = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
DEFAULT_U_INDEXES: List[int] = list(range(1, 42))

# Additional manuscript-cohort exclusions supported by evidence outside the
# released fixed-SOC raw-data subset. The main Step1-Step4 pipeline preserves
# the complete released dataset. The optional SOH cohort utility applies these
# rules together with the requested SOH threshold.
EXCLUDED_CELLS: List[dict] = [
    {
        "type_folder": "21Ah NMC",
        "no": 34,
        "id": "02LCC02100101A87Y0013547",
        "reason": "SOH > 1 in the historical random-SOC test",
        "reference_soc": 41.6497,
        "reference_q_ah": 21.0045,
        "reference_soh": 1.0002142857,
    }
]

WORKSTEP_SHEET_NAME_CANDIDATES: List[str] = [
    "\u5de5\u6b65\u5c42",
    "\u5de5\u4f5c\u6b65\u5c42",
    "Workstep_Layer",
]

RELEVANT_SHEET_NAME_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"^SOC\s*ALL$", flags=re.IGNORECASE),
    re.compile(r"^SOC\d+$", flags=re.IGNORECASE),
]
