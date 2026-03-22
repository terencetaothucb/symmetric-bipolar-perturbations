# -*- coding: utf-8 -*-
"""
Experiment_AllMaterials_SOC405070_W5000_FaiIrrevOnly_byCRate.py

Purpose:
- Run the fai_irrev-only experiment on all material folders under the Step4 output root.
- Use width_ms=5000.
- Use soc in {40, 50, 70}.
- Keep only one C-rate per run via keep_regex.
"""

from __future__ import annotations

import copy
from pathlib import Path

try:
    import PulseBat.Model_Codes.Model_backbone as bench
except ImportError:
    import Model_backbone as bench


DEFAULT_DATA_ROOT = Path(r"E:\Datasets\PulseBat_all\Data\Data_Process_Output\Step4_SelectedFeatures")
DEFAULT_OUT_DIR = Path(r"E:\Datasets\PulseBat_all\Data\Model_Output")


def crate_to_keep_regex(crate_str: str) -> str:
    """
    Convert a C-rate label into a regex used by build_feature_list(keep_regex=...).

    Examples:
      "0.5C" -> r"_0\.5C$"
      "1C" -> r"_1C$"
      "2.5C" -> r"_2\.5C$"
    """
    crate_str = crate_str.strip()
    if crate_str.endswith("C"):
        core = crate_str[:-1]
    else:
        core = crate_str
    core = core.replace(".", r"\.")
    return rf"_{core}C$"


def main() -> None:
    # -----------------------------
    # Fixed experimental settings
    # -----------------------------
    width_ms = 5000
    soc_list = [50, 40, 70]

    data_root = DEFAULT_DATA_ROOT
    out_dir_root = DEFAULT_OUT_DIR
    if not data_root.is_dir():
        raise FileNotFoundError(f"Step4 data root not found: {data_root}")

    materials = []
    for material_dir in sorted([path for path in data_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        if list(material_dir.glob(f"*_W_{width_ms}.xlsx")):
            materials.append(material_dir.name)

    feature_combo = "fai_irrev_Only"

    # Each run keeps only one C-rate from the selected feature family.
    # Reduce this list to run fewer cases, for example only ["1C"].
    crate_list = ["0.5C", "1C", "1.5C", "2C", "2.5C"]

    # -----------------------------
    # Global knobs
    # -----------------------------
    MODELS = ["linear", "ridge", "lasso", "en", "svm", "rf", "xgb", "gpr", "mlp", "transformer", "informer"]
    SEEDS = list(range(100))  # For quick checks, reduce this list, for example list(range(5)).
    TEST_SIZE = 0.2
    SHOW_PROGRESS = True

    bench_path = Path(bench.__file__).resolve()
    if not bench_path.exists():
        raise FileNotFoundError(f"Cannot find imported benchmark file at: {bench_path}")

    print("=== TestCase Matrix Runner (Single C-rate features) ===")
    print(f"Benchmark script: {bench_path}")
    print(f"Data root: {data_root}")
    print(f"Output root: {out_dir_root}")
    print(f"Materials: {materials}")
    print(f"SOCs: {soc_list}")
    print(f"Feature combo: {feature_combo}")
    print(f"C-rates: {crate_list}  (each run uses one keep_regex)")
    print(f"width_ms={width_ms}")
    print(f"Models: {MODELS}")
    print(f"Seeds: {len(SEEDS)} runs")
    print("========================================================\n")

    total = len(materials) * len(soc_list) * len(crate_list)
    job = 0

    for mat in materials:
        for soc in soc_list:
            for crate in crate_list:
                job += 1
                keep_regex = crate_to_keep_regex(crate)
                print(f"\n[{job}/{total}] RUN => material='{mat}', SOC={soc}, combo='{feature_combo}', crate='{crate}', keep_regex='{keep_regex}'")

                cfg = copy.deepcopy(bench.CONFIG)

                cfg["data_root"] = str(data_root)
                cfg["out_dir"] = str(out_dir_root)
                cfg["show_progress"] = SHOW_PROGRESS
                cfg["material"] = mat
                cfg["width_ms"] = width_ms
                cfg["soc"] = soc
                cfg["test_size"] = TEST_SIZE
                cfg["seeds"] = SEEDS
                cfg["models"] = MODELS
                cfg["feature_spec"] = dict(
                    name=f"{feature_combo}__{crate}",
                    combo=feature_combo,
                    include_groups=None,
                    exclude_groups=None,
                    add_features=None,
                    drop_features=None,
                    keep_regex=keep_regex,
                )

                try:
                    output_dir = bench.run(cfg)
                    print(f"[DONE] output: {output_dir}")
                except Exception as exc:
                    print(f"[SKIP] material='{mat}', SOC={soc}, crate='{crate}' -> {type(exc).__name__}: {exc}")

    print("\n=== ALL MATRIX RUNS FINISHED ===")


if __name__ == "__main__":
    main()
