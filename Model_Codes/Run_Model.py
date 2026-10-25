# -*- coding: utf-8 -*-
"""
Manuscript feature-comparison matrix runner.

Purpose:
- Run the manuscript feature cases on all material folders under the R0-augmented Step4 root.
- Use width_ms=5000.
- Use soc in {40, 50, 70}.
- Use the exact single-C-rate columns declared by Model_backbone.paper_feature_spec().
- Benchmark the ten released models.
"""

from __future__ import annotations

import copy
from pathlib import Path

try:
    import PulseBat.Model_Codes.Model_backbone as bench
except ImportError:
    import Model_backbone as bench


# Relative paths are resolved from the repository root rather than cwd.
DEFAULT_DATA_ROOT = Path("Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff")
DEFAULT_OUT_DIR = Path("Data/Model_Output")


def main() -> None:
    # -----------------------------
    # Fixed experimental settings
    # -----------------------------
    width_ms = 5000
    soc_list = [50, 40, 70]

    project_root = bench.find_project_root(Path(__file__).resolve().parent)
    data_root = (project_root / DEFAULT_DATA_ROOT.expanduser()).resolve()
    out_dir_root = (project_root / DEFAULT_OUT_DIR.expanduser()).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Step4 data root not found: {data_root}")

    materials = []
    for material_dir in sorted([path for path in data_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        if list(material_dir.glob(f"*_W_{width_ms}.xlsx")):
            materials.append(material_dir.name)

    # These are the six feature configurations reported in the manuscript.
    # Keep this list synchronized with Model_backbone.PAPER_FEATURE_CASES.
    experiment_cases = list(bench.PAPER_FEATURE_CASES)

    # Each run uses one C-rate-specific input definition.
    crate_list = ["0.5C", "1C", "1.5C", "2C", "2.5C"]

    # -----------------------------
    # Global knobs
    # -----------------------------
    MODELS = ["linear", "ridge", "lasso", "en", "svm", "rf", "gpr", "mlp", "transformer", "informer"]
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
    print(f"Feature cases: {experiment_cases}")
    print(f"C-rates: {crate_list}")
    print(f"width_ms={width_ms}")
    print(f"Models: {MODELS}")
    print(f"Seeds: {len(SEEDS)} runs")
    print("========================================================\n")

    total = len(materials) * len(soc_list) * len(experiment_cases) * len(crate_list)
    job = 0

    for mat in materials:
        for soc in soc_list:
            for case_name in experiment_cases:
                for crate in crate_list:
                    job += 1
                    feature_spec = bench.paper_feature_spec(case_name, crate)
                    feature_columns = feature_spec["add_features"]
                    print(
                        f"\n[{job}/{total}] RUN => material='{mat}', SOC={soc}, "
                        f"case='{case_name}', crate='{crate}', features={feature_columns}"
                    )

                    cfg = copy.deepcopy(bench.CONFIG)
                    cfg["data_root"] = DEFAULT_DATA_ROOT.as_posix()
                    cfg["out_dir"] = DEFAULT_OUT_DIR.as_posix()
                    cfg["show_progress"] = SHOW_PROGRESS
                    cfg["material"] = mat
                    cfg["width_ms"] = width_ms
                    cfg["soc"] = soc
                    cfg["test_size"] = TEST_SIZE
                    cfg["seeds"] = SEEDS
                    cfg["models"] = MODELS
                    cfg["feature_spec"] = feature_spec

                    try:
                        output_dir = bench.run(cfg)
                        print(f"[DONE] output: {output_dir}")
                    except Exception as exc:
                        print(
                            f"[SKIP] material='{mat}', SOC={soc}, case='{case_name}', "
                            f"crate='{crate}' -> {type(exc).__name__}: {exc}"
                        )

    print("\n=== ALL MATRIX RUNS FINISHED ===")


if __name__ == "__main__":
    main()
