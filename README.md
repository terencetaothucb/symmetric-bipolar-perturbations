# A rapid and scalable observable for battery capacity loss detection enabled by symmetric bipolar pulse perturbations

This repository contains the data-processing, modeling, experiment, and benchmarking workflow associated with the following manuscript.

**Associated manuscript**

*A rapid and scalable observable for battery capacity loss detection enabled by symmetric bipolar pulse perturbations*

- GitHub repository: https://github.com/terencetaothucb/symmetric-bipolar-perturbations
- Public dataset (Zenodo DOI): https://doi.org/10.5281/zenodo.23634967

After downloading the three archives and the additional 68 Ah LFP cell No. 6 workbook from Zenodo, arrange the working data using these relative paths:

```text
Data/
|-- Chinese_to_English_Dictionary/
|-- Raw_Data/
|-- Data_Process_Output/
`-- Model_Output/
```

- extract `Raw_Data.zip`, `Data_Process_Output.zip`, and `Model_Output.zip` into `Data/`; each extracted top-level folder must appear exactly once at the paths shown above
- place `LFP_C_68_B_6_SOC_5-80_Part_1-1_ID_02HCB68111211Y88F0000101.xlsx` in `Data/Raw_Data/68Ah LFP/`
- confirm that the unified modeling-input dataset is present at `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/`; this complete dataset supersedes the former `Step4_SelectedFeatures/` folder
- keep the translation dictionary at `Data/Chinese_to_English_Dictionary/`; if an archive places it inside `Data/Raw_Data/`, move the dictionary folder to this location before running the pipeline
- the working folders start from `Data/`, `Data_Process_Codes/`, and `Model_Codes/`
- `Raw_Data.zip` is only a transfer or upload artifact and is not part of the folder structure described in this README
- runtime folders such as `__pycache__` are not part of the documented structure
- `Data/` is distributed through Zenodo and is not part of the GitHub upload
- `Data/Model_Output/` contains one complete six-case reference suite rather than every manuscript run

## 1. Project Overview

The project is organized into two main technical parts:

1. Data processing, in which raw Excel workbooks are transformed step by step into modeling-ready data files.
2. Model building, in which the processed data are used for training, testing, and benchmarking experiments.

The current data processing route is:

`Data/Raw_Data -> Data/Data_Process_Output/Step1_Workstep -> Data/Data_Process_Output/Step2_CellLevel -> Data/Data_Process_Output/Step3_ByPulse -> Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff -> Data/Model_Output`

This repository implements the processing and modeling route shown above.

## 2. Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- Data_Process_Codes/
|   |-- run_pipeline.py
|   |-- step1_extract_workstep.py
|   |-- step2_extract_pulse_features.py
|   |-- step3_collect_by_pulse.py
|   |-- step4_extract_selected_features.py
|   |-- config.py
|   |-- utils.py
|   `-- optional/
|       `-- soh_gt1_summary.py
`-- Model_Codes/
    |-- Model_backbone.py
    `-- Run_Model.py
```

Only the two code folders and the two root files shown above are included in the GitHub upload. The `Data/` working directory is created locally after downloading the Zenodo files and is described in Section 4.

### 2.1 Released dataset and manuscript cohort

The associated [Zenodo record](https://zenodo.org/records/23634967) provides four downloadable files:

- `Raw_Data.zip`
- `Data_Process_Output.zip`
- `Model_Output.zip`
- `LFP_C_68_B_6_SOC_5-80_Part_1-1_ID_02HCB68111211Y88F0000101.xlsx`

The table reproduces the group statistics in manuscript Table 1 and adds the number of candidate cells in the released data. The real-capacity statistics, quantities, and data-sample counts describe the retained manuscript cohort. Some cells have multiple raw-workbook parts, so workbooks and cells must not be counted interchangeably.

| Material | Real capacity (Ah) | Form factor | Usage | Candidate cells in release | Quantity in Table 1 | Data samples in Table 1 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| NMC 811 | 15 (11.29; 12.39; 0.21) | Pouch | BEV1 | 83 | 83 | 114,000 |
| NMC 622 | 21 (15.68; 20.97; 1.30) | Pouch | BEV2 | 52 | 33 | 58,100 |
| LMO | 10 (5.68; 9.48; 1.10) | Pouch | HEV1 | 62 | 62 | 94,600 |
| LMO | 24 (15.35; 19.83; 1.02) | Pouch | HEV2 | 64 | 56 | 81,000 |
| LMO | 25 (10.66; 23.88; 3.13) | Pouch | HEV3 | 96 | 96 | 130,010 |
| LMO | 26 (20.47; 25.98; 1.68) | Pouch | HEV4 | 98 | 89 | 147,100 |
| LFP | 20 (13.33; 16.51; 0.79) | Prismatic | LSV1 | 102 | 102 | 138,500 |
| LFP | 35 (26.03; 33.68; 1.70) | Prismatic | HBESS1 | 56 | 56 | 85,300 |
| LFP | 68 (44.81; 59.36; 4.06) | Prismatic | GBESS1 | 96 | 96 | 136,500 |
| **Total** | | | | **709** | **673** | **985,110** |

Table 1 also reports a shared test design of `n_W = 10`, `n_SOC = 18`, and `n_P = 10`, with **98,511** total test points. Its `Quantity` column counts physical cells; `Data samples` is the separate count reported in Table 1.

The release covers **709 candidate cells**: `Raw_Data.zip` contains 708, and the separately supplied 68 Ah LFP cell No. 6 workbook completes the raw inventory. The released processed outputs were generated from the 709-cell candidate set. To obtain the **673-cell manuscript cohort**, exclude 35 cells with `SOH > 1` in the released fixed-SOC data and one 21 Ah NMC cell whose historical random-SOC test has `SOH > 1`. Filtering the released fixed-SOC data alone would retain 674 cells; the documented additional exclusion is required for 673. See Sections 4.1 and 7.5 for the cell identity and the exact filtering command.

The released raw data contain 799 workbooks, including multipart files. `Data_Process_Output/` contains 1,599 files: 1,598 Excel workbooks and one JSON provenance summary. `Model_Output/` contains 12,103 files: 12,090 CSV files and 13 JSON files. The translation dictionary contains one CSV file and one JSON file. The complete arranged `Data/` directory contains 14,503 files.

The public GitHub upload contains 12 files: `README.md`, `requirements.txt`, eight Python files under `Data_Process_Codes/`, and two Python files under `Model_Codes/`.

## 3. Folder-Level Description

### 3.1 `Data/`

This folder stores the working data used by the repository.

- `Chinese_to_English_Dictionary/`: CSV and JSON reference dictionaries mapping original Chinese headers and terms to English. This folder is stored alongside `Raw_Data/` and is not a battery-type input folder.
- `Raw_Data/`: raw Excel workbooks organized by battery type
- `Data_Process_Output/`: outputs generated by Step1-Step4
- `Model_Output/`: saved modeling outputs; the published release provides one complete six-case, 100-seed reference suite. Other experiment combinations can be reproduced by running the code.

### 3.2 `Data_Process_Codes/`

This folder contains the processing scripts that transform raw workbooks into modeling-ready features.

- `step1_extract_workstep.py`: extracts the workstep sheet from each raw workbook
- `step2_extract_pulse_features.py`: extracts cell-level pulse records and `U1-U41`
- `step3_collect_by_pulse.py`: reorganizes Step2 outputs by pulse width and by SOC sheet
- `step4_extract_selected_features.py`: computes `fai_irrev`, explicitly stored scalar-average `R0`/`Reff`, and `E_loss` from the Step3 voltage checkpoints
- `run_pipeline.py`: runs Step1-Step4 in sequence
- `config.py`: shared constants and matching rules
- `utils.py`: shared helper functions for file handling and parsing
- `optional/`: contains `soh_gt1_summary.py`, which reports cells above a selected SOH threshold and can rebuild a filtered copy retaining `SOH <= threshold` while also applying documented additional cohort exclusions

### 3.3 `Model_Codes/`

This folder contains the model-building code.

- `Model_backbone.py`: backbone for dataset loading, feature selection, training, evaluation, and result export
- `Run_Model.py`: experiment runner built on top of `Model_backbone.py`

For users who want to modify the benchmark design and run additional experiments, `Model_Codes/` is the main place to edit.

In `Model_backbone.py`, the feature-definition section controls which Step4 features are exposed to the model code:

```python
FEATURE_GROUPS = {
    "fai_irrev": [
        "fai_irrev_0.5C", "fai_irrev_1C", "fai_irrev_1.5C", "fai_irrev_2C", "fai_irrev_2.5C",
    ],
    "Reff_avg": ["Reff_avg_0.5C", "Reff_avg_1C", "Reff_avg_1.5C", "Reff_avg_2C", "Reff_avg_2.5C"],
    "R0_avg": ["R0_avg_0.5C", "R0_avg_1C", "R0_avg_1.5C", "R0_avg_2C", "R0_avg_2.5C"],
    "raw_U": ["U1", "U5", "U9", "U13", "U17", "U21", "U25", "U29", "U33", "U37", "U41"],
}

FEATURE_COMBOS = {
    "fai_irrev_Only": ["fai_irrev"],
    "Reff_avg_Only": ["Reff_avg"],
    "R0_avg_Only": ["R0_avg"],
    "raw_U_Only": ["raw_U"],
    "fai_irrev_Reff_avg": ["fai_irrev", "Reff_avg"],
    "fai_irrev_R0_avg": ["fai_irrev", "R0_avg"],
}
```

Here:

- `FEATURE_GROUPS` defines the base feature families
- `FEATURE_COMBOS` defines reusable multi-feature settings
- `build_feature_list(...)` resolves the final feature list from `combo`, `include_groups`, `exclude_groups`, `add_features`, `drop_features`, and `keep_regex`
- `paper_feature_columns(...)` and `paper_feature_spec(...)` resolve the exact single-C-rate raw-U, resistance, and feature-fusion inputs used in the manuscript comparisons

For each C-rate and polarity, Step4 uses:

```text
R0   = abs(U_step - U_pre) / abs(I)
Reff = abs(U_end  - U_pre) / abs(I)
abs(I) = C-rate * Qn
```

For each C-rate, the two pulse-polarity resistance estimates are averaged arithmetically. The resulting scalar columns are stored directly as `R0_avg_*` and `Reff_avg_*` in every relevant `SOC ALL` and numeric `SOCxx` sheet. The polarity-specific R0 and Reff values are calculation intermediates only and are not stored in the published Step4 workbooks. The model loader reads the stored average columns directly and does not derive them at runtime. The raw-voltage comparison uses three baseline checkpoints at each C-rate: `U1/U5/U9`, `U9/U13/U17`, `U17/U21/U25`, `U25/U29/U33`, or `U33/U37/U41` for 0.5C, 1C, 1.5C, 2C, or 2.5C, respectively.

This allows three common experiment modes:

1. single feature, single C-rate
   Use one feature group or combo and set `keep_regex` to retain only one C-rate, for example `_1C$`.
2. single feature, multiple C-rates
   Use one feature group or combo and keep `keep_regex=None`, so all available C-rates for that feature family are used together.
3. multiple features
   Use a multi-group entry in `FEATURE_COMBOS`, or modify `include_groups` / `exclude_groups` / `add_features` to assemble a custom feature set.

In the `CONFIG` block of `Model_backbone.py`, the main experiment settings are:

```python
CONFIG = dict(
    data_root="Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff",
    material="20Ah LFP",
    width_ms=5000,
    soc=50,
    label_col="SOH",
    feature_spec=dict(
        name="fai_irrev_Only",
        combo="fai_irrev_Only",
        include_groups=None,
        exclude_groups=None,
        add_features=None,
        drop_features=None,
        keep_regex=None,
    ),
    test_size=0.2,
    seeds=list(range(100)),
    models=["linear", "ridge", "lasso", "en", "svm", "rf", "gpr", "mlp", "transformer", "informer"],
    perm_n_jobs=1,
    rf_n_jobs=1,
    out_dir="Data/Model_Output",
)
```

- `material`: battery type folder to read
- `width_ms`: pulse width
- `soc`: SOC sheet to use
- `label_col`: target variable, currently `SOH`
- `feature_spec`: feature-selection specification
- `models`: list of models to benchmark
- `seeds`: repeated random splits
- `test_size`: train-test split ratio
- `torch_*`: training parameters for torch-based models
- `perm_n_jobs` and `rf_n_jobs`: process counts for permutation importance and random forest; the default value of 1 avoids excessive memory use on Windows

In the current codebase, the C-rate dimension is controlled through the feature names themselves, such as `fai_irrev_0.5C` or `Reff_avg_2C`. In practice, this is the parameter axis that corresponds to pulse amplitude selection in the model inputs. It is controlled through `keep_regex` in `Model_backbone.py` and through `crate_list` in `Run_Model.py`.

Example:

```python
feature_spec=dict(
    name="fai_irrev_Only__1C",
    combo="fai_irrev_Only",
    keep_regex=r"_1C$",
)
```

This keeps only the `1C` columns from the selected feature family. If `keep_regex=None`, all available C-rates in that feature family are used together.

`Run_Model.py` is a matrix-style runner for batch experiments. It currently loops over:

- all available materials under `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/`
- a selected `soc_list`
- a selected `crate_list`
- the manuscript comparison cases listed in `experiment_cases`

To expand or modify the experiment matrix, edit the following variables in `Run_Model.py`:

```python
width_ms = 5000
soc_list = [50, 40, 70]
experiment_cases = [
    "fai_irrev", "R0_avg", "Reff_avg", "raw_U",
    "fai_irrev_R0_avg", "fai_irrev_Reff_avg",
]
crate_list = ["0.5C", "1C", "1.5C", "2C", "2.5C"]

MODELS = ["linear", "ridge", "lasso", "en", "svm", "rf", "gpr", "mlp", "transformer", "informer"]
SEEDS = list(range(100))
TEST_SIZE = 0.2
```

- `width_ms`
- `soc_list`
- `experiment_cases`
- `crate_list`
- `MODELS`
- `SEEDS`
- `TEST_SIZE`

In practice:

- change `width_ms` to test another pulse width
- change `soc` or `soc_list` to test another SOC condition
- change `crate_list` or `keep_regex` to switch C-rate selection
- change `experiment_cases` or `feature_spec` to compare the six manuscript feature configurations
- change `models` to benchmark a different model set

### 3.4 `Data/Data_Process_Output/`

This folder stores the intermediate and final outputs of the processing pipeline.

- `Step1_Workstep/`: extracted workstep-sheet files; this is the output of `Step1` and the input of `Step2`
- `Step2_CellLevel/`: per-cell pulse extraction results containing metadata and `U1`-`U41`
- `Step3_ByPulse/`: pulse-width-wise aggregated workbooks with `SOC ALL` and `SOCxx` sheets
- `Step4_SelectedFeatures_with_R0_Reff/`: unified final modeling-input files containing all columns from the former Step4 dataset together with the released R0 additions; this is the only Step4 directory required by the current workflow

This README intentionally explains these folders at folder level only. The internal file counts are large, so they are not listed file by file here.

### 3.5 `Data/Model_Output/`

This folder stores model results generated by the code in `Model_Codes/`.

The published dataset contains one complete six-case example suite:

- `20261025-141538__20Ah_LFP__W5000__SOC40__0.5C__six_feature_cases__seeds0-99/`

This suite fixes the dataset to `20Ah LFP`, `W = 5000 ms`, `SOC = 40`, and `0.5C`, and stores the six manuscript feature comparisons as six separate run directories: `fai_irrev`, `R0_avg`, `Reff_avg`, `raw_U`, `fai_irrev_R0_avg`, and `fai_irrev_Reff_avg`. Each run contains 100 seeds (`0`-`99`) for the same ten-model set, i.e., 1,000 model-seed results per feature comparison. `sample_manifest.json` records the exact input columns and the corresponding output directory for every comparison.

For each modeling run, the code first selects one Step4 input workbook by:

- `material`
- `width_ms`
- `soc`
- `feature_spec`

More specifically, the model code reads:

- one workbook under `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/<material>/` matching `*_W_<width_ms>.xlsx`
- one SOC sheet inside that workbook, such as `SOC40` or `SOC50`
- one resolved feature list built from `feature_spec`
- one target column, normally `SOH`

The output folder name is generated as:

```text
<timestamp>__<material>__W<width_ms>__SOC<soc>__<feature_tag>
```

where:

- `<timestamp>` records the run time, for example `20260220-225843`
- `<material>` comes from `CONFIG["material"]`
- `W<width_ms>` comes from the selected pulse width
- `SOC<soc>` comes from the selected SOC sheet
- `<feature_tag>` is built from `feature_spec`

The `feature_tag` is not just `fai_irrev_Only`. It is assembled from the resolved feature settings in `Model_backbone.py`, including:

- `name`
- `combo`
- `include_groups`
- `exclude_groups`
- `keep_regex`
- `add_features`
- `drop_features`

Typical naming examples are:

- single feature, single C-rate:
  `20260220-225843__20Ah_LFP__W5000__SOC40__fai_irrev_Only__0.5C__combo-fai_irrev_Only__keep`
- single feature, multiple C-rates:
  `20260220-225843__20Ah_LFP__W5000__SOC40__fai_irrev_Only__combo-fai_irrev_Only`
- multiple features:
  `20260220-225843__20Ah_LFP__W5000__SOC40__fai_irrev_Reff_avg__1C__add2`

In other words, the run-folder name tells you which material, pulse width, SOC condition, and feature-selection rule were used to create that result. The model names themselves are stored inside the run folder rather than in the top-level run-folder name.

Inside a typical model-output folder:

- `summary/` stores aggregated metrics, configuration snapshots, merged prediction tables, and summarized feature-importance results
- `raw/predictions/` stores per-model and per-seed prediction outputs
- `raw/featimp/` stores per-model and per-seed feature-importance outputs

## 4. Data Availability and Placement

The code resolves relative paths from the repository root. Download the four files from the associated Zenodo record, create `Data/` if necessary, and extract the three ZIP archives into `Data/`. The final layout must contain `Data/Raw_Data/`, `Data/Data_Process_Output/`, and `Data/Model_Output/` exactly once. If an extraction tool creates a duplicated path such as `Data/Data_Process_Output/Data_Process_Output/`, move the inner folder up one level.

The repository expects the working raw data to be placed under:

- `Data/Raw_Data/`

The translation dictionaries are stored separately at `Data/Chinese_to_English_Dictionary/`. Keep this folder outside `Data/Raw_Data/`, whose subfolders are interpreted as battery types by the processing pipeline.

The processed outputs are generated under:

- `Data/Data_Process_Output/Step1_Workstep/`
- `Data/Data_Process_Output/Step2_CellLevel/`
- `Data/Data_Process_Output/Step3_ByPulse/`
- `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/`

`Step4_SelectedFeatures_with_R0_Reff/` is the complete Step4 dataset distributed in the Zenodo release. It contains all modeling inputs retained from the former `Step4_SelectedFeatures/` release plus the R0 additions. The former directory is therefore not required and should not be placed beside the unified directory in a fresh installation.

Place the separately downloaded 68 Ah LFP cell No. 6 raw workbook, ID `02HCB68111211Y88F0000101`, at:

`Data/Raw_Data/68Ah LFP/LFP_C_68_B_6_SOC_5-80_Part_1-1_ID_02HCB68111211Y88F0000101.xlsx`

Before running the code, verify these representative paths:

```text
Data/Raw_Data/10Ah LMO/
Data/Raw_Data/68Ah LFP/LFP_C_68_B_6_SOC_5-80_Part_1-1_ID_02HCB68111211Y88F0000101.xlsx
Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/20Ah LFP/LFP_20Ah_W_5000.xlsx
Data/Model_Output/20261025-141538__20Ah_LFP__W5000__SOC40__0.5C__six_feature_cases__seeds0-99/
```

The released processing dataset contains 709 batteries. The fixed-SOC records identify 35 batteries with `SOH > 1`. One additional 21 Ah NMC cell is excluded based on the historical random-SOC test described below. The optional cohort utility therefore produces the 673-battery manuscript cohort after 36 exclusions.

### 4.1 Additional exclusion identified by the random-SOC test

The 21 Ah NMC cell No. 34 (ID `02LCC02100101A87Y0013547`) has a maximum `SOH` of approximately `0.998933` in the released fixed-SOC records. However, its historical random-SOC test at `SOC = 41.6497%` recorded `Q = 21.0045 Ah`, corresponding to `SOH = 1.000214`, which exceeds the cohort threshold of 1. This cell was therefore excluded from the manuscript cohort together with the 35 cells whose `SOH > 1` is directly observable in the released fixed-SOC data.

The released raw-data subset contains the fixed-SOC experiments used in the downstream figures and model experiments; the historical random-SOC file is not included because random-SOC records are not used as downstream model inputs. The additional exclusion is recorded in `Data_Process_Codes/config.py`. The main Step1-Step4 pipeline preserves the complete 709-battery released dataset, including this cell. `Data_Process_Codes/optional/soh_gt1_summary.py` applies the `SOH > 1` rule and this additional exclusion when users request the 673-battery manuscript cohort. The additional rule is restricted to battery type `21Ah NMC`, cell No. 34, ID `02LCC02100101A87Y0013547`; it does not affect cells numbered 34 in other battery types.

## 5. Code Availability

All code needed for the current processing and modeling workflow is included in this repository under:

- [`Data_Process_Codes/`](./Data_Process_Codes/)
- [`Model_Codes/`](./Model_Codes/)

The processing scripts were simplified into one script per step so that each filename directly indicates its role in the workflow.

## 6. Computational Environment

The HPC and local workflows use one documented Python environment for data processing and model benchmarking. Its Python and core package versions match Supplementary Note 3:

```text
Python 3.10.16
numpy==1.25.1
pandas==2.0.3
scikit-learn==1.6.1
torch==2.1.2
```

The code also uses `openpyxl` for Excel input/output and `matplotlib` for the optional SOH-exclusion report. Supplementary Note 3 does not specify their versions, so they are included in `requirements.txt` without version pins.

On the original HPC system, the CUDA-enabled PyTorch build was loaded before activating the project virtual environment:

```bash
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2025-22-468/bamm_venv_gpu/bin/activate
```

For local use, select Python 3.10.16 and install the packages listed in `requirements.txt`. The PyTorch package version is shared across HPC and local workflows; CUDA support depends on the platform-specific build.

## 7. Reproducibility Workflow

All relative input and output paths in the processing commands and model configurations are resolved from the repository root (the directory containing this README), independently of the current working directory. Keep the documented folder structure when cloning or moving the repository. The default model paths do not require a machine-specific drive or directory.

### 7.1 Prepare the raw-data folder

Download the four files from the [Zenodo record](https://zenodo.org/records/23634967). Extract the three ZIP archives into `Data/`, place the separately downloaded 68 Ah LFP cell No. 6 workbook under `Data/Raw_Data/68Ah LFP/`, and confirm that the repository contains:

- `Data/Chinese_to_English_Dictionary/`
- `Data/Raw_Data/`
- `Data/Data_Process_Output/`
- `Data/Model_Output/`

To rerun Step1-Step4, the raw Excel workbooks must be under:

- `Data/Raw_Data/`

Keep `Chinese_to_English_Dictionary/` directly under `Data/`, alongside `Raw_Data/`. If it was extracted inside `Data/Raw_Data/`, move only that dictionary folder to `Data/Chinese_to_English_Dictionary/` before running the workflow. The `--raw-root` argument below continues to point to `Data/Raw_Data/`. Do not point `--raw-root` at `Data/`, because every subfolder under the selected raw root is interpreted as a battery type.

### 7.2 Run the full processing pipeline

From the repository root:

```powershell
python ".\Data_Process_Codes\run_pipeline.py" `
  --raw-root ".\Data\Raw_Data" `
  --step1-root ".\Data\Data_Process_Output\Step1_Workstep" `
  --step2-root ".\Data\Data_Process_Output\Step2_CellLevel" `
  --step3-root ".\Data\Data_Process_Output\Step3_ByPulse" `
  --step4-root ".\Data\Data_Process_Output\Step4_SelectedFeatures_with_R0_Reff"
```

### 7.3 Run from Step2 onward

If `Step1_Workstep/` has already been prepared, corrected, or manually curated, the remaining workflow can be run from Step2:

```powershell
python ".\Data_Process_Codes\step2_extract_pulse_features.py" `
  --input-root ".\Data\Data_Process_Output\Step1_Workstep" `
  --output-root ".\Data\Data_Process_Output\Step2_CellLevel"

python ".\Data_Process_Codes\step3_collect_by_pulse.py" `
  --input-root ".\Data\Data_Process_Output\Step2_CellLevel" `
  --output-root ".\Data\Data_Process_Output\Step3_ByPulse" `
  --overwrite

python ".\Data_Process_Codes\step4_extract_selected_features.py" `
  --input-root ".\Data\Data_Process_Output\Step3_ByPulse" `
  --output-root ".\Data\Data_Process_Output\Step4_SelectedFeatures_with_R0_Reff" `
  --overwrite
```

### 7.4 Run selected materials only

Each Step1-Step4 script can be limited to specific materials, for example:

```powershell
python ".\Data_Process_Codes\step2_extract_pulse_features.py" `
  --input-root ".\Data\Data_Process_Output\Step1_Workstep" `
  --output-root ".\Data\Data_Process_Output\Step2_CellLevel" `
  --materials "25Ah LMO"
```

### 7.5 Report SOH exclusions and rebuild the 673-cell manuscript cohort

The main Step1-Step4 pipeline preserves all 709 released batteries. To generate only the exclusion report without creating a filtered dataset, run:

```powershell
python ".\Data_Process_Codes\optional\soh_gt1_summary.py" `
  --input-root ".\Data\Data_Process_Output\Step4_SelectedFeatures_with_R0_Reff" `
  --report-root ".\Data\SOH_Exclusion_Report"
```

The report contains `summary_quantity_soh_gt1.xlsx/.csv`, `excluded_cells.xlsx/.csv`, and `soh_hist_grid_all_types.png`. With the default `--threshold 1.0`, it reports 35 cells whose maximum fixed-SOC value satisfies `SOH > 1` and one configured additional exclusion, giving 36 exclusions in total.

To create a separate 673-battery copy that retains `SOH <= 1` and applies the additional exclusion, run:

```powershell
python ".\Data_Process_Codes\optional\soh_gt1_summary.py" `
  --input-root ".\Data\Data_Process_Output\Step4_SelectedFeatures_with_R0_Reff" `
  --report-root ".\Data\SOH_Exclusion_Report" `
  --rebuild-output-root ".\Data\Data_Process_Output\Step4_SelectedFeatures_with_R0_Reff_SOH_LE_1"
```

The summary distinguishes `QuantitySOH_GT_Threshold`, `QuantityConfiguredExtraExclusions`, and `QuantityExcludedTotal`. The filtered copy preserves the original material folders, workbooks, and sheets while removing the excluded cell rows. The input directory is not modified.

To run the manuscript-cohort models from the filtered copy, set the model `data_root` to `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff_SOH_LE_1`.

### 7.6 Run the modeling code

For a single benchmark configuration, edit the experiment settings in the `CONFIG` block of `Model_Codes/Model_backbone.py` as needed and run:

The default paths are already set to:

- `data_root` to `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff`
- `out_dir` to `Data/Model_Output`

Both paths are relative to the repository root. The batch runner uses the same relative defaults in `DEFAULT_DATA_ROOT` and `DEFAULT_OUT_DIR`.

```powershell
python ".\Model_Codes\Model_backbone.py"
```

For batch experiments across multiple materials, SOC values, or C-rates, edit the matrix settings in `Model_Codes/Run_Model.py` and run:

```powershell
python ".\Model_Codes\Run_Model.py"
```

The main parameters you may want to modify are:

- `material` in `Model_backbone.py`
- `width_ms` in `Model_backbone.py` or `Run_Model.py`
- `soc` in `Model_backbone.py`, or `soc_list` in `Run_Model.py`
- `feature_spec` in `Model_backbone.py`
- `experiment_cases` and `crate_list` in `Run_Model.py`
- `models`, `seeds`, and `test_size` in either file

The modeling code reads features from:

- `Data/Data_Process_Output/Step4_SelectedFeatures_with_R0_Reff/`

and writes outputs to:

- `Data/Model_Output/`

### 7.7 Reproducibility note for deep-learning models

The random seed controls the train-test split and the stochastic operations during model fitting. However, in the current implementation, the Transformer and Informer networks are instantiated before the PyTorch seed is reset for each run. Their initial weights may therefore depend on the preceding global random-number-generator state. Consequently, independently repeated runs, changes in model execution order, or differences in GPU, CUDA, cuDNN, and PyTorch versions may lead to numerical differences in the deep-learning results. The released outputs should be regarded as reference results obtained in the documented environment rather than bitwise-reproducible outputs across all hardware and software configurations.

## 8. Citation

If you use this repository, please cite the associated manuscript:

Tao, S., Xia, C., Liang, C., Lu, M., Zhang, X., Zhou, G., and Zou, C. *A rapid and scalable observable for battery capacity loss detection enabled by symmetric bipolar pulse perturbations*. Manuscript in preparation.
