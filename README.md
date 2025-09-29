# ML Threshold Selection (Modular GUI)

A modular, production-ready toolkit for particle artifact filtering and threshold selection in XRCT datasets. The project provides a Tkinter GUI, robust ML pipelines, dual‑threshold prediction analysis, and scientifically sound fabric analysis (T and P'). All logic is fully modular under `src/ml_threshold_selection/`.

## Overview
- GUI for end‑to‑end workflow: data loading → feature analysis → model training → prediction & dual thresholds → fabric analysis → export
- Reproducible directory layout for training/test data and per‑sample configuration
- All outputs saved under `outputs/`

## Quick Start (GUI)
1. Place training files under `examples/data/train/` (CSV/XLSX).
2. Place test files under `examples/data/test/`.
3. Edit `examples/config/thresholds_voxels.csv` with columns: `SampleID, ExpertThreshold_mm3, VoxelSize_mm`.
4. Launch the app:
```bash
python main.py
```
5. In the GUI, proceed in order:
   - `📥 Load Thresholds Config` (auto‑fill thresholds & voxel sizes from CSV)
   - `1. Load Training Data` (pick files from `examples/data/train/`)
   - `4. Feature Analysis` (optional, interpretability)
   - `5. Train Model`
   - `6. Load Test Data` (pick a file from `examples/data/test/` → input voxel size if prompted)
   - `7. Predict Analysis`
   - `📦 Fabric Boxplots` (optional)
   - `📤 Export Results`
   - `❓ User Guide` to read methods and step‑by‑step instructions

Outputs (figures, two XLSX, one TXT report, and `last_time_model.pkl`) are written to `outputs/`.

## Quick Start (Script)
```bash
python scripts/repro_run.py \
  --train_dir examples/data/train \
  --config examples/config/thresholds_voxels.csv \
  --test_file examples/data/test/your_test.xlsx \
  --voxel_size_mm 0.03
```
Notes:
- `--voxel_size_mm` can be omitted if `SampleID` in the CSV matches the test file base name.
- Results go to `outputs/`.

## Data Requirements
- Required per‑particle columns:
  - `Volume3d (mm^3) `, `EigenVal1`, `EigenVal2`, `EigenVal3`,
    `EigenVec1X/Y/Z`, `EigenVec2X/Y/Z`, `EigenVec3X/Y/Z`
- Recommended: `SampleID` (per‑file default also works)

## Methods (Concise)
- Resolution‑aware 7D log‑ellipsoid tensor features standardized with `StandardScaler`
- Dual thresholds (voxel domain):
  - Loose = inflection point of artifact‑rate curve
  - Strict = threshold removing all particles with P > 0.05 (enforced ≥ Loose)
- Fabric analysis (T, P'):
  - For each volume threshold: bootstrap retained particles, compute log‑Euclidean mean fabric tensor, get eigenvalues → Jelínek (1981) T and P'

## Documentation
- User guide: `docs/USER_GUIDE_MODEL_AND_FEATURES_EN.md`
- Reproducibility guide: `docs/REPRODUCIBILITY.md`

## Project Structure
```
ML_Threshold_Selection/
├─ main.py                                   # minimal entrypoint (GUI bootstrap)
├─ src/
│  ├─ ml_threshold_selection/
│  │  ├─ app_controller.py                   # GUI controller (delegates everything)
│  │  ├─ ui_layout.py                        # Tkinter layout & bindings
│  │  ├─ ui_visualization.py                 # training/prediction charts
│  │  ├─ data_io.py                          # data loading & dialogs
│  │  ├─ training_pipeline.py                # train + metrics
│  │  ├─ prediction_analysis.py              # dual thresholds from predictions
│  │  ├─ export_results.py                   # XLSX & TXT export
│  │  ├─ labeling.py                         # labels from expert thresholds
│  │  ├─ feature_utils.py                    # simple feature helpers
│  │  ├─ fabric_thresholds.py                # volume threshold grid helpers
│  │  ├─ fabric_bootstrap.py                 # log‑E mean + bootstrap
│  │  ├─ fabric_logging.py                   # UILogger adapter
│  │  └─ fabric_pipeline.py                  # run_fabric_boxplots orchestrator
│  ├─ analysis/ellipsoid_feature_analyzer.py # feature analysis
│  └─ features/
│     ├─ ellipsoid_feature_engineering.py
│     ├─ ellipsoid_feature_engineering_legacy.py
│     └─ res_aware_feature_engineering.py
├─ examples/
│  ├─ data/
│  │  ├─ train/                              # put training CSV/XLSX here
│  │  └─ test/                               # put test CSV/XLSX here
│  └─ config/thresholds_voxels.csv           # SampleID → threshold(mm³), voxel(mm)
├─ scripts/repro_run.py                      # one‑command training+prediction
├─ docs/
│  ├─ USER_GUIDE_MODEL_AND_FEATURES_EN.md
│  └─ REPRODUCIBILITY.md
├─ outputs/                                  # figures, reports, models
├─ requirements.txt
├─ pyproject.toml / setup.py
└─ README.md
```

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# optional
pip install lightgbm
```

## License
MIT
