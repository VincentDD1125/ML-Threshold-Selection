# Reproducibility Guide

This guide explains how to reproduce our training and prediction results with your own or our sample data.

## 1. What we provide (recommended layout)
- `examples/data/train/` (you put training XLSX/CSV here)
- `examples/data/test/` (you put test XLSX/CSV here)
- `examples/config/thresholds_voxels.csv` (per-sample expert threshold in mm³ and voxel size in mm)

We include ready-to-fill templates; swap in your actual data to reproduce.

## 2. Required data formats
- Training/Test files must contain at least columns:
  - `Volume3d (mm^3) `, `EigenVal1`, `EigenVal2`, `EigenVal3`,
    `EigenVec1X/Y/Z`, `EigenVec2X/Y/Z`, `EigenVec3X/Y/Z`
  - `SampleID` is recommended (one sample ID per file is also acceptable)
- `examples/config/thresholds_voxels.csv` must contain:
  - `SampleID`, `ExpertThreshold_mm3`, `VoxelSize_mm`

## 3. Quick start (programmatic)
```bash
python scripts/repro_run.py \
  --train_dir examples/data/train \
  --config examples/config/thresholds_voxels.csv \
  --test_file examples/data/test/test_sample.xlsx \
  --voxel_size_mm 0.03
```
This will:
- Load all training files from `--train_dir`
- Read expert thresholds and voxel sizes from `--config`
- Train a model (LightGBM if available, else RandomForest)
- Run prediction analysis and export results under `outputs/`

## 4. GUI alternative
1. `python main.py`
2. Load Training Data → Input Expert Thresholds → Input Voxel Sizes → Train Model
3. Load Test Data → Predict Analysis → (optional) Fabric Boxplots
4. Export Results

## 5. Notes
- Ensure voxel sizes (mm) are correct; thresholds will be accurately converted to voxels via `a^3`.
- All outputs are under `outputs/`.
- To share a minimal reproducible package, include your train/test files and the CSV config.
