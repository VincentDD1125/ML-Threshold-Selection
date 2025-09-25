# ML Threshold Selection (XRCT Particles)

Resolution-aware, machine-learning-based threshold selection for XRCT particle analysis.

## 🎯 Key Features

- **Resolution-aware training**: Train and infer in voxel domain; UI displays voxel and mm³ side-by-side
- **Dual-threshold analysis**: Loose (inflection) and Strict (remove all P>0.05) thresholds
- **Expert-supervised**: Learn from expert absolute volume thresholds (mm³), internally converted to voxels
- **Joshua geometry**: 6 log-ellipsoid tensor features + voxel count, standardized
- **Interactive GUI**: Clean, two-row control layout; training and prediction visualizations
- **Model management**: Save/load models

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run GUI
```bash
python main.py
```

### Run in IDE
1. Open `main.py`
2. Run configuration: Python file

## 📊 Usage Steps (GUI)

### 1. Load training data
- Click "1. Load Training Data"; select multiple XLSX/CSV files
- The system extracts sample names and summary info

### 2. Input expert thresholds (absolute volume, mm³)
- Click "2. Input Expert Thresholds"
- Enter one per line: `SampleID:Threshold` (e.g., `sampleA:1.0e-06`)

### 3. Input voxel sizes (mm)
- Click "3. Input Voxel Sizes"; enter voxel edge length in mm per sample

### 4. Train model
- Click "4. Train Model"; AUC and metrics will be displayed

### 5. Predict analysis & visualization
- Click "5. Load Test Data" → select file
- Click "6. Predict Analysis" to log dual thresholds
- Click "Prediction Visualization" to view dual-threshold plots

## 📁 Data Format

### Required columns
- `Volume3d (mm^3) ` - Particle volume (note the trailing space)
- `EigenVal1`, `EigenVal2`, `EigenVal3` - Ellipsoid eigenvalues
- `EigenVec1X`, `EigenVec1Y`, `EigenVec1Z` - First principal axis direction
- `EigenVec2X`, `EigenVec2Y`, `EigenVec2Z` - Second principal axis direction
- `EigenVec3X`, `EigenVec3Y`, `EigenVec3Z` - Third principal axis direction

### Optional columns
- `BaryCenterX/Y/Z (mm) ` - Barycenter coordinates
- `Anisotropy`, `Elongation`, `Flatness` - Shape parameters
- `ExtentMin/Max1/2/3 (mm) ` - Bounding box
- `BinMom2x/y/z (mm^2) ` - Second moments
- `VoxelFaceArea`, `BorderVoxelCount` - Voxel info
- `GreyMass (mm^3) ` - Gray mass

## 🔧 Architecture

### ML models
- **LightGBM**: Preferred for best performance
- **RandomForest**: Stable fallback
- **Auto-degrade**: Fallback to RandomForest when LightGBM unavailable

### Feature engineering
- **Voxel domain**: continuous voxel count = volume(mm³) / voxel_mm³
- **Joshua 6**: L11, L22, L33, √2L12, √2L13, √2L23 (log-ellipsoid tensor)
- **Scaling**: StandardScaler (fit on training, reuse on prediction)

### Dual-threshold algorithm (voxel domain)
- **Loose (Inflection)**: inflection point from artifact-rate curve vs threshold
- **Strict (P>0.05)**: minimal voxel threshold that removes all particles with P>0.05
- Always enforce Strict ≥ Loose; both thresholds displayed in voxels and mm³

## 📈 Training Metrics

### Global metrics
- **AUC**: discrimination power
- **Accuracy**: proportion of correct predictions
- **Precision**: quality of artifact predictions
- **Recall**: fraction of actual artifacts identified
- **F1 score**: harmonic mean of precision and recall

### Per-sample
- **Accuracy**: per-sample accuracy
- **Threshold error**: percent error vs expert threshold

## 🎯 Project Structure

```
ML_Threshold_Selection/
├── main.py                       # GUI entry
├── src/
│   ├── ml_threshold_selection/   # Core package
│   ├── features/                 # Feature engineering
│   │   ├── joshua_feature_engineering.py
│   │   ├── joshua_feature_engineering_fixed.py
│   │   └── res_aware_feature_engineering.py
│   └── analysis/                 # Analyzers
│       └── joshua_feature_analyzer.py
├── scripts/                      # CLI scripts & experiments
├── assets/                       # Exported figures
├── docs/                         # Documentation (this file)
└── archive/                      # Deprecated/legacy code
```

## 💡 Tips

### Data preparation
1. **Consistent schema**: Ensure column names match
2. **Sample naming**: Use meaningful sample IDs
3. **Data quality**: Validate completeness

### Thresholding
1. **Scientific notation**: e.g., `1.0e-06`
2. **Precision**: Keep sufficient significant digits
3. **Validation**: Verify thresholds across trials

### Model training
1. **Data volume**: ≥100 particles per sample recommended
2. **Num samples**: ≥5–10 samples recommended
3. **Threshold quality**: Ensure expert thresholds are reliable

## 🔍 Troubleshooting

### Common issues

**1. "Full modules unavailable"**
- Falls back to simplified built-in modules
- Functionally works; minor performance differences

**2. "LightGBM unavailable"**
- Falls back to RandomForest
- Stable performance

**3. "File load failed"**
- Verify file format
- Ensure file not locked by other programs
- Check path and permissions

**4. "Missing required columns"**
- Check exact column names (note trailing spaces)
- Ensure all required columns are present

## 🎉 Summary

This system enables you to:
- **Process efficiently**: Batch multiple samples
- **Learn intelligently**: Learn threshold patterns automatically
- **Predict accurately**: Determine thresholds for new samples
- **Operate intuitively**: Easy-to-use GUI

**Get started: run `python main.py`!** 🚀

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)