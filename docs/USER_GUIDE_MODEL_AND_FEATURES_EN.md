# User Guide: Model Training, Prediction, Feature Analysis, and Fabric Analysis (GUI)

This guide explains how to use the graphical user interface to train a model, perform prediction and dual-threshold analysis, export results, run feature analysis, and generate fabric analysis (T and P') boxplots. The methods described are scientifically consistent and resolution-aware.

## 1) Data Requirements
- Required columns (per particle):
  - `Volume3d (mm^3) ` (note the space at the end if present in the source data)
  - Eigenvalues: `EigenVal1`, `EigenVal2`, `EigenVal3`
  - Eigenvectors: `EigenVec1X/Y/Z`, `EigenVec2X/Y/Z`, `EigenVec3X/Y/Z`
- Optional: `SampleID` (recommended for multi-sample training and per-sample voxel size input)
- File formats: CSV or XLSX

## 2) Launch
```bash
python main.py
```
Follow the button order in the GUI.

## 3) Training Workflow
1. Load training data
2. Input expert thresholds (absolute volume in mm³ per sample)
3. Input voxel sizes (edge length in mm per sample)
4. (Optional) Feature Analysis
5. Train Model (LightGBM preferred; RandomForest fallback if LightGBM is unavailable)

### What happens scientifically during training
- Labels are derived from expert thresholds in the voxel domain. For each particle with volume \(V_{mm^3}\) and voxel size \(a\) (mm), the voxel count is \(V_\text{vox} = V_{mm^3} / a^3\). A particle is labeled as artifact (1) if \(V_\text{vox} < T_\text{vox}\) for its sample; otherwise normal (0).
- Features are resolution-aware: we use a compact 7D log‑ellipsoid tensor representation (diagonals L11/L22/L33 and scaled off-diagonals \(\sqrt{2}\)·L12/L13/L23) plus a normalized voxel-count feature as needed. Features are standardized via `StandardScaler`.
- The classifier is trained on these features. After training, the GUI reports AUC, Accuracy, Precision, Recall, and F1.

### Visualizing training
Click "Training Visualization" to view ROC, top feature importances, probability distributions, and summary metrics.

## 4) Prediction and Dual Thresholds
1. Load test data
2. Enter the test sample voxel size (mm)
3. Click "Predict Analysis"

The app computes:
- Per-particle artifact probability via the trained model
- Two domain-specific thresholds (voxel domain), both also reported in mm³:
  - Loose: the inflection point of the artifact-rate curve with respect to voxel-count threshold (estimated from the smoothed second derivative)
  - Strict: the minimum voxel-count threshold that removes all particles with predicted probability \(P > 0.05\) (enforced to be \(\ge\) Loose if both exist)

Click "Prediction Visualization" to inspect prediction distributions and dual-threshold curves.

## 5) Exporting Results
Click "Export Results" to produce:
- Two XLSX files filtered by Loose/Strict thresholds
- One TXT report summarizing thresholds and retention
All outputs are saved under `outputs/`.

## 6) Session Persistence
Click "Load Last Time Model" to restore the entire session, including training data, expert thresholds, voxel sizes, fitted scaler, model, and prior analysis results.

## 7) Feature Analysis (Interpretability)
- Objective: quantify and interpret differences between particles labeled as removed (artifacts) vs kept (valid).
- Method: 7D log‑ellipsoid tensor feature set with resolution awareness. We compute effect sizes (Cohen's d), t‑tests (p‑values), and feature correlations.
- Usage: after steps 1–3 in training, click "Feature Analysis"; results appear in the log and can be cross-checked with the training visualization.

## 8) Fabric Analysis (T and P')
This analysis explores how fabric parameters vary across volume thresholds.

### Scientific procedure
- For each volume threshold (generated from the global minimum with a fixed log10 step of 0.25, also including the two dual thresholds), retain particles with \(V_{mm^3} \ge \text{threshold}\).
- For each threshold with enough particles (\(N \ge 50\)):
  1. Bootstrap the retained set \(B\) times (default 1000). Each bootstrap draws \(N\) particles with replacement.
  2. For each bootstrap sample, construct the per-particle log‑ellipsoid tensors from eigenvalues/eigenvectors and compute the log‑Euclidean mean fabric tensor \(\overline{\log E}\).
  3. Obtain eigenvalues of \(\overline{\log E}\), exponentiate them to get mean-ellipsoid principal values, then compute \(T\) and \(P'\) following Jelínek (1981).
- T and P' bootstrap samples are summarized via boxplots across thresholds. Loose and Strict thresholds are annotated.

### Usage
After completing "Load Test Data" and "Predict Analysis", click "Fabric Boxplots". The figures are saved to `outputs/`.

## 9) Best Practices
- Always enter correct voxel sizes (mm). All threshold conversions between voxels and mm³ depend on \(a^3\).
- Use Feature Analysis for interpretability before/after model training.
- Fabric bootstrap is computationally intensive; expect longer runtimes for large datasets.

## 10) Troubleshooting
- Missing LightGBM: the app falls back to RandomForest automatically.
- Missing columns: ensure all required columns are present verbatim.
- Outputs: all generated artifacts (figures, XLSX, TXT, models) are under `outputs/`.

---
For scripting or API usage, consult the modules in `src/ml_threshold_selection/` and reuse the functions directly (e.g., `training_pipeline.train_model_pipeline`, `prediction_analysis.compute_dual_thresholds`, `fabric_pipeline.run_fabric_boxplots`).
