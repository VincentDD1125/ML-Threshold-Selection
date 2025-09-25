# User Guide

This guide provides detailed instructions for using the ML Threshold Selection system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Supervised Learning](#supervised-learning)
5. [Semi-supervised Learning](#semi-supervised-learning)
6. [GUI Usage](#gui-usage)
7. [Command Line Usage](#command-line-usage)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip or conda package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-threshold-selection.git
cd ml-threshold-selection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install from PyPI (when available)

```bash
pip install ml-threshold-selection
```

## Quick Start

### Supervised Learning

```python
from ml_threshold_selection import SupervisedThresholdLearner, FeatureEngineer
import pandas as pd

# Load your data
df = pd.read_csv('your_particle_data.csv')

# Extract features
feature_engineer = FeatureEngineer()
features = feature_engineer.extract_all_features(df)

# Train model
learner = SupervisedThresholdLearner()
learner.train(features, df['label'].values)

# Analyze sample
results = learner.analyze_sample(df)
print(f"Optimal threshold: {results['threshold']:.2e} mm³")
```

### Semi-supervised Learning

```python
from ml_threshold_selection import SemiSupervisedThresholdLearner

# Initialize learner
learner = SemiSupervisedThresholdLearner()

# Add expert thresholds
learner.add_expert_threshold('sample_001', 1.23e-06, confidence=1.0)
learner.add_expert_threshold('sample_002', 2.45e-06, confidence=0.9)

# Load sample data
learner.load_sample_data('sample_001', 'sample_001_data.csv')
learner.load_sample_data('sample_002', 'sample_002_data.csv')

# Train model
learner.train(method='threshold_based', model_type='lightgbm')

# Analyze sample
results = learner.analyze_sample(df)
print(f"Optimal threshold: {results['threshold']:.2e} mm³")
```

## Data Preparation

### Required CSV Columns

Your particle data CSV must contain these columns:

**Essential columns:**
- `Volume3d (mm^3) `: Particle volume in mm³
- `EigenVal1`, `EigenVal2`, `EigenVal3`: Ellipsoid eigenvalues
- `EigenVec1X`, `EigenVec1Y`, `EigenVec1Z`: First principal axis direction
- `EigenVec2X`, `EigenVec2Y`, `EigenVec2Z`: Second principal axis direction
- `EigenVec3X`, `EigenVec3Y`, `EigenVec3Z`: Third principal axis direction

**Optional columns:**
- `ExtentMin1`, `ExtentMax1`: X-direction bounding box
- `ExtentMin2`, `ExtentMax2`: Y-direction bounding box
- `ExtentMin3`, `ExtentMax3`: Z-direction bounding box
- `VoxelFaceArea`: Voxel surface area
- `SampleID`: Sample identifier

**For supervised learning:**
- `label`: Artifact label (0=normal particle, 1=artifact)

### Data Quality

- Ensure all required columns are present
- Check for missing values (they will be filled with 0)
- Verify data types (numeric for all feature columns)
- Remove any non-particle rows

## Supervised Learning

### Step 1: Prepare Training Data

1. **Collect particle data** with all required columns
2. **Label particles** as normal (0) or artifact (1)
3. **Ensure balanced dataset** (aim for 20-40% artifacts)

### Step 2: Train Model

```python
from ml_threshold_selection import SupervisedThresholdLearner, FeatureEngineer

# Load and prepare data
df = pd.read_csv('training_data.csv')
feature_engineer = FeatureEngineer()
features = feature_engineer.extract_all_features(df)

# Train model
learner = SupervisedThresholdLearner()
results = learner.train(features, df['label'].values)

print(f"Training completed: {results['train_auc']:.3f} AUC")
```

### Step 3: Apply to New Samples

```python
# Load new sample
new_df = pd.read_csv('new_sample.csv')

# Analyze
results = learner.analyze_sample(new_df)
print(f"Threshold: {results['threshold']:.2e} mm³")
print(f"Removal rate: {results['removal_rate']:.1f}%")
```

## Semi-supervised Learning

### Step 1: Collect Expert Thresholds

Create a CSV file with expert-determined thresholds:

```csv
sample_id,threshold,confidence,notes
sample_001,1.23e-06,1.0,Determined via stereographic projection
sample_002,2.45e-06,0.9,Partially uncertain
```

### Step 2: Load Sample Data

```python
learner = SemiSupervisedThresholdLearner()

# Add thresholds
learner.add_expert_threshold('sample_001', 1.23e-06, 1.0)
learner.add_expert_threshold('sample_002', 2.45e-06, 0.9)

# Load data
learner.load_sample_data('sample_001', 'sample_001_data.csv')
learner.load_sample_data('sample_002', 'sample_002_data.csv')
```

### Step 3: Train Model

```python
# Choose pseudo-label method
results = learner.train(
    method='threshold_based',  # or 'threshold_with_features', 'threshold_with_uncertainty'
    model_type='lightgbm'      # or 'random_forest'
)

print(f"Training completed: {results['train_score']:.3f}")
```

## GUI Usage

### Launch GUI

```bash
# Supervised learning GUI
python -m ml_threshold_selection.gui.supervised_gui

# Semi-supervised learning GUI
python -m ml_threshold_selection.gui.semi_supervised_gui
```

### GUI Workflow

1. **Load Data**: Use file dialogs to load CSV files
2. **Configure Parameters**: Set training parameters and thresholds
3. **Train Model**: Click train button and wait for completion
4. **Analyze Results**: View plots and statistics
5. **Export Results**: Save analysis results and reports

## Command Line Usage

### Supervised Learning

```bash
# Run demo
python examples/scripts/run_supervised_demo.py

# Train model
python -m ml_threshold_selection.scripts.train_supervised \
    --data training_data.csv \
    --output model.joblib
```

### Semi-supervised Learning

```bash
# Run demo
python examples/scripts/run_semi_supervised_demo.py

# Train model
python -m ml_threshold_selection.scripts.train_semi_supervised \
    --thresholds expert_thresholds.csv \
    --data sample_data/ \
    --output model.joblib
```

## Troubleshooting

### Common Issues

**"KeyError: 'Volume3d (mm^3) '"**
- Check column names exactly match requirements
- Ensure space after "mm^3" in column name

**"Model not trained yet"**
- Call `train()` method before using `predict_proba()` or `analyze_sample()`

**"No training data available"**
- Ensure expert thresholds and sample data are loaded
- Check sample IDs match between thresholds and data

**"Feature mismatch"**
- Ensure all samples have same column structure
- Re-train model with consistent feature set

### Performance Issues

**Slow training:**
- Reduce number of features
- Use smaller dataset for testing
- Try Random Forest instead of LightGBM

**Memory issues:**
- Process samples in batches
- Reduce feature dimensionality
- Use data types with lower memory footprint

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/ml-threshold-selection/issues)
- Read [API Reference](api_reference.md)
- Join [Discussions](https://github.com/yourusername/ml-threshold-selection/discussions)

## Advanced Usage

### Custom Feature Engineering

```python
from ml_threshold_selection.feature_engineering import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def extract_custom_features(self, df):
        # Add your custom features here
        features = {}
        features['custom_metric'] = df['Volume3d (mm^3) '] / df['EigenVal1']
        return pd.DataFrame(features)
    
    def extract_all_features(self, df):
        # Combine standard and custom features
        standard_features = super().extract_all_features(df)
        custom_features = self.extract_custom_features(df)
        return pd.concat([standard_features, custom_features], axis=1)
```

### Custom Threshold Finder

```python
from ml_threshold_selection.threshold_finder import AdaptiveThresholdFinder

class CustomThresholdFinder(AdaptiveThresholdFinder):
    def find_threshold(self, volumes, probabilities, method='custom'):
        # Implement your custom threshold finding logic
        # ...
        return threshold, uncertainty
```

### Batch Processing

```python
import glob
from pathlib import Path

# Process multiple samples
sample_files = glob.glob('data/samples/*.csv')
results = []

for file_path in sample_files:
    df = pd.read_csv(file_path)
    analysis = learner.analyze_sample(df)
    results.append({
        'file': file_path,
        'threshold': analysis['threshold'],
        'removal_rate': analysis['removal_rate']
    })

# Save batch results
results_df = pd.DataFrame(results)
results_df.to_csv('batch_results.csv', index=False)
```
