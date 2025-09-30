# ML Threshold Selection

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/VincentDD1125/ML-Threshold-Selection/releases)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/VincentDD1125/ML-Threshold-Selection/actions)

A machine learning-driven toolkit for adaptive threshold selection in X-ray computed tomography (XRCT) particle analysis. This package provides both supervised and semi-supervised learning approaches to automatically determine optimal volume thresholds for artifact removal in particle datasets.

## 🚀 Key Features

- **Dual Learning Approaches**: Supervised learning with labeled data and semi-supervised learning with expert thresholds
- **Advanced Feature Engineering**: Resolution-aware 7D log-ellipsoid tensor features for robust particle characterization
- **Dual Threshold Prediction**: Automatic detection of both loose (inflection point) and strict (zero-artifact) thresholds
- **Fabric Analysis**: Comprehensive T and P' parameter analysis with bootstrap confidence intervals
- **Interactive GUI**: User-friendly Tkinter interface for end-to-end workflow
- **Scientific Rigor**: Reproducible methodology with detailed statistical validation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scientific Methodology](#scientific-methodology)
- [Data Requirements](#data-requirements)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/VincentDD1125/ML-Threshold-Selection.git
cd ML-Threshold-Selection

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Install from PyPI (Coming Soon)

```bash
pip install ml-threshold-selection
```

## 🚀 Quick Start

### GUI Interface (Recommended for Beginners)

1. **Prepare your data**:
   ```bash
   # Place training files in examples/data/train/
   # Place test files in examples/data/test/
   ```

2. **Determine expert thresholds**:
   - Use TomoFab or other stereographic projection tools
   - Analyze your particle data using lower hemisphere equal-area projections
   - Identify the optimal volume threshold for each sample that separates artifacts from real particles
   - Record these thresholds for input into the system

3. **Launch the application**:
   ```bash
   python main.py
   ```

4. **Follow the GUI workflow**:
   - Load training data
   - Input expert thresholds (pre-determined from stereographic analysis)
   - Input voxel sizes for each sample
   - Perform feature analysis (optional)
   - Train the model
   - Load test data
   - Run prediction analysis
   - Generate fabric boxplots (optional)
   - Export results

### Command Line Interface

#### Supervised Learning

```bash
# Run with example data
python examples/scripts/run_supervised_demo.py

# Run with your own data
python examples/scripts/run_supervised_demo.py \
  --train_dir /path/to/training/data \
  --test_file /path/to/test/file.xlsx \
  --voxel_size_mm 0.03
```

#### Semi-supervised Learning

```bash
# Run with expert thresholds
python examples/scripts/run_semi_supervised_demo.py \
  --thresholds examples/data/expert_thresholds.csv \
  --data_dir examples/data/train/
```

### Python API

#### Supervised Learning

```python
from src.ml_threshold_selection.supervised_learner import SupervisedThresholdLearner
from src.ml_threshold_selection.feature_engineering import FeatureEngineer
import pandas as pd

# Load your data
df = pd.read_excel('your_particle_data.xlsx')

# Extract features
feature_engineer = FeatureEngineer()
features = feature_engineer.extract_all_features(df)

# Train model
learner = SupervisedThresholdLearner()
learner.train(features, df['label'].values)

# Analyze new sample
results = learner.analyze_sample(df)
print(f"Optimal threshold: {results['threshold']:.2e} mm³")
print(f"Removal rate: {results['removal_rate']:.1f}%")
```

#### Semi-supervised Learning

```python
from src.ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner

# Initialize learner
learner = SemiSupervisedThresholdLearner()

# Add expert thresholds
learner.add_expert_threshold('sample_001', 1.23e-06, confidence=1.0)
learner.add_expert_threshold('sample_002', 2.45e-06, confidence=0.9)

# Load sample data
learner.load_sample_data('sample_001', 'sample_001_data.xlsx')
learner.load_sample_data('sample_002', 'sample_002_data.xlsx')

# Train model
learner.train(method='threshold_based', model_type='lightgbm')

# Analyze sample
results = learner.analyze_sample(df)
print(f"Optimal threshold: {results['threshold']:.2e} mm³")
```

## 🔬 Scientific Methodology

### Expert Threshold Determination

Before using this system, you must determine expert thresholds for your training samples using stereographic projection analysis:

1. **Stereographic Projection Analysis**:
   - Use TomoFab, MTEX, or similar software for lower hemisphere equal-area projections
   - Analyze particle orientations and fabric patterns
   - Identify the volume threshold where artifacts begin to dominate the fabric signal
   - This threshold represents the optimal separation between real particles and imaging artifacts

2. **Threshold Selection Criteria**:
   - **Too Low**: Includes too many artifacts, corrupting fabric analysis
   - **Too High**: Excludes real particles, reducing statistical power
   - **Optimal**: Balances artifact removal with particle retention for reliable fabric analysis

3. **Quality Control**:
   - Verify thresholds using multiple projection methods
   - Cross-validate with manual inspection of particle shapes
   - Ensure consistency across similar sample types

### Feature Engineering

The package employs a resolution-aware 7D log-ellipsoid tensor feature extraction approach:

1. **Ellipsoid Tensor Construction**: Each particle is represented as a 3×3 symmetric tensor based on its principal axes and eigenvalues
2. **Log-Euclidean Mapping**: Tensors are mapped to the log-Euclidean space for linear operations
3. **Resolution Normalization**: Features are normalized by voxel size to ensure cross-sample compatibility
4. **Dimensionality Reduction**: 7 key features are extracted: volume, aspect ratios, and orientation parameters

### Dual Threshold Strategy

The system predicts two complementary thresholds:

- **Loose Threshold**: Identified as the inflection point of the artifact rate curve, representing the optimal balance between artifact removal and particle retention
- **Strict Threshold**: Determined as the threshold that removes all particles with artifact probability > 0.05, ensuring zero false positives

### Fabric Analysis

Comprehensive fabric analysis using the Jelínek (1981) methodology:

1. **Bootstrap Sampling**: For each volume threshold, particles are resampled with replacement
2. **Log-Euclidean Mean**: The mean fabric tensor is computed using log-Euclidean averaging
3. **Eigenvalue Analysis**: Principal values are extracted and used to compute T and P' parameters
4. **Confidence Intervals**: 95% confidence intervals are calculated for statistical validation

### Machine Learning Pipeline

- **Feature Standardization**: All features are standardized using `StandardScaler`
- **Model Selection**: Support for LightGBM, Random Forest, and other scikit-learn classifiers
- **Cross-validation**: Built-in k-fold cross-validation for robust performance estimation
- **Hyperparameter Optimization**: Automated hyperparameter tuning for optimal performance

## 📊 Data Requirements

### Required Columns

Your particle data must contain these essential columns:

| Column Name | Description | Units |
|-------------|-------------|-------|
| `Volume3d (mm^3) ` | Particle volume | mm³ |
| `EigenVal1`, `EigenVal2`, `EigenVal3` | Ellipsoid eigenvalues | dimensionless |
| `EigenVec1X`, `EigenVec1Y`, `EigenVec1Z` | First principal axis direction | unit vector |
| `EigenVec2X`, `EigenVec2Y`, `EigenVec2Z` | Second principal axis direction | unit vector |
| `EigenVec3X`, `EigenVec3Y`, `EigenVec3Z` | Third principal axis direction | unit vector |

### Optional Columns

| Column Name | Description | Units |
|-------------|-------------|-------|
| `ExtentMin1`, `ExtentMax1` | X-direction bounding box | mm |
| `ExtentMin2`, `ExtentMax2` | Y-direction bounding box | mm |
| `ExtentMin3`, `ExtentMax3` | Z-direction bounding box | mm |
| `VoxelFaceArea` | Voxel surface area | mm² |
| `SampleID` | Sample identifier | string |

### Data Quality Guidelines

- **Missing Values**: Will be filled with 0 (ensure this is appropriate for your data)
- **Data Types**: All feature columns must be numeric
- **Volume Units**: Ensure volumes are in mm³ (not voxels)
- **Coordinate System**: Ensure consistent coordinate system across samples

## 💡 Usage Examples

### Example 1: Basic Supervised Learning

```python
import pandas as pd
from src.ml_threshold_selection.supervised_learner import SupervisedThresholdLearner
from src.ml_threshold_selection.feature_engineering import FeatureEngineer

# Load training data with labels
train_df = pd.read_excel('training_data.xlsx')
feature_engineer = FeatureEngineer()
features = feature_engineer.extract_all_features(train_df)

# Train model
learner = SupervisedThresholdLearner()
learner.train(features, train_df['label'].values)

# Analyze new sample
test_df = pd.read_excel('test_sample.xlsx')
results = learner.analyze_sample(test_df)

print(f"Predicted threshold: {results['threshold']:.2e} mm³")
print(f"Expected removal rate: {results['removal_rate']:.1f}%")
```

### Example 2: Semi-supervised Learning with Expert Knowledge

```python
from src.ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner

# Initialize learner
learner = SemiSupervisedThresholdLearner()

# Add expert thresholds from literature or manual analysis
expert_thresholds = {
    'sample_A': (1.2e-06, 1.0),  # (threshold, confidence)
    'sample_B': (2.1e-06, 0.9),
    'sample_C': (0.8e-06, 0.8)
}

for sample_id, (threshold, confidence) in expert_thresholds.items():
    learner.add_expert_threshold(sample_id, threshold, confidence)
    learner.load_sample_data(sample_id, f'{sample_id}_data.xlsx')

# Train with different methods
learner.train(method='threshold_based', model_type='lightgbm')
```

### Example 3: Fabric Analysis

```python
from src.ml_threshold_selection.fabric_pipeline import run_fabric_boxplots

# Run fabric analysis for a sample
results = run_fabric_boxplots(
    data=test_df,
    v_min_star=1e-06,  # Reference threshold
    sample_id='test_sample',
    output_dir='fabric_analysis_output'
)

# Access results
print(f"Inflection threshold: {results['inflection_threshold']:.2e} mm³")
print(f"Zero-artifact threshold: {results['zero_artifact_threshold']:.2e} mm³")
```

## 📚 API Reference

### Core Classes

#### `SupervisedThresholdLearner`

Main class for supervised learning approach.

```python
class SupervisedThresholdLearner:
    def train(self, features, labels, model_type='lightgbm')
    def analyze_sample(self, data)
    def predict_proba(self, features)
    def find_dual_thresholds(self, volumes, probabilities)
```

#### `SemiSupervisedThresholdLearner`

Main class for semi-supervised learning approach.

```python
class SemiSupervisedThresholdLearner:
    def add_expert_threshold(self, sample_id, threshold, confidence)
    def load_sample_data(self, sample_id, file_path)
    def train(self, method='threshold_based', model_type='lightgbm')
    def analyze_sample(self, data)
```

#### `FeatureEngineer`

Feature extraction and engineering.

```python
class FeatureEngineer:
    def extract_all_features(self, data)
    def extract_ellipsoid_features(self, data)
    def extract_resolution_aware_features(self, data)
```

### Utility Functions

#### `run_fabric_boxplots`

Generate fabric analysis plots and statistics.

```python
def run_fabric_boxplots(data, v_min_star, sample_id, output_dir=None):
    """
    Run comprehensive fabric analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Particle data with required columns
    v_min_star : float
        Reference threshold for grid generation
    sample_id : str
        Sample identifier
    output_dir : str, optional
        Output directory for plots and results
        
    Returns:
    --------
    dict : Analysis results including thresholds and statistics
    """
```

## 🏗️ Project Structure

```
ML_Threshold_Selection/
├── main.py                                    # Application entry point
├── src/
│   ├── ml_threshold_selection/
│   │   ├── app_controller.py                  # GUI controller
│   │   ├── ui_layout.py                       # Tkinter interface
│   │   ├── ui_visualization.py                # Plotting utilities
│   │   ├── data_io.py                         # Data loading/saving
│   │   ├── training_pipeline.py               # Training workflow
│   │   ├── prediction_analysis.py             # Prediction analysis
│   │   ├── export_results.py                  # Results export
│   │   ├── labeling.py                        # Label generation
│   │   ├── feature_utils.py                   # Feature utilities
│   │   ├── fabric_thresholds.py               # Threshold grid helpers
│   │   ├── fabric_bootstrap.py                # Bootstrap analysis
│   │   ├── fabric_logging.py                  # Logging utilities
│   │   └── fabric_pipeline.py                 # Fabric analysis pipeline
│   ├── analysis/
│   │   └── ellipsoid_feature_analyzer.py      # Feature analysis
│   └── features/
│       ├── ellipsoid_feature_engineering.py
│       ├── ellipsoid_feature_engineering_legacy.py
│       └── res_aware_feature_engineering.py
├── examples/
│   ├── data/
│   │   ├── train/                             # Training data
│   │   ├── test/                              # Test data
│   │   ├── expert_thresholds.csv              # Expert thresholds
│   │   └── sample_particles.csv               # Sample data
│   ├── config/
│   │   └── thresholds_voxels.csv              # Configuration
│   └── scripts/
│       ├── run_supervised_demo.py             # Supervised demo
│       ├── run_semi_supervised_demo.py        # Semi-supervised demo
│       └── validate_data.py                   # Data validation
├── docs/
│   ├── user_guide.md                          # User documentation
│   └── guides/                                # Detailed guides
├── tests/                                     # Test suite
├── outputs/                                   # Generated outputs
├── requirements.txt                           # Dependencies
├── pyproject.toml                            # Project configuration
└── setup.py                                  # Installation script
```

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/ml_threshold_selection --cov-report=html

# Run specific test
python -m pytest tests/test_feature_engineering.py -v
```

## 📈 Performance Optimization

### Memory Optimization

- Use `float32` for large datasets to reduce memory usage
- Process samples in batches for very large datasets
- Enable data type optimization in pandas

### Speed Optimization

- Use LightGBM for faster training (default)
- Enable parallel processing where available
- Consider feature selection for very high-dimensional data

### Example: Batch Processing

```python
import glob
from pathlib import Path

# Process multiple samples efficiently
sample_files = glob.glob('data/samples/*.xlsx')
results = []

for file_path in sample_files:
    df = pd.read_excel(file_path)
    analysis = learner.analyze_sample(df)
    results.append({
        'file': Path(file_path).stem,
        'threshold': analysis['threshold'],
        'removal_rate': analysis['removal_rate']
    })

# Save batch results
results_df = pd.DataFrame(results)
results_df.to_csv('batch_analysis_results.csv', index=False)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/VincentDD1125/ML-Threshold-Selection.git
cd ML-Threshold-Selection

# Create development environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [User Guide](docs/user_guide.md)
- **Issues**: [GitHub Issues](https://github.com/VincentDD1125/ML-Threshold-Selection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VincentDD1125/ML-Threshold-Selection/discussions)

## 🙏 Acknowledgments

- Based on the Jelínek (1981) fabric analysis methodology
- Inspired by modern machine learning approaches to geological data analysis
- Built with the scientific Python ecosystem (NumPy, Pandas, scikit-learn, Matplotlib)

---

**Citation**: If you use this software in your research, please cite:

```bibtex
@software{ml_threshold_selection,
  title={ML Threshold Selection: Machine Learning-Driven Adaptive Threshold Selection for XRCT Particle Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/VincentDD1125/ML-Threshold-Selection}
}
```