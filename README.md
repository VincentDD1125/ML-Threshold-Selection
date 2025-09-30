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

### GUI Interface (Recommended)

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

```bash
# Supervised learning demo
python examples/scripts/run_supervised_demo.py

# Semi-supervised learning demo
python examples/scripts/run_semi_supervised_demo.py \
  --thresholds examples/data/expert_thresholds.csv \
  --data_dir examples/data/train/
```

### Python API

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
```

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

### Data Quality Guidelines

- **Missing Values**: Will be filled with 0 (ensure this is appropriate for your data)
- **Data Types**: All feature columns must be numeric
- **Volume Units**: Ensure volumes are in mm³ (not voxels)
- **Coordinate System**: Ensure consistent coordinate system across samples

## 📚 Documentation

- **[User Guide](docs/user_guide.md)**: Complete user manual with step-by-step instructions
- **[Examples](examples/)**: Sample data and demonstration scripts

## 🏗️ Project Structure

```
ML_Threshold_Selection/
├── main.py                                    # Application entry point
├── src/ml_threshold_selection/                # Core package
├── examples/                                  # Sample data and scripts
├── docs/                                      # Documentation
├── tests/                                     # Test suite
├── requirements.txt                           # Dependencies
└── pyproject.toml                            # Project configuration
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/ml_threshold_selection --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

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

## 📚 Key References

- **Jelínek, V. (1981)**: Characterization of the magnetic fabric of rocks. *Tectonophysics*, 79(1-4), T63-T67
- **TomoFab Software**: https://github.com/ctlab/TomoFab

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