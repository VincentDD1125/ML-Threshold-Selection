# Project Structure Overview

This document provides a comprehensive overview of the ML Threshold Selection project structure.

## 📁 Directory Tree

```
ML_Threshold_Selection/
├── 📁 src/                            # Source code
│   ├── 📁 ml_threshold_selection/     # Core package
│   │   ├── 📄 __init__.py             # Package initialization
│   │   ├── 📄 feature_engineering.py  # Traditional feature extraction
│   │   ├── 📄 threshold_finder.py     # Adaptive threshold selection
│   │   ├── 📄 supervised_learner.py   # Supervised learning
│   │   ├── 📄 semi_supervised_learner.py # Semi-supervised learning
│   │   └── 📄 data_validator.py       # Data validation
│   ├── 📁 features/                   # Feature engineering modules
│   │   ├── 📄 joshua_feature_engineering.py
│   │   ├── 📄 joshua_feature_engineering_fixed.py
│   │   └── 📄 res_aware_feature_engineering.py
│   └── 📁 analysis/                   # Analysis modules
│       └── 📄 joshua_feature_analyzer.py
├── 📁 scripts/                        # CLI scripts and experiments
│   ├── 📄 compare_methods.py
│   ├── 📄 test_joshua_method.py
│   ├── 📄 test_joshua_method_fixed.py
│   └── 📄 test_compare_fixed_joshua_vs_traditional.py
├── 📁 docs/                           # Documentation
│   ├── 📄 INDEX.md                    # Documentation index
│   ├── 📄 user_guide.md               # User guide
│   ├── 📄 PROJECT_STRUCTURE.md        # This file
│   ├── 📄 CHANGELOG.md                # Version history
│   ├── 📄 CONTRIBUTING.md             # Contribution guidelines
│   ├── 📄 CONTRIBUTORS.md             # Contributors list
│   ├── 📄 DEPLOYMENT.md               # Deployment guide
│   └── 📁 guides/                     # Detailed guides
│       ├── 📄 ADVANCED_FEATURE_ANALYSIS_GUIDE.md
│       ├── 📄 COMPLETE_FEATURE_UPDATE_SUMMARY.md
│       ├── 📄 COMPREHENSIVE_FIXES_AND_VOXEL_EXPLANATION.md
│       ├── 📄 ELLIPSOID_FEATURE_ENGINEERING_GUIDE.md
│       ├── 📄 ENGLISH_CONVERSION_SUMMARY.md
│       ├── 📄 FEATURE_REMOVAL_AND_ENCODING_FIX.md
│       ├── 📄 FINAL_FIXES_SUMMARY.md
│       ├── 📄 PERFORMANCE_OPTIMIZATION_GUIDE.md
│       ├── 📄 PREDICTION_ACCURACY_FIX_GUIDE.md
│       ├── 📄 REDUNDANT_FEATURES_FIX_GUIDE.md
│       ├── 📄 SAMPLEID_FIX_GUIDE.md
│       ├── 📄 STRING_COLUMN_FIX_GUIDE.md
│       ├── 📄 VISUALIZATION_GUIDE.md
│       ├── 📄 VOLUME_FEATURE_CORRECTION_GUIDE.md
│       └── 📄 VOXEL_SIZE_INPUT_GUIDE.md
├── 📁 examples/                       # Example data and scripts
│   ├── 📁 data/                       # Sample data files
│   │   ├── 📄 sample_particles.csv    # Example particle data
│   │   ├── 📄 sample_particles.xlsx   # Example particle data (Excel)
│   │   ├── 📄 expert_thresholds.csv   # Example expert thresholds
│   │   └── 📄 expert_thresholds.xlsx  # Example expert thresholds (Excel)
│   └── 📁 scripts/                    # Example scripts
│       ├── 📄 run_supervised_demo.py
│       ├── 📄 run_semi_supervised_demo.py
│       └── 📄 validate_data.py
├── 📁 assets/                         # Exported figures and images
│   ├── 📄 *.png                       # Generated charts and plots
│   └── 📄 *.svg                       # Vector graphics
├── 📁 archive/                        # Deprecated/legacy code
│   └── 📄 feature_analysis_tool.py    # Legacy feature analysis
├── 📁 tests/                          # Test suite
│   └── 📄 test_feature_engineering.py
├── 📄 main.py                         # Main GUI application
├── 📄 README.md                       # Main documentation
├── 📄 LICENSE                         # MIT License
├── 📄 pyproject.toml                  # Modern Python packaging
├── 📄 requirements.txt                # Dependencies
└── 📄 setup.py                        # Package setup
```

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Threshold Selection                   │
├─────────────────────────────────────────────────────────────┤
│  Input Data → Feature Engineering → ML Classifier → Results │
│       ↓              ↓                ↓            ↓        │
│  Particle CSV → Feature (Ellipsoid) → LightGBM → Dual Thresholds│
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering Methods

1. **Feature Analysis (Primary)**
   - 7 features: VoxelCount + 6 log-ellipsoid tensor components
   - Resolution-aware: voxel domain training, mm³ display
   - StandardScaler applied to all features

2. **Traditional Method (Legacy)**
   - 25-30 geometric and statistical features
   - Deprecated in favor of feature analysis

### Dual Threshold Analysis

- **Loose Threshold (Inflection)**: Inflection point from artifact rate curve
- **Strict Threshold (P>0.05)**: Remove all particles with P > 0.05
- Both thresholds displayed in voxels and mm³

## 📦 Package Structure

### Core Package (`src/ml_threshold_selection/`)

- **`__init__.py`**: Package initialization and exports
- **`feature_engineering.py`**: Traditional feature extraction (legacy)
- **`threshold_finder.py`**: Adaptive threshold selection
- **`supervised_learner.py`**: Supervised learning system
- **`semi_supervised_learner.py`**: Semi-supervised learning system
- **`data_validator.py`**: Data validation utilities

### Feature Engineering (`src/features/`)

- **`joshua_feature_engineering.py`**: Original ellipsoid-tensor implementation
- **`joshua_feature_engineering_fixed.py`**: Fixed ellipsoid-tensor implementation
- **`res_aware_feature_engineering.py`**: Resolution-aware Joshua features (current)

### Analysis (`src/analysis/`)

- **`joshua_feature_analyzer.py`**: Feature analysis and visualization

### Scripts (`scripts/`)

- **`compare_methods.py`**: Method comparison utilities
- **`test_*.py`**: Various test and validation scripts

## 🧪 Testing Structure

### Test Organization

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **GUI Tests**: Test user interface functionality
- **Method Comparison**: Compare different feature engineering approaches

### Test Files

- `test_feature_engineering.py`: Feature extraction tests
- `test_joshua_method.py`: Joshua method tests
- `test_compare_fixed_joshua_vs_traditional.py`: Method comparison tests

## 📚 Documentation Structure

### User Documentation

- **`README.md`**: Main project documentation (root directory)
- **`docs/user_guide.md`**: Detailed usage instructions
- **`docs/INDEX.md`**: Documentation index

### Developer Documentation

- **`CONTRIBUTING.md`**: Contribution guidelines
- **`DEPLOYMENT.md`**: Deployment instructions
- **`PROJECT_STRUCTURE.md`**: This file

### Implementation Guides

- **`docs/guides/`**: Detailed implementation and fix guides
- **`docs/guides/JOSHUA_METHOD_IMPLEMENTATION.md`**: Joshua method details
- **`docs/guides/VOXEL_SIZE_INPUT_GUIDE.md`**: Voxel size handling

## 🔧 Configuration Files

### Package Configuration

- **`setup.py`**: Traditional Python packaging
- **`pyproject.toml`**: Modern Python packaging (PEP 518)
- **`requirements.txt`**: Dependencies list

### Development Configuration

- **`CHANGELOG.md`**: Version history
- **`CONTRIBUTORS.md`**: Contributors list

## 📊 Data Flow

### Current Workflow (Joshua Method)

```
Training Data → Voxel Size Input → Joshua Features → Model Training → Dual Thresholds
     ↓                ↓                ↓                ↓              ↓
Particle CSV → mm³ → VoxelCount+6 → LightGBM → Loose+Strict → Reports
```

### Legacy Workflow (Traditional Method)

```
Training Data → Traditional Features → Model Training → Single Threshold
     ↓                ↓                    ↓              ↓
Particle CSV → 25-30 Features → LightGBM → A(V) Curve → Reports
```

## 🚀 Key Features by Directory

### Core Application

| File | Purpose | Key Features |
|------|---------|--------------|
| `main.py` | Main GUI | Feature analysis, dual thresholds, visualization |

### Feature Engineering

| File | Purpose | Key Features |
|------|---------|--------------|
| `res_aware_feature_engineering.py` | Current features | VoxelCount + 6 ellipsoid-tensor features, StandardScaler |
| `joshua_feature_engineering_fixed.py` | Fixed features | Ceiling voxel count, minimum 1 |
| `joshua_feature_engineering.py` | Original features | Basic log-ellipsoid tensor |

### Analysis and Scripts

| File | Purpose | Key Features |
|------|---------|--------------|
| `joshua_feature_analyzer.py` | Feature analysis | Statistical analysis, visualization |
| `compare_methods.py` | Method comparison | ROC/AUC comparison, confusion matrices |

## 📈 Scalability Considerations

### Code Organization

- **Modular design**: Easy to extend and modify
- **Clear separation**: Features, analysis, scripts, docs
- **Type hints**: Better code maintainability
- **Documentation**: Comprehensive guides and examples

### Performance

- **Resolution-aware**: Voxel domain training eliminates resolution bias
- **Feature scaling**: StandardScaler ensures balanced learning
- **Efficient algorithms**: Optimized for large particle datasets
- **Memory management**: Careful handling of large arrays

### Extensibility

- **Plugin architecture**: Easy to add new feature engineering methods
- **Custom features**: User-defined feature sets
- **Multiple thresholds**: Flexible threshold selection strategies
- **API design**: Clean, consistent interfaces

## 🔧 Maintenance

### Regular Tasks

- **Dependency updates**: Keep packages current
- **Method validation**: Ensure feature method accuracy
- **Documentation updates**: Keep guides current
- **Performance monitoring**: Track training and prediction times

### Monitoring

- **Test coverage**: Maintain high coverage across methods
- **Code quality**: Use linting and formatting
- **Performance metrics**: Monitor AUC and threshold accuracy
- **User feedback**: Address issues and feature requests

---

This structure provides a solid foundation for a professional, maintainable, and extensible machine learning package focused on XRCT particle analysis. The modular design allows for easy development, testing, and deployment while keeping the codebase organized and accessible to both users and contributors.