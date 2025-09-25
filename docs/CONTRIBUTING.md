# Contributing to ML Threshold Selection

Thank you for your interest in contributing to ML Threshold Selection! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/yourusername/ml-threshold-selection/issues) page
- Search existing issues before creating a new one
- Use clear, descriptive titles
- Include steps to reproduce bugs
- Provide system information (OS, Python version, etc.)

### Suggesting Enhancements

- Use the [GitHub Discussions](https://github.com/yourusername/ml-threshold-selection/discussions) for feature requests
- Clearly describe the proposed enhancement
- Explain why it would be useful
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests** to ensure everything works
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 🛠️ Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- pip or conda

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/ml-threshold-selection.git
cd ml-threshold-selection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_threshold_selection

# Run specific test file
pytest tests/test_feature_engineering.py
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation

- Write docstrings for all public functions and classes
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Update README.md for user-facing changes
- Add examples for new features

### Testing

- Write tests for all new functionality
- Aim for >90% code coverage
- Use descriptive test names
- Test edge cases and error conditions

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add new feature for threshold optimization
fix: resolve issue with GUI display
docs: update installation instructions
test: add unit tests for feature engineering
refactor: improve code organization
```

## 🏗️ Project Structure

```
src/ml_threshold_selection/
├── __init__.py
├── supervised_learner.py      # Supervised learning system
├── semi_supervised_learner.py # Semi-supervised learning system
├── feature_engineering.py    # Feature extraction
├── threshold_finder.py       # Adaptive threshold selection
└── gui/                      # GUI components
    ├── __init__.py
    ├── supervised_gui.py
    └── semi_supervised_gui.py
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **GUI Tests**: Test user interface functionality
4. **Performance Tests**: Test with large datasets

### Test Data

- Use synthetic data for unit tests
- Include small real datasets for integration tests
- Ensure test data is properly licensed

### Test Naming

```python
def test_feature_extraction_with_valid_data():
    """Test feature extraction with valid input data."""
    pass

def test_feature_extraction_with_invalid_data():
    """Test feature extraction handles invalid input gracefully."""
    pass
```

## 📚 Documentation Guidelines

### Code Documentation

```python
def extract_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Extract geometric features from particle data.
    
    Args:
        df: DataFrame containing particle data with required columns
        
    Returns:
        DataFrame with extracted geometric features
        
    Raises:
        ValueError: If required columns are missing
        TypeError: If input data is not a DataFrame
    """
    pass
```

### User Documentation

- Update README.md for new features
- Add examples to documentation
- Include screenshots for GUI changes
- Document breaking changes

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared
- [ ] Tag created

## 🤔 Questions?

- **General questions**: Use [GitHub Discussions](https://github.com/yourusername/ml-threshold-selection/discussions)
- **Bug reports**: Use [GitHub Issues](https://github.com/yourusername/ml-threshold-selection/issues)
- **Security issues**: Email your.email@example.com

## 📋 Pull Request Template

When creating a pull request, please include:

### Description
Brief description of changes

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

### Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

### Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## 🎉 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to ML Threshold Selection! 🚀
