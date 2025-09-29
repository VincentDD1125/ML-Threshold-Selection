# Deployment Guide

This guide explains how to deploy the ML Threshold Selection system to GitHub and make it available to the community.

## 🚀 GitHub Deployment Steps

### 1. Create GitHub Repository

1. **Go to GitHub** and create a new repository:
   - Repository name: `ml-threshold-selection`
   - Description: "Machine learning-driven adaptive threshold selection for XRCT particle analysis"
   - Visibility: Public
   - Initialize with README: No (we already have one)

2. **Clone the repository locally:**
   ```bash
   git clone https://github.com/yourusername/ml-threshold-selection.git
   cd ml-threshold-selection
   ```

### 2. Upload Project Files

1. **Copy all project files** to the repository directory
2. **Initialize git and add files:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ML Threshold Selection v1.0.0"
   git branch -M main
   git remote add origin https://github.com/yourusername/ml-threshold-selection.git
   git push -u origin main
   ```

### 3. Configure Repository Settings

1. **Go to repository Settings**
2. **Set up branch protection:**
   - Go to "Branches" → "Add rule"
   - Branch name pattern: `main`
   - Enable "Require pull request reviews"
   - Enable "Require status checks to pass before merging"

3. **Configure GitHub Pages** (for documentation):
   - Go to "Pages" in Settings
   - Source: "Deploy from a branch"
   - Branch: `gh-pages`

### 4. Set Up CI/CD Pipeline

The GitHub Actions workflow is already configured in `.github/workflows/ci.yml`. It will:
- Run tests on multiple Python versions and OS
- Check code quality (flake8, black, mypy)
- Build and test the package
- Upload coverage reports

### 5. Create Release

1. **Go to "Releases"** in the repository
2. **Click "Create a new release"**
3. **Fill in details:**
   - Tag version: `v1.0.0`
   - Release title: `ML Threshold Selection v1.0.0`
   - Description: Copy from CHANGELOG.md
4. **Publish release**

## 📦 PyPI Package Deployment

### 1. Prepare for PyPI

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Test package build:**
   ```bash
   python -m build
   twine check dist/*
   ```

### 2. Upload to PyPI

1. **Install twine:**
   ```bash
   pip install twine
   ```

2. **Upload to TestPyPI first:**
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. **Test installation:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ ml-threshold-selection
   ```

4. **Upload to PyPI:**
   ```bash
   twine upload dist/*
   ```

## 🔧 Configuration Files

### Repository Configuration

- **`.github/workflows/ci.yml`**: Continuous Integration
- **`.github/ISSUE_TEMPLATE/`**: Issue templates
- **`.github/PULL_REQUEST_TEMPLATE.md`**: PR template
- **`.gitignore`**: Git ignore rules
- **`LICENSE`**: MIT License
- **`CONTRIBUTING.md`**: Contribution guidelines

### Package Configuration

- **`setup.py`**: Package setup
- **`pyproject.toml`**: Modern Python packaging
- **`requirements.txt`**: Dependencies
- **`MANIFEST.in`**: Include additional files

## 📚 Documentation Deployment

### 1. Sphinx Documentation

1. **Install Sphinx:**
   ```bash
   pip install sphinx sphinx-rtd-theme
   ```

2. **Initialize docs:**
   ```bash
   cd docs
   sphinx-quickstart
   ```

3. **Configure `conf.py`:**
   ```python
   import os
   import sys
   sys.path.insert(0, os.path.abspath('..'))
   
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.viewcode',
       'sphinx.ext.napoleon',
   ]
   
   html_theme = 'sphinx_rtd_theme'
   ```

4. **Build documentation:**
   ```bash
   sphinx-build -b html . _build/html
   ```

### 2. GitHub Pages

1. **Create `gh-pages` branch:**
   ```bash
   git checkout --orphan gh-pages
   git rm -rf .
   cp -r docs/_build/html/* .
   git add .
   git commit -m "Add documentation"
   git push origin gh-pages
   ```

2. **Enable GitHub Pages** in repository settings

## 🏷️ Version Management

### Semantic Versioning

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Process

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create git tag:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. **Create GitHub release**
5. **Build and upload to PyPI**

## 🔒 Security Considerations

### Repository Security

1. **Enable 2FA** on GitHub account
2. **Use SSH keys** for authentication
3. **Review dependency vulnerabilities** regularly
4. **Keep dependencies updated**

### Package Security

1. **Sign releases** with GPG
2. **Use secure upload** (HTTPS)
3. **Verify package integrity**
4. **Monitor for security issues**

## 📊 Community Management

### Issue Management

1. **Use issue templates** for bug reports and feature requests
2. **Label issues** appropriately
3. **Respond promptly** to community questions
4. **Close resolved issues**

### Pull Request Management

1. **Require reviews** for all PRs
2. **Run CI checks** before merging
3. **Use descriptive commit messages**
4. **Squash commits** when appropriate

### Community Guidelines

1. **Be welcoming** and inclusive
2. **Provide clear documentation**
3. **Respond to questions** helpfully
4. **Recognize contributors**

## 🚀 Launch Checklist

### Pre-Launch

- [ ] All tests passing
- [ ] Documentation complete
- [ ] README.md updated
- [ ] LICENSE file added
- [ ] Contributing guidelines written
- [ ] Issue templates created
- [ ] CI/CD pipeline working
- [ ] Code quality checks passing

### Launch Day

- [ ] Create GitHub repository
- [ ] Upload all files
- [ ] Configure repository settings
- [ ] Create first release
- [ ] Announce on social media
- [ ] Share with relevant communities

### Post-Launch

- [ ] Monitor issues and PRs
- [ ] Respond to community feedback
- [ ] Plan next release
- [ ] Update documentation as needed
- [ ] Celebrate success! 🎉

## 📞 Support

For deployment questions or issues:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/ml-threshold-selection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ml-threshold-selection/discussions)
- **Email**: your.email@example.com

---

**Happy Deploying! 🚀**
