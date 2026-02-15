# PyRamEx

**A Python Ramanome Analysis Toolkit for Machine Learning and Deep Learning**

[![CI/CD](https://github.com/openclaw/pyramex/actions/workflows/ci.yml/badge.svg)](https://github.com/openclaw/pyramex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/openclaw/pyramex/branch/main/graph/badge.svg)](https://codecov.io/gh/openclaw/pyramex)
[![PyPI version](https://badge.fury.io/py/pyramex.svg)](https://pypi.org/project/pyramex/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyramex.svg)](https://pypi.org/project/pyramex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**PyRamEx** is a Python reimplementation of [RamEx](https://github.com/qibebt-bioinfo/RamEx) (R package), specifically optimized for machine learning and deep learning workflows. It provides comprehensive tools for Raman spectroscopic data analysis with seamless integration with modern ML/DL frameworks.

### Key Features

✅ **ML/DL-Native Design** - NumPy/Pandas data structures, Scikit-learn/PyTorch/TensorFlow integration  
✅ **Method Chaining** - Fluent API for preprocessing pipelines  
✅ **Modern Python** - Type hints, async support, comprehensive testing  
✅ **GPU Acceleration** - Optional CUDA support (replaces OpenCL)  
✅ **Interactive Visualization** - Plotly/Matplotlib support  
✅ **Jupyter Friendly** - Designed for notebook-based exploration  

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install pyramex

# With ML/DL dependencies
pip install pyramex[ml]

# With GPU support
pip install pyramex[gpu]
```

### Basic Usage

```python
from pyramex import Ramanome, load_spectra

# Load data
data = load_spectra('path/to/spectra/')

# Preprocess with method chaining
data = data.smooth(window_size=5) \
           .remove_baseline(method='polyfit') \
           .normalize(method='minmax')

# Quality control
qc = data.quality_control(method='icod', threshold=0.05)
data_clean = data[qc.good_samples]

# Dimensionality reduction
data_clean.reduce(method='pca', n_components=2)
data_clean.plot_reduction(method='pca')

# Machine Learning integration
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = data_clean.to_sklearn_format()
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

---

## 📚 Documentation

- **Installation Guide**: [docs/installation.md](docs/installation.md)
- **Quick Start Tutorial**: [docs/tutorial.md](docs/tutorial.md)
- **API Reference**: [docs/api.md](docs/api.md)
- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **Developer Guide**: [docs/developer_guide.md](docs/developer_guide.md)

---

## 🎓 Comparison with RamEx (R)

| Feature | RamEx (R) | PyRamEx (Python) |
|---------|-----------|-------------------|
| **Language** | R | Python 3.8+ |
| **ML Integration** | Limited | Native (sklearn, PyTorch, TF) |
| **GPU Support** | OpenCL | CUDA (optional) |
| **Data Format** | S4 objects | NumPy/Pandas |
| **Visualization** | ggplot2 | Plotly/Matplotlib |
| **Interactivity** | Shiny | Jupyter + Streamlit |
| **API Style** | R functions | Python method chaining |

---

## 📊 Project Structure

```
pyramex/
├── pyramex/
│   ├── __init__.py              # Package entry point
│   ├── core/                    # Core data structures
│   ├── io/                      # Data loading
│   ├── preprocessing/           # Spectral preprocessing
│   ├── qc/                      # Quality control
│   ├── features/                # Feature engineering
│   ├── ml/                      # ML/DL integration
│   └── visualization/           # Plotting tools
├── tests/                       # Unit tests
├── examples/                    # Jupyter notebooks
├── docs/                        # Documentation
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── README.md                    # This file
└── .github/workflows/           # CI/CD
```

---

## 🔬 Features

### Data Loading
- Support for multiple Raman file formats
- Automatic format detection
- Batch loading from directories

### Preprocessing
- Smoothing (Savitzky-Golay)
- Baseline removal (polyfit, ALS, airPLS)
- Normalization (minmax, zscore, area, max, vecnorm)
- Spectral cutoff and derivatives

### Quality Control
- ICOD (Inverse Covariance-based Outlier Detection)
- MCD (Minimum Covariance Determinant)
- T2 (Hotelling's T-squared)
- SNR (Signal-to-Noise Ratio)
- Dis (Distance-based)

### Dimensionality Reduction
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- PCoA (Principal Coordinate Analysis)

### Machine Learning Integration
- Scikit-learn format conversion
- PyTorch Dataset creation
- TensorFlow Dataset creation
- Pre-defined model architectures (CNN, MLP)

### Visualization
- Static plots (Matplotlib)
- Interactive plots (Plotly)
- Spectral plots, reduction plots, QC plots

---

## 📖 Example: Complete Workflow

```python
from pyramex import Ramanome, load_spectra
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
data = load_spectra('data/spectra/')

# 2. Preprocess
data = data.smooth() \
           .remove_baseline() \
           .normalize()

# 3. Quality control
qc = data.quality_control(method='icod')
data = data[qc.good_samples]

# 4. Dimensionality reduction
data.reduce(method='pca', n_components=50)

# 5. Train ML model
X_train, X_test, y_train, y_test = data.to_sklearn_format()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/openclaw/pyramex.git
cd pyramex

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
black pyramex/
flake8 pyramex/
mypy pyramex/
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** PyRamEx is derived from [RamEx](https://github.com/qibebt-bioinfo/RamEx) (R package), which is licensed under GPL. The original RamEx license and attribution are preserved in the [NOTICE](NOTICE) file.

---

## 🙏 Acknowledgments

- Original [RamEx](https://github.com/qibebt-bioinfo/RamEx) team
- RamEx Paper: https://doi.org/10.1101/2025.03.10.642505
- Zhang Y., Jing G., et al. for the excellent work on RamEx

---

## 📞 Contact

- **Project Homepage**: https://github.com/openclaw/pyramex
- **Issues**: https://github.com/openclaw/pyramex/issues
- **Discussions**: https://github.com/openclaw/pyramex/discussions

---

## 📈 Roadmap

### v0.1.0-alpha (Current)
- ✅ Core functionality
- ✅ Basic preprocessing
- ✅ Quality control
- ✅ ML/DL integration

### v0.2.0-beta (Planned: March 2026)
- [ ] Complete unit tests
- [ ] Example datasets
- [ ] Streamlit web app
- [ ] GPU acceleration

### v0.3.0-rc (Planned: April 2026)
- [ ] Marker analysis
- [ ] IRCA analysis
- [ ] Phenotype analysis
- [ ] Spectral decomposition

### v1.0.0-stable (Planned: June 2026)
- [ ] Complete feature set
- [ ] Pre-trained models
- [ ] Plugin system
- [ ] Academic paper

---

*Developer: 小龙虾1号 🦞*  
*Status: 🟢 Active Development*

**Made with ❤️ for the Raman spectroscopy community**
