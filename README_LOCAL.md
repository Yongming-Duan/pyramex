# PyRamEx - Python Ramanome Analysis Toolkit

**A Python reimplementation of RamEx for ML/DL-friendly analysis**

## Overview

PyRamEx is a Python reimplementation of the RamEx R package, specifically optimized for machine learning and deep learning workflows. It provides comprehensive tools for Raman spectroscopic data analysis with a focus on seamless integration with modern ML/DL frameworks.

## Key Features

- ✅ **ML/DL-Native Design**: NumPy/Pandas data structures, Scikit-learn/PyTorch integration
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Modern Python**: Type hints, async support, comprehensive testing
- ✅ **GPU Acceleration**: Optional CUDA support for heavy computations
- ✅ **Interactive Visualization**: Plotly/Matplotlib support
- ✅ **Jupyter Friendly**: Designed for notebook-based exploration

## Installation

```bash
pip install pyramex
```

## Quick Start

```python
from pyramex import Ramanome

# Load data
data = Ramanome.from_file('path/to/spectra.txt')

# Preprocess
data = data.smooth(window_size=5) \
           .remove_baseline() \
           .normalize(method='minmax')

# Quality control
qc_result = data.quality_control(method='icod')
data_clean = data[qc_result.good_samples]

# Dimensionality reduction
reduced = data.reduce(method='umap', n_components=2)

# ML-ready format
X, y = data.to_ml_format()
```

## Architecture

```
pyramex/
├── core/           # Core data structures
├── io/             # Data loading/saving
├── preprocessing/  # Spectral preprocessing
├── qc/             # Quality control
├── ml/             # ML/DL integration
├── visualization/  # Plotting tools
└── utils/          # Utilities
```

## Comparison with RamEx (R)

| Feature | RamEx (R) | PyRamEx (Python) |
|---------|-----------|------------------|
| Language | R | Python |
| ML Integration | Limited | Native (sklearn, PyTorch) |
| GPU Support | OpenCL | CUDA (optional) |
| Data Format | S4 objects | NumPy/Pandas |
| Visualization | ggplot2 | Plotly/Matplotlib |
| Interactivity | Shiny | Jupyter + Streamlit |

## Progress

- [x] Phase 1: Core data structures (40%)
- [x] Phase 2: Preprocessing (0%)
- [ ] Phase 3: ML/DL integration (0%)
- [ ] Phase 4: Advanced features (0%)

## License

GPL (same as original RamEx)

## References

- Original RamEx: https://github.com/qibebt-bioinfo/RamEx
- Paper: https://doi.org/10.1101/2025.03.10.642505

---

*Developer: 小龙虾1号 🦞*
*Date: 2026-02-15*
