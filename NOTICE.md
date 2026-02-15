# PyRamEx - NOTICE file

## Attribution

This Python reimplementation (PyRamEx) is derived from RamEx (R package).

### Original RamEx

- **Repository:** https://github.com/qibebt-bioinfo/RamEx
- **Paper:** Zhang Y., Jing G., ..., Xu J., Sun L., 2025. 
  "RamEx: An R package for high-throughput microbial ramanome analyses 
  with accurate quality assessment"
  *bioRxiv*
  DOI: https://doi.org/10.1101/2025.03.10.642505

### RamEx License

RamEx is licensed under GPL (GNU General Public License).

### PyRamEx License

PyRamEx is licensed under MIT License to facilitate broader adoption in the 
Python ML/DL community, while maintaining attribution to the original RamEx project.

### Core Algorithms

The following core algorithms from RamEx have been reimplemented in Python:

- Quality Control: ICOD, MCD, T2, SNR, Dis
- Preprocessing: Savitzky-Golay smoothing, baseline removal (polyfit, ALS)
- Dimensionality Reduction: PCA, UMAP, t-SNE, PCoA
- Feature Engineering: Band intensity extraction, CDR calculation

### Improvements in PyRamEx

- ML/DL-native design (NumPy/Pandas instead of R S4 objects)
- GPU acceleration (CUDA instead of OpenCL)
- Seamless integration with Python ML/DL frameworks
- Modern Python API with method chaining
- Interactive visualization (Plotly)
- Jupyter notebook support

---

*This NOTICE file is maintained to respect the original RamEx project's contributions.*
