"""
Core data structures for PyRamEx
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Ramanome:
    """
    Core data structure for Raman spectroscopic data.
    
    This is a Python-native alternative to RamEx's S4 objects,
    designed for seamless ML/DL integration.
    
    Attributes:
        spectra: Spectral data matrix (n_samples x n_wavenumbers)
        wavenumbers: Wavenumber axis
        metadata: Sample metadata (DataFrame)
        processed: List of preprocessing steps applied
        quality: Quality control results
        features: Feature engineering results
        reductions: Dimensionality reduction results
        
    Example:
        >>> from pyramex import Ramanome
        >>> data = Ramanome(spectra, wavenumbers, metadata)
        >>> processed = data.smooth().normalize()
        >>> X, y = processed.to_ml_format()
    """
    
    # Core data
    spectra: np.ndarray  # Shape: (n_samples, n_wavenumbers)
    wavenumbers: np.ndarray  # Shape: (n_wavenumbers,)
    metadata: pd.DataFrame  # Shape: (n_samples, n_metadata)
    
    # Processing history
    processed: List[str] = field(default_factory=list)
    quality: Dict = field(default_factory=dict)
    features: Dict = field(default_factory=dict)
    reductions: Dict = field(default_factory=dict)
    
    # Internal state
    _original_spectra: Optional[np.ndarray] = field(init=False, default=None)
    _preprocessing_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize data."""
        # Store original data
        self._original_spectra = self.spectra.copy()
        
        # Validate shapes
        n_samples, n_wavenumbers = self.spectra.shape
        if self.wavenumbers.shape[0] != n_wavenumbers:
            raise ValueError(
                f"Wavenumber mismatch: spectra has {n_wavenumbers} points, "
                f"but wavenumbers has {self.wavenumbers.shape[0]} points"
            )
        
        if len(self.metadata) != n_samples:
            raise ValueError(
                f"Metadata mismatch: spectra has {n_samples} samples, "
                f"but metadata has {len(self.metadata)} rows"
            )
        
        # Ensure wavenumbers are sorted
        if not np.all(np.diff(self.wavenumbers) > 0):
            sort_idx = np.argsort(self.wavenumbers)
            self.wavenumbers = self.wavenumbers[sort_idx]
            self.spectra = self.spectra[:, sort_idx]
            self._original_spectra = self._original_spectra[:, sort_idx]
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.spectra.shape[0]
    
    @property
    def n_wavenumbers(self) -> int:
        """Number of wavenumber points."""
        return self.spectra.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of spectra matrix."""
        return self.spectra.shape
    
    def copy(self) -> 'Ramanome':
        """Create a deep copy."""
        return Ramanome(
            spectra=self.spectra.copy(),
            wavenumbers=self.wavenumbers.copy(),
            metadata=self.metadata.copy(),
            processed=self.processed.copy(),
            quality=self.quality.copy(),
            features=self.features.copy(),
            reductions=self.reductions.copy(),
        )
    
    def reset(self) -> 'Ramanome':
        """Reset to original data."""
        self.spectra = self._original_spectra.copy()
        self.processed = []
        self.quality = {}
        self.features = {}
        return self
    
    # Method chaining for preprocessing
    def smooth(self, window_size: int = 5, polyorder: int = 2) -> 'Ramanome':
        """Apply Savitzky-Golay smoothing."""
        # Implementation will be in preprocessing module
        from pyramex.preprocessing import smooth
        self.spectra = smooth(self.spectra, window_size, polyorder)
        self.processed.append(f"smooth(w={window_size}, p={polyorder})")
        return self
    
    def remove_baseline(self, method: str = 'polyfit', **kwargs) -> 'Ramanome':
        """Remove baseline."""
        from pyramex.preprocessing import remove_baseline
        self.spectra = remove_baseline(self.spectra, method, **kwargs)
        self.processed.append(f"baseline(method={method})")
        return self
    
    def normalize(self, method: str = 'minmax', **kwargs) -> 'Ramanome':
        """Normalize spectra."""
        from pyramex.preprocessing import normalize
        self.spectra = normalize(self.spectra, method, **kwargs)
        self.processed.append(f"normalize(method={method})")
        return self
    
    def cutoff(self, wavenumber_range: Tuple[float, float]) -> 'Ramanome':
        """Cutoff spectra to wavenumber range."""
        from pyramex.preprocessing import cutoff
        self.spectra, self.wavenumbers = cutoff(
            self.spectra, self.wavenumbers, wavenumber_range
        )
        self.processed.append(f"cutoff({wavenumber_range})")
        return self
    
    # Quality control
    def quality_control(
        self, 
        method: str = 'icod',
        **kwargs
    ) -> 'QualityResult':
        """
        Apply quality control.
        
        Args:
            method: QC method ('icod', 'mcd', 't2', 'snr')
            **kwargs: Method-specific parameters
            
        Returns:
            QualityResult object
        """
        from pyramex.qc import quality_control
        result = quality_control(self, method=method, **kwargs)
        self.quality[method] = result
        return result
    
    # Dimensionality reduction
    def reduce(
        self,
        method: str = 'pca',
        n_components: int = 2,
        **kwargs
    ) -> 'Ramanome':
        """
        Apply dimensionality reduction.
        
        Args:
            method: Reduction method ('pca', 'umap', 'tsne', 'pcoa')
            n_components: Number of components
            **kwargs: Method-specific parameters
            
        Returns:
            self (with reductions populated)
        """
        from pyramex.features import reduce
        result = reduce(
            self.spectra,
            method=method,
            n_components=n_components,
            **kwargs
        )
        self.reductions[method] = result
        return self
    
    # ML/DL integration
    def to_ml_format(
        self,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """
        Convert to ML-ready format.
        
        Args:
            return_metadata: If True, return (X, metadata)
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            metadata: (optional) Sample metadata
        """
        X = self.spectra.copy()
        if return_metadata:
            return X, self.metadata
        return X
    
    def to_tensor(
        self,
        add_channel: bool = True
    ) -> np.ndarray:
        """
        Convert to tensor format for deep learning.
        
        Args:
            add_channel: If True, add channel dimension
            
        Returns:
            Tensor of shape (n_samples, 1, n_wavenumbers) or (n_samples, n_wavenumbers)
        """
        X = self.spectra.copy()
        if add_channel:
            X = X[:, np.newaxis, :]
        return X
    
    # Visualization
    def plot(self, samples: Optional[List[int]] = None):
        """Plot spectra."""
        from pyramex.visualization import plot_spectra
        return plot_spectra(self, samples=samples)
    
    def plot_reduction(self, method: str = 'pca', **kwargs):
        """Plot dimensionality reduction."""
        from pyramex.visualization import plot_reduction
        return plot_reduction(self, method=method, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"Ramanome(n_samples={self.n_samples}, "
            f"n_wavenumbers={self.n_wavenumbers}, "
            f"processed={len(self.processed)} steps)"
        )


@dataclass
class QualityResult:
    """
    Quality control results.
    
    Attributes:
        good_samples: Boolean array indicating good samples
        quality_scores: Quality scores for each sample
        method: QC method used
        threshold: Threshold applied
        params: Method parameters
    """
    good_samples: np.ndarray
    quality_scores: np.ndarray
    method: str
    threshold: float
    params: Dict = field(default_factory=dict)
    
    @property
    def n_good(self) -> int:
        """Number of good samples."""
        return self.good_samples.sum()
    
    @property
    def n_bad(self) -> int:
        """Number of bad samples."""
        return (~self.good_samples).sum()
    
    @property
    def good_rate(self) -> float:
        """Ratio of good samples."""
        return self.good_samples.mean()
    
    def __repr__(self) -> str:
        return (
            f"QualityResult(method={self.method}, "
            f"good={self.n_good}/{len(self.good_samples)}, "
            f"rate={self.good_rate:.2%})"
        )
