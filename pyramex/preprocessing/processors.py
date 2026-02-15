"""
Spectral preprocessing functions
"""

from typing import Tuple
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.optimize import curve_fit


def smooth(
    spectra: np.ndarray,
    window_size: int = 5,
    polyorder: int = 2,
    axis: int = 1
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing.
    
    Args:
        spectra: Spectra matrix (n_samples, n_wavenumbers)
        window_size: Size of smoothing window (must be odd)
        polyorder: Order of polynomial
        axis: Axis along which to smooth
        
    Returns:
        Smoothed spectra
    """
    if window_size % 2 == 0:
        window_size += 1
    
    if window_size < polyorder + 2:
        raise ValueError(
            f"window_size ({window_size}) must be >= polyorder + 2 ({polyorder + 2})"
        )
    
    smoothed = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        smoothed[i] = savgol_filter(
            spectra[i],
            window_length=window_size,
            polyorder=polyorder
        )
    
    return smoothed


def remove_baseline(
    spectra: np.ndarray,
    method: str = 'polyfit',
    degree: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Remove baseline from spectra.
    
    Args:
        spectra: Spectra matrix
        method: Baseline removal method ('polyfit', 'als', 'airpls')
        degree: Polynomial degree for polyfit method
        **kwargs: Method-specific parameters
        
    Returns:
        Baseline-corrected spectra
    """
    if method == 'polyfit':
        return _baseline_polyfit(spectra, degree=degree)
    elif method == 'als':
        return _baseline_als(spectra, **kwargs)
    elif method == 'airpls':
        return _baseline_airpls(spectra, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _baseline_polyfit(spectra: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial fitting baseline removal."""
    corrected = np.zeros_like(spectra)
    n_wavenumbers = spectra.shape[1]
    x = np.arange(n_wavenumbers)
    
    for i in range(spectra.shape[0]):
        # Fit polynomial
        coeffs = np.polyfit(x, spectra[i], degree)
        baseline = np.polyval(coeffs, x)
        corrected[i] = spectra[i] - baseline
    
    return corrected


def _baseline_als(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline removal.
    
    Reference:
        Eilers, P. H., & Boelens, H. F. (2005).
        Baseline correction with asymmetric least squares smoothing.
        Leiden University Medical Centre Report.
    """
    corrected = np.zeros_like(spectra)
    n_wavenumbers = spectra.shape[1]
    
    # Design matrix (second derivative)
    D = diags([1, -2, 1], [0, -1, -2], shape=(n_wavenumbers, n_wavenumbers))
    
    for i in range(spectra.shape[0]):
        y = spectra[i]
        
        # Initialize
        w = np.ones(n_wavenumbers)
        
        for _ in range(n_iter):
            # Weighted least squares
            W = diags(w, 0)
            Z = W + lam * D.dot(D.T)
            z = np.linalg.solve(Z.toarray(), w * y)
            
            # Update weights
            w = p * (y > z) + (1 - p) * (y < z)
        
        corrected[i] = y - z
    
    return corrected


def _baseline_airpls(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10
) -> np.ndarray:
    """
    Adaptive Iteratively Reweighted Penalized Least Squares.
    
    Reference:
        Zhang, Z. M., et al. (2010).
        Baseline correction using adaptive iteratively reweighted
        penalized least squares.
        Analyst, 135(5), 1138-1146.
    """
    # Similar to ALS but with adaptive iteration
    # Simplified implementation
    return _baseline_als(spectra, lam=lam, p=p, n_iter=n_iter)


def normalize(
    spectra: np.ndarray,
    method: str = 'minmax',
    **kwargs
) -> np.ndarray:
    """
    Normalize spectra.
    
    Args:
        spectra: Spectra matrix
        method: Normalization method ('minmax', 'zscore', 'area', 'max', 'vecnorm')
        **kwargs: Method-specific parameters
        
    Returns:
        Normalized spectra
    """
    if method == 'minmax':
        return _normalize_minmax(spectra, **kwargs)
    elif method == 'zscore':
        return _normalize_zscore(spectra)
    elif method == 'area':
        return _normalize_area(spectra)
    elif method == 'max':
        return _normalize_max(spectra)
    elif method == 'vecnorm':
        return _normalize_vecnorm(spectra)
    else:
        raise ValueError(f"Unknown method: {method}")


def _normalize_minmax(
    spectra: np.ndarray,
    feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """Min-max normalization."""
    min_val = spectra.min(axis=1, keepdims=True)
    max_val = spectra.max(axis=1, keepdims=True)
    
    normalized = (spectra - min_val) / (max_val - min_val + 1e-10)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return normalized


def _normalize_zscore(spectra: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    
    return (spectra - mean) / (std + 1e-10)


def _normalize_area(spectra: np.ndarray) -> np.ndarray:
    """Area under curve normalization."""
    area = np.trapz(np.abs(spectra), axis=1, keepdims=True)
    return spectra / (area + 1e-10)


def _normalize_max(spectra: np.ndarray) -> np.ndarray:
    """Max normalization."""
    max_val = spectra.max(axis=1, keepdims=True)
    return spectra / (max_val + 1e-10)


def _normalize_vecnorm(spectra: np.ndarray) -> np.ndarray:
    """Vector (L2) normalization."""
    norm = np.linalg.norm(spectra, axis=1, keepdims=True)
    return spectra / (norm + 1e-10)


def cutoff(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    wavenumber_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cutoff spectra to wavenumber range.
    
    Args:
        spectra: Spectra matrix
        wavenumbers: Wavenumber array
        wavenumber_range: (min, max) wavenumber range
        
    Returns:
        (spectra_cutoff, wavenumbers_cutoff)
    """
    min_wn, max_wn = wavenumber_range
    
    mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
    
    return spectra[:, mask], wavenumbers[mask]


def derivative(
    spectra: np.ndarray,
    order: int = 1,
    window_size: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Calculate spectral derivative.
    
    Args:
        spectra: Spectra matrix
        order: Derivative order (1 or 2)
        window_size: Savitzky-Golay window size
        polyorder: Polynomial order
        
    Returns:
        Derivative spectra
    """
    deriv = np.zeros_like(spectra)
    
    for i in range(spectra.shape[0]):
        deriv[i] = savgol_filter(
            spectra[i],
            window_length=window_size,
            polyorder=polyorder,
            deriv=order
        )
    
    return deriv
