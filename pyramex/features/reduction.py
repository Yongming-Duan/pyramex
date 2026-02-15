"""
Dimensionality reduction and feature engineering
"""

from typing import Dict, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def reduce(
    spectra: np.ndarray,
    method: str = 'pca',
    n_components: int = 2,
    **kwargs
) -> Dict:
    """
    Apply dimensionality reduction.
    
    Args:
        spectra: Spectra matrix (n_samples, n_features)
        method: Reduction method ('pca', 'umap', 'tsne', 'pcoa')
        n_components: Number of components
        **kwargs: Method-specific parameters
        
    Returns:
        Dictionary with:
            - transformed: Reduced data (n_samples, n_components)
            - method: Method used
            - explained_variance: Explained variance (if applicable)
            - model: Fitted model (for prediction)
    """
    if method == 'pca':
        return _reduce_pca(spectra, n_components, **kwargs)
    elif method == 'umap':
        return _reduce_umap(spectra, n_components, **kwargs)
    elif method == 'tsne':
        return _reduce_tsne(spectra, n_components, **kwargs)
    elif method == 'pcoa':
        return _reduce_pcoa(spectra, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _reduce_pca(
    spectra: np.ndarray,
    n_components: int,
    **kwargs
) -> Dict:
    """
    Principal Component Analysis.
    
    Args:
        spectra: Spectra matrix
        n_components: Number of components
        **kwargs: Additional PCA parameters
        
    Returns:
        Reduction result dictionary
    """
    # Standardize first
    scaler = StandardScaler()
    spectra_scaled = scaler.fit_transform(spectra)
    
    # Apply PCA
    pca = PCA(n_components=n_components, **kwargs)
    transformed = pca.fit_transform(spectra_scaled)
    
    return {
        'transformed': transformed,
        'method': 'pca',
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'model': pca,
        'scaler': scaler,
        'components': pca.components_,
        'n_components': n_components,
    }


def _reduce_umap(
    spectra: np.ndarray,
    n_components: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    **kwargs
) -> Dict:
    """
    Uniform Manifold Approximation and Projection.
    
    Args:
        spectra: Spectra matrix
        n_components: Number of components (usually 2)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        **kwargs: Additional UMAP parameters
        
    Returns:
        Reduction result dictionary
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "UMAP requires 'umap-learn' package. "
            "Install with: pip install umap-learn"
        )
    
    # Pre-process with PCA if high dimensional
    if spectra.shape[1] > 100:
        pca = PCA(n_components=min(50, spectra.shape[0]-1))
        spectra = pca.fit_transform(spectra)
    
    # Apply UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        **kwargs
    )
    transformed = reducer.fit_transform(spectra)
    
    return {
        'transformed': transformed,
        'method': 'umap',
        'model': reducer,
        'n_components': n_components,
    }


def _reduce_tsne(
    spectra: np.ndarray,
    n_components: int,
    perplexity: float = 30.0,
    **kwargs
) -> Dict:
    """
    t-Distributed Stochastic Neighbor Embedding.
    
    Args:
        spectra: Spectra matrix
        n_components: Number of components (usually 2)
        perplexity: t-SNE perplexity parameter
        **kwargs: Additional t-SNE parameters
        
    Returns:
        Reduction result dictionary
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "t-SNE requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )
    
    # Pre-process with PCA if high dimensional
    if spectra.shape[1] > 50:
        pca = PCA(n_components=min(50, spectra.shape[0]-1))
        spectra = pca.fit_transform(spectra)
    
    # Apply t-SNE
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        **kwargs
    )
    transformed = reducer.fit_transform(spectra)
    
    return {
        'transformed': transformed,
        'method': 'tsne',
        'model': reducer,
        'n_components': n_components,
    }


def _reduce_pcoa(
    spectra: np.ndarray,
    n_components: int,
    **kwargs
) -> Dict:
    """
    Principal Coordinate Analysis (PCoA / MDS).
    
    Args:
        spectra: Spectra matrix
        n_components: Number of components
        **kwargs: Additional MDS parameters
        
    Returns:
        Reduction result dictionary
    """
    try:
        from sklearn.manifold import MDS
    except ImportError:
        raise ImportError(
            "PCoA requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )
    
    # Calculate distance matrix
    from sklearn.metrics.pairwise import euclidean_distances
    dist_matrix = euclidean_distances(spectra)
    
    # Apply MDS
    reducer = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        **kwargs
    )
    transformed = reducer.fit_transform(dist_matrix)
    
    return {
        'transformed': transformed,
        'method': 'pcoa',
        'model': reducer,
        'n_components': n_components,
    }


# Feature engineering
def extract_band_intensity(
    ramanome: 'Ramanome',
    bands: list
) -> np.ndarray:
    """
    Extract intensity from specific wavenumber bands.
    
    Args:
        ramanome: Ramanome object
        bands: List of wavenumber ranges [(min1, max1), (min2, max2), ...]
        
    Returns:
        Intensity matrix (n_samples, n_bands)
    """
    spectra = ramanome.spectra
    wavenumbers = ramanome.wavenumbers
    
    intensities = []
    for band in bands:
        if isinstance(band, (tuple, list)) and len(band) == 2:
            # Wavenumber range
            min_wn, max_wn = band
            mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
            intensity = spectra[:, mask].mean(axis=1)
        elif isinstance(band, (int, float)):
            # Single wavenumber
            idx = np.argmin(np.abs(wavenumbers - band))
            intensity = spectra[:, idx]
        else:
            raise ValueError(f"Invalid band specification: {band}")
        
        intensities.append(intensity)
    
    return np.column_stack(intensities)


def calculate_cdr(
    ramanome: 'Ramanome',
    band1: tuple,
    band2: tuple
) -> np.ndarray:
    """
    Calculate CDR (Cytoplasmic Ratio).
    
    Args:
        ramanome: Ramanome object
        band1: First band (min, max)
        band2: Second band (min, max)
        
    Returns:
        CDR values (n_samples,)
    """
    intensities = extract_band_intensity(ramanome, [band1, band2])
    
    # CDR = I1 / (I1 + I2)
    cdr = intensities[:, 0] / (intensities[:, 0] + intensities[:, 1] + 1e-10)
    
    return cdr
