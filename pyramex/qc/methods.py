"""
Quality control methods for Raman spectra
Simplified Python implementations of RamEx QC methods
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats
from sklearn.covariance import MinCovDet


def quality_control(
    ramanome: 'Ramanome',
    method: str = 'icod',
    **kwargs
) -> 'QualityResult':
    """
    Apply quality control to Ramanome data.
    
    Args:
        ramanome: Ramanome object
        method: QC method ('icod', 'mcd', 't2', 'snr', 'dis')
        **kwargs: Method-specific parameters
        
    Returns:
        QualityResult object
        
    Methods:
        icod: Inverse covariance-based outlier detection (recommended)
        mcd: Minimum covariance determinant
        t2: Hotelling's T-squared test
        snr: Signal-to-noise ratio
        dis: Distance-based outlier detection
    """
    from pyramex.core.ramanome import QualityResult
    
    if method == 'icod':
        return _qc_icod(ramanome, **kwargs)
    elif method == 'mcd':
        return _qc_mcd(ramanome, **kwargs)
    elif method == 't2':
        return _qc_t2(ramanome, **kwargs)
    elif method == 'snr':
        return _qc_snr(ramanome, **kwargs)
    elif method == 'dis':
        return _qc_dis(ramanome, **kwargs)
    else:
        raise ValueError(f"Unknown QC method: {method}")


def _qc_icod(
    ramanome: 'Ramanome',
    threshold: float = 0.05
) -> 'QualityResult':
    """
    Inverse Covariance-based Outlier Detection.
    
    Simplified implementation based on:
        Distance-based outlier detection using robust statistics.
    
    Args:
        ramanome: Ramanome object
        threshold: Outlier threshold (p-value)
        
    Returns:
        QualityResult
    """
    spectra = ramanome.spectra
    
    # Calculate robust Mahalanobis distance
    try:
        # Use MinCovDet for robust covariance estimation
        mcd = MinCovDet(support_fraction=0.8).fit(spectra)
        robust_cov = mcd.covariance_
        robust_mean = mcd.location_
        
        # Calculate inverse covariance
        inv_cov = np.linalg.inv(robust_cov + np.eye(robust_cov.shape[0]) * 1e-10)
        
        # Mahalanobis distance
        diff = spectra - robust_mean
        mahal = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        
        # Convert to p-values using chi-squared distribution
        p_values = 1 - stats.chi2.cdf(mahal**2, df=spectra.shape[1])
        
        good_samples = p_values > threshold
        quality_scores = p_values
        
        return QualityResult(
            good_samples=good_samples,
            quality_scores=quality_scores,
            method='icod',
            threshold=threshold,
            params={'method': 'inverse_covariance'}
        )
        
    except Exception as e:
        # Fallback to simple distance-based method
        return _qc_dis(ramanome, threshold=threshold)


def _qc_mcd(
    ramanome: 'Ramanome',
    threshold: float = 0.95
) -> 'QualityResult':
    """
    Minimum Covariance Determinant outlier detection.
    
    Args:
        ramanome: Ramanome object
        threshold: Quantile threshold (keep samples below this)
        
    Returns:
        QualityResult
    """
    spectra = ramanome.spectra
    
    try:
        mcd = MinCovDet(support_fraction=0.75).fit(spectra)
        mahal = mcd.mahalanobis(spectra)
        
        # Samples with distance below threshold are good
        quantile_threshold = np.quantile(mahal, threshold)
        good_samples = mahal <= quantile_threshold
        
        # Convert to scores (higher = better)
        max_dist = mahal.max()
        quality_scores = 1 - (mahal / (max_dist + 1e-10))
        
        return QualityResult(
            good_samples=good_samples,
            quality_scores=quality_scores,
            method='mcd',
            threshold=threshold,
            params={'support_fraction': 0.75}
        )
        
    except Exception as e:
        # Fallback
        return _qc_dis(ramanome, threshold=threshold)


def _qc_t2(
    ramanome: 'Ramanome',
    alpha: float = 0.95
) -> 'QualityResult':
    """
    Hotelling's T-squared test for outlier detection.
    
    Args:
        ramanome: Ramanome object
        alpha: Confidence level
        
    Returns:
        QualityResult
    """
    spectra = ramanome.spectra
    n_samples, n_features = spectra.shape
    
    # Calculate mean and covariance
    mean = np.mean(spectra, axis=0)
    cov = np.cov(spectra.T)
    
    # Inverse covariance
    inv_cov = np.linalg.pinv(cov)
    
    # T-squared statistic
    diff = spectra - mean
    t2 = np.sum(diff @ inv_cov * diff, axis=1)
    
    # Critical value
    critical_value = (
        (n_samples - 1) * n_features /
        (n_samples - n_features) *
        stats.f.ppf(alpha, n_features, n_samples - n_features)
    )
    
    good_samples = t2 <= critical_value
    
    # Convert to scores
    quality_scores = 1 - (t2 / (t2.max() + 1e-10))
    
    return QualityResult(
        good_samples=good_samples,
        quality_scores=quality_scores,
        method='t2',
        threshold=critical_value,
        params={'alpha': alpha}
    )


def _qc_snr(
    ramanome: 'Ramanome',
    method: str = 'easy',
    threshold: float = 10.0
) -> 'QualityResult':
    """
    Signal-to-Noise Ratio quality control.
    
    Args:
        ramanome: Ramanome object
        method: SNR calculation method ('easy', 'advanced')
        threshold: Minimum SNR threshold
        
    Returns:
        QualityResult
    """
    spectra = ramanome.spectra
    wavenumbers = ramanome.wavenumbers
    
    if method == 'easy':
        # Simple method: signal max / noise std
        signal = spectra.max(axis=1)
        noise = spectra.std(axis=1)
        snr = signal / (noise + 1e-10)
        
    elif method == 'advanced':
        # Advanced method: peak signal / baseline noise
        # Estimate baseline (minimum rolling window)
        from scipy.ndimage import minimum_filter1d
        
        baseline = minimum_filter1d(spectra, size=50, axis=1)
        noise = spectra - baseline
        
        signal = spectra.max(axis=1)
        snr = signal / (noise.std(axis=1) + 1e-10)
    
    good_samples = snr >= threshold
    
    # Normalize scores to 0-1
    quality_scores = (snr - snr.min()) / (snr.max() - snr.min() + 1e-10)
    
    return QualityResult(
        good_samples=good_samples,
        quality_scores=quality_scores,
        method='snr',
        threshold=threshold,
        params={'calculation': method}
    )


def _qc_dis(
    ramanome: 'Ramanome',
    threshold: float = 0.05
) -> 'QualityResult':
    """
    Distance-based outlier detection (fallback method).
    
    Args:
        ramanome: Ramanome object
        threshold: Outlier threshold
        
    Returns:
        QualityResult
    """
    spectra = ramanome.spectra
    
    # Calculate distance from median spectrum
    median = np.median(spectra, axis=0)
    distances = np.linalg.norm(spectra - median, axis=1)
    
    # Use percentile-based threshold
    quantile_threshold = np.quantile(distances, 1 - threshold)
    good_samples = distances <= quantile_threshold
    
    # Convert to scores (higher = better)
    quality_scores = 1 - (distances / (distances.max() + 1e-10))
    
    return QualityResult(
        good_samples=good_samples,
        quality_scores=quality_scores,
        method='dis',
        threshold=quantile_threshold,
        params={'metric': 'euclidean'}
    )
