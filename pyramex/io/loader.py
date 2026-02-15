"""
Data loading utilities for PyRamEx
Supports multiple Raman spectroscopy file formats
"""

from typing import Union, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob


def load_spectra(
    path: Union[str, Path],
    format: str = 'auto',
    **kwargs
) -> 'Ramanome':
    """
    Load Raman spectroscopic data from file(s).
    
    Supported formats:
    - Type 1: Two-column text (wavenumber, intensity)
    - Type 2: Mapping matrix (first column = wavenumbers, rest = spectra)
    - Type 3: Coordinate scan (metadata columns + wavenumber + intensity)
    
    Args:
        path: File or directory path
        format: File format ('auto', 'two_col', 'matrix', 'coords')
        **kwargs: Additional parameters
        
    Returns:
        Ramanome object
        
    Examples:
        >>> data = load_spectra('data.txt')
        >>> data = load_spectra('spectra_dir/', format='auto')
    """
    path = Path(path)
    
    if path.is_file():
        # Single file
        return load_single_file(path, format=format, **kwargs)
    elif path.is_dir():
        # Directory with multiple files
        return load_directory(path, format=format, **kwargs)
    else:
        raise ValueError(f"Path not found: {path}")


def load_single_file(
    file_path: Path,
    format: str = 'auto'
) -> 'Ramanome':
    """Load a single spectrum file."""
    from pyramex.core.ramanome import Ramanome
    
    # Read file
    data = pd.read_csv(file_path, sep=None, engine='python', header=None)
    data = data.values
    
    # Auto-detect format
    if format == 'auto':
        format = detect_format(data)
    
    # Parse based on format
    if format == 'two_col':
        wavenumbers = data[:, 0]
        spectra = data[:, 1:2].T  # Shape: (1, n_wavenumbers)
        metadata = pd.DataFrame({'file': [file_path.name]})
        
    elif format == 'matrix':
        wavenumbers = data[:, 0]
        spectra = data[:, 1:].T  # Shape: (n_samples, n_wavenumbers)
        metadata = pd.DataFrame({
            'sample': [f'S{i}' for i in range(spectra.shape[0])]
        })
        
    elif format == 'coords':
        # Detect number of metadata columns
        n_meta = detect_metadata_columns(data)
        metadata_vals = data[:, :n_meta]
        wavenumbers = data[:, n_meta]
        spectra = data[:, n_meta+1:].T
        
        metadata = pd.DataFrame(
            metadata_vals,
            columns=[f'meta_{i}' for i in range(n_meta)]
        )
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return Ramanome(
        spectra=spectra,
        wavenumbers=wavenumbers,
        metadata=metadata
    )


def load_directory(
    dir_path: Path,
    format: str = 'auto',
    pattern: str = '*.txt'
) -> 'Ramanome':
    """Load all spectrum files from a directory."""
    from pyramex.core.ramanome import Ramanome
    
    files = sorted(glob(str(dir_path / pattern)))
    
    if not files:
        raise ValueError(f"No files found matching {pattern} in {dir_path}")
    
    spectra_list = []
    metadata_list = []
    wavenumbers = None
    
    for file_path in files:
        file_path = Path(file_path)
        data = pd.read_csv(file_path, sep=None, engine='python', header=None)
        data = data.values
        
        if format == 'auto':
            format = detect_format(data)
        
        if format == 'two_col':
            wn = data[:, 0]
            spec = data[:, 1]
            
            if wavenumbers is None:
                wavenumbers = wn
            elif not np.allclose(wavenumbers, wn):
                # Interpolate to common wavenumber grid
                spec = np.interp(wavenumbers, wn, spec)
            
            spectra_list.append(spec)
            metadata_list.append({
                'file': file_path.name,
                'path': str(file_path)
            })
            
        elif format == 'matrix':
            raise NotImplementedError(
                "Matrix format in directory mode not yet implemented"
            )
    
    spectra = np.vstack(spectra_list)
    metadata = pd.DataFrame(metadata_list)
    
    return Ramanome(
        spectra=spectra,
        wavenumbers=wavenumbers,
        metadata=metadata
    )


def detect_format(data: np.ndarray) -> str:
    """
    Auto-detect file format from data shape.
    
    Rules:
    - 2 columns: two_col
    - First column evenly spaced (wavenumbers), rest are spectra: matrix
    - Multiple columns with coordinate-like patterns: coords
    """
    n_rows, n_cols = data.shape
    
    if n_cols == 2:
        return 'two_col'
    
    # Check if first column looks like wavenumbers
    first_col = data[:, 0]
    if is_wavenumber_like(first_col):
        if n_cols > 2:
            return 'matrix'
    
    # Check for coordinate pattern
    if has_coordinate_pattern(data):
        return 'coords'
    
    # Default
    return 'two_col'


def is_wavenumber_like(arr: np.ndarray) -> bool:
    """Check if array looks like wavenumbers."""
    # Should be monotonically increasing
    if not np.all(np.diff(arr) > 0):
        return False
    
    # Should be in typical Raman range (200-4000 cm-1)
    if arr.min() < 100 or arr.max() > 5000:
        return False
    
    # Should be reasonably spaced
    spacing = np.diff(arr)
    if spacing.std() / spacing.mean() > 0.5:
        return False
    
    return True


def has_coordinate_pattern(data: np.ndarray) -> bool:
    """Check if data has coordinate columns."""
    # Heuristic: coordinate columns have small integer values
    n_rows, n_cols = data.shape
    
    # Check first few columns (except last 2 which are wavenumber and intensity)
    n_candidates = min(n_cols - 2, 5)
    
    for i in range(n_candidates):
        col = data[:, i]
        # Coordinates are typically small numbers
        if col.max() > 1000:
            return False
        # Coordinates often have low variance or are integers
        unique_ratio = len(np.unique(col)) / len(col)
        if unique_ratio > 0.9:
            continue
    
    return True


def detect_metadata_columns(data: np.ndarray) -> int:
    """
    Detect number of metadata columns in coordinate format.
    
    Pattern: metadata columns + wavenumber column + intensity column(s)
    """
    n_rows, n_cols = data.shape
    
    # Check from the end: last column is intensity, second-to-last is wavenumber
    # Everything before that is metadata
    
    # Find wavenumber column (should be monotonic)
    for i in range(n_cols - 1, 0, -1):
        if is_wavenumber_like(data[:, i]):
            return i
    
    # Default: assume last 2 columns are wavenumber + intensity
    return n_cols - 2


# Convenience functions
def read_spec(path: Union[str, Path]) -> 'Ramanome':
    """Convenience alias for load_spectra."""
    return load_spectra(path)
