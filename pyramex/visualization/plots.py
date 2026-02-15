"""
Visualization utilities for PyRamEx
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_spectra(
    ramanome: 'Ramanome',
    samples: Optional[List[int]] = None,
    n_samples: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False
):
    """
    Plot Raman spectra.
    
    Args:
        ramanome: Ramanome object
        samples: Specific sample indices to plot (None for random)
        n_samples: Number of samples to plot (if samples=None)
        figsize: Figure size
        return_fig: If True, return figure
        
    Returns:
        matplotlib Figure (if return_fig=True)
    """
    spectra = ramanome.spectra
    wavenumbers = ramanome.wavenumbers
    
    # Select samples
    if samples is None:
        n_plot = min(n_samples, len(spectra))
        indices = np.random.choice(len(spectra), n_plot, replace=False)
    else:
        indices = samples
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectra
    for idx in indices:
        ax.plot(wavenumbers, spectra[idx], alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Raman Spectra ({len(indices)} samples)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Raman convention
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_reduction(
    ramanome: 'Ramanome',
    method: str = 'pca',
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    return_fig: bool = False
):
    """
    Plot dimensionality reduction results.
    
    Args:
        ramanome: Ramanome object
        method: Reduction method ('pca', 'umap', 'tsne', 'pcoa')
        color_by: Metadata column for coloring
        figsize: Figure size
        return_fig: If True, return figure
        
    Returns:
        matplotlib Figure (if return_fig=True)
    """
    if method not in ramanome.reductions:
        raise ValueError(f"Reduction {method} not found. Run reduce() first.")
    
    reduction = ramanome.reductions[method]
    transformed = reduction['transformed']
    
    # Determine dimensions
    n_components = transformed.shape[1]
    
    # Create figure
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_by and color_by in ramanome.metadata.columns:
            colors = ramanome.metadata[color_by]
            scatter = ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                c=colors,
                alpha=0.6,
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(f'{method.upper()} Plot')
        ax.grid(True, alpha=0.3)
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if color_by and color_by in ramanome.metadata.columns:
            colors = ramanome.metadata[color_by]
            scatter = ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                c=colors,
                alpha=0.6,
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                alpha=0.6
            )
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_zlabel(f'{method.upper()} 3')
        ax.set_title(f'{method.upper()} Plot')
        
    else:
        raise ValueError(f"Cannot plot {n_components} dimensions")
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_quality_control(
    ramanome: 'Ramanome',
    method: str = 'icod',
    figsize: Tuple[int, int] = (12, 5),
    return_fig: bool = False
):
    """
    Plot quality control results.
    
    Args:
        ramanome: Ramanome object
        method: QC method
        figsize: Figure size
        return_fig: If True, return figure
        
    Returns:
        matplotlib Figure (if return_fig=True)
    """
    if method not in ramanome.quality:
        raise ValueError(f"QC {method} not found. Run quality_control() first.")
    
    qc_result = ramanome.quality[method]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Quality scores
    axes[0].bar(
        range(len(qc_result.quality_scores)),
        qc_result.quality_scores,
        color=['green' if g else 'red' for g in qc_result.good_samples]
    )
    axes[0].axhline(y=qc_result.threshold, color='red', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Quality Score')
    axes[0].set_title('Quality Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Good vs Bad samples
    good_bad = ['Good', 'Bad']
    counts = [qc_result.n_good, qc_result.n_bad]
    colors_pie = ['green', 'red']
    
    axes[1].pie(
        counts,
        labels=good_bad,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90
    )
    axes[1].set_title(f'Sample Quality ({qc_result.good_rate:.1%} Good)')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def plot_preprocessing_steps(
    ramanome: 'Ramanome',
    figsize: Tuple[int, int] = (15, 10),
    return_fig: bool = False
):
    """
    Plot preprocessing steps to visualize effects.
    
    Args:
        ramanome: Ramanome object
        figsize: Figure size
        return_fig: If True, return figure
        
    Returns:
        matplotlib Figure (if return_fig=True)
    """
    if not ramanome.processed:
        print("No preprocessing steps applied.")
        return None
    
    n_steps = len(ramanome.processed)
    n_samples_plot = min(5, ramanome.n_samples)
    
    fig, axes = plt.subplots(n_steps, 1, figsize=figsize)
    if n_steps == 1:
        axes = [axes]
    
    # We need to reconstruct each preprocessing step
    # For simplicity, just plot the final processed spectra
    for i, ax in enumerate(axes):
        sample_idx = np.random.choice(ramanome.n_samples, n_samples_plot, replace=False)
        
        for idx in sample_idx:
            ax.plot(
                ramanome.wavenumbers,
                ramanome.spectra[idx],
                alpha=0.7,
                linewidth=1
            )
        
        ax.set_title(f'Step {i+1}: {ramanome.processed[i]}')
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def interactive_plot(
    ramanome: 'Ramanome',
    samples: Optional[List[int]] = None,
    n_samples: int = 10
):
    """
    Create interactive Plotly plot.
    
    Args:
        ramanome: Ramanome object
        samples: Specific sample indices to plot
        n_samples: Number of samples to plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    spectra = ramanome.spectra
    wavenumbers = ramanome.wavenumbers
    
    # Select samples
    if samples is None:
        n_plot = min(n_samples, len(spectra))
        indices = np.random.choice(len(spectra), n_plot, replace=False)
    else:
        indices = samples
    
    # Create figure
    fig = go.Figure()
    
    for idx in indices:
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=spectra[idx],
            mode='lines',
            name=f'Sample {idx}',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Raman Spectra (Interactive)',
        xaxis_title='Wavenumber (cm⁻¹)',
        yaxis_title='Intensity',
        hovermode='x unified'
    )
    
    fig.show()
