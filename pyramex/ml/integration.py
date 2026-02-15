"""
Machine Learning and Deep Learning integration utilities
"""

from typing import Tuple, Union, Optional
import numpy as np
from sklearn.model_selection import train_test_split


def to_sklearn_format(
    ramanome: 'Ramanome',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Convert Ramanome to scikit-learn compatible format.
    
    Args:
        ramanome: Ramanome object
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        (X_train, X_test, y_train, y_test) if labels available
        (X_train, X_test) if no labels
    """
    X = ramanome.spectra
    y = None
    
    # Check for labels in metadata
    if 'label' in ramanome.metadata.columns:
        y = ramanome.metadata['label'].values
    
    # Split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        return X_train, X_test


def to_torch_dataset(
    ramanome: 'Ramanome',
    transform=None
):
    """
    Convert Ramanome to PyTorch Dataset.
    
    Args:
        ramanome: Ramanome object
        transform: Optional transform function
        
    Returns:
        PyTorch Dataset
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError(
            "PyTorch is required. Install with: pip install torch"
        )
    
    class RamanomeDataset(Dataset):
        def __init__(self, ramanome, transform=None):
            self.ramanome = ramanome
            self.transform = transform
            
            # Prepare data
            self.spectra = torch.FloatTensor(ramanome.spectra)
            
            # Add channel dimension for CNN: (n_samples, 1, n_wavenumbers)
            self.spectra = self.spectra.unsqueeze(1)
            
            # Get labels if available
            if 'label' in ramanome.metadata.columns:
                self.labels = ramanome.metadata['label'].values
            else:
                self.labels = None
        
        def __len__(self):
            return len(self.spectra)
        
        def __getitem__(self, idx):
            spectrum = self.spectra[idx]
            
            if self.transform:
                spectrum = self.transform(spectrum)
            
            if self.labels is not None:
                return spectrum, self.labels[idx]
            else:
                return spectrum
    
    return RamanomeDataset(ramanome, transform=transform)


def to_tf_dataset(
    ramanome: 'Ramanome',
    batch_size: int = 32,
    shuffle: bool = True
):
    """
    Convert Ramanome to TensorFlow Dataset.
    
    Args:
        ramanome: Ramanome object
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        )
    
    # Prepare data
    spectra = ramanome.spectra.astype(np.float32)
    
    # Add channel dimension for CNN: (n_samples, 1, n_wavenumbers)
    spectra = spectra[:, np.newaxis, :]
    
    # Get labels if available
    if 'label' in ramanome.metadata.columns:
        labels = ramanome.metadata['label'].values.astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((spectra, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(spectra)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(spectra))
    
    dataset = dataset.batch(batch_size)
    
    return dataset


def create_dataloader(
    ramanome: 'Ramanome',
    framework: str = 'sklearn',
    **kwargs
):
    """
    Create data loader for specified framework.
    
    Args:
        ramanome: Ramanome object
        framework: 'sklearn', 'torch', 'tensorflow'
        **kwargs: Framework-specific parameters
        
    Returns:
        Data loader appropriate for framework
    """
    if framework == 'sklearn':
        return to_sklearn_format(ramanome, **kwargs)
    elif framework == 'torch':
        return to_torch_dataset(ramanome, **kwargs)
    elif framework == 'tensorflow':
        return to_tf_dataset(ramanome, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# Example model architectures
def create_cnn_model(
    input_length: int,
    n_classes: int = None,
    dropout: float = 0.2
):
    """
    Create a simple CNN for spectral classification.
    
    Args:
        input_length: Number of wavenumber points
        n_classes: Number of classes (None for regression)
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required")
    
    class SpectralCNN(nn.Module):
        def __init__(self, input_length, n_classes=None, dropout=0.2):
            super().__init__()
            
            self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool1d(2)
            
            # Calculate size after convolutions and pooling
            conv_output_length = input_length // 4  # After 2 pooling layers
            
            self.fc1 = nn.Linear(64 * conv_output_length, 128)
            self.dropout = nn.Dropout(dropout)
            
            if n_classes:
                self.fc2 = nn.Linear(128, n_classes)
            else:
                self.fc2 = nn.Linear(128, 1)
        
        def forward(self, x):
            # Conv layers
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # FC layers
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x
    
    return SpectralCNN(input_length, n_classes, dropout)


def create_mlp_model(
    input_length: int,
    n_classes: int = None,
    hidden_dims: list = [256, 128, 64]
):
    """
    Create a simple MLP for spectral analysis.
    
    Args:
        input_length: Number of wavenumber points
        n_classes: Number of classes (None for regression)
        hidden_dims: Hidden layer dimensions
        
    Returns:
        scikit-learn MLP model
    """
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    
    if n_classes:
        model = MLPClassifier(
            hidden_layer_sizes=hidden_dims,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2
        )
    
    return model
