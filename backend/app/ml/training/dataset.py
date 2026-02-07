"""
PyTorch Dataset for Institution Features
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import torch
from torch.utils.data import Dataset

from app.ml.features.extractor import InstitutionFeatures


class InstitutionDataset(Dataset):
    """
    PyTorch Dataset for institution feature vectors and default labels
    """
    
    def __init__(
        self,
        features: List[InstitutionFeatures],
        labels: List[int],
        normalize: bool = True,
    ):
        """
        Args:
            features: List of InstitutionFeatures objects
            labels: Binary labels (0=no default, 1=default)
            normalize: Whether to normalize features
        """
        assert len(features) == len(labels), "Features and labels must have same length"
        
        self.features = features
        self.labels = labels
        
        # Convert to tensors
        self.X = torch.stack([
            torch.from_numpy(f.to_array()) for f in features
        ])
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Normalization
        if normalize:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (features, label)
        """
        return self.X[idx], self.y[idx]
    
    def get_normalization_params(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get mean and std for normalization"""
        return self.mean, self.std


class SequenceDataset(Dataset):
    """
    Dataset for time series sequences of institution states
    
    Used for LSTM forecasting
    """
    
    def __init__(
        self,
        sequences: List[np.ndarray],
        targets: List[np.ndarray],
        window_size: int = 20,
    ):
        """
        Args:
            sequences: List of historical sequences [window_size, feature_dim]
            targets: List of target sequences [forecast_horizon, feature_dim]
            window_size: Length of input sequence
        """
        assert len(sequences) == len(targets), "Sequences and targets must match"
        
        self.sequences = [torch.from_numpy(s).float() for s in sequences]
        self.targets = [torch.from_numpy(t).float() for t in targets]
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class GraphDataset:
    """
    Dataset for graph neural network training
    
    Stores graph snapshots with node features and cascade labels
    Note: Requires torch_geometric
    """
    
    def __init__(self):
        self.graphs = []
        self.labels = []
    
    def add_graph(self, graph_data, label: int):
        """
        Add a graph snapshot
        
        Args:
            graph_data: PyTorch Geometric Data object
            label: Cascade classification (0=no_cascade, 1=local, 2=systemic)
        """
        self.graphs.append(graph_data)
        self.labels.append(label)
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx]
