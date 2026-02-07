"""
Graph Neural Network for Cascade Risk Classification

Uses GCN layers to classify network structures by cascade risk.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Define dummy classes to prevent import errors
    class GCNConv:
        pass
    class Data:
        pass


class CascadeClassifierGNN(nn.Module):
    """
    Graph Convolutional Network for Cascade Risk Classification
    
    Architecture:
    - 3 GCN layers with ReLU activations
    - Global mean pooling
    - 3-class classification: no_cascade (0), local_cascade (1), systemic_cascade (2)
    
    Requires: torch-geometric
    """
    
    def __init__(
        self,
        node_feature_dim: int = 10,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            node_feature_dim: Dimension of node features
            hidden_channels: Hidden layer dimension
            num_layers: Number of GCN layers
            num_classes: Number of cascade classes
            dropout: Dropout probability
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )
        
        self.node_feature_dim = node_feature_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: node_feature_dim -> hidden_channels
        self.convs.append(GCNConv(node_feature_dim, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last GCN layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes] (None for single graph)
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # No dropout after last conv
                x = self.dropout_layer(x)
        
        # Global pooling
        if batch is None:
            # Single graph: mean over all nodes
            x = x.mean(dim=0, keepdim=True)
        else:
            # Batch of graphs: mean per graph
            x = global_mean_pool(x, batch)
        
        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cascade class with probabilities
        
        Returns:
            Tuple of (class_predictions, probabilities)
        """
        logits = self.forward(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        return predictions, probs


class GraphDataConverter:
    """
    Convert NetworkX graphs to PyTorch Geometric Data objects
    """
    
    @staticmethod
    def networkx_to_pyg(
        graph,
        node_features: dict,
        label: int,
    ):
        """
        Convert NetworkX graph to PyG Data object
        
        Args:
            graph: NetworkX DiGraph
            node_features: Dict mapping node_id -> feature array
            label: Cascade classification label
        
        Returns:
            torch_geometric.data.Data object
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric is required")
        
        import torch
        from torch_geometric.data import Data
        
        # Create node index mapping
        node_list = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Extract features
        x_list = []
        for node in node_list:
            if node in node_features:
                x_list.append(node_features[node])
            else:
                # Use zero features for missing nodes
                feature_dim = len(next(iter(node_features.values())))
                x_list.append([0.0] * feature_dim)
        
        x = torch.tensor(x_list, dtype=torch.float)
        
        # Extract edges
        edge_index_list = []
        for source, target in graph.edges():
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            edge_index_list.append([source_idx, target_idx])
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
        )
        
        return data
