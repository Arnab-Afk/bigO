"""
Default Probability Prediction Neural Network

Feedforward neural network that predicts institution default probability
from extracted features.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultPredictorModel(nn.Module):
    """
    Neural network for default probability prediction
    
    Architecture:
    - Input: 20-dimensional feature vector
    - 3 hidden layers with batch normalization and dropout
    - Output: Single probability [0,1] with sigmoid activation
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Tuple[int, int, int] = (128, 64, 32),
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: Sizes of hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Hidden layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dims[2], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Default probabilities of shape (batch_size, 1)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer with sigmoid
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        
        return x
    
    def predict_with_confidence(
        self, x: torch.Tensor, num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            x: Input features
            num_samples: Number of forward passes for uncertainty estimation
        
        Returns:
            Tuple of (mean_prediction, confidence_score)
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Confidence: inverse of standard deviation
        std = predictions.std(dim=0)
        confidence = 1.0 / (1.0 + std)
        
        self.eval()  # Back to eval mode
        
        return mean_pred, confidence
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using gradient-based attribution
        
        Args:
            x: Input features (requires_grad=True)
        
        Returns:
            Feature importance scores
        """
        x.requires_grad = True
        output = self.forward(x)
        
        # Compute gradients
        output.backward(torch.ones_like(output))
        
        # Importance = absolute gradient
        importance = torch.abs(x.grad)
        
        return importance


class EnsembleDefaultPredictor:
    """
    Ensemble of multiple DefaultPredictorModel instances for improved robustness
    """
    
    def __init__(
        self,
        num_models: int = 5,
        input_dim: int = 20,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            num_models: Number of models in ensemble
            input_dim: Input feature dimension
            device: Torch device (CPU/GPU)
        """
        self.num_models = num_models
        self.device = device or torch.device("cpu")
        
        # Create ensemble
        self.models = [
            DefaultPredictorModel(input_dim=input_dim).to(self.device)
            for _ in range(num_models)
        ]
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction
        
        Args:
            x: Input features
        
        Returns:
            Tuple of (mean_prediction, ensemble_confidence)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Mean and variance
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # Confidence: inverse of variance
        confidence = 1.0 / (1.0 + variance)
        
        return mean_pred, confidence
    
    def to(self, device: torch.device):
        """Move all models to device"""
        self.device = device
        for model in self.models:
            model.to(device)
        return self
    
    def train(self):
        """Set all models to train mode"""
        for model in self.models:
            model.train()
    
    def eval(self):
        """Set all models to eval mode"""
        for model in self.models:
            model.eval()
