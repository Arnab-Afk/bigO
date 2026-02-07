"""
LSTM State Forecaster

Bidirectional LSTM for predicting future institution states.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class StateForecastLSTM(nn.Module):
    """
    Bidirectional LSTM for Time Series Forecasting
    
    Predicts future institution states from historical sequences.
    
    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Fully connected output layer
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of state features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            forecast_horizon: Number of future timesteps to predict
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Output projection
        self.fc_out = nn.Linear(lstm_output_dim, forecast_horizon * input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            hidden: Initial hidden state (optional)
        
        Returns:
            Forecasted sequence [batch_size, forecast_horizon, input_dim]
        """
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights: [batch_size, seq_len, 1]
        
        # Context vector: weighted sum of LSTM outputs
        context = (lstm_out * attention_weights).sum(dim=1)
        # context: [batch_size, hidden_dim * num_directions]
        
        # Apply dropout
        context = self.dropout(context)
        
        # Project to output
        output = self.fc_out(context)
        # output: [batch_size, forecast_horizon * input_dim]
        
        # Reshape to sequence
        output = output.view(batch_size, self.forecast_horizon, self.input_dim)
        
        return output
    
    def predict_multi_step(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Multi-step ahead prediction
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            num_steps: Number of steps to predict (uses forecast_horizon if None)
        
        Returns:
            Predictions [batch_size, num_steps, input_dim]
        """
        if num_steps is None:
            num_steps = self.forecast_horizon
        
        predictions = self.forward(x)
        
        # If num_steps > forecast_horizon, truncate
        # If num_steps < forecast_horizon, pad
        if num_steps != self.forecast_horizon:
            if num_steps < self.forecast_horizon:
                predictions = predictions[:, :num_steps, :]
            # Padding not implemented for simplicity
        
        return predictions
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state
        
        Returns:
            Tuple of (h_0, c_0)
        """
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        return h_0, c_0


class EarlyWarningSystem:
    """
    Early warning system using LSTM forecasts
    
    Detects institutions trending toward default
    """
    
    def __init__(
        self,
        model: StateForecastLSTM,
        device: torch.device,
        warning_threshold: float = 0.7,
    ):
        """
        Args:
            model: Trained StateForecastLSTM
            device: torch device
            warning_threshold: Threshold for raising warning
        """
        self.model = model
        self.device = device
        self.warning_threshold = warning_threshold
        self.model.eval()
    
    def detect_risk(
        self,
        historical_sequence: torch.Tensor,
    ) -> dict:
        """
        Detect default risk from historical sequence
        
        Args:
            historical_sequence: [batch_size, seq_len, input_dim]
        
        Returns:
            Dictionary with risk indicators
        """
        with torch.no_grad():
            forecast = self.model.predict_multi_step(historical_sequence)
        
        # Extract capital ratio and default probability trends
        # Assuming: [capital_ratio, liquidity_buffer, credit_exposure,
        #            default_probability, stress_level, risk_appetite, ml_conf]
        capital_ratio_forecast = forecast[:, :, 0]  # [batch_size, forecast_horizon]
        default_prob_forecast = forecast[:, :, 3]
        stress_forecast = forecast[:, :, 4]
        
        # Detect deteriorating trends
        capital_declining = (capital_ratio_forecast[:, -1] < capital_ratio_forecast[:, 0]).float()
        default_prob_rising = (default_prob_forecast[:, -1] > default_prob_forecast[:, 0]).float()
        stress_increasing = (stress_forecast[:, -1] > stress_forecast[:, 0]).float()
        
        # Combined risk score
        risk_score = (
            capital_declining * 0.4 +
            default_prob_rising * 0.4 +
            stress_increasing * 0.2
        )
        
        warnings = (risk_score > self.warning_threshold).cpu().numpy()
        
        return {
            'warnings': warnings,
            'risk_scores': risk_score.cpu().numpy(),
            'forecasted_capital_ratio': capital_ratio_forecast.cpu().numpy(),
            'forecasted_default_prob': default_prob_forecast.cpu().numpy(),
        }
