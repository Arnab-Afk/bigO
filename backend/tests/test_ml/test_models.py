"""
Tests for ML Models
"""

import pytest
import torch
import numpy as np

from app.ml.models.default_predictor import DefaultPredictorModel, EnsembleDefaultPredictor
from app.ml.models.state_forecaster import StateForecastLSTM, EarlyWarningSystem


class TestDefaultPredictorModel:
    """Test default prediction neural network"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = DefaultPredictorModel(
            input_dim=20,
            hidden_dims=(128, 64, 32),
            dropout_rate=0.3,
        )
        
        assert model.input_dim == 20
        assert model.hidden_dims == (128, 64, 32)
        assert model.dropout_rate == 0.3
    
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        model = DefaultPredictorModel()
        model.eval()
        
        # Create batch of features
        batch_size = 16
        x = torch.randn(batch_size, 20)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_predict_with_confidence(self):
        """Test prediction with uncertainty estimation"""
        model = DefaultPredictorModel()
        
        x = torch.randn(4, 20)
        
        mean_pred, confidence = model.predict_with_confidence(x, num_samples=5)
        
        assert mean_pred.shape == (4, 1)
        assert confidence.shape == (4, 1)
        assert torch.all(mean_pred >= 0) and torch.all(mean_pred <= 1)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_model_trainable(self):
        """Test model can be trained on dummy data"""
        model = DefaultPredictorModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        # Dummy data
        x = torch.randn(32, 20)
        y = torch.randint(0, 2, (32, 1)).float()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0  # Loss should be positive


class TestEnsemblePredictor:
    """Test ensemble model"""
    
    def test_ensemble_initialization(self):
        """Test ensemble can be initialized"""
        ensemble = EnsembleDefaultPredictor(
            num_models=3,
            input_dim=20,
        )
        
        assert len(ensemble.models) == 3
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        ensemble = EnsembleDefaultPredictor(num_models=3)
        ensemble.eval()
        
        x = torch.randn(8, 20)
        
        mean_pred, confidence = ensemble.predict(x)
        
        assert mean_pred.shape == (8, 1)
        assert confidence.shape == (8, 1)


class TestStateForecastLSTM:
    """Test LSTM forecaster"""
    
    def test_lstm_initialization(self):
        """Test LSTM can be initialized"""
        model = StateForecastLSTM(
            input_dim=7,
            hidden_dim=128,
            num_layers=2,
            forecast_horizon=10,
        )
        
        assert model.input_dim == 7
        assert model.hidden_dim == 128
        assert model.forecast_horizon == 10
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass"""
        model = StateForecastLSTM()
        model.eval()
        
        # Input: [batch_size, seq_len, input_dim]
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 7)
        
        output = model(x)
        
        # Output: [batch_size, forecast_horizon, input_dim]
        assert output.shape == (batch_size, 10, 7)
    
    def test_lstm_multi_step_prediction(self):
        """Test multi-step prediction"""
        model = StateForecastLSTM(forecast_horizon=10)
        model.eval()
        
        x = torch.randn(2, 20, 7)
        
        predictions = model.predict_multi_step(x, num_steps=5)
        
        assert predictions.shape == (2, 5, 7)


class TestEarlyWarningSystem:
    """Test early warning system"""
    
    def test_early_warning_initialization(self):
        """Test early warning system can be initialized"""
        model = StateForecastLSTM()
        ews = EarlyWarningSystem(
            model=model,
            device=torch.device('cpu'),
            warning_threshold=0.7,
        )
        
        assert ews.warning_threshold == 0.7
    
    def test_risk_detection(self):
        """Test risk detection"""
        model = StateForecastLSTM()
        ews = EarlyWarningSystem(
            model=model,
            device=torch.device('cpu'),
        )
        
        # Dummy historical sequence
        x = torch.randn(3, 20, 7)
        
        result = ews.detect_risk(x)
        
        assert 'warnings' in result
        assert 'risk_scores' in result
        assert 'forecasted_capital_ratio' in result
        assert len(result['warnings']) == 3
