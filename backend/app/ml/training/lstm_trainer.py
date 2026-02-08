"""
LSTM Trainer for State Forecasting

Trains StateForecastLSTM model on historical simulation data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.ml.models.state_forecaster import StateForecastLSTM

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """
    Dataset for time series sequences

    Converts simulation history into training sequences for LSTM.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Args:
            sequences: Input sequences [num_samples, seq_len, input_dim]
            targets: Target sequences [num_samples, forecast_horizon, input_dim]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMTrainer:
    """
    Trainer for StateForecastLSTM model

    Handles:
    - Training data generation from simulation
    - Model training with early stopping
    - Model evaluation and metrics
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 10,
        seq_length: int = 20,
        dropout: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            input_dim: Dimension of state features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            forecast_horizon: Number of timesteps to predict
            seq_length: Length of input sequence
            dropout: Dropout probability
            device: torch device (auto-detect if None)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.dropout = dropout

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model
        self.model = StateForecastLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            dropout=dropout,
        ).to(self.device)

        logger.info(f"Initialized LSTMTrainer on {self.device}")

    def generate_sequences(
        self,
        simulation_history: List[Dict],
        agent_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training sequences from simulation history

        Args:
            simulation_history: List of simulation snapshots
            agent_ids: List of agent IDs to include (all if None)

        Returns:
            Tuple of (input_sequences, target_sequences)
            - input_sequences: [num_samples, seq_len, input_dim]
            - target_sequences: [num_samples, forecast_horizon, input_dim]
        """
        logger.info("Generating training sequences from simulation history")

        # Extract agent states over time
        # State vector: [capital_ratio, liquidity_buffer, credit_exposure,
        #                default_probability, stress_level, risk_appetite, ml_conf]

        agent_histories = {}

        for snapshot in simulation_history:
            agent_states = snapshot.get('agent_states', {})

            for agent_id, state in agent_states.items():
                # Filter by agent type (only banks)
                if state.get('type') != 'bank':
                    continue

                # Filter by agent_ids if specified
                if agent_ids is not None and agent_id not in agent_ids:
                    continue

                if agent_id not in agent_histories:
                    agent_histories[agent_id] = []

                # Extract state features
                # Normalize capital ratio and liquidity
                capital_ratio = state.get('crar', 0.0) / 100.0  # Convert % to ratio
                liquidity_buffer = state.get('liquidity', 0.0) / max(state.get('capital', 1.0), 1.0)

                # Credit exposure (normalized by capital)
                credit_exposure = 0.0  # Placeholder - would need network data

                # Default probability (from ML model if available)
                default_probability = 0.0  # Placeholder

                # Stress level
                stress_level = state.get('npa_ratio', 0.0) / 100.0

                # Risk appetite
                risk_appetite = state.get('risk_appetite', 0.5)

                # ML confidence
                ml_conf = 0.8  # Placeholder

                state_vector = [
                    capital_ratio,
                    liquidity_buffer,
                    credit_exposure,
                    default_probability,
                    stress_level,
                    risk_appetite,
                    ml_conf,
                ]

                agent_histories[agent_id].append(state_vector)

        # Generate sequences
        sequences = []
        targets = []

        for agent_id, history in agent_histories.items():
            history_array = np.array(history)

            # Need at least seq_length + forecast_horizon timesteps
            if len(history_array) < self.seq_length + self.forecast_horizon:
                continue

            # Sliding window
            for i in range(len(history_array) - self.seq_length - self.forecast_horizon + 1):
                input_seq = history_array[i:i + self.seq_length]
                target_seq = history_array[i + self.seq_length:i + self.seq_length + self.forecast_horizon]

                sequences.append(input_seq)
                targets.append(target_seq)

        if not sequences:
            logger.warning("No valid sequences generated from simulation history")
            return np.array([]), np.array([])

        sequences = np.array(sequences)
        targets = np.array(targets)

        logger.info(f"Generated {len(sequences)} sequences from {len(agent_histories)} agents")

        return sequences, targets

    def train_forecaster(
        self,
        train_sequences: np.ndarray,
        train_targets: np.ndarray,
        val_sequences: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train LSTM forecaster

        Args:
            train_sequences: Training input sequences
            train_targets: Training target sequences
            val_sequences: Validation input sequences (optional)
            val_targets: Validation target sequences (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history with losses
        """
        logger.info(f"Training LSTM forecaster for {epochs} epochs")

        # Create datasets
        train_dataset = SequenceDataset(train_sequences, train_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_sequences is not None and val_targets is not None:
            val_dataset = SequenceDataset(val_sequences, val_targets)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        predictions = self.model(batch_x)
                        loss = criterion(predictions, batch_y)

                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {avg_val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}"
                    )

        logger.info("Training completed")

        return history

    def evaluate_forecast_accuracy(
        self,
        test_sequences: np.ndarray,
        test_targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy on test set

        Args:
            test_sequences: Test input sequences
            test_targets: Test target sequences

        Returns:
            Dictionary with evaluation metrics (RMSE, MAE, etc.)
        """
        logger.info("Evaluating forecast accuracy")

        self.model.eval()

        test_dataset = SequenceDataset(test_sequences, test_targets)
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
        )

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Per-feature metrics
        feature_names = [
            'capital_ratio', 'liquidity_buffer', 'credit_exposure',
            'default_probability', 'stress_level', 'risk_appetite', 'ml_conf'
        ]

        per_feature_rmse = {}
        for i, feature in enumerate(feature_names):
            feature_mse = np.mean((all_predictions[:, :, i] - all_targets[:, :, i]) ** 2)
            per_feature_rmse[f'rmse_{feature}'] = np.sqrt(feature_mse)

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse),
            **per_feature_rmse,
        }

        logger.info(f"Evaluation metrics - RMSE: {rmse:.6f}, MAE: {mae:.6f}")

        return metrics

    def save_model(self, path: Path) -> None:
        """Save model to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'forecast_horizon': self.forecast_horizon,
                'seq_length': self.seq_length,
                'dropout': self.dropout,
            }
        }, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        logger.info(f"Model loaded from {path}")
