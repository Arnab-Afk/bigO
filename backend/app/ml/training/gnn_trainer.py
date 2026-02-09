"""
GNN Trainer for Cascade Risk Classification

Trains CascadeClassifierGNN model on graph datasets to predict cascade risk levels.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from app.ml.config import ml_config
from app.ml.models.cascade_classifier import CascadeClassifierGNN, GraphDataConverter

logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    Trainer for Cascade Classifier GNN

    Handles training, validation, and evaluation of GNN models for
    cascade risk prediction on financial networks.
    """

    def __init__(
        self,
        model: Optional[CascadeClassifierGNN] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: CascadeClassifierGNN instance (creates new if None)
            device: torch device (auto-detects if None)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for GNN training. "
                "Install with: pip install torch-geometric"
            )

        # Setup device
        if device is None:
            if ml_config.ENABLE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for GNN training")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for GNN training")
        else:
            self.device = device

        # Initialize model
        self.model = model or CascadeClassifierGNN()
        self.model.to(self.device)

        # Training state
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
        }

    def train_cascade_classifier(
        self,
        train_graphs: List[Data],
        val_graphs: List[Data],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Train GNN on graph datasets

        Args:
            train_graphs: List of PyG Data objects with labels
            val_graphs: List of PyG Data objects with labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints

        Returns:
            Dictionary with training metrics
        """
        save_dir = save_dir or ml_config.ML_MODELS_PATH / "cascade_classifier"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create dataloaders
        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=batch_size,
            shuffle=False,
        )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        logger.info(f"Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)

            # Update scheduler
            scheduler.step(val_loss)

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Early stopping check
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self._save_checkpoint(save_dir / "best_model.pt", val_metrics)
                logger.info(f"New best model saved (accuracy: {val_metrics['accuracy']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Save final model
        self._save_checkpoint(save_dir / "final_model.pt", val_metrics)

        # Save training history
        history_path = save_dir / "training_history.npz"
        np.savez(
            history_path,
            train_loss=self.training_history['train_loss'],
            val_loss=self.training_history['val_loss'],
            val_accuracy=self.training_history['val_accuracy'],
            val_f1=self.training_history['val_f1'],
        )

        return {
            'best_val_loss': self.best_loss,
            'best_val_accuracy': self.best_accuracy,
            'best_val_f1': max(self.training_history['val_f1']),
            'epochs_trained': len(self.training_history['train_loss']),
            'save_dir': str(save_dir),
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_graphs = 0

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

        return total_loss / num_graphs

    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_graphs = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)

                predictions = torch.argmax(logits, dim=1)

                total_loss += loss.item() * batch.num_graphs
                num_graphs += batch.num_graphs
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / num_graphs

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'f1': f1,
        }

        return avg_loss, metrics

    def evaluate_classifier(
        self,
        test_graphs: List[Data],
        batch_size: int = 32,
    ) -> Dict:
        """
        Evaluate classifier on test set

        Returns accuracy, F1 score, and confusion matrix

        Args:
            test_graphs: List of test graphs
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        test_loader = DataLoader(
            test_graphs,
            batch_size=batch_size,
            shuffle=False,
        )

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                logits = self.model(batch.x, batch.edge_index, batch.batch)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Classification report
        class_names = ['no_cascade', 'local_cascade', 'systemic_cascade']
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        )

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(f"Classification Report:\n{report}")

        return {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist(),
        }

    def _save_checkpoint(self, path: Path, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'config': {
                'node_feature_dim': self.model.node_feature_dim,
                'hidden_channels': self.model.hidden_channels,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes,
                'dropout': self.model.dropout,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded GNN checkpoint from {path}")
        return checkpoint.get('metrics', {})


def generate_cascade_labels(
    network,
    agent_states: Dict,
    default_threshold: float = 0.2,
) -> int:
    """
    Generate cascade classification label based on network state

    Labels:
    - 0: no_cascade - No significant defaults
    - 1: local_cascade - 1-3 defaults, isolated
    - 2: systemic_cascade - 4+ defaults or system health < 30%

    Args:
        network: NetworkX graph
        agent_states: Dictionary of agent states
        default_threshold: Health threshold for considering default

    Returns:
        Cascade label (0, 1, or 2)
    """
    # Count defaults
    num_defaults = sum(
        1 for state in agent_states.values()
        if state.get('health', 1.0) < default_threshold or not state.get('alive', True)
    )

    # Calculate system health
    if agent_states:
        avg_health = np.mean([
            state.get('health', 1.0)
            for state in agent_states.values()
            if state.get('alive', True)
        ])
    else:
        avg_health = 0.0

    # Classify cascade level
    if num_defaults == 0 and avg_health > 0.7:
        return 0  # no_cascade
    elif num_defaults <= 3 and avg_health > 0.3:
        return 1  # local_cascade
    else:
        return 2  # systemic_cascade
