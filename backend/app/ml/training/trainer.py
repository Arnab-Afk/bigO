"""
Training Pipeline for Default Predictor Model

Includes:
- Training loop with validation
- Early stopping
- Hyperparameter tuning with Optuna
- Model checkpointing
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from app.ml.config import ml_config
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.training.dataset import InstitutionDataset

logger = logging.getLogger(__name__)


class DefaultPredictorTrainer:
    """
    Trainer for Default Predictor Neural Network
    """
    
    def __init__(
        self,
        model: Optional[DefaultPredictorModel] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: DefaultPredictorModel instance (creates new if None)
            device: torch device (auto-detects if None)
        """
        # Setup device
        if device is None:
            if ml_config.ENABLE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for training")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for training")
        else:
            self.device = device
        
        # Initialize model
        self.model = model or DefaultPredictorModel()
        self.model.to(self.device)
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
        }
    
    def train(
        self,
        dataset: InstitutionDataset,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        validation_split: float = None,
        early_stopping_patience: int = None,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Train the model
        
        Args:
            dataset: InstitutionDataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with training metrics
        """
        # Use config defaults if not specified
        epochs = epochs or ml_config.TRAINING_EPOCHS
        batch_size = batch_size or ml_config.BATCH_SIZE
        learning_rate = learning_rate or ml_config.LEARNING_RATE
        validation_split = validation_split or ml_config.VALIDATION_SPLIT
        early_stopping_patience = early_stopping_patience or ml_config.EARLY_STOPPING_PATIENCE
        save_dir = save_dir or ml_config.ML_MODELS_PATH / "default_predictor"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Split dataset with stratification to ensure both classes in train/val
        # Get labels for stratification
        all_labels = [dataset[i][1].item() for i in range(len(dataset))]
        indices = list(range(len(dataset)))
        
        # Check if we have both classes
        unique_labels = set(all_labels)
        if len(unique_labels) < 2:
            logger.warning(
                f"Dataset has only one class: {unique_labels}. "
                "Cannot train a binary classifier. Please provide data with both classes."
            )
            raise ValueError("Dataset must contain both positive and negative samples")
        
        # Stratified split
        try:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=validation_split,
                stratify=all_labels,
                random_state=42
            )
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        except ValueError as e:
            # Fallback to regular split if stratification fails
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
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
            self.training_history['val_auc'].append(val_metrics['auc'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self._save_checkpoint(save_dir / "best_model.pt", val_metrics)
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
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
            val_auc=self.training_history['val_auc'],
            val_f1=self.training_history['val_f1'],
        )
        
        return {
            'best_val_loss': self.best_loss,
            'best_val_auc': max(self.training_history['val_auc']),
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
        
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
        
        return total_loss / len(loader.dataset)
    
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
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Check if we have both classes in validation set
        unique_labels = np.unique(all_labels)
        
        if len(unique_labels) < 2:
            # Only one class present - cannot compute AUC
            logger.warning(
                f"Only one class present in validation set: {unique_labels}. "
                "Metrics may not be reliable. Consider more data or different split."
            )
            auc = 0.5  # Random baseline
            best_f1 = 0.0
        else:
            # Compute AUC
            try:
                auc = roc_auc_score(all_labels, all_predictions)
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
                auc = 0.5
            
            # F1 score (use optimal threshold)
            thresholds = np.linspace(0.1, 0.9, 20)
            f1_scores = []
            for t in thresholds:
                try:
                    f1 = f1_score(all_labels, (all_predictions > t).astype(int), zero_division=0)
                    f1_scores.append(f1)
                except:
                    f1_scores.append(0.0)
            best_f1 = max(f1_scores) if f1_scores else 0.0
        
        metrics = {
            'auc': auc,
            'f1': best_f1,
        }
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, path: Path, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout_rate': self.model.dropout_rate,
            },
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metrics', {})
    
    def hyperparameter_search(
        self,
        dataset: InstitutionDataset,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict:
        """
        Hyperparameter optimization using Optuna
        
        Args:
            dataset: Training dataset
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping hyperparameter search")
            return {}
        
        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            hidden_dim1 = trial.suggest_int("hidden_dim1", 64, 256, step=32)
            hidden_dim2 = trial.suggest_int("hidden_dim2", 32, 128, step=16)
            hidden_dim3 = trial.suggest_int("hidden_dim3", 16, 64, step=8)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            
            # Create model with suggested hyperparameters
            model = DefaultPredictorModel(
                hidden_dims=(hidden_dim1, hidden_dim2, hidden_dim3),
                dropout_rate=dropout_rate,
            )
            
            trainer = DefaultPredictorTrainer(model=model, device=self.device)
            
            # Train with early stopping
            result = trainer.train(
                dataset=dataset,
                epochs=50,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=5,
            )
            
            return result['best_val_loss']
        
        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value:.4f}")
        
        return study.best_params
