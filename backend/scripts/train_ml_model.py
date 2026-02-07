#!/usr/bin/env python3
"""
ML Training Script

Quick script to train ML models for RUDRA platform.

Usage:
    python scripts/train_ml_model.py --model default_predictor --simulations 100
    python scripts/train_ml_model.py --model cascade_classifier --simulations 50
    python scripts/train_ml_model.py --model state_forecaster --simulations 80
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data.synthetic_generator import SyntheticDataGenerator
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.registry.model_manager import ModelRegistry
from app.ml.config import ml_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_default_predictor(args):
    """Train default probability predictor"""
    logger.info("=== Training Default Predictor ===")
    
    # 1. Generate training data
    logger.info(f"Generating synthetic data ({args.simulations} simulations)...")
    generator = SyntheticDataGenerator()
    features_list, labels_list = generator.generate_balanced_dataset(
        target_samples=args.simulations * 50,
        default_ratio=0.3,
    )
    
    logger.info(f"Generated {len(features_list)} samples")
    logger.info(f"Defaults: {sum(labels_list)}, Non-defaults: {len(labels_list) - sum(labels_list)}")
    
    # 2. Create dataset
    logger.info("Creating dataset...")
    dataset = InstitutionDataset(
        features=features_list,
        labels=labels_list,
        normalize=True,
    )
    
    # 3. Initialize model
    logger.info("Initializing model...")
    model = DefaultPredictorModel(
        input_dim=20,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
    )
    
    trainer = DefaultPredictorTrainer(model=model)
    
    # 4. Hyperparameter search (optional)
    if args.hyperparameter_search:
        logger.info("Running hyperparameter search...")
        best_params = trainer.hyperparameter_search(
            dataset=dataset,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )
        
        if best_params:
            logger.info(f"Best hyperparameters: {best_params}")
            # Create new model with best params
            model = DefaultPredictorModel(
                hidden_dims=(
                    best_params.get('hidden_dim1', 128),
                    best_params.get('hidden_dim2', 64),
                    best_params.get('hidden_dim3', 32),
                ),
                dropout_rate=best_params.get('dropout_rate', 0.3),
            )
            trainer = DefaultPredictorTrainer(model=model)
            args.learning_rate = best_params.get('learning_rate', args.learning_rate)
            args.batch_size = best_params.get('batch_size', args.batch_size)
    
    # 5. Train
    logger.info("Starting training...")
    results = trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
    logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Epochs trained: {results['epochs_trained']}")
    logger.info(f"Model saved to: {results['save_dir']}")
    
    # 6. Register model (if MLflow available)
    if args.register:
        logger.info("Registering model with MLflow...")
        registry = ModelRegistry()
        registry.register_model(
            model_name="default_predictor",
            model_path=Path(results['save_dir']) / "best_model.pt",
            metrics={
                'val_auc': results['best_val_auc'],
                'val_f1': results['best_val_f1'],
                'val_loss': results['best_val_loss'],
            },
            version=args.version,
            stage="Staging",
        )
        logger.info("Model registered!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ML models for RUDRA')
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='default_predictor',
        choices=['default_predictor', 'cascade_classifier', 'state_forecaster'],
        help='Model to train'
    )
    
    # Data generation
    parser.add_argument(
        '--simulations',
        type=int,
        default=100,
        help='Number of simulations for synthetic data'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    # Hyperparameter search
    parser.add_argument(
        '--hyperparameter-search',
        action='store_true',
        help='Perform hyperparameter search with Optuna'
    )
    
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=20,
        help='Number of Optuna trials'
    )
    
    parser.add_argument(
        '--optuna-timeout',
        type=int,
        default=3600,
        help='Optuna timeout in seconds'
    )
    
    # Model registry
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register model with MLflow'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0.0',
        help='Model version string'
    )
    
    args = parser.parse_args()
    
    # Train selected model
    if args.model == 'default_predictor':
        train_default_predictor(args)
    elif args.model == 'cascade_classifier':
        logger.error("Cascade classifier training not yet implemented")
        sys.exit(1)
    elif args.model == 'state_forecaster':
        logger.error("State forecaster training not yet implemented")
        sys.exit(1)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
