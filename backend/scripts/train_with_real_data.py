#!/usr/bin/env python3
"""
Train ML Model with Real External Data

Usage examples:

1. From single CSV file:
   python scripts/train_with_real_data.py --csv data/my_data.csv

2. From network files:
   python scripts/train_with_real_data.py --network-data \
       --institutions data/institutions.csv \
       --exposures data/exposures.csv \
       --states data/states.csv

3. Create sample data and train:
   python scripts/train_with_real_data.py --create-sample --csv sample_data.csv
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data.real_data_loader import RealDataLoader, create_sample_csv
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.registry.model_manager import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train ML model with real external data'
    )
    
    # Data source options
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        '--csv',
        type=str,
        help='Path to single CSV file with all data'
    )
    data_source.add_argument(
        '--network-data',
        action='store_true',
        help='Load from multiple network files'
    )
    data_source.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample CSV data for testing'
    )
    
    # Network data files
    parser.add_argument(
        '--institutions',
        type=str,
        help='Path to institutions CSV (for network-data)'
    )
    parser.add_argument(
        '--exposures',
        type=str,
        help='Path to exposures CSV (for network-data)'
    )
    parser.add_argument(
        '--states',
        type=str,
        help='Path to states CSV (for network-data)'
    )
    
    # CSV column names
    parser.add_argument(
        '--id-col',
        type=str,
        default='institution_id',
        help='Institution ID column name'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='defaulted',
        help='Label column name'
    )
    parser.add_argument(
        '--timestamp-col',
        type=str,
        default='timestamp',
        help='Timestamp column name (optional)'
    )
    
    # Sample data generation
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of samples to generate (for --create-sample)'
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
    
    # Model output
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register model with MLflow'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0.0',
        help='Model version'
    )
    
    args = parser.parse_args()
    
    # Initialize data loader
    loader = RealDataLoader()
    
    # Load data based on source
    if args.create_sample:
        # Create sample CSV
        csv_path = args.csv or 'sample_data.csv'
        logger.info(f"Creating sample CSV with {args.sample_size} samples...")
        create_sample_csv(csv_path, num_samples=args.sample_size)
        
        # Load it
        features, labels = loader.load_from_csv(
            csv_path,
            institution_id_col=args.id_col,
            timestamp_col=args.timestamp_col,
            label_col=args.label_col,
        )
    
    elif args.csv:
        # Load from single CSV
        logger.info(f"Loading data from {args.csv}")
        features, labels = loader.load_from_csv(
            args.csv,
            institution_id_col=args.id_col,
            timestamp_col=args.timestamp_col,
            label_col=args.label_col,
        )
    
    elif args.network_data:
        # Load from network files
        if not all([args.institutions, args.exposures, args.states]):
            parser.error(
                "--network-data requires --institutions, --exposures, and --states"
            )
        
        logger.info("Loading network data from multiple files")
        features, labels = loader.load_from_network_data(
            institutions_file=args.institutions,
            exposures_file=args.exposures,
            states_file=args.states,
        )
    
    # Check data quality
    if len(features) < 100:
        logger.warning(
            f"Only {len(features)} samples loaded. Consider using more data."
        )
    
    default_ratio = sum(labels) / len(labels)
    logger.info(f"Default ratio: {default_ratio:.2%}")
    
    if default_ratio < 0.05 or default_ratio > 0.95:
        logger.warning(
            f"Imbalanced dataset (default ratio: {default_ratio:.2%}). "
            "Consider balancing or adjusting class weights."
        )
    
    # Create dataset
    logger.info("Creating PyTorch dataset...")
    dataset = InstitutionDataset(
        features=features,
        labels=labels,
        normalize=True,
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = DefaultPredictorModel(
        input_dim=20,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
    )
    
    trainer = DefaultPredictorTrainer(model=model)
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Report results
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
    logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Epochs trained: {results['epochs_trained']}")
    logger.info(f"Model saved to: {results['save_dir']}")
    
    # Register model
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
        logger.info("Model registered successfully!")
    
    logger.info("\nNext steps:")
    logger.info("1. Test the model: pytest tests/test_ml/")
    logger.info("2. Run inference: python examples/ml_example.py")
    logger.info("3. Deploy to API: Configure ML serving in production")


if __name__ == '__main__':
    main()
