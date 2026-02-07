#!/usr/bin/env python3
"""
Train ML Model with RBI Bank Data

Trains the default predictor model using the cleaned RBI bank data.

Usage:
    python scripts/train_rbi_data.py
    python scripts/train_rbi_data.py --epochs 150 --hyperparameter-search
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.data.real_data_loader import RealDataLoader
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.registry.model_manager import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_data(csv_path: str):
    """Inspect the CSV data before training"""
    logger.info("=" * 60)
    logger.info("DATA INSPECTION")
    logger.info("=" * 60)
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"\nColumns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['institution_id', 'defaulted']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info("\nYour CSV must have:")
        logger.info("  - 'institution_id' column (bank identifier)")
        logger.info("  - 'defaulted' column (0 or 1)")
        return False
    
    # Check data quality
    logger.info(f"\nMissing values per column:")
    missing_vals = df.isnull().sum()
    for col, count in missing_vals.items():
        if count > 0:
            logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check label distribution
    if 'defaulted' in df.columns:
        default_count = df['defaulted'].sum()
        default_ratio = default_count / len(df)
        logger.info(f"\nLabel distribution:")
        logger.info(f"  Defaults: {default_count} ({default_ratio*100:.1f}%)")
        logger.info(f"  Non-defaults: {len(df) - default_count} ({(1-default_ratio)*100:.1f}%)")
        
        if default_ratio < 0.05 or default_ratio > 0.95:
            logger.warning(f"⚠️  Highly imbalanced dataset (default ratio: {default_ratio:.1%})")
    
    # Show sample
    logger.info(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    logger.info("\n" + "=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train ML model with RBI bank data'
    )
    
    # Data path
    parser.add_argument(
        '--csv',
        type=str,
        default='app/ml/data/rbi_banks_ml_ready.csv',
        help='Path to cleaned RBI CSV file'
    )
    
    # Column names (adjust if your CSV has different names)
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
        help='Default label column name (0 or 1)'
    )
    parser.add_argument(
        '--timestamp-col',
        type=str,
        default='timestamp',
        help='Timestamp column name (if exists)'
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
    
    # Model registry
    parser.add_argument(
        '--register',
        action='store_true',
        help='Register model with MLflow'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0.0-rbi',
        help='Model version'
    )
    
    # Skip inspection
    parser.add_argument(
        '--skip-inspection',
        action='store_true',
        help='Skip data inspection'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv).exists():
        logger.error(f"File not found: {args.csv}")
        logger.info("\nMake sure you have created 'rbi_banks_ml_ready.csv' in app/ml/data/")
        sys.exit(1)
    
    # Inspect data
    if not args.skip_inspection:
        if not inspect_data(args.csv):
            logger.error("\nData inspection failed. Please fix the issues and try again.")
            sys.exit(1)
        
        # Ask for confirmation
        response = input("\nProceed with training? [Y/n]: ").strip().lower()
        if response and response != 'y':
            logger.info("Training cancelled.")
            sys.exit(0)
    
    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    loader = RealDataLoader()
    
    try:
        features, labels = loader.load_from_csv(
            args.csv,
            institution_id_col=args.id_col,
            label_col=args.label_col,
            timestamp_col=args.timestamp_col if args.timestamp_col in pd.read_csv(args.csv).columns else None,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check column names match: --id-col, --label-col")
        logger.info("  2. Ensure 'defaulted' column has 0/1 values")
        logger.info("  3. Check for missing values in key columns")
        sys.exit(1)
    
    logger.info(f"✓ Loaded {len(features)} samples")
    
    # Create dataset
    logger.info("\n" + "=" * 60)
    logger.info("CREATING DATASET")
    logger.info("=" * 60)
    
    dataset = InstitutionDataset(
        features=features,
        labels=labels,
        normalize=True,
    )
    logger.info(f"✓ Dataset created with {len(dataset)} samples")
    
    # Initialize model
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 60)
    
    model = DefaultPredictorModel(
        input_dim=20,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.3,
    )
    logger.info(f"✓ Model initialized")
    logger.info(f"  Architecture: 20 → 128 → 64 → 32 → 1")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DefaultPredictorTrainer(model=model)
    
    # Hyperparameter search (optional)
    if args.hyperparameter_search:
        logger.info("\n" + "=" * 60)
        logger.info("HYPERPARAMETER SEARCH")
        logger.info("=" * 60)
        
        best_params = trainer.hyperparameter_search(
            dataset=dataset,
            n_trials=args.optuna_trials,
        )
        
        if best_params:
            logger.info(f"✓ Best hyperparameters found:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            
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
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    
    results = trainer.train(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"✓ Best validation AUC: {results['best_val_auc']:.4f}")
    logger.info(f"✓ Best validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"✓ Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"✓ Epochs trained: {results['epochs_trained']}")
    logger.info(f"✓ Model saved to: {results['save_dir']}")
    
    # Performance interpretation
    auc = results['best_val_auc']
    if auc >= 0.90:
        performance = "Excellent 🎉"
    elif auc >= 0.85:
        performance = "Good ✓"
    elif auc >= 0.75:
        performance = "Fair (consider more data/tuning)"
    else:
        performance = "Poor (needs improvement)"
    
    logger.info(f"\nModel Performance: {performance}")
    
    # Register model
    if args.register:
        logger.info("\n" + "=" * 60)
        logger.info("REGISTERING MODEL")
        logger.info("=" * 60)
        
        registry = ModelRegistry()
        registry.register_model(
            model_name="default_predictor_rbi",
            model_path=Path(results['save_dir']) / "best_model.pt",
            metrics={
                'val_auc': results['best_val_auc'],
                'val_f1': results['best_val_f1'],
                'val_loss': results['best_val_loss'],
                'data_source': 'RBI',
            },
            version=args.version,
            stage="Staging",
        )
        logger.info("✓ Model registered with MLflow")
    
    # Next steps
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Test the model:")
    logger.info("   pytest tests/test_ml/ -v")
    logger.info("\n2. Run inference on new data:")
    logger.info("   python examples/ml_example.py")
    logger.info("\n3. Use in simulations:")
    logger.info("   from app.engine.simulation import SimulationEngine")
    logger.info("   engine = SimulationEngine(enable_ml=True)")
    logger.info("\n4. View training history:")
    logger.info(f"   Check: {results['save_dir']}/training_history.npz")


if __name__ == '__main__':
    main()
