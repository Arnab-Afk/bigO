"""
Celery Tasks for ML Training

Background tasks for:
- Model training
- Periodic retraining
- Model evaluation
- Data generation
"""

import logging
from pathlib import Path
from typing import Dict

from celery import shared_task

from app.ml.data.synthetic_generator import SyntheticDataGenerator
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.registry.model_manager import ModelRegistry
from app.ml.config import ml_config

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="ml.train_default_predictor")
def train_default_predictor_task(
    self,
    num_simulations: int = 100,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hyperparameter_search: bool = False,
) -> Dict:
    """
    Train default predictor model
    
    Args:
        num_simulations: Number of simulations for synthetic data
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hyperparameter_search: Whether to perform hyperparameter search
    
    Returns:
        Training results
    """
    try:
        logger.info(
            f"Starting default predictor training: "
            f"{num_simulations} sims, {epochs} epochs"
        )
        
        # Update task state
        self.update_state(state='GENERATING_DATA', meta={'progress': 0})
        
        # 1. Generate synthetic training data
        generator = SyntheticDataGenerator()
        features_list, labels_list = generator.generate_balanced_dataset(
            target_samples=num_simulations * 50,
            default_ratio=0.3,
        )
        
        logger.info(f"Generated {len(features_list)} training samples")
        
        # 2. Create dataset
        self.update_state(state='PREPARING_DATASET', meta={'progress': 20})
        dataset = InstitutionDataset(
            features=features_list,
            labels=labels_list,
            normalize=True,
        )
        
        # 3. Initialize model and trainer
        self.update_state(state='INITIALIZING_MODEL', meta={'progress': 30})
        model = DefaultPredictorModel()
        trainer = DefaultPredictorTrainer(model=model)
        
        # 4. Hyperparameter search (optional)
        best_params = {}
        if hyperparameter_search:
            self.update_state(state='HYPERPARAMETER_SEARCH', meta={'progress': 40})
            logger.info("Starting hyperparameter search...")
            best_params = trainer.hyperparameter_search(
                dataset=dataset,
                n_trials=20,
                timeout=3600,  # 1 hour
            )
            
            # Create new model with best params
            if best_params:
                model = DefaultPredictorModel(
                    hidden_dims=(
                        best_params.get('hidden_dim1', 128),
                        best_params.get('hidden_dim2', 64),
                        best_params.get('hidden_dim3', 32),
                    ),
                    dropout_rate=best_params.get('dropout_rate', 0.3),
                )
                trainer = DefaultPredictorTrainer(model=model)
                learning_rate = best_params.get('learning_rate', learning_rate)
                batch_size = best_params.get('batch_size', batch_size)
        
        # 5. Train model
        self.update_state(state='TRAINING', meta={'progress': 50})
        logger.info("Starting model training...")
        
        training_results = trainer.train(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        logger.info(
            f"Training complete: "
            f"AUC={training_results['best_val_auc']:.4f}, "
            f"F1={training_results['best_val_f1']:.4f}"
        )
        
        # 6. Save normalization parameters
        self.update_state(state='SAVING_MODEL', meta={'progress': 90})
        mean, std = dataset.get_normalization_params()
        if mean is not None and std is not None:
            # Save normalization params with model
            import torch
            save_dir = Path(training_results['save_dir'])
            checkpoint = torch.load(save_dir / "best_model.pt")
            checkpoint['normalization_mean'] = mean
            checkpoint['normalization_std'] = std
            checkpoint['version'] = ml_config.DEFAULT_PREDICTOR_VERSION
            torch.save(checkpoint, save_dir / "best_model.pt")
        
        # 7. Register model (if MLflow available)
        self.update_state(state='REGISTERING', meta={'progress': 95})
        registry = ModelRegistry()
        registry.register_model(
            model_name="default_predictor",
            model_path=Path(training_results['save_dir']) / "best_model.pt",
            metrics={
                'val_auc': training_results['best_val_auc'],
                'val_f1': training_results['best_val_f1'],
                'val_loss': training_results['best_val_loss'],
            },
            version=ml_config.DEFAULT_PREDICTOR_VERSION,
            stage="Staging",
            parameters={
                'num_simulations': num_simulations,
                'epochs': training_results['epochs_trained'],
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                **best_params,
            },
        )
        
        self.update_state(state='SUCCESS', meta={'progress': 100})
        
        return {
            'status': 'SUCCESS',
            'metrics': {
                'val_auc': training_results['best_val_auc'],
                'val_f1': training_results['best_val_f1'],
                'val_loss': training_results['best_val_loss'],
            },
            'epochs_trained': training_results['epochs_trained'],
            'model_path': training_results['save_dir'],
            'hyperparameters': best_params if hyperparameter_search else {},
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


@shared_task(name="ml.periodic_retraining")
def periodic_retraining_task() -> Dict:
    """
    Periodic model retraining task
    
    Can be scheduled via Celery Beat to retrain models regularly.
    """
    logger.info("Starting periodic retraining...")
    
    # Trigger training with default parameters
    result = train_default_predictor_task.apply_async(
        kwargs={
            'num_simulations': 200,
            'epochs': 100,
            'hyperparameter_search': False,
        }
    )
    
    return {
        'status': 'SCHEDULED',
        'task_id': result.id,
        'message': 'Periodic retraining scheduled',
    }


@shared_task(name="ml.evaluate_model")
def evaluate_model_task(
    model_path: str,
    num_test_simulations: int = 50,
) -> Dict:
    """
    Evaluate model performance on test data
    
    Args:
        model_path: Path to model checkpoint
        num_test_simulations: Number of test simulations
    
    Returns:
        Evaluation metrics
    """
    try:
        import torch
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
        
        logger.info(f"Evaluating model: {model_path}")
        
        # Generate test data
        generator = SyntheticDataGenerator()
        features_list, labels_list = generator.generate_default_prediction_dataset(
            num_simulations=num_test_simulations,
            timesteps_per_sim=30,
        )
        
        # Create dataset
        dataset = InstitutionDataset(
            features=features_list,
            labels=labels_list,
            normalize=True,
        )
        
        # Load model
        model = DefaultPredictorModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Make predictions
        import numpy as np
        predictions = []
        labels = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset[i]
                pred = model(x.unsqueeze(0))
                predictions.append(pred.item())
                labels.append(y.item())
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Compute metrics
        auc = roc_auc_score(labels, predictions)
        
        # Optimal threshold for F1
        thresholds = np.linspace(0.1, 0.9, 20)
        f1_scores = [
            f1_score(labels, (predictions > t).astype(int))
            for t in thresholds
        ]
        best_f1 = max(f1_scores)
        
        logger.info(f"Evaluation: AUC={auc:.4f}, F1={best_f1:.4f}")
        
        return {
            'status': 'SUCCESS',
            'metrics': {
                'auc': float(auc),
                'f1': float(best_f1),
            },
            'num_test_samples': len(labels),
            'model_path': model_path,
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            'status': 'FAILURE',
            'error': str(e),
        }


@shared_task(name="ml.generate_training_data")
def generate_training_data_task(
    num_simulations: int = 100,
    output_path: str = "ml_models/training_data",
) -> Dict:
    """
    Generate and save training data
    
    Args:
        num_simulations: Number of simulations
        output_path: Directory to save data
    
    Returns:
        Generation results
    """
    try:
        import pickle
        
        logger.info(f"Generating training data: {num_simulations} simulations")
        
        generator = SyntheticDataGenerator()
        features_list, labels_list = generator.generate_balanced_dataset(
            target_samples=num_simulations * 50,
            default_ratio=0.3,
        )
        
        # Save data
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "features.pkl", "wb") as f:
            pickle.dump(features_list, f)
        
        with open(output_dir / "labels.pkl", "wb") as f:
            pickle.dump(labels_list, f)
        
        logger.info(f"Saved {len(features_list)} samples to {output_dir}")
        
        return {
            'status': 'SUCCESS',
            'num_samples': len(features_list),
            'num_defaults': sum(labels_list),
            'output_path': str(output_dir),
        }
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=True)
        return {
            'status': 'FAILURE',
            'error': str(e),
        }
