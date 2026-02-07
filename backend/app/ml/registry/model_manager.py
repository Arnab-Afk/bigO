"""
Model Registry and Management with MLflow

Handles model versioning, serving, and A/B testing.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from app.ml.config import ml_config

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    model_name: str
    version: str
    stage: str  # "Production", "Staging", "Archived"
    metrics: Dict[str, float]
    registered_at: datetime
    model_path: Optional[Path] = None


class ModelRegistry:
    """
    Model Registry using MLflow
    
    Features:
    - Model versioning with semantic versioning
    - Experiment tracking
    - Model staging (Production, Staging, Archived)
    - A/B testing support
    - Rollback capabilities
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri or ml_config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or ml_config.MLFLOW_EXPERIMENT_NAME
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient()
            logger.info(f"MLflow registry initialized: {self.tracking_uri}")
        else:
            logger.warning("MLflow not available. Model registry disabled.")
            self.client = None
    
    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metrics: Dict[str, float],
        version: str,
        stage: str = "Staging",
        parameters: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Register a new model version
        
        Args:
            model_name: Name of the model
            model_path: Path to model checkpoint
            metrics: Performance metrics
            version: Model version string
            stage: Stage (Production/Staging/Archived)
            parameters: Training parameters
        
        Returns:
            Model URI or None if MLflow unavailable
        """
        if not MLFLOW_AVAILABLE or not self.client:
            logger.warning("MLflow unavailable, saving locally only")
            return None
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_{version}"):
                # Log parameters
                if parameters:
                    mlflow.log_params(parameters)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model artifact
                mlflow.log_artifact(str(model_path))
                
                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, model_name)
                
                logger.info(
                    f"Registered model {model_name} version {version} "
                    f"with stage {stage}"
                )
                
                return model_uri
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def get_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[ModelMetadata]:
        """
        Get model by name and stage
        
        Args:
            model_name: Model name
            stage: Model stage (Production/Staging/Archived)
        
        Returns:
            ModelMetadata or None
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return None
        
        try:
            # Get model versions
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            
            if not versions:
                logger.warning(f"No {stage} model found for {model_name}")
                return None
            
            latest_version = versions[0]
            
            # Get run details
            run = self.client.get_run(latest_version.run_id)
            
            metadata = ModelMetadata(
                model_name=model_name,
                version=latest_version.version,
                stage=latest_version.current_stage,
                metrics=run.data.metrics,
                registered_at=datetime.fromtimestamp(latest_version.creation_timestamp / 1000),
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None
    
    def list_models(
        self,
        model_name: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """
        List all registered models
        
        Args:
            model_name: Filter by model name (optional)
        
        Returns:
            List of ModelMetadata
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return []
        
        try:
            models = []
            
            if model_name:
                # Get specific model versions
                versions = self.client.search_model_versions(f"name='{model_name}'")
            else:
                # Get all models
                versions = self.client.search_model_versions("")
            
            for version in versions:
                try:
                    run = self.client.get_run(version.run_id)
                    metadata = ModelMetadata(
                        model_name=version.name,
                        version=version.version,
                        stage=version.current_stage,
                        metrics=run.data.metrics,
                        registered_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                    )
                    models.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to get metadata for version {version.version}: {e}")
                    continue
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production",
    ) -> bool:
        """
        Promote model to a different stage
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage
        
        Returns:
            Success status
        """
        if not MLFLOW_AVAILABLE or not self.client:
            logger.warning("MLflow unavailable")
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def rollback_model(
        self,
        model_name: str,
        target_version: str,
    ) -> bool:
        """
        Rollback to a previous model version
        
        Args:
            model_name: Model name
            target_version: Version to rollback to
        
        Returns:
            Success status
        """
        # Demote current production model
        current_prod = self.get_model(model_name, stage="Production")
        if current_prod:
            self.promote_model(model_name, current_prod.version, stage="Archived")
        
        # Promote target version to production
        return self.promote_model(model_name, target_version, stage="Production")
    
    def compare_models(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict:
        """
        Compare metrics between two model versions
        
        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version
        
        Returns:
            Comparison dictionary
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return {}
        
        try:
            # Get both versions
            version_a_obj = self.client.get_model_version(model_name, version_a)
            version_b_obj = self.client.get_model_version(model_name, version_b)
            
            # Get runs
            run_a = self.client.get_run(version_a_obj.run_id)
            run_b = self.client.get_run(version_b_obj.run_id)
            
            # Compare metrics
            metrics_a = run_a.data.metrics
            metrics_b = run_b.data.metrics
            
            comparison = {
                'model_name': model_name,
                'version_a': version_a,
                'version_b': version_b,
                'metrics_a': metrics_a,
                'metrics_b': metrics_b,
                'differences': {
                    metric: metrics_b.get(metric, 0) - metrics_a.get(metric, 0)
                    for metric in set(metrics_a.keys()) | set(metrics_b.keys())
                },
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
