"""
ML Configuration Settings
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class MLConfig(BaseSettings):
    """Machine Learning configuration"""
    
    # Model paths
    ML_MODELS_PATH: Path = Field(
        default=Path("ml_models"),
        description="Base path for storing trained models"
    )
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="rudra-ml",
        description="MLflow experiment name"
    )
    
    # Inference settings
    MODEL_SERVING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for model predictions"
    )
    
    # Hardware settings
    ENABLE_GPU: bool = Field(
        default=False,
        description="Enable GPU acceleration if available"
    )
    
    # Default predictor settings
    DEFAULT_PREDICTOR_VERSION: str = Field(
        default="v1.0.0",
        description="Default predictor model version"
    )
    DEFAULT_PREDICTOR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum confidence score to use ML prediction"
    )
    
    # Training settings
    TRAINING_EPOCHS: int = Field(default=100, description="Maximum training epochs")
    EARLY_STOPPING_PATIENCE: int = Field(default=10, description="Early stopping patience")
    LEARNING_RATE: float = Field(default=0.001, description="Initial learning rate")
    BATCH_SIZE: int = Field(default=64, description="Training batch size")
    VALIDATION_SPLIT: float = Field(default=0.2, description="Validation data split ratio")
    
    # Feature engineering
    FEATURE_WINDOW_SIZE: int = Field(
        default=20,
        description="Historical window size for time series features"
    )
    
    # Cascade classifier settings
    CASCADE_GNN_LAYERS: int = Field(default=3, description="Number of GNN layers")
    CASCADE_HIDDEN_DIM: int = Field(default=64, description="GNN hidden dimension")
    
    # LSTM forecaster settings
    LSTM_LAYERS: int = Field(default=2, description="Number of LSTM layers")
    LSTM_HIDDEN_DIM: int = Field(default=128, description="LSTM hidden dimension")
    FORECAST_HORIZON: int = Field(default=10, description="Forecast timesteps ahead")
    
    # Performance monitoring
    MAX_INFERENCE_LATENCY_MS: float = Field(
        default=50.0,
        description="Maximum acceptable inference latency in milliseconds"
    )
    
    class Config:
        env_prefix = "RUDRA_ML_"
        case_sensitive = False


# Global configuration instance
ml_config = MLConfig()
