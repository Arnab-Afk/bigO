"""
ML API Endpoints

Provides REST API for ML predictions, model management, and training.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
import torch

from app.ml.inference.predictor import DefaultPredictor
from app.ml.features.extractor import FeatureExtractor
from app.ml.registry.model_manager import ModelRegistry
from app.ml.config import ml_config
from app.engine.game_theory import AgentState

router = APIRouter(prefix="/ml", tags=["machine-learning"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for default probability prediction"""
    institution_id: UUID
    capital_ratio: float = Field(..., ge=0.0, le=1.0)
    liquidity_buffer: float = Field(..., ge=0.0, le=1.0)
    credit_exposure: float = Field(..., ge=0.0)
    default_probability: float = Field(..., ge=0.0, le=1.0)
    stress_level: float = Field(..., ge=0.0, le=1.0)
    risk_appetite: float = Field(0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response with ML prediction"""
    institution_id: UUID
    predicted_probability: float
    confidence: float
    model_version: str
    inference_time_ms: float
    should_use_prediction: bool


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    institutions: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    total_inference_time_ms: float


class TrainModelRequest(BaseModel):
    """Request to train a new model"""
    num_simulations: int = Field(100, ge=10, le=1000)
    epochs: int = Field(100, ge=10, le=500)
    batch_size: int = Field(64, ge=16, le=256)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.01)
    hyperparameter_search: bool = False


class TrainModelResponse(BaseModel):
    """Response for training request"""
    task_id: str
    status: str
    message: str


class ModelInfoResponse(BaseModel):
    """Model information"""
    model_loaded: bool
    model_version: str
    device: str
    confidence_threshold: float
    normalization_enabled: bool


class ModelMetadataResponse(BaseModel):
    """Model metadata from registry"""
    model_name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    registered_at: datetime


class ForecastRequest(BaseModel):
    """Request for early warning forecast"""
    institution_id: UUID
    historical_states: List[List[float]]  # [timesteps, features]


class ForecastResponse(BaseModel):
    """Response with forecast"""
    institution_id: UUID
    warning: bool
    risk_score: float
    forecasted_capital_ratio: List[float]
    forecasted_default_prob: List[float]


# ============================================================================
# Global State
# ============================================================================

_predictor: Optional[DefaultPredictor] = None
_model_registry: Optional[ModelRegistry] = None


def get_predictor() -> DefaultPredictor:
    """Get or initialize default predictor"""
    global _predictor
    if _predictor is None:
        try:
            model_path = ml_config.ML_MODELS_PATH / "default_predictor" / "best_model.pt"
            feature_extractor = FeatureExtractor()
            _predictor = DefaultPredictor(
                model_path=model_path,
                feature_extractor=feature_extractor,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize ML predictor: {e}"
            )
    return _predictor


def get_model_registry() -> ModelRegistry:
    """Get or initialize model registry"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/health", summary="Check ML service health")
async def health_check() -> Dict:
    """Check if ML service is available"""
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        return {
            "status": "healthy",
            "ml_enabled": info["model_loaded"],
            "model_version": info["model_version"],
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.post("/predict", response_model=PredictionResponse, summary="Predict default probability")
async def predict_default(request: PredictionRequest) -> PredictionResponse:
    """
    Predict default probability for a single institution
    
    Uses ML model to predict probability based on current state.
    """
    predictor = get_predictor()
    
    # Create minimal network (single node)
    import networkx as nx
    network = nx.DiGraph()
    network.add_node(request.institution_id)
    
    # Create agent state
    agent_state = AgentState(
        agent_id=request.institution_id,
        capital_ratio=request.capital_ratio,
        liquidity_buffer=request.liquidity_buffer,
        credit_exposure=request.credit_exposure,
        default_probability=request.default_probability,
        stress_level=request.stress_level,
        risk_appetite=request.risk_appetite,
    )
    
    # Predict
    result = predictor.predict(
        institution_id=request.institution_id,
        agent_state=agent_state,
        network=network,
        all_agent_states={request.institution_id: agent_state},
        use_confidence=True,
    )
    
    return PredictionResponse(
        institution_id=result.institution_id,
        predicted_probability=result.probability,
        confidence=result.confidence,
        model_version=result.model_version,
        inference_time_ms=result.inference_time_ms,
        should_use_prediction=predictor.should_use_ml_prediction(result),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch prediction")
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict default probabilities for multiple institutions
    
    More efficient than individual predictions.
    """
    import time
    predictor = get_predictor()
    
    # Create network with all institutions
    import networkx as nx
    network = nx.DiGraph()
    agent_states = {}
    
    for inst_req in request.institutions:
        network.add_node(inst_req.institution_id)
        agent_states[inst_req.institution_id] = AgentState(
            agent_id=inst_req.institution_id,
            capital_ratio=inst_req.capital_ratio,
            liquidity_buffer=inst_req.liquidity_buffer,
            credit_exposure=inst_req.credit_exposure,
            default_probability=inst_req.default_probability,
            stress_level=inst_req.stress_level,
            risk_appetite=inst_req.risk_appetite,
        )
    
    # Batch predict
    start_time = time.time()
    results = predictor.predict_batch(
        agent_states=agent_states,
        network=network,
    )
    total_time = (time.time() - start_time) * 1000
    
    # Convert to response
    predictions = [
        PredictionResponse(
            institution_id=result.institution_id,
            predicted_probability=result.probability,
            confidence=result.confidence,
            model_version=result.model_version,
            inference_time_ms=result.inference_time_ms,
            should_use_prediction=predictor.should_use_ml_prediction(result),
        )
        for result in results.values()
    ]
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_inference_time_ms=total_time,
    )


@router.get("/model/info", response_model=ModelInfoResponse, summary="Get model information")
async def get_model_info() -> ModelInfoResponse:
    """Get information about the loaded ML model"""
    predictor = get_predictor()
    info = predictor.get_model_info()
    
    return ModelInfoResponse(**info)


@router.post("/train", response_model=TrainModelResponse, summary="Train new model")
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
) -> TrainModelResponse:
    """
    Trigger model training job
    
    Runs as background task. Returns task_id for tracking.
    """
    from app.tasks.ml_tasks import train_default_predictor_task
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Schedule training task
    task = train_default_predictor_task.apply_async(
        kwargs={
            'num_simulations': request.num_simulations,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
            'learning_rate': request.learning_rate,
            'hyperparameter_search': request.hyperparameter_search,
        },
        task_id=task_id,
    )
    
    return TrainModelResponse(
        task_id=task_id,
        status="PENDING",
        message="Training task scheduled. Use /ml/train/status/{task_id} to check progress.",
    )


@router.get("/train/status/{task_id}", summary="Check training status")
async def get_training_status(task_id: str) -> Dict:
    """Check status of a training task"""
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": result.state,
        "result": result.result if result.ready() else None,
    }


@router.get("/registry/models", response_model=List[ModelMetadataResponse], summary="List registered models")
async def list_models(model_name: Optional[str] = None) -> List[ModelMetadataResponse]:
    """List all registered models in MLflow"""
    registry = get_model_registry()
    models = registry.list_models(model_name=model_name)
    
    return [
        ModelMetadataResponse(
            model_name=m.model_name,
            version=m.version,
            stage=m.stage,
            metrics=m.metrics,
            registered_at=m.registered_at,
        )
        for m in models
    ]


@router.post("/registry/promote", summary="Promote model version")
async def promote_model(
    model_name: str,
    version: str,
    stage: str = "Production",
) -> Dict:
    """Promote a model version to a different stage"""
    registry = get_model_registry()
    success = registry.promote_model(model_name, version, stage)
    
    if success:
        return {
            "status": "success",
            "message": f"Promoted {model_name} v{version} to {stage}",
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to promote model",
        )


@router.post("/registry/rollback", summary="Rollback to previous model")
async def rollback_model(
    model_name: str,
    target_version: str,
) -> Dict:
    """Rollback to a previous model version"""
    registry = get_model_registry()
    success = registry.rollback_model(model_name, target_version)
    
    if success:
        return {
            "status": "success",
            "message": f"Rolled back {model_name} to v{target_version}",
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rollback model",
        )


@router.get("/forecast/early-warning", summary="Early warning system")
async def early_warning_forecast(
    institution_id: UUID,
    historical_states: List[List[float]],
) -> ForecastResponse:
    """
    Get early warning forecast for an institution
    
    Uses LSTM model to predict future states and detect risk.
    """
    # This is a placeholder - full implementation requires loaded LSTM model
    # For now, return a simple response
    
    import numpy as np
    
    # Simple heuristic: check if capital ratio is declining
    if len(historical_states) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 historical states",
        )
    
    capital_ratios = [state[0] for state in historical_states]
    is_declining = capital_ratios[-1] < capital_ratios[0]
    
    risk_score = 0.7 if is_declining else 0.3
    
    # Forecast next 10 timesteps (simple linear extrapolation)
    trend = (capital_ratios[-1] - capital_ratios[0]) / len(capital_ratios)
    forecasted_capital = [
        max(0.0, capital_ratios[-1] + trend * i)
        for i in range(1, 11)
    ]
    
    forecasted_default_prob = [
        min(1.0, 0.1 + 0.05 * i) if is_declining else 0.05
        for i in range(10)
    ]
    
    return ForecastResponse(
        institution_id=institution_id,
        warning=risk_score > 0.5,
        risk_score=risk_score,
        forecasted_capital_ratio=forecasted_capital,
        forecasted_default_prob=forecasted_default_prob,
    )
