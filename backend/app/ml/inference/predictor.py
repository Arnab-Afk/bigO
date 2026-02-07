"""
Real-time Inference Engine for Default Prediction

Provides low-latency prediction during simulation execution.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

import numpy as np
import torch

from app.ml.config import ml_config
from app.ml.features.extractor import FeatureExtractor, InstitutionFeatures
from app.ml.models.default_predictor import DefaultPredictorModel
from app.engine.game_theory import AgentState

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a default probability prediction"""
    institution_id: UUID
    probability: float
    confidence: float
    model_version: str
    inference_time_ms: float
    features_used: Optional[InstitutionFeatures] = None


class DefaultPredictor:
    """
    Default Probability Inference Engine
    
    Loads trained model and provides real-time predictions during simulations.
    Includes:
    - Fast batch inference
    - Confidence scoring via Monte Carlo Dropout
    - Feature normalization
    - Fallback to prior if confidence is low
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            feature_extractor: FeatureExtractor instance
            device: torch device
        """
        # Setup device
        if device is None:
            if ml_config.ENABLE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Load model
        self.model = None
        self.model_version = "unknown"
        self.normalization_mean = None
        self.normalization_std = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning(
                f"No model found at {model_path}, predictions will use fallback"
            )
        
        self.confidence_threshold = ml_config.DEFAULT_PREDICTOR_CONFIDENCE_THRESHOLD
    
    def load_model(self, model_path: Path):
        """
        Load trained model from checkpoint
        
        Args:
            model_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract config
            config = checkpoint.get('config', {})
            
            # Create model
            self.model = DefaultPredictorModel(
                input_dim=config.get('input_dim', 20),
                hidden_dims=config.get('hidden_dims', (128, 64, 32)),
                dropout_rate=config.get('dropout_rate', 0.3),
            )
            
            # Load state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load normalization parameters if available
            if 'normalization_mean' in checkpoint:
                self.normalization_mean = checkpoint['normalization_mean'].to(self.device)
                self.normalization_std = checkpoint['normalization_std'].to(self.device)
            
            # Extract version
            self.model_version = checkpoint.get('version', ml_config.DEFAULT_PREDICTOR_VERSION)
            
            logger.info(f"Loaded model from {model_path} (version: {self.model_version})")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None
    
    def predict(
        self,
        institution_id: UUID,
        agent_state: AgentState,
        network,
        all_agent_states: Dict[UUID, AgentState],
        centralities: Optional[Dict] = None,
        defaulted_institutions: Optional[set] = None,
        use_confidence: bool = True,
    ) -> PredictionResult:
        """
        Predict default probability for a single institution
        
        Args:
            institution_id: Institution UUID
            agent_state: AgentState
            network: NetworkX graph
            all_agent_states: All agent states
            centralities: Pre-computed centralities
            defaulted_institutions: Set of defaulted institutions
            use_confidence: Whether to compute confidence via MC Dropout
        
        Returns:
            PredictionResult with probability and confidence
        """
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(
            institution_id=institution_id,
            agent_state=agent_state,
            network=network,
            all_agent_states=all_agent_states,
            centralities=centralities,
            defaulted_institutions=defaulted_institutions,
        )
        
        # Check if model is loaded
        if self.model is None:
            # Fallback: use prior default probability
            inference_time = (time.time() - start_time) * 1000
            return PredictionResult(
                institution_id=institution_id,
                probability=agent_state.default_probability,
                confidence=0.5,  # Low confidence
                model_version="fallback",
                inference_time_ms=inference_time,
                features_used=features,
            )
        
        # Convert to tensor
        X = torch.from_numpy(features.to_array()).unsqueeze(0).to(self.device)
        
        # Normalize
        if self.normalization_mean is not None and self.normalization_std is not None:
            X = (X - self.normalization_mean) / self.normalization_std
        
        # Predict
        with torch.no_grad():
            if use_confidence:
                probability, confidence = self.model.predict_with_confidence(
                    X, num_samples=10
                )
                probability = probability.item()
                confidence = confidence.item()
            else:
                probability = self.model(X).item()
                confidence = 1.0  # High confidence when not using MC dropout
        
        inference_time = (time.time() - start_time) * 1000
        
        # Check performance constraint
        if inference_time > ml_config.MAX_INFERENCE_LATENCY_MS:
            logger.warning(
                f"Inference latency {inference_time:.2f}ms exceeds "
                f"threshold {ml_config.MAX_INFERENCE_LATENCY_MS}ms"
            )
        
        return PredictionResult(
            institution_id=institution_id,
            probability=probability,
            confidence=confidence,
            model_version=self.model_version,
            inference_time_ms=inference_time,
            features_used=features,
        )
    
    def predict_batch(
        self,
        agent_states: Dict[UUID, AgentState],
        network,
        defaulted_institutions: Optional[set] = None,
    ) -> Dict[UUID, PredictionResult]:
        """
        Batch prediction for all institutions
        
        Args:
            agent_states: Dictionary of all agent states
            network: NetworkX graph
            defaulted_institutions: Set of defaulted institutions
        
        Returns:
            Dictionary mapping institution_id to PredictionResult
        """
        start_time = time.time()
        
        # Extract features for all institutions
        features_dict = self.feature_extractor.extract_batch_features(
            agent_states=agent_states,
            network=network,
            defaulted_institutions=defaulted_institutions,
        )
        
        # Fallback if no model
        if self.model is None:
            results = {}
            for inst_id, features in features_dict.items():
                results[inst_id] = PredictionResult(
                    institution_id=inst_id,
                    probability=agent_states[inst_id].default_probability,
                    confidence=0.5,
                    model_version="fallback",
                    inference_time_ms=0.0,
                    features_used=features,
                )
            return results
        
        # Convert to batch tensor
        institution_ids = list(features_dict.keys())
        X_batch = torch.stack([
            torch.from_numpy(features_dict[inst_id].to_array())
            for inst_id in institution_ids
        ]).to(self.device)
        
        # Normalize
        if self.normalization_mean is not None and self.normalization_std is not None:
            X_batch = (X_batch - self.normalization_mean) / self.normalization_std
        
        # Predict
        with torch.no_grad():
            predictions = self.model(X_batch).cpu().numpy().flatten()
        
        total_time = (time.time() - start_time) * 1000
        per_institution_time = total_time / len(institution_ids)
        
        # Build results
        results = {}
        for i, inst_id in enumerate(institution_ids):
            results[inst_id] = PredictionResult(
                institution_id=inst_id,
                probability=float(predictions[i]),
                confidence=0.85,  # Default confidence for batch
                model_version=self.model_version,
                inference_time_ms=per_institution_time,
                features_used=features_dict[inst_id],
            )
        
        logger.debug(
            f"Batch prediction: {len(institution_ids)} institutions in {total_time:.2f}ms "
            f"({per_institution_time:.2f}ms per institution)"
        )
        
        return results
    
    def should_use_ml_prediction(self, prediction: PredictionResult) -> bool:
        """
        Determine if ML prediction should be used or fall back to prior
        
        Args:
            prediction: PredictionResult
        
        Returns:
            True if confidence is above threshold
        """
        return prediction.confidence >= self.confidence_threshold
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'normalization_enabled': self.normalization_mean is not None,
        }
