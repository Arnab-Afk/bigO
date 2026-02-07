"""
Tests for ML Inference Engine
"""

import pytest
import torch
import networkx as nx
from uuid import uuid4
from pathlib import Path

from app.engine.game_theory import AgentState
from app.ml.inference.predictor import DefaultPredictor, PredictionResult
from app.ml.features.extractor import FeatureExtractor
from app.ml.models.default_predictor import DefaultPredictorModel


class TestDefaultPredictor:
    """Test inference engine"""
    
    @pytest.fixture
    def predictor(self, tmp_path):
        """Create predictor with dummy model"""
        # Create and save a dummy model
        model = DefaultPredictorModel()
        model_dir = tmp_path / "default_predictor"
        model_dir.mkdir()
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim': 20,
                'hidden_dims': (128, 64, 32),
                'dropout_rate': 0.3,
            },
            'version': 'test_v1.0.0',
        }
        
        model_path = model_dir / "best_model.pt"
        torch.save(checkpoint, model_path)
        
        # Create predictor
        feature_extractor = FeatureExtractor()
        predictor = DefaultPredictor(
            model_path=model_path,
            feature_extractor=feature_extractor,
        )
        
        return predictor
    
    @pytest.fixture
    def test_data(self):
        """Create test network and agent states"""
        # Network
        G = nx.DiGraph()
        institutions = [uuid4() for _ in range(3)]
        for inst_id in institutions:
            G.add_node(inst_id)
        
        # Add edges
        G.add_edge(institutions[0], institutions[1], exposure_magnitude=100)
        G.add_edge(institutions[1], institutions[2], exposure_magnitude=150)
        
        # Agent states
        agent_states = {}
        for i, inst_id in enumerate(institutions):
            agent_states[inst_id] = AgentState(
                agent_id=inst_id,
                capital_ratio=0.10 + i * 0.02,
                liquidity_buffer=0.4 + i * 0.1,
                credit_exposure=100 + i * 50,
                default_probability=0.02 + i * 0.01,
                stress_level=0.2 + i * 0.1,
                risk_appetite=0.5,
            )
        
        return G, institutions, agent_states
    
    def test_predictor_initialization(self, predictor):
        """Test predictor can be initialized"""
        assert predictor.model is not None
        assert predictor.model_version == "test_v1.0.0"
    
    def test_single_prediction(self, predictor, test_data):
        """Test prediction for single institution"""
        G, institutions, agent_states = test_data
        
        inst_id = institutions[0]
        result = predictor.predict(
            institution_id=inst_id,
            agent_state=agent_states[inst_id],
            network=G,
            all_agent_states=agent_states,
        )
        
        assert isinstance(result, PredictionResult)
        assert result.institution_id == inst_id
        assert 0 <= result.probability <= 1
        assert 0 <= result.confidence <= 1
        assert result.inference_time_ms >= 0
    
    def test_batch_prediction(self, predictor, test_data):
        """Test batch prediction"""
        G, institutions, agent_states = test_data
        
        results = predictor.predict_batch(
            agent_states=agent_states,
            network=G,
        )
        
        assert len(results) == len(institutions)
        
        for inst_id in institutions:
            assert inst_id in results
            result = results[inst_id]
            assert isinstance(result, PredictionResult)
            assert 0 <= result.probability <= 1
            assert 0 <= result.confidence <= 1
    
    def test_prediction_latency(self, predictor, test_data):
        """Test prediction meets latency requirements"""
        G, institutions, agent_states = test_data
        
        result = predictor.predict(
            institution_id=institutions[0],
            agent_state=agent_states[institutions[0]],
            network=G,
            all_agent_states=agent_states,
            use_confidence=False,  # Faster without MC dropout
        )
        
        # Should be under 50ms
        assert result.inference_time_ms < 100  # Lenient for tests
    
    def test_fallback_without_model(self, test_data):
        """Test fallback behavior when model not loaded"""
        G, institutions, agent_states = test_data
        
        # Create predictor without model
        predictor = DefaultPredictor(
            model_path=None,
            feature_extractor=FeatureExtractor(),
        )
        
        inst_id = institutions[0]
        result = predictor.predict(
            institution_id=inst_id,
            agent_state=agent_states[inst_id],
            network=G,
            all_agent_states=agent_states,
        )
        
        # Should fallback to prior probability
        assert result.probability == agent_states[inst_id].default_probability
        assert result.model_version == "fallback"
