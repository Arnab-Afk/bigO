"""
Tests for ML Feature Extraction
"""

import pytest
import networkx as nx
import numpy as np
from uuid import uuid4

from app.engine.game_theory import AgentState
from app.ml.features.extractor import FeatureExtractor, InstitutionFeatures


class TestFeatureExtractor:
    """Test feature extraction functionality"""
    
    @pytest.fixture
    def network(self):
        """Create test network"""
        G = nx.DiGraph()
        
        # Add 5 institutions
        institutions = [uuid4() for _ in range(5)]
        for inst_id in institutions:
            G.add_node(inst_id)
        
        # Add some exposures
        G.add_edge(institutions[0], institutions[1], exposure_magnitude=100)
        G.add_edge(institutions[0], institutions[2], exposure_magnitude=150)
        G.add_edge(institutions[1], institutions[3], exposure_magnitude=200)
        G.add_edge(institutions[2], institutions[3], exposure_magnitude=120)
        G.add_edge(institutions[3], institutions[4], exposure_magnitude=180)
        
        return G, institutions
    
    @pytest.fixture
    def agent_states(self, network):
        """Create test agent states"""
        _, institutions = network
        
        states = {}
        for i, inst_id in enumerate(institutions):
            states[inst_id] = AgentState(
                agent_id=inst_id,
                capital_ratio=0.08 + i * 0.02,
                liquidity_buffer=0.3 + i * 0.1,
                credit_exposure=100 + i * 50,
                default_probability=0.01 + i * 0.01,
                stress_level=0.1 + i * 0.05,
                risk_appetite=0.5,
            )
        
        return states
    
    def test_extract_features_single_institution(self, network, agent_states):
        """Test extracting features for a single institution"""
        G, institutions = network
        inst_id = institutions[0]
        
        extractor = FeatureExtractor()
        
        features = extractor.extract_features(
            institution_id=inst_id,
            agent_state=agent_states[inst_id],
            network=G,
            all_agent_states=agent_states,
        )
        
        assert isinstance(features, InstitutionFeatures)
        assert features.institution_id == inst_id
        assert features.feature_dim == 20
        
        # Check feature values
        assert 0 <= features.capital_ratio <= 1
        assert 0 <= features.liquidity_buffer <= 1
        assert features.leverage > 0
        assert 0 <= features.default_probability_prior <= 1
        
        # Check network features
        assert 0 <= features.degree_centrality <= 1
        assert 0 <= features.pagerank <= 1
    
    def test_feature_to_array(self, network, agent_states):
        """Test converting features to numpy array"""
        G, institutions = network
        
        extractor = FeatureExtractor()
        features = extractor.extract_features(
            institution_id=institutions[0],
            agent_state=agent_states[institutions[0]],
            network=G,
            all_agent_states=agent_states,
        )
        
        arr = features.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20,)
        assert arr.dtype == np.float32
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
    
    def test_extract_batch_features(self, network, agent_states):
        """Test batch feature extraction"""
        G, institutions = network
        
        extractor = FeatureExtractor()
        features_dict = extractor.extract_batch_features(
            agent_states=agent_states,
            network=G,
        )
        
        assert len(features_dict) == len(institutions)
        
        for inst_id in institutions:
            assert inst_id in features_dict
            assert isinstance(features_dict[inst_id], InstitutionFeatures)
    
    def test_neighborhood_features(self, network, agent_states):
        """Test neighborhood stress features"""
        G, institutions = network
        
        # Mark one institution as defaulted
        defaulted = {institutions[1]}
        
        extractor = FeatureExtractor()
        features = extractor.extract_features(
            institution_id=institutions[0],
            agent_state=agent_states[institutions[0]],
            network=G,
            all_agent_states=agent_states,
            defaulted_institutions=defaulted,
        )
        
        # Institution 0 is connected to 1 and 2
        # Institution 1 is defaulted
        assert features.neighbor_default_count >= 1
        assert 0 <= features.neighbor_avg_stress <= 1
        assert 0 <= features.neighbor_max_stress <= 1
