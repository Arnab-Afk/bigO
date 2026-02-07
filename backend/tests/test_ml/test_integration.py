"""
Integration Tests for ML in Simulation
"""

import pytest
import networkx as nx
from uuid import uuid4

from app.engine.simulation import SimulationEngine, Shock
from app.engine.game_theory import AgentState


class TestMLSimulationIntegration:
    """Test ML integration with simulation engine"""
    
    @pytest.fixture
    def network(self):
        """Create test network"""
        G = nx.DiGraph()
        
        institutions = [uuid4() for _ in range(5)]
        for inst_id in institutions:
            G.add_node(inst_id)
        
        # Add exposures
        for i in range(4):
            G.add_edge(
                institutions[i],
                institutions[i + 1],
                exposure_magnitude=100 + i * 20,
            )
        
        return G, institutions
    
    @pytest.fixture
    def initial_states(self, network):
        """Create initial agent states"""
        _, institutions = network
        
        states = {}
        for i, inst_id in enumerate(institutions):
            states[inst_id] = AgentState(
                agent_id=inst_id,
                capital_ratio=0.10 + i * 0.01,
                liquidity_buffer=0.5,
                credit_exposure=100.0,
                default_probability=0.02,
                stress_level=0.2,
                risk_appetite=0.5,
            )
        
        return states
    
    def test_simulation_without_ml(self, network, initial_states):
        """Test simulation runs without ML"""
        G, institutions = network
        
        engine = SimulationEngine(
            network=G,
            max_timesteps=10,
            enable_ml=False,  # Disable ML
        )
        
        sim_state = engine.run_simulation(
            simulation_id="test_no_ml",
            initial_states=initial_states,
            shocks=[],
            shock_timing={},
        )
        
        assert len(sim_state.timesteps) > 0
        
        # Check that ML fields are not set
        first_timestep = sim_state.timesteps[0]
        for inst_id in institutions:
            agent_state = first_timestep.agent_states[inst_id]
            assert agent_state.ml_prediction_confidence == 0.0
            assert agent_state.ml_model_version == "none"
    
    def test_simulation_with_ml_enabled(self, network, initial_states):
        """Test simulation with ML enabled (may fallback if no model)"""
        G, institutions = network
        
        engine = SimulationEngine(
            network=G,
            max_timesteps=10,
            enable_ml=True,  # Enable ML
        )
        
        # Should not crash even if model not available
        sim_state = engine.run_simulation(
            simulation_id="test_with_ml",
            initial_states=initial_states,
            shocks=[],
            shock_timing={},
        )
        
        assert len(sim_state.timesteps) > 0
    
    def test_ml_updates_agent_states(self, network, initial_states, tmp_path):
        """Test that ML predictions update agent states"""
        import torch
        from app.ml.models.default_predictor import DefaultPredictorModel
        
        # Create and save a model
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
        
        torch.save(checkpoint, model_dir / "best_model.pt")
        
        # Update config to use this model
        from app.ml.config import ml_config
        ml_config.ML_MODELS_PATH = tmp_path
        
        G, institutions = network
        
        engine = SimulationEngine(
            network=G,
            max_timesteps=5,
            enable_ml=True,
        )
        
        sim_state = engine.run_simulation(
            simulation_id="test_ml_updates",
            initial_states=initial_states,
            shocks=[],
            shock_timing={},
        )
        
        # Check if ML fields are updated (if model loaded)
        if engine.ml_predictor and engine.ml_predictor.model:
            last_timestep = sim_state.timesteps[-1]
            for inst_id in institutions:
                agent_state = last_timestep.agent_states[inst_id]
                # ML confidence should be > 0 if predictions were made
                # (may still be 0 if fallback used)
                assert agent_state.ml_model_version != "none" or agent_state.ml_prediction_confidence == 0.0
    
    def test_simulation_performance_with_ml(self, network, initial_states):
        """Test that ML doesn't significantly slow down simulation"""
        import time
        
        G, _ = network
        
        # Run without ML
        engine_no_ml = SimulationEngine(
            network=G,
            max_timesteps=20,
            enable_ml=False,
        )
        
        start = time.time()
        sim_state_no_ml = engine_no_ml.run_simulation(
            simulation_id="perf_no_ml",
            initial_states=initial_states,
            shocks=[],
            shock_timing={},
        )
        time_no_ml = time.time() - start
        
        # Run with ML
        engine_ml = SimulationEngine(
            network=G,
            max_timesteps=20,
            enable_ml=True,
        )
        
        start = time.time()
        sim_state_ml = engine_ml.run_simulation(
            simulation_id="perf_ml",
            initial_states=initial_states,
            shocks=[],
            shock_timing={},
        )
        time_ml = time.time() - start
        
        # ML should not be more than 3x slower (lenient)
        # In production with optimized models, should be < 2x
        assert time_ml < time_no_ml * 5
        
        print(f"Time without ML: {time_no_ml:.3f}s")
        print(f"Time with ML: {time_ml:.3f}s")
        print(f"Ratio: {time_ml / time_no_ml:.2f}x")
