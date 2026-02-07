"""
Example: Using ML for Default Prediction

Demonstrates how to:
1. Train a default predictor model
2. Use it for inference
3. Integrate with simulations
"""

import logging
from pathlib import Path
from uuid import uuid4

import networkx as nx
import torch

from app.engine.game_theory import AgentState
from app.engine.simulation import SimulationEngine, Shock
from app.ml.data.synthetic_generator import SyntheticDataGenerator
from app.ml.training.dataset import InstitutionDataset
from app.ml.training.trainer import DefaultPredictorTrainer
from app.ml.models.default_predictor import DefaultPredictorModel
from app.ml.inference.predictor import DefaultPredictor
from app.ml.features.extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_train_model():
    """Example 1: Train a default predictor model"""
    logger.info("=== Example 1: Training Default Predictor ===")
    
    # Generate synthetic training data
    logger.info("Generating training data...")
    generator = SyntheticDataGenerator()
    features_list, labels_list = generator.generate_balanced_dataset(
        target_samples=5000,  # Small dataset for quick demo
        default_ratio=0.3,
    )
    
    logger.info(f"Generated {len(features_list)} samples")
    
    # Create dataset
    dataset = InstitutionDataset(features_list, labels_list, normalize=True)
    
    # Create and train model
    logger.info("Training model...")
    model = DefaultPredictorModel()
    trainer = DefaultPredictorTrainer(model=model)
    
    results = trainer.train(
        dataset=dataset,
        epochs=20,  # Quick training for demo
        batch_size=64,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Best AUC: {results['best_val_auc']:.4f}")
    logger.info(f"Best F1: {results['best_val_f1']:.4f}")
    logger.info(f"Model saved to: {results['save_dir']}")
    
    return Path(results['save_dir']) / "best_model.pt"


def example_2_inference(model_path):
    """Example 2: Use trained model for inference"""
    logger.info("\n=== Example 2: Inference with Trained Model ===")
    
    # Create test network
    G = nx.DiGraph()
    institutions = [uuid4() for _ in range(5)]
    
    for inst in institutions:
        G.add_node(inst)
    
    # Add exposures
    G.add_edge(institutions[0], institutions[1], exposure_magnitude=100)
    G.add_edge(institutions[1], institutions[2], exposure_magnitude=150)
    G.add_edge(institutions[2], institutions[3], exposure_magnitude=120)
    
    # Create agent states
    agent_states = {}
    for i, inst_id in enumerate(institutions):
        agent_states[inst_id] = AgentState(
            agent_id=inst_id,
            capital_ratio=0.08 + i * 0.02,
            liquidity_buffer=0.4 + i * 0.1,
            credit_exposure=100 + i * 30,
            default_probability=0.02,
            stress_level=0.2 + i * 0.05,
            risk_appetite=0.5,
        )
    
    # Load predictor
    logger.info(f"Loading model from {model_path}...")
    feature_extractor = FeatureExtractor()
    predictor = DefaultPredictor(
        model_path=model_path,
        feature_extractor=feature_extractor,
    )
    
    # Make predictions
    logger.info("Making predictions...")
    results = predictor.predict_batch(
        agent_states=agent_states,
        network=G,
    )
    
    # Display results
    logger.info("\nPrediction Results:")
    for inst_id, result in results.items():
        logger.info(
            f"Institution {str(inst_id)[:8]}...: "
            f"P(default)={result.probability:.4f}, "
            f"Confidence={result.confidence:.2f}, "
            f"Time={result.inference_time_ms:.2f}ms"
        )


def example_3_simulation_with_ml(model_path):
    """Example 3: Run simulation with ML predictions"""
    logger.info("\n=== Example 3: Simulation with ML ===")
    
    # Create network
    G = nx.DiGraph()
    institutions = [uuid4() for _ in range(5)]
    
    for inst in institutions:
        G.add_node(inst)
    
    for i in range(4):
        G.add_edge(
            institutions[i],
            institutions[i + 1],
            exposure_magnitude=150,
        )
    
    # Create initial states
    initial_states = {}
    for i, inst_id in enumerate(institutions):
        initial_states[inst_id] = AgentState(
            agent_id=inst_id,
            capital_ratio=0.10,
            liquidity_buffer=0.5,
            credit_exposure=100.0,
            default_probability=0.02,
            stress_level=0.2,
            risk_appetite=0.5,
        )
    
    # Create shock
    shock = Shock(
        shock_id="shock_1",
        shock_type="liquidity_freeze",
        target_institutions=[institutions[0]],
        magnitude=0.4,
    )
    
    # Update ML config to use our model
    from app.ml.config import ml_config
    ml_config.ML_MODELS_PATH = model_path.parent.parent
    
    # Run simulation WITH ML
    logger.info("Running simulation WITH ML...")
    engine_ml = SimulationEngine(
        network=G.copy(),
        max_timesteps=20,
        enable_ml=True,
    )
    
    sim_state_ml = engine_ml.run_simulation(
        simulation_id="sim_with_ml",
        initial_states=initial_states.copy(),
        shocks=[shock],
        shock_timing={5: ["shock_1"]},
    )
    
    # Run simulation WITHOUT ML for comparison
    logger.info("Running simulation WITHOUT ML...")
    engine_no_ml = SimulationEngine(
        network=G.copy(),
        max_timesteps=20,
        enable_ml=False,
    )
    
    sim_state_no_ml = engine_no_ml.run_simulation(
        simulation_id="sim_without_ml",
        initial_states=initial_states.copy(),
        shocks=[shock],
        shock_timing={5: ["shock_1"]},
    )
    
    # Compare results
    logger.info("\n=== Comparison ===")
    logger.info(f"With ML: {len(sim_state_ml.final_defaults)} defaults")
    logger.info(f"Without ML: {len(sim_state_no_ml.final_defaults)} defaults")
    
    # Show ML prediction info
    if engine_ml.ml_predictor and engine_ml.ml_predictor.model:
        logger.info("\nML Predictions in final timestep:")
        final_timestep = sim_state_ml.timesteps[-1]
        for inst_id in institutions:
            if inst_id in final_timestep.agent_states:
                state = final_timestep.agent_states[inst_id]
                logger.info(
                    f"  {str(inst_id)[:8]}...: "
                    f"P(default)={state.default_probability:.4f}, "
                    f"ML_conf={state.ml_prediction_confidence:.2f}, "
                    f"Version={state.ml_model_version}"
                )


def main():
    """Run all examples"""
    logger.info("===== ML Integration Examples =====\n")
    
    # Example 1: Train model
    model_path = example_1_train_model()
    
    # Example 2: Inference
    example_2_inference(model_path)
    
    # Example 3: Simulation with ML
    example_3_simulation_with_ml(model_path)
    
    logger.info("\n===== All examples complete! =====")


if __name__ == "__main__":
    main()
