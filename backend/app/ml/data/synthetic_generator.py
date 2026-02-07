"""
Synthetic Training Data Generator

Generates training data from simulation runs for ML model training.
"""

import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import networkx as nx
import numpy as np

from app.engine.game_theory import AgentState
from app.engine.simulation import SimulationEngine, Shock
from app.ml.features.extractor import FeatureExtractor, InstitutionFeatures

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic training data from simulation runs
    
    Creates labeled datasets for ML training:
    - Default prediction: features + binary label (defaulted/not)
    - Cascade classification: graph structure + cascade type
    - Time series forecasting: sequences + future states
    """
    
    def __init__(
        self,
        network: Optional[nx.DiGraph] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        """
        Args:
            network: Financial network (creates synthetic if None)
            feature_extractor: FeatureExtractor instance
        """
        self.network = network or self._create_synthetic_network()
        self.feature_extractor = feature_extractor or FeatureExtractor()
    
    def generate_default_prediction_dataset(
        self,
        num_simulations: int = 100,
        timesteps_per_sim: int = 50,
        shock_probability: float = 0.3,
    ) -> Tuple[List[InstitutionFeatures], List[int]]:
        """
        Generate dataset for default prediction
        
        Args:
            num_simulations: Number of simulation runs
            timesteps_per_sim: Timesteps per simulation
            shock_probability: Probability of applying shock
        
        Returns:
            Tuple of (features_list, labels_list)
        """
        features_list = []
        labels_list = []
        
        logger.info(
            f"Generating default prediction dataset: "
            f"{num_simulations} simulations × {timesteps_per_sim} timesteps"
        )
        
        for sim_idx in range(num_simulations):
            # Create initial states
            initial_states = self._create_initial_states()
            
            # Create shocks
            shocks = []
            shock_timing = {}
            if np.random.random() < shock_probability:
                shock_timestep = np.random.randint(5, min(20, timesteps_per_sim))
                shock = self._create_random_shock()
                shocks.append(shock)
                shock_timing[shock_timestep] = [shock.shock_id]
            
            # Run simulation (without ML to avoid circular dependency)
            engine = SimulationEngine(
                network=self.network.copy(),
                max_timesteps=timesteps_per_sim,
                enable_ml=False,
            )
            
            try:
                sim_state = engine.run_simulation(
                    simulation_id=f"synthetic_{sim_idx}",
                    initial_states=initial_states,
                    shocks=shocks,
                    shock_timing=shock_timing,
                )
                
                # Extract features and labels from each timestep
                defaulted_institutions = set()
                
                for timestep_state in sim_state.timesteps:
                    # Update defaults
                    for default_event in timestep_state.defaults:
                        defaulted_institutions.add(default_event.institution_id)
                    
                    # Extract features for all institutions
                    for inst_id, agent_state in timestep_state.agent_states.items():
                        features = self.feature_extractor.extract_features(
                            institution_id=inst_id,
                            agent_state=agent_state,
                            network=self.network,
                            all_agent_states=timestep_state.agent_states,
                            defaulted_institutions=defaulted_institutions,
                        )
                        
                        # Label: will this institution default in next N timesteps?
                        # For simplicity: 1 if already defaulted
                        label = 1 if inst_id in defaulted_institutions else 0
                        
                        features_list.append(features)
                        labels_list.append(label)
                
            except Exception as e:
                logger.warning(f"Simulation {sim_idx} failed: {e}")
                continue
            
            if (sim_idx + 1) % 10 == 0:
                logger.info(f"Generated {sim_idx + 1}/{num_simulations} simulations")
        
        logger.info(
            f"Generated {len(features_list)} samples "
            f"(Defaults: {sum(labels_list)}, Non-defaults: {len(labels_list) - sum(labels_list)})"
        )
        
        return features_list, labels_list
    
    def _create_synthetic_network(self, num_institutions: int = 20) -> nx.DiGraph:
        """Create a synthetic financial network"""
        G = nx.DiGraph()
        
        # Add institutions
        institutions = [uuid4() for _ in range(num_institutions)]
        for inst_id in institutions:
            G.add_node(
                inst_id,
                institution_type="bank",
                total_assets=np.random.uniform(1000, 10000),
            )
        
        # Add exposures (scale-free-like network)
        for i, source in enumerate(institutions):
            # Each institution has 2-5 counterparties
            num_counterparties = np.random.randint(2, 6)
            targets = np.random.choice(
                institutions,
                size=min(num_counterparties, len(institutions) - 1),
                replace=False,
            )
            
            for target in targets:
                if source != target:
                    G.add_edge(
                        source,
                        target,
                        exposure_magnitude=np.random.uniform(10, 500),
                        exposure_type="counterparty_credit",
                    )
        
        logger.info(
            f"Created synthetic network: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        
        return G
    
    def _create_initial_states(self) -> Dict[UUID, AgentState]:
        """Create random initial agent states"""
        states = {}
        
        for node_id in self.network.nodes():
            states[node_id] = AgentState(
                agent_id=node_id,
                capital_ratio=np.random.uniform(0.08, 0.20),
                liquidity_buffer=np.random.uniform(0.2, 0.8),
                credit_exposure=np.random.uniform(50, 500),
                default_probability=np.random.uniform(0.001, 0.05),
                stress_level=np.random.uniform(0.0, 0.4),
                risk_appetite=np.random.uniform(0.3, 0.7),
            )
        
        return states
    
    def _create_random_shock(self) -> Shock:
        """Create a random shock"""
        shock_types = [
            "institution_default",
            "liquidity_freeze",
            "market_volatility",
            "credit_downgrade",
        ]
        
        shock_type = np.random.choice(shock_types)
        num_targets = np.random.randint(1, min(4, len(self.network.nodes())))
        target_institutions = list(
            np.random.choice(list(self.network.nodes()), size=num_targets, replace=False)
        )
        
        if shock_type == "institution_default":
            magnitude = 1.0
        elif shock_type == "liquidity_freeze":
            magnitude = np.random.uniform(0.3, 0.7)
        elif shock_type == "market_volatility":
            magnitude = np.random.uniform(0.2, 0.5)
        else:  # credit_downgrade
            magnitude = np.random.uniform(0.1, 0.3)
        
        return Shock(
            shock_id=f"shock_{uuid4().hex[:8]}",
            shock_type=shock_type,
            target_institutions=target_institutions,
            magnitude=magnitude,
        )
    
    def generate_balanced_dataset(
        self,
        target_samples: int = 10000,
        default_ratio: float = 0.3,
    ) -> Tuple[List[InstitutionFeatures], List[int]]:
        """
        Generate balanced dataset with specified default ratio
        
        Args:
            target_samples: Target number of samples
            default_ratio: Desired ratio of default samples
        
        Returns:
            Balanced features and labels
        """
        features_list = []
        labels_list = []
        
        # Generate until we have enough samples
        num_simulations_per_batch = 10
        
        while len(features_list) < target_samples:
            batch_features, batch_labels = self.generate_default_prediction_dataset(
                num_simulations=num_simulations_per_batch,
                timesteps_per_sim=30,
                shock_probability=0.5,  # Higher shock probability for more defaults
            )
            
            features_list.extend(batch_features)
            labels_list.extend(batch_labels)
        
        # Balance the dataset
        default_indices = [i for i, label in enumerate(labels_list) if label == 1]
        non_default_indices = [i for i, label in enumerate(labels_list) if label == 0]
        
        target_defaults = int(target_samples * default_ratio)
        target_non_defaults = target_samples - target_defaults
        
        # Sample with replacement if needed
        selected_default_indices = np.random.choice(
            default_indices,
            size=min(target_defaults, len(default_indices)),
            replace=len(default_indices) < target_defaults,
        )
        selected_non_default_indices = np.random.choice(
            non_default_indices,
            size=min(target_non_defaults, len(non_default_indices)),
            replace=len(non_default_indices) < target_non_defaults,
        )
        
        selected_indices = np.concatenate([
            selected_default_indices,
            selected_non_default_indices,
        ])
        np.random.shuffle(selected_indices)
        
        balanced_features = [features_list[i] for i in selected_indices]
        balanced_labels = [labels_list[i] for i in selected_indices]
        
        logger.info(
            f"Balanced dataset: {len(balanced_features)} samples "
            f"(Defaults: {sum(balanced_labels)}, "
            f"Non-defaults: {len(balanced_labels) - sum(balanced_labels)})"
        )
        
        return balanced_features, balanced_labels
