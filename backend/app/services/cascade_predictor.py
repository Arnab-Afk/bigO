"""
Cascade Predictor Service

Real-time cascade risk prediction using trained GNN models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
import torch

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from app.ml.models.cascade_classifier import CascadeClassifierGNN, GraphDataConverter
from app.ml.config import ml_config

logger = logging.getLogger(__name__)


class CascadePredictorService:
    """
    Service for predicting cascade risk in financial networks

    Uses trained CascadeClassifierGNN to classify network states and
    identify vulnerable subgraphs.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: torch device (auto-detects if None)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for cascade prediction. "
                "Install with: pip install torch-geometric"
            )

        # Setup device
        if device is None:
            if ml_config.ENABLE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Load model
        self.model = None
        self.model_loaded = False

        if model_path is None:
            model_path = ml_config.ML_MODELS_PATH / "cascade_classifier" / "best_model.pt"

        if model_path.exists():
            try:
                self._load_model(model_path)
                logger.info(f"Loaded cascade classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load cascade classifier: {e}")
        else:
            logger.warning(f"Cascade classifier not found at {model_path}")

    def _load_model(self, model_path: Path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with saved config
        config = checkpoint.get('config', {})
        self.model = CascadeClassifierGNN(
            node_feature_dim=config.get('node_feature_dim', 10),
            hidden_channels=config.get('hidden_channels', 64),
            num_layers=config.get('num_layers', 3),
            num_classes=config.get('num_classes', 3),
            dropout=config.get('dropout', 0.3),
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    def predict_cascade_risk(
        self,
        network: nx.DiGraph,
        agent_states: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify current network state for cascade risk

        Args:
            network: NetworkX DiGraph of financial network
            agent_states: Dictionary of agent states with metrics

        Returns:
            Dictionary with:
            - risk_level: int (0=none, 1=local, 2=systemic)
            - risk_label: str (no_cascade, local_cascade, systemic_cascade)
            - probability: List[float] (probability distribution over classes)
            - confidence: float (max probability)
        """
        if not self.model_loaded:
            # Fallback to heuristic if model not loaded
            return self._heuristic_prediction(network, agent_states)

        # Convert network to PyG Data
        node_features = self._extract_node_features(network, agent_states)
        graph_data = GraphDataConverter.networkx_to_pyg(
            network,
            node_features,
            label=0,  # Dummy label for inference
        )

        # Predict
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            predictions, probabilities = self.model.predict(
                graph_data.x,
                graph_data.edge_index,
                batch=None,
            )

        risk_level = predictions.item()
        probs = probabilities[0].cpu().numpy()
        confidence = float(np.max(probs))

        risk_labels = ['no_cascade', 'local_cascade', 'systemic_cascade']
        risk_label = risk_labels[risk_level]

        return {
            'risk_level': risk_level,
            'risk_label': risk_label,
            'probability': probs.tolist(),
            'confidence': confidence,
            'model_loaded': True,
        }

    def _heuristic_prediction(
        self,
        network: nx.DiGraph,
        agent_states: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Heuristic-based cascade prediction when model not available

        Uses simple rules based on agent health and network structure.
        """
        # Count unhealthy/dead agents
        num_unhealthy = sum(
            1 for state in agent_states.values()
            if state.get('health', 1.0) < 0.3 or not state.get('alive', True)
        )

        # Calculate average health
        alive_agents = [
            state for state in agent_states.values()
            if state.get('alive', True)
        ]
        if alive_agents:
            avg_health = np.mean([state.get('health', 1.0) for state in alive_agents])
        else:
            avg_health = 0.0

        # Classify based on heuristics
        if num_unhealthy == 0 and avg_health > 0.7:
            risk_level = 0
            probs = [0.8, 0.15, 0.05]
        elif num_unhealthy <= 3 and avg_health > 0.3:
            risk_level = 1
            probs = [0.2, 0.6, 0.2]
        else:
            risk_level = 2
            probs = [0.1, 0.2, 0.7]

        risk_labels = ['no_cascade', 'local_cascade', 'systemic_cascade']

        return {
            'risk_level': risk_level,
            'risk_label': risk_labels[risk_level],
            'probability': probs,
            'confidence': float(max(probs)),
            'model_loaded': False,
        }

    def identify_vulnerable_subgraphs(
        self,
        network: nx.DiGraph,
        agent_states: Dict[str, Any],
        health_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect weak clusters in the network

        Identifies subgraphs with high risk of cascading failures.

        Args:
            network: NetworkX DiGraph
            agent_states: Dictionary of agent states
            health_threshold: Threshold for considering nodes vulnerable

        Returns:
            List of vulnerable clusters with metadata
        """
        vulnerable_clusters = []

        # Find weakly connected components
        if network.is_directed():
            undirected = network.to_undirected()
        else:
            undirected = network

        # Get connected components
        components = list(nx.connected_components(undirected))

        for component in components:
            component_nodes = list(component)

            # Calculate cluster metrics
            component_states = {
                node_id: agent_states[node_id]
                for node_id in component_nodes
                if node_id in agent_states
            }

            if not component_states:
                continue

            # Average health in cluster
            avg_health = np.mean([
                state.get('health', 1.0)
                for state in component_states.values()
                if state.get('alive', True)
            ])

            # Count defaults
            num_defaults = sum(
                1 for state in component_states.values()
                if not state.get('alive', True)
            )

            # Identify as vulnerable if low health or has defaults
            if avg_health < health_threshold or num_defaults > 0:
                # Calculate centrality for nodes in cluster
                subgraph = network.subgraph(component_nodes)
                centrality = nx.degree_centrality(subgraph)

                # Find most central (critical) nodes
                critical_nodes = sorted(
                    centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                vulnerable_clusters.append({
                    'cluster_id': len(vulnerable_clusters),
                    'size': len(component_nodes),
                    'nodes': component_nodes,
                    'avg_health': float(avg_health),
                    'num_defaults': num_defaults,
                    'critical_nodes': [
                        {'node_id': node, 'centrality': centrality[node]}
                        for node, _ in critical_nodes
                    ],
                    'risk_score': float(1.0 - avg_health + 0.1 * num_defaults),
                })

        # Sort by risk score (descending)
        vulnerable_clusters.sort(key=lambda x: x['risk_score'], reverse=True)

        return vulnerable_clusters

    def compute_cascade_probability(
        self,
        network: nx.DiGraph,
        agent_states: Dict[str, Any],
        shock_node: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute probability of cascade for each node

        Uses PageRank-style propagation to estimate cascade spread probability.

        Args:
            network: NetworkX DiGraph
            agent_states: Dictionary of agent states
            shock_node: Initial shock node (if any)

        Returns:
            Dictionary mapping node_id to cascade probability
        """
        cascade_probs = {}

        # Initialize probabilities
        for node_id in network.nodes():
            if node_id in agent_states:
                health = agent_states[node_id].get('health', 1.0)
                # Base probability inversely related to health
                base_prob = max(0.0, 1.0 - health)
                cascade_probs[node_id] = base_prob
            else:
                cascade_probs[node_id] = 0.0

        # If shock node specified, boost its probability
        if shock_node and shock_node in cascade_probs:
            cascade_probs[shock_node] = min(1.0, cascade_probs[shock_node] + 0.5)

        # Propagate risk through network (iterative)
        for iteration in range(5):
            new_probs = cascade_probs.copy()

            for node_id in network.nodes():
                # Get predecessors (creditors who could transmit losses)
                predecessors = list(network.predecessors(node_id))

                if predecessors:
                    # Average risk from neighbors
                    neighbor_risk = np.mean([
                        cascade_probs.get(pred, 0.0)
                        for pred in predecessors
                    ])

                    # Combine own risk with neighbor risk
                    own_risk = cascade_probs[node_id]
                    combined_risk = 0.7 * own_risk + 0.3 * neighbor_risk

                    new_probs[node_id] = min(1.0, combined_risk)

            cascade_probs = new_probs

        return cascade_probs

    def _extract_node_features(
        self,
        network: nx.DiGraph,
        agent_states: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Extract node features for GNN input

        Creates feature vectors for each node based on agent state.

        Returns:
            Dictionary mapping node_id to feature array
        """
        node_features = {}

        for node_id in network.nodes():
            if node_id not in agent_states:
                # Use zero features for missing nodes
                features = np.zeros(10)
            else:
                state = agent_states[node_id]

                # Extract features (10-dimensional)
                features = np.array([
                    state.get('health', 1.0),
                    state.get('capital', 0.0) / 1e6,  # Normalize to millions
                    state.get('liquidity', 0.0) / 1e6,
                    state.get('npa_ratio', 0.0) / 100,
                    state.get('crar', 10.0) / 100,
                    state.get('risk_appetite', 0.5),
                    float(state.get('alive', True)),
                    network.in_degree(node_id) / 10.0,  # Normalize degree
                    network.out_degree(node_id) / 10.0,
                    len(list(network.neighbors(node_id))) / 10.0,  # Total degree
                ])

            node_features[node_id] = features

        return node_features

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model_loaded
