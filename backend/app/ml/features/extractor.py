"""
Feature Extractor for Institution Default Prediction

Extracts 20+ features from institution state, network metrics, and market signals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import UUID

import networkx as nx
import numpy as np

from app.engine.game_theory import AgentState
from app.engine.network import NetworkAnalyzer, NodeCentrality


@dataclass
class InstitutionFeatures:
    """Container for all features of a single institution"""
    
    # Institution ID
    institution_id: UUID
    
    # Financial features (6)
    capital_ratio: float
    liquidity_buffer: float
    leverage: float  # 1/capital_ratio
    credit_exposure: float
    risk_appetite: float
    stress_level: float
    
    # Network topology features (6)
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    in_degree: float
    out_degree: float
    
    # Market signals (4)
    default_probability_prior: float  # Current/prior estimate
    credit_spread: float
    volatility: float
    market_pressure: float
    
    # Neighborhood stress features (4)
    neighbor_avg_stress: float
    neighbor_max_stress: float
    neighbor_default_count: int
    neighbor_avg_capital_ratio: float
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input"""
        return np.array([
            # Financial (6)
            self.capital_ratio,
            self.liquidity_buffer,
            self.leverage,
            self.credit_exposure,
            self.risk_appetite,
            self.stress_level,
            # Network (6)
            self.degree_centrality,
            self.betweenness_centrality,
            self.eigenvector_centrality,
            self.pagerank,
            self.in_degree,
            self.out_degree,
            # Market signals (4)
            self.default_probability_prior,
            self.credit_spread,
            self.volatility,
            self.market_pressure,
            # Neighborhood (4)
            self.neighbor_avg_stress,
            self.neighbor_max_stress,
            float(self.neighbor_default_count),
            self.neighbor_avg_capital_ratio,
        ], dtype=np.float32)
    
    @property
    def feature_dim(self) -> int:
        """Total number of features"""
        return 20


class FeatureExtractor:
    """
    Extracts features from institution states and network structure
    
    Integrates with:
    - AgentState from game_theory.py
    - NetworkAnalyzer from network.py
    """
    
    def __init__(self, network_analyzer: Optional[NetworkAnalyzer] = None):
        """
        Args:
            network_analyzer: NetworkAnalyzer instance for computing centralities
        """
        self.network_analyzer = network_analyzer
    
    def extract_features(
        self,
        institution_id: UUID,
        agent_state: AgentState,
        network: nx.DiGraph,
        all_agent_states: Dict[UUID, AgentState],
        centralities: Optional[Dict[UUID, NodeCentrality]] = None,
        defaulted_institutions: Optional[set] = None,
    ) -> InstitutionFeatures:
        """
        Extract all features for a single institution
        
        Args:
            institution_id: Institution UUID
            agent_state: Current AgentState
            network: NetworkX graph of exposures
            all_agent_states: States of all institutions
            centralities: Pre-computed centrality metrics
            defaulted_institutions: Set of defaulted institution IDs
        
        Returns:
            InstitutionFeatures object with all 20+ features
        """
        defaulted = defaulted_institutions or set()
        
        # 1. Financial features (directly from AgentState)
        capital_ratio = agent_state.capital_ratio
        liquidity_buffer = agent_state.liquidity_buffer
        leverage = 1.0 / max(agent_state.capital_ratio, 0.01)
        credit_exposure = agent_state.credit_exposure
        risk_appetite = agent_state.risk_appetite
        stress_level = agent_state.stress_level
        
        # 2. Network topology features
        if centralities and institution_id in centralities:
            cent = centralities[institution_id]
            degree_centrality = cent.degree_centrality
            betweenness_centrality = cent.betweenness_centrality
            eigenvector_centrality = cent.eigenvector_centrality
            pagerank = cent.pagerank
        else:
            # Fallback: compute basic metrics
            degree_centrality = nx.degree_centrality(network).get(institution_id, 0.0)
            betweenness_centrality = 0.0
            eigenvector_centrality = 0.0
            pagerank = 0.0
        
        # In/out degree
        if institution_id in network:
            in_degree = network.in_degree(institution_id)
            out_degree = network.out_degree(institution_id)
            # Normalize by network size
            n = len(network)
            in_degree = in_degree / max(n - 1, 1)
            out_degree = out_degree / max(n - 1, 1)
        else:
            in_degree = 0.0
            out_degree = 0.0
        
        # 3. Market signals
        default_probability_prior = agent_state.default_probability
        
        # Credit spread (synthetic: based on default prob and stress)
        credit_spread = self._compute_credit_spread(
            default_probability_prior, stress_level
        )
        
        # Volatility (synthetic: based on stress and liquidity)
        volatility = self._compute_volatility(stress_level, liquidity_buffer)
        
        # Market pressure (synthetic: combination of factors)
        market_pressure = self._compute_market_pressure(
            stress_level, credit_exposure, capital_ratio
        )
        
        # 4. Neighborhood stress features
        neighbors = self._get_neighbors(network, institution_id)
        
        if neighbors:
            neighbor_states = [
                all_agent_states[nid] for nid in neighbors 
                if nid in all_agent_states
            ]
            
            if neighbor_states:
                neighbor_avg_stress = np.mean([s.stress_level for s in neighbor_states])
                neighbor_max_stress = np.max([s.stress_level for s in neighbor_states])
                neighbor_avg_capital_ratio = np.mean([s.capital_ratio for s in neighbor_states])
            else:
                neighbor_avg_stress = 0.0
                neighbor_max_stress = 0.0
                neighbor_avg_capital_ratio = 1.0
            
            neighbor_default_count = sum(1 for nid in neighbors if nid in defaulted)
        else:
            neighbor_avg_stress = 0.0
            neighbor_max_stress = 0.0
            neighbor_default_count = 0
            neighbor_avg_capital_ratio = 1.0
        
        return InstitutionFeatures(
            institution_id=institution_id,
            capital_ratio=capital_ratio,
            liquidity_buffer=liquidity_buffer,
            leverage=leverage,
            credit_exposure=credit_exposure,
            risk_appetite=risk_appetite,
            stress_level=stress_level,
            degree_centrality=degree_centrality,
            betweenness_centrality=betweenness_centrality,
            eigenvector_centrality=eigenvector_centrality,
            pagerank=pagerank,
            in_degree=in_degree,
            out_degree=out_degree,
            default_probability_prior=default_probability_prior,
            credit_spread=credit_spread,
            volatility=volatility,
            market_pressure=market_pressure,
            neighbor_avg_stress=neighbor_avg_stress,
            neighbor_max_stress=neighbor_max_stress,
            neighbor_default_count=neighbor_default_count,
            neighbor_avg_capital_ratio=neighbor_avg_capital_ratio,
        )
    
    def extract_batch_features(
        self,
        agent_states: Dict[UUID, AgentState],
        network: nx.DiGraph,
        defaulted_institutions: Optional[set] = None,
    ) -> Dict[UUID, InstitutionFeatures]:
        """
        Extract features for all institutions in batch
        
        Args:
            agent_states: Dictionary of all agent states
            network: NetworkX graph
            defaulted_institutions: Set of defaulted institutions
        
        Returns:
            Dictionary mapping institution_id to InstitutionFeatures
        """
        # Pre-compute centralities for efficiency
        centralities = {}
        if self.network_analyzer:
            try:
                centralities = self.network_analyzer.compute_all_centralities(network)
            except Exception:
                # Fallback if centrality computation fails
                pass
        
        features = {}
        for inst_id, agent_state in agent_states.items():
            features[inst_id] = self.extract_features(
                institution_id=inst_id,
                agent_state=agent_state,
                network=network,
                all_agent_states=agent_states,
                centralities=centralities,
                defaulted_institutions=defaulted_institutions,
            )
        
        return features
    
    def _get_neighbors(self, network: nx.DiGraph, node_id: UUID) -> List[UUID]:
        """Get all neighbors (predecessors + successors) of a node"""
        if node_id not in network:
            return []
        
        predecessors = set(network.predecessors(node_id))
        successors = set(network.successors(node_id))
        return list(predecessors | successors)
    
    def _compute_credit_spread(
        self, default_prob: float, stress: float
    ) -> float:
        """
        Compute synthetic credit spread based on default probability and stress
        
        Credit spread model: spread = base_rate + risk_premium
        """
        base_rate = 0.02  # 2% base
        risk_premium = default_prob * 10.0 + stress * 5.0
        return base_rate + risk_premium
    
    def _compute_volatility(
        self, stress: float, liquidity: float
    ) -> float:
        """
        Compute synthetic volatility based on stress and liquidity
        
        Higher stress and lower liquidity => higher volatility
        """
        # Inverse relationship with liquidity
        liquidity_factor = 1.0 - liquidity
        return (stress * 0.5 + liquidity_factor * 0.5)
    
    def _compute_market_pressure(
        self, stress: float, exposure: float, capital_ratio: float
    ) -> float:
        """
        Compute synthetic market pressure indicator
        
        Combines stress, exposure, and capital adequacy
        """
        exposure_normalized = min(exposure / 1000.0, 1.0)  # Normalize exposure
        capital_pressure = 1.0 - capital_ratio  # Lower capital = higher pressure
        
        return (stress * 0.4 + exposure_normalized * 0.3 + capital_pressure * 0.3)
