"""
Contagion Propagation Engine

Implements multiple contagion mechanisms for cascade simulation.
Based on Technical Documentation Section 7.2.2.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from uuid import UUID

import networkx as nx
import numpy as np


class ContagionMechanism(str, enum.Enum):
    """Types of contagion mechanisms"""
    CREDIT_CONTAGION = "credit_contagion"
    LIQUIDITY_SPIRAL = "liquidity_spiral"
    MARGIN_SPIRAL = "margin_spiral"
    INFORMATION_CONTAGION = "information_contagion"
    FIRE_SALE = "fire_sale"


@dataclass
class DefaultEvent:
    """Records a default event"""
    institution_id: UUID
    timestep: int
    cause: str
    losses_inflicted: float = 0.0
    triggered_by: List[UUID] = field(default_factory=list)


@dataclass
class CascadeRound:
    """Records one round of cascade propagation"""
    round_number: int
    defaults: List[DefaultEvent]
    total_losses: float
    affected_institutions: Set[UUID]


@dataclass
class PropagationState:
    """State during contagion propagation"""
    capital_levels: Dict[UUID, float]
    liquidity_levels: Dict[UUID, float]
    stress_levels: Dict[UUID, float]
    defaulted: Set[UUID]
    asset_prices: Dict[str, float] = field(default_factory=lambda: {"default": 1.0})


class ContagionPropagator:
    """
    Contagion propagation engine
    
    Simulates how defaults and stress propagate through the financial network
    via multiple transmission channels.
    """
    
    def __init__(
        self,
        network: nx.DiGraph,
        fire_sale_impact: float = 0.1,
        margin_sensitivity: float = 0.2,
        information_weight: float = 0.15,
    ):
        """
        Args:
            network: Financial network graph
            fire_sale_impact: Price impact of fire sales (α)
            margin_sensitivity: Sensitivity to volatility increases (β)
            information_weight: Weight of information contagion
        """
        self.network = network
        self.fire_sale_impact = fire_sale_impact
        self.margin_sensitivity = margin_sensitivity
        self.information_weight = information_weight
    
    def propagate_shock(
        self,
        initial_state: PropagationState,
        shocked_institutions: List[UUID],
        max_rounds: int = 10,
    ) -> Tuple[PropagationState, List[CascadeRound]]:
        """
        Propagate an initial shock through the network
        
        Args:
            initial_state: Starting state of all institutions
            shocked_institutions: Initially affected institutions
            max_rounds: Maximum cascade rounds
        
        Returns:
            Tuple of (final_state, cascade_history)
        """
        state = initial_state
        cascade_history = []
        
        # Track new defaults each round
        newly_defaulted = set(shocked_institutions)
        state.defaulted.update(newly_defaulted)
        
        for round_num in range(max_rounds):
            if not newly_defaulted:
                break
            
            round_defaults = []
            round_losses = 0.0
            affected = set()
            
            # Apply each contagion mechanism
            state = self._apply_credit_contagion(state, newly_defaulted)
            state = self._apply_liquidity_spiral(state, newly_defaulted)
            state = self._apply_margin_spiral(state, newly_defaulted)
            state = self._apply_information_contagion(state, newly_defaulted)
            
            # Detect new defaults
            new_defaults_this_round = self._detect_defaults(state)
            
            for inst_id in new_defaults_this_round:
                default_event = DefaultEvent(
                    institution_id=inst_id,
                    timestep=round_num,
                    cause=self._determine_default_cause(state, inst_id),
                    triggered_by=list(newly_defaulted),
                )
                round_defaults.append(default_event)
                
                # Calculate losses inflicted
                losses = self._calculate_losses_inflicted(inst_id)
                default_event.losses_inflicted = losses
                round_losses += losses
                affected.add(inst_id)
            
            # Record cascade round
            cascade_history.append(
                CascadeRound(
                    round_number=round_num,
                    defaults=round_defaults,
                    total_losses=round_losses,
                    affected_institutions=affected,
                )
            )
            
            # Update for next iteration
            newly_defaulted = new_defaults_this_round
            state.defaulted.update(newly_defaulted)
        
        return state, cascade_history
    
    def _apply_credit_contagion(
        self,
        state: PropagationState,
        defaulted: Set[UUID]
    ) -> PropagationState:
        """
        Credit contagion: Loss transmitted through counterparty exposures
        
        Formula: Loss_j = Σ_i LGD_i × Exposure_{ij} × 1_{default_i}
        """
        for defaulted_inst in defaulted:
            if defaulted_inst not in self.network:
                continue
            
            # Find all creditors (predecessors in the graph)
            for creditor in self.network.predecessors(defaulted_inst):
                if creditor in state.defaulted:
                    continue
                
                # Get exposure data
                edge_data = self.network[creditor][defaulted_inst]
                exposure = edge_data.get('exposure_magnitude', 0.0)
                recovery_rate = edge_data.get('recovery_rate', 0.55)
                lgd = 1.0 - recovery_rate
                
                # Apply loss
                loss = lgd * exposure
                state.capital_levels[creditor] = state.capital_levels.get(
                    creditor, 10000.0
                ) - loss
                
                # Increase stress
                stress_increase = loss / 10000.0  # Normalized
                state.stress_levels[creditor] = min(
                    1.0,
                    state.stress_levels.get(creditor, 0.0) + stress_increase
                )
        
        return state
    
    def _apply_liquidity_spiral(
        self,
        state: PropagationState,
        defaulted: Set[UUID]
    ) -> PropagationState:
        """
        Liquidity spiral: Fire sales depress asset prices
        
        Formula: P_{t+1} = P_t × (1 - α × ΔS_t)
        """
        # Calculate total fire sales
        total_sales = sum(
            state.capital_levels.get(inst_id, 0.0)
            for inst_id in defaulted
        )
        
        if total_sales > 0:
            # Update asset prices
            for asset in state.asset_prices:
                price_impact = self.fire_sale_impact * (total_sales / 1000000)
                new_price = state.asset_prices[asset] * (1 - price_impact)
                state.asset_prices[asset] = max(0.1, new_price)  # Floor at 10%
            
            # Reduce liquidity for all institutions due to price drops
            for inst_id in state.liquidity_levels:
                if inst_id not in state.defaulted:
                    liquidity_loss = 0.05 * price_impact
                    state.liquidity_levels[inst_id] = max(
                        0.0,
                        state.liquidity_levels[inst_id] - liquidity_loss
                    )
        
        return state
    
    def _apply_margin_spiral(
        self,
        state: PropagationState,
        defaulted: Set[UUID]
    ) -> PropagationState:
        """
        Margin spiral: Increased volatility leads to margin calls
        
        Formula: Margin_t = VaR(Position_t) × (1 + β × Volatility_t)
        """
        # Volatility increases with number of defaults
        volatility_increase = len(defaulted) * 0.05
        
        for inst_id in state.capital_levels:
            if inst_id not in state.defaulted:
                # Calculate additional margin requirement
                base_margin = state.capital_levels[inst_id] * 0.1
                additional_margin = (
                    base_margin * self.margin_sensitivity * volatility_increase
                )
                
                # Reduce liquidity to meet margin
                state.liquidity_levels[inst_id] = max(
                    0.0,
                    state.liquidity_levels.get(inst_id, 1.0) - 
                    (additional_margin / state.capital_levels[inst_id])
                )
        
        return state
    
    def _apply_information_contagion(
        self,
        state: PropagationState,
        defaulted: Set[UUID]
    ) -> PropagationState:
        """
        Information contagion: Belief updates causing coordinated actions
        
        Formula: P(default_j | default_i) > P(default_j)
        """
        for defaulted_inst in defaulted:
            if defaulted_inst not in self.network:
                continue
            
            # Increase stress for connected institutions
            for neighbor in self.network.neighbors(defaulted_inst):
                if neighbor not in state.defaulted:
                    # Information contagion increases stress
                    edge_data = self.network[defaulted_inst][neighbor]
                    contagion_prob = edge_data.get('contagion_probability', 0.0)
                    
                    stress_increase = self.information_weight * contagion_prob
                    state.stress_levels[neighbor] = min(
                        1.0,
                        state.stress_levels.get(neighbor, 0.0) + stress_increase
                    )
        
        return state
    
    def _detect_defaults(self, state: PropagationState) -> Set[UUID]:
        """
        Detect institutions that have newly defaulted
        
        Criteria:
        - Capital ratio < 0 (insolvency)
        - Liquidity buffer < 0.05 (illiquidity)
        - Stress level > 0.95 (panic)
        """
        new_defaults = set()
        
        for inst_id in state.capital_levels:
            if inst_id in state.defaulted:
                continue
            
            capital = state.capital_levels.get(inst_id, 0.0)
            liquidity = state.liquidity_levels.get(inst_id, 1.0)
            stress = state.stress_levels.get(inst_id, 0.0)
            
            # Check default conditions
            if capital <= 0:
                new_defaults.add(inst_id)
            elif liquidity < 0.05:
                new_defaults.add(inst_id)
            elif stress > 0.95:
                new_defaults.add(inst_id)
        
        return new_defaults
    
    def _determine_default_cause(self, state: PropagationState, inst_id: UUID) -> str:
        """Determine the primary cause of default"""
        capital = state.capital_levels.get(inst_id, 0.0)
        liquidity = state.liquidity_levels.get(inst_id, 1.0)
        stress = state.stress_levels.get(inst_id, 0.0)
        
        if capital <= 0:
            return "insolvency"
        elif liquidity < 0.05:
            return "illiquidity"
        elif stress > 0.95:
            return "panic"
        else:
            return "unknown"
    
    def _calculate_losses_inflicted(self, inst_id: UUID) -> float:
        """Calculate total losses this institution will inflict on others"""
        total_loss = 0.0
        
        if inst_id in self.network:
            for creditor in self.network.predecessors(inst_id):
                edge_data = self.network[creditor][inst_id]
                exposure = edge_data.get('exposure_magnitude', 0.0)
                recovery_rate = edge_data.get('recovery_rate', 0.55)
                lgd = 1.0 - recovery_rate
                total_loss += lgd * exposure
        
        return total_loss
    
    def compute_contagion_matrix(self) -> np.ndarray:
        """
        Compute contagion matrix showing direct transmission probabilities
        
        Returns:
            Matrix C where C[i,j] = probability that i's default causes j's default
        """
        nodes = list(self.network.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        for i, source in enumerate(nodes):
            if source in self.network:
                for target in self.network.successors(source):
                    j = node_to_idx[target]
                    edge_data = self.network[source][target]
                    matrix[i, j] = edge_data.get('contagion_probability', 0.0)
        
        return matrix
    
    def estimate_cascade_size(
        self,
        shocked_institutions: List[UUID],
        monte_carlo_runs: int = 1000
    ) -> Dict[str, float]:
        """
        Estimate expected cascade size via Monte Carlo simulation
        
        Args:
            shocked_institutions: Initially shocked institutions
            monte_carlo_runs: Number of simulation runs
        
        Returns:
            Dictionary with statistics (mean, std, percentiles)
        """
        cascade_sizes = []
        
        for _ in range(monte_carlo_runs):
            # Create random initial state
            initial_state = self._create_random_state()
            
            # Run propagation
            final_state, _ = self.propagate_shock(
                initial_state=initial_state,
                shocked_institutions=shocked_institutions,
                max_rounds=10
            )
            
            cascade_sizes.append(len(final_state.defaulted))
        
        cascade_array = np.array(cascade_sizes)
        
        return {
            "mean": float(np.mean(cascade_array)),
            "std": float(np.std(cascade_array)),
            "min": float(np.min(cascade_array)),
            "max": float(np.max(cascade_array)),
            "p50": float(np.percentile(cascade_array, 50)),
            "p95": float(np.percentile(cascade_array, 95)),
            "p99": float(np.percentile(cascade_array, 99)),
        }
    
    def _create_random_state(self) -> PropagationState:
        """Create random initial state for Monte Carlo"""
        nodes = list(self.network.nodes())
        
        return PropagationState(
            capital_levels={
                node: float(np.random.uniform(5000, 15000))
                for node in nodes
            },
            liquidity_levels={
                node: float(np.random.uniform(0.3, 1.0))
                for node in nodes
            },
            stress_levels={
                node: float(np.random.uniform(0.0, 0.3))
                for node in nodes
            },
            defaulted=set(),
        )
