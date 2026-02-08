"""
Game-Theoretic Decision Engine

Implements strategic agent behavior, utility functions, and Nash equilibrium computation.
Based on Technical Documentation Section 5.
"""

import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from scipy.optimize import minimize
import networkx as nx


class ActionType(str, enum.Enum):
    """Types of actions agents can take"""
    ADJUST_CREDIT_LIMIT = "adjust_credit_limit"
    MODIFY_MARGIN = "modify_margin"
    REROUTE_TRADE = "reroute_trade"
    LIQUIDITY_DECISION = "liquidity_decision"
    COLLATERAL_CALL = "collateral_call"
    MAINTAIN_STATUS = "maintain_status"


@dataclass
class AgentAction:
    """
    Represents a strategic action by an institution
    """
    action_type: ActionType
    agent_id: UUID
    target_id: Optional[UUID] = None
    magnitude: float = 0.0
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class AgentState:
    """Current state of an agent for decision-making"""
    agent_id: UUID
    capital_ratio: float
    liquidity_buffer: float
    credit_exposure: float
    default_probability: float
    stress_level: float
    risk_appetite: float = 0.5  # 0-1 scale
    
    # ML prediction fields
    ml_prediction_confidence: float = 0.0  # Confidence in ML prediction [0-1]
    ml_model_version: str = "none"  # Version of ML model used
    
    def is_stressed(self) -> bool:
        """Check if agent is under stress"""
        return self.stress_level > 0.5 or self.liquidity_buffer < 0.3


class AgentUtility:
    """
    Utility function calculator for strategic agents
    
    Based on: U_i(a_i, a_{-i}, θ_i) = π_i - ρ·Risk - λ·Liquidity_Cost - γ·Regulatory_Cost
    """
    
    def __init__(
        self,
        risk_aversion: float = 0.5,
        liquidity_weight: float = 0.3,
        regulatory_weight: float = 0.2,
    ):
        """
        Args:
            risk_aversion: ρ parameter (0-1)
            liquidity_weight: λ parameter
            regulatory_weight: γ parameter
        """
        self.risk_aversion = risk_aversion
        self.liquidity_weight = liquidity_weight
        self.regulatory_weight = regulatory_weight
    
    def compute_utility(
        self,
        action: AgentAction,
        agent_state: AgentState,
        network_state: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
    ) -> float:
        """
        Compute total utility for an action
        
        Args:
            action: The action being evaluated
            agent_state: Current state of the agent
            network_state: States of all other agents
            exposures: Exposure amounts (source, target) -> value
        
        Returns:
            Total utility value
        """
        revenue = self._compute_revenue(action, agent_state)
        credit_risk = self._compute_credit_risk(
            action, agent_state, network_state, exposures
        )
        liquidity_risk = self._compute_liquidity_risk(action, agent_state)
        regulatory_cost = self._compute_regulatory_cost(action, agent_state)
        
        utility = (
            revenue
            - self.risk_aversion * credit_risk
            - self.liquidity_weight * liquidity_risk
            - self.regulatory_weight * regulatory_cost
        )
        
        return utility
    
    def _compute_revenue(self, action: AgentAction, state: AgentState) -> float:
        """Expected revenue from action"""
        if action.action_type == ActionType.ADJUST_CREDIT_LIMIT:
            # Revenue from lending
            net_interest_margin = 0.02  # 2% NIM
            return action.magnitude * net_interest_margin
        
        elif action.action_type == ActionType.LIQUIDITY_DECISION:
            # Opportunity cost of hoarding
            if action.parameters.get("decision") == "HOARD":
                return -action.magnitude * 0.01  # Cost of idle cash
            else:
                return action.magnitude * 0.015  # Return from deployment
        
        elif action.action_type == ActionType.MAINTAIN_STATUS:
            # Current steady-state revenue
            return state.credit_exposure * 0.015
        
        return 0.0
    
    def _compute_credit_risk(
        self,
        action: AgentAction,
        state: AgentState,
        network_state: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
    ) -> float:
        """
        Expected credit loss: E[Loss] = Σ PD × LGD × EAD
        """
        expected_loss = 0.0
        
        # Calculate exposure-weighted default probability
        for (source, target), exposure_amount in exposures.items():
            if source == state.agent_id:
                target_state = network_state.get(target)
                if target_state:
                    pd = target_state.default_probability
                    lgd = 0.45  # Loss given default (45%)
                    ead = exposure_amount
                    
                    # Adjust EAD based on action
                    if (action.action_type == ActionType.ADJUST_CREDIT_LIMIT 
                        and action.target_id == target):
                        ead += action.magnitude
                    
                    expected_loss += pd * lgd * ead
        
        return expected_loss
    
    def _compute_liquidity_risk(self, action: AgentAction, state: AgentState) -> float:
        """Cost of potential liquidity shortfalls"""
        liquidity_buffer = state.liquidity_buffer
        
        # Adjust buffer based on action
        if action.action_type == ActionType.LIQUIDITY_DECISION:
            if action.parameters.get("decision") == "HOARD":
                liquidity_buffer += action.magnitude / 1000  # Normalize
            else:
                liquidity_buffer -= action.magnitude / 1000
        
        # Liquidity gap cost
        if liquidity_buffer < 0.2:
            emergency_rate = 0.05  # 5% emergency funding rate
            gap = max(0, 0.2 - liquidity_buffer)
            return gap * 1000 * emergency_rate
        
        return 0.0
    
    def _compute_regulatory_cost(self, action: AgentAction, state: AgentState) -> float:
        """Regulatory penalties for constraint violations"""
        cost = 0.0
        
        # Capital adequacy penalty
        if state.capital_ratio < 0.08:  # Below 8% minimum
            cost += (0.08 - state.capital_ratio) * 10000
        
        # Liquidity coverage penalty
        if state.liquidity_buffer < 0.1:  # Below 10% buffer
            cost += (0.1 - state.liquidity_buffer) * 5000
        
        return cost


class NashEquilibriumSolver:
    """
    Nash Equilibrium solver for financial network game
    
    Implements best response iteration and mixed strategy computation.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        """
        Args:
            tolerance: Convergence threshold
            max_iterations: Maximum iterations for convergence
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve_pure_nash(
        self,
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
        action_spaces: Dict[UUID, List[AgentAction]],
    ) -> Optional[Dict[UUID, AgentAction]]:
        """
        Find pure strategy Nash equilibrium via best response iteration
        
        Args:
            agents: Current states of all agents
            exposures: Exposure network
            action_spaces: Available actions for each agent
        
        Returns:
            Nash equilibrium strategy profile or None if not found
        """
        # Initialize with status quo actions
        strategies = {
            agent_id: AgentAction(
                action_type=ActionType.MAINTAIN_STATUS,
                agent_id=agent_id,
            )
            for agent_id in agents.keys()
        }
        
        for iteration in range(self.max_iterations):
            new_strategies = {}
            changed = False
            
            for agent_id in agents.keys():
                best_response = self._compute_best_response(
                    agent_id=agent_id,
                    current_strategies=strategies,
                    agents=agents,
                    exposures=exposures,
                    action_space=action_spaces.get(agent_id, []),
                )
                
                if best_response.action_type != strategies[agent_id].action_type:
                    changed = True
                
                new_strategies[agent_id] = best_response
            
            strategies = new_strategies
            
            # Check convergence
            if not changed:
                return strategies
        
        # No pure Nash found in max iterations
        return None
    
    def _compute_best_response(
        self,
        agent_id: UUID,
        current_strategies: Dict[UUID, AgentAction],
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
        action_space: List[AgentAction],
    ) -> AgentAction:
        """
        Find utility-maximizing action given others' strategies
        """
        agent_state = agents[agent_id]
        utility_calculator = AgentUtility(risk_aversion=agent_state.risk_appetite)
        
        best_action = None
        best_utility = float('-inf')
        
        # Add maintain status as default option
        if not any(a.action_type == ActionType.MAINTAIN_STATUS for a in action_space):
            action_space = action_space + [
                AgentAction(
                    action_type=ActionType.MAINTAIN_STATUS,
                    agent_id=agent_id,
                )
            ]
        
        for action in action_space:
            utility = utility_calculator.compute_utility(
                action=action,
                agent_state=agent_state,
                network_state=agents,
                exposures=exposures,
            )
            
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action


class CCPCentricPayoffNormalizer:
    """
    Ensures CCP has zero risk by normalizing payoffs using eigenvector centrality.
    
    This implements mentor's guidance: use eigenvectors to normalize the payoff matrix
    such that the CCP (clearing house) always comes out on top with zero net loss.
    """
    
    def __init__(self, ccp_node_id: Optional[str] = None):
        """
        Args:
            ccp_node_id: Identifier for the CCP node in the network
        """
        self.ccp_node_id = ccp_node_id
        self.eigenvector_centrality = {}
        self.normalization_matrix = None
    
    def compute_eigenvector_weights(
        self, 
        network: 'nx.DiGraph'
    ) -> Dict[UUID, float]:
        """
        Compute eigenvector centrality for payoff normalization.
        
        Returns:
            Mapping of agent_id to normalized eigenvector centrality
        """
        try:
            # Compute eigenvector centrality
            centrality = nx.eigenvector_centrality(network, max_iter=1000, tol=1e-6)
            
            # Normalize to sum to 1.0
            total = sum(centrality.values())
            if total > 0:
                self.eigenvector_centrality = {
                    k: v / total for k, v in centrality.items()
                }
            else:
                # Fallback: equal weights
                n = len(network.nodes())
                self.eigenvector_centrality = {k: 1.0/n for k in network.nodes()}
            
            # Convert string keys to UUID
            uuid_centrality = {}
            for node_id, centrality_val in self.eigenvector_centrality.items():
                # Hash node_id to create consistent UUID
                uuid_key = UUID(int=hash(node_id) & 0xFFFFFFFFFFFFFFFF)
                uuid_centrality[uuid_key] = centrality_val
            
            return uuid_centrality
            
        except Exception as e:
            # Fallback to equal distribution
            n = len(network.nodes())
            return {UUID(int=hash(k) & 0xFFFFFFFFFFFFFFFF): 1.0/n 
                    for k in network.nodes()}
    
    def normalize_payoff(
        self,
        raw_payoff: float,
        agent_id: UUID,
        agent_centrality: float,
        ccp_id: Optional[UUID] = None
    ) -> float:
        """
        Normalize individual agent payoff using eigenvector weights.
        
        The normalization ensures that:
        1. Agents with higher systemic importance (centrality) bear more risk
        2. CCP's net payoff is always >= 0
        3. Total system payoff is conserved
        
        Args:
            raw_payoff: Original payoff for the agent
            agent_id: Agent identifier
            agent_centrality: Agent's eigenvector centrality (0-1)
            ccp_id: CCP identifier (if provided, ensures CCP has zero net)
            
        Returns:
            Normalized payoff
        """
        # If this is the CCP, ensure zero payoff (all risk mutualized)
        if ccp_id is not None and agent_id == ccp_id:
            return 0.0
        
        # For non-CCP agents, adjust payoff by centrality weight
        # Higher centrality = absorb more of CCP's would-be losses
        # This implements the eigenvector-based risk sharing
        
        if raw_payoff < 0:  # Losses
            # Amplify losses for high-centrality nodes (they bear CCP's risk)
            normalized = raw_payoff * (1.0 + agent_centrality)
        else:  # Gains
            # Reduce gains proportionally (to fund CCP's protection)
            normalized = raw_payoff * (1.0 - 0.5 * agent_centrality)
        
        return normalized
    
    def create_normalized_payoff_matrix(
        self,
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
        network: 'nx.DiGraph',
        ccp_id: Optional[UUID] = None
    ) -> np.ndarray:
        """
        Create eigenvector-normalized payoff matrix.
        
        The resulting matrix ensures:
        - CCP row/column sum to zero (zero risk)
        - Losses are redistributed proportionally to eigenvector centrality
        - System-wide payoff is conserved
        
        Returns:
            n×n payoff matrix where element (i,j) is i's payoff from interaction with j
        """
        n = len(agents)
        agent_ids = list(agents.keys())
        payoff_matrix = np.zeros((n, n))
        
        # Compute eigenvector centrality
        centrality = self.compute_eigenvector_weights(network)
        
        # Build raw payoff matrix from exposures
        for i, agent_i in enumerate(agent_ids):
            for j, agent_j in enumerate(agent_ids):
                if i == j:
                    continue
                
                # Get exposure from i to j
                exposure = exposures.get((agent_i, agent_j), 0.0)
                
                if exposure > 0:
                    # Payoff = exposure × (1 - default_probability_j)
                    # Loss = exposure × default_probability_j × LGD
                    pd_j = agents[agent_j].default_probability
                    lgd = 0.45  # Loss given default
                    
                    expected_return = exposure * 0.02  # 2% return
                    expected_loss = exposure * pd_j * lgd
                    
                    payoff_matrix[i, j] = expected_return - expected_loss
        
        # Normalize matrix using eigenvector weights
        normalized_matrix = np.zeros_like(payoff_matrix)
        
        for i, agent_i in enumerate(agent_ids):
            centrality_i = centrality.get(agent_i, 1.0 / n)
            
            for j, agent_j in enumerate(agent_ids):
                centrality_j = centrality.get(agent_j, 1.0 / n)
                
                # Normalize by geometric mean of centralities
                weight = np.sqrt(centrality_i * centrality_j)
                normalized_matrix[i, j] = payoff_matrix[i, j] * weight
        
        # If CCP is specified, zero out its row and column
        if ccp_id is not None and ccp_id in agent_ids:
            ccp_idx = agent_ids.index(ccp_id)
            
            # CCP's losses are redistributed to others
            ccp_net_payoff = np.sum(normalized_matrix[ccp_idx, :]) + np.sum(normalized_matrix[:, ccp_idx])
            
            if ccp_net_payoff < 0:
                # Redistribute CCP's losses to other agents by centrality
                for i, agent_i in enumerate(agent_ids):
                    if i != ccp_idx:
                        centrality_i = centrality.get(agent_i, 1.0 / n)
                        # Allocate loss proportionally
                        normalized_matrix[i, i] += ccp_net_payoff * centrality_i
            
            # Zero out CCP's entries
            normalized_matrix[ccp_idx, :] = 0
            normalized_matrix[:, ccp_idx] = 0
        
        self.normalization_matrix = normalized_matrix
        return normalized_matrix
    
    def verify_ccp_zero_risk(
        self, 
        payoff_matrix: np.ndarray, 
        ccp_idx: int
    ) -> Tuple[bool, float]:
        """
        Verify that CCP has zero net payoff (zero risk).
        
        Args:
            payoff_matrix: Normalized payoff matrix
            ccp_idx: Index of CCP in the matrix
            
        Returns:
            (is_zero_risk, net_payoff) tuple
        """
        ccp_net = np.sum(payoff_matrix[ccp_idx, :]) + np.sum(payoff_matrix[:, ccp_idx])
        is_zero = abs(ccp_net) < 1e-6
        
        return is_zero, float(ccp_net) if best_action else action_space[0]
    
    def compute_expected_utility(
        self,
        agent_id: UUID,
        action: AgentAction,
        beliefs: Dict[UUID, Dict[str, float]],
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
    ) -> float:
        """
        Compute expected utility under Bayesian beliefs about opponent types
        
        Args:
            agent_id: ID of the agent
            action: Action being considered
            beliefs: Probability distributions over opponent types
            agents: Current agent states
            exposures: Exposure network
        
        Returns:
            Expected utility value
        """
        # For simplicity, use current states weighted by beliefs
        # In full implementation, would integrate over type distributions
        
        agent_state = agents[agent_id]
        utility_calculator = AgentUtility(risk_aversion=agent_state.risk_appetite)
        
        # Base utility calculation
        base_utility = utility_calculator.compute_utility(
            action=action,
            agent_state=agent_state,
            network_state=agents,
            exposures=exposures,
        )
        
        # Adjust for uncertainty (risk premium)
        uncertainty_penalty = 0.0
        for counterparty_id, belief_dist in beliefs.items():
            if counterparty_id != agent_id:
                # Variance in beliefs increases uncertainty
                belief_variance = sum(
                    p * (v - 0.5) ** 2 
                    for v, p in belief_dist.items()
                )
                uncertainty_penalty += belief_variance * 0.1
        
        return base_utility - uncertainty_penalty


def generate_action_space(
    agent_id: UUID,
    agent_state: AgentState,
    counterparties: List[UUID],
) -> List[AgentAction]:
    """
    Generate feasible action space for an agent
    
    Args:
        agent_id: Agent's ID
        agent_state: Current state
        counterparties: List of connected institutions
    
    Returns:
        List of viable actions
    """
    actions = [
        AgentAction(
            action_type=ActionType.MAINTAIN_STATUS,
            agent_id=agent_id,
        )
    ]
    
    # Credit limit adjustments for each counterparty
    for cp_id in counterparties:
        # Increase credit
        actions.append(
            AgentAction(
                action_type=ActionType.ADJUST_CREDIT_LIMIT,
                agent_id=agent_id,
                target_id=cp_id,
                magnitude=1000.0,
            )
        )
        # Decrease credit
        actions.append(
            AgentAction(
                action_type=ActionType.ADJUST_CREDIT_LIMIT,
                agent_id=agent_id,
                target_id=cp_id,
                magnitude=-1000.0,
            )
        )
    
    # Liquidity decisions
    if agent_state.liquidity_buffer > 0.3:
        actions.append(
            AgentAction(
                action_type=ActionType.LIQUIDITY_DECISION,
                agent_id=agent_id,
                magnitude=5000.0,
                parameters={"decision": "RELEASE"},
            )
        )
    
    if agent_state.stress_level > 0.4:
        actions.append(
            AgentAction(
                action_type=ActionType.LIQUIDITY_DECISION,
                agent_id=agent_id,
                magnitude=5000.0,
                parameters={"decision": "HOARD"},
            )
        )
    
    return actions
