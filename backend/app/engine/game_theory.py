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

from app.engine.nash_solvers import (
    LemkeHowsonSolver,
    SupportEnumerationSolver,
    CorrelatedEquilibriumSolver,
    MixedStrategy,
    MixedNashEquilibrium,
)


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


@dataclass
class PayoffComponents:
    """Decomposed payoff for a single agent at a single timestep"""
    agent_id: UUID
    timestep: int
    total_utility: float
    revenue: float
    credit_risk_cost: float
    liquidity_risk_cost: float
    regulatory_cost: float
    action_taken: str
    target_id: Optional[UUID] = None

    def to_dict(self) -> Dict:
        return {
            "agent_id": str(self.agent_id),
            "timestep": self.timestep,
            "total_utility": self.total_utility,
            "revenue": self.revenue,
            "credit_risk_cost": self.credit_risk_cost,
            "liquidity_risk_cost": self.liquidity_risk_cost,
            "regulatory_cost": self.regulatory_cost,
            "action_taken": self.action_taken,
            "target_id": str(self.target_id) if self.target_id else None,
        }


@dataclass
class PayoffMatrixEntry:
    """A single cell in the pairwise payoff matrix"""
    agent_i_id: UUID
    agent_j_id: UUID
    agent_i_action: str
    agent_j_action: str
    agent_i_payoff: float
    agent_j_payoff: float


@dataclass
class PayoffMatrix:
    """Complete pairwise payoff matrix for a pair of agents"""
    agent_i_id: UUID
    agent_j_id: UUID
    entries: List[PayoffMatrixEntry]
    nash_equilibria: List[Tuple[str, str]]

    def to_dict(self) -> Dict:
        """Serialize to dict for JSONB storage"""
        matrix = {}
        for entry in self.entries:
            key = f"{entry.agent_i_action}|{entry.agent_j_action}"
            matrix[key] = {
                "agent_i_payoff": entry.agent_i_payoff,
                "agent_j_payoff": entry.agent_j_payoff,
            }
        return {
            "agent_i": str(self.agent_i_id),
            "agent_j": str(self.agent_j_id),
            "matrix": matrix,
            "nash_equilibria": [
                {"action_i": ne[0], "action_j": ne[1]}
                for ne in self.nash_equilibria
            ],
        }


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

    def compute_utility_components(
        self,
        action: AgentAction,
        agent_state: AgentState,
        network_state: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
        timestep: int = 0,
    ) -> "PayoffComponents":
        """
        Compute utility with full component decomposition.

        Returns PayoffComponents with the breakdown instead of a scalar.
        """
        revenue = self._compute_revenue(action, agent_state)
        credit_risk = self._compute_credit_risk(
            action, agent_state, network_state, exposures
        )
        liquidity_risk = self._compute_liquidity_risk(action, agent_state)
        regulatory_cost = self._compute_regulatory_cost(action, agent_state)

        total = (
            revenue
            - self.risk_aversion * credit_risk
            - self.liquidity_weight * liquidity_risk
            - self.regulatory_weight * regulatory_cost
        )

        action_name = (
            action.action_type.value
            if isinstance(action.action_type, ActionType)
            else str(action.action_type)
        )

        return PayoffComponents(
            agent_id=agent_state.agent_id,
            timestep=timestep,
            total_utility=total,
            revenue=revenue,
            credit_risk_cost=self.risk_aversion * credit_risk,
            liquidity_risk_cost=self.liquidity_weight * liquidity_risk,
            regulatory_cost=self.regulatory_weight * regulatory_cost,
            action_taken=action_name,
            target_id=action.target_id,
        )

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

        elif action.action_type == ActionType.MODIFY_MARGIN:
            # Tightening margins: reduces risk but costs opportunity
            if action.magnitude > 0:  # Tighten
                return -action.magnitude * 0.005
            else:  # Loosen
                return abs(action.magnitude) * 0.008

        elif action.action_type == ActionType.REROUTE_TRADE:
            # Transaction cost of rerouting
            return -abs(action.magnitude) * 0.003

        elif action.action_type == ActionType.COLLATERAL_CALL:
            # Small revenue from collateral interest
            return action.magnitude * 0.002

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

        # LGD varies by exposure type
        lgd_map = {
            "interbank_lending": 0.45,
            "derivatives": 0.60,
            "repo": 0.20,
            "securities_lending": 0.30,
            "collateral": 0.15,
            "credit_line": 0.55,
            "clearing_margin": 0.25,
            "settlement": 0.35,
            "trade_finance": 0.50,
        }

        # Calculate exposure-weighted default probability
        for (source, target), exposure_data in exposures.items():
            if source == state.agent_id:
                target_state = network_state.get(target)
                if target_state:
                    pd = target_state.default_probability

                    # Extract exposure amount and type-aware LGD
                    if isinstance(exposure_data, dict):
                        ead = exposure_data.get("magnitude", 0.0)
                        lgd = lgd_map.get(exposure_data.get("type", ""), 0.45)
                    else:
                        ead = float(exposure_data)
                        lgd = 0.45  # Default LGD

                    # Adjust EAD based on action
                    if (action.action_type == ActionType.ADJUST_CREDIT_LIMIT
                        and action.target_id == target):
                        ead += action.magnitude

                    if (action.action_type == ActionType.COLLATERAL_CALL
                        and action.target_id == target):
                        ead = max(0, ead - action.magnitude * 0.5)

                    if (action.action_type == ActionType.MODIFY_MARGIN
                        and action.target_id == target
                        and action.magnitude > 0):
                        ead *= 0.9  # 10% reduction from tighter margins

                    if (action.action_type == ActionType.REROUTE_TRADE
                        and action.target_id == target):
                        ead = max(0, ead - abs(action.magnitude))

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
        beliefs: Optional[Dict[UUID, Dict[str, float]]] = None,
    ) -> Optional[Dict[UUID, AgentAction]]:
        """
        Find pure strategy Nash equilibrium via best response iteration

        Args:
            agents: Current states of all agents
            exposures: Exposure network
            action_spaces: Available actions for each agent
            beliefs: Bayesian belief distributions over counterparty states

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
                    beliefs=beliefs,
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
        beliefs: Optional[Dict[UUID, Dict[str, float]]] = None,
    ) -> AgentAction:
        """
        Find utility-maximizing action given others' strategies and beliefs
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
            if beliefs is not None:
                # Use belief-weighted expected utility (incomplete information)
                utility = self.compute_expected_utility(
                    agent_id=agent_id,
                    action=action,
                    beliefs=beliefs,
                    agents=agents,
                    exposures=exposures,
                )
            else:
                utility = utility_calculator.compute_utility(
                    action=action,
                    agent_state=agent_state,
                    network_state=agents,
                    exposures=exposures,
                )

            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action if best_action else action_space[0]

    def compute_expected_utility(
        self,
        agent_id: UUID,
        action: AgentAction,
        beliefs: Dict[UUID, Dict[str, float]],
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
    ) -> float:
        """
        Compute expected utility under Bayesian beliefs about opponent types.

        Integrates base utility with uncertainty penalty from belief variance,
        capturing the risk premium agents demand under incomplete information.
        """
        agent_state = agents[agent_id]
        utility_calculator = AgentUtility(risk_aversion=agent_state.risk_appetite)

        # Base utility calculation
        base_utility = utility_calculator.compute_utility(
            action=action,
            agent_state=agent_state,
            network_state=agents,
            exposures=exposures,
        )

        # Adjust for uncertainty (risk premium from incomplete information)
        uncertainty_penalty = 0.0
        # Map belief states to default probability multipliers
        state_pd_map = {
            "healthy": 0.01,
            "stressed": 0.05,
            "distressed": 0.20,
            "defaulted": 1.00,
        }

        for counterparty_id, belief_dist in beliefs.items():
            if counterparty_id == agent_id:
                continue

            if isinstance(belief_dist, dict):
                probs = belief_dist
            elif hasattr(belief_dist, 'beliefs'):
                probs = belief_dist.beliefs
            else:
                continue

            # Expected default probability from beliefs
            expected_pd = sum(
                probs.get(state, 0.0) * pd
                for state, pd in state_pd_map.items()
            )

            # Variance in beliefs increases uncertainty
            mean_prob = sum(probs.values()) / max(len(probs), 1)
            belief_variance = sum(
                (p - mean_prob) ** 2 for p in probs.values()
            ) / max(len(probs), 1)

            # Uncertainty penalty: higher variance = more cautious
            uncertainty_penalty += belief_variance * 0.1

            # Additional penalty if expected PD is high
            uncertainty_penalty += expected_pd * 0.05

        return base_utility - uncertainty_penalty

    def compute_pairwise_payoff_matrix(
        self,
        agent_i_id: UUID,
        agent_j_id: UUID,
        agents: Dict[UUID, AgentState],
        exposures: Dict[Tuple[UUID, UUID], float],
        action_space_i: List[AgentAction],
        action_space_j: List[AgentAction],
    ) -> "PayoffMatrix":
        """
        Compute the complete pairwise payoff matrix between two agents.

        For each (action_i, action_j) pair, computes both agents' utilities.
        Also identifies pure-strategy Nash equilibria in this 2-player sub-game.
        """
        entries = []
        state_i = agents[agent_i_id]
        state_j = agents[agent_j_id]
        utility_i = AgentUtility(risk_aversion=state_i.risk_appetite)
        utility_j = AgentUtility(risk_aversion=state_j.risk_appetite)

        payoff_table_i: Dict[Tuple[str, str], float] = {}
        payoff_table_j: Dict[Tuple[str, str], float] = {}

        def _action_key(action: AgentAction) -> str:
            key = (
                action.action_type.value
                if isinstance(action.action_type, ActionType)
                else str(action.action_type)
            )
            if action.target_id:
                key += f"_{str(action.target_id)[:8]}"
            if action.parameters and action.parameters.get("decision"):
                key += f"_{action.parameters['decision']}"
            return key

        for action_i in action_space_i:
            for action_j in action_space_j:
                ui = utility_i.compute_utility(action_i, state_i, agents, exposures)
                uj = utility_j.compute_utility(action_j, state_j, agents, exposures)

                ai_key = _action_key(action_i)
                aj_key = _action_key(action_j)

                entries.append(PayoffMatrixEntry(
                    agent_i_id=agent_i_id,
                    agent_j_id=agent_j_id,
                    agent_i_action=ai_key,
                    agent_j_action=aj_key,
                    agent_i_payoff=ui,
                    agent_j_payoff=uj,
                ))
                payoff_table_i[(ai_key, aj_key)] = ui
                payoff_table_j[(ai_key, aj_key)] = uj

        # Find Nash equilibria in the 2-player matrix
        nash_eq = self._find_pure_nash_2player(payoff_table_i, payoff_table_j)

        return PayoffMatrix(
            agent_i_id=agent_i_id,
            agent_j_id=agent_j_id,
            entries=entries,
            nash_equilibria=nash_eq,
        )

    def _find_pure_nash_2player(
        self,
        table_i: Dict[Tuple[str, str], float],
        table_j: Dict[Tuple[str, str], float],
    ) -> List[Tuple[str, str]]:
        """Find pure-strategy Nash equilibria in a 2-player normal-form game"""
        action_keys_i = list(set(k[0] for k in table_i.keys()))
        action_keys_j = list(set(k[1] for k in table_i.keys()))

        nash = []
        for ai in action_keys_i:
            for aj in action_keys_j:
                # Check if ai is best response to aj for player i
                i_payoffs_given_aj = {
                    a: table_i.get((a, aj), float('-inf'))
                    for a in action_keys_i
                }
                best_i = max(i_payoffs_given_aj.values())

                # Check if aj is best response to ai for player j
                j_payoffs_given_ai = {
                    a: table_j.get((ai, a), float('-inf'))
                    for a in action_keys_j
                }
                best_j = max(j_payoffs_given_ai.values())

                current_i = table_i.get((ai, aj), float('-inf'))
                current_j = table_j.get((ai, aj), float('-inf'))

                if (abs(current_i - best_i) < self.tolerance and
                        abs(current_j - best_j) < self.tolerance):
                    nash.append((ai, aj))

        return nash


class MixedStrategyNashSolver:
    """
    Mixed Strategy Nash Equilibrium Solver

    Extends pure Nash solver with mixed strategy computation.
    Supports:
    - 2-player games via Lemke-Howson
    - N-player games via support enumeration
    - Correlated equilibrium computation
    """

    def __init__(self, tolerance: float = 1e-6, max_support_size: int = 3):
        """
        Args:
            tolerance: Convergence threshold
            max_support_size: Maximum support size for enumeration
        """
        self.tolerance = tolerance
        self.max_support_size = max_support_size
        self.lemke_howson = LemkeHowsonSolver(tolerance=tolerance)
        self.support_enumeration = SupportEnumerationSolver(tolerance=tolerance)
        self.ce_solver = CorrelatedEquilibriumSolver(tolerance=tolerance)

    def solve_mixed_nash_2player(
        self,
        agent_i_id: UUID,
        agent_j_id: UUID,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        action_names_i: List[str],
        action_names_j: List[str],
    ) -> MixedNashEquilibrium:
        """
        Solve for mixed strategy Nash equilibrium in 2-player game using Lemke-Howson

        Args:
            agent_i_id: First player ID
            agent_j_id: Second player ID
            payoff_matrix_i: Payoff matrix for player i (rows=i's actions, cols=j's actions)
            payoff_matrix_j: Payoff matrix for player j
            action_names_i: Names of player i's actions
            action_names_j: Names of player j's actions

        Returns:
            Mixed Nash equilibrium with strategies for both players
        """
        # Filter dominated strategies first
        reduced_i, remaining_i = self.support_enumeration.filter_dominated_strategies(
            payoff_matrix_i
        )
        reduced_j, remaining_j = self.support_enumeration.filter_dominated_strategies(
            payoff_matrix_j.T
        )

        # Reconstruct reduced matrices
        reduced_payoff_i = payoff_matrix_i[np.ix_(remaining_i, remaining_j)]
        reduced_payoff_j = payoff_matrix_j[np.ix_(remaining_i, remaining_j)]

        # Solve using Lemke-Howson
        strategy_i_reduced, strategy_j_reduced = self.lemke_howson.solve(
            reduced_payoff_i, reduced_payoff_j
        )

        # Map back to full action space
        strategy_i = np.zeros(len(action_names_i))
        strategy_j = np.zeros(len(action_names_j))

        for idx, action_idx in enumerate(remaining_i):
            strategy_i[action_idx] = strategy_i_reduced[idx]
        for idx, action_idx in enumerate(remaining_j):
            strategy_j[action_idx] = strategy_j_reduced[idx]

        # Identify supports
        support_i = {action_names_i[i] for i in range(len(strategy_i)) if strategy_i[i] > self.tolerance}
        support_j = {action_names_j[j] for j in range(len(strategy_j)) if strategy_j[j] > self.tolerance}

        # Build mixed strategies
        mixed_i = MixedStrategy(
            agent_id=agent_i_id,
            action_probabilities={
                action_names_i[i]: strategy_i[i]
                for i in range(len(strategy_i))
                if strategy_i[i] > self.tolerance
            },
            support=support_i,
            expected_payoff=float(strategy_i @ payoff_matrix_i @ strategy_j),
        )

        mixed_j = MixedStrategy(
            agent_id=agent_j_id,
            action_probabilities={
                action_names_j[j]: strategy_j[j]
                for j in range(len(strategy_j))
                if strategy_j[j] > self.tolerance
            },
            support=support_j,
            expected_payoff=float(strategy_i @ payoff_matrix_j @ strategy_j),
        )

        is_pure = len(support_i) == 1 and len(support_j) == 1

        return MixedNashEquilibrium(
            strategies={agent_i_id: mixed_i, agent_j_id: mixed_j},
            is_pure=is_pure,
            support_size=len(support_i) + len(support_j),
            convergence_error=0.0,
        )

    def support_enumeration(
        self,
        agent_i_id: UUID,
        agent_j_id: UUID,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        action_names_i: List[str],
        action_names_j: List[str],
    ) -> List[MixedNashEquilibrium]:
        """
        Find all Nash equilibria using support enumeration

        Args:
            agent_i_id: First player ID
            agent_j_id: Second player ID
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            action_names_i: Names of player i's actions
            action_names_j: Names of player j's actions

        Returns:
            List of all found Nash equilibria
        """
        equilibria = []

        # Generate all support pairs
        supports = self.support_enumeration.enumerate_supports(
            payoff_matrix_i.shape[0],
            payoff_matrix_j.shape[1],
            max_support_size=self.max_support_size,
        )

        for supp_i, supp_j in supports:
            result = self.support_enumeration.solve_for_support(
                payoff_matrix_i, payoff_matrix_j,
                supp_i, supp_j
            )

            if result is not None:
                strategy_i, strategy_j = result

                # Build equilibrium
                support_i_names = {action_names_i[i] for i in supp_i}
                support_j_names = {action_names_j[j] for j in supp_j}

                mixed_i = MixedStrategy(
                    agent_id=agent_i_id,
                    action_probabilities={
                        action_names_i[i]: strategy_i[i]
                        for i in range(len(strategy_i))
                        if strategy_i[i] > self.tolerance
                    },
                    support=support_i_names,
                    expected_payoff=float(strategy_i @ payoff_matrix_i @ strategy_j),
                )

                mixed_j = MixedStrategy(
                    agent_id=agent_j_id,
                    action_probabilities={
                        action_names_j[j]: strategy_j[j]
                        for j in range(len(strategy_j))
                        if strategy_j[j] > self.tolerance
                    },
                    support=support_j_names,
                    expected_payoff=float(strategy_i @ payoff_matrix_j @ strategy_j),
                )

                is_pure = len(support_i_names) == 1 and len(support_j_names) == 1

                equilibria.append(MixedNashEquilibrium(
                    strategies={agent_i_id: mixed_i, agent_j_id: mixed_j},
                    is_pure=is_pure,
                    support_size=len(support_i_names) + len(support_j_names),
                ))

        return equilibria

    def compute_correlated_equilibrium(
        self,
        agent_i_id: UUID,
        agent_j_id: UUID,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        action_names_i: List[str],
        action_names_j: List[str],
        objective: str = "welfare",
    ) -> Dict:
        """
        Compute correlated equilibrium using linear programming

        Args:
            agent_i_id: First player ID
            agent_j_id: Second player ID
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            action_names_i: Names of player i's actions
            action_names_j: Names of player j's actions
            objective: Optimization objective

        Returns:
            Dictionary with joint distribution and expected payoffs
        """
        distribution = self.ce_solver.compute_correlated_equilibrium(
            payoff_matrix_i, payoff_matrix_j, objective
        )

        # Compute marginals
        marginal_i = distribution.sum(axis=1)
        marginal_j = distribution.sum(axis=0)

        # Expected payoffs
        expected_payoff_i = float(np.sum(distribution * payoff_matrix_i))
        expected_payoff_j = float(np.sum(distribution * payoff_matrix_j))

        return {
            "agent_i_id": str(agent_i_id),
            "agent_j_id": str(agent_j_id),
            "joint_distribution": {
                f"{action_names_i[i]}|{action_names_j[j]}": float(distribution[i, j])
                for i in range(len(action_names_i))
                for j in range(len(action_names_j))
                if distribution[i, j] > self.tolerance
            },
            "marginal_i": {
                action_names_i[i]: float(marginal_i[i])
                for i in range(len(action_names_i))
                if marginal_i[i] > self.tolerance
            },
            "marginal_j": {
                action_names_j[j]: float(marginal_j[j])
                for j in range(len(action_names_j))
                if marginal_j[j] > self.tolerance
            },
            "expected_payoff_i": expected_payoff_i,
            "expected_payoff_j": expected_payoff_j,
            "social_welfare": expected_payoff_i + expected_payoff_j,
        }

    def check_nash_equilibrium(
        self,
        agent_i_id: UUID,
        agent_j_id: UUID,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        strategy_i: Dict[str, float],
        strategy_j: Dict[str, float],
        action_names_i: List[str],
        action_names_j: List[str],
    ) -> Dict:
        """
        Check if given strategies form a Nash equilibrium

        Args:
            agent_i_id: First player ID
            agent_j_id: Second player ID
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            strategy_i: Mixed strategy for player i (action_name -> probability)
            strategy_j: Mixed strategy for player j
            action_names_i: Names of player i's actions
            action_names_j: Names of player j's actions

        Returns:
            Dictionary with verification results
        """
        # Convert strategies to arrays
        strat_i = np.array([strategy_i.get(name, 0.0) for name in action_names_i])
        strat_j = np.array([strategy_j.get(name, 0.0) for name in action_names_j])

        # Normalize
        if strat_i.sum() > 0:
            strat_i /= strat_i.sum()
        if strat_j.sum() > 0:
            strat_j /= strat_j.sum()

        # Expected payoffs
        expected_i = float(strat_i @ payoff_matrix_i @ strat_j)
        expected_j = float(strat_i @ payoff_matrix_j @ strat_j)

        # Check for profitable deviations
        deviations_i = []
        for i, action in enumerate(action_names_i):
            pure_i = np.zeros_like(strat_i)
            pure_i[i] = 1.0
            dev_payoff = float(pure_i @ payoff_matrix_i @ strat_j)
            if dev_payoff > expected_i + self.tolerance:
                deviations_i.append({
                    "action": action,
                    "payoff": dev_payoff,
                    "gain": dev_payoff - expected_i,
                })

        deviations_j = []
        for j, action in enumerate(action_names_j):
            pure_j = np.zeros_like(strat_j)
            pure_j[j] = 1.0
            dev_payoff = float(strat_i @ payoff_matrix_j @ pure_j)
            if dev_payoff > expected_j + self.tolerance:
                deviations_j.append({
                    "action": action,
                    "payoff": dev_payoff,
                    "gain": dev_payoff - expected_j,
                })

        is_equilibrium = len(deviations_i) == 0 and len(deviations_j) == 0

        return {
            "is_nash_equilibrium": is_equilibrium,
            "expected_payoff_i": expected_i,
            "expected_payoff_j": expected_j,
            "profitable_deviations_i": deviations_i,
            "profitable_deviations_j": deviations_j,
            "epsilon_equilibrium": is_equilibrium or (
                max(
                    [0] + [d["gain"] for d in deviations_i] + [d["gain"] for d in deviations_j]
                ) < self.tolerance * 10
            ),
        }


def generate_action_space(
    agent_id: UUID,
    agent_state: AgentState,
    counterparties: List[UUID],
) -> List[AgentAction]:
    """
    Generate feasible action space for an agent.

    Action magnitudes scale with agent state for realism.
    All 6 action types are generated when conditions are met.

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

    # Agent-dependent magnitudes
    credit_magnitude = max(500, min(5000, agent_state.credit_exposure * 0.1))
    liquidity_magnitude = max(1000, min(10000, agent_state.credit_exposure * 0.05))
    margin_magnitude = max(200, min(2000, agent_state.credit_exposure * 0.05))

    # Credit limit adjustments for each counterparty
    for cp_id in counterparties:
        # Increase credit
        actions.append(
            AgentAction(
                action_type=ActionType.ADJUST_CREDIT_LIMIT,
                agent_id=agent_id,
                target_id=cp_id,
                magnitude=credit_magnitude,
            )
        )
        # Decrease credit
        actions.append(
            AgentAction(
                action_type=ActionType.ADJUST_CREDIT_LIMIT,
                agent_id=agent_id,
                target_id=cp_id,
                magnitude=-credit_magnitude,
            )
        )

    # Liquidity decisions
    if agent_state.liquidity_buffer > 0.3:
        actions.append(
            AgentAction(
                action_type=ActionType.LIQUIDITY_DECISION,
                agent_id=agent_id,
                magnitude=liquidity_magnitude,
                parameters={"decision": "RELEASE"},
            )
        )

    if agent_state.stress_level > 0.4:
        actions.append(
            AgentAction(
                action_type=ActionType.LIQUIDITY_DECISION,
                agent_id=agent_id,
                magnitude=liquidity_magnitude,
                parameters={"decision": "HOARD"},
            )
        )

    # Margin modifications — tighten when stressed, loosen when risk-seeking
    for cp_id in counterparties:
        if agent_state.stress_level > 0.3:
            actions.append(
                AgentAction(
                    action_type=ActionType.MODIFY_MARGIN,
                    agent_id=agent_id,
                    target_id=cp_id,
                    magnitude=margin_magnitude,
                    parameters={"direction": "tighten"},
                )
            )
        if agent_state.risk_appetite > 0.6:
            actions.append(
                AgentAction(
                    action_type=ActionType.MODIFY_MARGIN,
                    agent_id=agent_id,
                    target_id=cp_id,
                    magnitude=-margin_magnitude,
                    parameters={"direction": "loosen"},
                )
            )

    # Trade rerouting — when stressed and multiple counterparties available
    if agent_state.is_stressed() and len(counterparties) > 1:
        for cp_id in counterparties:
            actions.append(
                AgentAction(
                    action_type=ActionType.REROUTE_TRADE,
                    agent_id=agent_id,
                    target_id=cp_id,
                    magnitude=credit_magnitude,
                    parameters={"from": str(cp_id)},
                )
            )

    # Collateral calls — when liquidity is low
    if agent_state.liquidity_buffer < 0.4:
        for cp_id in counterparties:
            actions.append(
                AgentAction(
                    action_type=ActionType.COLLATERAL_CALL,
                    agent_id=agent_id,
                    target_id=cp_id,
                    magnitude=margin_magnitude * 2,
                )
            )

    return actions
