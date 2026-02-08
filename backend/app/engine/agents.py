"""
Agent-Based Model: Financial Network Agents
============================================
Polymorphic agent classes for strategic simulation of financial networks.
Each agent type has:
- Policy Variables (Adjustable levers)
- State Variables (Computed health metrics)
- Strategy (Game-theoretic decision rules)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the financial ecosystem"""
    BANK = "bank"
    CCP = "ccp"
    SECTOR = "sector"
    REGULATOR = "regulator"


class AgentMode(Enum):
    """Behavioral modes for agents"""
    NORMAL = "normal"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"
    DISTRESS = "distress"
    DEFAULT = "default"


@dataclass
class AgentState:
    """Snapshot of an agent's state at time t"""
    timestep: int
    health_score: float
    mode: AgentMode
    liquidity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "health_score": self.health_score,
            "mode": self.mode.value,
            "liquidity": self.liquidity
        }


class Agent(ABC):
    """
    Base class for all agents in the financial ecosystem.
    Implements the core Agent protocol for time-stepped simulation.
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.alive = True
        self.mode = AgentMode.NORMAL
        self.timestep = 0
        self.history: List[AgentState] = []
        
    @abstractmethod
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        Observe the environment and neighbors.
        Updates internal perception variables.
        """
        pass
    
    @abstractmethod
    def decide(self) -> Dict[str, Any]:
        """
        Execute strategic decision-making (The Game Theory layer).
        Returns a dict of policy decisions.
        """
        pass
    
    @abstractmethod
    def act(self, network: 'nx.DiGraph') -> List[Dict[str, Any]]:
        """
        Execute actions on the network based on decisions.
        Returns a list of transactions/events.
        """
        pass
    
    @abstractmethod
    def compute_health(self) -> float:
        """
        Calculate a normalized health score [0, 1].
        0 = Default, 1 = Fully Healthy
        """
        pass
    
    def step(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute one full cycle: Perceive -> Decide -> Act
        """
        if not self.alive:
            return []
        
        self.perceive(network, global_state)
        self.decide()
        events = self.act(network)
        
        # Record state for history
        health = self.compute_health()
        state = AgentState(
            timestep=self.timestep,
            health_score=health,
            mode=self.mode,
            liquidity=getattr(self, 'liquidity', 0.0)
        )
        self.history.append(state)
        self.timestep += 1
        
        return events
    
    def default(self):
        """Mark agent as defaulted"""
        self.alive = False
        self.mode = AgentMode.DEFAULT
        logger.warning(f"{self.agent_type.value} {self.agent_id} has DEFAULTED at t={self.timestep}")


class BankAgent(Agent):
    """
    Banks: The core nodes holding balance sheets and making credit decisions.
    
    Policy Variables (User/AI Adjustable):
    - credit_supply_limit: Maximum total lending
    - lending_spread: Interest premium (basis points)
    - interbank_limit: Max lending to other banks
    - risk_appetite: Willingness to lend to risky sectors [0, 1]
    
    State Variables (Computed):
    - capital: Tier-1 Capital (equity)
    - risk_weighted_assets: RWA for CRAR calculation
    - crar: Capital Adequacy Ratio (%)
    - npa_ratio: Non-Performing Assets (%)
    - liquidity: Available cash for immediate obligations
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_capital: float,
        initial_assets: float,
        initial_liquidity: float,
        initial_npa_ratio: float = 0.0,
        initial_crar: float = 12.0,
        regulatory_min_crar: float = 9.0
    ):
        super().__init__(agent_id, AgentType.BANK)
        
        # Balance Sheet Components
        self.capital = initial_capital
        self.risk_weighted_assets = initial_assets
        self.liquidity = initial_liquidity
        self.npa_ratio = initial_npa_ratio
        self.crar = initial_crar
        self.regulatory_min_crar = regulatory_min_crar
        self.debt_obligations = 0.0  # Total debt service obligations
        
        # Policy Variables (Start Conservative)
        self.credit_supply_limit = initial_capital * 10  # 10x leverage
        self.lending_spread = 200  # 200 bps over base rate
        self.interbank_limit = initial_capital * 0.5
        self.risk_appetite = 0.5  # Moderate
        
        # Perception Variables (Updated each step)
        self.perceived_systemic_stress = 0.0
        self.neighbor_defaults = 0
        self.neighbor_avg_health = 1.0
        
        # ML Integration (Probability of Default Predictor)
        self.default_predictor = None  # Will be injected with XGBoost model
        self.ml_risk_advisor = None  # ML-based risk mitigation advisor
        self.is_user_controlled = False  # Flag for user-controlled entities
        
        # Store references for ML processing
        self._current_network = None
        self._current_global_state = None
        self._all_agent_states = {}
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        Observe neighbors and global market conditions.
        """
        # Store references for ML processing
        self._current_network = network
        self._current_global_state = global_state
        
        # Count failed neighbors
        self.neighbor_defaults = 0
        neighbor_health_scores = []
        
        for neighbor_id in network.neighbors(self.agent_id):
            neighbor = network.nodes[neighbor_id].get('agent')
            if neighbor and not neighbor.alive:
                self.neighbor_defaults += 1
            elif neighbor:
                neighbor_health_scores.append(neighbor.compute_health())
        
        self.neighbor_avg_health = np.mean(neighbor_health_scores) if neighbor_health_scores else 1.0
        
        # Global liquidity perception
        global_liquidity = global_state.get('system_liquidity', 1.0)
        market_volatility = global_state.get('market_volatility', 0.0)
        
        # Compute stress perception
        self.perceived_systemic_stress = (
            (1.0 - self.neighbor_avg_health) * 0.4 +
            (self.neighbor_defaults / max(len(list(network.neighbors(self.agent_id))), 1)) * 0.3 +
            market_volatility * 0.2 +
            (1.0 - global_liquidity) * 0.1
        )
        
    def decide(self) -> Dict[str, Any]:
        """
        ML-Enhanced Game Theory Strategy:
        Uses machine learning to predict risk and make proactive risk-reducing decisions.
        
        New Approach:
        1. ML Risk Assessment: Predict default probability and systemic risk
        2. Risk Mitigation: Generate optimal actions to reduce predicted risk
        3. Policy Optimization: Adjust parameters to minimize network-wide risk
        4. Execution: Apply risk-reducing actions
        """
        decisions = {}
        
        # Use ML risk advisor if available
        if self.ml_risk_advisor and self._current_network:
            try:
                # Convert self to AgentState for ML processing
                from app.engine.game_theory import AgentState as GTAgentState
                from uuid import UUID
                
                agent_state = GTAgentState(
                    agent_id=UUID(int=hash(self.agent_id) & 0xFFFFFFFFFFFFFFFF),
                    capital_ratio=self.crar / 10.0,  # Normalize to 0-1 range
                    liquidity_buffer=self.liquidity / max(self.capital, 1.0),
                    credit_exposure=self.credit_supply_limit / max(self.capital, 1.0),
                    default_probability=0.0,  # Will be calculated by ML
                    stress_level=self.perceived_systemic_stress,
                    risk_appetite=self.risk_appetite,
                )
                
                # Get ML risk assessment
                risk_assessment = self.ml_risk_advisor.assess_risk(
                    agent_id=agent_state.agent_id,
                    agent_state=agent_state,
                    network=self._current_network,
                    all_agent_states=self._all_agent_states,
                )
                
                # Store ML predictions for reporting
                decisions['ml_default_probability'] = risk_assessment.default_probability
                decisions['ml_risk_level'] = risk_assessment.current_risk_level.value
                decisions['ml_confidence'] = risk_assessment.ml_confidence
                
                # Apply ML-recommended actions for risk reduction
                if risk_assessment.recommended_actions:
                    decisions['ml_recommendations'] = [
                        {
                            'action': action.action_type.value,
                            'magnitude': action.magnitude,
                            'risk_reduction': action.expected_risk_reduction,
                            'reasoning': action.reasoning
                        }
                        for action in risk_assessment.recommended_actions[:3]  # Top 3
                    ]
                    
                    # Execute ML-guided risk reduction (gradual)
                    self._execute_ml_risk_reduction(risk_assessment)
                    decisions['action'] = 'ML_GUIDED_RISK_REDUCTION'
                
                # ONLY optimize policies if in distress or high risk
                # Don't constantly fiddle with policies in normal conditions
                from app.ml.risk_mitigation import RiskLevel
                if risk_assessment.current_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    current_policies = {
                        'credit_supply_limit': self.credit_supply_limit / max(self.capital, 1.0),
                        'risk_appetite': self.risk_appetite,
                        'interbank_limit': self.interbank_limit / max(self.capital, 1.0),
                    }
                    
                    optimized_policies = self.ml_risk_advisor.optimize_policy_parameters(
                        agent_state=agent_state,
                        risk_assessment=risk_assessment,
                        current_policies=current_policies,
                    )
                    
                    # Apply optimized policies GRADUALLY (blend with current)
                    # Move only 20% of the way toward optimal each step
                    blend_factor = 0.2
                    target_credit = optimized_policies['credit_supply_limit'] * self.capital
                    target_risk_app = optimized_policies['risk_appetite']
                    target_interbank = optimized_policies['interbank_limit'] * self.capital
                    
                    self.credit_supply_limit = self.credit_supply_limit * (1 - blend_factor) + target_credit * blend_factor
                    self.risk_appetite = self.risk_appetite * (1 - blend_factor) + target_risk_app * blend_factor
                    self.interbank_limit = self.interbank_limit * (1 - blend_factor) + target_interbank * blend_factor
                    
                    decisions['optimized_policies'] = optimized_policies
                else:
                    # In normal/low risk, just maintain current policies
                    decisions['action'] = 'STEADY_STATE_ML'
                
            except Exception as e:
                logger.warning(f"ML risk advisor failed for {self.agent_id}: {e}")
                # Fall back to traditional strategy
                self._traditional_decision_strategy(decisions)
        else:
            # Traditional strategy if ML not available
            self._traditional_decision_strategy(decisions)
        
        decisions['credit_supply_limit'] = self.credit_supply_limit
        decisions['risk_appetite'] = self.risk_appetite
        decisions['interbank_limit'] = self.interbank_limit
        
        return decisions
    
    def _traditional_decision_strategy(self, decisions: Dict[str, Any]):
        """
        Enhanced risk-responsive strategy (fallback when ML not available).
        Makes proactive policy adjustments based on health metrics.
        """
        # Compute current health status
        current_health = self.compute_health()
        
        # CRITICAL DISTRESS: CRAR below minimum
        if self.crar < self.regulatory_min_crar:
            self.mode = AgentMode.DISTRESS
            # Emergency deleveraging - aggressive but gradual
            self.credit_supply_limit *= 0.7  # Don't crash to 0.5, be gradual
            self.interbank_limit *= 0.5
            self.risk_appetite = max(0.15, self.risk_appetite * 0.5)  # Don't go to 0.1 immediately
            decisions['action'] = 'EMERGENCY_DELEVERAGING'
            logger.warning(f"{self.agent_id}: CRITICAL DISTRESS - CRAR {self.crar:.2f}% below min {self.regulatory_min_crar:.2f}%")
        
        # HIGH RISK: Health below 40% or significant system stress
        elif current_health < 0.4 or self.perceived_systemic_stress > 0.6 or self.neighbor_defaults > 0:
            self.mode = AgentMode.DEFENSIVE
            # Proactive risk reduction
            self.credit_supply_limit *= 0.85  # More gradual reduction
            self.interbank_limit *= 0.7
            self.risk_appetite = max(0.3, self.risk_appetite * 0.8)
            
            # Build liquidity buffer
            if self.liquidity / max(self.capital, 1.0) < 0.25:
                # Redirect some lending capacity to liquidity
                liquidity_target = self.capital * 0.25
                liquidity_gap = max(0, liquidity_target - self.liquidity)
                self.liquidity += min(liquidity_gap * 0.2, self.capital * 0.05)  # Gradual increase
            
            decisions['action'] = 'DEFENSIVE_RISK_REDUCTION'
            logger.info(f"{self.agent_id}: DEFENSIVE MODE - Health {current_health:.2f}, Stress {self.perceived_systemic_stress:.2f}")
        
        # MODERATE RISK: Health 40-60% or moderate stress
        elif current_health < 0.6 or self.perceived_systemic_stress > 0.4:
            self.mode = AgentMode.NORMAL
            # Cautious adjustments
            
                # Reduce NPA exposure if high
            if self.npa_ratio > 5.0:  # Above 5% NPA
                self.credit_supply_limit *= 0.92  # Slight reduction
                self.risk_appetite = max(0.4, self.risk_appetite * 0.9)
                decisions['action'] = 'CAUTIOUS_DELEVERAGING'
            else:
                decisions['action'] = 'STEADY_STATE'
            
            # Ensure minimum liquidity buffer
            min_liquidity_ratio = 0.15
            if self.liquidity / max(self.capital, 1.0) < min_liquidity_ratio:
                liquidity_increase = self.capital * 0.02  # Small gradual increase
                self.liquidity += liquidity_increase
                decisions['liquidity_action'] = 'BUILD_BUFFER'
            
            logger.debug(f"{self.agent_id}: MODERATE RISK - Health {current_health:.2f}")
        
        # LOW RISK: Healthy conditions, can expand cautiously
        elif current_health > 0.7 and self.perceived_systemic_stress < 0.3 and self.crar > self.regulatory_min_crar + 3:
            self.mode = AgentMode.AGGRESSIVE
            # Measured expansion (not reckless)
            self.credit_supply_limit *= 1.05  # Modest expansion
            self.risk_appetite = min(0.8, self.risk_appetite * 1.1)  # Cap at 0.8, not 0.9
            decisions['action'] = 'MEASURED_EXPANSION'
            logger.debug(f"{self.agent_id}: EXPANDING - Health {current_health:.2f}, CRAR {self.crar:.2f}%")
        
        else:
            # Normal Mode - maintain status quo
            self.mode = AgentMode.NORMAL
            decisions['action'] = 'STEADY_STATE'
            
            # Still do minor adjustments for risk management
            # If NPA is creeping up, be slightly more conservative
            if self.npa_ratio > 3.0:
                self.risk_appetite = max(0.4, self.risk_appetite * 0.95)
                decisions['micro_adjustment'] = 'NPA_RESPONSE'
    
    def _execute_ml_risk_reduction(self, risk_assessment):
        """
        Execute ML-recommended risk reduction actions.
        Focus on CAPITAL PRESERVATION and gradual adjustments.
        """
        from app.ml.risk_mitigation import RiskLevel
        
        # CRITICAL: Don't make changes that reduce capital or income
        # Only take the top 1-2 most impactful actions, and do them gradually
        top_actions = risk_assessment.recommended_actions[:2]
        
        for action in top_actions:
            if action.action_type.value == 'liquidity_decision' and action.magnitude > 0:
                # Gradual liquidity building - only if we have capacity
                if self.liquidity / max(self.capital, 1.0) < 0.5:
                    # Small, gradual increase (max 5% of capital)
                    liquidity_increase = min(action.magnitude * self.capital * 0.05, self.capital * 0.05)
                    self.liquidity += liquidity_increase
                    # DON'T reduce credit limit - maintain income
                    logger.info(f"{self.agent_id}: Increased liquidity by {liquidity_increase:.2f}")
            
            elif action.action_type.value == 'adjust_credit_limit' and action.magnitude < 0:
                # Very gradual credit reduction (max 5% per step)
                reduction = max(action.magnitude, -0.05)
                self.credit_supply_limit *= (1 + reduction)
                logger.info(f"{self.agent_id}: Reduced credit limit by {-reduction * 100:.1f}%")
            
            elif action.action_type.value == 'modify_margin':
                # Gradual margin increase (max 50 bps per step)
                margin_increase = min(action.magnitude * 100, 50)
                self.lending_spread += margin_increase
                logger.info(f"{self.agent_id}: Increased lending spread by {margin_increase:.0f} bps")
        
        # Set mode based on ML risk assessment - but don't panic
        if risk_assessment.current_risk_level == RiskLevel.CRITICAL and self.crar < self.regulatory_min_crar:
            self.mode = AgentMode.DISTRESS
        elif risk_assessment.current_risk_level == RiskLevel.HIGH:
            self.mode = AgentMode.DEFENSIVE
        else:
            self.mode = AgentMode.NORMAL
    
    def act(self, network: 'nx.DiGraph') -> List[Dict[str, Any]]:
        """
        Execute lending decisions and update balance sheet.
        """
        events = []
        
        # Check if bank should default
        if self.capital <= 0 or self.crar < 0:
            self.default()
            events.append({
                'type': 'BANK_DEFAULT',
                'agent_id': self.agent_id,
                'timestep': self.timestep,
                'capital': self.capital,
                'crar': self.crar
            })
            return events
        
        # Update network node attributes
        network.nodes[self.agent_id]['crar'] = self.crar
        network.nodes[self.agent_id]['liquidity'] = self.liquidity
        network.nodes[self.agent_id]['mode'] = self.mode.value
        
        return events
    
    def compute_health(self) -> float:
        """
        Health score based on CRAR and Liquidity.
        """
        if not self.alive:
            return 0.0
        
        # CRAR component: normalized around regulatory minimum
        crar_health = min(1.0, max(0.0, (self.crar - self.regulatory_min_crar) / 10.0))
        
        # Liquidity component: normalized
        if self.capital > 0:
            liquidity_health = min(1.0, self.liquidity / (self.capital * 0.2))  # 20% of capital is "good"
        else:
            liquidity_health = 0.0  # No capital = no liquidity health
        
        # NPA component: inverse relationship
        npa_health = max(0.0, 1.0 - (self.npa_ratio / 15.0))  # 15% NPA = 0 health
        
        # Clamp final health to [0, 1] range
        health = crar_health * 0.5 + liquidity_health * 0.3 + npa_health * 0.2
        return min(1.0, max(0.0, health))
    
    def apply_shock(self, loss_amount: float, source: str = "market"):
        """
        Apply a capital loss to the bank.
        """
        self.capital -= loss_amount
        if self.capital <= 0:
            self.capital = 0
        
        # Recalculate CRAR
        if self.risk_weighted_assets > 0:
            self.crar = (self.capital / self.risk_weighted_assets) * 100
        
        logger.info(f"{self.agent_id} absorbed loss of {loss_amount:.2f} from {source}. New CRAR: {self.crar:.2f}%")
    
    def set_predictor(self, model):
        """Inject trained XGBoost model for PD estimation"""
        self.default_predictor = model


class CCPAgent(Agent):
    """
    Central Counterparty (Clearing House): The central node managed by the user.
    
    Policy Variables:
    - initial_margin_requirement: % collateral required (e.g., 10%)
    - variation_margin_threshold: Sensitivity to price changes
    - default_fund_contribution: Required contribution from members
    
    State Variables:
    - default_fund_size: Total pool available
    - margin_buffer: Current excess collateral held
    - active_exposures: Total notional exposure cleared
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_default_fund: float,
        initial_margin_requirement: float = 10.0  # Percentage
    ):
        super().__init__(agent_id, AgentType.CCP)
        
        # State Variables
        self.default_fund_size = initial_default_fund
        self.margin_buffer = 0.0
        self.active_exposures = 0.0
        
        # Policy Variables (User Controlled)
        self.initial_margin_requirement = initial_margin_requirement
        self.variation_margin_threshold = 0.02  # 2% price move triggers call
        self.haircut_rate = 0.0  # Loss sharing rate
        
        # Custom User Rules (Injected via API)
        self.policy_rules: List[Dict[str, Any]] = []
        
        # ML Integration
        self.ml_risk_advisor = None
        self._current_network = None
        self._current_global_state = None
        self._all_agent_states = {}
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        CCP observes system-wide stress.
        """
        # Store references for ML processing
        self._current_network = network
        self._current_global_state = global_state
        
        # Calculate total system NPA
        total_npa = 0.0
        bank_count = 0
        
        for node_id, data in network.nodes(data=True):
            agent = data.get('agent')
            if isinstance(agent, BankAgent) and agent.alive:
                total_npa += agent.npa_ratio
                bank_count += 1
        
        system_npa = total_npa / bank_count if bank_count > 0 else 0.0
        global_state['system_npa'] = system_npa
        
    def decide(self) -> Dict[str, Any]:
        """
        CCP decision-making with ML-guided risk reduction.
        Uses ML to predict systemic risk and adjust margin requirements.
        """
        decisions = {}
        
        # Use ML risk advisor if available
        if self.ml_risk_advisor and self._current_network:
            try:
                # Assess network-wide risk
                bank_risks = []
                for node_id, data in self._current_network.nodes(data=True):
                    agent = data.get('agent')
                    if isinstance(agent, BankAgent) and agent.alive:
                        from app.engine.game_theory import AgentState as GTAgentState
                        from uuid import UUID
                        
                        agent_state = GTAgentState(
                            agent_id=UUID(int=hash(agent.agent_id) & 0xFFFFFFFFFFFFFFFF),
                            capital_ratio=agent.crar / 10.0,
                            liquidity_buffer=agent.liquidity / max(agent.capital, 1.0),
                            credit_exposure=agent.credit_supply_limit / max(agent.capital, 1.0),
                            default_probability=0.0,
                            stress_level=agent.perceived_systemic_stress,
                            risk_appetite=agent.risk_appetite,
                        )
                        
                        risk_assessment = self.ml_risk_advisor.assess_risk(
                            agent_id=agent_state.agent_id,
                            agent_state=agent_state,
                            network=self._current_network,
                            all_agent_states=self._all_agent_states,
                        )
                        bank_risks.append(risk_assessment.default_probability)
                
                # Calculate system-wide risk
                avg_default_prob = np.mean(bank_risks) if bank_risks else 0.0
                max_default_prob = max(bank_risks) if bank_risks else 0.0
                
                decisions['avg_system_risk'] = avg_default_prob
                decisions['max_member_risk'] = max_default_prob
                
                # Adjust margins based on ML predictions
                if avg_default_prob > 0.4:  # High systemic risk
                    margin_increase = min(0.2, (avg_default_prob - 0.4) * 0.5)
                    self.initial_margin_requirement *= (1 + margin_increase)
                    decisions['action'] = 'INCREASE_MARGINS'
                    decisions['margin_adjustment'] = margin_increase
                    logger.info(f"CCP: Increased margins by {margin_increase * 100:.1f}% due to high systemic risk")
                
                elif avg_default_prob < 0.15:  # Low risk
                    margin_decrease = min(0.1, (0.15 - avg_default_prob) * 0.3)
                    self.initial_margin_requirement *= (1 - margin_decrease)
                    decisions['action'] = 'REDUCE_MARGINS'
                    decisions['margin_adjustment'] = -margin_decrease
                    logger.info(f"CCP: Reduced margins by {margin_decrease * 100:.1f}% due to low risk")
                
                else:
                    decisions['action'] = 'MAINTAIN_MARGINS'
                
            except Exception as e:
                logger.warning(f"ML risk advisor failed for CCP: {e}")
                # Fall back to rule-based system
                self._execute_policy_rules(decisions)
        else:
            # Execute traditional rule-based system
            self._execute_policy_rules(decisions)
        
        return decisions
    
    def _execute_policy_rules(self, decisions: Dict[str, Any]):
        """
        Execute risk-responsive policy adjustments (fallback when ML not available).
        CCPs proactively adjust margins based on system health.
        """
        # Calculate system health metrics
        if self._current_network:
            alive_banks = []
            total_banks = 0
            total_npa = 0.0
            avg_crar = 0.0
            
            for node_id, data in self._current_network.nodes(data=True):
                agent = data.get('agent')
                if isinstance(agent, BankAgent):
                    total_banks += 1
                    if agent.alive:
                        alive_banks.append(agent)
                        total_npa += agent.npa_ratio
                        avg_crar += agent.crar
            
            if alive_banks:
                survival_rate = len(alive_banks) / max(total_banks, 1)
                avg_npa = total_npa / len(alive_banks)
                avg_crar = avg_crar / len(alive_banks)
                
                # DEFENSIVE: System under stress
                if survival_rate < 0.7 or avg_npa > 8.0 or avg_crar < 11.0:
                    # Increase margins to protect against further defaults
                    margin_increase = 0.15  # 15% increase
                    self.initial_margin_requirement *= (1 + margin_increase)
                    # Also increase default fund contribution
                    self.default_fund_size *= 1.1
                    decisions['action'] = 'DEFENSIVE_MARGIN_INCREASE'
                    decisions['reason'] = f'System stress: {survival_rate*100:.1f}% survival, {avg_npa:.1f}% NPA'
                    logger.warning(f"CCP: DEFENSIVE MODE - Increased margins by {margin_increase*100:.1f}%")
                
                # CAUTIOUS: Moderate risk
                elif survival_rate < 0.85 or avg_npa > 5.0:
                    # Small margin increase
                    margin_increase = 0.05
                    self.initial_margin_requirement *= (1 + margin_increase)
                    decisions['action'] = 'CAUTIOUS_MARGIN_INCREASE'
                    logger.info(f"CCP: CAUTIOUS - Slightly increased margins")
                
                # RELAXED: Low risk environment
                elif survival_rate > 0.95 and avg_npa < 2.0 and avg_crar > 13.0:
                    # Can reduce margins slightly
                    margin_decrease = 0.03
                    self.initial_margin_requirement *= (1 - margin_decrease)
                    decisions['action'] = 'MARGIN_RELAXATION'
                    logger.info(f"CCP: Low risk - Relaxed margins by {margin_decrease*100:.1f}%")
                
                else:
                    decisions['action'] = 'MAINTAIN_MARGINS'
                
                # Ensure margins don't go too high or too low
                self.initial_margin_requirement = min(0.30, max(0.03, self.initial_margin_requirement))
                decisions['current_margin'] = self.initial_margin_requirement
                
        # Execute any custom user-defined rules
        for rule in self.policy_rules:
            condition = rule.get('condition', lambda state: False)
            action = rule.get('action', lambda: {})
            
            if condition(self):
                action_result = action()
                decisions.update(action_result)
    
    def act(self, network: 'nx.DiGraph') -> List[Dict[str, Any]]:
        """
        CCP actions: Margin calls, default handling.
        """
        events = []
        
        # Check for member defaults
        for node_id, data in network.nodes(data=True):
            agent = data.get('agent')
            if isinstance(agent, BankAgent) and not agent.alive:
                # Bank defaulted - CCP must use default fund
                loss = abs(agent.capital)  # The capital deficit
                if loss > 0:
                    self.default_fund_size -= loss
                    events.append({
                        'type': 'CCP_DEFAULT_FUND_USAGE',
                        'failed_member': node_id,
                        'loss': loss,
                        'remaining_fund': self.default_fund_size
                    })
        
        return events
    
    def compute_health(self) -> float:
        """
        CCP health based on default fund adequacy.
        """
        # Simplified: Fund size relative to total exposures
        if self.active_exposures == 0:
            return 1.0
        
        coverage = self.default_fund_size / self.active_exposures
        return min(1.0, max(0.0, coverage))
    
    def add_policy_rule(self, rule: Dict[str, Any]):
        """Allow user to inject custom policy rules"""
        self.policy_rules.append(rule)


class SectorAgent(Agent):
    """
    Sector Nodes: Represent borrowing sectors (Real Estate, Commodities, etc.)
    
    Policy Variables:
    - leverage_appetite: How much debt to take on
    
    State Variables:
    - economic_health: Stochastic health index [0, 1]
    - debt_load: Total debt owed to banks
    - revenue_volatility: Stochastic shock susceptibility
    """
    
    def __init__(
        self,
        agent_id: str,
        sector_name: str,
        initial_health: float = 0.8,
        base_volatility: float = 0.1
    ):
        super().__init__(agent_id, AgentType.SECTOR)
        
        self.sector_name = sector_name
        self.economic_health = initial_health
        self.debt_load = 0.0
        self.revenue_volatility = base_volatility
        
        # Policy
        self.leverage_appetite = 0.5
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        Sectors are affected by global economic shocks.
        """
        # External shock (will be applied by simulation engine)
        shock_magnitude = global_state.get(f'shock_{self.sector_name}', 0.0)
        if shock_magnitude != 0:
            self.economic_health = max(0.0, min(1.0, self.economic_health + shock_magnitude))
    
    def decide(self) -> Dict[str, Any]:
        """
        Sectors don't have complex strategies - they react to their health.
        """
        if self.economic_health < 0.3:
            self.mode = AgentMode.DISTRESS
        elif self.economic_health < 0.6:
            self.mode = AgentMode.DEFENSIVE
        else:
            self.mode = AgentMode.NORMAL
        
        return {'health': self.economic_health}
    
    def act(self, network: 'nx.DiGraph') -> List[Dict[str, Any]]:
        """
        Sectors generate losses for connected banks based on health.
        REDUCED SEVERITY for more stable simulations.
        """
        events = []
        
        # Find all banks lending to this sector (incoming edges)
        for predecessor in network.predecessors(self.agent_id):
            edge_data = network.edges[predecessor, self.agent_id]
            exposure = edge_data.get('weight', 0.0)
            
            bank = network.nodes[predecessor].get('agent')
            if isinstance(bank, BankAgent) and bank.alive:
                # If sector health drops, banks take proportional loss
                # BUT MUCH LESS SEVERE - only when health is really bad
                if self.economic_health < 0.5:  # Stress threshold (was 0.7)
                    loss_rate = (0.5 - self.economic_health) * 0.02  # Max 1% loss (was 7%)
                    loss_amount = exposure * loss_rate
                    
                    bank.apply_shock(loss_amount, source=f"sector_{self.sector_name}")
                    bank.npa_ratio += loss_rate * 50  # Less NPA increase
                    
                    events.append({
                        'type': 'SECTOR_LOSS_PROPAGATION',
                        'sector': self.sector_name,
                        'bank': predecessor,
                        'loss': loss_amount,
                        'exposure': exposure
                    })
        
        return events
    
    def compute_health(self) -> float:
        """
        Sector health is directly the economic_health variable.
        """
        return min(1.0, max(0.0, self.economic_health))


class RegulatorAgent(Agent):
    """
    Regulator: The Central Bank setting monetary policy.
    
    Policy Variables:
    - base_repo_rate: Cost of money
    - min_crar_requirement: Regulatory floor
    
    State Variables:
    - system_liquidity: Total liquidity in the banking system
    """
    
    def __init__(
        self,
        agent_id: str,
        base_repo_rate: float = 6.0,
        min_crar: float = 9.0
    ):
        super().__init__(agent_id, AgentType.REGULATOR)
        
        self.base_repo_rate = base_repo_rate
        self.min_crar_requirement = min_crar
        self.system_liquidity = 1.0  # Normalized
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        Regulator monitors system-wide metrics.
        """
        total_liquidity = 0.0
        total_capital = 0.0
        
        for node_id, data in network.nodes(data=True):
            agent = data.get('agent')
            if isinstance(agent, BankAgent):
                total_liquidity += agent.liquidity
                total_capital += agent.capital
        
        self.system_liquidity = total_liquidity / (total_capital + 1e-6)
        global_state['system_liquidity'] = self.system_liquidity
    
    def decide(self) -> Dict[str, Any]:
        """
        Adjust repo rate based on system stress.
        """
        # Counter-cyclical policy: Lower rates if stress is high
        system_stress = 1.0 - self.system_liquidity
        
        if system_stress > 0.6:
            # Crisis mode: Cut rates
            self.base_repo_rate = max(2.0, self.base_repo_rate - 0.5)
        elif system_stress < 0.2:
            # Overheating: Raise rates
            self.base_repo_rate = min(10.0, self.base_repo_rate + 0.25)
        
        return {'repo_rate': self.base_repo_rate}
    
    def act(self, network: 'nx.DiGraph') -> List[Dict[str, Any]]:
        """
        Broadcast repo rate to the system.
        """
        return [{
            'type': 'REPO_RATE_UPDATE',
            'rate': self.base_repo_rate
        }]
    
    def compute_health(self) -> float:
        """
        Regulator health is system liquidity.
        """
        return min(1.0, max(0.0, self.system_liquidity))
