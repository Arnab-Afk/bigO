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
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        Observe neighbors and global market conditions.
        """
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
        Game Theory Strategy: Prisoner's Dilemma in Banking.
        
        Strategies:
        1. Panic Rule: If stress is high, hoard liquidity (defensive).
        2. Greed Rule: If stress is low and CRAR is strong, expand lending.
        3. Survival Rule: If CRAR near regulatory min, deleverage aggressively.
        """
        decisions = {}
        
        # Check for distress
        if self.crar < self.regulatory_min_crar:
            self.mode = AgentMode.DISTRESS
            # Emergency deleveraging
            self.credit_supply_limit *= 0.5
            self.interbank_limit *= 0.3
            self.risk_appetite = 0.1
            decisions['action'] = 'EMERGENCY_DELEVERAGING'
            
        elif self.perceived_systemic_stress > 0.5 or self.neighbor_defaults > 0:
            # Defensive Mode (The Nash Equilibrium Trap)
            self.mode = AgentMode.DEFENSIVE
            self.credit_supply_limit *= 0.7  # Cut lending
            self.interbank_limit *= 0.5      # Reduce interbank exposure
            self.risk_appetite *= 0.6        # Become risk-averse
            decisions['action'] = 'DEFENSIVE_HOARDING'
            
        elif self.perceived_systemic_stress < 0.2 and self.crar > self.regulatory_min_crar + 3:
            # Aggressive Mode (Maximize yield)
            self.mode = AgentMode.AGGRESSIVE
            self.credit_supply_limit *= 1.1
            self.risk_appetite = min(0.9, self.risk_appetite * 1.2)
            decisions['action'] = 'AGGRESSIVE_EXPANSION'
            
        else:
            # Normal Mode
            self.mode = AgentMode.NORMAL
            decisions['action'] = 'STEADY_STATE'
        
        decisions['credit_supply_limit'] = self.credit_supply_limit
        decisions['risk_appetite'] = self.risk_appetite
        decisions['interbank_limit'] = self.interbank_limit
        
        return decisions
    
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
        liquidity_health = min(1.0, self.liquidity / (self.capital * 0.2))  # 20% of capital is "good"
        
        # NPA component: inverse relationship
        npa_health = max(0.0, 1.0 - (self.npa_ratio / 15.0))  # 15% NPA = 0 health
        
        return (crar_health * 0.5 + liquidity_health * 0.3 + npa_health * 0.2)
    
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
        
    def perceive(self, network: 'nx.DiGraph', global_state: Dict[str, Any]) -> None:
        """
        CCP observes system-wide stress.
        """
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
        Execute user-defined policy rules.
        Example Rule: IF system_npa > 8% THEN increase_haircuts_by 5%
        """
        decisions = {}
        
        # Execute custom rules
        for rule in self.policy_rules:
            # Rule execution logic (simplified)
            condition = rule.get('condition', lambda state: False)
            action = rule.get('action', lambda: {})
            
            if condition(self):
                action_result = action()
                decisions.update(action_result)
        
        return decisions
    
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
        return min(1.0, coverage)
    
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
        """
        events = []
        
        # Find all banks lending to this sector (incoming edges)
        for predecessor in network.predecessors(self.agent_id):
            edge_data = network.edges[predecessor, self.agent_id]
            exposure = edge_data.get('weight', 0.0)
            
            bank = network.nodes[predecessor].get('agent')
            if isinstance(bank, BankAgent) and bank.alive:
                # If sector health drops, banks take proportional loss
                if self.economic_health < 0.7:  # Stress threshold
                    loss_rate = (0.7 - self.economic_health) * 0.1  # Max 7% loss if health=0
                    loss_amount = exposure * loss_rate
                    
                    bank.apply_shock(loss_amount, source=f"sector_{self.sector_name}")
                    bank.npa_ratio += loss_rate * 100  # Increase NPA
                    
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
        return self.economic_health


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
        return self.system_liquidity
