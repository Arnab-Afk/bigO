"""
Financial Ecosystem Simulation Engine
=====================================
Agent-Based Model with time-stepped simulation of strategic interactions.

Conway's Game of Life for Finance:
- Survival: Bank.Capital > Regulatory_Min
- Death: Bank.Capital <= 0
- Neighbor Interaction: Weighted edges transmit losses
- Feedback Loops: Defensive actions create cascading liquidity crunches
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

from .agents import (
    Agent, BankAgent, CCPAgent, SectorAgent, RegulatorAgent,
    AgentType, AgentMode
)
from .sector_shocks import (
    SectorType, ShockSeverity, SectorState, SectorShockScenario,
    create_default_sector_states, get_shock_scenario,
    compute_bank_loss_from_sector_shock, SECTOR_SHOCK_LIBRARY
)

logger = logging.getLogger(__name__)


class ShockType(Enum):
    """Types of exogenous shocks"""
    # Legacy shocks (kept for backward compatibility)
    SECTOR_CRISIS = "sector_crisis"
    LIQUIDITY_SQUEEZE = "liquidity_squeeze"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    ASSET_PRICE_CRASH = "asset_price_crash"
    
    # New sector-specific shocks
    REAL_ESTATE_SHOCK = "real_estate_shock"
    INFRASTRUCTURE_SHOCK = "infrastructure_shock"
    MANUFACTURING_SHOCK = "manufacturing_shock"
    AGRICULTURE_SHOCK = "agriculture_shock"
    ENERGY_SHOCK = "energy_shock"
    EXPORT_SHOCK = "export_shock"
    SERVICES_SHOCK = "services_shock"
    MSME_SHOCK = "msme_shock"
    TECHNOLOGY_SHOCK = "technology_shock"
    RETAIL_SHOCK = "retail_shock"
    
    NONE = "none"


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    max_timesteps: int = 100
    enable_shocks: bool = True
    shock_probability: float = 0.05  # Reduced from 0.1 to 0.05 for stability
    contagion_multiplier: float = 0.5  # Reduced from 1.0 to make losses less severe
    enable_regulator: bool = True
    enable_ml: bool = False  # ML is OPT-IN, not forced
    random_seed: Optional[int] = None


@dataclass
class SimulationSnapshot:
    """Complete state of the system at time t"""
    timestep: int
    network_state: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    events: List[Dict[str, Any]]
    global_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestep': self.timestep,
            'network_state': self.network_state,
            'agent_states': self.agent_states,
            'events': self.events,
            'global_metrics': self.global_metrics
        }


class FinancialEcosystem:
    """
    The main simulation engine that orchestrates the Agent-Based Model.
    
    Architecture:
    - Network: networkx DiGraph where nodes are Agents, edges are Exposures
    - Time: Discrete steps t=0, t=1, ..., t=T
    - Order of Operations per Step:
        1. Exogenous Shock (optional)
        2. Agent Perception (parallel)
        3. Agent Decision (parallel)
        4. Agent Action (sequential, order matters)
        5. Contagion Propagation
        6. State Recording
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.network = nx.DiGraph()
        self.timestep = 0
        self.global_state: Dict[str, Any] = {
            'system_liquidity': 1.0,
            'market_volatility': 0.0,
            'system_npa': 0.0,
            'base_repo_rate': 6.0
        }
        self.history: List[SimulationSnapshot] = []
        self.agents: Dict[str, Agent] = {}
        
        # Initialize sector states for multi-sector shock tracking
        self.sector_states: Dict[SectorType, SectorState] = create_default_sector_states()
        # logger.info(f"Initialized {len(self.sector_states)} sector states for shock simulation")
        
        # Initialize ML risk advisor
        from app.ml.risk_mitigation import initialize_risk_advisor
        from app.ml.inference.predictor import DefaultPredictor
        from pathlib import Path
        
        try:
            # Try to load trained ML model
            model_path = Path("ml_models/default_predictor/model.pt")
            
            # ML is OPT-IN, not forced on by default
            # Set enable_ml=False in config to disable
            enable_ml = getattr(config, 'enable_ml', False)
            
            if enable_ml and model_path.exists():
                default_predictor = DefaultPredictor(model_path=model_path)
                self.ml_risk_advisor = initialize_risk_advisor(
                    default_predictor=default_predictor,
                    risk_aversion=0.3,  # Lower risk aversion for stability
                )
                # logger.info("ML risk advisor initialized with trained model")
            elif enable_ml:
                # Use advisor without ML model (uses heuristics)
                self.ml_risk_advisor = initialize_risk_advisor(
                    default_predictor=None,
                    risk_aversion=0.3,  # Lower risk aversion
                )
                # logger.info("ML risk advisor initialized with heuristic fallback")
            else:
                self.ml_risk_advisor = None
                # logger.info("ML risk advisor DISABLED - using traditional strategy")
        except Exception as e:
            logger.warning(f"Failed to initialize ML risk advisor: {e}")
            self.ml_risk_advisor = None
        
        # Initialize random seed for reproducibility
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        # logger.info(f"FinancialEcosystem initialized with config: {config}")
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the ecosystem and inject ML risk advisor.
        Auto-connects RBI and CCP to other nodes for better network integration.
        """
        self.agents[agent.agent_id] = agent
        self.network.add_node(
            agent.agent_id,
            agent=agent,
            agent_type=agent.agent_type.value,
            alive=agent.alive
        )
        
        # Inject ML risk advisor into agent
        if self.ml_risk_advisor:
            if isinstance(agent, (BankAgent, CCPAgent)):
                agent.ml_risk_advisor = self.ml_risk_advisor
                # logger.debug(f"Injected ML risk advisor into {agent.agent_id}")
        
        # Auto-connect RBI/Regulator to all banks
        if isinstance(agent, RegulatorAgent):
            # Connect regulator to all existing banks
            for agent_id, a in self.agents.items():
                if isinstance(a, BankAgent) and agent_id != agent.agent_id:
                    self.network.add_edge(agent.agent_id, agent_id, weight=0, type='regulatory')
                    # logger.debug(f"Connected regulator {agent.agent_id} to {agent_id}")
        
        # If adding a bank and regulator exists, connect them
        if isinstance(agent, BankAgent):
            for a in self.agents.values():
                if isinstance(a, RegulatorAgent):
                    self.network.add_edge(a.agent_id, agent.agent_id, weight=0, type='regulatory')
                    # logger.debug(f"Connected bank {agent.agent_id} to regulator {a.agent_id}")
        
        # Auto-connect CCP to all banks (bidirectional clearing relationships)
        if isinstance(agent, CCPAgent):
            for agent_id, a in self.agents.items():
                if isinstance(a, BankAgent) and agent_id != agent.agent_id:
                    # Bank to CCP (clearing obligations)
                    clearing_amount = a.capital * 0.05  # 5% of capital as initial margin
                    self.network.add_edge(agent_id, agent.agent_id, weight=clearing_amount, type='clearing')
                    # CCP to Bank (potential default fund claims)
                    self.network.add_edge(agent.agent_id, agent_id, weight=0, type='clearing_reciprocal')
                    # logger.debug(f"Connected CCP {agent.agent_id} to {agent_id}")
        
        # If adding a bank and CCP exists, connect them
        if isinstance(agent, BankAgent):
            for a in self.agents.values():
                if isinstance(a, CCPAgent):
                    clearing_amount = agent.capital * 0.05
                    self.network.add_edge(agent.agent_id, a.agent_id, weight=clearing_amount, type='clearing')
                    self.network.add_edge(a.agent_id, agent.agent_id, weight=0, type='clearing_reciprocal')
                    a.active_exposures += clearing_amount
                    # logger.debug(f"Connected bank {agent.agent_id} to CCP {a.agent_id}")
        
        # logger.info(f"Added {agent.agent_type.value} agent: {agent.agent_id}")
    
    def add_exposure(
        self,
        creditor_id: str,
        debtor_id: str,
        exposure_amount: float,
        edge_type: str = "loan"
    ) -> None:
        """
        Add a directed edge representing financial exposure.
        creditor -> debtor with weight = exposure amount
        """
        if creditor_id not in self.agents or debtor_id not in self.agents:
            logger.error(f"Cannot create edge: Invalid agent IDs {creditor_id} -> {debtor_id}")
            return
        
        self.network.add_edge(
            creditor_id,
            debtor_id,
            weight=exposure_amount,
            edge_type=edge_type,
            initial_weight=exposure_amount
        )
        logger.debug(f"Added exposure: {creditor_id} -> {debtor_id} = {exposure_amount}")
    
    def apply_shock(
        self,
        shock_type: ShockType,
        target: Optional[str] = None,
        magnitude: float = -0.2,
        severity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply an exogenous shock to the system using new multi-sector framework.
        
        Args:
            shock_type: Type of shock
            target: Target agent/sector ID (if applicable)
            magnitude: Shock magnitude (for legacy shocks, typically negative)
            severity: Shock severity for new sector shocks ("mild", "moderate", "severe", "crisis")
        
        Returns:
            Dict describing the shock event
        """
        shock_event = {
            'type': 'EXOGENOUS_SHOCK',
            'shock_type': shock_type.value,
            'timestep': self.timestep,
            'magnitude': magnitude,
            'severity': severity
        }
        
        # Map shock types to sectors
        shock_to_sector_map = {
            ShockType.REAL_ESTATE_SHOCK: SectorType.REAL_ESTATE,
            ShockType.INFRASTRUCTURE_SHOCK: SectorType.INFRASTRUCTURE,
            ShockType.MANUFACTURING_SHOCK: SectorType.MANUFACTURING,
            ShockType.AGRICULTURE_SHOCK: SectorType.AGRICULTURE,
            ShockType.ENERGY_SHOCK: SectorType.ENERGY,
            ShockType.EXPORT_SHOCK: SectorType.EXPORT_ORIENTED,
            ShockType.SERVICES_SHOCK: SectorType.SERVICES,
            ShockType.MSME_SHOCK: SectorType.MSME,
            ShockType.TECHNOLOGY_SHOCK: SectorType.TECHNOLOGY,
            ShockType.RETAIL_SHOCK: SectorType.RETAIL_TRADE,
        }
        
        # Handle new sector-specific shocks
        if shock_type in shock_to_sector_map:
            sector_type = shock_to_sector_map[shock_type]
            shock_severity = ShockSeverity(severity) if severity else ShockSeverity.MODERATE
            
            # Get pre-defined shock scenario
            scenario = get_shock_scenario(sector_type, shock_severity)
            if scenario:
                # Apply shock to sector state
                sector_state = self.sector_states[sector_type]
                changes = sector_state.apply_shock(scenario)
                
                shock_event.update({
                    'sector': sector_type.value,
                    'trigger_event': scenario.trigger_event,
                    'indicator_changes': changes,
                    'sector_health': sector_state.indicators.economic_health,
                    'affected_banks': []
                })
                
                # Propagate to banks based on exposure
                total_losses = 0.0
                for agent in self.agents.values():
                    if isinstance(agent, BankAgent) and agent.alive:
                        # Assume banks have sector exposures (simplified)
                        # In real implementation, exposures would be tracked
                        bank_exposure = agent.risk_weighted_assets * 0.10  # Assume 10% exposure to shocked sector
                        
                        loss = compute_bank_loss_from_sector_shock(
                            sector_state,
                            bank_exposure,
                            agent.risk_weighted_assets
                        )
                        
                        if loss > 0:
                            agent.apply_shock(loss, source=f"sector_{sector_type.value}")
                            shock_event['affected_banks'].append({
                                'bank_id': agent.agent_id,
                                'loss': loss,
                                'exposure': bank_exposure
                            })
                            total_losses += loss
                
                shock_event['total_bank_losses'] = total_losses
                
                # Apply spillover to correlated sectors
                for correlated_sector in scenario.correlated_sectors:
                    spillover_state = self.sector_states[correlated_sector]
                    spillover_impact = scenario.spillover_intensity * 0.5
                    
                    spillover_state.indicators.demand_index *= (1.0 - spillover_impact * 0.3)
                    spillover_state.indicators.business_confidence *= (1.0 - spillover_impact * 0.4)
                    spillover_state.indicators.economic_health = spillover_state.indicators.compute_overall_health()
                
                # Only log severe shocks
                if scenario.severity in [ShockSeverity.SEVERE, ShockSeverity.CRISIS]:
                    logger.warning(
                    f"SHOCK: {scenario.trigger_event} | "
                    f"Sector: {sector_type.value} | "
                    f"Health: {sector_state.indicators.economic_health:.2%} | "
                    f"Bank losses: ${total_losses:,.0f}"
                )
                
                return shock_event
        
        # Handle legacy shocks (backward compatibility)
        if shock_type == ShockType.SECTOR_CRISIS:
            if target and target in self.agents:
                agent = self.agents[target]
                if isinstance(agent, SectorAgent):
                    agent.economic_health = max(0.0, agent.economic_health + magnitude)
                    shock_event['target'] = target
                    shock_event['new_health'] = agent.economic_health
                    logger.warning(f"SHOCK: {target} health dropped to {agent.economic_health:.2f}")
            else:
                # Random sector
                sector_agents = [a for a in self.agents.values() if isinstance(a, SectorAgent)]
                if sector_agents:
                    target_agent = np.random.choice(sector_agents)
                    target_agent.economic_health = max(0.0, target_agent.economic_health + magnitude)
                    shock_event['target'] = target_agent.agent_id
                    shock_event['new_health'] = target_agent.economic_health
        
        elif shock_type == ShockType.LIQUIDITY_SQUEEZE:
            # Global liquidity reduction
            self.global_state['system_liquidity'] *= (1.0 + magnitude)
            shock_event['new_liquidity'] = self.global_state['system_liquidity']
        
        elif shock_type == ShockType.INTEREST_RATE_SHOCK:
            # Spike in interest rates
            rate_change = abs(magnitude) * 300  # 300 bps
            self.global_state['base_repo_rate'] += rate_change
            shock_event['new_rate'] = self.global_state['base_repo_rate']
        
        elif shock_type == ShockType.ASSET_PRICE_CRASH:
            # All banks lose a fraction of capital
            for agent in self.agents.values():
                if isinstance(agent, BankAgent) and agent.alive:
                    loss = agent.capital * abs(magnitude)
                    agent.apply_shock(loss, source="asset_crash")
            shock_event['affected_banks'] = len([a for a in self.agents.values() if isinstance(a, BankAgent)])
        
        return shock_event
    
    def step(self) -> SimulationSnapshot:
        """
        Execute one time step of the simulation.
        
        Order of Operations:
        1. Stochastic Shock (optional)
        2. Regulator observes and acts
        3. All agents perceive (parallel)
        4. All agents decide (parallel)
        5. Banks act (sequential, to handle interdependencies)
        6. Sectors act (propagate losses)
        7. CCP acts (handle defaults)
        8. Contagion: Distribute losses across network
        9. Update global metrics
        10. Record snapshot
        """
        # logger.info(f"========== TIMESTEP {self.timestep} ==========")
        
        all_events: List[Dict[str, Any]] = []
        
        # Step 1: Stochastic Shock
        if self.config.enable_shocks and np.random.random() < self.config.shock_probability:
            shock_event = self.apply_shock(ShockType.SECTOR_CRISIS, magnitude=-0.15)
            all_events.append(shock_event)
        
        # Step 2: Regulator acts first (sets monetary policy)
        regulator_agents = [a for a in self.agents.values() if isinstance(a, RegulatorAgent)]
        for regulator in regulator_agents:
            events = regulator.step(self.network, self.global_state)
            all_events.extend(events)
        
        # Step 3 & 4: All agents perceive and decide (can be parallel)
        for agent in self.agents.values():
            if agent.alive and not isinstance(agent, RegulatorAgent):
                agent.perceive(self.network, self.global_state)
                agent.decide()
        
        # Step 5: Banks act (sequential for causal consistency)
        bank_agents = [a for a in self.agents.values() if isinstance(a, BankAgent)]
        for bank in bank_agents:
            events = bank.act(self.network)
            all_events.extend(events)
        
        # Step 6: Sectors act (propagate sector health to banks)
        sector_agents = [a for a in self.agents.values() if isinstance(a, SectorAgent)]
        for sector in sector_agents:
            events = sector.act(self.network)
            all_events.extend(events)
        
        # Step 7: CCP handles defaults
        ccp_agents = [a for a in self.agents.values() if isinstance(a, CCPAgent)]
        for ccp in ccp_agents:
            events = ccp.act(self.network)
            all_events.extend(events)
        
        # Step 7a: Dynamic network evolution - create new nodes (small probability)
        new_node_events = self._maybe_create_new_node()
        all_events.extend(new_node_events)
        
        # Step 7b: Dynamic edge creation - form new loans/transactions
        new_edge_events = self._maybe_create_new_edges()
        all_events.extend(new_edge_events)
        
        # Step 8: Contagion - Interbank loss propagation
        contagion_events = self._propagate_contagion()
        all_events.extend(contagion_events)
        
        # Step 9: Update global metrics
        self._update_global_metrics()
        
        # Step 10: Record snapshot
        snapshot = self._create_snapshot(all_events)
        self.history.append(snapshot)
        
        self.timestep += 1
        
        return snapshot
    
    def _maybe_create_new_node(self) -> List[Dict[str, Any]]:
        """
        Dynamically create a new node (bank/CCP) with small probability.
        Probability: 3% per timestep
        """
        events = []
        
        # Small chance to create a new bank (3%)
        if np.random.random() < 0.03:
            bank_count = len([a for a in self.agents.values() if isinstance(a, BankAgent)])
            new_bank_id = f"BANK_{bank_count + 1}"
            
            # Create new bank with realistic capital values (₹500K - ₹2M range)
            capital = np.random.uniform(500_000, 2_000_000)
            rwa = capital * np.random.uniform(8.0, 15.0)
            
            new_bank = BankAgent(
                agent_id=new_bank_id,
                initial_capital=capital,
                initial_assets=rwa,
                initial_liquidity=capital * np.random.uniform(0.15, 0.25),
                initial_npa_ratio=np.random.uniform(2.0, 5.0),
                initial_crar=np.random.uniform(10.0, 14.0),
                regulatory_min_crar=9.0
            )
            
            # Set policy variables after initialization
            new_bank.credit_supply_limit = capital * np.random.uniform(3, 6)
            new_bank.risk_appetite = np.random.uniform(0.4, 0.7)
            
            # Add to ecosystem
            self.agents[new_bank_id] = new_bank
            self.network.add_node(
                new_bank_id,
                agent_type=AgentType.BANK.value,
                agent=new_bank
            )
            
            # Connect to RBI/Regulator if it exists
            regulator_agents = [a for a in self.agents.values() if isinstance(a, RegulatorAgent)]
            if regulator_agents:
                regulator = regulator_agents[0]
                self.network.add_edge(regulator.agent_id, new_bank_id, weight=0, type='regulatory')
            
            # Connect to CCP if it exists
            ccp_agents = [a for a in self.agents.values() if isinstance(a, CCPAgent)]
            if ccp_agents:
                ccp = ccp_agents[0]
                self.network.add_edge(new_bank_id, ccp.agent_id, weight=capital * 0.1, type='clearing')
                ccp.active_exposures += capital * 0.1
            
            events.append({
                'type': 'NEW_BANK_CREATED',
                'bank_id': new_bank_id,
                'capital': capital,
                'timestep': self.timestep
            })
            
            # logger.info(f"NEW NODE: {new_bank_id} created with capital {capital:.2f}")
        
        return events
    
    def _maybe_create_new_edges(self) -> List[Dict[str, Any]]:
        """
        Dynamically create new loans/exposures between banks.
        Probability: 5% per timestep for each bank to make a new loan
        """
        events = []
        
        bank_agents = [a for a in self.agents.values() if isinstance(a, BankAgent) and a.alive]
        
        for bank in bank_agents:
            # Each bank has 5% chance to create new interbank loan
            if np.random.random() < 0.05:
                # Find potential borrowers (other alive banks not already connected strongly)
                potential_borrowers = [
                    b for b in bank_agents 
                    if b.agent_id != bank.agent_id and b.alive
                ]
                
                if potential_borrowers:
                    borrower = np.random.choice(potential_borrowers)
                    
                    # Loan size based on lender's capacity and risk appetite
                    max_loan = min(
                        bank.credit_supply_limit * 0.05,  # 5% of credit limit
                        bank.capital * bank.risk_appetite * 0.1  # 10% of risk-adjusted capital
                    )
                    
                    if max_loan > 10:  # Minimum viable loan
                        loan_amount = np.random.uniform(max_loan * 0.3, max_loan)
                        
                        # Add or update edge
                        if self.network.has_edge(bank.agent_id, borrower.agent_id):
                            # Increase existing exposure
                            current_weight = self.network.edges[bank.agent_id, borrower.agent_id]['weight']
                            self.network.edges[bank.agent_id, borrower.agent_id]['weight'] = current_weight + loan_amount
                        else:
                            # Create new edge
                            self.network.add_edge(
                                bank.agent_id,
                                borrower.agent_id,
                                weight=loan_amount,
                                type='interbank'
                            )
                        
                        # Update borrower's liquidity and lender's exposure
                        borrower.liquidity += loan_amount
                        borrower.debt_obligations += loan_amount * 0.05  # Interest burden
                        
                        events.append({
                            'type': 'NEW_LOAN_CREATED',
                            'lender': bank.agent_id,
                            'borrower': borrower.agent_id,
                            'amount': loan_amount,
                            'timestep': self.timestep
                        })
                        
                        # logger.info(f"NEW LOAN: {bank.agent_id} -> {borrower.agent_id}: {loan_amount:.2f}")
        
        return events
    
    def _propagate_contagion(self) -> List[Dict[str, Any]]:
        """
        Implement the cascading failure mechanism.
        
        Contagion Logic:
        - If Bank A defaults, all its creditors (Banks with edges A -> Creditor) take a haircut
        - The haircut is proportional to the exposure weight
        - This can trigger secondary failures (cascade)
        """
        contagion_events = []
        
        # Find all defaulted banks
        defaulted_banks = [agent for agent in self.agents.values() 
                          if isinstance(agent, BankAgent) and not agent.alive]
        
        for failed_bank in defaulted_banks:
            # Find all creditors (who lent to this bank)
            for creditor_id in self.network.predecessors(failed_bank.agent_id):
                edge_data = self.network.edges[creditor_id, failed_bank.agent_id]
                exposure = edge_data.get('weight', 0.0)
                
                if exposure > 0:
                    # Creditor takes a loss (haircut)
                    haircut_rate = 0.5  # 50% recovery rate
                    loss = exposure * haircut_rate
                    
                    creditor = self.agents.get(creditor_id)
                    if creditor and isinstance(creditor, BankAgent) and creditor.alive:
                        creditor.apply_shock(loss, source=f"contagion_from_{failed_bank.agent_id}")
                        
                        contagion_events.append({
                            'type': 'CONTAGION_LOSS',
                            'failed_bank': failed_bank.agent_id,
                            'creditor': creditor_id,
                            'exposure': exposure,
                            'loss': loss,
                            'new_capital': creditor.capital
                        })
                        
                        # Only log significant contagion events (>10% capital loss)
                        if loss > creditor.capital * 0.1:
                            logger.warning(f"CONTAGION: {creditor_id} lost {loss:.2f} from {failed_bank.agent_id} default")
        
        return contagion_events
    
    def _update_global_metrics(self) -> None:
        """
        Compute system-wide metrics that properly account for dead nodes.
        """
        banks = [a for a in self.agents.values() if isinstance(a, BankAgent)]
        
        if banks:
            alive_banks = [b for b in banks if b.alive]
            total_banks = len(banks)
            
            self.global_state['survival_rate'] = len(alive_banks) / total_banks if total_banks > 0 else 0.0
            self.global_state['avg_crar'] = np.mean([b.crar for b in alive_banks]) if alive_banks else 0.0
            self.global_state['total_defaults'] = total_banks - len(alive_banks)
            self.global_state['avg_npa'] = np.mean([b.npa_ratio for b in alive_banks]) if alive_banks else 0.0
            
            # System health: weighted average including dead nodes as 0
            # This properly penalizes system for defaults
            total_capital = sum(b.capital for b in alive_banks)
            if total_capital > 0:
                # Weight by capital for banks that are alive, dead banks contribute 0
                weighted_health = sum(b.compute_health() * b.capital for b in alive_banks)
                # Account for dead banks (they have 0 health but had capital before default)
                dead_banks = [b for b in banks if not b.alive]
                # Assume dead banks had average capital before default
                avg_initial_capital = total_capital / len(alive_banks) if alive_banks else 1000
                dead_capital_weight = len(dead_banks) * avg_initial_capital
                total_capital_including_dead = total_capital + dead_capital_weight
                
                self.global_state['system_health'] = weighted_health / total_capital_including_dead if total_capital_including_dead > 0 else 0.0
            else:
                # All banks are dead
                self.global_state['system_health'] = 0.0
            
            # Clamp system health to [0, 1]
            self.global_state['system_health'] = min(1.0, max(0.0, self.global_state['system_health']))
            
            # Market volatility proxy: variance in CRAR
            if len(alive_banks) > 1:
                self.global_state['market_volatility'] = np.std([b.crar for b in alive_banks]) / 10.0
            else:
                self.global_state['market_volatility'] = 1.0
            
            # Total system liquidity and capital
            self.global_state['system_liquidity'] = sum(b.liquidity for b in alive_banks)
            self.global_state['system_capital'] = total_capital
            self.global_state['system_npa'] = sum(b.npa_ratio * b.capital for b in alive_banks) / total_capital if total_capital > 0 else 0.0
    
    def _create_snapshot(self, events: List[Dict[str, Any]]) -> SimulationSnapshot:
        """
        Create a complete snapshot of the current state.
        """
        agent_states = {}
        for agent_id, agent in self.agents.items():
            agent_states[agent_id] = {
                'alive': agent.alive,
                'mode': agent.mode.value,
                'health': agent.compute_health(),
                'type': agent.agent_type.value
            }
            
            # Add type-specific data - use agent_type to avoid isinstance issues
            if agent.agent_type == AgentType.BANK:
                agent_states[agent_id].update({
                    'capital': float(agent.capital),
                    'crar': float(agent.crar),
                    'liquidity': float(agent.liquidity),
                    'npa_ratio': float(agent.npa_ratio),
                    'risk_appetite': float(agent.risk_appetite),
                    'credit_supply_limit': float(agent.credit_supply_limit),
                    'interbank_limit': float(agent.interbank_limit)
                })
            elif agent.agent_type == AgentType.SECTOR:
                agent_states[agent_id].update({
                    'economic_health': float(agent.economic_health),
                    'debt_load': float(agent.debt_load)
                })
            elif agent.agent_type == AgentType.CCP:
                agent_states[agent_id].update({
                    'default_fund_size': float(agent.default_fund_size),
                    'margin_buffer': float(agent.margin_buffer),
                    'initial_margin_requirement': float(agent.initial_margin_requirement)
                })
        
        # Network state for visualization
        network_state = {
            'nodes': [
                {
                    'id': node,
                    'type': self.network.nodes[node].get('agent_type'),
                    'alive': self.network.nodes[node].get('agent').alive,
                    'health': self.network.nodes[node].get('agent').compute_health()
                }
                for node in self.network.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 0),
                    'type': data.get('edge_type', 'loan')
                }
                for u, v, data in self.network.edges(data=True)
            ]
        }
        
        return SimulationSnapshot(
            timestep=self.timestep,
            network_state=network_state,
            agent_states=agent_states,
            events=events,
            global_metrics=self.global_state.copy()
        )
    
    def run(self, steps: int = None) -> List[SimulationSnapshot]:
        """
        Run the simulation for N steps.
        
        Args:
            steps: Number of steps to run (default: config.max_timesteps)
        
        Returns:
            List of snapshots for each timestep
        """
        steps = steps or self.config.max_timesteps
        # logger.info(f"Starting simulation run for {steps} steps")
        
        snapshots = []
        for _ in range(steps):
            # Check termination conditions
            banks = [a for a in self.agents.values() if isinstance(a, BankAgent)]
            alive_banks = [b for b in banks if b.alive]
            
            if len(alive_banks) == 0:
                logger.warning("All banks have defaulted. Terminating simulation.")
                break
            
            snapshot = self.step()
            snapshots.append(snapshot)
        
        # logger.info(f"Simulation completed. Final timestep: {self.timestep}")
        return snapshots
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Get network topology statistics.
        """
        return {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.network.degree()]) if self.network.number_of_nodes() > 0 else 0,
            'density': nx.density(self.network),
            'num_banks': len([a for a in self.agents.values() if isinstance(a, BankAgent)]),
            'num_sectors': len([a for a in self.agents.values() if isinstance(a, SectorAgent)]),
            'num_ccps': len([a for a in self.agents.values() if isinstance(a, CCPAgent)]),
        }
    
    def export_network(self, filepath: str, format: str = 'gexf') -> None:
        """
        Export network to file for external visualization.
        
        Args:
            filepath: Output file path
            format: 'gexf', 'graphml', 'json'
        """
        if format == 'gexf':
            nx.write_gexf(self.network, filepath)
        elif format == 'graphml':
            nx.write_graphml(self.network, filepath)
        elif format == 'json':
            from networkx.readwrite import json_graph
            import json
            data = json_graph.node_link_data(self.network)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        # logger.info(f"Network exported to {filepath} ({format} format)")
    
    def reset(self) -> None:
        """
        Reset the simulation to initial state.
        """
        self.timestep = 0
        self.history.clear()
        
        # Reset all agents
        for agent in self.agents.values():
            agent.alive = True
            agent.mode = AgentMode.NORMAL
            agent.timestep = 0
            agent.history.clear()
        
        # Reset global state
        self.global_state = {
            'system_liquidity': 1.0,
            'market_volatility': 0.0,
            'system_npa': 0.0,
            'base_repo_rate': 6.0
        }
        
        # logger.info("Simulation reset to initial state")


class SimulationFactory:
    """
    Factory for creating pre-configured simulation scenarios.
    """
    
    @staticmethod
    def create_default_scenario(config: SimulationConfig, user_entity: Optional[Dict[str, Any]] = None) -> FinancialEcosystem:
        """
        Create a simple default scenario with randomized parameters.
        """
        sim = FinancialEcosystem(config)
        
        # Randomize number of banks (15-25 for better performance)
        num_banks = np.random.randint(15, 26)
        
        # Add banks with realistic capital values (₹500K - ₹5M range)
        for i in range(num_banks):
            # Base capital scales with bank size: 500K + (i * 200K) with ±100K variation
            base_capital = 500_000 + (i * 200_000)
            capital = base_capital + np.random.uniform(-100_000, 100_000)
            
            # Risk-weighted assets: 8-14x capital (Basel norms)
            rwa = capital * np.random.uniform(8.0, 14.0)
            
            bank = BankAgent(
                agent_id=f"BANK_{i+1}",
                initial_capital=capital,
                initial_assets=rwa,
                initial_liquidity=capital * np.random.uniform(0.15, 0.30),  # 15-30% of capital
                initial_npa_ratio=np.random.uniform(2.0, 8.0),  # 2-8% NPA
                initial_crar=(capital / rwa) * 100
            )
            sim.add_agent(bank)
        
        # Add 20-30 sectors with random health (5x increase)
        num_sectors = np.random.randint(20, 31)
        sector_names = [
            ("SECTOR_REAL_ESTATE", "Real Estate"),
            ("SECTOR_COMMODITIES", "Commodities"),
            ("SECTOR_INFRASTRUCTURE", "Infrastructure"),
            ("SECTOR_MANUFACTURING", "Manufacturing"),
            ("SECTOR_TECHNOLOGY", "Technology"),
            ("SECTOR_RETAIL", "Retail"),
            ("SECTOR_AGRICULTURE", "Agriculture"),
            ("SECTOR_ENERGY", "Energy"),
            ("SECTOR_HEALTHCARE", "Healthcare"),
            ("SECTOR_TELECOM", "Telecommunications"),
            ("SECTOR_FINANCE", "Financial Services"),
            ("SECTOR_TRANSPORT", "Transportation"),
            ("SECTOR_MINING", "Mining"),
            ("SECTOR_CHEMICALS", "Chemicals"),
            ("SECTOR_AUTOMOTIVE", "Automotive"),
            ("SECTOR_CONSTRUCTION", "Construction"),
            ("SECTOR_UTILITIES", "Utilities"),
            ("SECTOR_MEDIA", "Media & Entertainment"),
            ("SECTOR_HOSPITALITY", "Hospitality & Tourism"),
            ("SECTOR_EDUCATION", "Education"),
            ("SECTOR_PHARMA", "Pharmaceuticals"),
            ("SECTOR_AEROSPACE", "Aerospace & Defense"),
            ("SECTOR_TEXTILES", "Textiles"),
            ("SECTOR_FOOD_PROCESSING", "Food Processing"),
            ("SECTOR_IT_SERVICES", "IT Services"),
            ("SECTOR_ECOMMERCE", "E-Commerce"),
            ("SECTOR_LOGISTICS", "Logistics"),
            ("SECTOR_INSURANCE", "Insurance"),
            ("SECTOR_REAL_ESTATE_COMM", "Commercial Real Estate"),
            ("SECTOR_METALS", "Metals & Steel"),
        ]
        
        for i in range(num_sectors):
            sector = SectorAgent(
                sector_names[i][0], 
                sector_names[i][1], 
                initial_health=np.random.uniform(0.4, 0.9)
            )
            sim.add_agent(sector)
        
        # Add 20-25 CCPs (Central Counterparties / Clearing Houses)
        num_ccps = np.random.randint(20, 26)
        ccp_names = [
            ("NSCCL", "NSE Clearing"),
            ("ICCL", "BSE Clearing"),
            ("MCXCCL", "MCX Clearing"),
            ("NCDEX_CL", "NCDEX Clearing"),
            ("CDSL", "Central Depository Services"),
            ("NSDL", "National Securities Depository"),
            ("CCIL", "Clearing Corporation of India"),
            ("ICEX_CL", "ICEX Clearing"),
            ("DGCX_CL", "Dubai Gold Clearing"),
            ("LME_CL", "London Metal Exchange Clearing"),
            ("CME_CL", "CME Clearing"),
            ("EUREX_CL", "Eurex Clearing"),
            ("ICE_CL", "ICE Clear"),
            ("LCH_CL", "LCH Clearnet"),
            ("OCC_CL", "Options Clearing Corp"),
            ("DTCC_CL", "DTCC Clearing"),
            ("JSCC_CL", "Japan Securities Clearing"),
            ("HKEX_CL", "HKEX Clearing"),
            ("SGX_CL", "SGX Clearing"),
            ("ASX_CL", "ASX Clear"),
            ("BME_CL", "BME Clearing"),
            ("SIX_CL", "SIX x-clear"),
            ("KELER_CL", "KELER Clearing"),
            ("CDS_CL", "Canadian Derivatives Clearing"),
            ("BMV_CL", "BMV Clearing"),
        ]
        for i in range(num_ccps):
            ccp_id, ccp_name = ccp_names[i % len(ccp_names)]
            if i >= len(ccp_names):
                ccp_id = f"{ccp_id}_{i}"
            # CCPs have large default funds: ₹5M - ₹7M range
            default_fund = 5_000_000 + np.random.uniform(-1_000_000, 2_000_000)
            ccp = CCPAgent(ccp_id, initial_default_fund=default_fund)
            sim.add_agent(ccp)
        
        # Add Multiple Regulators (RBI, SEBI, IRDAI, PFRDA)
        regulators = [
            ("RBI", 6.0),  # Reserve Bank of India - Banking
            ("SEBI", 6.0),  # Securities and Exchange Board - Capital Markets
            ("IRDAI", 6.0),  # Insurance Regulatory Authority
            ("PFRDA", 6.0),  # Pension Fund Regulatory Authority
        ]
        for reg_id, rate in regulators:
            regulator = RegulatorAgent(reg_id, base_repo_rate=rate)
            sim.add_agent(regulator)
        
        # Create randomized exposures (Banks -> Sectors)
        bank_ids = [f"BANK_{i+1}" for i in range(num_banks)]
        sector_ids = [sector_names[i][0] for i in range(num_sectors)]
        
        for bank_id in bank_ids:
            bank = sim.agents[bank_id]
            # Each bank lends to 2-3 random sectors
            num_sector_loans = min(num_sectors, np.random.randint(2, num_sectors + 1))
            selected_sectors = np.random.choice(sector_ids, size=num_sector_loans, replace=False)
            
            for sector_id in selected_sectors:
                exposure = bank.capital * np.random.uniform(1.5, 3.5)
                sim.add_exposure(bank_id, sector_id, exposure_amount=exposure)
        
        # Create randomized interbank network
        for i in range(len(bank_ids) - 1):
            if np.random.random() > 0.3:  # 70% chance of connection
                creditor_id = bank_ids[i]
                debtor_id = bank_ids[i + 1]
                exposure = sim.agents[creditor_id].interbank_limit * np.random.uniform(0.2, 0.6)
                sim.add_exposure(creditor_id, debtor_id, exposure_amount=exposure)
        
        # Add some random additional interbank links
        for _ in range(np.random.randint(0, 3)):
            creditor_idx = np.random.randint(0, len(bank_ids))
            debtor_idx = np.random.randint(0, len(bank_ids))
            if creditor_idx != debtor_idx:
                creditor_id = bank_ids[creditor_idx]
                debtor_id = bank_ids[debtor_idx]
                exposure = sim.agents[creditor_id].interbank_limit * np.random.uniform(0.1, 0.4)
                sim.add_exposure(creditor_id, debtor_id, exposure_amount=exposure)
        
        # Add user entity if provided
        if user_entity:
            SimulationFactory.add_user_entity(sim, user_entity)
        
        # logger.info(f"Created default scenario with {num_banks} banks, {num_sectors} sectors, {num_ccps} CCPs, 4 regulators")
        return sim
    
    @staticmethod
    def add_user_entity(sim: FinancialEcosystem, user_entity: Dict[str, Any]) -> None:
        """
        Add a user-controlled entity to the simulation.
        """
        entity_type = user_entity.get('type', 'bank')
        entity_id = user_entity.get('id', 'USER_ENTITY')
        policies = user_entity.get('policies', {})
        
        if entity_type == 'bank':
            # Create user bank with realistic capital (₹1.3M - ₹1.7M)
            capital = 1_500_000 + np.random.uniform(-200_000, 200_000)
            rwa = capital * 10
            bank = BankAgent(
                agent_id=entity_id,
                initial_capital=capital,
                initial_assets=rwa,
                initial_liquidity=capital * (policies.get('liquidityBuffer', 15) / 100),
                initial_npa_ratio=policies.get('npaThreshold', 5.0),
                initial_crar=policies.get('minCapitalRatio', 11.5)
            )
            bank.risk_appetite = policies.get('riskAppetite', 0.6)
            bank.max_exposure_per_counterparty = policies.get('maxExposurePerCounterparty', 25) / 100
            bank.is_user_controlled = True  # Mark as user-controlled for decision alerts
            sim.add_agent(bank)
            
            # Connect user bank to random sectors and other banks
            sector_agents = [a for a in sim.agents.values() if isinstance(a, SectorAgent)]
            bank_agents = [a for a in sim.agents.values() if isinstance(a, BankAgent) and a.agent_id != entity_id]
            
            # Lend to 2-3 sectors
            if sector_agents:
                num_sector_loans = min(len(sector_agents), np.random.randint(2, 4))
                selected_sectors = np.random.choice(sector_agents, size=num_sector_loans, replace=False)
                for sector in selected_sectors:
                    exposure = capital * np.random.uniform(1.5, 3.0)
                    sim.add_exposure(entity_id, sector.agent_id, exposure_amount=exposure)
            
            # Interbank connections (3-5 banks)
            if bank_agents:
                num_connections = min(len(bank_agents), np.random.randint(3, 6))
                selected_banks = np.random.choice(bank_agents, size=num_connections, replace=False)
                for other_bank in selected_banks:
                    if np.random.random() > 0.5:
                        # Lend to other bank
                        exposure = bank.interbank_limit * np.random.uniform(0.2, 0.5)
                        sim.add_exposure(entity_id, other_bank.agent_id, exposure_amount=exposure)
                    else:
                        # Borrow from other bank
                        exposure = other_bank.interbank_limit * np.random.uniform(0.2, 0.5)
                        sim.add_exposure(other_bank.agent_id, entity_id, exposure_amount=exposure)
        
        elif entity_type == 'clearing_house':
            # Create user CCP with specified policies
            default_fund = policies.get('defaultFundSize', 5000000)
            ccp = CCPAgent(entity_id, initial_default_fund=default_fund)
            ccp.initial_margin = policies.get('initialMargin', 10) / 100
            ccp.haircut_rate = policies.get('haircut', 5) / 100
            sim.add_agent(ccp)
        
        elif entity_type == 'regulator':
            # Create user regulator with specified policies
            regulator = RegulatorAgent(
                entity_id,
                base_repo_rate=policies.get('baseRepoRate', 6.5)
            )
            regulator.minimum_crar = policies.get('minimumCRAR', 9.0)
            regulator.crisis_threshold = policies.get('crisisInterventionThreshold', 0.6)
            sim.add_agent(regulator)
        
        elif entity_type == 'sector':
            # Create user sector with specified policies
            sector = SectorAgent(
                entity_id,
                "User Sector",
                initial_health=policies.get('economicHealth', 0.8)
            )
            sector.debt_load = policies.get('debtLoad', 45) / 100
            sim.add_agent(sector)
            
            # Connect random banks to this sector
            bank_agents = [a for a in sim.agents.values() if isinstance(a, BankAgent)]
            num_connections = min(len(bank_agents), np.random.randint(5, 12))
            selected_banks = np.random.choice(bank_agents, size=num_connections, replace=False)
            for bank in selected_banks:
                exposure = bank.capital * np.random.uniform(1.0, 2.5)
                sim.add_exposure(bank.agent_id, entity_id, exposure_amount=exposure)
        
        # logger.info(f"Added user entity: {entity_id} ({entity_type})")
