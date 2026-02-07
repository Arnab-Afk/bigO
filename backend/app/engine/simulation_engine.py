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

logger = logging.getLogger(__name__)


class ShockType(Enum):
    """Types of exogenous shocks"""
    SECTOR_CRISIS = "sector_crisis"
    LIQUIDITY_SQUEEZE = "liquidity_squeeze"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    ASSET_PRICE_CRASH = "asset_price_crash"
    NONE = "none"


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    max_timesteps: int = 100
    enable_shocks: bool = True
    shock_probability: float = 0.1
    contagion_multiplier: float = 1.0
    enable_regulator: bool = True
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
        
        # Initialize random seed for reproducibility
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        logger.info(f"FinancialEcosystem initialized with config: {config}")
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the ecosystem.
        """
        self.agents[agent.agent_id] = agent
        self.network.add_node(
            agent.agent_id,
            agent=agent,
            agent_type=agent.agent_type.value,
            alive=agent.alive
        )
        logger.info(f"Added {agent.agent_type.value} agent: {agent.agent_id}")
    
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
    
    def apply_shock(self, shock_type: ShockType, target: Optional[str] = None, magnitude: float = -0.2) -> Dict[str, Any]:
        """
        Apply an exogenous shock to the system.
        
        Args:
            shock_type: Type of shock
            target: Target agent/sector ID (if applicable)
            magnitude: Shock magnitude (typically negative for crisis)
        
        Returns:
            Dict describing the shock event
        """
        shock_event = {
            'type': 'EXOGENOUS_SHOCK',
            'shock_type': shock_type.value,
            'timestep': self.timestep,
            'magnitude': magnitude
        }
        
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
        logger.info(f"========== TIMESTEP {self.timestep} ==========")
        
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
                        
                        logger.warning(f"CONTAGION: {creditor_id} lost {loss:.2f} from {failed_bank.agent_id} default")
        
        return contagion_events
    
    def _update_global_metrics(self) -> None:
        """
        Compute system-wide metrics.
        """
        banks = [a for a in self.agents.values() if isinstance(a, BankAgent)]
        
        if banks:
            alive_banks = [b for b in banks if b.alive]
            total_banks = len(banks)
            
            self.global_state['survival_rate'] = len(alive_banks) / total_banks
            self.global_state['avg_crar'] = np.mean([b.crar for b in alive_banks]) if alive_banks else 0.0
            self.global_state['total_defaults'] = total_banks - len(alive_banks)
            self.global_state['avg_npa'] = np.mean([b.npa_ratio for b in alive_banks]) if alive_banks else 0.0
            
            # Market volatility proxy: variance in CRAR
            if len(alive_banks) > 1:
                self.global_state['market_volatility'] = np.std([b.crar for b in alive_banks]) / 10.0
            else:
                self.global_state['market_volatility'] = 1.0
    
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
            
            # Add type-specific data
            if isinstance(agent, BankAgent):
                agent_states[agent_id].update({
                    'capital': agent.capital,
                    'crar': agent.crar,
                    'liquidity': agent.liquidity,
                    'npa_ratio': agent.npa_ratio,
                    'risk_appetite': agent.risk_appetite
                })
            elif isinstance(agent, SectorAgent):
                agent_states[agent_id].update({
                    'economic_health': agent.economic_health,
                    'debt_load': agent.debt_load
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
        logger.info(f"Starting simulation run for {steps} steps")
        
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
        
        logger.info(f"Simulation completed. Final timestep: {self.timestep}")
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
        
        logger.info(f"Network exported to {filepath} ({format} format)")
    
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
        
        logger.info("Simulation reset to initial state")


class SimulationFactory:
    """
    Factory for creating pre-configured simulation scenarios.
    """
    
    @staticmethod
    def create_default_scenario(config: SimulationConfig) -> FinancialEcosystem:
        """
        Create a simple default scenario for testing.
        """
        sim = FinancialEcosystem(config)
        
        # Add 5 banks
        for i in range(5):
            bank = BankAgent(
                agent_id=f"BANK_{i+1}",
                initial_capital=1000 + i * 200,
                initial_assets=10000 + i * 2000,
                initial_liquidity=500 + i * 100,
                initial_npa_ratio=2.0 + i * 0.5,
                initial_crar=12.0 + i
            )
            sim.add_agent(bank)
        
        # Add 2 sectors
        real_estate = SectorAgent("SECTOR_REAL_ESTATE", "Real Estate", initial_health=0.8)
        commodities = SectorAgent("SECTOR_COMMODITIES", "Commodities", initial_health=0.75)
        sim.add_agent(real_estate)
        sim.add_agent(commodities)
        
        # Add CCP
        ccp = CCPAgent("CCP_MAIN", initial_default_fund=5000)
        sim.add_agent(ccp)
        
        # Add Regulator
        regulator = RegulatorAgent("RBI", base_repo_rate=6.0)
        sim.add_agent(regulator)
        
        # Create exposures (Banks -> Sectors)
        for i in range(5):
            sim.add_exposure(f"BANK_{i+1}", "SECTOR_REAL_ESTATE", exposure_amount=2000 + i * 500)
            sim.add_exposure(f"BANK_{i+1}", "SECTOR_COMMODITIES", exposure_amount=1500 + i * 300)
        
        # Interbank exposures
        sim.add_exposure("BANK_1", "BANK_2", exposure_amount=500)
        sim.add_exposure("BANK_2", "BANK_3", exposure_amount=600)
        sim.add_exposure("BANK_3", "BANK_4", exposure_amount=400)
        
        return sim
