"""
RUDRA Computational Engine

Core game-theoretic, network analysis, and simulation modules.

Includes:
- Legacy static analysis modules (game_theory, network, contagion, simulation, bayesian)
- NEW: Agent-Based Model (ABM) for dynamic time-stepped simulations
"""

from app.engine.game_theory import (
    AgentUtility,
    NashEquilibriumSolver,
    AgentAction,
    ActionType,
)
from app.engine.network import NetworkAnalyzer, NetworkMetrics
from app.engine.contagion import ContagionPropagator, ContagionMechanism
from app.engine.simulation import SimulationEngine, SimulationState
from app.engine.bayesian import BayesianBeliefUpdater, SignalProcessor

# Agent-Based Model components
from app.engine.simulation_engine import (
    FinancialEcosystem,
    SimulationConfig,
    SimulationSnapshot,
    SimulationFactory,
    ShockType
)
from app.engine.agents import (
    Agent,
    BankAgent,
    CCPAgent,
    SectorAgent,
    RegulatorAgent,
    AgentType,
    AgentMode
)
from app.engine.initial_state_loader import (
    InitialStateLoader,
    load_ecosystem_from_data
)
from app.engine.visualization import (
    NetworkVisualizer,
    prepare_dashboard_data
)

__all__ = [
    # Legacy modules
    "AgentUtility",
    "NashEquilibriumSolver",
    "AgentAction",
    "ActionType",
    "NetworkAnalyzer",
    "NetworkMetrics",
    "ContagionPropagator",
    "ContagionMechanism",
    "SimulationEngine",
    "SimulationState",
    "BayesianBeliefUpdater",
    "SignalProcessor",
    
    # Agent-Based Model
    "FinancialEcosystem",
    "SimulationConfig",
    "SimulationSnapshot",
    "SimulationFactory",
    "ShockType",
    "Agent",
    "BankAgent",
    "CCPAgent",
    "SectorAgent",
    "RegulatorAgent",
    "AgentType",
    "AgentMode",
    "InitialStateLoader",
    "load_ecosystem_from_data",
    "NetworkVisualizer",
    "prepare_dashboard_data",
]
