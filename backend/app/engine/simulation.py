"""
Simulation Execution Engine

Orchestrates the complete simulation loop integrating game theory, network analysis,
and contagion propagation.
Based on Technical Documentation Section 7.2.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID
import logging

import networkx as nx
import numpy as np

from app.engine.game_theory import (
    AgentAction,
    AgentState,
    AgentUtility,
    NashEquilibriumSolver,
    generate_action_space,
)
from app.engine.contagion import (
    ContagionPropagator,
    PropagationState,
    DefaultEvent,
    CascadeRound,
)
from app.engine.bayesian import BayesianBeliefUpdater, SignalProcessor
from app.engine.network import NetworkAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Shock:
    """Exogenous shock to the system"""
    shock_id: str
    shock_type: str
    target_institutions: List[UUID]
    magnitude: float
    parameters: Dict = field(default_factory=dict)


@dataclass
class TimestepState:
    """Complete state at a single timestep"""
    timestep: int
    agent_states: Dict[UUID, AgentState]
    agent_actions: Dict[UUID, AgentAction]
    defaults: List[DefaultEvent]
    propagation_state: PropagationState
    network_metrics: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SimulationState:
    """Complete simulation state and history"""
    simulation_id: str
    timesteps: List[TimestepState]
    cascade_history: List[CascadeRound]
    final_defaults: Set[UUID]
    total_losses: float
    converged: bool
    convergence_step: Optional[int] = None


class SimulationEngine:
    """
    Main simulation orchestrator
    
    Executes the discrete-time simulation loop:
    1. Apply shocks
    2. Agent decision phase (game theory)
    3. Action execution
    4. Propagation phase (contagion)
    5. Default detection and cascade
    """
    
    def __init__(
        self,
        network: nx.DiGraph,
        convergence_threshold: float = 1e-6,
        max_timesteps: int = 100,
        enable_ml: bool = False,
    ):
        """
        Args:
            network: Financial network graph
            convergence_threshold: Threshold for early stopping
            max_timesteps: Maximum simulation timesteps
            enable_ml: Whether to use ML predictions
        """
        self.network = network
        self.convergence_threshold = convergence_threshold
        self.max_timesteps = max_timesteps
        self.enable_ml = enable_ml
        
        # Initialize sub-engines
        self.nash_solver = NashEquilibriumSolver(tolerance=convergence_threshold)
        self.contagion_propagator = ContagionPropagator(network)
        self.belief_updater = BayesianBeliefUpdater()
        self.signal_processor = SignalProcessor()
        self.network_analyzer = NetworkAnalyzer(network)
        
        # Initialize ML predictor if enabled
        self.ml_predictor = None
        if enable_ml:
            try:
                from app.ml.inference.predictor import DefaultPredictor
                from app.ml.features.extractor import FeatureExtractor
                from app.ml.config import ml_config
                from pathlib import Path
                
                model_path = ml_config.ML_MODELS_PATH / "default_predictor" / "best_model.pt"
                feature_extractor = FeatureExtractor(network_analyzer=self.network_analyzer)
                self.ml_predictor = DefaultPredictor(
                    model_path=model_path,
                    feature_extractor=feature_extractor,
                )
                logger.info("ML predictor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML predictor: {e}. Using fallback.")
                self.ml_predictor = None
    
    def run_simulation(
        self,
        simulation_id: str,
        initial_states: Dict[UUID, AgentState],
        shocks: List[Shock],
        shock_timing: Dict[int, List[str]],
    ) -> SimulationState:
        """
        Execute complete simulation run
        
        Args:
            simulation_id: Unique simulation identifier
            initial_states: Starting states for all agents
            shocks: List of shock definitions
            shock_timing: Mapping of timestep -> shock IDs
        
        Returns:
            Complete simulation state with history
        """
        # Initialize
        current_states = initial_states.copy()
        history: List[TimestepState] = []
        all_cascades: List[CascadeRound] = []
        all_defaults: Set[UUID] = set()
        
        # Initialize beliefs
        self.belief_updater.initialize_beliefs(list(current_states.keys()))
        
        # Record initial state
        initial_timestep = TimestepState(
            timestep=0,
            agent_states=current_states.copy(),
            agent_actions={},
            defaults=[],
            propagation_state=self._create_propagation_state(current_states),
            network_metrics=self._compute_network_metrics(),
        )
        history.append(initial_timestep)
        
        # Main simulation loop
        for t in range(1, self.max_timesteps + 1):
            # 1. Apply scheduled shocks
            if t in shock_timing:
                for shock_id in shock_timing[t]:
                    shock = self._get_shock(shocks, shock_id)
                    if shock:
                        current_states = self._apply_shock(current_states, shock)
            
            # 1.5. Update default probabilities with ML predictions (if enabled)
            if self.enable_ml and self.ml_predictor:
                current_states = self._update_ml_predictions(current_states, all_defaults)
            
            # 2. Agent decision phase
            decisions = self._agent_decision_phase(current_states, t)
            
            # 3. Execute actions
            current_states = self._execute_actions(current_states, decisions)
            
            # 4. Propagation phase
            prop_state = self._create_propagation_state(current_states)
            
            # Detect any stressed institutions for propagation
            stressed = [
                inst_id for inst_id, state in current_states.items()
                if state.is_stressed() and inst_id not in all_defaults
            ]
            
            if stressed:
                prop_state, cascades = self.contagion_propagator.propagate_shock(
                    initial_state=prop_state,
                    shocked_institutions=stressed,
                    max_rounds=5
                )
                
                # Update states from propagation
                current_states = self._update_from_propagation(
                    current_states, prop_state
                )
                
                all_cascades.extend(cascades)
            else:
                cascades = []
            
            # 5. Collect defaults
            new_defaults = [
                DefaultEvent(
                    institution_id=inst_id,
                    timestep=t,
                    cause="simulation",
                )
                for inst_id in prop_state.defaulted
                if inst_id not in all_defaults
            ]
            all_defaults.update(prop_state.defaulted)
            
            # 6. Update beliefs based on observations
            self._update_beliefs(current_states, new_defaults)
            
            # Record timestep
            timestep_state = TimestepState(
                timestep=t,
                agent_states=current_states.copy(),
                agent_actions=decisions,
                defaults=new_defaults,
                propagation_state=prop_state,
                network_metrics=self._compute_network_metrics(),
            )
            history.append(timestep_state)
            
            # Check convergence
            if self._check_convergence(history):
                return SimulationState(
                    simulation_id=simulation_id,
                    timesteps=history,
                    cascade_history=all_cascades,
                    final_defaults=all_defaults,
                    total_losses=self._calculate_total_losses(all_cascades),
                    converged=True,
                    convergence_step=t,
                )
        
        # Completed without convergence
        return SimulationState(
            simulation_id=simulation_id,
            timesteps=history,
            cascade_history=all_cascades,
            final_defaults=all_defaults,
            total_losses=self._calculate_total_losses(all_cascades),
            converged=False,
        )
    
    def _agent_decision_phase(
        self,
        states: Dict[UUID, AgentState],
        timestep: int
    ) -> Dict[UUID, AgentAction]:
        """
        Each agent computes optimal action given beliefs and network state
        """
        decisions = {}
        
        # Get exposure graph
        exposures = self._get_exposure_dict()
        
        # Generate action spaces
        action_spaces = {}
        for agent_id, state in states.items():
            counterparties = list(self.network.successors(agent_id))
            action_spaces[agent_id] = generate_action_space(
                agent_id, state, counterparties
            )
        
        # Compute Nash equilibrium (or best responses)
        equilibrium = self.nash_solver.solve_pure_nash(
            agents=states,
            exposures=exposures,
            action_spaces=action_spaces,
        )
        
        if equilibrium:
            decisions = equilibrium
        else:
            # Fallback: each agent plays maintain status
            for agent_id in states.keys():
                decisions[agent_id] = AgentAction(
                    action_type="maintain_status",
                    agent_id=agent_id,
                )
        
        return decisions
    
    def _execute_actions(
        self,
        states: Dict[UUID, AgentState],
        actions: Dict[UUID, AgentAction]
    ) -> Dict[UUID, AgentState]:
        """
        Apply agent actions to update states
        """
        new_states = {k: v for k, v in states.items()}
        
        for agent_id, action in actions.items():
            state = new_states[agent_id]
            
            if action.action_type == "adjust_credit_limit":
                # Adjust exposure
                if action.magnitude > 0:
                    state.credit_exposure += action.magnitude
                else:
                    state.credit_exposure = max(0, state.credit_exposure + action.magnitude)
            
            elif action.action_type == "liquidity_decision":
                decision = action.parameters.get("decision")
                if decision == "HOARD":
                    # Increase liquidity buffer
                    state.liquidity_buffer = min(1.0, state.liquidity_buffer + 0.1)
                elif decision == "RELEASE":
                    # Decrease liquidity buffer but gain revenue
                    state.liquidity_buffer = max(0.0, state.liquidity_buffer - 0.05)
        
        return new_states
    
    def _apply_shock(
        self,
        states: Dict[UUID, AgentState],
        shock: Shock
    ) -> Dict[UUID, AgentState]:
        """Apply exogenous shock to system"""
        new_states = {k: v for k, v in states.items()}
        
        for inst_id in shock.target_institutions:
            if inst_id not in new_states:
                continue
            
            state = new_states[inst_id]
            
            if shock.shock_type == "institution_default":
                # Force default
                state.capital_ratio = 0.0
                state.default_probability = 1.0
            
            elif shock.shock_type == "liquidity_freeze":
                # Drain liquidity
                state.liquidity_buffer = max(0.0, state.liquidity_buffer - shock.magnitude)
            
            elif shock.shock_type == "market_volatility":
                # Increase stress
                state.stress_level = min(1.0, state.stress_level + shock.magnitude)
            
            elif shock.shock_type == "credit_downgrade":
                # Increase default probability
                state.default_probability = min(
                    1.0, state.default_probability + shock.magnitude
                )
        
        return new_states
    
    def _create_propagation_state(
        self,
        agent_states: Dict[UUID, AgentState]
    ) -> PropagationState:
        """Convert agent states to propagation state"""
        return PropagationState(
            capital_levels={
                inst_id: state.capital_ratio * 10000  # Denormalize
                for inst_id, state in agent_states.items()
            },
            liquidity_levels={
                inst_id: state.liquidity_buffer
                for inst_id, state in agent_states.items()
            },
            stress_levels={
                inst_id: state.stress_level
                for inst_id, state in agent_states.items()
            },
            defaulted={
                inst_id
                for inst_id, state in agent_states.items()
                if state.capital_ratio <= 0
            },
        )
    
    def _update_from_propagation(
        self,
        agent_states: Dict[UUID, AgentState],
        prop_state: PropagationState
    ) -> Dict[UUID, AgentState]:
        """Update agent states from propagation results"""
        new_states = {k: v for k, v in agent_states.items()}
        
        for inst_id, state in new_states.items():
            state.capital_ratio = prop_state.capital_levels.get(inst_id, 0) / 10000
            state.liquidity_buffer = prop_state.liquidity_levels.get(inst_id, 0)
            state.stress_level = prop_state.stress_levels.get(inst_id, 0)
            
            if inst_id in prop_state.defaulted:
                state.default_probability = 1.0
        
        return new_states
    
    def _update_ml_predictions(
        self,
        states: Dict[UUID, AgentState],
        defaulted_institutions: Set[UUID],
    ) -> Dict[UUID, AgentState]:
        """
        Update default probabilities using ML predictions
        
        Args:
            states: Current agent states
            defaulted_institutions: Set of defaulted institutions
        
        Returns:
            Updated states with ML predictions
        """
        if not self.ml_predictor:
            return states
        
        try:
            # Batch prediction for all non-defaulted institutions
            non_defaulted_states = {
                inst_id: state for inst_id, state in states.items()
                if inst_id not in defaulted_institutions
            }
            
            if not non_defaulted_states:
                return states
            
            # Get ML predictions
            predictions = self.ml_predictor.predict_batch(
                agent_states=non_defaulted_states,
                network=self.network,
                defaulted_institutions=defaulted_institutions,
            )
            
            # Update states with ML predictions
            updated_states = states.copy()
            for inst_id, prediction in predictions.items():
                if self.ml_predictor.should_use_ml_prediction(prediction):
                    # Use ML prediction
                    updated_states[inst_id].default_probability = prediction.probability
                    updated_states[inst_id].ml_prediction_confidence = prediction.confidence
                    updated_states[inst_id].ml_model_version = prediction.model_version
                else:
                    # Keep prior/bayesian prediction
                    updated_states[inst_id].ml_prediction_confidence = prediction.confidence
                    updated_states[inst_id].ml_model_version = f"{prediction.model_version}_low_conf"
            
            return updated_states
            
        except Exception as e:
            logger.error(f"Failed to update ML predictions: {e}")
            return states
    
    def _update_beliefs(
        self,
        states: Dict[UUID, AgentState],
        defaults: List[DefaultEvent]
    ):
        """Update Bayesian beliefs based on observations"""
        # Generate signals from defaults
        for default_event in defaults:
            # All agents observe defaults
            for observer_id in states.keys():
                if observer_id != default_event.institution_id:
                    signal = self.signal_processor.generate_signal(
                        true_state="defaulted",
                        observed_by=observer_id,
                        about=default_event.institution_id,
                        signal_type="default_event",
                    )
                    self.belief_updater.update_belief(
                        default_event.institution_id, signal
                    )
    
    def _compute_network_metrics(self) -> Dict:
        """Compute current network metrics"""
        metrics = self.network_analyzer.compute_network_metrics()
        return {
            "node_count": metrics.node_count,
            "edge_count": metrics.edge_count,
            "density": metrics.density,
            "concentration_index": metrics.concentration_index,
            "interconnectedness": metrics.interconnectedness_score,
        }
    
    def _get_exposure_dict(self) -> Dict[Tuple[UUID, UUID], float]:
        """Extract exposure values from network"""
        exposures = {}
        for source, target, data in self.network.edges(data=True):
            exposures[(source, target)] = data.get('exposure_magnitude', 0.0)
        return exposures
    
    def _get_shock(self, shocks: List[Shock], shock_id: str) -> Optional[Shock]:
        """Find shock by ID"""
        for shock in shocks:
            if shock.shock_id == shock_id:
                return shock
        return None
    
    def _check_convergence(self, history: List[TimestepState]) -> bool:
        """Check if simulation has converged"""
        if len(history) < 5:
            return False
        
        # Check if states are stable over last 5 timesteps
        recent = history[-5:]
        
        # Calculate variance in key metrics
        stress_variance = np.var([
            np.mean([s.stress_level for s in ts.agent_states.values()])
            for ts in recent
        ])
        
        return stress_variance < self.convergence_threshold
    
    def _calculate_total_losses(self, cascades: List[CascadeRound]) -> float:
        """Calculate total system losses from cascades"""
        return sum(cascade.total_losses for cascade in cascades)
