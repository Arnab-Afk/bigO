"""
Agent-Based Model (ABM) Simulation API
=====================================
FastAPI endpoints for real-time strategic financial network simulation.

Endpoints:
- POST /abm/initialize - Create a new simulation instance
- POST /abm/{sim_id}/step - Advance one time step
- POST /abm/{sim_id}/run - Run multiple steps
- GET /abm/{sim_id}/state - Get current state
- POST /abm/{sim_id}/shock - Apply exogenous shock
- POST /abm/{sim_id}/policy - Update CCP policy rules
- GET /abm/{sim_id}/history - Get simulation history
- DELETE /abm/{sim_id}/reset - Reset simulation
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import uuid4
import logging
from pathlib import Path

from app.engine.simulation_engine import (
    FinancialEcosystem,
    SimulationConfig,
    SimulationSnapshot,
    ShockType,
    SimulationFactory
)
from app.engine.initial_state_loader import load_ecosystem_from_data
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for active simulations
# In production, use Redis or a database
ACTIVE_SIMULATIONS: Dict[str, FinancialEcosystem] = {}
PENDING_DECISIONS: Dict[str, Dict[str, Any]] = {}  # Store user decisions waiting for approval
DECISION_COOLDOWNS: Dict[str, Dict[str, int]] = {}  # Track last decision timestep per agent per sim


# ============================================================================
# Request/Response Schemas
# ============================================================================

class SimulationInitRequest(BaseModel):
    """Request to initialize a new simulation"""
    name: str = Field(..., description="Simulation name")
    max_timesteps: int = Field(100, ge=1, le=1000, description="Maximum timesteps")
    enable_shocks: bool = Field(True, description="Enable random shocks")
    shock_probability: float = Field(0.1, ge=0.0, le=1.0, description="Probability of shock per step")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    use_real_data: bool = Field(True, description="Use CSV data or synthetic data")
    data_source: str = Field("backend/ccp_ml/data", description="Path to CSV data directory")
    user_entity: Optional[Dict[str, Any]] = Field(None, description="User-controlled entity parameters")


class SimulationInitResponse(BaseModel):
    """Response after initialization"""
    simulation_id: str
    name: str
    config: Dict[str, Any]
    network_stats: Dict[str, Any]
    initial_state: Dict[str, Any]


class StepRequest(BaseModel):
    """Request to step the simulation"""
    num_steps: int = Field(1, ge=1, le=100, description="Number of steps to execute")


class StepResponse(BaseModel):
    """Response after stepping"""
    simulation_id: str
    current_timestep: int
    snapshots: List[Dict[str, Any]]
    pending_decision: Optional[Dict[str, Any]] = Field(None, description="Alert requiring user decision")


class UserDecisionRequest(BaseModel):
    """User's response to a decision alert"""
    decision_id: str = Field(..., description="ID of the decision")
    approved: bool = Field(..., description="Whether user approved the action")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters if user wants to modify")


class ShockRequest(BaseModel):
    """Request to apply an exogenous shock"""
    shock_type: str = Field(
        ...,
        description="Type of shock: real_estate_shock, manufacturing_shock, agriculture_shock, energy_shock, export_shock, services_shock, msme_shock, technology_shock, retail_shock, infrastructure_shock, sector_crisis, liquidity_squeeze, interest_rate_shock, asset_price_crash"
    )
    target: Optional[str] = Field(None, description="Target agent ID (for legacy sector_crisis)")
    magnitude: float = Field(-0.2, ge=-1.0, le=1.0, description="Shock magnitude for legacy shocks")
    severity: Optional[str] = Field(
        "moderate",
        description="Severity for sector shocks: mild, moderate, severe, crisis"
    )


class PolicyRuleRequest(BaseModel):
    """Request to add a CCP policy rule"""
    ccp_id: str = Field("CCP_MAIN", description="CCP agent ID")
    rule_name: str = Field(..., description="Rule name")
    condition: str = Field(..., description="Python condition expression (e.g., 'system_npa > 8.0')")
    action: str = Field(..., description="Python action expression (e.g., 'haircut_rate += 0.05')")


class UpdateAgentPolicyRequest(BaseModel):
    """Request to update user agent policies"""
    agent_id: str = Field(..., description="Agent ID to update")
    policies: Dict[str, Any] = Field(..., description="Policy key-value pairs to update")


class StateResponse(BaseModel):
    """Current state of the simulation"""
    simulation_id: str
    timestep: int
    global_metrics: Dict[str, float]
    agent_states: Dict[str, Dict[str, Any]]
    network_state: Dict[str, Any]


class HistoryResponse(BaseModel):
    """Historical snapshots"""
    simulation_id: str
    total_timesteps: int
    snapshots: List[Dict[str, Any]]


# ============================================================================
# Helper Functions
# ============================================================================

def _find_user_agent(ecosystem: FinancialEcosystem):
    """Find the user-controlled agent in the ecosystem."""
    for agent_id, agent in ecosystem.agents.items():
        if hasattr(agent, 'is_user_controlled') and agent.is_user_controlled:
            return agent
        # Fallback: check for common user IDs
        if 'USER' in agent_id.upper() or agent_id.startswith('BANK_USER'):
            return agent
    return None


def _check_for_risky_decision(sim_id: str, ecosystem: FinancialEcosystem, user_agent, snapshot) -> Optional[Dict[str, Any]]:
    """
    Check if user agent faces a high-risk situation that requires approval.
    Returns decision alert details if action needed, None otherwise.
    Includes cooldown mechanism to prevent alert spam.
    """
    from app.engine.agents import BankAgent, AgentMode
    
    if not isinstance(user_agent, BankAgent):
        return None
    
    # Don't show decisions if user agent is dead
    if not user_agent.alive:
        return None
    
    # Check cooldown - don't show same alert within 5 timesteps of last action
    cooldown_timesteps = 5
    if sim_id in DECISION_COOLDOWNS:
        last_decision_timestep = DECISION_COOLDOWNS[sim_id].get(user_agent.agent_id, -999)
        if ecosystem.timestep - last_decision_timestep < cooldown_timesteps:
            return None  # Still in cooldown period
    
    # Only alert on significant risk situations
    health = user_agent.compute_health()
    
    # CRITICAL: CRAR below regulatory minimum
    if user_agent.crar < user_agent.regulatory_min_crar:
        target_crar = user_agent.regulatory_min_crar + 2.0
        target_rwa = (user_agent.capital / target_crar) * 100
        target_liquidity = user_agent.capital * 0.3
        return {
            'title': '🚨 Critical Capital Shortage',
            'description': f'Your CRAR ({user_agent.crar:.1f}%) is below the regulatory minimum ({user_agent.regulatory_min_crar:.1f}%). You must deleverage to avoid regulatory action.',
            'risk_level': 'critical',
            'current_metrics': {
                'crar': user_agent.crar,
                'capital': user_agent.capital,
                'health': health,
                'liquidity': user_agent.liquidity
            },
            'recommended_action': {
                'type': 'deleverage',
                'description': f'Reduce RWA from ${user_agent.risk_weighted_assets/1e6:.1f}M to ${target_rwa/1e6:.1f}M | Boost liquidity to ${target_liquidity/1e6:.1f}M | Cut credit supply by 40% | Reduce risk appetite to 20%',
                'impact': 'CRAR will increase to ~{:.1f}% (safe zone)'.format(target_crar),
                'metrics_change': {
                    'crar': {'from': user_agent.crar, 'to': target_crar},
                    'rwa': {'from': user_agent.risk_weighted_assets, 'to': target_rwa},
                    'liquidity': {'from': user_agent.liquidity, 'to': target_liquidity},
                    'risk_appetite': {'from': user_agent.risk_appetite, 'to': 0.2}
                }
            },
            'alternative_action': {
                'type': 'maintain',
                'description': 'Continue current strategy (high risk of default)',
                'impact': 'Risk of regulatory penalties and potential default'
            }
        }
    
    # HIGH RISK: Health below 30% or multiple neighbor defaults
    elif health < 0.3 or user_agent.neighbor_defaults >= 2:
        target_liquidity = user_agent.capital * 0.25
        new_interbank_limit = user_agent.interbank_limit * 0.8
        new_credit_limit = user_agent.credit_supply_limit * 0.85
        new_risk_appetite = max(0.3, user_agent.risk_appetite * 0.8)
        return {
            'title': '⚠️ High Risk Detected',
            'description': f'Your health score ({health*100:.1f}%) is concerning. {user_agent.neighbor_defaults} of your counterparties have defaulted. Consider defensive measures.',
            'risk_level': 'high',
            'current_metrics': {
                'health': health,
                'neighbor_defaults': user_agent.neighbor_defaults,
                'npa_ratio': user_agent.npa_ratio,
                'liquidity_ratio': user_agent.liquidity / max(user_agent.capital, 1)
            },
            'recommended_action': {
                'type': 'defensive',
                'description': f'Reduce interbank limit from ${user_agent.interbank_limit/1e6:.1f}M to ${new_interbank_limit/1e6:.1f}M | Build liquidity from ${user_agent.liquidity/1e6:.1f}M to ${target_liquidity/1e6:.1f}M | Lower risk appetite from {user_agent.risk_appetite*100:.0f}% to {new_risk_appetite*100:.0f}%',
                'impact': 'Reduced contagion risk from neighbor failures',
                'metrics_change': {
                    'interbank_limit': {'from': user_agent.interbank_limit, 'to': new_interbank_limit},
                    'liquidity': {'from': user_agent.liquidity, 'to': target_liquidity},
                    'risk_appetite': {'from': user_agent.risk_appetite, 'to': new_risk_appetite}
                }
            },
            'alternative_action': {
                'type': 'maintain',
                'description': 'Maintain current positions',
                'impact': 'Higher returns if markets stabilize, but risk of losses'
            }
        }
    
    # SYSTEMIC STRESS: High global stress with moderate health
    elif user_agent.perceived_systemic_stress > 0.7 and health < 0.6:
        new_risk_appetite = 0.3
        new_credit_limit = user_agent.credit_supply_limit * 0.9
        return {
            'title': '📉 Market Stress Alert',
            'description': f'System stress is high ({user_agent.perceived_systemic_stress*100:.1f}%). Your health is moderate. Reduce risk exposure?',
            'risk_level': 'medium',
            'current_metrics': {
                'health': health,
                'systemic_stress': user_agent.perceived_systemic_stress,
                'risk_appetite': user_agent.risk_appetite
            },
            'recommended_action': {
                'type': 'reduce_risk',
                'description': f'Lower risk appetite from {user_agent.risk_appetite*100:.0f}% to {new_risk_appetite*100:.0f}% | Reduce credit supply from ${user_agent.credit_supply_limit/1e6:.1f}M to ${new_credit_limit/1e6:.1f}M',
                'impact': 'Better protection against market shocks',
                'metrics_change': {
                    'risk_appetite': {'from': user_agent.risk_appetite, 'to': new_risk_appetite},
                    'credit_supply_limit': {'from': user_agent.credit_supply_limit, 'to': new_credit_limit}
                }
            },
            'alternative_action': {
                'type': 'maintain',
                'description': 'Trust your current strategy',
                'impact': 'Potential for higher gains if crisis passes quickly'
            }
        }
    
    return None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/initialize", response_model=SimulationInitResponse, status_code=201)
async def initialize_simulation(request: SimulationInitRequest) -> SimulationInitResponse:
    """
    Initialize a new Agent-Based Model simulation.
    
    Creates agents from real CSV data (if available) or synthetic data,
    constructs the financial network, and prepares for time-stepping.
    """
    try:
        sim_id = str(uuid4())
        
        # Create configuration
        config = SimulationConfig(
            max_timesteps=request.max_timesteps,
            enable_shocks=request.enable_shocks,
            shock_probability=request.shock_probability,
            random_seed=request.random_seed
        )
        
        # Build ecosystem
        if request.use_real_data:
            data_dir = Path(request.data_source)
            if not data_dir.exists():
                logger.warning(f"Data directory {data_dir} not found. Using synthetic data.")
                ecosystem = SimulationFactory.create_default_scenario(config, user_entity=request.user_entity)
            else:
                ecosystem = load_ecosystem_from_data(
                    str(data_dir),
                    max_timesteps=request.max_timesteps,
                    enable_shocks=request.enable_shocks,
                    random_seed=request.random_seed
                )
                # Add user entity if provided
                if request.user_entity:
                    SimulationFactory.add_user_entity(ecosystem, request.user_entity)
        else:
            ecosystem = SimulationFactory.create_default_scenario(config, user_entity=request.user_entity)
        
        # Store in active simulations
        ACTIVE_SIMULATIONS[sim_id] = ecosystem
        
        # Get network stats
        network_stats = ecosystem.get_network_stats()
        
        # Get initial state
        initial_snapshot = ecosystem._create_snapshot([])
        
        # logger.info(f"Initialized simulation {sim_id}: {request.name}")
        
        return SimulationInitResponse(
            simulation_id=sim_id,
            name=request.name,
            config={
                "max_timesteps": request.max_timesteps,
                "enable_shocks": request.enable_shocks,
                "shock_probability": request.shock_probability,
                "random_seed": request.random_seed
            },
            network_stats=network_stats,
            initial_state=initial_snapshot.to_dict()
        )
    
    except Exception as e:
        logger.error(f"Error initializing simulation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize simulation: {str(e)}")


@router.post("/{sim_id}/step", response_model=StepResponse)
async def step_simulation(sim_id: str, request: StepRequest) -> StepResponse:
    """
    Advance the simulation by N time steps.
    
    Executes the full Agent cycle: Perceive -> Decide -> Act -> Contagion.
    Pauses if user entity faces a risky decision that needs approval.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        snapshots = []
        pending_decision = None
        
        for step_num in range(request.num_steps):
            # Check if there's a pending decision from previous step
            if sim_id in PENDING_DECISIONS:
                pending_decision = PENDING_DECISIONS[sim_id]
                # logger.info(f"Simulation {sim_id} paused - waiting for user decision")
                break
            
            snapshot = ecosystem.step()
            snapshots.append(snapshot.to_dict())
            
            # Check if user entity needs to make a risky decision
            user_agent = _find_user_agent(ecosystem)
            if user_agent:
                risk_decision = _check_for_risky_decision(sim_id, ecosystem, user_agent, snapshot)
                if risk_decision:
                    decision_id = str(uuid4())
                    PENDING_DECISIONS[sim_id] = {
                        'decision_id': decision_id,
                        'agent_id': user_agent.agent_id,
                        'timestep': ecosystem.timestep,
                        **risk_decision
                    }
                    pending_decision = PENDING_DECISIONS[sim_id]
                    # logger.info(f"User decision required for {sim_id}: {risk_decision['title']}")
                    break
        
        return StepResponse(
            simulation_id=sim_id,
            current_timestep=ecosystem.timestep,
            snapshots=snapshots,
            pending_decision=pending_decision
        )
    
    except Exception as e:
        logger.error(f"Error stepping simulation {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to step simulation: {str(e)}")


@router.post("/{sim_id}/decision")
async def respond_to_decision(sim_id: str, request: UserDecisionRequest):
    """
    User responds to a risk alert decision.
    Applies the chosen action and removes pending decision to resume simulation.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    if sim_id not in PENDING_DECISIONS:
        raise HTTPException(status_code=404, detail=f"No pending decision for simulation {sim_id}")
    
    pending = PENDING_DECISIONS[sim_id]
    
    if pending['decision_id'] != request.decision_id:
        raise HTTPException(status_code=400, detail="Invalid decision ID")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    user_agent = ecosystem.agents.get(pending['agent_id'])
    
    if not user_agent:
        raise HTTPException(status_code=404, detail="User agent not found")
    
    try:
        from app.engine.agents import BankAgent
        
        if isinstance(user_agent, BankAgent):
            if request.approved:
                # Apply recommended action
                action_type = pending['recommended_action']['type']
                
                if action_type == 'deleverage':
                    # CRITICAL: Fix CRAR by reducing RWA and building capital
                    # Target: CRAR >= regulatory_min + 2% buffer
                    target_crar = user_agent.regulatory_min_crar + 2.0
                    current_crar = user_agent.crar
                    
                    if current_crar < target_crar:
                        # Calculate needed RWA reduction to achieve target CRAR
                        # CRAR = (capital / RWA) * 100
                        # target_crar = (capital / new_RWA) * 100
                        # new_RWA = (capital / target_crar) * 100
                        target_rwa = (user_agent.capital / target_crar) * 100
                        
                        # Reduce RWA by selling assets/calling in loans
                        if target_rwa < user_agent.risk_weighted_assets:
                            user_agent.risk_weighted_assets = target_rwa
                            # logger.info(f"Reduced RWA from {user_agent.risk_weighted_assets:.2f} to {target_rwa:.2f}")
                        
                        # Recalculate CRAR
                        user_agent.crar = (user_agent.capital / user_agent.risk_weighted_assets) * 100 if user_agent.risk_weighted_assets > 0 else 100.0
                    
                    # Reduce future lending capacity
                    user_agent.credit_supply_limit *= 0.6
                    user_agent.interbank_limit *= 0.5
                    user_agent.risk_appetite = max(0.2, user_agent.risk_appetite * 0.6)
                    
                    # Boost liquidity buffer
                    liquidity_target = user_agent.capital * 0.3
                    user_agent.liquidity = max(user_agent.liquidity, liquidity_target)
                    
                    # logger.info(f"Applied DELEVERAGE action for {user_agent.agent_id}: CRAR {current_crar:.2f}% -> {user_agent.crar:.2f}%")
                
                elif action_type == 'defensive':
                    user_agent.interbank_limit *= 0.8
                    user_agent.credit_supply_limit *= 0.85
                    user_agent.risk_appetite = max(0.3, user_agent.risk_appetite * 0.8)
                    # Build liquidity
                    liquidity_target = user_agent.capital * 0.25
                    if user_agent.liquidity < liquidity_target:
                        user_agent.liquidity += (liquidity_target - user_agent.liquidity) * 0.5
                    # logger.info(f"Applied DEFENSIVE action for {user_agent.agent_id}")
                
                elif action_type == 'reduce_risk':
                    user_agent.risk_appetite = 0.3
                    user_agent.credit_supply_limit *= 0.9
                    # logger.info(f"Applied REDUCE_RISK action for {user_agent.agent_id}")
                
                result_message = f"Applied {action_type} action successfully"
            else:
                # User rejected recommendation - no action taken
                result_message = "User chose to maintain current strategy"
                # logger.info(f"User rejected recommendation for {user_agent.agent_id}")
            
            # Apply custom parameters if provided
            if request.custom_params:
                if 'risk_appetite' in request.custom_params:
                    user_agent.risk_appetite = request.custom_params['risk_appetite']
                if 'credit_supply_limit' in request.custom_params:
                    user_agent.credit_supply_limit = request.custom_params['credit_supply_limit']
                if 'interbank_limit' in request.custom_params:
                    user_agent.interbank_limit = request.custom_params['interbank_limit']
                result_message += " (with custom parameters)"
        
        # Set cooldown to prevent immediate re-alerting
        sim_cooldowns = DECISION_COOLDOWNS.get(sim_id, {})
        sim_cooldowns[user_agent.agent_id] = ecosystem.timestep
        DECISION_COOLDOWNS[sim_id] = sim_cooldowns
        
        # Remove pending decision to resume simulation
        del PENDING_DECISIONS[sim_id]
        
        return {
            'success': True,
            'message': result_message,
            'agent_id': user_agent.agent_id,
            'new_state': {
                'risk_appetite': user_agent.risk_appetite,
                'credit_supply_limit': user_agent.credit_supply_limit,
                'interbank_limit': user_agent.interbank_limit,
                'health': user_agent.compute_health()
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing decision for {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process decision: {str(e)}")


@router.post("/{sim_id}/run", response_model=StepResponse)
async def run_simulation(
    sim_id: str,
    steps: int = Query(10, ge=1, le=500, description="Number of steps to run")
) -> StepResponse:
    """
    Run the simulation for multiple steps (same as step, but with query param).
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        snapshots = ecosystem.run(steps=steps)
        
        return StepResponse(
            simulation_id=sim_id,
            current_timestep=ecosystem.timestep,
            snapshots=[s.to_dict() for s in snapshots]
        )
    
    except Exception as e:
        logger.error(f"Error running simulation {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to run simulation: {str(e)}")


@router.get("/{sim_id}/state", response_model=StateResponse)
async def get_simulation_state(sim_id: str) -> StateResponse:
    """
    Get the current state of the simulation without advancing time.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        # Create snapshot without stepping
        snapshot = ecosystem._create_snapshot([])
        
        return StateResponse(
            simulation_id=sim_id,
            timestep=ecosystem.timestep,
            global_metrics=ecosystem.global_state,
            agent_states=snapshot.agent_states,
            network_state=snapshot.network_state
        )
    
    except Exception as e:
        logger.error(f"Error getting state for {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@router.post("/{sim_id}/shock", response_model=Dict[str, Any])
async def apply_shock(sim_id: str, request: ShockRequest) -> Dict[str, Any]:
    """
    Apply an exogenous shock to the simulation.
    
    Available shock types:
    - sector_crisis: Crash a specific sector's economic health
    - liquidity_squeeze: Reduce global liquidity
    - interest_rate_shock: Spike interest rates
    - asset_price_crash: Global asset price decline
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        # Map string to enum
        shock_type_map = {
            "sector_crisis": ShockType.SECTOR_CRISIS,
            "liquidity_squeeze": ShockType.LIQUIDITY_SQUEEZE,
            "interest_rate_shock": ShockType.INTEREST_RATE_SHOCK,
            "asset_price_crash": ShockType.ASSET_PRICE_CRASH,
            # New sector-specific shocks
            "real_estate_shock": ShockType.REAL_ESTATE_SHOCK,
            "infrastructure_shock": ShockType.INFRASTRUCTURE_SHOCK,
            "manufacturing_shock": ShockType.MANUFACTURING_SHOCK,
            "agriculture_shock": ShockType.AGRICULTURE_SHOCK,
            "energy_shock": ShockType.ENERGY_SHOCK,
            "export_shock": ShockType.EXPORT_SHOCK,
            "services_shock": ShockType.SERVICES_SHOCK,
            "msme_shock": ShockType.MSME_SHOCK,
            "technology_shock": ShockType.TECHNOLOGY_SHOCK,
            "retail_shock": ShockType.RETAIL_SHOCK,
        }
        
        shock_type = shock_type_map.get(request.shock_type.lower())
        if shock_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid shock type: {request.shock_type}. Valid: {list(shock_type_map.keys())}"
            )
        
        shock_event = ecosystem.apply_shock(
            shock_type=shock_type,
            target=request.target,
            magnitude=request.magnitude,
            severity=request.severity
        )
        
        # logger.info(f"Applied shock to {sim_id}: {shock_event}")
        
        return {
            "simulation_id": sim_id,
            "shock_applied": True,
            "shock_event": shock_event
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying shock to {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to apply shock: {str(e)}")


@router.post("/{sim_id}/agent-policy", response_model=Dict[str, Any])
async def update_agent_policy(sim_id: str, request: UpdateAgentPolicyRequest) -> Dict[str, Any]:
    """
    Update policies for a user-controlled agent (bank, CCP, etc.).
    
    This allows the frontend to update agent properties like risk_appetite,
    min_capital_ratio, liquidity_buffer, etc. during the simulation.
    
    Example for Bank:
    ```
    {
        "agent_id": "user_bank_123",
        "policies": {
            "risk_appetite": 0.65,
            "min_capital_ratio": 11.5,
            "liquidity_buffer": 0.15
        }
    }
    ```
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        # Get the agent
        agent = ecosystem.get_agent(request.agent_id)
        
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
        
        # Map frontend policy names to backend attributes
        policy_mapping = {
            # Bank policies
            'riskAppetite': 'risk_appetite',
            'minCapitalRatio': 'regulatory_min_crar',
            'liquidityBuffer': None,  # Not a direct attribute, handle specially
            'maxExposurePerCounterparty': 'interbank_limit',
            # CCP policies
            'initialMargin': 'initial_margin_requirement',
            'haircut': 'haircut_rate',
            'stressTestMultiplier': None,  # Not implemented in current agent
            # Regulator policies
            'baseRepoRate': 'base_repo_rate',
            'minimumCRAR': 'min_crar_requirement',
            'crisisInterventionThreshold': None,  # Not a direct attribute
            # Sector policies
            'economicHealth': 'economic_health',
            'debtLoad': 'debt_load',
            'volatility': 'volatility'
        }
        
        updated_fields = {}
        
        # Update agent attributes
        for frontend_key, value in request.policies.items():
            backend_key = policy_mapping.get(frontend_key, frontend_key)
            
            # Skip if not mapped
            if backend_key is None:
                logger.warning(f"Policy '{frontend_key}' not mapped to agent attribute, skipping")
                continue
            
            # Special handling for percentage values that need conversion
            if frontend_key == 'minCapitalRatio':
                # Frontend sends as percentage (e.g., 11.5), backend expects decimal (e.g., 0.115)
                value = value / 100.0
            elif frontend_key == 'liquidityBuffer':
                # Frontend sends as percentage (e.g., 15), backend expects decimal (e.g., 0.15)
                # For banks, update liquidity directly as a percentage of capital
                value = value / 100.0
                if hasattr(agent, 'capital'):
                    agent.liquidity = agent.capital * value
                    updated_fields[frontend_key] = value
                    logger.info(f"Updated bank liquidity to {agent.liquidity:.2f} ({value*100:.1f}% of capital)")
                continue
            elif frontend_key == 'maxExposurePerCounterparty':
                # Frontend sends as percentage (e.g., 25), backend uses absolute value
                # Convert to actual money amount based on agent's capital
                value_percentage = value / 100.0
                if hasattr(agent, 'capital'):
                    value = agent.capital * value_percentage  # Convert to absolute amount
            
            if hasattr(agent, backend_key):
                setattr(agent, backend_key, value)
                updated_fields[frontend_key] = value
                logger.info(f"Updated {request.agent_id}.{backend_key} to {value}")
            else:
                logger.warning(f"Agent {request.agent_id} does not have attribute: {backend_key}")
        
        return {
            "simulation_id": sim_id,
            "agent_id": request.agent_id,
            "updated_policies": updated_fields,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent policy in {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update agent policy: {str(e)}")


@router.post("/{sim_id}/policy", response_model=Dict[str, Any])
async def update_ccp_policy(sim_id: str, request: PolicyRuleRequest) -> Dict[str, Any]:
    """
    Add a custom policy rule to the CCP agent.
    
    This allows users to inject game theory strategies dynamically.
    
    Example:
    ```
    {
        "ccp_id": "CCP_MAIN",
        "rule_name": "Emergency Haircut Rule",
        "condition": "system_npa > 8.0",
        "action": "self.haircut_rate += 0.05"
    }
    ```
    
    ⚠️ WARNING: This uses eval() and is unsafe in production.
    For production, implement a DSL or use a rules engine.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        # Get the CCP agent
        ccp_agent = ecosystem.get_agent(request.ccp_id)
        
        if ccp_agent is None:
            raise HTTPException(status_code=404, detail=f"CCP {request.ccp_id} not found")
        
        # Create rule (UNSAFE: For demo purposes only)
        def condition_fn(agent):
            # Simple eval (very unsafe, use a proper DSL in production)
            try:
                # Expose safe variables
                system_npa = ecosystem.global_state.get('system_npa', 0.0)
                system_liquidity = ecosystem.global_state.get('system_liquidity', 1.0)
                
                # Evaluate condition
                return eval(request.condition, {
                    'system_npa': system_npa,
                    'system_liquidity': system_liquidity,
                    'agent': agent
                })
            except Exception as e:
                logger.error(f"Error evaluating condition: {e}")
                return False
        
        def action_fn():
            # Execute action (also unsafe)
            try:
                exec(request.action, {'self': ccp_agent})
                return {"action_executed": request.action}
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                return {"error": str(e)}
        
        rule = {
            'name': request.rule_name,
            'condition': condition_fn,
            'action': action_fn
        }
        
        ccp_agent.add_policy_rule(rule)
        
        # logger.info(f"Added policy rule '{request.rule_name}' to {request.ccp_id} in simulation {sim_id}")
        
        return {
            "simulation_id": sim_id,
            "ccp_id": request.ccp_id,
            "rule_added": request.rule_name,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating CCP policy in {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update policy: {str(e)}")


@router.get("/{sim_id}/history", response_model=HistoryResponse)
async def get_simulation_history(
    sim_id: str,
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit number of snapshots")
) -> HistoryResponse:
    """
    Retrieve the complete history of the simulation.
    
    Returns all recorded snapshots (or last N if limit is specified).
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        history = ecosystem.history
        
        if limit:
            history = history[-limit:]
        
        return HistoryResponse(
            simulation_id=sim_id,
            total_timesteps=len(ecosystem.history),
            snapshots=[s.to_dict() for s in history]
        )
    
    except Exception as e:
        logger.error(f"Error getting history for {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/{sim_id}/reset", response_model=Dict[str, Any])
async def reset_simulation(sim_id: str) -> Dict[str, Any]:
    """
    Reset the simulation to time t=0 with initial conditions.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        ecosystem.reset()
        
        # logger.info(f"Reset simulation {sim_id}")
        
        return {
            "simulation_id": sim_id,
            "status": "reset",
            "timestep": ecosystem.timestep
        }
    
    except Exception as e:
        logger.error(f"Error resetting simulation {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset simulation: {str(e)}")


@router.delete("/{sim_id}", response_model=Dict[str, Any])
async def delete_simulation(sim_id: str) -> Dict[str, Any]:
    """
    Delete a simulation and free resources.
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    try:
        del ACTIVE_SIMULATIONS[sim_id]
        
        # logger.info(f"Deleted simulation {sim_id}")
        
        return {
            "simulation_id": sim_id,
            "status": "deleted"
        }
    
    except Exception as e:
        logger.error(f"Error deleting simulation {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete simulation: {str(e)}")


@router.get("/list", response_model=Dict[str, Any])
async def list_simulations() -> Dict[str, Any]:
    """
    List all active simulations.
    """
    simulations = []
    
    for sim_id, ecosystem in ACTIVE_SIMULATIONS.items():
        simulations.append({
            "simulation_id": sim_id,
            "timestep": ecosystem.timestep,
            "num_agents": len(ecosystem.agents),
            "network_stats": ecosystem.get_network_stats()
        })
    
    return {
        "total": len(simulations),
        "simulations": simulations
    }


@router.post("/{sim_id}/export", response_model=Dict[str, Any])
async def export_network(
    sim_id: str,
    format: str = Query("json", regex="^(gexf|graphml|json)$", description="Export format")
) -> Dict[str, Any]:
    """
    Export the network to a file for external visualization (Gephi, Cytoscape).
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        from pathlib import Path
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        filename = f"network_{sim_id}_{ecosystem.timestep}.{format}"
        filepath = export_dir / filename
        
        ecosystem.export_network(str(filepath), format=format)
        
        return {
            "simulation_id": sim_id,
            "exported": True,
            "filepath": str(filepath),
            "format": format
        }
    
    except Exception as e:
        logger.error(f"Error exporting network for {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export network: {str(e)}")
