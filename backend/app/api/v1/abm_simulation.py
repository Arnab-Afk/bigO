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


class ShockRequest(BaseModel):
    """Request to apply an exogenous shock"""
    shock_type: str = Field(..., description="Type of shock: sector_crisis, liquidity_squeeze, interest_rate_shock, asset_price_crash")
    target: Optional[str] = Field(None, description="Target agent ID (for sector_crisis)")
    magnitude: float = Field(-0.2, ge=-1.0, le=1.0, description="Shock magnitude")


class PolicyRuleRequest(BaseModel):
    """Request to add a CCP policy rule"""
    ccp_id: str = Field("CCP_MAIN", description="CCP agent ID")
    rule_name: str = Field(..., description="Rule name")
    condition: str = Field(..., description="Python condition expression (e.g., 'system_npa > 8.0')")
    action: str = Field(..., description="Python action expression (e.g., 'haircut_rate += 0.05')")


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
                ecosystem = SimulationFactory.create_default_scenario(config)
            else:
                ecosystem = load_ecosystem_from_data(
                    str(data_dir),
                    max_timesteps=request.max_timesteps,
                    enable_shocks=request.enable_shocks,
                    random_seed=request.random_seed
                )
        else:
            ecosystem = SimulationFactory.create_default_scenario(config)
        
        # Store in active simulations
        ACTIVE_SIMULATIONS[sim_id] = ecosystem
        
        # Get network stats
        network_stats = ecosystem.get_network_stats()
        
        # Get initial state
        initial_snapshot = ecosystem._create_snapshot([])
        
        logger.info(f"Initialized simulation {sim_id}: {request.name}")
        
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
    
    Executes the full Agent cycle: Perceive -> Decide -> Act -> Contagion
    """
    if sim_id not in ACTIVE_SIMULATIONS:
        raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
    
    ecosystem = ACTIVE_SIMULATIONS[sim_id]
    
    try:
        snapshots = []
        for _ in range(request.num_steps):
            snapshot = ecosystem.step()
            snapshots.append(snapshot.to_dict())
        
        return StepResponse(
            simulation_id=sim_id,
            current_timestep=ecosystem.timestep,
            snapshots=snapshots
        )
    
    except Exception as e:
        logger.error(f"Error stepping simulation {sim_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to step simulation: {str(e)}")


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
            "asset_price_crash": ShockType.ASSET_PRICE_CRASH
        }
        
        shock_type = shock_type_map.get(request.shock_type.lower())
        if shock_type is None:
            raise HTTPException(status_code=400, detail=f"Invalid shock type: {request.shock_type}")
        
        shock_event = ecosystem.apply_shock(
            shock_type=shock_type,
            target=request.target,
            magnitude=request.magnitude
        )
        
        logger.info(f"Applied shock to {sim_id}: {shock_event}")
        
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
        
        logger.info(f"Added policy rule '{request.rule_name}' to {request.ccp_id} in simulation {sim_id}")
        
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
        
        logger.info(f"Reset simulation {sim_id}")
        
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
        
        logger.info(f"Deleted simulation {sim_id}")
        
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
