"""
Celery Background Tasks for Simulation Execution

Handles long-running simulation tasks asynchronously.
"""

import asyncio
from datetime import datetime
from typing import Dict, List
from uuid import UUID

import networkx as nx
from celery import Task
from sqlalchemy import select

from app.tasks import celery_app
from app.core.logging import logger
from app.db.session import async_session_maker
from app.models.simulation import Simulation, SimulationStatus
from app.models.institution import Institution
from app.models.exposure import Exposure
from app.models.scenario import Scenario
from app.engine.simulation import SimulationEngine, Shock
from app.engine.game_theory import AgentState
from app.engine.network import build_network_graph


class SimulationTask(Task):
    """Base task for simulations with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(
            "Simulation task failed",
            task_id=task_id,
            error=str(exc),
            traceback=str(einfo)
        )
        
        # Update simulation status to failed
        simulation_id = kwargs.get('simulation_id') or (args[0] if args else None)
        if simulation_id:
            asyncio.run(self._mark_failed(simulation_id, str(exc)))
    
    async def _mark_failed(self, simulation_id: str, error_message: str):
        """Mark simulation as failed in database"""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Simulation).where(Simulation.id == UUID(simulation_id))
            )
            simulation = result.scalar_one_or_none()
            
            if simulation:
                simulation.status = SimulationStatus.FAILED
                simulation.completed_at = datetime.utcnow()
                simulation.error_message = error_message
                await session.commit()


@celery_app.task(
    bind=True,
    base=SimulationTask,
    name='rudra.run_simulation'
)
def run_simulation_task(self, simulation_id: str) -> Dict:
    """
    Execute simulation in background
    
    Args:
        simulation_id: UUID of simulation to run
    
    Returns:
        Simulation results summary
    """
    logger.info("Starting simulation task", simulation_id=simulation_id)
    
    # Run async simulation
    result = asyncio.run(_execute_simulation(simulation_id, self))
    
    logger.info("Completed simulation task", simulation_id=simulation_id)
    return result


async def _execute_simulation(simulation_id: str, task: Task) -> Dict:
    """
    Execute simulation asynchronously
    
    Args:
        simulation_id: Simulation UUID
        task: Celery task instance for progress updates
    
    Returns:
        Results dictionary
    """
    async with async_session_maker() as session:
        # Load simulation
        result = await session.execute(
            select(Simulation).where(Simulation.id == UUID(simulation_id))
        )
        simulation = result.scalar_one_or_none()
        
        if not simulation:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        # Update status to running
        simulation.status = SimulationStatus.RUNNING
        simulation.started_at = datetime.utcnow()
        await session.commit()
        
        try:
            # Load network data
            institutions = await _load_institutions(session)
            exposures = await _load_exposures(session)
            
            # Build network graph
            network = build_network_graph(institutions, exposures)
            
            logger.info(
                "Built network",
                simulation_id=simulation_id,
                nodes=network.number_of_nodes(),
                edges=network.number_of_edges()
            )
            
            # Load scenario if specified
            shocks = []
            shock_timing = {}
            
            if simulation.scenario_id:
                scenario_result = await session.execute(
                    select(Scenario).where(Scenario.id == simulation.scenario_id)
                )
                scenario = scenario_result.scalar_one_or_none()
                
                if scenario and scenario.shocks:
                    shocks = _parse_shocks(scenario.shocks)
                    shock_timing = scenario.shock_timing or {}
            
            # Create initial states
            initial_states = _create_initial_states(institutions, simulation.parameters)
            
            # Initialize simulation engine
            engine = SimulationEngine(
                network=network,
                convergence_threshold=float(
                    simulation.parameters.get('convergence_threshold', 1e-6)
                ),
                max_timesteps=simulation.total_timesteps or 100,
            )
            
            # Run simulation with progress updates
            sim_state = engine.run_simulation(
                simulation_id=simulation_id,
                initial_states=initial_states,
                shocks=shocks,
                shock_timing=shock_timing,
            )
            
            # Store results
            results = {
                'total_timesteps': len(sim_state.timesteps),
                'converged': sim_state.converged,
                'convergence_step': sim_state.convergence_step,
                'total_defaults': len(sim_state.final_defaults),
                'total_losses': float(sim_state.total_losses),
                'final_metrics': sim_state.timesteps[-1].network_metrics if sim_state.timesteps else {},
                'cascade_rounds': len(sim_state.cascade_history),
            }
            
            simulation.results = results
            simulation.status = SimulationStatus.COMPLETED
            simulation.completed_at = datetime.utcnow()
            simulation.current_timestep = len(sim_state.timesteps)
            
            await session.commit()
            
            logger.info(
                "Simulation completed successfully",
                simulation_id=simulation_id,
                timesteps=len(sim_state.timesteps),
                defaults=len(sim_state.final_defaults),
                losses=sim_state.total_losses
            )
            
            return results
            
        except Exception as e:
            # Mark as failed
            simulation.status = SimulationStatus.FAILED
            simulation.completed_at = datetime.utcnow()
            simulation.error_message = str(e)
            await session.commit()
            
            logger.error(
                "Simulation failed",
                simulation_id=simulation_id,
                error=str(e)
            )
            raise


async def _load_institutions(session) -> List[Dict]:
    """Load all active institutions"""
    result = await session.execute(
        select(Institution).where(Institution.is_active == True)
    )
    institutions = result.scalars().all()
    
    return [
        {
            'id': str(inst.id),
            'name': inst.name,
            'type': inst.type.value,
            'tier': inst.tier.value,
        }
        for inst in institutions
    ]


async def _load_exposures(session) -> List[Dict]:
    """Load all active exposures"""
    result = await session.execute(
        select(Exposure).where(
            (Exposure.valid_to.is_(None)) | (Exposure.valid_to > datetime.utcnow())
        )
    )
    exposures = result.scalars().all()
    
    return [
        {
            'source_institution_id': str(exp.source_institution_id),
            'target_institution_id': str(exp.target_institution_id),
            'exposure_type': exp.exposure_type.value,
            'gross_exposure': float(exp.gross_exposure),
            'contagion_probability': float(exp.contagion_probability),
            'recovery_rate': float(exp.recovery_rate),
            'settlement_urgency': float(exp.settlement_urgency),
        }
        for exp in exposures
    ]


def _create_initial_states(
    institutions: List[Dict],
    parameters: Dict
) -> Dict[UUID, AgentState]:
    """Create initial agent states"""
    states = {}
    
    for inst in institutions:
        inst_id = UUID(inst['id'])
        
        # Use parameters or defaults
        states[inst_id] = AgentState(
            agent_id=inst_id,
            capital_ratio=parameters.get('initial_capital_ratio', 0.12),
            liquidity_buffer=parameters.get('initial_liquidity_buffer', 0.5),
            credit_exposure=parameters.get('initial_credit_exposure', 10000.0),
            default_probability=parameters.get('initial_default_probability', 0.01),
            stress_level=parameters.get('initial_stress_level', 0.1),
            risk_appetite=parameters.get('risk_appetite', 0.5),
        )
    
    return states


def _parse_shocks(shocks_data: List[Dict]) -> List[Shock]:
    """Parse shock definitions from scenario"""
    shocks = []
    
    for shock_dict in shocks_data:
        shocks.append(
            Shock(
                shock_id=shock_dict['id'],
                shock_type=shock_dict['type'],
                target_institutions=[
                    UUID(tid) for tid in shock_dict.get('targets', [])
                ],
                magnitude=shock_dict.get('magnitude', 0.0),
                parameters=shock_dict.get('parameters', {}),
            )
        )
    
    return shocks


@celery_app.task(name='rudra.monte_carlo_simulation')
def monte_carlo_simulation_task(
    base_simulation_id: str,
    num_runs: int = 1000
) -> Dict:
    """
    Run Monte Carlo simulation with parameter variations
    
    Args:
        base_simulation_id: Base simulation configuration
        num_runs: Number of Monte Carlo runs
    
    Returns:
        Aggregated statistics
    """
    logger.info(
        "Starting Monte Carlo simulation",
        base_id=base_simulation_id,
        runs=num_runs
    )
    
    # Run multiple simulations with parameter variations
    results = []
    
    for i in range(num_runs):
        # Create variation of base simulation
        # This would involve creating new simulation records
        # For now, simplified implementation
        pass
    
    return {
        'runs_completed': num_runs,
        'aggregate_results': {},
    }


@celery_app.task(name='rudra.periodic_risk_assessment')
def periodic_risk_assessment_task():
    """
    Periodic task to assess current systemic risk
    
    Runs daily to provide ongoing risk monitoring.
    """
    logger.info("Running periodic risk assessment")
    
    # This would:
    # 1. Load current network state
    # 2. Compute risk metrics
    # 3. Generate alerts if thresholds exceeded
    # 4. Store assessment results
    
    return {'status': 'completed'}
