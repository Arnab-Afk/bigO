"""
Simulation API endpoints
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.simulation import Simulation, SimulationStatus, SimulationResult
from app.models.scenario import Scenario
from app.schemas.simulation import (
    SimulationCreate,
    SimulationResponse,
    SimulationResultResponse,
    SimulationListResponse,
)
from app.tasks.simulation_tasks import run_simulation_task
from app.core.logging import logger

router = APIRouter()


@router.get("", response_model=SimulationListResponse)
async def list_simulations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[SimulationStatus] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> SimulationListResponse:
    """
    List all simulations with optional filtering.
    """
    query = select(Simulation)
    count_query = select(func.count(Simulation.id))
    
    if status_filter:
        query = query.where(Simulation.status == status_filter)
        count_query = count_query.where(Simulation.status == status_filter)
    
    total = await db.scalar(count_query)
    
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit).order_by(Simulation.created_at.desc())
    
    result = await db.execute(query)
    simulations = result.scalars().all()
    
    pages = (total + limit - 1) // limit if limit > 0 else 0
    
    items = []
    for sim in simulations:
        response = SimulationResponse.model_validate(sim)
        response.progress = sim.progress
        response.duration_seconds = sim.duration_seconds
        items.append(response)
    
    return SimulationListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SimulationResponse:
    """
    Get a specific simulation by ID.
    """
    query = select(Simulation).where(Simulation.id == simulation_id)
    result = await db.execute(query)
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    response = SimulationResponse.model_validate(simulation)
    response.progress = simulation.progress
    response.duration_seconds = simulation.duration_seconds
    
    return response


@router.post("", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    simulation_data: SimulationCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> SimulationResponse:
    """
    Create and queue a new simulation.
    """
    # Verify scenario exists if provided
    if simulation_data.scenario_id:
        scenario_query = select(Scenario.id).where(
            Scenario.id == simulation_data.scenario_id
        )
        scenario_result = await db.execute(scenario_query)
        if not scenario_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario {simulation_data.scenario_id} not found"
            )
    
    # Create simulation
    simulation = Simulation(
        **simulation_data.model_dump(),
        status=SimulationStatus.PENDING
    )
    db.add(simulation)
    await db.flush()
    await db.refresh(simulation)
    await db.commit()
    
    # Dispatch to Celery for background execution
    try:
        task = run_simulation_task.delay(str(simulation.id))
        # logger.info(
        #     "Simulation task queued",
        #     simulation_id=str(simulation.id),
        #     task_id=task.id
        # )
    except Exception as e:
        logger.error(
            "Failed to queue simulation",
            simulation_id=str(simulation.id),
            error=str(e)
        )
        # Don't fail the request, simulation can be started manually
    
    response = SimulationResponse.model_validate(simulation)
    response.progress = 0.0
    
    return response


@router.post("/{simulation_id}/start", response_model=SimulationResponse)
async def start_simulation(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SimulationResponse:
    """
    Start a pending simulation.
    """
    query = select(Simulation).where(Simulation.id == simulation_id)
    result = await db.execute(query)
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    if simulation.status != SimulationStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation is not in PENDING status (current: {simulation.status.value})"
        )
    
    # Update status and dispatch task
    simulation.status = SimulationStatus.QUEUED
    simulation.started_at = datetime.utcnow()
    await db.commit()
    
    # Dispatch to Celery
    try:
        task = run_simulation_task.delay(str(simulation.id))
        # logger.info(
        #     "Simulation task started",
        #     simulation_id=str(simulation.id),
        #     task_id=task.id
        # )
    except Exception as e:
        logger.error(
            "Failed to start simulation",
            simulation_id=str(simulation.id),
            error=str(e)
        )
        simulation.status = SimulationStatus.FAILED
        simulation.error_message = f"Failed to queue task: {str(e)}"
        await db.commit()
    
    await db.refresh(simulation)
    
    response = SimulationResponse.model_validate(simulation)
    response.progress = simulation.progress
    
    return response


@router.post("/{simulation_id}/cancel", response_model=SimulationResponse)
async def cancel_simulation(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SimulationResponse:
    """
    Cancel a running or queued simulation.
    """
    query = select(Simulation).where(Simulation.id == simulation_id)
    result = await db.execute(query)
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    if simulation.status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation cannot be cancelled (status: {simulation.status.value})"
        )
    
    simulation.status = SimulationStatus.CANCELLED
    simulation.completed_at = datetime.utcnow()
    
    await db.flush()
    await db.refresh(simulation)
    
    response = SimulationResponse.model_validate(simulation)
    response.progress = simulation.progress
    response.duration_seconds = simulation.duration_seconds
    
    return response


@router.get("/{simulation_id}/results", response_model=List[SimulationResultResponse])
async def get_simulation_results(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> List[SimulationResultResponse]:
    """
    Get results of a completed simulation.
    """
    # Verify simulation exists
    sim_query = select(Simulation).where(Simulation.id == simulation_id)
    sim_result = await db.execute(sim_query)
    simulation = sim_result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    if simulation.status != SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation not completed (status: {simulation.status.value})"
        )
    
    # Get results
    results_query = select(SimulationResult).where(
        SimulationResult.simulation_id == simulation_id
    )
    results = await db.execute(results_query)
    
    return [
        SimulationResultResponse.model_validate(r)
        for r in results.scalars().all()
    ]


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_simulation(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a simulation and its results.
    """
    query = select(Simulation).where(Simulation.id == simulation_id)
    result = await db.execute(query)
    simulation = result.scalar_one_or_none()
    
    if not simulation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found"
        )
    
    if simulation.status == SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running simulation"
        )
    
    await db.delete(simulation)
