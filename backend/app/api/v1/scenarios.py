"""
Scenario API endpoints
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.scenario import Scenario, Shock
from app.schemas.scenario import (
    ScenarioCreate,
    ScenarioResponse,
    ScenarioListResponse,
    ShockCreate,
    ShockResponse,
)

router = APIRouter()


@router.get("", response_model=ScenarioListResponse)
async def list_scenarios(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_template: Optional[bool] = Query(None, description="Filter by template status"),
    db: AsyncSession = Depends(get_db),
) -> ScenarioListResponse:
    """
    List all scenarios with optional filtering.
    """
    query = select(Scenario).options(selectinload(Scenario.shocks))
    count_query = select(func.count(Scenario.id))
    
    if category:
        query = query.where(Scenario.category == category)
        count_query = count_query.where(Scenario.category == category)
    
    if is_template is not None:
        query = query.where(Scenario.is_template == is_template)
        count_query = count_query.where(Scenario.is_template == is_template)
    
    total = await db.scalar(count_query)
    
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit).order_by(Scenario.name)
    
    result = await db.execute(query)
    scenarios = result.scalars().unique().all()
    
    pages = (total + limit - 1) // limit if limit > 0 else 0
    
    return ScenarioListResponse(
        items=[ScenarioResponse.model_validate(s) for s in scenarios],
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )


@router.get("/templates", response_model=List[ScenarioResponse])
async def list_template_scenarios(
    db: AsyncSession = Depends(get_db),
) -> List[ScenarioResponse]:
    """
    List all template scenarios.
    """
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.is_template == True)
        .order_by(Scenario.name)
    )
    result = await db.execute(query)
    scenarios = result.scalars().unique().all()
    
    return [ScenarioResponse.model_validate(s) for s in scenarios]


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(
    scenario_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    """
    Get a specific scenario by ID.
    """
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == scenario_id)
    )
    result = await db.execute(query)
    scenario = result.scalar_one_or_none()
    
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario {scenario_id} not found"
        )
    
    return ScenarioResponse.model_validate(scenario)


@router.post("", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
async def create_scenario(
    scenario_data: ScenarioCreate,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    """
    Create a new scenario with shocks.
    """
    # Extract shocks data
    shocks_data = scenario_data.shocks
    scenario_dict = scenario_data.model_dump(exclude={"shocks"})
    
    # Create scenario
    scenario = Scenario(**scenario_dict)
    db.add(scenario)
    await db.flush()
    
    # Create shocks
    for shock_data in shocks_data:
        shock = Shock(
            **shock_data.model_dump(),
            scenario_id=scenario.id
        )
        db.add(shock)
    
    await db.flush()
    
    # Reload with shocks
    await db.refresh(scenario)
    
    # Fetch with shocks
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == scenario.id)
    )
    result = await db.execute(query)
    scenario = result.scalar_one()
    
    return ScenarioResponse.model_validate(scenario)


@router.put("/{scenario_id}", response_model=ScenarioResponse)
async def update_scenario(
    scenario_id: UUID,
    scenario_data: ScenarioCreate,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    """
    Update an existing scenario.
    """
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == scenario_id)
    )
    result = await db.execute(query)
    scenario = result.scalar_one_or_none()
    
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario {scenario_id} not found"
        )
    
    # Update scenario fields
    update_data = scenario_data.model_dump(exclude={"shocks"})
    for field, value in update_data.items():
        setattr(scenario, field, value)
    
    # Delete existing shocks
    for shock in scenario.shocks:
        await db.delete(shock)
    
    # Create new shocks
    for shock_data in scenario_data.shocks:
        shock = Shock(
            **shock_data.model_dump(),
            scenario_id=scenario.id
        )
        db.add(shock)
    
    await db.flush()
    await db.refresh(scenario)
    
    # Reload with updated shocks
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == scenario.id)
    )
    result = await db.execute(query)
    scenario = result.scalar_one()
    
    return ScenarioResponse.model_validate(scenario)


@router.delete("/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scenario(
    scenario_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a scenario.
    """
    query = select(Scenario).where(Scenario.id == scenario_id)
    result = await db.execute(query)
    scenario = result.scalar_one_or_none()
    
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario {scenario_id} not found"
        )
    
    await db.delete(scenario)


@router.post("/{scenario_id}/duplicate", response_model=ScenarioResponse)
async def duplicate_scenario(
    scenario_id: UUID,
    new_name: Optional[str] = Query(None, description="Name for the duplicate"),
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    """
    Duplicate an existing scenario.
    """
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == scenario_id)
    )
    result = await db.execute(query)
    original = result.scalar_one_or_none()
    
    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario {scenario_id} not found"
        )
    
    # Create duplicate
    duplicate = Scenario(
        name=new_name or f"{original.name} (Copy)",
        description=original.description,
        category=original.category,
        is_template=False,  # Duplicates are not templates by default
        num_timesteps=original.num_timesteps,
        convergence_threshold=original.convergence_threshold,
        base_volatility=original.base_volatility,
        liquidity_premium=original.liquidity_premium,
        risk_aversion_min=original.risk_aversion_min,
        risk_aversion_max=original.risk_aversion_max,
        information_delay=original.information_delay,
        shock_timing=original.shock_timing.copy(),
        parameters=original.parameters.copy(),
    )
    db.add(duplicate)
    await db.flush()
    
    # Duplicate shocks
    for original_shock in original.shocks:
        shock = Shock(
            name=original_shock.name,
            description=original_shock.description,
            shock_type=original_shock.shock_type,
            target_type=original_shock.target_type,
            target_id=original_shock.target_id,
            magnitude=original_shock.magnitude,
            duration=original_shock.duration,
            trigger_timestep=original_shock.trigger_timestep,
            parameters=original_shock.parameters.copy(),
            scenario_id=duplicate.id,
        )
        db.add(shock)
    
    await db.flush()
    await db.refresh(duplicate)
    
    # Reload with shocks
    query = (
        select(Scenario)
        .options(selectinload(Scenario.shocks))
        .where(Scenario.id == duplicate.id)
    )
    result = await db.execute(query)
    duplicate = result.scalar_one()
    
    return ScenarioResponse.model_validate(duplicate)
