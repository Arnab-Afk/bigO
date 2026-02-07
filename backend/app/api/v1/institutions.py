"""
Institution API endpoints
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.institution import Institution, InstitutionType, SystemicTier
from app.models.institution_state import InstitutionState
from app.schemas.institution import (
    InstitutionCreate,
    InstitutionUpdate,
    InstitutionResponse,
    InstitutionListResponse,
    InstitutionWithState,
)
from app.schemas.common import PaginationParams

router = APIRouter()


@router.get("", response_model=InstitutionListResponse)
async def list_institutions(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    type: Optional[InstitutionType] = Query(None, description="Filter by type"),
    tier: Optional[SystemicTier] = Query(None, description="Filter by tier"),
    jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search by name or external_id"),
    db: AsyncSession = Depends(get_db),
) -> InstitutionListResponse:
    """
    List all institutions with optional filtering and pagination.
    """
    # Build query
    query = select(Institution)
    count_query = select(func.count(Institution.id))
    
    # Apply filters
    if type:
        query = query.where(Institution.type == type)
        count_query = count_query.where(Institution.type == type)
    
    if tier:
        query = query.where(Institution.tier == tier)
        count_query = count_query.where(Institution.tier == tier)
    
    if jurisdiction:
        query = query.where(Institution.jurisdiction == jurisdiction)
        count_query = count_query.where(Institution.jurisdiction == jurisdiction)
    
    if is_active is not None:
        query = query.where(Institution.is_active == is_active)
        count_query = count_query.where(Institution.is_active == is_active)
    
    if search:
        search_filter = or_(
            Institution.name.ilike(f"%{search}%"),
            Institution.external_id.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)
    
    # Get total count
    total = await db.scalar(count_query)
    
    # Apply pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit).order_by(Institution.name)
    
    # Execute query
    result = await db.execute(query)
    institutions = result.scalars().all()
    
    # Calculate pages
    pages = (total + limit - 1) // limit if limit > 0 else 0
    
    return InstitutionListResponse(
        items=[InstitutionResponse.model_validate(inst) for inst in institutions],
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )


@router.get("/{institution_id}", response_model=InstitutionWithState)
async def get_institution(
    institution_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> InstitutionWithState:
    """
    Get a specific institution by ID with its current state.
    """
    # Get institution
    query = select(Institution).where(Institution.id == institution_id)
    result = await db.execute(query)
    institution = result.scalar_one_or_none()
    
    if not institution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {institution_id} not found"
        )
    
    # Get latest state
    state_query = (
        select(InstitutionState)
        .where(InstitutionState.institution_id == institution_id)
        .order_by(InstitutionState.timestamp.desc())
        .limit(1)
    )
    state_result = await db.execute(state_query)
    current_state = state_result.scalar_one_or_none()
    
    # Build response
    response = InstitutionWithState.model_validate(institution)
    
    if current_state:
        response.current_state = {
            "timestamp": current_state.timestamp,
            "capital_ratio": current_state.capital_ratio,
            "liquidity_buffer": current_state.liquidity_buffer,
            "total_credit_exposure": current_state.total_credit_exposure,
            "default_probability": current_state.default_probability,
            "stress_level": current_state.stress_level,
            "risk_score": current_state.risk_score,
        }
    
    return response


@router.post("", response_model=InstitutionResponse, status_code=status.HTTP_201_CREATED)
async def create_institution(
    institution_data: InstitutionCreate,
    db: AsyncSession = Depends(get_db),
) -> InstitutionResponse:
    """
    Create a new institution.
    """
    # Check for duplicate external_id
    existing_query = select(Institution).where(
        Institution.external_id == institution_data.external_id
    )
    existing = await db.execute(existing_query)
    
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Institution with external_id '{institution_data.external_id}' already exists"
        )
    
    # Create institution
    institution = Institution(**institution_data.model_dump(by_alias=True))
    db.add(institution)
    await db.flush()
    await db.refresh(institution)
    
    return InstitutionResponse.model_validate(institution)


@router.put("/{institution_id}", response_model=InstitutionResponse)
async def update_institution(
    institution_id: UUID,
    institution_data: InstitutionUpdate,
    db: AsyncSession = Depends(get_db),
) -> InstitutionResponse:
    """
    Update an existing institution.
    """
    # Get institution
    query = select(Institution).where(Institution.id == institution_id)
    result = await db.execute(query)
    institution = result.scalar_one_or_none()
    
    if not institution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {institution_id} not found"
        )
    
    # Update fields
    update_data = institution_data.model_dump(exclude_unset=True, by_alias=True)
    for field, value in update_data.items():
        setattr(institution, field, value)
    
    await db.flush()
    await db.refresh(institution)
    
    return InstitutionResponse.model_validate(institution)


@router.delete("/{institution_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_institution(
    institution_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete an institution.
    """
    query = select(Institution).where(Institution.id == institution_id)
    result = await db.execute(query)
    institution = result.scalar_one_or_none()
    
    if not institution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {institution_id} not found"
        )
    
    await db.delete(institution)


@router.get("/{institution_id}/states", response_model=List[dict])
async def get_institution_states(
    institution_id: UUID,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> List[dict]:
    """
    Get historical states for an institution.
    """
    # Verify institution exists
    inst_query = select(Institution.id).where(Institution.id == institution_id)
    inst_result = await db.execute(inst_query)
    
    if not inst_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Institution {institution_id} not found"
        )
    
    # Get states
    query = (
        select(InstitutionState)
        .where(InstitutionState.institution_id == institution_id)
        .order_by(InstitutionState.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    states = result.scalars().all()
    
    return [state.to_dict() for state in states]
