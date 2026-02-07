"""
Exposure API endpoints
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.exposure import Exposure, ExposureType
from app.models.institution import Institution
from app.schemas.exposure import (
    ExposureCreate,
    ExposureUpdate,
    ExposureResponse,
    ExposureListResponse,
)

router = APIRouter()


@router.get("", response_model=ExposureListResponse)
async def list_exposures(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    source_id: Optional[UUID] = Query(None, description="Filter by source institution"),
    target_id: Optional[UUID] = Query(None, description="Filter by target institution"),
    exposure_type: Optional[ExposureType] = Query(None, description="Filter by type"),
    min_exposure: Optional[float] = Query(None, description="Minimum exposure amount"),
    include_expired: bool = Query(False, description="Include expired exposures"),
    db: AsyncSession = Depends(get_db),
) -> ExposureListResponse:
    """
    List all exposures with optional filtering and pagination.
    """
    from datetime import datetime
    
    # Build query
    query = select(Exposure)
    count_query = select(func.count(Exposure.id))
    
    # Apply filters
    if source_id:
        query = query.where(Exposure.source_institution_id == source_id)
        count_query = count_query.where(Exposure.source_institution_id == source_id)
    
    if target_id:
        query = query.where(Exposure.target_institution_id == target_id)
        count_query = count_query.where(Exposure.target_institution_id == target_id)
    
    if exposure_type:
        query = query.where(Exposure.exposure_type == exposure_type)
        count_query = count_query.where(Exposure.exposure_type == exposure_type)
    
    if min_exposure:
        query = query.where(Exposure.gross_exposure >= min_exposure)
        count_query = count_query.where(Exposure.gross_exposure >= min_exposure)
    
    if not include_expired:
        now = datetime.utcnow()
        valid_filter = or_(
            Exposure.valid_to.is_(None),
            Exposure.valid_to > now
        )
        query = query.where(valid_filter)
        count_query = count_query.where(valid_filter)
    
    # Get total count
    total = await db.scalar(count_query)
    
    # Apply pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit).order_by(Exposure.gross_exposure.desc())
    
    # Execute query
    result = await db.execute(query)
    exposures = result.scalars().all()
    
    # Calculate pages
    pages = (total + limit - 1) // limit if limit > 0 else 0
    
    return ExposureListResponse(
        items=[ExposureResponse.model_validate(exp) for exp in exposures],
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )


@router.get("/{exposure_id}", response_model=ExposureResponse)
async def get_exposure(
    exposure_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExposureResponse:
    """
    Get a specific exposure by ID.
    """
    query = select(Exposure).where(Exposure.id == exposure_id)
    result = await db.execute(query)
    exposure = result.scalar_one_or_none()
    
    if not exposure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exposure {exposure_id} not found"
        )
    
    response = ExposureResponse.model_validate(exposure)
    response.effective_exposure = exposure.effective_exposure
    response.loss_given_default = exposure.loss_given_default
    
    return response


@router.post("", response_model=ExposureResponse, status_code=status.HTTP_201_CREATED)
async def create_exposure(
    exposure_data: ExposureCreate,
    db: AsyncSession = Depends(get_db),
) -> ExposureResponse:
    """
    Create a new exposure relationship.
    """
    # Verify source institution exists
    source_query = select(Institution.id).where(
        Institution.id == exposure_data.source_institution_id
    )
    source_result = await db.execute(source_query)
    if not source_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source institution {exposure_data.source_institution_id} not found"
        )
    
    # Verify target institution exists
    target_query = select(Institution.id).where(
        Institution.id == exposure_data.target_institution_id
    )
    target_result = await db.execute(target_query)
    if not target_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target institution {exposure_data.target_institution_id} not found"
        )
    
    # Create exposure
    exposure = Exposure(**exposure_data.model_dump(by_alias=True))
    db.add(exposure)
    await db.flush()
    await db.refresh(exposure)
    
    return ExposureResponse.model_validate(exposure)


@router.put("/{exposure_id}", response_model=ExposureResponse)
async def update_exposure(
    exposure_id: UUID,
    exposure_data: ExposureUpdate,
    db: AsyncSession = Depends(get_db),
) -> ExposureResponse:
    """
    Update an existing exposure.
    """
    query = select(Exposure).where(Exposure.id == exposure_id)
    result = await db.execute(query)
    exposure = result.scalar_one_or_none()
    
    if not exposure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exposure {exposure_id} not found"
        )
    
    # Update fields
    update_data = exposure_data.model_dump(exclude_unset=True, by_alias=True)
    for field, value in update_data.items():
        setattr(exposure, field, value)
    
    await db.flush()
    await db.refresh(exposure)
    
    return ExposureResponse.model_validate(exposure)


@router.delete("/{exposure_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_exposure(
    exposure_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete an exposure.
    """
    query = select(Exposure).where(Exposure.id == exposure_id)
    result = await db.execute(query)
    exposure = result.scalar_one_or_none()
    
    if not exposure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exposure {exposure_id} not found"
        )
    
    await db.delete(exposure)


@router.get("/matrix/summary")
async def get_exposure_matrix(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get exposure matrix summary for network visualization.
    """
    from datetime import datetime
    
    # Get all active exposures
    now = datetime.utcnow()
    query = (
        select(Exposure)
        .where(or_(Exposure.valid_to.is_(None), Exposure.valid_to > now))
    )
    result = await db.execute(query)
    exposures = result.scalars().all()
    
    # Build matrix data
    matrix = []
    for exp in exposures:
        matrix.append({
            "source_id": str(exp.source_institution_id),
            "target_id": str(exp.target_institution_id),
            "exposure_type": exp.exposure_type.value,
            "gross_exposure": float(exp.gross_exposure),
            "contagion_probability": float(exp.contagion_probability),
        })
    
    # Calculate summary stats
    total_exposure = sum(e["gross_exposure"] for e in matrix)
    
    return {
        "edges": matrix,
        "total_edges": len(matrix),
        "total_exposure": total_exposure,
    }
