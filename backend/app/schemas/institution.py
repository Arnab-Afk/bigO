"""
Institution Pydantic schemas for request/response validation
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.models.institution import InstitutionType, SystemicTier
from app.schemas.common import BaseSchema, TimestampMixin, IDMixin


class InstitutionBase(BaseSchema):
    """Base institution schema with common fields"""
    
    external_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="External identifier (e.g., LEI, BIC)"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Institution name"
    )
    short_name: Optional[str] = Field(
        None,
        max_length=50,
        description="Short name or ticker"
    )
    type: InstitutionType = Field(
        default=InstitutionType.BANK,
        description="Type of financial institution"
    )
    tier: SystemicTier = Field(
        default=SystemicTier.TIER_3,
        description="Systemic importance tier"
    )
    jurisdiction: Optional[str] = Field(
        None,
        max_length=10,
        description="ISO 3166-1 alpha-2 country code"
    )
    region: Optional[str] = Field(
        None,
        max_length=50,
        description="Geographic region"
    )
    description: Optional[str] = Field(
        None,
        description="Description of the institution"
    )
    metadata_: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        alias="metadata",
        description="Additional metadata"
    )


class InstitutionCreate(InstitutionBase):
    """Schema for creating a new institution"""
    
    is_active: bool = Field(
        default=True,
        description="Whether institution is active"
    )


class InstitutionUpdate(BaseSchema):
    """Schema for updating an institution (all fields optional)"""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    short_name: Optional[str] = Field(None, max_length=50)
    type: Optional[InstitutionType] = None
    tier: Optional[SystemicTier] = None
    jurisdiction: Optional[str] = Field(None, max_length=10)
    region: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    metadata_: Optional[Dict[str, Any]] = Field(None, alias="metadata")


class InstitutionResponse(InstitutionBase, IDMixin, TimestampMixin):
    """Schema for institution response"""
    
    is_active: bool
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class InstitutionStateSnapshot(BaseSchema):
    """Embedded state snapshot for institution response"""
    
    timestamp: datetime
    capital_ratio: Decimal
    liquidity_buffer: Decimal
    total_credit_exposure: Decimal
    default_probability: Decimal
    stress_level: Decimal
    risk_score: Optional[Decimal] = None


class InstitutionWithState(InstitutionResponse):
    """Institution response with current state"""
    
    current_state: Optional[InstitutionStateSnapshot] = None
    total_outbound_exposure: Optional[Decimal] = None
    total_inbound_exposure: Optional[Decimal] = None


class InstitutionListResponse(BaseSchema):
    """Paginated list of institutions"""
    
    items: List[InstitutionResponse]
    total: int
    page: int
    limit: int
    pages: int


class InstitutionSummary(BaseSchema):
    """Brief institution summary for references"""
    
    id: UUID
    external_id: str
    name: str
    type: InstitutionType
    tier: SystemicTier
