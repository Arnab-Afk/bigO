"""
Exposure Pydantic schemas
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.models.exposure import ExposureType
from app.schemas.common import BaseSchema, TimestampMixin, IDMixin


class ExposureBase(BaseSchema):
    """Base exposure schema"""
    
    exposure_type: ExposureType = Field(
        default=ExposureType.INTERBANK_LENDING,
        description="Type of financial exposure"
    )
    
    # Amounts
    gross_exposure: Decimal = Field(
        ...,
        ge=0,
        description="Gross exposure amount"
    )
    net_exposure: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Net exposure after netting"
    )
    
    # Collateral
    collateral_value: Optional[Decimal] = Field(
        default=0,
        ge=0,
        description="Collateral value"
    )
    collateral_haircut: Decimal = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Collateral haircut"
    )
    
    # Risk Parameters
    recovery_rate: Decimal = Field(
        default=0.55,
        ge=0,
        le=1,
        description="Expected recovery rate"
    )
    contagion_probability: Decimal = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Probability of stress transmission"
    )
    settlement_urgency: Decimal = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Settlement time criticality"
    )
    
    # Dates
    effective_date: date = Field(
        ...,
        description="Date exposure became effective"
    )
    maturity_date: Optional[date] = Field(
        None,
        description="Maturity date"
    )
    
    # Reference
    currency: str = Field(
        default="USD",
        max_length=3,
        description="Currency code"
    )
    netting_agreement_id: Optional[str] = Field(
        None,
        description="Netting agreement reference"
    )


class ExposureCreate(ExposureBase):
    """Schema for creating a new exposure"""
    
    source_institution_id: UUID = Field(
        ...,
        description="Source institution ID (creditor)"
    )
    target_institution_id: UUID = Field(
        ...,
        description="Target institution ID (debtor)"
    )
    valid_from: datetime = Field(
        ...,
        description="Start of validity period"
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="End of validity period"
    )
    metadata_: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        alias="metadata"
    )
    
    @model_validator(mode='after')
    def validate_different_institutions(self):
        if self.source_institution_id == self.target_institution_id:
            raise ValueError("Source and target institutions must be different")
        return self


class ExposureUpdate(BaseSchema):
    """Schema for updating an exposure"""
    
    exposure_type: Optional[ExposureType] = None
    gross_exposure: Optional[Decimal] = Field(None, ge=0)
    net_exposure: Optional[Decimal] = Field(None, ge=0)
    collateral_value: Optional[Decimal] = Field(None, ge=0)
    collateral_haircut: Optional[Decimal] = Field(None, ge=0, le=1)
    recovery_rate: Optional[Decimal] = Field(None, ge=0, le=1)
    contagion_probability: Optional[Decimal] = Field(None, ge=0, le=1)
    settlement_urgency: Optional[Decimal] = Field(None, ge=0, le=1)
    maturity_date: Optional[date] = None
    valid_to: Optional[datetime] = None
    metadata_: Optional[Dict[str, Any]] = Field(None, alias="metadata")


class ExposureResponse(ExposureBase, IDMixin, TimestampMixin):
    """Schema for exposure response"""
    
    source_institution_id: UUID
    target_institution_id: UUID
    valid_from: datetime
    valid_to: Optional[datetime] = None
    
    # Computed fields
    effective_exposure: Optional[Decimal] = None
    loss_given_default: Optional[Decimal] = None
    
    class Config:
        from_attributes = True


class ExposureListResponse(BaseSchema):
    """Paginated list of exposures"""
    
    items: List[ExposureResponse]
    total: int
    page: int
    limit: int
    pages: int


class ExposureSummary(BaseSchema):
    """Exposure summary for network visualization"""
    
    source_id: UUID
    target_id: UUID
    exposure_type: ExposureType
    gross_exposure: Decimal
    contagion_probability: Decimal


class ExposureMatrixEntry(BaseSchema):
    """Entry for exposure matrix"""
    
    source_id: UUID
    source_name: str
    target_id: UUID
    target_name: str
    exposure: Decimal
    exposure_type: ExposureType
