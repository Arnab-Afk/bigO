"""
Institution State Pydantic schemas
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import Field

from app.schemas.common import BaseSchema, TimestampMixin, IDMixin


class InstitutionStateBase(BaseSchema):
    """Base institution state schema"""
    
    timestamp: datetime = Field(
        ...,
        description="Point in time for this state"
    )
    
    # Capital Metrics
    capital_ratio: Decimal = Field(
        ...,
        ge=0,
        le=1,
        description="Tier 1 capital ratio"
    )
    leverage_ratio: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Leverage ratio"
    )
    total_capital: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total capital"
    )
    risk_weighted_assets: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Risk-weighted assets"
    )
    
    # Liquidity Metrics
    liquidity_coverage_ratio: Optional[Decimal] = Field(
        None,
        ge=0,
        description="LCR"
    )
    net_stable_funding_ratio: Optional[Decimal] = Field(
        None,
        ge=0,
        description="NSFR"
    )
    liquidity_buffer: Decimal = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Normalized liquidity buffer"
    )
    
    # Exposure Metrics
    total_credit_exposure: Decimal = Field(
        default=0,
        ge=0,
        description="Total credit exposure"
    )
    total_market_exposure: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total market risk exposure"
    )
    
    # Risk Metrics
    default_probability: Decimal = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Estimated PD"
    )
    stress_level: Decimal = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Current stress level"
    )
    risk_score: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="Composite risk score"
    )
    
    # Behavioral Parameters
    risk_appetite: Decimal = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Risk appetite coefficient"
    )
    margin_sensitivity: Decimal = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Sensitivity to margin changes"
    )


class InstitutionStateCreate(InstitutionStateBase):
    """Schema for creating a new institution state"""
    
    institution_id: UUID = Field(
        ...,
        description="Institution ID"
    )
    source: Optional[str] = Field(
        None,
        description="Data source"
    )
    metadata_: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        alias="metadata"
    )


class InstitutionStateResponse(InstitutionStateBase, IDMixin, TimestampMixin):
    """Schema for institution state response"""
    
    institution_id: UUID
    source: Optional[str] = None
    
    class Config:
        from_attributes = True


class InstitutionStateTimeSeries(BaseSchema):
    """Time series of institution states"""
    
    institution_id: UUID
    states: list[InstitutionStateResponse]
    start_time: datetime
    end_time: datetime
    count: int
