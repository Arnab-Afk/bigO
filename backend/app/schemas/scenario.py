"""
Scenario and Shock Pydantic schemas
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field

from app.models.scenario import ShockType
from app.schemas.common import BaseSchema, TimestampMixin, IDMixin


class ShockBase(BaseSchema):
    """Base shock schema"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Shock name"
    )
    description: Optional[str] = Field(
        None,
        description="Shock description"
    )
    shock_type: ShockType = Field(
        ...,
        description="Type of shock"
    )
    target_type: str = Field(
        default="institution",
        description="Target type"
    )
    target_id: Optional[str] = Field(
        None,
        description="Target ID"
    )
    magnitude: Decimal = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Shock magnitude"
    )
    duration: int = Field(
        default=1,
        ge=1,
        description="Duration in timesteps"
    )
    trigger_timestep: int = Field(
        default=1,
        ge=0,
        description="Trigger timestep"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters"
    )


class ShockCreate(ShockBase):
    """Schema for creating a shock"""
    pass


class ShockResponse(ShockBase, IDMixin, TimestampMixin):
    """Schema for shock response"""
    
    scenario_id: UUID
    
    class Config:
        from_attributes = True


class ScenarioBase(BaseSchema):
    """Base scenario schema"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Scenario name"
    )
    description: Optional[str] = Field(
        None,
        description="Scenario description"
    )
    category: str = Field(
        default="custom",
        description="Scenario category"
    )
    is_template: bool = Field(
        default=False,
        description="Whether this is a template"
    )
    
    # Simulation Parameters
    num_timesteps: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of timesteps"
    )
    convergence_threshold: Decimal = Field(
        default=Decimal("0.000001"),
        description="Convergence threshold"
    )
    
    # Risk Parameters
    base_volatility: Decimal = Field(
        default=Decimal("0.15"),
        ge=0,
        le=1,
        description="Base volatility"
    )
    liquidity_premium: Decimal = Field(
        default=Decimal("0.02"),
        ge=0,
        description="Liquidity premium"
    )
    
    # Agent Behavior
    risk_aversion_min: Decimal = Field(
        default=Decimal("0.3"),
        ge=0,
        le=1,
        description="Min risk aversion"
    )
    risk_aversion_max: Decimal = Field(
        default=Decimal("0.7"),
        ge=0,
        le=1,
        description="Max risk aversion"
    )
    information_delay: int = Field(
        default=1,
        ge=0,
        description="Information delay"
    )
    
    # Additional
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters"
    )


class ScenarioCreate(ScenarioBase):
    """Schema for creating a scenario"""
    
    shocks: List[ShockCreate] = Field(
        default_factory=list,
        description="Shocks in this scenario"
    )
    shock_timing: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Shock timing mapping"
    )


class ScenarioResponse(ScenarioBase, IDMixin, TimestampMixin):
    """Schema for scenario response"""
    
    shock_timing: Dict[str, List[str]] = Field(default_factory=dict)
    shocks: List[ShockResponse] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ScenarioListResponse(BaseSchema):
    """Paginated list of scenarios"""
    
    items: List[ScenarioResponse]
    total: int
    page: int
    limit: int
    pages: int


class ScenarioSummary(BaseSchema):
    """Brief scenario summary"""
    
    id: UUID
    name: str
    category: str
    num_timesteps: int
    num_shocks: int
    is_template: bool
