"""
Simulation Pydantic schemas
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field

from app.models.simulation import SimulationStatus
from app.schemas.common import BaseSchema, TimestampMixin, IDMixin


class SimulationBase(BaseSchema):
    """Base simulation schema"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Simulation name"
    )
    description: Optional[str] = Field(
        None,
        description="Simulation description"
    )
    total_timesteps: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Total planned timesteps"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Simulation parameters"
    )


class SimulationCreate(SimulationBase):
    """Schema for creating a new simulation"""
    
    scenario_id: Optional[UUID] = Field(
        None,
        description="Scenario ID to use"
    )


class SimulationResponse(SimulationBase, IDMixin, TimestampMixin):
    """Schema for simulation response"""
    
    scenario_id: Optional[UUID] = None
    status: SimulationStatus
    current_timestep: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Computed
    progress: float = Field(description="Progress percentage")
    duration_seconds: Optional[float] = None
    
    class Config:
        from_attributes = True


class SimulationResultResponse(BaseSchema, IDMixin, TimestampMixin):
    """Schema for simulation result response"""
    
    simulation_id: UUID
    result_type: str
    
    # Metrics
    total_defaults: int
    max_cascade_depth: int
    survival_rate: Decimal
    final_systemic_stress: Decimal
    total_system_loss: Decimal
    time_to_first_default: Optional[int] = None
    
    # Detailed data
    metrics_data: Dict[str, Any] = Field(default_factory=dict)
    timeline_data: Optional[Dict[str, Any]] = None
    cascade_data: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class SimulationListResponse(BaseSchema):
    """Paginated list of simulations"""
    
    items: List[SimulationResponse]
    total: int
    page: int
    limit: int
    pages: int


class SimulationProgress(BaseSchema):
    """Real-time simulation progress"""
    
    simulation_id: UUID
    status: SimulationStatus
    current_timestep: int
    total_timesteps: int
    progress: float
    current_defaults: int
    current_stress_level: Decimal


class TimestepData(BaseSchema):
    """Data for a single simulation timestep"""
    
    timestep: int
    node_states: Dict[str, Dict[str, Any]]
    actions: Dict[str, Dict[str, Any]]
    defaults: List[str]
    metrics: Dict[str, float]


class SimulationTimeline(BaseSchema):
    """Timeline of simulation states"""

    simulation_id: UUID
    timesteps: List[TimestepData]
    start_timestep: int
    end_timestep: int


# --- Payoff Schemas ---


class PayoffAgentSummary(BaseSchema):
    """Per-agent payoff summary"""
    final_utility: float
    cumulative_utility: float
    avg_utility: float
    total_revenue: float = 0.0
    total_credit_risk_cost: float = 0.0
    total_liquidity_risk_cost: float = 0.0
    total_regulatory_cost: float = 0.0


class PayoffSummaryResponse(BaseSchema):
    """Full payoff analysis response"""
    simulation_id: UUID
    payoff_summary: Dict[str, PayoffAgentSummary]
    payoff_matrices: List[Dict[str, Any]]


class PayoffTimelineEntry(BaseSchema):
    """Single timestep payoff for an agent"""
    timestep: int
    total: float
    revenue: float
    credit_risk: float
    liquidity_risk: float
    regulatory: float
    action: str


class PayoffTimelineResponse(BaseSchema):
    """Per-agent payoff evolution"""
    simulation_id: UUID
    agent_id: Optional[UUID] = None
    timeline: Dict[str, Any]


class PayoffMatrixResponse(BaseSchema):
    """Pairwise payoff matrices with Nash equilibria"""
    simulation_id: UUID
    matrices: List[Dict[str, Any]]
    count: int
