"""Pydantic schemas package"""
from app.schemas.institution import (
    InstitutionCreate,
    InstitutionUpdate,
    InstitutionResponse,
    InstitutionListResponse,
    InstitutionWithState,
)
from app.schemas.institution_state import (
    InstitutionStateCreate,
    InstitutionStateResponse,
)
from app.schemas.exposure import (
    ExposureCreate,
    ExposureUpdate,
    ExposureResponse,
    ExposureListResponse,
)
from app.schemas.simulation import (
    SimulationCreate,
    SimulationResponse,
    SimulationResultResponse,
)
from app.schemas.scenario import (
    ScenarioCreate,
    ScenarioResponse,
    ShockCreate,
    ShockResponse,
)
from app.schemas.common import (
    PaginationParams,
    PaginatedResponse,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    # Institution
    "InstitutionCreate",
    "InstitutionUpdate",
    "InstitutionResponse",
    "InstitutionListResponse",
    "InstitutionWithState",
    # Institution State
    "InstitutionStateCreate",
    "InstitutionStateResponse",
    # Exposure
    "ExposureCreate",
    "ExposureUpdate",
    "ExposureResponse",
    "ExposureListResponse",
    # Simulation
    "SimulationCreate",
    "SimulationResponse",
    "SimulationResultResponse",
    # Scenario
    "ScenarioCreate",
    "ScenarioResponse",
    "ShockCreate",
    "ShockResponse",
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "HealthResponse",
    "ErrorResponse",
]
