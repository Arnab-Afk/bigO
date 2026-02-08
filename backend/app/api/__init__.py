"""API routes package"""
from fastapi import APIRouter

from app.api.v1 import institutions, exposures, network, simulations, scenarios, health, ml, ccp

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(
    health.router,
    tags=["Health"]
)

api_router.include_router(
    institutions.router,
    prefix="/institutions",
    tags=["Institutions"]
)

api_router.include_router(
    exposures.router,
    prefix="/exposures",
    tags=["Exposures"]
)

api_router.include_router(
    network.router,
    prefix="/network",
    tags=["Network"]
)

api_router.include_router(
    simulations.router,
    prefix="/simulations",
    tags=["Simulations"]
)

api_router.include_router(
    scenarios.router,
    prefix="/scenarios",
    tags=["Scenarios"]
)

# ML and CCP endpoints
api_router.include_router(
    ml.router,
    tags=["Machine Learning"]
)

api_router.include_router(
    ccp.router,
    tags=["CCP Risk Analysis"]
)

