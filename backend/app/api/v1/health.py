"""
Health check endpoints
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from app.core.config import settings
from app.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Health check endpoint
    
    Returns the status of the API and its dependent services.
    """
    services = {}
    
    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        services["postgresql"] = "healthy"
    except Exception as e:
        services["postgresql"] = f"unhealthy: {str(e)}"
    
    # Check Redis (optional, wrapped in try-catch)
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        services["redis"] = "healthy"
        await r.close()
    except Exception as e:
        services["redis"] = f"unavailable: {str(e)}"
    
    # Determine overall status
    postgres_healthy = services.get("postgresql") == "healthy"
    status = "healthy" if postgres_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        services=services
    )


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Readiness check for Kubernetes
    
    Returns 200 if the service is ready to accept traffic.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        return {"ready": False}


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes
    
    Returns 200 if the service is alive.
    """
    return {"alive": True}
