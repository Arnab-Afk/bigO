"""
Pytest configuration and fixtures
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app


# Test database URL
TEST_DATABASE_URL = settings.DATABASE_URL.replace(
    settings.POSTGRES_DB,
    f"{settings.POSTGRES_DB}_test"
)


# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=NullPool,
)

# Create test session factory
TestSessionFactory = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test"""
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestSessionFactory() as session:
        yield session
        await session.rollback()
    
    # Drop all tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with overridden database"""
    
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_institution_data() -> dict:
    """Sample institution data for tests"""
    return {
        "external_id": "TEST-001",
        "name": "Test Bank",
        "short_name": "TB",
        "type": "bank",
        "tier": "tier_2",
        "jurisdiction": "US",
        "region": "North America",
        "description": "Test institution",
        "is_active": True,
    }


@pytest.fixture
def sample_exposure_data() -> dict:
    """Sample exposure data for tests"""
    return {
        "exposure_type": "interbank_lending",
        "gross_exposure": "1000000000.00",
        "net_exposure": "800000000.00",
        "collateral_value": "200000000.00",
        "collateral_haircut": "0.05",
        "recovery_rate": "0.55",
        "contagion_probability": "0.15",
        "settlement_urgency": "0.5",
        "effective_date": "2024-01-01",
        "currency": "USD",
    }
