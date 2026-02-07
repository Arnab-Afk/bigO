"""
Institution API endpoint tests
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint"""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_list_institutions_empty(client: AsyncClient):
    """Test listing institutions when empty"""
    response = await client.get("/api/v1/institutions")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_create_institution(client: AsyncClient, sample_institution_data: dict):
    """Test creating an institution"""
    response = await client.post("/api/v1/institutions", json=sample_institution_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == sample_institution_data["name"]
    assert data["external_id"] == sample_institution_data["external_id"]
    assert "id" in data


@pytest.mark.asyncio
async def test_get_institution(client: AsyncClient, sample_institution_data: dict):
    """Test getting a specific institution"""
    # Create institution first
    create_response = await client.post("/api/v1/institutions", json=sample_institution_data)
    assert create_response.status_code == 201
    created = create_response.json()
    
    # Get the institution
    response = await client.get(f"/api/v1/institutions/{created['id']}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created["id"]
    assert data["name"] == sample_institution_data["name"]


@pytest.mark.asyncio
async def test_get_institution_not_found(client: AsyncClient):
    """Test getting a non-existent institution"""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/institutions/{fake_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_institution(client: AsyncClient, sample_institution_data: dict):
    """Test updating an institution"""
    # Create institution first
    create_response = await client.post("/api/v1/institutions", json=sample_institution_data)
    created = create_response.json()
    
    # Update it
    update_data = {"name": "Updated Test Bank"}
    response = await client.put(f"/api/v1/institutions/{created['id']}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Test Bank"


@pytest.mark.asyncio
async def test_delete_institution(client: AsyncClient, sample_institution_data: dict):
    """Test deleting an institution"""
    # Create institution first
    create_response = await client.post("/api/v1/institutions", json=sample_institution_data)
    created = create_response.json()
    
    # Delete it
    response = await client.delete(f"/api/v1/institutions/{created['id']}")
    assert response.status_code == 204
    
    # Verify it's gone
    get_response = await client.get(f"/api/v1/institutions/{created['id']}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_create_duplicate_institution(client: AsyncClient, sample_institution_data: dict):
    """Test creating duplicate institution fails"""
    # Create first
    await client.post("/api/v1/institutions", json=sample_institution_data)
    
    # Try to create duplicate
    response = await client.post("/api/v1/institutions", json=sample_institution_data)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_institutions_with_filter(client: AsyncClient, sample_institution_data: dict):
    """Test listing institutions with filters"""
    # Create institution
    await client.post("/api/v1/institutions", json=sample_institution_data)
    
    # Filter by type
    response = await client.get("/api/v1/institutions?type=bank")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    
    # Filter by wrong type
    response = await client.get("/api/v1/institutions?type=ccp")
    data = response.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_institutions_pagination(client: AsyncClient, sample_institution_data: dict):
    """Test institution list pagination"""
    # Create multiple institutions
    for i in range(5):
        data = sample_institution_data.copy()
        data["external_id"] = f"TEST-{i:03d}"
        data["name"] = f"Test Bank {i}"
        await client.post("/api/v1/institutions", json=data)
    
    # Get first page
    response = await client.get("/api/v1/institutions?page=1&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["pages"] == 3
