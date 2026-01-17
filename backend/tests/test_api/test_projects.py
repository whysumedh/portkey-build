"""Tests for project API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, sample_project_data: dict):
    """Test creating a new project."""
    response = await client.post("/api/v1/projects", json=sample_project_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == sample_project_data["name"]
    assert data["agent_purpose"] == sample_project_data["agent_purpose"]
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_projects(client: AsyncClient, sample_project_data: dict):
    """Test listing projects."""
    # Create a project first
    await client.post("/api/v1/projects", json=sample_project_data)
    
    response = await client.get("/api/v1/projects")
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_get_project(client: AsyncClient, sample_project_data: dict):
    """Test getting a specific project."""
    # Create a project first
    create_response = await client.post("/api/v1/projects", json=sample_project_data)
    project_id = create_response.json()["id"]
    
    response = await client.get(f"/api/v1/projects/{project_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == project_id
    assert data["name"] == sample_project_data["name"]


@pytest.mark.asyncio
async def test_update_project(client: AsyncClient, sample_project_data: dict):
    """Test updating a project."""
    # Create a project first
    create_response = await client.post("/api/v1/projects", json=sample_project_data)
    project_id = create_response.json()["id"]
    
    update_data = {"name": "Updated Agent Name"}
    response = await client.patch(f"/api/v1/projects/{project_id}", json=update_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Agent Name"
    assert data["version"] == 2  # Version should increment


@pytest.mark.asyncio
async def test_delete_project(client: AsyncClient, sample_project_data: dict):
    """Test deleting a project."""
    # Create a project first
    create_response = await client.post("/api/v1/projects", json=sample_project_data)
    project_id = create_response.json()["id"]
    
    response = await client.delete(f"/api/v1/projects/{project_id}")
    
    assert response.status_code == 204
    
    # Verify it's deleted
    get_response = await client.get(f"/api/v1/projects/{project_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_project(client: AsyncClient):
    """Test getting a project that doesn't exist."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/projects/{fake_id}")
    
    assert response.status_code == 404
