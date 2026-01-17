"""Project management API routes."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.project import Project, SuccessCriteria, ToleranceLevels
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
)
from app.services.ingestion.log_ingestion import LogIngestionService

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """
    Create a new project for an AI agent.
    
    A project represents exactly ONE AI agent being optimized.
    If selected_log_ids is provided, those logs will be imported from Portkey
    and associated with this project.
    """
    logger.info("Creating project", name=project_data.name)
    
    # Create the project
    project = Project(
        name=project_data.name,
        description=project_data.description,
        agent_purpose=project_data.agent_purpose,
        portkey_virtual_key=project_data.portkey_virtual_key,
        portkey_config_id=project_data.portkey_config_id,
        current_model=project_data.current_model,
        current_provider=project_data.current_provider,
        selected_log_ids=project_data.selected_log_ids,
        log_filter_metadata=project_data.log_filter_metadata,
    )
    db.add(project)
    await db.flush()  # Get the project ID
    
    # Create success criteria if provided
    if project_data.success_criteria:
        criteria = SuccessCriteria(
            project_id=project.id,
            **project_data.success_criteria.model_dump(),
        )
        db.add(criteria)
    else:
        # Create default success criteria
        criteria = SuccessCriteria(project_id=project.id)
        db.add(criteria)
    
    # Create tolerance levels if provided
    if project_data.tolerance_levels:
        tolerances = ToleranceLevels(
            project_id=project.id,
            **project_data.tolerance_levels.model_dump(),
        )
        db.add(tolerances)
    else:
        # Create default tolerance levels
        tolerances = ToleranceLevels(project_id=project.id)
        db.add(tolerances)
    
    await db.commit()
    
    # Import selected logs from Portkey if provided
    import_stats = None
    if project_data.selected_log_ids:
        logger.info(
            "Importing selected logs",
            project_id=str(project.id),
            log_count=len(project_data.selected_log_ids),
        )
        ingestion_service = LogIngestionService(db)
        import_stats = await ingestion_service.import_logs_by_ids(
            project_id=project.id,
            log_ids=project_data.selected_log_ids,
        )
        await db.commit()
        logger.info(
            "Logs imported",
            project_id=str(project.id),
            imported=import_stats.get("imported", 0),
            skipped=import_stats.get("skipped", 0),
        )
    
    await db.refresh(project)
    
    # Load relationships
    result = await db.execute(
        select(Project)
        .options(
            selectinload(Project.success_criteria),
            selectinload(Project.tolerance_levels),
        )
        .where(Project.id == project.id)
    )
    project = result.scalar_one()
    
    logger.info("Project created", project_id=str(project.id))
    return project


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    is_active: bool | None = None,
) -> ProjectListResponse:
    """List all projects with pagination."""
    
    # Build query
    query = select(Project).options(
        selectinload(Project.success_criteria),
        selectinload(Project.tolerance_levels),
    )
    
    if is_active is not None:
        query = query.where(Project.is_active == is_active)
    
    # Get total count
    count_query = select(func.count(Project.id))
    if is_active is not None:
        count_query = count_query.where(Project.is_active == is_active)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Project.created_at.desc())
    result = await db.execute(query)
    projects = result.scalars().all()
    
    return ProjectListResponse(
        items=projects,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """Get a project by ID."""
    result = await db.execute(
        select(Project)
        .options(
            selectinload(Project.success_criteria),
            selectinload(Project.tolerance_levels),
        )
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    project_data: ProjectUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """Update a project."""
    result = await db.execute(
        select(Project)
        .options(
            selectinload(Project.success_criteria),
            selectinload(Project.tolerance_levels),
        )
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    # Update project fields
    update_data = project_data.model_dump(exclude_unset=True)
    
    # Handle success criteria separately
    if "success_criteria" in update_data and update_data["success_criteria"]:
        criteria_data = update_data.pop("success_criteria")
        if project.success_criteria:
            for key, value in criteria_data.items():
                setattr(project.success_criteria, key, value)
        else:
            criteria = SuccessCriteria(project_id=project.id, **criteria_data)
            db.add(criteria)
    
    # Handle tolerance levels separately
    if "tolerance_levels" in update_data and update_data["tolerance_levels"]:
        tolerance_data = update_data.pop("tolerance_levels")
        if project.tolerance_levels:
            for key, value in tolerance_data.items():
                setattr(project.tolerance_levels, key, value)
        else:
            tolerances = ToleranceLevels(project_id=project.id, **tolerance_data)
            db.add(tolerances)
    
    # Update remaining fields
    for key, value in update_data.items():
        if value is not None:
            setattr(project, key, value)
    
    # Increment version
    project.version += 1
    
    await db.commit()
    await db.refresh(project)
    
    logger.info("Project updated", project_id=str(project_id), version=project.version)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a project and all associated data."""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    await db.delete(project)
    await db.commit()
    
    logger.info("Project deleted", project_id=str(project_id))
