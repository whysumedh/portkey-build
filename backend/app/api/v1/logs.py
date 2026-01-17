"""Log management API routes."""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.project import Project
from app.schemas.log_entry import (
    LogEntryResponse,
    LogEntryListResponse,
    LogSyncRequest,
    LogSyncResponse,
    LogStatsResponse,
)
from app.services.ingestion.log_ingestion import LogIngestionService

logger = get_logger(__name__)
router = APIRouter()


@router.post("/{project_id}/sync", response_model=LogSyncResponse)
async def sync_project_logs(
    project_id: uuid.UUID,
    sync_request: LogSyncRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> LogSyncResponse:
    """
    Sync logs from Portkey for a project.
    
    This endpoint triggers a sync of production logs from Portkey's
    observability system into the local database for analysis.
    """
    # Get project
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    if not project.portkey_virtual_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project does not have a Portkey virtual key configured",
        )
    
    logger.info("Starting log sync", project_id=str(project_id))
    
    # Perform sync
    ingestion_service = LogIngestionService(db)
    sync_result = await ingestion_service.sync_project_logs(
        project=project,
        start_date=sync_request.start_date,
        end_date=sync_request.end_date,
        limit=sync_request.limit,
        include_prompt=sync_request.include_prompt,
    )
    
    return LogSyncResponse(**sync_result)


@router.get("/{project_id}", response_model=LogEntryListResponse)
async def list_project_logs(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    model: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    has_error: bool | None = None,
) -> LogEntryListResponse:
    """
    List logs for a project with pagination and filters.
    """
    # Verify project exists
    project_result = await db.execute(
        select(Project.id).where(Project.id == project_id)
    )
    if not project_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    # Build query
    query = select(LogEntry).where(LogEntry.project_id == project_id)
    count_query = select(func.count(LogEntry.id)).where(LogEntry.project_id == project_id)
    
    # Apply filters
    if start_date:
        query = query.where(LogEntry.timestamp >= start_date)
        count_query = count_query.where(LogEntry.timestamp >= start_date)
    if end_date:
        query = query.where(LogEntry.timestamp <= end_date)
        count_query = count_query.where(LogEntry.timestamp <= end_date)
    if model:
        query = query.where(LogEntry.model == model)
        count_query = count_query.where(LogEntry.model == model)
    if status_filter:
        query = query.where(LogEntry.status == status_filter)
        count_query = count_query.where(LogEntry.status == status_filter)
    if has_error is not None:
        if has_error:
            query = query.where(LogEntry.error_message.isnot(None))
            count_query = count_query.where(LogEntry.error_message.isnot(None))
        else:
            query = query.where(LogEntry.error_message.is_(None))
            count_query = count_query.where(LogEntry.error_message.is_(None))
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(LogEntry.timestamp.desc())
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return LogEntryListResponse(
        items=logs,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}/stats", response_model=LogStatsResponse)
async def get_log_stats(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LogStatsResponse:
    """
    Get statistics about logs for a project.
    """
    # Verify project exists
    project_result = await db.execute(
        select(Project.id).where(Project.id == project_id)
    )
    if not project_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    
    ingestion_service = LogIngestionService(db)
    stats = await ingestion_service.get_log_stats(project_id)
    
    return LogStatsResponse(**stats)


@router.get("/{project_id}/entry/{log_id}", response_model=LogEntryResponse)
async def get_log_entry(
    project_id: uuid.UUID,
    log_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LogEntry:
    """
    Get a specific log entry by ID.
    """
    result = await db.execute(
        select(LogEntry).where(
            LogEntry.id == log_id,
            LogEntry.project_id == project_id,
        )
    )
    log_entry = result.scalar_one_or_none()
    
    if not log_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log entry {log_id} not found in project {project_id}",
        )
    
    return log_entry


@router.get("/{project_id}/models", response_model=list[str])
async def get_project_models(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[str]:
    """
    Get list of unique models used in project logs.
    """
    result = await db.execute(
        select(LogEntry.model)
        .where(LogEntry.project_id == project_id)
        .distinct()
    )
    return [row[0] for row in result.all()]


@router.get("/{project_id}/providers", response_model=list[str])
async def get_project_providers(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[str]:
    """
    Get list of unique providers used in project logs.
    """
    result = await db.execute(
        select(LogEntry.provider)
        .where(LogEntry.project_id == project_id)
        .distinct()
    )
    return [row[0] for row in result.all()]
