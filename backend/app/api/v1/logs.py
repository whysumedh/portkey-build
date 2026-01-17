"""Log management API routes."""

import uuid
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from pydantic import BaseModel
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
from app.services.ingestion.portkey_client import get_portkey_client, PortkeyClientError

logger = get_logger(__name__)
router = APIRouter()


# ===========================
# Pydantic models for Portkey logs
# ===========================

class PortkeyLogResponse(BaseModel):
    """Response model for a single Portkey log."""
    id: str
    trace_id: str | None = None
    span_id: str | None = None
    created_at: str | None = None
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    response_time: float | None = None
    status: str | None = None
    is_success: bool | None = None
    request: dict | None = None
    response: dict | None = None
    metadata: dict | None = None


class PortkeyLogsListResponse(BaseModel):
    """Response model for list of Portkey logs."""
    logs: list[dict[str, Any]]
    total: int
    workspace_id: str
    export_id: str | None = None


class PortkeyExportStatusResponse(BaseModel):
    """Response model for export status."""
    export_id: str
    status: str
    created_at: str | None = None
    filters: dict | None = None


# ===========================
# Direct Portkey Logs Endpoints (with local caching)
# ===========================

class CachedLogsResponse(BaseModel):
    """Response model for cached logs from database."""
    logs: list[dict[str, Any]]
    total: int
    from_cache: bool
    last_synced: datetime | None = None


@router.get("/portkey/logs", response_model=CachedLogsResponse)
async def get_portkey_logs(
    db: Annotated[AsyncSession, Depends(get_db)],
    workspace_id: str | None = Query(default="2d469afe-6e46-4929-ab71-21de003b711d", description="Portkey workspace ID"),
    refresh: bool = Query(default=False, description="Force refresh from Portkey API"),
    hours: int = Query(default=24, ge=1, le=168, description="Hours of logs to fetch when refreshing (default 24, max 168)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum logs to return"),
) -> CachedLogsResponse:
    """
    Get logs from local cache (database) or refresh from Portkey.
    
    - By default, returns logs from the local database (fast, no API calls)
    - Pass `refresh=true` to fetch new logs from Portkey and store them
    - Logs without a project_id are workspace-level logs (log pool)
    """
    
    # If refresh requested, fetch from Portkey and store in DB
    if refresh:
        logger.info("Refreshing logs from Portkey", workspace_id=workspace_id)
        client = get_portkey_client()
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        try:
            portkey_logs = await client.fetch_all_logs(
                workspace_id=workspace_id,
                start_date=start_date,
                end_date=end_date,
                max_logs=limit,
            )
            
            # Store logs in database (without project_id = workspace log pool)
            stored_count = 0
            for log_data in portkey_logs:
                log_id = log_data.get("id") or log_data.get("log_id")
                if not log_id:
                    continue
                
                # Check if log already exists
                existing = await db.execute(
                    select(LogEntry.id).where(LogEntry.portkey_log_id == log_id)
                )
                if existing.scalar_one_or_none():
                    continue
                
                # Create log entry (no project_id = workspace pool)
                try:
                    log_entry = _convert_portkey_log_to_entry(log_data)
                    db.add(log_entry)
                    stored_count += 1
                except Exception as e:
                    logger.warning(f"Failed to store log {log_id}: {e}")
            
            if stored_count > 0:
                await db.commit()
                logger.info(f"Stored {stored_count} new logs in database")
            
        except PortkeyClientError as e:
            logger.error(f"Failed to fetch Portkey logs: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch logs from Portkey: {str(e)}",
            )
    
    # Return logs from database (workspace-level logs without project_id)
    query = (
        select(LogEntry)
        .where(LogEntry.project_id.is_(None))
        .order_by(LogEntry.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    db_logs = result.scalars().all()
    
    # Get total count
    count_result = await db.execute(
        select(func.count(LogEntry.id)).where(LogEntry.project_id.is_(None))
    )
    total = count_result.scalar() or 0
    
    # Get last synced time
    last_synced_result = await db.execute(
        select(func.max(LogEntry.ingested_at)).where(LogEntry.project_id.is_(None))
    )
    last_synced = last_synced_result.scalar()
    
    # Convert to dict format for response
    logs_data = [_log_entry_to_dict(log) for log in db_logs]
    
    return CachedLogsResponse(
        logs=logs_data,
        total=total,
        from_cache=not refresh,
        last_synced=last_synced,
    )


def _convert_portkey_log_to_entry(portkey_data: dict) -> LogEntry:
    """Convert Portkey export format log to LogEntry (without project_id)."""
    import uuid as uuid_module
    from datetime import timezone
    
    log_id = portkey_data.get("id") or portkey_data.get("log_id") or str(uuid_module.uuid4())
    
    # Parse timestamp
    time_str = portkey_data.get("time_of_generation") or portkey_data.get("created_at")
    if isinstance(time_str, str):
        try:
            timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)
    
    # Get model and provider
    model = portkey_data.get("ai_model") or portkey_data.get("model") or "unknown"
    provider = portkey_data.get("ai_org") or portkey_data.get("ai_provider") or portkey_data.get("provider") or "unknown"
    
    # Get tokens
    input_tokens = portkey_data.get("req_units") or portkey_data.get("usage", {}).get("prompt_tokens", 0) or 0
    output_tokens = portkey_data.get("res_units") or portkey_data.get("usage", {}).get("completion_tokens", 0) or 0
    total_tokens = portkey_data.get("total_units") or portkey_data.get("usage", {}).get("total_tokens", 0) or 0
    
    # Get latency and cost
    latency = portkey_data.get("response_time") or portkey_data.get("latency_ms") or 0
    cost = portkey_data.get("cost") or 0.0
    
    # Determine status
    is_success = portkey_data.get("is_success")
    if is_success is True:
        status_val = "success"
    elif is_success is False:
        status_val = "error"
    else:
        status_val = portkey_data.get("status", "success")
    
    # Extract prompt
    prompt_str = ""
    messages = portkey_data.get("request", {}).get("messages", [])
    if messages:
        for msg in messages:
            if msg.get("role") == "user":
                prompt_str = msg.get("content", "")
                break
    
    return LogEntry(
        id=uuid_module.uuid4(),
        project_id=None,  # Workspace-level log (pool)
        portkey_log_id=log_id,
        trace_id=portkey_data.get("trace_id"),
        span_id=portkey_data.get("span_id"),
        timestamp=timestamp,
        endpoint=portkey_data.get("endpoint", "/chat/completions"),
        prompt=prompt_str,
        prompt_hash=LogEntry.compute_hash(prompt_str),
        model=model,
        provider=provider,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
        latency_ms=float(latency),
        cost_usd=float(cost),
        status=status_val,
        refusal=bool(portkey_data.get("refusal", False)),
        log_metadata=portkey_data.get("metadata"),
    )


def _log_entry_to_dict(log: LogEntry) -> dict[str, Any]:
    """Convert LogEntry to dict for API response."""
    return {
        "id": str(log.portkey_log_id),
        "db_id": str(log.id),
        "trace_id": log.trace_id,
        "span_id": log.span_id,
        "time_of_generation": log.timestamp.isoformat() if log.timestamp else None,
        "created_at": log.timestamp.isoformat() if log.timestamp else None,
        "ai_model": log.model,
        "ai_org": log.provider,
        "ai_provider": log.provider,
        "model": log.model,
        "provider": log.provider,
        "req_units": log.input_tokens,
        "res_units": log.output_tokens,
        "total_units": log.total_tokens,
        "prompt_tokens": log.input_tokens,
        "completion_tokens": log.output_tokens,
        "total_tokens": log.total_tokens,
        "cost": log.cost_usd,
        "response_time": log.latency_ms,
        "latency_ms": log.latency_ms,
        "status": log.status,
        "is_success": log.status == "success",
        "refusal": log.refusal,
        "metadata": log.log_metadata,
        "project_id": str(log.project_id) if log.project_id else None,
        "ingested_at": log.ingested_at.isoformat() if log.ingested_at else None,
    }


@router.get("/portkey/logs/{log_id}")
async def get_portkey_log_by_id(log_id: str) -> dict[str, Any]:
    """
    Fetch a specific log entry from Portkey by ID.
    """
    client = get_portkey_client()
    
    try:
        log = await client.get_log_by_id(log_id)
        return log
    except PortkeyClientError as e:
        logger.error(f"Failed to fetch Portkey log {log_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch log from Portkey: {str(e)}",
        )


@router.get("/portkey/exports")
async def list_portkey_exports(
    workspace_id: str | None = Query(default=None, description="Portkey workspace ID"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """
    List existing log exports for a workspace.
    """
    client = get_portkey_client()
    
    try:
        exports = await client.list_log_exports(
            workspace_id=workspace_id,
            limit=limit,
        )
        return exports
    except PortkeyClientError as e:
        logger.error(f"Failed to list Portkey exports: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to list exports from Portkey: {str(e)}",
        )


@router.post("/portkey/exports/create")
async def create_portkey_export(
    workspace_id: str | None = Query(default=None),
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
) -> dict[str, Any]:
    """
    Create a new log export job (does not wait for completion).
    """
    client = get_portkey_client()
    
    try:
        export = await client.create_log_export(
            workspace_id=workspace_id,
            start_date=start_date,
            end_date=end_date,
        )
        
        export_id = export.get("id") or export.get("export_id")
        if export_id:
            # Start the export
            await client.start_log_export(export_id)
        
        return export
    except PortkeyClientError as e:
        logger.error(f"Failed to create Portkey export: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to create export: {str(e)}",
        )


@router.get("/portkey/exports/{export_id}/status")
async def get_portkey_export_status(export_id: str) -> dict[str, Any]:
    """
    Get the status of a log export job.
    """
    client = get_portkey_client()
    
    try:
        return await client.get_log_export(export_id)
    except PortkeyClientError as e:
        logger.error(f"Failed to get export status: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get export status: {str(e)}",
        )


@router.get("/portkey/exports/{export_id}/download")
async def download_portkey_export(export_id: str) -> dict[str, Any]:
    """
    Get the download URL for a completed export.
    """
    client = get_portkey_client()
    
    try:
        return await client.download_log_export(export_id)
    except PortkeyClientError as e:
        logger.error(f"Failed to get export download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get download URL: {str(e)}",
        )


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
