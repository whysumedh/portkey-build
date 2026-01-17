"""Evaluation API routes - run model evaluations with AI judges."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.project import Project
from app.models.evaluation import EvaluationRun, ReplayRun, EvaluationStatus
from app.schemas.evaluation import (
    EvaluationRunCreate,
    EvaluationRunResponse,
    EvaluationRunListResponse,
)
from app.services.selector.model_selector import ModelSelector
from app.services.analytics.engine import AnalyticsEngine

logger = get_logger(__name__)
router = APIRouter()


async def run_evaluation_background(
    evaluation_run_id: uuid.UUID,
    db_url: str,
):
    """Background task to run the full evaluation pipeline."""
    # This would be implemented with a separate database session
    # For production, use Celery or similar task queue
    logger.info(f"Background evaluation started: {evaluation_run_id}")


@router.post("", response_model=EvaluationRunResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(
    evaluation_data: EvaluationRunCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> EvaluationRun:
    """
    Create and start a new evaluation run.
    
    An evaluation run:
    1. Samples historical prompts from the project
    2. Replays them on candidate models
    3. Runs AI judges on the outputs
    4. Produces recommendations
    
    The evaluation runs asynchronously in the background.
    """
    # Verify project exists
    result = await db.execute(
        select(Project)
        .options(
            selectinload(Project.success_criteria),
            selectinload(Project.tolerance_levels),
        )
        .where(Project.id == evaluation_data.project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {evaluation_data.project_id} not found",
        )
    
    # Calculate time range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=evaluation_data.time_range_days)
    
    # Create evaluation run
    evaluation_run = EvaluationRun(
        project_id=project.id,
        trigger_type=evaluation_data.trigger_type,
        sample_size=evaluation_data.sample_size,
        candidate_models=evaluation_data.candidate_models,
        logs_start_date=start_date,
        logs_end_date=end_date,
        status=EvaluationStatus.PENDING.value,
    )
    
    db.add(evaluation_run)
    await db.commit()
    await db.refresh(evaluation_run)
    
    # Note: In production, trigger background evaluation via Celery
    # background_tasks.add_task(run_evaluation_background, evaluation_run.id, str(settings.database_url))
    
    logger.info(
        "Evaluation run created",
        evaluation_id=str(evaluation_run.id),
        project_id=str(project.id),
    )
    
    return evaluation_run


@router.get("/{project_id}", response_model=EvaluationRunListResponse)
async def list_evaluation_runs(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
) -> EvaluationRunListResponse:
    """
    List evaluation runs for a project.
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
    query = select(EvaluationRun).where(EvaluationRun.project_id == project_id)
    count_query = select(func.count(EvaluationRun.id)).where(
        EvaluationRun.project_id == project_id
    )
    
    if status_filter:
        query = query.where(EvaluationRun.status == status_filter)
        count_query = count_query.where(EvaluationRun.status == status_filter)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(EvaluationRun.created_at.desc())
    result = await db.execute(query)
    runs = result.scalars().all()
    
    return EvaluationRunListResponse(
        items=runs,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}/{evaluation_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(
    project_id: uuid.UUID,
    evaluation_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EvaluationRun:
    """
    Get a specific evaluation run with replay details.
    """
    result = await db.execute(
        select(EvaluationRun)
        .options(selectinload(EvaluationRun.replay_runs))
        .where(
            EvaluationRun.id == evaluation_id,
            EvaluationRun.project_id == project_id,
        )
    )
    evaluation_run = result.scalar_one_or_none()
    
    if not evaluation_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run {evaluation_id} not found",
        )
    
    return evaluation_run


@router.post("/{project_id}/{evaluation_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_evaluation_run(
    project_id: uuid.UUID,
    evaluation_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Cancel a running evaluation.
    """
    result = await db.execute(
        select(EvaluationRun).where(
            EvaluationRun.id == evaluation_id,
            EvaluationRun.project_id == project_id,
        )
    )
    evaluation_run = result.scalar_one_or_none()
    
    if not evaluation_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run {evaluation_id} not found",
        )
    
    if evaluation_run.status not in [EvaluationStatus.PENDING.value, EvaluationStatus.RUNNING.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel evaluation in status: {evaluation_run.status}",
        )
    
    evaluation_run.status = EvaluationStatus.CANCELLED.value
    await db.commit()
    
    return {"status": "cancelled", "evaluation_id": str(evaluation_id)}
