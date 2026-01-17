"""Scheduler API routes for managing evaluation schedules."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.services.scheduler.scheduler import get_scheduler, SchedulerMode

logger = get_logger(__name__)
router = APIRouter()


class SchedulerStatus(BaseModel):
    """Scheduler status response."""
    running: bool
    mode: str
    jobs: list[dict]


class AdHocEvaluationRequest(BaseModel):
    """Request for ad-hoc evaluation."""
    project_id: uuid.UUID
    candidate_models: list[str] | None = None
    sample_size: int | None = None


class AdHocEvaluationResponse(BaseModel):
    """Response for ad-hoc evaluation."""
    evaluation_run_id: uuid.UUID
    message: str


@router.get("/status", response_model=SchedulerStatus)
async def get_scheduler_status() -> SchedulerStatus:
    """Get current scheduler status."""
    scheduler = get_scheduler()
    return SchedulerStatus(
        running=scheduler.is_running,
        mode=scheduler._mode.value,
        jobs=scheduler.get_scheduled_jobs(),
    )


@router.post("/trigger", response_model=AdHocEvaluationResponse)
async def trigger_ad_hoc_evaluation(
    request: AdHocEvaluationRequest,
) -> AdHocEvaluationResponse:
    """
    Trigger an ad-hoc evaluation for a project.
    
    This immediately starts an evaluation run outside of the
    normal schedule.
    """
    scheduler = get_scheduler()
    
    try:
        evaluation_id = await scheduler.trigger_ad_hoc_evaluation(
            project_id=request.project_id,
            candidate_models=request.candidate_models,
            sample_size=request.sample_size,
        )
        
        return AdHocEvaluationResponse(
            evaluation_run_id=evaluation_id,
            message="Evaluation triggered successfully",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to trigger evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger evaluation: {str(e)}",
        )


@router.post("/mode/{mode}")
async def set_scheduler_mode(mode: str) -> dict:
    """
    Set scheduler mode.
    
    Modes:
    - disabled: No automatic evaluations
    - scheduled: Normal scheduled evaluations
    - shadow: Shadow evaluation mode (async, no impact on prod)
    """
    try:
        scheduler_mode = SchedulerMode(mode)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {mode}. Valid modes: disabled, scheduled, shadow",
        )
    
    scheduler = get_scheduler()
    scheduler.set_mode(scheduler_mode)
    
    return {"mode": mode, "message": f"Scheduler mode set to {mode}"}
