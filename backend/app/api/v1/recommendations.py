"""Recommendation API routes - model recommendations with explainability."""

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.project import Project
from app.models.recommendation import Recommendation
from app.schemas.recommendation import (
    RecommendationResponse,
    RecommendationListResponse,
    RecommendationAcknowledge,
)
from app.services.recommendation.recommender import RecommendationEngine

logger = get_logger(__name__)
router = APIRouter()


@router.post("/{project_id}/generate", response_model=RecommendationResponse)
async def generate_recommendation(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    evaluation_run_id: uuid.UUID | None = None,
) -> Recommendation:
    """
    Generate a new recommendation for a project.
    
    This analyzes aggregated metrics and produces a ranked
    list of model candidates with confidence scores and explanations.
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

    engine = RecommendationEngine(db)
    
    try:
        recommendation = await engine.generate_recommendation(
            project_id=project_id,
            evaluation_run_id=evaluation_run_id,
        )
        return recommendation
    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendation: {str(e)}",
        )


@router.get("/{project_id}", response_model=RecommendationListResponse)
async def list_recommendations(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    acknowledged: bool | None = None,
) -> RecommendationListResponse:
    """
    List recommendations for a project.
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
    query = (
        select(Recommendation)
        .options(selectinload(Recommendation.candidates))
        .where(Recommendation.project_id == project_id)
    )
    count_query = select(func.count(Recommendation.id)).where(
        Recommendation.project_id == project_id
    )
    
    if acknowledged is not None:
        query = query.where(Recommendation.acknowledged == acknowledged)
        count_query = count_query.where(Recommendation.acknowledged == acknowledged)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(Recommendation.created_at.desc())
    result = await db.execute(query)
    recommendations = result.scalars().all()
    
    return RecommendationListResponse(
        items=recommendations,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{project_id}/latest", response_model=RecommendationResponse)
async def get_latest_recommendation(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Recommendation:
    """
    Get the latest recommendation for a project.
    """
    result = await db.execute(
        select(Recommendation)
        .options(selectinload(Recommendation.candidates))
        .where(Recommendation.project_id == project_id)
        .order_by(Recommendation.created_at.desc())
        .limit(1)
    )
    recommendation = result.scalar_one_or_none()
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendations found for project {project_id}",
        )
    
    return recommendation


@router.get("/{project_id}/{recommendation_id}", response_model=RecommendationResponse)
async def get_recommendation(
    project_id: uuid.UUID,
    recommendation_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Recommendation:
    """
    Get a specific recommendation with full details.
    """
    result = await db.execute(
        select(Recommendation)
        .options(selectinload(Recommendation.candidates))
        .where(
            Recommendation.id == recommendation_id,
            Recommendation.project_id == project_id,
        )
    )
    recommendation = result.scalar_one_or_none()
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation {recommendation_id} not found",
        )
    
    return recommendation


@router.post("/{project_id}/{recommendation_id}/acknowledge", response_model=RecommendationResponse)
async def acknowledge_recommendation(
    project_id: uuid.UUID,
    recommendation_id: uuid.UUID,
    acknowledge_data: RecommendationAcknowledge,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Recommendation:
    """
    Acknowledge a recommendation.
    
    Recommendations are advisory only - this endpoint records
    that the recommendation was reviewed and what action was taken.
    
    Action types:
    - "accepted": User will implement the recommendation
    - "rejected": User chose not to implement
    - "deferred": User will consider later
    - "partial": User implemented partially
    """
    result = await db.execute(
        select(Recommendation)
        .options(selectinload(Recommendation.candidates))
        .where(
            Recommendation.id == recommendation_id,
            Recommendation.project_id == project_id,
        )
    )
    recommendation = result.scalar_one_or_none()
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation {recommendation_id} not found",
        )
    
    recommendation.acknowledged = True
    recommendation.acknowledged_at = datetime.now(timezone.utc)
    recommendation.acknowledged_by = acknowledge_data.acknowledged_by
    recommendation.action_taken = acknowledge_data.action_taken
    
    await db.commit()
    await db.refresh(recommendation)
    
    logger.info(
        "Recommendation acknowledged",
        recommendation_id=str(recommendation_id),
        action=acknowledge_data.action_taken,
    )
    
    return recommendation
