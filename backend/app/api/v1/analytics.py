"""Analytics API routes - safe interface for statistical analysis."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.project import Project
from app.schemas.analytics import AnalysisRequest, AnalysisResponse
from app.services.analytics.request_handler import (
    AnalysisRequestHandler,
    validate_analysis_request,
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=AnalysisResponse)
async def run_analysis(
    request: AnalysisRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnalysisResponse:
    """
    Execute a structured analysis request.
    
    This is the safe interface for statistical analysis.
    LLMs and users submit structured requests instead of arbitrary code.
    
    ## Supported Analysis Types
    
    - **distribution**: Histogram of a field
      - params: `field`, `bins` (optional), `normalize` (optional)
    - **percentile**: Percentile values for a metric
      - params: `metric`, `percentiles` (optional, default [50, 75, 90, 95, 99])
    - **correlation**: Correlation between two fields
      - params: `x`, `y`, `method` (optional: pearson, spearman, kendall)
    - **aggregation**: Group-by aggregations
      - params: `metric`, `group_by` (optional), `aggregations` (optional)
    - **clustering**: K-means clustering
      - params: `features` (list), `n_clusters` (optional)
    - **sample**: Stratified sampling for replay
      - params: `n`, `stratify_by` (optional), `random_seed` (optional)
    
    ## Allowed Fields
    
    - prompt_length, input_tokens, output_tokens, total_tokens
    - latency_ms, cost_usd, refusal, status
    - model, provider, timestamp
    - hour_of_day, day_of_week (derived)
    """
    # Verify project exists
    result = await db.execute(
        select(Project.id).where(Project.id == request.project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {request.project_id} not found",
        )

    # Validate request structure
    validation_errors = validate_analysis_request(request.model_dump())
    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": validation_errors},
        )

    # Execute analysis
    handler = AnalysisRequestHandler(db)
    response = await handler.handle_request(request)
    
    return response


@router.get("/{project_id}/summary")
async def get_project_analytics_summary(
    project_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """
    Get comprehensive summary statistics for a project's logs.
    
    Returns:
    - Total log count and date range
    - Models and providers used
    - Status distribution
    - Refusal rate
    - Numeric summaries (tokens, latency, cost) with percentiles
    """
    # Verify project exists
    result = await db.execute(
        select(Project.id).where(Project.id == project_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    handler = AnalysisRequestHandler(db)
    
    try:
        return await handler.get_summary(project_id)
    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute summary: {str(e)}",
        )


@router.get("/{project_id}/allowed-fields")
async def get_allowed_fields() -> dict[str, list[str]]:
    """
    Get list of fields allowed for analysis.
    
    This endpoint documents which fields can be used in analysis requests.
    """
    from app.services.analytics.engine import AnalyticsEngine
    
    return {
        "allowed_fields": list(AnalyticsEngine.ALLOWED_FIELDS.keys()),
        "analysis_types": [
            "distribution",
            "percentile", 
            "correlation",
            "aggregation",
            "clustering",
            "sample",
        ],
    }
