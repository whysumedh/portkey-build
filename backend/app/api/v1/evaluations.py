"""Evaluation API routes - replay-based model evaluation pipeline."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.project import Project
from app.models.log_entry import LogEntry
from app.models.evaluation import EvaluationRun, ReplayRun, EvaluationStatus, AggregatedMetrics
from app.schemas.evaluation import (
    EvaluationRunCreate,
    EvaluationRunResponse,
    EvaluationRunListResponse,
)

logger = get_logger(__name__)
router = APIRouter()


# ========================
# Request/Response Models
# ========================

class AnalyzeLogsRequest(BaseModel):
    """Request to analyze selected logs."""
    log_ids: list[str]  # UUIDs as strings


class AnalysisReportResponse(BaseModel):
    """Response with log analysis report."""
    analysis_id: str
    project_id: str
    log_count: int
    date_range: dict[str, str | None]
    token_distribution: dict[str, Any]
    latency_metrics: dict[str, float]
    cost_breakdown: dict[str, Any]
    prompt_complexity: dict[str, Any]
    error_patterns: dict[str, Any]
    model_performance: list[dict[str, Any]]
    key_insights: list[str]
    criteria_assessment: dict[str, Any]


class SelectModelsRequest(BaseModel):
    """Request to select candidate models."""
    analysis_id: str | None = None  # Optional, can re-analyze
    log_ids: list[str] | None = None  # Required if no analysis_id
    max_candidates: int = 5


class CandidateModelResponse(BaseModel):
    """Response for a candidate model."""
    provider: str
    model: str
    rank: int
    reasoning: str
    expected_cost_per_request: float
    expected_latency_ms: float
    strengths: list[str]
    concerns: list[str]
    tier: str | None = None
    use_cases: list[str] = []
    quality_score: int | None = None
    speed_score: int | None = None


class ModelSelectionResponse(BaseModel):
    """Response with selected candidate models."""
    selection_id: str
    project_id: str
    candidates: list[CandidateModelResponse]
    selection_reasoning: str
    key_requirements: list[str]
    confidence: float


class StartReplayRequest(BaseModel):
    """Request to start replay evaluation."""
    log_ids: list[str]  # Logs to replay
    candidate_models: list[dict[str, str]]  # [{"provider": "x", "model": "y"}]
    include_current_model: bool = True  # Include current model as baseline


class ReplayStatusResponse(BaseModel):
    """Response with replay status."""
    evaluation_run_id: str
    status: str
    progress: dict[str, int]  # {"completed": x, "total": y}
    models_status: list[dict[str, Any]]


class ModelResultSummary(BaseModel):
    """Summary of results for a single model."""
    provider: str
    model: str
    total_evaluated: int
    avg_quality_score: float
    avg_comparison_score: float
    comparison_verdicts: dict[str, int]
    total_cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float
    quality_distribution: dict[str, int]


class EvaluationResultsResponse(BaseModel):
    """Full evaluation results response."""
    evaluation_run_id: str
    project_id: str
    status: str
    completed_at: str | None
    total_logs_evaluated: int
    model_results: list[ModelResultSummary]
    recommended_model: str | None
    recommended_provider: str | None
    recommendation_reasoning: str
    recommendation_confidence: float


# ========================
# Analysis Endpoint
# ========================

@router.post("/{project_id}/analyze", response_model=AnalysisReportResponse)
async def analyze_logs(
    project_id: uuid.UUID,
    request: AnalyzeLogsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnalysisReportResponse:
    """
    Analyze selected logs for model selection.
    
    This generates statistical insights about the selected logs that
    inform the Model Selector Agent's decision-making.
    """
    # Verify project exists
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
    
    # Parse log IDs
    try:
        log_ids = [uuid.UUID(lid) for lid in request.log_ids]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid log ID format: {e}",
        )
    
    if not log_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No log IDs provided",
        )
    
    # Run analysis
    from app.services.analysis.log_analyzer import LogAnalyzer
    
    analyzer = LogAnalyzer(db)
    try:
        report = await analyzer.analyze_logs(
            log_ids=log_ids,
            project=project,
            success_criteria=project.success_criteria,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    logger.info(
        "Log analysis completed",
        project_id=str(project_id),
        log_count=report.log_count,
    )
    
    return AnalysisReportResponse(**report.to_dict())


# ========================
# Model Selection Endpoint
# ========================

@router.post("/{project_id}/select-models", response_model=ModelSelectionResponse)
async def select_candidate_models(
    project_id: uuid.UUID,
    request: SelectModelsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelSelectionResponse:
    """
    Select candidate models for replay evaluation.
    
    Uses Claude 3.5 Sonnet to analyze project criteria and log statistics,
    then selects the best candidate models to evaluate.
    """
    # Verify project exists
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
    
    # Get log IDs for analysis
    if request.log_ids:
        try:
            log_ids = [uuid.UUID(lid) for lid in request.log_ids]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid log ID format: {e}",
            )
    else:
        # Use project's selected logs or get recent logs
        if project.selected_log_ids:
            log_ids = [uuid.UUID(lid) for lid in project.selected_log_ids]
        else:
            # Get recent logs for this project
            log_result = await db.execute(
                select(LogEntry.id)
                .where(LogEntry.project_id == project_id)
                .order_by(LogEntry.timestamp.desc())
                .limit(100)
            )
            log_ids = [row[0] for row in log_result.all()]
    
    if not log_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No logs available for analysis",
        )
    
    # Run log analysis
    from app.services.analysis.log_analyzer import LogAnalyzer
    from app.services.selector.model_selector_agent import ModelSelectorAgent
    
    analyzer = LogAnalyzer(db)
    try:
        analysis_report = await analyzer.analyze_logs(
            log_ids=log_ids,
            project=project,
            success_criteria=project.success_criteria,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis failed: {e}",
        )
    
    # Run model selection
    selector = ModelSelectorAgent(db)
    try:
        selection_result = await selector.select_models(
            project=project,
            analysis_report=analysis_report,
            max_candidates=request.max_candidates,
        )
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model selection failed: {e}",
        )
    
    logger.info(
        "Model selection completed",
        project_id=str(project_id),
        num_candidates=len(selection_result.candidates),
        confidence=selection_result.confidence,
    )
    
    return ModelSelectionResponse(
        selection_id=selection_result.selection_id,
        project_id=str(project_id),
        candidates=[
            CandidateModelResponse(
                provider=c.provider,
                model=c.model,
                rank=c.rank,
                reasoning=c.reasoning,
                expected_cost_per_request=c.expected_cost_per_request,
                expected_latency_ms=c.expected_latency_ms,
                strengths=c.strengths,
                concerns=c.concerns,
                tier=c.tier,
                use_cases=c.use_cases,
                quality_score=c.quality_score,
                speed_score=c.speed_score,
            )
            for c in selection_result.candidates
        ],
        selection_reasoning=selection_result.selection_reasoning,
        key_requirements=selection_result.key_requirements,
        confidence=selection_result.confidence,
    )


# ========================
# Replay Endpoints
# ========================

@router.post("/{project_id}/start-replay", response_model=ReplayStatusResponse)
async def start_replay_evaluation(
    project_id: uuid.UUID,
    request: StartReplayRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> ReplayStatusResponse:
    """
    Start replay evaluation for candidate models.
    
    This will:
    1. Replay selected logs through each candidate model via Portkey
    2. Track real costs and latency
    3. Run AI judges to compare responses
    4. Generate comparison results
    
    The replay runs asynchronously. Poll the status endpoint for progress.
    """
    # Verify project exists
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
    
    # Parse log IDs
    try:
        log_ids = [uuid.UUID(lid) for lid in request.log_ids]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid log ID format: {e}",
        )
    
    if not log_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No log IDs provided",
        )
    
    if not request.candidate_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No candidate models provided",
        )
    
    # Include current model as baseline if requested
    models = list(request.candidate_models)
    if request.include_current_model and project.current_model and project.current_provider:
        baseline = {"provider": project.current_provider, "model": project.current_model, "is_baseline": True}
        if baseline not in models:
            models.insert(0, baseline)
    
    # Create evaluation run
    evaluation_run = EvaluationRun(
        project_id=project.id,
        trigger_type="manual",
        sample_size=len(log_ids),
        candidate_models=[f"{m['provider']}/{m['model']}" for m in models],
        logs_start_date=datetime.now(timezone.utc) - timedelta(days=30),
        logs_end_date=datetime.now(timezone.utc),
        status=EvaluationStatus.PENDING.value,
    )
    
    db.add(evaluation_run)
    await db.commit()
    await db.refresh(evaluation_run)
    
    # Start replay in background
    background_tasks.add_task(
        run_replay_evaluation_background,
        evaluation_run_id=evaluation_run.id,
        log_ids=log_ids,
        models=models,
    )
    
    logger.info(
        "Replay evaluation started",
        evaluation_run_id=str(evaluation_run.id),
        log_count=len(log_ids),
        model_count=len(models),
    )
    
    return ReplayStatusResponse(
        evaluation_run_id=str(evaluation_run.id),
        status="pending",
        progress={"completed": 0, "total": len(log_ids) * len(models)},
        models_status=[
            {"provider": m["provider"], "model": m["model"], "status": "pending"}
            for m in models
        ],
    )


async def run_replay_evaluation_background(
    evaluation_run_id: uuid.UUID,
    log_ids: list[uuid.UUID],
    models: list[dict[str, str]],
):
    """Background task to run the full replay evaluation pipeline."""
    from app.core.database import get_db_context
    from app.services.replay.replay_engine import ReplayEngine
    from app.services.evaluation.orchestrator import EvaluationOrchestrator
    
    logger.info(f"Background replay evaluation started: {evaluation_run_id}")
    
    try:
        async with get_db_context() as session:
            # Load evaluation run
            result = await session.execute(
                select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id)
            )
            evaluation_run = result.scalar_one_or_none()
            
            if not evaluation_run:
                logger.error(f"Evaluation run {evaluation_run_id} not found")
                return
            
            # Run replay
            replay_engine = ReplayEngine(session)
            replay_results = await replay_engine.execute_replay_run(
                evaluation_run=evaluation_run,
                sample_log_ids=log_ids,
                models=models,
            )
            
            # Run judge evaluation
            orchestrator = EvaluationOrchestrator(session)
            eval_summary = await orchestrator.evaluate_full_run(
                evaluation_run_id=evaluation_run_id,
            )
            
            # Populate aggregated metrics from replay summaries
            # This allows the recommendation engine to use the replay data
            for model_summary in eval_summary.model_summaries:
                if model_summary.total_evaluated > 0:
                    agg_metrics = AggregatedMetrics(
                        project_id=evaluation_run.project_id,
                        model=model_summary.model,
                        provider=model_summary.provider,
                        time_window="7d",  # Use standard window
                        window_start=datetime.now(timezone.utc) - timedelta(days=7),
                        window_end=datetime.now(timezone.utc),
                        total_requests=model_summary.total_evaluated,
                        successful_requests=model_summary.total_evaluated - model_summary.comparison_verdicts.get("worse", 0),
                        failed_requests=0,
                        refusals=0,
                        avg_latency_ms=model_summary.avg_latency_ms,
                        p50_latency_ms=model_summary.avg_latency_ms,
                        p95_latency_ms=model_summary.p95_latency_ms,
                        p99_latency_ms=model_summary.p95_latency_ms * 1.1,
                        latency_std_dev=0.0,
                        total_cost_usd=model_summary.total_cost_usd,
                        avg_cost_per_request=model_summary.total_cost_usd / model_summary.total_evaluated if model_summary.total_evaluated > 0 else 0,
                        total_input_tokens=0,
                        total_output_tokens=0,
                        avg_quality_score=model_summary.avg_overall_score,
                        avg_correctness_score=model_summary.avg_scores_by_judge.get("comparison", model_summary.avg_overall_score),
                        avg_helpfulness_score=model_summary.avg_scores_by_judge.get("helpfulness", model_summary.avg_overall_score),
                        avg_safety_score=1.0,  # Default to safe
                        success_rate=1.0 - (model_summary.comparison_verdicts.get("worse", 0) / model_summary.total_evaluated) if model_summary.total_evaluated > 0 else 1.0,
                        refusal_rate=0.0,
                        error_rate=0.0,
                        quality_variance=model_summary.disagreement_rate,
                        judge_disagreement_rate=model_summary.disagreement_rate,
                        drift_detected=False,
                    )
                    session.add(agg_metrics)
            
            # Update evaluation run status
            evaluation_run.status = EvaluationStatus.COMPLETED.value
            evaluation_run.completed_at = datetime.now(timezone.utc)
            await session.commit()
            
            logger.info(
                f"Background replay evaluation completed: {evaluation_run_id}",
                total_evaluated=eval_summary.total_logs_evaluated,
                recommended_model=eval_summary.recommended_model,
            )
            
    except Exception as e:
        logger.error(f"Background replay evaluation failed: {e}", exc_info=True)
        
        # Mark as failed
        try:
            async with get_db_context() as session:
                result = await session.execute(
                    select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id)
                )
                evaluation_run = result.scalar_one_or_none()
                if evaluation_run:
                    evaluation_run.status = EvaluationStatus.FAILED.value
                    evaluation_run.error_message = str(e)
                    await session.commit()
        except Exception:
            pass


@router.get("/{project_id}/{evaluation_id}/status", response_model=ReplayStatusResponse)
async def get_replay_status(
    project_id: uuid.UUID,
    evaluation_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReplayStatusResponse:
    """
    Get the status of a replay evaluation.
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
    
    # Calculate progress
    total = evaluation_run.replays_total or 0
    completed = evaluation_run.replays_completed or 0
    
    # Get per-model status
    models_status = []
    for replay_run in evaluation_run.replay_runs:
        models_status.append({
            "provider": replay_run.provider,
            "model": replay_run.model,
            "status": replay_run.status,
            "successful": replay_run.successful_completions,
            "failed": replay_run.failed_completions,
            "cost_usd": replay_run.total_cost_usd,
        })
    
    return ReplayStatusResponse(
        evaluation_run_id=str(evaluation_id),
        status=evaluation_run.status,
        progress={"completed": completed, "total": total},
        models_status=models_status,
    )


@router.get("/{project_id}/{evaluation_id}/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results(
    project_id: uuid.UUID,
    evaluation_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EvaluationResultsResponse:
    """
    Get full results of a completed evaluation.
    
    Includes per-model comparison results, quality scores, costs,
    and the final recommendation.
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
    
    if evaluation_run.status != EvaluationStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation not complete. Status: {evaluation_run.status}",
        )
    
    # Get evaluation summary
    from app.services.evaluation.orchestrator import EvaluationOrchestrator
    
    orchestrator = EvaluationOrchestrator(db)
    summary = await orchestrator.evaluate_full_run(evaluation_id)
    
    return EvaluationResultsResponse(
        evaluation_run_id=str(evaluation_id),
        project_id=str(project_id),
        status=evaluation_run.status,
        completed_at=evaluation_run.completed_at.isoformat() if evaluation_run.completed_at else None,
        total_logs_evaluated=summary.total_logs_evaluated,
        model_results=[
            ModelResultSummary(
                provider=ms.provider,
                model=ms.model,
                total_evaluated=ms.total_evaluated,
                avg_quality_score=ms.avg_quality_score,
                avg_comparison_score=ms.avg_comparison_score,
                comparison_verdicts=ms.comparison_verdicts,
                total_cost_usd=ms.total_cost_usd,
                avg_latency_ms=ms.avg_latency_ms,
                p95_latency_ms=ms.p95_latency_ms,
                quality_distribution=ms.quality_distribution,
            )
            for ms in summary.model_summaries
        ],
        recommended_model=summary.recommended_model,
        recommended_provider=summary.recommended_provider,
        recommendation_reasoning=summary.recommendation_reasoning,
        recommendation_confidence=summary.recommendation_confidence,
    )


# ========================
# Legacy Endpoints (kept for compatibility)
# ========================

@router.post("", response_model=EvaluationRunResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(
    evaluation_data: EvaluationRunCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    background_tasks: BackgroundTasks,
) -> EvaluationRun:
    """
    Create and start a new evaluation run (legacy endpoint).
    
    For the new replay-based flow, use:
    1. POST /{project_id}/analyze - Analyze logs
    2. POST /{project_id}/select-models - Select candidates
    3. POST /{project_id}/start-replay - Start evaluation
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
    """List evaluation runs for a project."""
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
    """Get a specific evaluation run with replay details."""
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
    """Cancel a running evaluation."""
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
