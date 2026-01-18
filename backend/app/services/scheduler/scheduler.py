"""
Evaluation Scheduler - Manages periodic and ad-hoc evaluations.

Responsibilities:
- Schedule periodic re-evaluations
- Support ad-hoc evaluation triggers
- Manage shadow evaluation mode
- Prevent evaluation pile-up
"""

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_context
from app.core.logging import get_logger
from app.models.project import Project
from app.models.evaluation import EvaluationRun, EvaluationStatus, AggregatedMetrics

logger = get_logger(__name__)


class SchedulerMode(str, Enum):
    """Evaluation scheduler modes."""
    DISABLED = "disabled"
    SCHEDULED = "scheduled"
    SHADOW = "shadow"  # Shadow evaluation mode


class EvaluationScheduler:
    """
    Manages scheduled and ad-hoc evaluation runs.
    
    Features:
    - Cron-based periodic evaluations
    - Ad-hoc trigger support
    - Shadow mode for testing
    - Prevents concurrent runs for same project
    """

    def __init__(self):
        self._scheduler: AsyncIOScheduler | None = None
        self._mode = SchedulerMode.DISABLED
        self._running_evaluations: set[uuid.UUID] = set()

    def start(self) -> None:
        """Start the scheduler."""
        if not settings.enable_scheduler:
            logger.info("Scheduler is disabled in settings")
            return
        
        self._scheduler = AsyncIOScheduler()
        
        # Add default evaluation job
        self._scheduler.add_job(
            self._run_scheduled_evaluations,
            CronTrigger.from_crontab(settings.evaluation_schedule_cron),
            id="scheduled_evaluations",
            name="Scheduled Model Evaluations",
            replace_existing=True,
        )
        
        # Add log sync job (more frequent)
        self._scheduler.add_job(
            self._run_log_sync,
            IntervalTrigger(hours=6),
            id="log_sync",
            name="Log Sync from Portkey",
            replace_existing=True,
        )
        
        # Add aggregation job
        self._scheduler.add_job(
            self._run_aggregation,
            IntervalTrigger(hours=1),
            id="metric_aggregation",
            name="Metric Aggregation",
            replace_existing=True,
        )
        
        self._scheduler.start()
        self._mode = SchedulerMode.SCHEDULED
        logger.info("Evaluation scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown()
            self._scheduler = None
            self._mode = SchedulerMode.DISABLED
            logger.info("Evaluation scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._scheduler is not None and self._scheduler.running

    def set_mode(self, mode: SchedulerMode) -> None:
        """Set scheduler mode."""
        self._mode = mode
        logger.info(f"Scheduler mode set to: {mode}")

    async def _run_scheduled_evaluations(self) -> None:
        """Run evaluations for all active projects."""
        logger.info("Starting scheduled evaluation run")
        
        async with get_db_context() as session:
            # Get all active projects
            result = await session.execute(
                select(Project).where(Project.is_active == True)
            )
            projects = result.scalars().all()
            
            for project in projects:
                # Skip if already running
                if project.id in self._running_evaluations:
                    logger.info(f"Skipping project {project.id} - evaluation already running")
                    continue
                
                # Check if we need an evaluation
                if await self._should_run_evaluation(session, project):
                    await self._trigger_evaluation(session, project)

    async def _run_log_sync(self) -> None:
        """Sync logs for all active projects."""
        logger.info("Starting scheduled log sync")
        
        async with get_db_context() as session:
            from app.services.ingestion.log_ingestion import LogIngestionService
            
            result = await session.execute(
                select(Project).where(
                    Project.is_active == True,
                    Project.portkey_virtual_key.isnot(None),
                )
            )
            projects = result.scalars().all()
            
            ingestion_service = LogIngestionService(session)
            
            for project in projects:
                try:
                    await ingestion_service.sync_project_logs(project)
                    logger.info(f"Log sync completed for project {project.id}")
                except Exception as e:
                    logger.error(f"Log sync failed for project {project.id}: {e}")

    async def _run_aggregation(self) -> None:
        """Run metric aggregation for all projects."""
        logger.info("Starting scheduled aggregation")
        
        async with get_db_context() as session:
            from app.services.aggregation.state_store import StateStore
            from app.models.log_entry import LogEntry
            
            # Get projects with recent logs
            result = await session.execute(
                select(Project).where(Project.is_active == True)
            )
            projects = result.scalars().all()
            
            state_store = StateStore(session)
            
            for project in projects:
                # Get unique models for this project
                models_result = await session.execute(
                    select(LogEntry.model, LogEntry.provider)
                    .where(LogEntry.project_id == project.id)
                    .distinct()
                )
                
                for model, provider in models_result.all():
                    try:
                        for window in ["7d", "30d"]:
                            await state_store.compute_aggregates(
                                project_id=project.id,
                                model=model,
                                provider=provider,
                                time_window=window,
                            )
                        await session.commit()
                    except Exception as e:
                        logger.error(f"Aggregation failed for {model}: {e}")
                        await session.rollback()

    async def _should_run_evaluation(
        self,
        session: AsyncSession,
        project: Project,
    ) -> bool:
        """Check if an evaluation should run for a project."""
        # Check last evaluation time
        if project.last_evaluation:
            hours_since = (datetime.now(timezone.utc) - project.last_evaluation).total_seconds() / 3600
            if hours_since < 24:  # Minimum 24 hours between evaluations
                return False
        
        # Check if we have enough new logs
        from app.models.log_entry import LogEntry
        
        last_eval_time = project.last_evaluation or (datetime.now(timezone.utc) - timedelta(days=30))
        
        result = await session.execute(
            select(func.count(LogEntry.id)).where(
                LogEntry.project_id == project.id,
                LogEntry.timestamp > last_eval_time,
            )
        )
        new_log_count = result.scalar() or 0
        
        # Need at least 50 new logs
        return new_log_count >= 50

    async def _trigger_evaluation(
        self,
        session: AsyncSession,
        project: Project,
    ) -> None:
        """Trigger an evaluation for a project."""
        logger.info(f"Triggering evaluation for project {project.id}")
        
        self._running_evaluations.add(project.id)
        
        try:
            # Get candidate models
            from app.services.selector.model_selector import ModelSelector
            
            selector = ModelSelector(session)
            selection = await selector.select_models(project)
            
            if not selection.selected_models:
                logger.info(f"No candidate models for project {project.id}")
                return
            
            # Create evaluation run
            evaluation_run = EvaluationRun(
                project_id=project.id,
                trigger_type="scheduled",
                sample_size=settings.default_replay_sample_size,
                candidate_models=[
                    f"{m.provider}/{m.model}" for m in selection.selected_models
                ],
                logs_start_date=datetime.now(timezone.utc) - timedelta(days=30),
                logs_end_date=datetime.now(timezone.utc),
                status=EvaluationStatus.PENDING.value,
            )
            session.add(evaluation_run)
            
            # Update project
            project.last_evaluation = datetime.now(timezone.utc)
            
            await session.commit()
            logger.info(f"Evaluation run created: {evaluation_run.id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger evaluation for {project.id}: {e}")
            await session.rollback()
        finally:
            self._running_evaluations.discard(project.id)

    async def trigger_ad_hoc_evaluation(
        self,
        project_id: uuid.UUID,
        candidate_models: list[str] | None = None,
        sample_size: int | None = None,
    ) -> uuid.UUID:
        """
        Trigger an ad-hoc evaluation with full replay pipeline.
        
        This will:
        1. Select candidate models (if not provided)
        2. Sample logs from the project
        3. Create an evaluation run
        4. Start replay evaluation in the background
        
        Args:
            project_id: Project to evaluate
            candidate_models: Optional specific models to test
            sample_size: Optional custom sample size
            
        Returns:
            ID of the created evaluation run
        """
        import asyncio
        from app.services.selector.model_selector import ModelSelector
        from app.models.log_entry import LogEntry
        from sqlalchemy.orm import selectinload
        from sqlalchemy import func
        
        actual_sample_size = sample_size or settings.default_replay_sample_size
        
        async with get_db_context() as session:
            # Load project with relations
            result = await session.execute(
                select(Project)
                .options(
                    selectinload(Project.success_criteria),
                    selectinload(Project.tolerance_levels),
                )
                .where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()
            
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Run model selection if no candidates provided
            if not candidate_models:
                selector = ModelSelector(session)
                selection = await selector.select_models(project)
                candidate_models = [
                    f"{m.provider}/{m.model}" for m in selection.selected_models
                ]
            
            # ALWAYS ensure at least 3 models for comparison (using OpenAI models)
            # This is critical for meaningful evaluations
            default_openai_models = [
                "openai/@openai/gpt-4o",
                "openai/@openai/gpt-4o-mini", 
                "openai/@openai/o3-mini",
            ]
            
            # Add default models if we don't have enough
            seen_models = set(candidate_models)
            for model in default_openai_models:
                if len(candidate_models) >= 3:
                    break
                if model not in seen_models:
                    candidate_models.append(model)
                    seen_models.add(model)
            
            if not candidate_models:
                raise ValueError("No candidate models selected - all models were excluded by pruning rules")
            
            logger.info(f"Using {len(candidate_models)} candidate models: {candidate_models}")
            
            # Sample logs from the project for replay
            log_result = await session.execute(
                select(LogEntry.id)
                .where(LogEntry.project_id == project_id)
                .where(LogEntry.status == "success")  # Only replay successful logs
                .order_by(func.random())
                .limit(actual_sample_size)
            )
            log_ids = [row[0] for row in log_result.fetchall()]
            
            if not log_ids:
                raise ValueError(f"No logs found for project {project_id}")
            
            # Convert candidate model strings to dict format
            models = []
            for cm in candidate_models:
                if "/" in cm:
                    provider, model = cm.split("/", 1)
                    models.append({"provider": provider, "model": model})
            
            # Create evaluation run
            evaluation_run = EvaluationRun(
                project_id=project.id,
                trigger_type="ad_hoc",
                sample_size=len(log_ids),
                candidate_models=candidate_models,
                logs_start_date=datetime.now(timezone.utc) - timedelta(days=30),
                logs_end_date=datetime.now(timezone.utc),
                status=EvaluationStatus.PENDING.value,
                replays_total=len(log_ids) * len(models),
            )
            session.add(evaluation_run)
            await session.commit()
            
            evaluation_run_id = evaluation_run.id
            logger.info(
                f"Ad-hoc evaluation run created: {evaluation_run_id}",
                log_count=len(log_ids),
                model_count=len(models),
            )
        
        # Start replay evaluation in background task
        asyncio.create_task(
            self._run_replay_background(evaluation_run_id, log_ids, models)
        )
        
        return evaluation_run_id
    
    async def _run_replay_background(
        self,
        evaluation_run_id: uuid.UUID,
        log_ids: list[uuid.UUID],
        models: list[dict[str, str]],
    ):
        """Background task to run the replay evaluation pipeline."""
        from app.services.replay.replay_engine import ReplayEngine
        from app.services.evaluation.orchestrator import EvaluationOrchestrator
        from app.services.recommendation.recommender import RecommendationEngine
        
        logger.info(f"Background replay evaluation started: {evaluation_run_id}")
        
        try:
            async with get_db_context() as session:
                # Get evaluation run
                result = await session.execute(
                    select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id)
                )
                evaluation_run = result.scalar_one_or_none()
                
                if not evaluation_run:
                    logger.error(f"Evaluation run {evaluation_run_id} not found")
                    return
                
                # Run replay engine
                replay_engine = ReplayEngine(session)
                orchestrator = EvaluationOrchestrator(session)
                recommender = RecommendationEngine(session)
                
                try:
                    # Run replays - this updates status to RUNNING internally
                    replay_results = await replay_engine.execute_replay_run(
                        evaluation_run=evaluation_run,
                        sample_log_ids=log_ids,
                        models=models,
                    )
                    
                    logger.info(
                        f"Replays completed: {evaluation_run_id}",
                        replays_completed=evaluation_run.replays_completed,
                    )
                    
                    # Run judge evaluations for all replay runs
                    eval_summary = await orchestrator.evaluate_full_run(evaluation_run_id)
                    
                    logger.info(
                        f"Judgments completed: {evaluation_run_id}",
                        recommended_model=eval_summary.recommended_model,
                    )
                    
                    # Populate aggregated metrics from replay summaries BEFORE recommendation
                    # This allows the recommendation engine to use the replay data
                    for model_summary in eval_summary.model_summaries:
                        if model_summary.total_evaluated > 0:
                            agg_metrics = AggregatedMetrics(
                                project_id=evaluation_run.project_id,
                                model=model_summary.model,
                                provider=model_summary.provider,
                                time_window="7d",
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
                                avg_safety_score=1.0,
                                success_rate=1.0 - (model_summary.comparison_verdicts.get("worse", 0) / model_summary.total_evaluated) if model_summary.total_evaluated > 0 else 1.0,
                                refusal_rate=0.0,
                                error_rate=0.0,
                                quality_variance=model_summary.disagreement_rate,
                                judge_disagreement_rate=model_summary.disagreement_rate,
                                drift_detected=False,
                            )
                            session.add(agg_metrics)
                    
                    # Flush to ensure metrics are available for recommendation
                    await session.flush()
                    
                    # Generate recommendation
                    await recommender.generate_recommendation(
                        project_id=evaluation_run.project_id,
                        evaluation_run_id=evaluation_run_id,
                    )
                    
                    evaluation_run.status = EvaluationStatus.COMPLETED.value
                    evaluation_run.completed_at = datetime.now(timezone.utc)
                    
                    logger.info(
                        f"Evaluation completed: {evaluation_run_id}",
                        replays_completed=evaluation_run.replays_completed,
                    )
                    
                except Exception as e:
                    evaluation_run.status = EvaluationStatus.FAILED.value
                    evaluation_run.error_message = str(e)
                    logger.error(f"Evaluation failed: {evaluation_run_id}: {e}", exc_info=True)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Background replay task failed: {evaluation_run_id}: {e}", exc_info=True)

    def get_scheduled_jobs(self) -> list[dict[str, Any]]:
        """Get list of scheduled jobs."""
        if not self._scheduler:
            return []
        
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return jobs


# Singleton instance
_scheduler: EvaluationScheduler | None = None


def get_scheduler() -> EvaluationScheduler:
    """Get or create the scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = EvaluationScheduler()
    return _scheduler


# Import for type hints
from sqlalchemy import func
