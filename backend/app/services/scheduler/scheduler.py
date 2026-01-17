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
from app.models.evaluation import EvaluationRun, EvaluationStatus

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
        Trigger an ad-hoc evaluation.
        
        Args:
            project_id: Project to evaluate
            candidate_models: Optional specific models to test
            sample_size: Optional custom sample size
            
        Returns:
            ID of the created evaluation run
        """
        async with get_db_context() as session:
            # Load project
            result = await session.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()
            
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Get candidates if not specified
            if not candidate_models:
                from app.services.selector.model_selector import ModelSelector
                selector = ModelSelector(session)
                selection = await selector.select_models(project)
                candidate_models = [
                    f"{m.provider}/{m.model}" for m in selection.selected_models
                ]
            
            # Create evaluation run
            evaluation_run = EvaluationRun(
                project_id=project.id,
                trigger_type="ad_hoc",
                sample_size=sample_size or settings.default_replay_sample_size,
                candidate_models=candidate_models,
                logs_start_date=datetime.now(timezone.utc) - timedelta(days=30),
                logs_end_date=datetime.now(timezone.utc),
                status=EvaluationStatus.PENDING.value,
            )
            session.add(evaluation_run)
            await session.commit()
            
            logger.info(f"Ad-hoc evaluation triggered: {evaluation_run.id}")
            return evaluation_run.id

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
