"""
State Store - Manages aggregated metrics over time windows.

Responsibilities:
- Compute and store time-windowed aggregates (7d, 30d)
- Track metric history for trend analysis
- Support versioning of metrics
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.evaluation import AggregatedMetrics, TimeWindow, JudgeResult, ReplayResult

logger = get_logger(__name__)


class StateStore:
    """
    Manages aggregated state and metrics for projects.
    
    Key features:
    - Time-windowed aggregation
    - Metric versioning
    - Efficient incremental updates
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def compute_aggregates(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        time_window: str = "7d",
    ) -> AggregatedMetrics:
        """
        Compute aggregated metrics for a project-model combination.
        
        Args:
            project_id: The project to aggregate for
            model: Model identifier
            provider: Provider identifier
            time_window: Time window (7d, 30d, 90d)
            
        Returns:
            AggregatedMetrics with computed values
        """
        # Parse time window
        window_days = {"7d": 7, "30d": 30, "90d": 90}.get(time_window, 7)
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=window_days)

        logger.info(
            f"Computing aggregates for {provider}/{model}",
            project_id=str(project_id),
            time_window=time_window,
        )

        # Query log entries
        log_stats = await self._compute_log_stats(
            project_id, model, provider, window_start, window_end
        )

        # Query evaluation metrics
        eval_stats = await self._compute_eval_stats(
            project_id, model, provider, window_start, window_end
        )

        # Check for existing aggregate
        existing = await self.session.execute(
            select(AggregatedMetrics).where(
                AggregatedMetrics.project_id == project_id,
                AggregatedMetrics.model == model,
                AggregatedMetrics.provider == provider,
                AggregatedMetrics.time_window == time_window,
            )
        )
        metrics = existing.scalar_one_or_none()

        if metrics:
            # Update existing
            self._update_metrics(metrics, log_stats, eval_stats, window_start, window_end)
        else:
            # Create new
            metrics = AggregatedMetrics(
                project_id=project_id,
                model=model,
                provider=provider,
                time_window=time_window,
                window_start=window_start,
                window_end=window_end,
            )
            self._update_metrics(metrics, log_stats, eval_stats, window_start, window_end)
            self.session.add(metrics)

        await self.session.flush()
        return metrics

    async def _compute_log_stats(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, Any]:
        """Compute statistics from log entries."""
        # Basic aggregates
        result = await self.session.execute(
            select(
                func.count(LogEntry.id).label("total"),
                func.sum(func.cast(LogEntry.refusal, Integer)).label("refusals"),
                func.count(LogEntry.id).filter(LogEntry.status == "error").label("errors"),
                func.count(LogEntry.id).filter(LogEntry.status == "success").label("successes"),
                func.avg(LogEntry.latency_ms).label("avg_latency"),
                func.stddev(LogEntry.latency_ms).label("latency_std"),
                func.sum(LogEntry.cost_usd).label("total_cost"),
                func.sum(LogEntry.input_tokens).label("total_input"),
                func.sum(LogEntry.output_tokens).label("total_output"),
            )
            .where(
                LogEntry.project_id == project_id,
                LogEntry.model == model,
                LogEntry.provider == provider,
                LogEntry.timestamp >= window_start,
                LogEntry.timestamp <= window_end,
            )
        )
        row = result.one()

        # Get latency percentiles
        latency_result = await self.session.execute(
            select(LogEntry.latency_ms)
            .where(
                LogEntry.project_id == project_id,
                LogEntry.model == model,
                LogEntry.provider == provider,
                LogEntry.timestamp >= window_start,
                LogEntry.timestamp <= window_end,
                LogEntry.latency_ms.isnot(None),
            )
            .order_by(LogEntry.latency_ms)
        )
        latencies = [r[0] for r in latency_result.all()]

        p50 = p95 = p99 = 0.0
        if latencies:
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))

        total = row.total or 0
        return {
            "total_requests": total,
            "successful_requests": row.successes or 0,
            "failed_requests": row.errors or 0,
            "refusals": row.refusals or 0,
            "avg_latency_ms": float(row.avg_latency or 0),
            "latency_std_dev": float(row.latency_std or 0),
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "total_cost_usd": float(row.total_cost or 0),
            "total_input_tokens": int(row.total_input or 0),
            "total_output_tokens": int(row.total_output or 0),
            "success_rate": (row.successes or 0) / total if total > 0 else 0,
            "refusal_rate": (row.refusals or 0) / total if total > 0 else 0,
            "error_rate": (row.errors or 0) / total if total > 0 else 0,
            "avg_cost_per_request": (row.total_cost or 0) / total if total > 0 else 0,
        }

    async def _compute_eval_stats(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, Any]:
        """Compute statistics from evaluation results."""
        from app.models.evaluation import ReplayRun
        
        # Get judge results for this model
        result = await self.session.execute(
            select(
                JudgeResult.judge_type,
                func.avg(JudgeResult.score).label("avg_score"),
                func.stddev(JudgeResult.score).label("score_std"),
            )
            .join(ReplayResult, JudgeResult.replay_result_id == ReplayResult.id)
            .join(ReplayRun, ReplayResult.replay_run_id == ReplayRun.id)
            .where(
                ReplayRun.model == model,
                ReplayRun.provider == provider,
                ReplayRun.created_at >= window_start,
                ReplayRun.created_at <= window_end,
            )
            .group_by(JudgeResult.judge_type)
        )
        
        judge_scores = {}
        quality_variance = None
        
        for row in result.all():
            judge_scores[row.judge_type] = float(row.avg_score or 0)
            if row.judge_type == "quality" and row.score_std:
                quality_variance = float(row.score_std)

        # Check for judge disagreements
        disagreement_result = await self.session.execute(
            select(func.count(ReplayResult.id))
            .join(ReplayRun, ReplayResult.replay_run_id == ReplayRun.id)
            .where(
                ReplayRun.model == model,
                ReplayRun.provider == provider,
                ReplayRun.created_at >= window_start,
                ReplayRun.created_at <= window_end,
            )
        )
        total_evals = disagreement_result.scalar() or 0

        return {
            "avg_quality_score": judge_scores.get("quality"),
            "avg_correctness_score": judge_scores.get("correctness"),
            "avg_helpfulness_score": judge_scores.get("helpfulness"),
            "avg_safety_score": judge_scores.get("safety"),
            "quality_variance": quality_variance,
            "judge_disagreement_rate": 0.0,  # Would need more complex query
        }

    def _update_metrics(
        self,
        metrics: AggregatedMetrics,
        log_stats: dict[str, Any],
        eval_stats: dict[str, Any],
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        """Update metrics object with computed stats."""
        metrics.window_start = window_start
        metrics.window_end = window_end
        
        # Log stats
        metrics.total_requests = log_stats["total_requests"]
        metrics.successful_requests = log_stats["successful_requests"]
        metrics.failed_requests = log_stats["failed_requests"]
        metrics.refusals = log_stats["refusals"]
        metrics.avg_latency_ms = log_stats["avg_latency_ms"]
        metrics.latency_std_dev = log_stats["latency_std_dev"]
        metrics.p50_latency_ms = log_stats["p50_latency_ms"]
        metrics.p95_latency_ms = log_stats["p95_latency_ms"]
        metrics.p99_latency_ms = log_stats["p99_latency_ms"]
        metrics.total_cost_usd = log_stats["total_cost_usd"]
        metrics.total_input_tokens = log_stats["total_input_tokens"]
        metrics.total_output_tokens = log_stats["total_output_tokens"]
        metrics.success_rate = log_stats["success_rate"]
        metrics.refusal_rate = log_stats["refusal_rate"]
        metrics.error_rate = log_stats["error_rate"]
        metrics.avg_cost_per_request = log_stats["avg_cost_per_request"]
        
        # Eval stats
        metrics.avg_quality_score = eval_stats["avg_quality_score"]
        metrics.avg_correctness_score = eval_stats["avg_correctness_score"]
        metrics.avg_helpfulness_score = eval_stats["avg_helpfulness_score"]
        metrics.avg_safety_score = eval_stats["avg_safety_score"]
        metrics.quality_variance = eval_stats["quality_variance"]
        metrics.judge_disagreement_rate = eval_stats["judge_disagreement_rate"]

    async def get_latest_aggregates(
        self,
        project_id: uuid.UUID,
        time_window: str = "7d",
    ) -> list[AggregatedMetrics]:
        """Get latest aggregated metrics for all models in a project."""
        result = await self.session.execute(
            select(AggregatedMetrics)
            .where(
                AggregatedMetrics.project_id == project_id,
                AggregatedMetrics.time_window == time_window,
            )
            .order_by(AggregatedMetrics.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_metric_history(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        metric: str,
        lookback_days: int = 90,
    ) -> list[tuple[datetime, float]]:
        """Get historical values for a specific metric."""
        # This would query historical snapshots
        # For now, return empty - would need a separate history table
        return []


# Import for type annotation
from sqlalchemy import Integer
