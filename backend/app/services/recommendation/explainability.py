"""
Explainability Engine - Generates explanations for recommendations.

Responsibilities:
- Generate key insights
- Find example cases (improvements, regressions)
- Track judge disagreements
- Make reasoning auditable
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.evaluation import (
    AggregatedMetrics,
    ReplayRun,
    ReplayResult,
    JudgeResult,
)

logger = get_logger(__name__)


@dataclass
class ExampleCase:
    """An example case for explainability."""
    prompt_hash: str
    prompt_preview: str
    current_score: float
    recommended_score: float
    difference: float
    judge_type: str
    rationale: str


@dataclass
class JudgeDisagreement:
    """A case where judges disagreed."""
    prompt_hash: str
    judge_scores: dict[str, float]
    max_disagreement: float
    conflicting_judges: list[str]


class ExplainabilityEngine:
    """
    Generates explanations and examples for recommendations.
    
    Provides:
    - Key insights in plain language
    - Example improvements and regressions
    - Judge disagreement cases
    - Auditable reasoning chain
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def generate_insights(
        self,
        recommended: AggregatedMetrics,
        current: AggregatedMetrics | None,
        all_candidates: list[tuple[AggregatedMetrics, float, dict]],
    ) -> list[str]:
        """Generate key insights about the recommendation."""
        insights = []
        
        # Insight 1: Why recommended model is best
        if recommended.avg_quality_score and recommended.avg_quality_score > 0.8:
            insights.append(
                f"{recommended.model} shows consistently high quality scores "
                f"(avg: {recommended.avg_quality_score:.2f})"
            )
        
        # Insight 2: Cost comparison
        if current and recommended.avg_cost_per_request < current.avg_cost_per_request * 0.8:
            savings = (1 - recommended.avg_cost_per_request / current.avg_cost_per_request) * 100
            insights.append(
                f"Switching to {recommended.model} could reduce costs by {savings:.1f}%"
            )
        
        # Insight 3: Latency comparison
        if current and recommended.avg_latency_ms < current.avg_latency_ms * 0.9:
            improvement = (1 - recommended.avg_latency_ms / current.avg_latency_ms) * 100
            insights.append(
                f"{recommended.model} is {improvement:.1f}% faster on average"
            )
        
        # Insight 4: Reliability
        if recommended.success_rate > 0.99:
            insights.append(
                f"{recommended.model} has excellent reliability ({recommended.success_rate*100:.1f}% success rate)"
            )
        
        # Insight 5: Trade-off insight
        if current:
            if (recommended.avg_cost_per_request < current.avg_cost_per_request and
                recommended.avg_quality_score and current.avg_quality_score and
                recommended.avg_quality_score < current.avg_quality_score):
                quality_diff = (current.avg_quality_score - recommended.avg_quality_score) * 100
                cost_diff = (1 - recommended.avg_cost_per_request / current.avg_cost_per_request) * 100
                insights.append(
                    f"Trade-off: {cost_diff:.1f}% cost savings with {quality_diff:.1f}% quality reduction"
                )
        
        # Insight 6: Sample size confidence
        insights.append(
            f"Analysis based on {recommended.total_requests} production requests"
        )
        
        return insights

    async def find_example_improvements(
        self,
        project_id: uuid.UUID,
        recommended_model: str,
        current_model: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find examples where recommended model outperformed current."""
        # This would query replay results to find specific examples
        # For now, return placeholder structure
        return []

    async def find_example_regressions(
        self,
        project_id: uuid.UUID,
        recommended_model: str,
        current_model: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find examples where recommended model underperformed current."""
        return []

    async def find_judge_disagreements(
        self,
        project_id: uuid.UUID,
        model: str,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find cases where judges significantly disagreed."""
        # Query for replay results with high score variance
        result = await self.session.execute(
            select(
                ReplayResult.id,
                ReplayResult.prompt_hash,
                func.stddev(JudgeResult.score).label("score_std"),
            )
            .join(JudgeResult, JudgeResult.replay_result_id == ReplayResult.id)
            .join(ReplayRun, ReplayResult.replay_run_id == ReplayRun.id)
            .where(ReplayRun.model == model)
            .group_by(ReplayResult.id, ReplayResult.prompt_hash)
            .having(func.stddev(JudgeResult.score) > threshold)
            .order_by(func.stddev(JudgeResult.score).desc())
            .limit(limit)
        )
        
        disagreements = []
        for row in result.all():
            # Get detailed judge scores
            scores_result = await self.session.execute(
                select(JudgeResult.judge_type, JudgeResult.score)
                .where(JudgeResult.replay_result_id == row.id)
            )
            scores = {r.judge_type: r.score for r in scores_result.all()}
            
            disagreements.append({
                "prompt_hash": row.prompt_hash,
                "judge_scores": scores,
                "score_std": float(row.score_std),
            })
        
        return disagreements

    async def generate_trade_off_analysis(
        self,
        recommended: AggregatedMetrics,
        current: AggregatedMetrics,
    ) -> dict[str, Any]:
        """Generate detailed trade-off analysis."""
        analysis = {
            "summary": [],
            "details": {},
        }
        
        # Cost analysis
        cost_change = (recommended.avg_cost_per_request - current.avg_cost_per_request) / current.avg_cost_per_request * 100
        analysis["details"]["cost"] = {
            "current": current.avg_cost_per_request,
            "recommended": recommended.avg_cost_per_request,
            "change_percent": cost_change,
            "monthly_impact": self._estimate_monthly_impact(
                current.avg_cost_per_request,
                recommended.avg_cost_per_request,
                current.total_requests,
            ),
        }
        
        if cost_change < -10:
            analysis["summary"].append(f"Cost reduction of {abs(cost_change):.1f}%")
        elif cost_change > 10:
            analysis["summary"].append(f"Cost increase of {cost_change:.1f}%")
        
        # Latency analysis
        latency_change = (recommended.avg_latency_ms - current.avg_latency_ms) / current.avg_latency_ms * 100
        analysis["details"]["latency"] = {
            "current_avg": current.avg_latency_ms,
            "current_p95": current.p95_latency_ms,
            "recommended_avg": recommended.avg_latency_ms,
            "recommended_p95": recommended.p95_latency_ms,
            "change_percent": latency_change,
        }
        
        if latency_change < -10:
            analysis["summary"].append(f"Latency improvement of {abs(latency_change):.1f}%")
        elif latency_change > 10:
            analysis["summary"].append(f"Latency increase of {latency_change:.1f}%")
        
        # Quality analysis
        if current.avg_quality_score and recommended.avg_quality_score:
            quality_change = (recommended.avg_quality_score - current.avg_quality_score) * 100
            analysis["details"]["quality"] = {
                "current": current.avg_quality_score,
                "recommended": recommended.avg_quality_score,
                "change_percent": quality_change,
            }
            
            if quality_change > 5:
                analysis["summary"].append(f"Quality improvement of {quality_change:.1f}%")
            elif quality_change < -5:
                analysis["summary"].append(f"Quality reduction of {abs(quality_change):.1f}%")
        
        return analysis

    def _estimate_monthly_impact(
        self,
        current_cost: float,
        recommended_cost: float,
        weekly_requests: int,
    ) -> dict[str, float]:
        """Estimate monthly cost impact."""
        monthly_requests = weekly_requests * 4  # Rough estimate
        current_monthly = current_cost * monthly_requests
        recommended_monthly = recommended_cost * monthly_requests
        
        return {
            "current_monthly_usd": current_monthly,
            "recommended_monthly_usd": recommended_monthly,
            "savings_usd": current_monthly - recommended_monthly,
        }

    async def build_audit_trail(
        self,
        recommendation_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Build an audit trail for a recommendation."""
        # This would gather all the data used to make the recommendation
        # for compliance and debugging purposes
        return {
            "recommendation_id": str(recommendation_id),
            "data_sources": [
                "aggregated_metrics",
                "evaluation_runs",
                "judge_results",
            ],
            "algorithms_used": [
                "weighted_scoring",
                "confidence_calculation",
                "trade_off_analysis",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


from datetime import datetime, timezone
