"""
Recommendation Engine - Generates model recommendations.

Responsibilities:
- Combine evaluation results with project constraints
- Apply confidence gating
- Generate ranked recommendations with trade-offs
- Support "NO RECOMMENDATION" when confidence is low
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logging import get_logger
from app.models.project import Project
from app.models.evaluation import AggregatedMetrics, EvaluationRun, ReplayRun
from app.models.recommendation import (
    Recommendation,
    CandidateModel,
    RecommendationStatus,
)
from app.services.recommendation.explainability import ExplainabilityEngine

logger = get_logger(__name__)


@dataclass
class TradeOff:
    """Trade-off analysis between models."""
    metric: str
    current_value: float
    recommended_value: float
    change_percent: float
    direction: str  # "better", "worse", "same"
    impact: str  # "positive", "negative", "neutral"


@dataclass
class RecommendationResult:
    """Result of the recommendation process."""
    status: str
    recommended_model: str | None
    recommended_provider: str | None
    confidence: float
    trade_offs: list[TradeOff]
    risk_notes: list[str]
    key_insights: list[str]
    reasoning: str


class RecommendationEngine:
    """
    Generates model recommendations with confidence gating.
    
    Key principles:
    - Advisory only (no auto-switching)
    - Confidence gating (no recommendation if uncertain)
    - Full explainability
    - Clear trade-off quantification
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.explainability = ExplainabilityEngine(session)
        self._confidence_threshold = settings.confidence_threshold

    async def generate_recommendation(
        self,
        project_id: uuid.UUID,
        evaluation_run_id: uuid.UUID | None = None,
    ) -> Recommendation:
        """
        Generate a recommendation for a project.
        
        Args:
            project_id: The project to recommend for
            evaluation_run_id: Optional specific evaluation run
            
        Returns:
            Recommendation with ranked models and explanations
        """
        logger.info(
            "Generating recommendation",
            project_id=str(project_id),
            evaluation_run_id=str(evaluation_run_id) if evaluation_run_id else None,
        )

        # Load project with criteria
        project = await self._load_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Get aggregated metrics for all models
        metrics = await self._load_metrics(project_id)
        
        if not metrics:
            return await self._create_no_recommendation(
                project=project,
                evaluation_run_id=evaluation_run_id,
                reason="No aggregated metrics available",
                status=RecommendationStatus.INSUFFICIENT_DATA,
            )

        # Rank candidates
        ranked_candidates = self._rank_models(metrics, project)
        
        if not ranked_candidates:
            return await self._create_no_recommendation(
                project=project,
                evaluation_run_id=evaluation_run_id,
                reason="No models meet minimum criteria",
                status=RecommendationStatus.NO_RECOMMENDATION,
            )

        # Calculate confidence
        confidence = self._calculate_confidence(ranked_candidates, metrics, project)

        # Check confidence threshold
        if confidence < self._confidence_threshold:
            return await self._create_no_recommendation(
                project=project,
                evaluation_run_id=evaluation_run_id,
                reason=f"Confidence ({confidence:.2f}) below threshold ({self._confidence_threshold})",
                status=RecommendationStatus.LOW_CONFIDENCE,
                candidates=ranked_candidates,
                confidence=confidence,
            )

        # Generate full recommendation
        return await self._create_recommendation(
            project=project,
            evaluation_run_id=evaluation_run_id,
            candidates=ranked_candidates,
            confidence=confidence,
        )

    async def _load_project(self, project_id: uuid.UUID) -> Project | None:
        """Load project with relationships."""
        result = await self.session.execute(
            select(Project)
            .options(
                selectinload(Project.success_criteria),
                selectinload(Project.tolerance_levels),
            )
            .where(Project.id == project_id)
        )
        return result.scalar_one_or_none()

    async def _load_metrics(
        self,
        project_id: uuid.UUID,
        time_window: str = "7d",
    ) -> list[AggregatedMetrics]:
        """Load aggregated metrics for all models."""
        result = await self.session.execute(
            select(AggregatedMetrics)
            .where(
                AggregatedMetrics.project_id == project_id,
                AggregatedMetrics.time_window == time_window,
            )
            .order_by(AggregatedMetrics.created_at.desc())
        )
        return list(result.scalars().all())

    def _rank_models(
        self,
        metrics: list[AggregatedMetrics],
        project: Project,
    ) -> list[tuple[AggregatedMetrics, float, dict]]:
        """
        Rank models by composite score.
        
        Returns list of (metrics, score, breakdown) tuples.
        """
        criteria = project.success_criteria
        tolerances = project.tolerance_levels
        
        # Weight factors based on project preferences
        weights = {
            "quality": 0.35,
            "latency": 0.20,
            "cost": 0.25,
            "reliability": 0.20,
        }
        
        # Adjust weights based on cost sensitivity
        if tolerances:
            if tolerances.cost_sensitivity == "high":
                weights["cost"] = 0.40
                weights["quality"] = 0.25
            elif tolerances.cost_sensitivity == "low":
                weights["cost"] = 0.15
                weights["quality"] = 0.40

        ranked = []
        
        for m in metrics:
            # Skip if insufficient data
            if m.total_requests < 10:
                continue
            
            # Calculate component scores (0-1, higher is better)
            quality_score = (m.avg_quality_score or 0.5)
            latency_score = 1.0 - min((m.avg_latency_ms or 0) / 10000, 1.0)
            cost_score = 1.0 - min((m.avg_cost_per_request or 0) / 0.10, 1.0)
            reliability_score = m.success_rate * (1 - m.refusal_rate)
            
            breakdown = {
                "quality": quality_score,
                "latency": latency_score,
                "cost": cost_score,
                "reliability": reliability_score,
            }
            
            # Compute weighted score
            total_score = sum(breakdown[k] * weights[k] for k in weights)
            
            # Apply penalties
            if criteria:
                # Penalty for exceeding latency
                if m.avg_latency_ms > criteria.max_latency_ms:
                    total_score *= 0.8
                
                # Penalty for high refusal rate
                if m.refusal_rate > criteria.max_refusal_rate:
                    total_score *= 0.7
            
            ranked.append((m, total_score, breakdown))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked

    def _calculate_confidence(
        self,
        ranked: list[tuple[AggregatedMetrics, float, dict]],
        all_metrics: list[AggregatedMetrics],
        project: Project,
    ) -> float:
        """Calculate confidence in the recommendation."""
        if not ranked:
            return 0.0
        
        top_metrics, top_score, _ = ranked[0]
        
        # Factor 1: Sample size
        sample_confidence = min(top_metrics.total_requests / 100, 1.0)
        
        # Factor 2: Score gap (clear winner = higher confidence)
        gap_confidence = 1.0
        if len(ranked) >= 2:
            second_score = ranked[1][1]
            gap = top_score - second_score
            gap_confidence = min(gap / 0.1, 1.0)  # 0.1 gap = full confidence
        
        # Factor 3: Quality variance (lower = higher confidence)
        variance_confidence = 1.0
        if top_metrics.quality_variance:
            variance_confidence = max(0, 1.0 - top_metrics.quality_variance)
        
        # Factor 4: Recent data
        recency_confidence = 1.0
        # Could check how recent the data is
        
        # Combine factors
        confidence = (
            sample_confidence * 0.3 +
            gap_confidence * 0.3 +
            variance_confidence * 0.25 +
            recency_confidence * 0.15
        )
        
        return round(confidence, 2)

    async def _create_no_recommendation(
        self,
        project: Project,
        evaluation_run_id: uuid.UUID | None,
        reason: str,
        status: RecommendationStatus,
        candidates: list | None = None,
        confidence: float = 0.0,
    ) -> Recommendation:
        """Create a no-recommendation result."""
        recommendation = Recommendation(
            project_id=project.id,
            evaluation_run_id=evaluation_run_id,
            status=status.value,
            current_model=project.current_model,
            current_provider=project.current_provider,
            confidence_score=confidence,
            risk_level="low",
            risk_notes=[reason],
            reasoning=f"No recommendation: {reason}",
            key_insights=[reason],
        )
        
        self.session.add(recommendation)
        await self.session.flush()
        
        # Add candidates if available
        if candidates:
            for i, (metrics, score, breakdown) in enumerate(candidates[:5]):
                candidate = CandidateModel(
                    recommendation_id=recommendation.id,
                    model=metrics.model,
                    provider=metrics.provider,
                    is_current=(metrics.model == project.current_model),
                    rank=i + 1,
                    avg_latency_ms=metrics.avg_latency_ms,
                    p95_latency_ms=metrics.p95_latency_ms,
                    avg_cost_per_request=metrics.avg_cost_per_request,
                    success_rate=metrics.success_rate,
                    refusal_rate=metrics.refusal_rate,
                    quality_score=metrics.avg_quality_score or 0,
                    correctness_score=metrics.avg_correctness_score or 0,
                    helpfulness_score=metrics.avg_helpfulness_score or 0,
                    safety_score=metrics.avg_safety_score or 0,
                    overall_score=score,
                )
                self.session.add(candidate)
        
        await self.session.commit()
        await self.session.refresh(recommendation)
        
        return recommendation

    async def _create_recommendation(
        self,
        project: Project,
        evaluation_run_id: uuid.UUID | None,
        candidates: list[tuple[AggregatedMetrics, float, dict]],
        confidence: float,
    ) -> Recommendation:
        """Create a full recommendation."""
        top_metrics, top_score, top_breakdown = candidates[0]
        
        # Find current model metrics
        current_metrics = None
        for m, _, _ in candidates:
            if m.model == project.current_model:
                current_metrics = m
                break

        # Calculate trade-offs vs current
        trade_offs = {}
        if current_metrics:
            trade_offs = self._calculate_trade_offs(current_metrics, top_metrics)

        # Determine risk level
        risk_level = self._assess_risk_level(top_metrics, current_metrics, candidates)
        
        # Generate risk notes
        risk_notes = self._generate_risk_notes(
            top_metrics, current_metrics, candidates, project
        )
        
        # Generate key insights
        key_insights = await self.explainability.generate_insights(
            top_metrics, current_metrics, candidates
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            top_metrics, current_metrics, trade_offs, confidence
        )

        # Create recommendation
        recommendation = Recommendation(
            project_id=project.id,
            evaluation_run_id=evaluation_run_id,
            status=RecommendationStatus.RECOMMENDED.value,
            recommended_model=top_metrics.model,
            recommended_provider=top_metrics.provider,
            current_model=project.current_model,
            current_provider=project.current_provider,
            confidence_score=confidence,
            risk_level=risk_level,
            risk_notes=risk_notes,
            trade_off_summary=trade_offs,
            reasoning=reasoning,
            key_insights=key_insights,
        )
        
        self.session.add(recommendation)
        await self.session.flush()
        
        # Add all candidates
        for i, (metrics, score, breakdown) in enumerate(candidates[:10]):
            candidate = CandidateModel(
                recommendation_id=recommendation.id,
                model=metrics.model,
                provider=metrics.provider,
                is_current=(metrics.model == project.current_model),
                rank=i + 1,
                avg_latency_ms=metrics.avg_latency_ms,
                p95_latency_ms=metrics.p95_latency_ms,
                avg_cost_per_request=metrics.avg_cost_per_request,
                success_rate=metrics.success_rate,
                refusal_rate=metrics.refusal_rate,
                quality_score=metrics.avg_quality_score or 0,
                correctness_score=metrics.avg_correctness_score or 0,
                helpfulness_score=metrics.avg_helpfulness_score or 0,
                safety_score=metrics.avg_safety_score or 0,
                overall_score=score,
                trade_offs=breakdown,
                selection_reason=self._explain_selection(metrics, score, breakdown),
            )
            self.session.add(candidate)
        
        await self.session.commit()
        await self.session.refresh(recommendation)
        
        logger.info(
            "Recommendation created",
            recommendation_id=str(recommendation.id),
            recommended_model=top_metrics.model,
            confidence=confidence,
        )
        
        return recommendation

    def _calculate_trade_offs(
        self,
        current: AggregatedMetrics,
        recommended: AggregatedMetrics,
    ) -> dict[str, Any]:
        """Calculate trade-offs between current and recommended model."""
        def calc_change(curr, rec):
            if curr and curr > 0:
                return ((rec or 0) - curr) / curr * 100
            return 0

        return {
            "cost": {
                "current": current.avg_cost_per_request,
                "recommended": recommended.avg_cost_per_request,
                "change_pct": calc_change(current.avg_cost_per_request, recommended.avg_cost_per_request),
            },
            "latency": {
                "current": current.avg_latency_ms,
                "recommended": recommended.avg_latency_ms,
                "change_pct": calc_change(current.avg_latency_ms, recommended.avg_latency_ms),
            },
            "quality": {
                "current": current.avg_quality_score,
                "recommended": recommended.avg_quality_score,
                "change_pct": calc_change(current.avg_quality_score, recommended.avg_quality_score),
            },
            "reliability": {
                "current": current.success_rate,
                "recommended": recommended.success_rate,
                "change_pct": calc_change(current.success_rate, recommended.success_rate),
            },
        }

    def _assess_risk_level(
        self,
        recommended: AggregatedMetrics,
        current: AggregatedMetrics | None,
        candidates: list,
    ) -> str:
        """Assess the risk level of the recommendation."""
        if not current:
            return "medium"  # No baseline to compare
        
        # Check for significant degradation risk
        if recommended.avg_quality_score and current.avg_quality_score:
            quality_drop = current.avg_quality_score - recommended.avg_quality_score
            if quality_drop > 0.1:
                return "high"
            if quality_drop > 0.05:
                return "medium"
        
        # Check sample size
        if recommended.total_requests < 50:
            return "medium"
        
        return "low"

    def _generate_risk_notes(
        self,
        recommended: AggregatedMetrics,
        current: AggregatedMetrics | None,
        candidates: list,
        project: Project,
    ) -> list[str]:
        """Generate risk notes for the recommendation."""
        notes = []
        
        # Sample size warning
        if recommended.total_requests < 100:
            notes.append(f"Limited sample size ({recommended.total_requests} requests)")
        
        # Quality variance warning
        if recommended.quality_variance and recommended.quality_variance > 0.2:
            notes.append("High quality variance observed - results may be inconsistent")
        
        # Latency increase warning
        if current and recommended.avg_latency_ms > current.avg_latency_ms * 1.2:
            notes.append("Recommended model has higher average latency")
        
        # New model warning
        if recommended.total_requests < 50:
            notes.append("Model has limited production history")
        
        return notes

    def _build_reasoning(
        self,
        recommended: AggregatedMetrics,
        current: AggregatedMetrics | None,
        trade_offs: dict,
        confidence: float,
    ) -> str:
        """Build human-readable reasoning for the recommendation."""
        parts = []
        
        parts.append(f"Recommending {recommended.provider}/{recommended.model} with {confidence*100:.0f}% confidence.")
        
        if current:
            # Highlight improvements
            if trade_offs.get("cost", {}).get("change_pct", 0) < -10:
                parts.append(f"Cost savings of {abs(trade_offs['cost']['change_pct']):.1f}%.")
            
            if trade_offs.get("latency", {}).get("change_pct", 0) < -10:
                parts.append(f"Latency improvement of {abs(trade_offs['latency']['change_pct']):.1f}%.")
            
            if trade_offs.get("quality", {}).get("change_pct", 0) > 5:
                parts.append(f"Quality improvement of {trade_offs['quality']['change_pct']:.1f}%.")
        
        parts.append(f"Based on {recommended.total_requests} requests analyzed.")
        
        return " ".join(parts)

    def _explain_selection(
        self,
        metrics: AggregatedMetrics,
        score: float,
        breakdown: dict,
    ) -> str:
        """Explain why a model was selected at its rank."""
        strengths = []
        
        if breakdown.get("quality", 0) > 0.7:
            strengths.append("high quality")
        if breakdown.get("cost", 0) > 0.7:
            strengths.append("cost-effective")
        if breakdown.get("latency", 0) > 0.7:
            strengths.append("fast response")
        if breakdown.get("reliability", 0) > 0.9:
            strengths.append("highly reliable")
        
        if strengths:
            return f"Selected for: {', '.join(strengths)}"
        return f"Overall score: {score:.3f}"
