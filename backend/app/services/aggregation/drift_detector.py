"""
Drift Detector - Detects performance changes over time.

Responsibilities:
- Compare metrics across time windows
- Detect statistically significant changes
- Alert on degradation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.evaluation import AggregatedMetrics

logger = get_logger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    metric: str
    direction: str  # "increase" or "decrease"
    baseline_value: float
    current_value: float
    change_percent: float
    severity: str  # "low", "medium", "high"
    message: str
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DriftReport:
    """Complete drift report for a model."""
    model: str
    provider: str
    project_id: uuid.UUID
    baseline_window: str
    current_window: str
    alerts: list[DriftAlert] = field(default_factory=list)
    overall_drift_detected: bool = False
    drift_score: float = 0.0  # 0 = no drift, 1 = severe drift
    recommendations: list[str] = field(default_factory=list)


class DriftDetector:
    """
    Detects drift in model performance metrics.
    
    Compares metrics between time windows to identify:
    - Latency degradation
    - Quality drops
    - Increased error/refusal rates
    - Cost changes
    """

    # Thresholds for drift detection
    THRESHOLDS = {
        "latency_ms": {"warning": 0.15, "critical": 0.30},  # 15% / 30% increase
        "quality_score": {"warning": -0.05, "critical": -0.10},  # 5% / 10% decrease
        "error_rate": {"warning": 0.02, "critical": 0.05},  # 2% / 5% increase
        "refusal_rate": {"warning": 0.03, "critical": 0.07},  # 3% / 7% increase
        "cost_per_request": {"warning": 0.10, "critical": 0.25},  # 10% / 25% increase
    }

    def __init__(self, session: AsyncSession):
        self.session = session

    async def detect_drift(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        baseline_window: str = "30d",
        current_window: str = "7d",
    ) -> DriftReport:
        """
        Detect drift between two time windows.
        
        Args:
            project_id: Project to analyze
            model: Model to check
            provider: Provider to check
            baseline_window: Longer window for baseline
            current_window: Recent window to compare
            
        Returns:
            DriftReport with detected changes
        """
        logger.info(
            f"Detecting drift for {provider}/{model}",
            baseline=baseline_window,
            current=current_window,
        )

        # Get metrics for both windows
        baseline = await self._get_metrics(project_id, model, provider, baseline_window)
        current = await self._get_metrics(project_id, model, provider, current_window)

        alerts = []
        
        if baseline and current:
            # Check each metric
            alerts.extend(self._check_latency_drift(baseline, current))
            alerts.extend(self._check_quality_drift(baseline, current))
            alerts.extend(self._check_reliability_drift(baseline, current))
            alerts.extend(self._check_cost_drift(baseline, current))

        # Calculate overall drift score
        drift_score = self._calculate_drift_score(alerts)
        overall_drift = drift_score > 0.3  # Threshold for flagging

        # Update metrics with drift info
        if current and alerts:
            current.drift_detected = overall_drift
            current.drift_metrics = {
                "alerts": [a.__dict__ for a in alerts],
                "score": drift_score,
            }
            await self.session.flush()

        # Generate recommendations
        recommendations = self._generate_recommendations(alerts)

        return DriftReport(
            model=model,
            provider=provider,
            project_id=project_id,
            baseline_window=baseline_window,
            current_window=current_window,
            alerts=alerts,
            overall_drift_detected=overall_drift,
            drift_score=drift_score,
            recommendations=recommendations,
        )

    async def _get_metrics(
        self,
        project_id: uuid.UUID,
        model: str,
        provider: str,
        time_window: str,
    ) -> AggregatedMetrics | None:
        """Get the most recent metrics for a window."""
        result = await self.session.execute(
            select(AggregatedMetrics)
            .where(
                AggregatedMetrics.project_id == project_id,
                AggregatedMetrics.model == model,
                AggregatedMetrics.provider == provider,
                AggregatedMetrics.time_window == time_window,
            )
            .order_by(AggregatedMetrics.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def _check_latency_drift(
        self,
        baseline: AggregatedMetrics,
        current: AggregatedMetrics,
    ) -> list[DriftAlert]:
        """Check for latency drift."""
        alerts = []
        
        if baseline.avg_latency_ms and baseline.avg_latency_ms > 0:
            change = (current.avg_latency_ms - baseline.avg_latency_ms) / baseline.avg_latency_ms
            
            if change > self.THRESHOLDS["latency_ms"]["critical"]:
                alerts.append(DriftAlert(
                    metric="avg_latency_ms",
                    direction="increase",
                    baseline_value=baseline.avg_latency_ms,
                    current_value=current.avg_latency_ms,
                    change_percent=change * 100,
                    severity="high",
                    message=f"Latency increased by {change*100:.1f}% (critical threshold: {self.THRESHOLDS['latency_ms']['critical']*100}%)",
                ))
            elif change > self.THRESHOLDS["latency_ms"]["warning"]:
                alerts.append(DriftAlert(
                    metric="avg_latency_ms",
                    direction="increase",
                    baseline_value=baseline.avg_latency_ms,
                    current_value=current.avg_latency_ms,
                    change_percent=change * 100,
                    severity="medium",
                    message=f"Latency increased by {change*100:.1f}%",
                ))

        # Also check p95 latency
        if baseline.p95_latency_ms and baseline.p95_latency_ms > 0:
            change = (current.p95_latency_ms - baseline.p95_latency_ms) / baseline.p95_latency_ms
            
            if change > self.THRESHOLDS["latency_ms"]["critical"]:
                alerts.append(DriftAlert(
                    metric="p95_latency_ms",
                    direction="increase",
                    baseline_value=baseline.p95_latency_ms,
                    current_value=current.p95_latency_ms,
                    change_percent=change * 100,
                    severity="high",
                    message=f"P95 latency increased by {change*100:.1f}%",
                ))

        return alerts

    def _check_quality_drift(
        self,
        baseline: AggregatedMetrics,
        current: AggregatedMetrics,
    ) -> list[DriftAlert]:
        """Check for quality drift."""
        alerts = []
        
        quality_metrics = [
            ("avg_quality_score", "Quality"),
            ("avg_correctness_score", "Correctness"),
            ("avg_helpfulness_score", "Helpfulness"),
        ]
        
        for metric, name in quality_metrics:
            baseline_val = getattr(baseline, metric)
            current_val = getattr(current, metric)
            
            if baseline_val and baseline_val > 0:
                change = (current_val or 0) - baseline_val
                
                if change < self.THRESHOLDS["quality_score"]["critical"]:
                    alerts.append(DriftAlert(
                        metric=metric,
                        direction="decrease",
                        baseline_value=baseline_val,
                        current_value=current_val or 0,
                        change_percent=change * 100,
                        severity="high",
                        message=f"{name} score dropped by {abs(change)*100:.1f}%",
                    ))
                elif change < self.THRESHOLDS["quality_score"]["warning"]:
                    alerts.append(DriftAlert(
                        metric=metric,
                        direction="decrease",
                        baseline_value=baseline_val,
                        current_value=current_val or 0,
                        change_percent=change * 100,
                        severity="medium",
                        message=f"{name} score decreased by {abs(change)*100:.1f}%",
                    ))

        return alerts

    def _check_reliability_drift(
        self,
        baseline: AggregatedMetrics,
        current: AggregatedMetrics,
    ) -> list[DriftAlert]:
        """Check for reliability drift (errors, refusals)."""
        alerts = []
        
        # Error rate
        error_change = current.error_rate - baseline.error_rate
        if error_change > self.THRESHOLDS["error_rate"]["critical"]:
            alerts.append(DriftAlert(
                metric="error_rate",
                direction="increase",
                baseline_value=baseline.error_rate,
                current_value=current.error_rate,
                change_percent=error_change * 100,
                severity="high",
                message=f"Error rate increased by {error_change*100:.1f} percentage points",
            ))
        elif error_change > self.THRESHOLDS["error_rate"]["warning"]:
            alerts.append(DriftAlert(
                metric="error_rate",
                direction="increase",
                baseline_value=baseline.error_rate,
                current_value=current.error_rate,
                change_percent=error_change * 100,
                severity="medium",
                message=f"Error rate increased by {error_change*100:.1f} percentage points",
            ))

        # Refusal rate
        refusal_change = current.refusal_rate - baseline.refusal_rate
        if refusal_change > self.THRESHOLDS["refusal_rate"]["critical"]:
            alerts.append(DriftAlert(
                metric="refusal_rate",
                direction="increase",
                baseline_value=baseline.refusal_rate,
                current_value=current.refusal_rate,
                change_percent=refusal_change * 100,
                severity="high",
                message=f"Refusal rate increased by {refusal_change*100:.1f} percentage points",
            ))
        elif refusal_change > self.THRESHOLDS["refusal_rate"]["warning"]:
            alerts.append(DriftAlert(
                metric="refusal_rate",
                direction="increase",
                baseline_value=baseline.refusal_rate,
                current_value=current.refusal_rate,
                change_percent=refusal_change * 100,
                severity="medium",
                message=f"Refusal rate increased by {refusal_change*100:.1f} percentage points",
            ))

        return alerts

    def _check_cost_drift(
        self,
        baseline: AggregatedMetrics,
        current: AggregatedMetrics,
    ) -> list[DriftAlert]:
        """Check for cost drift."""
        alerts = []
        
        if baseline.avg_cost_per_request and baseline.avg_cost_per_request > 0:
            change = (current.avg_cost_per_request - baseline.avg_cost_per_request) / baseline.avg_cost_per_request
            
            if change > self.THRESHOLDS["cost_per_request"]["critical"]:
                alerts.append(DriftAlert(
                    metric="avg_cost_per_request",
                    direction="increase",
                    baseline_value=baseline.avg_cost_per_request,
                    current_value=current.avg_cost_per_request,
                    change_percent=change * 100,
                    severity="high",
                    message=f"Cost per request increased by {change*100:.1f}%",
                ))
            elif change > self.THRESHOLDS["cost_per_request"]["warning"]:
                alerts.append(DriftAlert(
                    metric="avg_cost_per_request",
                    direction="increase",
                    baseline_value=baseline.avg_cost_per_request,
                    current_value=current.avg_cost_per_request,
                    change_percent=change * 100,
                    severity="medium",
                    message=f"Cost per request increased by {change*100:.1f}%",
                ))

        return alerts

    def _calculate_drift_score(self, alerts: list[DriftAlert]) -> float:
        """Calculate overall drift score from alerts."""
        if not alerts:
            return 0.0
        
        severity_weights = {"low": 0.2, "medium": 0.5, "high": 1.0}
        total_weight = sum(severity_weights.get(a.severity, 0.5) for a in alerts)
        
        # Normalize to 0-1 scale (assume max 5 high-severity alerts = 1.0)
        return min(total_weight / 5.0, 1.0)

    def _generate_recommendations(self, alerts: list[DriftAlert]) -> list[str]:
        """Generate recommendations based on detected drift."""
        recommendations = []
        
        high_severity = [a for a in alerts if a.severity == "high"]
        
        if any(a.metric.endswith("latency_ms") for a in high_severity):
            recommendations.append(
                "Consider investigating model provider issues or switching to a faster model"
            )
        
        if any("quality" in a.metric or "correctness" in a.metric for a in high_severity):
            recommendations.append(
                "Quality degradation detected - review recent model updates or prompt changes"
            )
        
        if any(a.metric == "error_rate" for a in high_severity):
            recommendations.append(
                "High error rate - check API quotas, network issues, or model availability"
            )
        
        if any(a.metric == "refusal_rate" for a in high_severity):
            recommendations.append(
                "Increased refusals - review prompts for policy compliance or consider model alternatives"
            )
        
        if any(a.metric == "avg_cost_per_request" for a in high_severity):
            recommendations.append(
                "Cost increase detected - review prompt lengths or consider more cost-effective models"
            )
        
        if not recommendations and alerts:
            recommendations.append(
                "Minor drift detected - continue monitoring but no immediate action required"
            )
        
        return recommendations
