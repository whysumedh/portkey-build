"""
Model Selector Agent.

A lightweight AI agent that:
- Consumes project intent + constraints
- Consumes statistical artifacts from analytics engine
- Applies deterministic pruning rules first
- Selects TOP N candidate models for replay
- Outputs structured JSON only
- Explains why models were selected or excluded
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.project import Project
from app.services.analytics.engine import AnalyticsEngine
from app.services.selector.pruning_rules import (
    PruningRulesEngine,
    ModelCandidate,
    create_candidate_from_metrics,
)
from app.services.ingestion.portkey_client import PortkeyClient, get_portkey_client

logger = get_logger(__name__)


@dataclass
class ModelSelection:
    """Result of model selection."""
    selected_models: list[ModelCandidate]
    excluded_models: list[ModelCandidate]
    current_model: ModelCandidate | None
    selection_reasoning: str
    pruning_rules_applied: list[str]
    statistical_artifacts_used: list[str]
    confidence: float = 0.0


@dataclass
class CandidateScore:
    """Scored candidate for ranking."""
    candidate: ModelCandidate
    overall_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    ranking_reason: str = ""


class ModelSelector:
    """
    Model selection agent that chooses candidate models for evaluation.
    
    The selector:
    - NEVER sees raw logs (only statistical artifacts)
    - NEVER executes code
    - Outputs structured JSON only
    - Explains why models were selected or excluded
    """

    def __init__(
        self,
        session: AsyncSession,
        portkey_client: PortkeyClient | None = None,
    ):
        self.session = session
        self.portkey = portkey_client or get_portkey_client()

    async def select_models(
        self,
        project: Project,
        top_n: int = 3,
        include_current: bool = True,
    ) -> ModelSelection:
        """
        Select candidate models for a project.
        
        Process:
        1. Gather candidate models from project history + model catalog
        2. Compute metrics for each candidate
        3. Apply deterministic pruning rules
        4. Rank remaining candidates
        5. Select top N
        
        Args:
            project: The project to select models for
            top_n: Number of candidates to select
            include_current: Whether to always include current model
            
        Returns:
            ModelSelection with selected and excluded models
        """
        logger.info(
            "Starting model selection",
            project_id=str(project.id),
            top_n=top_n,
        )

        # Step 1: Gather candidate models
        candidates = await self._gather_candidates(project)
        
        if not candidates:
            logger.warning("No candidate models found")
            return ModelSelection(
                selected_models=[],
                excluded_models=[],
                current_model=None,
                selection_reasoning="No candidate models found in logs or catalog",
                pruning_rules_applied=[],
                statistical_artifacts_used=[],
                confidence=0.0,
            )

        # Step 2: Compute metrics for each candidate
        candidates = await self._compute_candidate_metrics(project.id, candidates)

        # Identify current model
        current_model = None
        if project.current_model:
            for c in candidates:
                if c.model == project.current_model:
                    current_model = c
                    break

        # Step 3: Apply pruning rules
        pruning_engine = PruningRulesEngine(project)
        pruning_result = pruning_engine.apply_rules(candidates)

        # Step 4: Rank remaining candidates
        scored_candidates = self._rank_candidates(
            pruning_result.passed,
            project,
        )

        # Step 5: Select top N
        selected = []
        for scored in scored_candidates[:top_n]:
            selected.append(scored.candidate)

        # Always include current model if requested
        if include_current and current_model and not current_model.excluded:
            if current_model not in selected:
                selected.append(current_model)

        # Build reasoning
        reasoning = self._build_selection_reasoning(
            selected,
            pruning_result.excluded,
            scored_candidates,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            selected,
            pruning_result.passed,
            pruning_result.excluded,
        )

        logger.info(
            "Model selection complete",
            selected_count=len(selected),
            excluded_count=len(pruning_result.excluded),
            confidence=confidence,
        )

        return ModelSelection(
            selected_models=selected,
            excluded_models=pruning_result.excluded,
            current_model=current_model,
            selection_reasoning=reasoning,
            pruning_rules_applied=pruning_result.rules_applied,
            statistical_artifacts_used=[
                "latency_percentiles",
                "cost_per_request",
                "refusal_rate",
                "error_rate",
                "sample_count",
            ],
            confidence=confidence,
        )

    async def _gather_candidates(self, project: Project) -> list[ModelCandidate]:
        """Gather candidate models from logs and catalog."""
        candidates = []
        seen_models = set()

        # Get models from project logs
        result = await self.session.execute(
            select(LogEntry.model, LogEntry.provider)
            .where(LogEntry.project_id == project.id)
            .distinct()
        )
        
        for row in result.all():
            model_key = f"{row.provider}:{row.model}"
            if model_key not in seen_models:
                candidates.append(ModelCandidate(
                    model=row.model,
                    provider=row.provider,
                ))
                seen_models.add(model_key)

        # Try to get additional models from Portkey catalog
        try:
            catalog = await self.portkey.get_model_catalog()
            for model_info in catalog.get("data", []):
                model_key = f"{model_info.get('provider')}:{model_info.get('model')}"
                if model_key not in seen_models:
                    candidates.append(ModelCandidate(
                        model=model_info.get("model", ""),
                        provider=model_info.get("provider", ""),
                    ))
                    seen_models.add(model_key)
        except Exception as e:
            logger.warning(f"Could not fetch model catalog: {e}")

        return candidates

    async def _compute_candidate_metrics(
        self,
        project_id: uuid.UUID,
        candidates: list[ModelCandidate],
    ) -> list[ModelCandidate]:
        """Compute metrics for each candidate from log data."""
        
        for candidate in candidates:
            # Query metrics for this model
            result = await self.session.execute(
                select(
                    func.count(LogEntry.id).label("count"),
                    func.avg(LogEntry.latency_ms).label("avg_latency"),
                    func.avg(LogEntry.cost_usd).label("avg_cost"),
                )
                .where(
                    LogEntry.project_id == project_id,
                    LogEntry.model == candidate.model,
                    LogEntry.provider == candidate.provider,
                )
            )
            row = result.one()
            
            if row.count > 0:
                candidate.sample_size = row.count
                candidate.avg_latency_ms = float(row.avg_latency or 0)
                candidate.avg_cost_per_request = float(row.avg_cost or 0)

                # Get percentiles
                latency_result = await self.session.execute(
                    select(LogEntry.latency_ms)
                    .where(
                        LogEntry.project_id == project_id,
                        LogEntry.model == candidate.model,
                        LogEntry.provider == candidate.provider,
                    )
                    .order_by(LogEntry.latency_ms)
                )
                latencies = [r[0] for r in latency_result.all() if r[0] is not None]
                
                if latencies:
                    import numpy as np
                    candidate.p95_latency_ms = float(np.percentile(latencies, 95))

                # Get refusal and error rates
                status_result = await self.session.execute(
                    select(
                        func.sum(func.cast(LogEntry.refusal, Integer)).label("refusals"),
                        func.count(LogEntry.id).filter(LogEntry.status == "error").label("errors"),
                    )
                    .where(
                        LogEntry.project_id == project_id,
                        LogEntry.model == candidate.model,
                        LogEntry.provider == candidate.provider,
                    )
                )
                status_row = status_result.one()
                
                total = candidate.sample_size
                candidate.refusal_rate = (status_row.refusals or 0) / total if total > 0 else 0
                candidate.error_rate = (status_row.errors or 0) / total if total > 0 else 0

        return candidates

    def _rank_candidates(
        self,
        candidates: list[ModelCandidate],
        project: Project,
    ) -> list[CandidateScore]:
        """Rank candidates by a composite score."""
        scored = []
        
        # Get criteria weights based on project settings
        criteria = project.success_criteria
        tolerances = project.tolerance_levels
        
        # Default weights
        weights = {
            "latency": 0.25,
            "cost": 0.25,
            "quality": 0.30,
            "reliability": 0.20,
        }
        
        # Adjust weights based on cost sensitivity
        if tolerances and tolerances.cost_sensitivity == "high":
            weights["cost"] = 0.40
            weights["quality"] = 0.20
        elif tolerances and tolerances.cost_sensitivity == "low":
            weights["cost"] = 0.10
            weights["quality"] = 0.40

        for candidate in candidates:
            # Normalize metrics to 0-1 scale (higher is better)
            latency_score = 1.0 - min(candidate.avg_latency_ms / 10000, 1.0)  # Assume 10s is worst
            cost_score = 1.0 - min(candidate.avg_cost_per_request / 0.10, 1.0)  # Assume $0.10 is worst
            quality_score = candidate.quality_score if candidate.quality_score > 0 else 0.5  # Default to 0.5
            reliability_score = 1.0 - candidate.refusal_rate - candidate.error_rate
            reliability_score = max(0, reliability_score)

            breakdown = {
                "latency": latency_score,
                "cost": cost_score,
                "quality": quality_score,
                "reliability": reliability_score,
            }

            overall = sum(breakdown[k] * weights[k] for k in weights)

            scored.append(CandidateScore(
                candidate=candidate,
                overall_score=overall,
                score_breakdown=breakdown,
                ranking_reason=self._explain_ranking(breakdown, weights),
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.overall_score, reverse=True)
        
        return scored

    def _explain_ranking(
        self,
        breakdown: dict[str, float],
        weights: dict[str, float],
    ) -> str:
        """Generate human-readable ranking explanation."""
        parts = []
        for metric, score in sorted(breakdown.items(), key=lambda x: x[1] * weights.get(x[0], 0), reverse=True):
            weight = weights.get(metric, 0)
            contribution = score * weight
            parts.append(f"{metric}={score:.2f}×{weight:.2f}={contribution:.2f}")
        return " + ".join(parts)

    def _build_selection_reasoning(
        self,
        selected: list[ModelCandidate],
        excluded: list[ModelCandidate],
        scored: list[CandidateScore],
    ) -> str:
        """Build human-readable selection reasoning."""
        lines = []
        
        lines.append(f"Selected {len(selected)} candidate models for evaluation.")
        
        if selected:
            lines.append("\nSelected models:")
            for i, candidate in enumerate(selected, 1):
                score = next((s for s in scored if s.candidate == candidate), None)
                if score:
                    lines.append(f"  {i}. {candidate.provider}/{candidate.model} (score: {score.overall_score:.3f})")
                else:
                    lines.append(f"  {i}. {candidate.provider}/{candidate.model}")
        
        if excluded:
            lines.append(f"\nExcluded {len(excluded)} models:")
            for candidate in excluded[:5]:  # Show top 5 exclusions
                reasons = ", ".join(candidate.exclusion_reasons[:2])
                lines.append(f"  - {candidate.provider}/{candidate.model}: {reasons}")
            if len(excluded) > 5:
                lines.append(f"  ... and {len(excluded) - 5} more")
        
        return "\n".join(lines)

    def _calculate_confidence(
        self,
        selected: list[ModelCandidate],
        passed: list[ModelCandidate],
        excluded: list[ModelCandidate],
    ) -> float:
        """Calculate confidence in the selection."""
        if not selected:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Sample size of selected models
        # 2. Diversity of options (more passed = more confidence)
        # 3. Clear winner (gap between top candidates)
        
        avg_sample_size = sum(c.sample_size for c in selected) / len(selected)
        sample_confidence = min(avg_sample_size / 100, 1.0)  # 100 samples = max confidence
        
        diversity_confidence = min(len(passed) / 3, 1.0)  # 3+ options = max confidence
        
        # Combined confidence
        confidence = (sample_confidence * 0.6 + diversity_confidence * 0.4)
        
        return round(confidence, 2)


# Import for type hints
from sqlalchemy import Integer
