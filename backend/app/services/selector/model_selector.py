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
from app.services.catalog.model_catalog_service import ModelCatalogService
from app.models.model_catalog import ModelCatalogEntry

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
    ):
        self.session = session
        self.catalog_service = ModelCatalogService(session)

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

        # Step 5: Select top N (minimum 3 models always)
        min_models = max(top_n, 3)  # Always select at least 3 models
        selected = []
        for scored in scored_candidates[:min_models]:
            selected.append(scored.candidate)

        # Always include current model if requested
        if include_current and current_model and not current_model.excluded:
            if current_model not in selected:
                selected.append(current_model)
        
        # Ensure we have at least 3 models by adding from catalog if needed
        if len(selected) < 3:
            # Add more models from catalog that weren't pruned
            seen_models = {(c.provider, c.model) for c in selected}
            for scored in scored_candidates:
                if (scored.candidate.provider, scored.candidate.model) not in seen_models:
                    selected.append(scored.candidate)
                    seen_models.add((scored.candidate.provider, scored.candidate.model))
                    if len(selected) >= 3:
                        break

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
        """
        Gather candidate models from logs and model catalog.
        
        Uses project requirements to intelligently filter and prioritize models:
        - Infers use cases from agent purpose
        - Considers cost sensitivity for tier selection
        - Prioritizes models that match the project's needs
        """
        candidates = []
        seen_models = set()

        # Step 1: Get models from project logs (these have real usage data)
        result = await self.session.execute(
            select(LogEntry.model, LogEntry.provider)
            .where(LogEntry.project_id == project.id)
            .distinct()
        )
        
        log_models = []
        for row in result.all():
            if row.model and row.provider:
                model_key = f"{row.provider}:{row.model}"
                if model_key not in seen_models:
                    candidates.append(ModelCandidate(
                        model=row.model,
                        provider=row.provider,
                    ))
                    seen_models.add(model_key)
                    log_models.append(f"{row.provider}/{row.model}")
        
        logger.info(
            "Found models in project logs",
            model_count=len(log_models),
            models=log_models,
        )

        # Step 2: Infer project requirements for intelligent catalog selection
        # Infer use cases from agent purpose
        # Convert SQLAlchemy model to dict for inference
        success_criteria_dict = None
        if project.success_criteria:
            success_criteria_dict = {
                "accuracy_target": project.success_criteria.min_accuracy,
                "latency_target_ms": project.success_criteria.max_latency_ms,
            }
        
        use_cases = self.catalog_service.infer_use_cases_from_project(
            agent_purpose=project.agent_purpose,
            success_criteria=success_criteria_dict,
        )
        
        # Get preferred tiers based on cost sensitivity
        cost_sensitivity = "medium"
        if project.tolerance_levels:
            cost_sensitivity = project.tolerance_levels.cost_sensitivity or "medium"
        
        preferred_tiers = self.catalog_service.get_tier_recommendations(cost_sensitivity)
        
        # Get minimum scores based on success criteria
        min_quality_score = None
        min_speed_score = None
        if project.success_criteria:
            if project.success_criteria.min_accuracy and project.success_criteria.min_accuracy > 0.9:
                min_quality_score = 7  # Require good quality for high accuracy targets
            if project.success_criteria.max_latency_ms and project.success_criteria.max_latency_ms < 3000:
                min_speed_score = 7  # Require good speed for low latency targets
        
        logger.info(
            "Inferred project requirements",
            use_cases=use_cases,
            preferred_tiers=preferred_tiers,
            cost_sensitivity=cost_sensitivity,
            min_quality=min_quality_score,
            min_speed=min_speed_score,
        )

        # Step 3: Add candidates from the model catalog based on inferred requirements
        catalog_models = await self.catalog_service.get_models_for_recommendation(
            current_provider=project.current_provider,
            current_model=project.current_model,
            use_cases=use_cases,
            preferred_tiers=preferred_tiers,
            min_quality_score=min_quality_score,
            min_speed_score=min_speed_score,
        )
        
        catalog_added = []
        for catalog_entry in catalog_models:
            model_key = f"{catalog_entry.provider}:{catalog_entry.model}"
            if model_key not in seen_models:
                # Create candidate with full catalog info
                candidate = ModelCandidate(
                    model=catalog_entry.model,
                    provider=catalog_entry.provider,
                )
                
                # Set estimated cost from catalog pricing
                if catalog_entry.input_price_per_token > 0:
                    # Estimate avg cost assuming 1000 input + 500 output tokens
                    estimated_cost = (
                        (1000 * catalog_entry.input_price_per_token / 100) +
                        (500 * catalog_entry.output_price_per_token / 100)
                    )
                    candidate.avg_cost_per_request = estimated_cost
                
                # Set quality score from catalog
                if catalog_entry.quality_score:
                    candidate.quality_score = catalog_entry.quality_score / 10.0  # Normalize to 0-1
                
                # Store tier and use case info for explanation
                candidate.metadata = {
                    "tier": catalog_entry.tier,
                    "use_cases": catalog_entry.use_cases,
                    "recommended_for": catalog_entry.recommended_for,
                    "quality_score": catalog_entry.quality_score,
                    "speed_score": catalog_entry.speed_score,
                }
                
                candidates.append(candidate)
                seen_models.add(model_key)
                
                tier_info = f" ({catalog_entry.tier})" if catalog_entry.tier else ""
                catalog_added.append(f"{catalog_entry.provider}/{catalog_entry.model}{tier_info}")
        
        logger.info(
            "Added models from catalog based on project requirements",
            catalog_count=len(catalog_added),
            total_candidates=len(candidates),
            models_added=catalog_added[:10],  # Log first 10
        )

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
            cost_score = 1.0 - min(candidate.avg_cost_per_request / 1.00, 1.0)  # Assume $1.00 is worst
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
        """Build human-readable selection reasoning with tier and use case info."""
        lines = []
        
        lines.append(f"Selected {len(selected)} candidate models for evaluation.")
        
        if selected:
            lines.append("\n📊 Selected models:")
            for i, candidate in enumerate(selected, 1):
                score = next((s for s in scored if s.candidate == candidate), None)
                
                # Build model info line with tier
                tier_badge = ""
                if candidate.tier:
                    tier_icons = {"budget": "💚", "standard": "💙", "premium": "💜", "enterprise": "👑"}
                    tier_badge = f" [{tier_icons.get(candidate.tier, '')} {candidate.tier.upper()}]"
                
                if score:
                    lines.append(f"  {i}. {candidate.provider}/{candidate.model}{tier_badge} (score: {score.overall_score:.3f})")
                else:
                    lines.append(f"  {i}. {candidate.provider}/{candidate.model}{tier_badge}")
                
                # Add recommendation reason if available
                if candidate.recommended_for:
                    lines.append(f"      → {candidate.recommended_for}")
                
                # Show use cases
                if candidate.use_cases:
                    lines.append(f"      Use cases: {', '.join(candidate.use_cases)}")
        
        # Group excluded models by tier
        if excluded:
            lines.append(f"\n❌ Excluded {len(excluded)} models:")
            
            # Group by tier for better readability
            by_tier = {"budget": [], "standard": [], "premium": [], "enterprise": [], "unknown": []}
            for candidate in excluded:
                tier = candidate.tier or "unknown"
                by_tier[tier].append(candidate)
            
            shown = 0
            for tier, tier_candidates in by_tier.items():
                if tier_candidates and shown < 5:
                    for candidate in tier_candidates[:2]:
                        reasons = ", ".join(candidate.exclusion_reasons[:2])
                        lines.append(f"  - {candidate.provider}/{candidate.model} [{tier}]: {reasons}")
                        shown += 1
                        if shown >= 5:
                            break
            
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
