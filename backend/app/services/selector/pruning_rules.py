"""
Deterministic Pruning Rules Engine.

Applies hard rules to filter out candidate models that don't meet
project constraints. This runs BEFORE AI-based model selection.
"""

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.models.project import Project

logger = get_logger(__name__)


@dataclass
class ModelCandidate:
    """A candidate model with its metrics."""
    model: str
    provider: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_cost_per_request: float = 0.0
    refusal_rate: float = 0.0
    error_rate: float = 0.0
    quality_score: float = 0.0
    sample_size: int = 0
    
    # Exclusion tracking
    excluded: bool = False
    exclusion_reasons: list[str] = None
    
    def __post_init__(self):
        if self.exclusion_reasons is None:
            self.exclusion_reasons = []
    
    def exclude(self, reason: str) -> None:
        """Mark this candidate as excluded with a reason."""
        self.excluded = True
        self.exclusion_reasons.append(reason)


@dataclass
class PruningResult:
    """Result of the pruning process."""
    passed: list[ModelCandidate]
    excluded: list[ModelCandidate]
    total_candidates: int
    rules_applied: list[str]


class PruningRulesEngine:
    """
    Engine for applying deterministic pruning rules to model candidates.
    
    Rules are applied in order:
    1. Latency constraints (hard limit)
    2. Cost constraints (based on sensitivity)
    3. Refusal rate constraints
    4. Error rate constraints
    5. Minimum sample size
    
    All rules are deterministic and explainable.
    """

    def __init__(self, project: Project):
        self.project = project
        self.criteria = project.success_criteria
        self.tolerances = project.tolerance_levels

    def apply_rules(
        self,
        candidates: list[ModelCandidate],
    ) -> PruningResult:
        """
        Apply all pruning rules to candidates.
        
        Returns a PruningResult with passed and excluded candidates.
        """
        rules_applied = []
        
        # Apply each rule in sequence
        candidates = self._apply_latency_rule(candidates, rules_applied)
        candidates = self._apply_cost_rule(candidates, rules_applied)
        candidates = self._apply_refusal_rule(candidates, rules_applied)
        candidates = self._apply_error_rule(candidates, rules_applied)
        candidates = self._apply_sample_size_rule(candidates, rules_applied)

        passed = [c for c in candidates if not c.excluded]
        excluded = [c for c in candidates if c.excluded]

        logger.info(
            "Pruning complete",
            total=len(candidates),
            passed=len(passed),
            excluded=len(excluded),
            rules=rules_applied,
        )

        return PruningResult(
            passed=passed,
            excluded=excluded,
            total_candidates=len(candidates),
            rules_applied=rules_applied,
        )

    def _apply_latency_rule(
        self,
        candidates: list[ModelCandidate],
        rules_applied: list[str],
    ) -> list[ModelCandidate]:
        """Apply latency constraints."""
        if not self.criteria or not self.tolerances:
            return candidates

        max_latency = self.criteria.max_latency_ms
        max_p95 = self.criteria.max_latency_p95_ms
        absolute_max = self.tolerances.absolute_max_latency_ms

        rules_applied.append(f"latency_rule(max={max_latency}ms, p95_max={max_p95}ms, absolute={absolute_max}ms)")

        for candidate in candidates:
            if candidate.excluded:
                continue
                
            # Hard limit: absolute maximum
            if candidate.p95_latency_ms > absolute_max:
                candidate.exclude(
                    f"P95 latency ({candidate.p95_latency_ms:.0f}ms) exceeds absolute max ({absolute_max:.0f}ms)"
                )
                continue

            # Soft limit with tolerance
            tolerance = self.tolerances.latency_tolerance_pct
            adjusted_max = max_latency * (1 + tolerance)
            
            if candidate.avg_latency_ms > adjusted_max:
                candidate.exclude(
                    f"Avg latency ({candidate.avg_latency_ms:.0f}ms) exceeds max ({adjusted_max:.0f}ms with {tolerance*100:.0f}% tolerance)"
                )

        return candidates

    def _apply_cost_rule(
        self,
        candidates: list[ModelCandidate],
        rules_applied: list[str],
    ) -> list[ModelCandidate]:
        """Apply cost constraints based on sensitivity."""
        if not self.criteria or not self.tolerances:
            return candidates

        max_cost = self.criteria.max_cost_per_request_usd
        sensitivity = self.tolerances.cost_sensitivity
        tolerance = self.tolerances.cost_tolerance_pct

        # Adjust tolerance based on sensitivity
        if sensitivity == "high":
            tolerance = tolerance / 2  # Stricter for high sensitivity
        elif sensitivity == "low":
            tolerance = tolerance * 2  # More lenient for low sensitivity

        adjusted_max = max_cost * (1 + tolerance)
        rules_applied.append(f"cost_rule(max=${max_cost:.4f}, sensitivity={sensitivity}, adjusted=${adjusted_max:.4f})")

        for candidate in candidates:
            if candidate.excluded:
                continue
                
            if candidate.avg_cost_per_request > adjusted_max:
                candidate.exclude(
                    f"Cost (${candidate.avg_cost_per_request:.4f}/req) exceeds max (${adjusted_max:.4f} with {sensitivity} sensitivity)"
                )

        return candidates

    def _apply_refusal_rule(
        self,
        candidates: list[ModelCandidate],
        rules_applied: list[str],
    ) -> list[ModelCandidate]:
        """Apply refusal rate constraints."""
        if not self.criteria or not self.tolerances:
            return candidates

        max_refusal = self.criteria.max_refusal_rate
        absolute_max = self.tolerances.absolute_max_refusal_rate

        rules_applied.append(f"refusal_rule(max={max_refusal*100:.1f}%, absolute={absolute_max*100:.1f}%)")

        for candidate in candidates:
            if candidate.excluded:
                continue
                
            # Hard limit
            if candidate.refusal_rate > absolute_max:
                candidate.exclude(
                    f"Refusal rate ({candidate.refusal_rate*100:.1f}%) exceeds absolute max ({absolute_max*100:.1f}%)"
                )
                continue

            # Soft limit
            if candidate.refusal_rate > max_refusal:
                candidate.exclude(
                    f"Refusal rate ({candidate.refusal_rate*100:.1f}%) exceeds target ({max_refusal*100:.1f}%)"
                )

        return candidates

    def _apply_error_rule(
        self,
        candidates: list[ModelCandidate],
        rules_applied: list[str],
    ) -> list[ModelCandidate]:
        """Apply error rate constraints."""
        max_error_rate = 0.05  # 5% max error rate (hardcoded reasonable default)
        
        rules_applied.append(f"error_rule(max={max_error_rate*100:.1f}%)")

        for candidate in candidates:
            if candidate.excluded:
                continue
                
            if candidate.error_rate > max_error_rate:
                candidate.exclude(
                    f"Error rate ({candidate.error_rate*100:.1f}%) exceeds max ({max_error_rate*100:.1f}%)"
                )

        return candidates

    def _apply_sample_size_rule(
        self,
        candidates: list[ModelCandidate],
        rules_applied: list[str],
    ) -> list[ModelCandidate]:
        """Require minimum sample size for reliable metrics."""
        min_samples = 10  # Need at least 10 samples for statistical significance
        
        rules_applied.append(f"sample_size_rule(min={min_samples})")

        for candidate in candidates:
            if candidate.excluded:
                continue
                
            if candidate.sample_size < min_samples:
                candidate.exclude(
                    f"Insufficient data ({candidate.sample_size} samples < {min_samples} minimum)"
                )

        return candidates


def create_candidate_from_metrics(
    model: str,
    provider: str,
    metrics: dict[str, Any],
) -> ModelCandidate:
    """Create a ModelCandidate from aggregated metrics."""
    return ModelCandidate(
        model=model,
        provider=provider,
        avg_latency_ms=metrics.get("avg_latency_ms", 0.0),
        p95_latency_ms=metrics.get("p95_latency_ms", 0.0),
        avg_cost_per_request=metrics.get("avg_cost_per_request", 0.0),
        refusal_rate=metrics.get("refusal_rate", 0.0),
        error_rate=metrics.get("error_rate", 0.0),
        quality_score=metrics.get("quality_score", 0.0),
        sample_size=metrics.get("sample_size", 0),
    )
