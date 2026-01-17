"""
Evaluation Orchestrator - Coordinates multiple judges.

Responsibilities:
- Run all judges on replay results
- Track disagreements and variance
- Aggregate scores across judges
- Flag high-disagreement cases
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.evaluation import JudgeResult as JudgeResultModel, ReplayResult
from app.services.evaluation.judge_base import BaseJudge, JudgeResult
from app.services.evaluation.correctness_judge import CorrectnessJudge
from app.services.evaluation.safety_judge import SafetyJudge
from app.services.evaluation.quality_judge import QualityJudge, HelpfulnessJudge
from app.services.ingestion.portkey_client import PortkeyClient, get_portkey_client

logger = get_logger(__name__)


@dataclass
class AggregatedJudgment:
    """Aggregated judgment across all judges."""
    overall_score: float
    confidence: float
    judge_scores: dict[str, float]
    judge_confidences: dict[str, float]
    disagreement_level: float  # 0 = perfect agreement, 1 = complete disagreement
    high_disagreement: bool
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results for a model."""
    model: str
    provider: str
    total_evaluated: int
    avg_overall_score: float
    avg_scores_by_judge: dict[str, float]
    disagreement_rate: float
    high_disagreement_cases: int
    safety_issues: int
    quality_distribution: dict[str, int]  # "low", "medium", "high" counts


class EvaluationOrchestrator:
    """
    Orchestrates evaluation across multiple judges.
    
    Key responsibilities:
    - Initialize and manage judges
    - Run evaluations with proper concurrency
    - Aggregate and analyze results
    - Track judge disagreements
    """

    def __init__(
        self,
        session: AsyncSession,
        portkey_client: PortkeyClient | None = None,
        judge_model: str = "gpt-4o-mini",
        judge_provider: str = "openai",
    ):
        self.session = session
        self.portkey = portkey_client or get_portkey_client()
        self.judge_model = judge_model
        self.judge_provider = judge_provider
        
        # Initialize judges
        self.judges: list[BaseJudge] = [
            CorrectnessJudge(self.portkey),
            SafetyJudge(self.portkey),
            QualityJudge(self.portkey),
            HelpfulnessJudge(self.portkey),
        ]
        
        self._disagreement_threshold = settings.judge_disagreement_threshold

    async def evaluate_completion(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
        replay_result_id: UUID | None = None,
    ) -> AggregatedJudgment:
        """
        Evaluate a single completion with all judges.
        
        Args:
            prompt: The original user prompt
            completion: The model's completion
            context: Additional context
            replay_result_id: Optional ID to store results
            
        Returns:
            AggregatedJudgment with scores from all judges
        """
        results: list[JudgeResult] = []
        
        # Run judges concurrently
        tasks = []
        for judge in self.judges:
            task = self._run_judge(judge, prompt, completion, context)
            tasks.append(task)
        
        judge_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in judge_results:
            if isinstance(result, JudgeResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Judge failed: {result}")
        
        # Store results in database if replay_result_id provided
        if replay_result_id:
            await self._store_results(replay_result_id, results)
        
        # Aggregate results
        return self._aggregate_results(results)

    async def _run_judge(
        self,
        judge: BaseJudge,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None,
    ) -> JudgeResult:
        """Run a single judge with heuristic fallback."""
        # Try heuristic first
        heuristic_result = judge.evaluate_heuristic(prompt, completion, context)
        if heuristic_result:
            return heuristic_result
        
        # Run full evaluation
        return await judge.evaluate(
            prompt=prompt,
            completion=completion,
            context=context,
            model=self.judge_model,
            provider=self.judge_provider,
        )

    async def _store_results(
        self,
        replay_result_id: UUID,
        results: list[JudgeResult],
    ) -> None:
        """Store judge results in the database."""
        for result in results:
            db_result = JudgeResultModel(
                replay_result_id=replay_result_id,
                judge_type=result.judge_type,
                judge_version=result.judge_version,
                score=result.score,
                confidence=result.confidence,
                rationale=result.rationale,
                details=result.details,
                judge_prompt_hash=result.prompt_hash,
            )
            self.session.add(db_result)
        
        await self.session.flush()

    def _aggregate_results(self, results: list[JudgeResult]) -> AggregatedJudgment:
        """Aggregate results from multiple judges."""
        if not results:
            return AggregatedJudgment(
                overall_score=0.0,
                confidence=0.0,
                judge_scores={},
                judge_confidences={},
                disagreement_level=1.0,
                high_disagreement=True,
                flags=["no_judge_results"],
            )
        
        # Collect scores
        scores = {}
        confidences = {}
        
        for result in results:
            scores[result.judge_type] = result.score
            confidences[result.judge_type] = result.confidence
        
        # Calculate weighted average (weight by confidence)
        total_confidence = sum(confidences.values())
        if total_confidence > 0:
            weighted_score = sum(
                scores[k] * confidences[k] for k in scores
            ) / total_confidence
        else:
            weighted_score = statistics.mean(scores.values()) if scores else 0.0
        
        avg_confidence = statistics.mean(confidences.values()) if confidences else 0.0
        
        # Calculate disagreement
        if len(scores) >= 2:
            score_values = list(scores.values())
            score_std = statistics.stdev(score_values)
            # Normalize to 0-1 (max std for 0-1 range is 0.5)
            disagreement_level = min(score_std / 0.5, 1.0)
        else:
            disagreement_level = 0.0
        
        high_disagreement = disagreement_level > self._disagreement_threshold
        
        # Generate flags
        flags = []
        if high_disagreement:
            flags.append("high_judge_disagreement")
        
        # Check safety specifically
        safety_score = scores.get("safety", 1.0)
        if safety_score < 0.5:
            flags.append("safety_concern")
        
        # Low confidence
        if avg_confidence < 0.5:
            flags.append("low_confidence")
        
        return AggregatedJudgment(
            overall_score=weighted_score,
            confidence=avg_confidence,
            judge_scores=scores,
            judge_confidences=confidences,
            disagreement_level=disagreement_level,
            high_disagreement=high_disagreement,
            flags=flags,
            details={
                "judge_count": len(results),
                "score_std": statistics.stdev(list(scores.values())) if len(scores) >= 2 else 0,
            },
        )

    async def evaluate_replay_run(
        self,
        replay_results: list[tuple[ReplayResult, str, str]],  # (result, prompt, completion)
        progress_callback: callable = None,
    ) -> EvaluationSummary:
        """
        Evaluate all results from a replay run.
        
        Args:
            replay_results: List of (ReplayResult, prompt, completion) tuples
            progress_callback: Optional progress callback
            
        Returns:
            EvaluationSummary with aggregated metrics
        """
        all_judgments: list[AggregatedJudgment] = []
        
        for i, (result, prompt, completion) in enumerate(replay_results):
            if not completion:
                continue
                
            judgment = await self.evaluate_completion(
                prompt=prompt,
                completion=completion,
                replay_result_id=result.id,
            )
            all_judgments.append(judgment)
            
            if progress_callback:
                await progress_callback(i + 1, len(replay_results))
        
        return self._summarize_judgments(all_judgments, replay_results[0][0] if replay_results else None)

    def _summarize_judgments(
        self,
        judgments: list[AggregatedJudgment],
        sample_result: ReplayResult | None,
    ) -> EvaluationSummary:
        """Summarize all judgments for a model."""
        if not judgments:
            return EvaluationSummary(
                model="unknown",
                provider="unknown",
                total_evaluated=0,
                avg_overall_score=0.0,
                avg_scores_by_judge={},
                disagreement_rate=0.0,
                high_disagreement_cases=0,
                safety_issues=0,
                quality_distribution={"low": 0, "medium": 0, "high": 0},
            )
        
        # Aggregate scores by judge type
        scores_by_judge: dict[str, list[float]] = {}
        for judgment in judgments:
            for judge_type, score in judgment.judge_scores.items():
                if judge_type not in scores_by_judge:
                    scores_by_judge[judge_type] = []
                scores_by_judge[judge_type].append(score)
        
        avg_by_judge = {
            k: statistics.mean(v) for k, v in scores_by_judge.items()
        }
        
        # Quality distribution
        quality_dist = {"low": 0, "medium": 0, "high": 0}
        for judgment in judgments:
            if judgment.overall_score < 0.4:
                quality_dist["low"] += 1
            elif judgment.overall_score < 0.7:
                quality_dist["medium"] += 1
            else:
                quality_dist["high"] += 1
        
        return EvaluationSummary(
            model=sample_result.replay_run.model if sample_result else "unknown",
            provider=sample_result.replay_run.provider if sample_result else "unknown",
            total_evaluated=len(judgments),
            avg_overall_score=statistics.mean([j.overall_score for j in judgments]),
            avg_scores_by_judge=avg_by_judge,
            disagreement_rate=statistics.mean([j.disagreement_level for j in judgments]),
            high_disagreement_cases=sum(1 for j in judgments if j.high_disagreement),
            safety_issues=sum(1 for j in judgments if "safety_concern" in j.flags),
            quality_distribution=quality_dist,
        )
