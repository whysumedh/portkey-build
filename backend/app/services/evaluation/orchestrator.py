"""
Evaluation Orchestrator - Coordinates AI judges for replay evaluation.

Responsibilities:
- Run judges on replay results comparing to original responses
- Use Portkey SDK for all judge LLM calls
- Track disagreements and variance
- Aggregate scores across judges
- Support both comparison and standalone quality evaluation
"""

import asyncio
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from portkey_ai import AsyncPortkey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.evaluation import (
    JudgeResult as JudgeResultModel,
    ReplayResult,
    ReplayRun,
)
from app.services.evaluation.judge_base import BaseJudge, JudgeResult
from app.services.evaluation.comparison_judge import ComparisonJudge, QualityComparisonJudge
from app.services.evaluation.quality_judge import QualityJudge, HelpfulnessJudge

logger = get_logger(__name__)


@dataclass
class AggregatedJudgment:
    """Aggregated judgment across all judges."""
    replay_result_id: UUID | None = None
    overall_score: float = 0.0
    confidence: float = 0.0
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_confidences: dict[str, float] = field(default_factory=dict)
    comparison_verdict: str = "unknown"  # better, equivalent, worse
    disagreement_level: float = 0.0
    high_disagreement: bool = False
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluationSummary:
    """Summary of evaluation results for a single model."""
    model: str
    provider: str
    replay_run_id: UUID | None
    total_evaluated: int
    avg_overall_score: float
    avg_comparison_score: float  # Specifically for comparison judge
    avg_quality_score: float
    avg_scores_by_judge: dict[str, float]
    disagreement_rate: float
    high_disagreement_cases: int
    quality_distribution: dict[str, int]  # "low", "medium", "high" counts
    comparison_verdicts: dict[str, int]  # "better", "equivalent", "worse" counts
    
    # Cost and performance from replay
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


@dataclass  
class EvaluationRunSummary:
    """Summary of a complete evaluation run across all models."""
    evaluation_run_id: UUID
    project_id: UUID
    completed_at: str
    total_logs_evaluated: int
    model_summaries: list[ModelEvaluationSummary]
    recommended_model: str | None = None
    recommended_provider: str | None = None
    recommendation_reasoning: str = ""
    recommendation_confidence: float = 0.0


class EvaluationOrchestrator:
    """
    Orchestrates AI judge evaluation for replay results using Portkey SDK.
    
    Key responsibilities:
    - Initialize judges for comparison and quality evaluation
    - Run evaluations with proper concurrency
    - Compare replay responses to originals
    - Aggregate and analyze results
    - Generate model comparison summaries
    """

    def __init__(
        self,
        session: AsyncSession,
        judge_model: str | None = None,
    ):
        self.session = session
        # Judge model uses @provider/model format (e.g., @openai/o4-mini-deep-research)
        self.judge_model = judge_model or settings.judge_model
        
        # Initialize Portkey client - uses @provider/model format, no virtual keys needed
        self.portkey = AsyncPortkey(
            api_key=settings.portkey_api_key,
        )
        
        # Initialize judges - focusing on comparison and quality for replay
        self.comparison_judge = ComparisonJudge(portkey_client=self)
        self.quality_judge = QualityComparisonJudge(portkey_client=self)
        self.helpfulness_judge = HelpfulnessJudge(portkey_client=self)
        
        self.judges: list[BaseJudge] = [
            self.comparison_judge,
            self.quality_judge,
            self.helpfulness_judge,
        ]
        
        self._disagreement_threshold = settings.judge_disagreement_threshold
        self._semaphore = asyncio.Semaphore(settings.judge_max_concurrent)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute chat completion via Portkey SDK.
        Model uses @provider/model format (e.g., @openai/o4-mini-deep-research).
        Used by judges for evaluation calls.
        """
        response = await self.portkey.chat.completions.create(
            model=model or self.judge_model,
            messages=messages,
            **kwargs,
        )
        
        # Convert SDK response to dict format expected by judges
        return {
            "choices": [
                {
                    "message": {
                        "content": response.choices[0].message.content if response.choices else ""
                    }
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }
        }

    async def evaluate_replay_result(
        self,
        replay_result: ReplayResult,
        prompt: str,
        original_completion: str | None = None,
    ) -> AggregatedJudgment:
        """
        Evaluate a single replay result against the original response.
        
        Args:
            replay_result: The replay result to evaluate
            prompt: The original user prompt
            original_completion: The original model's response (for comparison)
            
        Returns:
            AggregatedJudgment with scores from all judges
        """
        completion = replay_result.completion_text
        
        if not completion:
            return AggregatedJudgment(
                replay_result_id=replay_result.id,
                overall_score=0.0,
                confidence=0.0,
                flags=["no_completion"],
                comparison_verdict="worse",
            )
        
        # Build context with original completion
        context = {
            "original_completion": original_completion or replay_result.original_completion,
            "original_model": getattr(replay_result, 'original_model', None),
        }
        
        # Run judges concurrently with rate limiting
        results: list[JudgeResult] = []
        tasks = []
        
        for judge in self.judges:
            task = self._run_judge_with_semaphore(judge, prompt, completion, context)
            tasks.append(task)
        
        judge_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in judge_results:
            if isinstance(result, JudgeResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Judge failed: {result}")
        
        # Store results in database
        await self._store_results(replay_result.id, results)
        
        # Aggregate results
        return self._aggregate_results(results, replay_result.id)

    async def _run_judge_with_semaphore(
        self,
        judge: BaseJudge,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None,
    ) -> JudgeResult:
        """Run a judge with concurrency limiting."""
        async with self._semaphore:
            return await self._run_judge(judge, prompt, completion, context)

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
        
        # Run full evaluation - model uses @provider/model format
        return await judge.evaluate(
            prompt=prompt,
            completion=completion,
            context=context,
            model=self.judge_model,
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

    def _aggregate_results(
        self,
        results: list[JudgeResult],
        replay_result_id: UUID | None = None,
    ) -> AggregatedJudgment:
        """Aggregate results from multiple judges."""
        if not results:
            return AggregatedJudgment(
                replay_result_id=replay_result_id,
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
            disagreement_level = min(score_std / 0.5, 1.0)
        else:
            disagreement_level = 0.0
        
        high_disagreement = disagreement_level > self._disagreement_threshold
        
        # Determine comparison verdict from comparison judge
        comparison_verdict = "unknown"
        comparison_result = next(
            (r for r in results if r.judge_type == "comparison"), None
        )
        if comparison_result and comparison_result.details:
            comparison_verdict = comparison_result.details.get("comparison_verdict", "unknown")
        
        # Generate flags
        flags = []
        if high_disagreement:
            flags.append("high_judge_disagreement")
        if comparison_verdict == "worse":
            flags.append("quality_regression")
        if avg_confidence < 0.5:
            flags.append("low_confidence")
        
        return AggregatedJudgment(
            replay_result_id=replay_result_id,
            overall_score=weighted_score,
            confidence=avg_confidence,
            judge_scores=scores,
            judge_confidences=confidences,
            comparison_verdict=comparison_verdict,
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
        replay_run: ReplayRun,
        progress_callback: callable = None,
    ) -> ModelEvaluationSummary:
        """
        Evaluate all results from a replay run.
        
        Args:
            replay_run: The replay run to evaluate
            progress_callback: Optional progress callback
            
        Returns:
            ModelEvaluationSummary with aggregated metrics for this model
        """
        # Load replay results with prompts
        result = await self.session.execute(
            select(ReplayResult).where(ReplayResult.replay_run_id == replay_run.id)
        )
        replay_results = list(result.unique().scalars().all())
        
        if not replay_results:
            return ModelEvaluationSummary(
                model=replay_run.model,
                provider=replay_run.provider,
                replay_run_id=replay_run.id,
                total_evaluated=0,
                avg_overall_score=0.0,
                avg_comparison_score=0.0,
                avg_quality_score=0.0,
                avg_scores_by_judge={},
                disagreement_rate=0.0,
                high_disagreement_cases=0,
                quality_distribution={"low": 0, "medium": 0, "high": 0},
                comparison_verdicts={"better": 0, "equivalent": 0, "worse": 0, "unknown": 0},
                total_cost_usd=replay_run.total_cost_usd or 0.0,
                avg_latency_ms=replay_run.avg_latency_ms or 0.0,
                p95_latency_ms=replay_run.p95_latency_ms or 0.0,
            )
        
        # Load original log entries to get prompts
        log_ids = [r.original_log_id for r in replay_results]
        from app.models.log_entry import LogEntry
        log_result = await self.session.execute(
            select(LogEntry).where(LogEntry.id.in_(log_ids))
        )
        logs_by_id = {log.id: log for log in log_result.scalars().all()}
        
        # Evaluate each replay result
        all_judgments: list[AggregatedJudgment] = []
        
        for i, replay_result in enumerate(replay_results):
            if not replay_result.completion_text:
                continue
            
            # Get original prompt
            log = logs_by_id.get(replay_result.original_log_id)
            if not log:
                continue
            
            prompt = log.prompt or ""
            
            # Get original completion
            original_completion = replay_result.original_completion
            if not original_completion and log.response_data:
                choices = log.response_data.get("choices", [])
                if choices:
                    original_completion = choices[0].get("message", {}).get("content", "")
            
            judgment = await self.evaluate_replay_result(
                replay_result=replay_result,
                prompt=prompt,
                original_completion=original_completion,
            )
            all_judgments.append(judgment)
            
            if progress_callback:
                await progress_callback(i + 1, len(replay_results))
        
        return self._summarize_model_evaluation(
            replay_run=replay_run,
            judgments=all_judgments,
        )

    def _summarize_model_evaluation(
        self,
        replay_run: ReplayRun,
        judgments: list[AggregatedJudgment],
    ) -> ModelEvaluationSummary:
        """Summarize all judgments for a model."""
        if not judgments:
            return ModelEvaluationSummary(
                model=replay_run.model,
                provider=replay_run.provider,
                replay_run_id=replay_run.id,
                total_evaluated=0,
                avg_overall_score=0.0,
                avg_comparison_score=0.0,
                avg_quality_score=0.0,
                avg_scores_by_judge={},
                disagreement_rate=0.0,
                high_disagreement_cases=0,
                quality_distribution={"low": 0, "medium": 0, "high": 0},
                comparison_verdicts={"better": 0, "equivalent": 0, "worse": 0, "unknown": 0},
                total_cost_usd=replay_run.total_cost_usd or 0.0,
                avg_latency_ms=replay_run.avg_latency_ms or 0.0,
                p95_latency_ms=replay_run.p95_latency_ms or 0.0,
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
        
        # Comparison verdicts
        verdicts = {"better": 0, "equivalent": 0, "worse": 0, "unknown": 0}
        for judgment in judgments:
            verdict = judgment.comparison_verdict
            if verdict in verdicts:
                verdicts[verdict] += 1
            else:
                verdicts["unknown"] += 1
        
        # Calculate specific averages
        comparison_scores = scores_by_judge.get("comparison", [])
        quality_scores = scores_by_judge.get("quality_standalone", [])
        
        return ModelEvaluationSummary(
            model=replay_run.model,
            provider=replay_run.provider,
            replay_run_id=replay_run.id,
            total_evaluated=len(judgments),
            avg_overall_score=statistics.mean([j.overall_score for j in judgments]),
            avg_comparison_score=statistics.mean(comparison_scores) if comparison_scores else 0.0,
            avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
            avg_scores_by_judge=avg_by_judge,
            disagreement_rate=statistics.mean([j.disagreement_level for j in judgments]),
            high_disagreement_cases=sum(1 for j in judgments if j.high_disagreement),
            quality_distribution=quality_dist,
            comparison_verdicts=verdicts,
            total_cost_usd=replay_run.total_cost_usd or 0.0,
            avg_latency_ms=replay_run.avg_latency_ms or 0.0,
            p95_latency_ms=replay_run.p95_latency_ms or 0.0,
        )

    async def evaluate_full_run(
        self,
        evaluation_run_id: UUID,
        progress_callback: callable = None,
    ) -> EvaluationRunSummary:
        """
        Evaluate all replay runs for an evaluation.
        
        Args:
            evaluation_run_id: The evaluation run ID
            progress_callback: Optional progress callback
            
        Returns:
            EvaluationRunSummary with all model comparisons
        """
        from app.models.evaluation import EvaluationRun
        
        # Load evaluation run
        eval_result = await self.session.execute(
            select(EvaluationRun).where(EvaluationRun.id == evaluation_run_id)
        )
        evaluation_run = eval_result.scalar_one_or_none()
        
        if not evaluation_run:
            raise ValueError(f"Evaluation run {evaluation_run_id} not found")
        
        # Load all replay runs
        replay_result = await self.session.execute(
            select(ReplayRun).where(ReplayRun.evaluation_run_id == evaluation_run_id)
        )
        replay_runs = list(replay_result.unique().scalars().all())
        
        # Evaluate each model
        model_summaries = []
        for replay_run in replay_runs:
            summary = await self.evaluate_replay_run(
                replay_run=replay_run,
                progress_callback=progress_callback,
            )
            model_summaries.append(summary)
        
        # Determine recommended model
        recommended_model = None
        recommended_provider = None
        recommendation_reasoning = ""
        recommendation_confidence = 0.0
        
        if model_summaries:
            # Score models by: quality * (1 - cost_factor)
            # Higher quality and lower cost = better
            scored_models = []
            max_cost = max(s.total_cost_usd for s in model_summaries) or 1
            
            for summary in model_summaries:
                cost_factor = summary.total_cost_usd / max_cost if max_cost > 0 else 0
                # Quality score weighted more than cost
                combined_score = (summary.avg_overall_score * 0.7) + ((1 - cost_factor) * 0.3)
                scored_models.append((summary, combined_score))
            
            scored_models.sort(key=lambda x: x[1], reverse=True)
            best = scored_models[0][0]
            
            # Only recommend if quality is acceptable
            if best.avg_overall_score >= 0.6:
                recommended_model = best.model
                recommended_provider = best.provider
                recommendation_confidence = min(best.avg_overall_score, 0.9)
                
                # Build reasoning
                verdicts = best.comparison_verdicts
                better_pct = verdicts["better"] / best.total_evaluated * 100 if best.total_evaluated > 0 else 0
                equiv_pct = verdicts["equivalent"] / best.total_evaluated * 100 if best.total_evaluated > 0 else 0
                
                recommendation_reasoning = (
                    f"{recommended_provider}/{recommended_model} scored {best.avg_overall_score:.2f} overall. "
                    f"{better_pct:.0f}% responses were better than original, {equiv_pct:.0f}% equivalent. "
                    f"Total cost: ${best.total_cost_usd:.4f}, avg latency: {best.avg_latency_ms:.0f}ms."
                )
        
        return EvaluationRunSummary(
            evaluation_run_id=evaluation_run_id,
            project_id=evaluation_run.project_id,
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_logs_evaluated=sum(s.total_evaluated for s in model_summaries),
            model_summaries=model_summaries,
            recommended_model=recommended_model,
            recommended_provider=recommended_provider,
            recommendation_reasoning=recommendation_reasoning,
            recommendation_confidence=recommendation_confidence,
        )


def get_evaluation_orchestrator(session: AsyncSession) -> EvaluationOrchestrator:
    """Get a configured evaluation orchestrator instance."""
    return EvaluationOrchestrator(session)
