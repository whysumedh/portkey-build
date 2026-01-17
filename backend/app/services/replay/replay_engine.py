"""
Replay Engine - Deterministic prompt replay for model evaluation.

The replay engine:
- Replays historical prompts deterministically
- Executes selected candidate models via Portkey
- Tracks: input tokens, output tokens, retries, latency, errors
- Never stores actual completions (only hashes for verification)
"""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.evaluation import (
    EvaluationRun,
    ReplayRun,
    ReplayResult,
    EvaluationStatus,
)
from app.services.ingestion.portkey_client import PortkeyClient, get_portkey_client

logger = get_logger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for a replay run."""
    model: str
    provider: str
    temperature: float = 0.0  # Deterministic by default
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    trace_prefix: str = "replay"


@dataclass
class SingleReplayResult:
    """Result of replaying a single prompt."""
    original_log_id: uuid.UUID
    prompt_hash: str
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error_message: str | None = None
    refusal: bool = False
    completion_hash: str | None = None
    completion_text: str | None = None  # Stored temporarily for evaluation


@dataclass
class ModelReplayResult:
    """Aggregated results for a model's replay run."""
    model: str
    provider: str
    total_prompts: int
    successful: int
    failed: int
    refusals: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    results: list[SingleReplayResult] = field(default_factory=list)


class ReplayEngine:
    """
    Engine for replaying prompts through candidate models.
    
    Key properties:
    - Deterministic replay (temperature=0, fixed seeds)
    - No completion storage (only hashes)
    - Concurrent execution with rate limiting
    - Comprehensive error tracking
    """

    def __init__(
        self,
        session: AsyncSession,
        portkey_client: PortkeyClient | None = None,
        max_concurrent: int = 5,
    ):
        self.session = session
        self.portkey = portkey_client or get_portkey_client()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_replay_run(
        self,
        evaluation_run: EvaluationRun,
        sample_log_ids: list[uuid.UUID],
        models: list[dict[str, str]],  # [{"model": "...", "provider": "..."}]
        progress_callback: callable = None,
    ) -> list[ModelReplayResult]:
        """
        Execute a complete replay run for multiple models.
        
        Args:
            evaluation_run: The parent evaluation run
            sample_log_ids: IDs of log entries to replay
            models: List of model/provider dicts to test
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ModelReplayResult for each model
        """
        logger.info(
            "Starting replay run",
            evaluation_run_id=str(evaluation_run.id),
            sample_size=len(sample_log_ids),
            model_count=len(models),
        )

        # Load sample prompts
        prompts = await self._load_prompts(sample_log_ids)
        
        if not prompts:
            logger.warning("No prompts to replay")
            return []

        # Update evaluation run status
        evaluation_run.status = EvaluationStatus.RUNNING.value
        evaluation_run.started_at = datetime.now(timezone.utc)
        evaluation_run.replays_total = len(prompts) * len(models)
        await self.session.commit()

        results = []
        
        for model_info in models:
            config = ReplayConfig(
                model=model_info["model"],
                provider=model_info["provider"],
            )
            
            # Create replay run record
            replay_run = ReplayRun(
                evaluation_run_id=evaluation_run.id,
                model=config.model,
                provider=config.provider,
                is_baseline=(model_info.get("is_baseline", False)),
                status=EvaluationStatus.RUNNING.value,
                started_at=datetime.now(timezone.utc),
                total_prompts=len(prompts),
            )
            self.session.add(replay_run)
            await self.session.flush()

            try:
                # Execute replay for this model
                model_result = await self._replay_model(
                    replay_run=replay_run,
                    prompts=prompts,
                    config=config,
                    progress_callback=progress_callback,
                )
                results.append(model_result)

                # Update replay run with results
                replay_run.status = EvaluationStatus.COMPLETED.value
                replay_run.completed_at = datetime.now(timezone.utc)
                replay_run.successful_completions = model_result.successful
                replay_run.failed_completions = model_result.failed
                replay_run.refusals = model_result.refusals
                replay_run.total_input_tokens = model_result.total_input_tokens
                replay_run.total_output_tokens = model_result.total_output_tokens
                replay_run.total_cost_usd = model_result.total_cost_usd
                replay_run.avg_latency_ms = model_result.avg_latency_ms
                replay_run.p50_latency_ms = model_result.p50_latency_ms
                replay_run.p95_latency_ms = model_result.p95_latency_ms
                replay_run.p99_latency_ms = model_result.p99_latency_ms

            except Exception as e:
                logger.error(f"Replay failed for {config.model}: {e}")
                replay_run.status = EvaluationStatus.FAILED.value
                replay_run.error_message = str(e)
                replay_run.completed_at = datetime.now(timezone.utc)

            await self.session.commit()

            # Update overall progress
            evaluation_run.replays_completed += len(prompts)
            await self.session.commit()

        return results

    async def _load_prompts(
        self,
        log_ids: list[uuid.UUID],
    ) -> list[dict[str, Any]]:
        """Load prompts from log entries."""
        result = await self.session.execute(
            select(
                LogEntry.id,
                LogEntry.prompt,
                LogEntry.system_prompt,
                LogEntry.context,
                LogEntry.tool_calls,
                LogEntry.prompt_hash,
            )
            .where(LogEntry.id.in_(log_ids))
        )
        
        prompts = []
        for row in result.all():
            if row.prompt:  # Only include if we have the prompt
                prompts.append({
                    "log_id": row.id,
                    "prompt": row.prompt,
                    "system_prompt": row.system_prompt,
                    "context": row.context,
                    "tool_calls": row.tool_calls,
                    "prompt_hash": row.prompt_hash,
                })
        
        return prompts

    async def _replay_model(
        self,
        replay_run: ReplayRun,
        prompts: list[dict[str, Any]],
        config: ReplayConfig,
        progress_callback: callable = None,
    ) -> ModelReplayResult:
        """Replay all prompts through a single model."""
        
        logger.info(
            f"Replaying {len(prompts)} prompts through {config.provider}/{config.model}"
        )

        # Execute replays concurrently with rate limiting
        tasks = []
        for prompt_data in prompts:
            task = self._replay_single_with_semaphore(
                prompt_data=prompt_data,
                config=config,
                replay_run_id=replay_run.id,
            )
            tasks.append(task)

        results: list[SingleReplayResult] = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        refusal_count = 0
        
        for r in results:
            if isinstance(r, Exception):
                failed_count += 1
                logger.error(f"Replay task failed: {r}")
            elif r.success:
                successful_results.append(r)
                if r.refusal:
                    refusal_count += 1
            else:
                failed_count += 1
                successful_results.append(r)  # Include for analysis

        # Calculate latency percentiles
        latencies = [r.latency_ms for r in successful_results if r.latency_ms > 0]
        
        import numpy as np
        if latencies:
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            avg = float(np.mean(latencies))
        else:
            p50 = p95 = p99 = avg = 0.0

        # Save individual results to database
        for r in successful_results:
            replay_result = ReplayResult(
                replay_run_id=replay_run.id,
                original_log_id=r.original_log_id,
                prompt_hash=r.prompt_hash,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                latency_ms=r.latency_ms,
                cost_usd=r.cost_usd,
                success=r.success,
                error_message=r.error_message,
                refusal=r.refusal,
                completion_hash=r.completion_hash,
            )
            self.session.add(replay_result)
        
        await self.session.flush()

        return ModelReplayResult(
            model=config.model,
            provider=config.provider,
            total_prompts=len(prompts),
            successful=len([r for r in successful_results if r.success]),
            failed=failed_count,
            refusals=refusal_count,
            total_input_tokens=sum(r.input_tokens for r in successful_results),
            total_output_tokens=sum(r.output_tokens for r in successful_results),
            total_cost_usd=sum(r.cost_usd for r in successful_results),
            avg_latency_ms=avg,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            results=successful_results,
        )

    async def _replay_single_with_semaphore(
        self,
        prompt_data: dict[str, Any],
        config: ReplayConfig,
        replay_run_id: uuid.UUID,
    ) -> SingleReplayResult:
        """Execute a single replay with concurrency limiting."""
        async with self._semaphore:
            return await self._replay_single(prompt_data, config, replay_run_id)

    async def _replay_single(
        self,
        prompt_data: dict[str, Any],
        config: ReplayConfig,
        replay_run_id: uuid.UUID,
    ) -> SingleReplayResult:
        """Execute a single prompt replay."""
        import time
        
        log_id = prompt_data["log_id"]
        prompt = prompt_data["prompt"]
        system_prompt = prompt_data.get("system_prompt")
        prompt_hash = prompt_data["prompt_hash"]

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        
        try:
            # Execute through Portkey
            response = await self.portkey.chat_completion(
                messages=messages,
                model=config.model,
                provider=config.provider,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                trace_id=f"{config.trace_prefix}-{replay_run_id}-{log_id}",
            )

            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data
            usage = response.get("usage", {})
            choices = response.get("choices", [])
            completion_text = ""
            
            if choices:
                completion_text = choices[0].get("message", {}).get("content", "")

            # Check for refusal
            refusal = self._detect_refusal(completion_text, response)

            # Hash completion for verification (don't store actual text)
            completion_hash = hashlib.sha256(completion_text.encode()).hexdigest()

            return SingleReplayResult(
                original_log_id=log_id,
                prompt_hash=prompt_hash,
                success=True,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                cost_usd=response.get("cost", 0.0),
                refusal=refusal,
                completion_hash=completion_hash,
                completion_text=completion_text,  # Kept temporarily for evaluation
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Replay failed: {e}")
            
            return SingleReplayResult(
                original_log_id=log_id,
                prompt_hash=prompt_hash,
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    def _detect_refusal(self, completion: str, response: dict) -> bool:
        """Detect if the model refused to respond."""
        # Check for common refusal patterns
        refusal_patterns = [
            "I cannot",
            "I'm not able to",
            "I apologize, but I cannot",
            "I'm sorry, but I can't",
            "I won't be able to",
            "I must decline",
            "I can't assist with",
            "I'm unable to",
        ]
        
        completion_lower = completion.lower()
        for pattern in refusal_patterns:
            if pattern.lower() in completion_lower:
                return True
        
        # Check if response indicates refusal
        if response.get("refusal"):
            return True
            
        return False
