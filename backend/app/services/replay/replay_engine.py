"""
Replay Engine - Deterministic prompt replay for model evaluation.

The replay engine:
- Replays historical prompts through candidate models via Portkey
- Uses the Portkey SDK with @provider/model format
- Tracks real costs from Portkey's response metadata
- Stores completions for AI judge comparison
- Executes with configurable concurrency
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from portkey_ai import AsyncPortkey
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

logger = get_logger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for a replay run."""
    model: str  # Model in @provider/model format (e.g., @openai/gpt-4o-mini)
    provider: str  # Provider name for reference/logging
    temperature: float = 0.0  # Deterministic by default
    max_tokens: int = 4096  # Default max tokens, required by some providers (e.g., Anthropic)
    timeout_seconds: float = 120.0
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
    completion_text: str | None = None  # Stored for judge evaluation
    original_completion: str | None = None  # Original response for comparison


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
    Engine for replaying prompts through candidate models using Portkey SDK.
    
    Key properties:
    - Uses Portkey SDK with @provider/model format
    - Tracks real costs from Portkey's response metadata
    - Deterministic replay (temperature=0 by default)
    - Stores completions for AI judge comparison
    - Concurrent execution with configurable rate limiting
    """

    def __init__(
        self,
        session: AsyncSession,
        max_concurrent: int | None = None,
    ):
        self.session = session
        self.max_concurrent = max_concurrent or settings.replay_max_concurrent
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Single Portkey client - uses @provider/model format, no virtual keys needed
        self.portkey = AsyncPortkey(
            api_key=settings.portkey_api_key,
        )

    def _get_portkey_model_string(self, provider: str, model: str) -> str:
        """
        Convert provider/model to Portkey's @provider/model format.
        
        If model already starts with @, return as-is.
        Otherwise, construct @provider/model format.
        """
        if model.startswith("@"):
            return model
        return f"@{provider}/{model}"

    async def execute_replay_run(
        self,
        evaluation_run: EvaluationRun,
        sample_log_ids: list[uuid.UUID],
        models: list[dict[str, str]],  # [{"model": "...", "provider": "..."}]
        progress_callback: callable = None,
    ) -> list[ModelReplayResult]:
        """
        Execute a complete replay run for multiple models.
        
        Models should use @provider/model format (e.g., @openai/gpt-4o-mini).
        
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

        # Load sample prompts with original completions
        prompts = await self._load_prompts_with_completions(sample_log_ids)
        
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
            # Convert to @provider/model format if needed
            model_string = self._get_portkey_model_string(
                model_info["provider"], 
                model_info["model"]
            )
            
            config = ReplayConfig(
                model=model_string,
                provider=model_info["provider"],
                temperature=settings.replay_temperature,
                timeout_seconds=settings.replay_timeout_seconds,
            )
            
            # Create replay run record
            replay_run = ReplayRun(
                evaluation_run_id=evaluation_run.id,
                model=config.model,
                provider=config.provider,
                is_baseline=model_info.get("is_baseline", False),
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

                logger.info(
                    f"Replay completed for {config.provider}/{config.model}",
                    successful=model_result.successful,
                    failed=model_result.failed,
                    total_cost=model_result.total_cost_usd,
                )

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

    async def _load_prompts_with_completions(
        self,
        log_ids: list[uuid.UUID],
    ) -> list[dict[str, Any]]:
        """Load prompts and original completions from log entries."""
        result = await self.session.execute(
            select(LogEntry).where(LogEntry.id.in_(log_ids))
        )
        logs = result.scalars().all()
        
        prompts = []
        for log in logs:
            if not log.prompt and not log.request_data:
                continue  # Skip logs without prompt data
            
            # Extract original completion from response_data
            original_completion = None
            if log.response_data:
                choices = log.response_data.get("choices", [])
                if choices:
                    original_completion = choices[0].get("message", {}).get("content", "")
            
            # Get messages from request_data if available
            messages = None
            if log.request_data:
                messages = log.request_data.get("messages", [])
            
            prompts.append({
                "log_id": log.id,
                "prompt": log.prompt,
                "system_prompt": log.system_prompt,
                "messages": messages,  # Full message history if available
                "tools": log.tool_calls,
                "prompt_hash": log.prompt_hash,
                "original_completion": original_completion,
                "original_model": log.model,
                "original_provider": log.provider,
            })
        
        return prompts

    async def _replay_model(
        self,
        replay_run: ReplayRun,
        prompts: list[dict[str, Any]],
        config: ReplayConfig,
        progress_callback: callable = None,
    ) -> ModelReplayResult:
        """Replay all prompts through a single model using Portkey SDK."""
        
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
                completion_text=r.completion_text,  # Store for judge evaluation
                original_completion=r.original_completion,  # Store for comparison
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
        """Execute a single prompt replay using Portkey SDK with @provider/model format."""
        log_id = prompt_data["log_id"]
        prompt_hash = prompt_data["prompt_hash"]
        original_completion = prompt_data.get("original_completion")

        # Build messages
        messages = prompt_data.get("messages")
        if not messages:
            # Fallback to simple prompt format
            messages = []
            if prompt_data.get("system_prompt"):
                messages.append({"role": "system", "content": prompt_data["system_prompt"]})
            if prompt_data.get("prompt"):
                messages.append({"role": "user", "content": prompt_data["prompt"]})
        
        if not messages:
            return SingleReplayResult(
                original_log_id=log_id,
                prompt_hash=prompt_hash,
                success=False,
                error_message="No messages to replay",
                original_completion=original_completion,
            )

        start_time = time.time()
        
        try:
            # Build request kwargs - model is already in @provider/model format
            request_kwargs = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
            }
            
            # Always set max_tokens (required by some providers like Anthropic)
            request_kwargs["max_tokens"] = config.max_tokens
            
            # Add tools if present in original request
            tools = prompt_data.get("tools")
            if tools:
                request_kwargs["tools"] = tools
            
            # Execute through Portkey SDK with @provider/model format
            # Note: metadata is not included as it requires 'store' to be enabled for OpenAI
            response = await self.portkey.chat.completions.create(**request_kwargs)

            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response data from Portkey SDK response
            completion_text = ""
            if response.choices:
                message = response.choices[0].message
                completion_text = message.content or ""

            # Get usage from response
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

            # Get cost from Portkey response (if available in headers/metadata)
            # Portkey includes cost in the response when using their SDK
            cost_usd = 0.0
            if hasattr(response, '_raw_response'):
                raw = response._raw_response
                if hasattr(raw, 'headers'):
                    cost_header = raw.headers.get('x-portkey-cost')
                    if cost_header:
                        try:
                            cost_usd = float(cost_header)
                        except (ValueError, TypeError):
                            pass
            
            # Fallback: estimate cost from model pricing if not in response
            if cost_usd == 0.0:
                cost_usd = await self._estimate_cost(
                    config.provider, config.model, input_tokens, output_tokens
                )

            # Check for refusal
            refusal = self._detect_refusal(completion_text, response)

            # Hash completion for verification
            completion_hash = hashlib.sha256(completion_text.encode()).hexdigest()

            return SingleReplayResult(
                original_log_id=log_id,
                prompt_hash=prompt_hash,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                refusal=refusal,
                completion_hash=completion_hash,
                completion_text=completion_text,
                original_completion=original_completion,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Replay failed for log {log_id}: {e}")
            
            return SingleReplayResult(
                original_log_id=log_id,
                prompt_hash=prompt_hash,
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
                original_completion=original_completion,
            )

    async def _estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost from model catalog if not returned by Portkey."""
        from app.services.catalog.model_catalog_service import ModelCatalogService
        
        try:
            catalog = ModelCatalogService(self.session)
            
            # Clean up model name - remove @ prefix if present
            clean_model = model.lstrip('@')
            clean_provider = provider.lstrip('@')
            
            # Try to get model from catalog
            model_entry = await catalog.get_model(clean_provider, clean_model)
            
            # If not found, try parsing @provider/model format
            if not model_entry and '/' in clean_model:
                parts = clean_model.split('/', 1)
                if len(parts) == 2:
                    clean_provider = parts[0]
                    clean_model = parts[1]
                    model_entry = await catalog.get_model(clean_provider, clean_model)
            
            if model_entry and model_entry.input_price_per_token > 0:
                # Prices are in cents per token
                input_cost = (input_tokens * model_entry.input_price_per_token) / 100
                output_cost = (output_tokens * model_entry.output_price_per_token) / 100
                return input_cost + output_cost
            
            # Fallback to known model pricing if catalog doesn't have it
            cost = self._get_fallback_cost(clean_provider, clean_model, input_tokens, output_tokens)
            if cost > 0:
                return cost
                
        except Exception as e:
            logger.warning(f"Could not estimate cost: {e}")
        
        return 0.0
    
    def _get_fallback_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Fallback pricing for common models."""
        # Pricing per 1M tokens (in USD)
        PRICING = {
            # OpenAI
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "o1": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            # Anthropic
            "claude-opus-4-5": {"input": 15.00, "output": 75.00},
            "claude-sonnet-4": {"input": 3.00, "output": 15.00},
            "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
            # Google
            "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
            "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
        }
        
        # Try exact match
        model_lower = model.lower()
        for model_name, prices in PRICING.items():
            if model_name.lower() in model_lower or model_lower in model_name.lower():
                input_cost = (input_tokens / 1_000_000) * prices["input"]
                output_cost = (output_tokens / 1_000_000) * prices["output"]
                return input_cost + output_cost
        
        return 0.0

    def _detect_refusal(self, completion: str, response: Any) -> bool:
        """Detect if the model refused to respond."""
        if not completion:
            return False
        
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
            "I cannot help with",
            "I'm not allowed to",
        ]
        
        completion_lower = completion.lower()
        for pattern in refusal_patterns:
            if pattern.lower() in completion_lower:
                return True
        
        # Check response object for refusal flag (some models/providers return this)
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'finish_reason') and choice.finish_reason == 'content_filter':
                return True
            if hasattr(choice.message, 'refusal') and choice.message.refusal:
                return True
            
        return False


# Convenience function for creating replay engine
def get_replay_engine(session: AsyncSession) -> ReplayEngine:
    """Get a configured replay engine instance."""
    return ReplayEngine(session)
