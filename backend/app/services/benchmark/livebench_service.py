"""
LiveBench Service - Fetches and manages LiveBench benchmark data.

LiveBench is a benchmark for LLMs that evaluates models across multiple dimensions:
- Reasoning: Logical reasoning and problem-solving
- Coding: Code generation, review, and debugging
- Agentic Coding: Autonomous coding with tool use
- Mathematics: Mathematical problem-solving
- Data Analysis: Data interpretation and analysis
- Language: Language understanding and generation
- Instruction Following (IF): Following complex instructions

This service fetches benchmark results and stores them in the model catalog
for use in model selection.

Reference: https://livebench.ai/ and https://github.com/LiveBench/LiveBench
"""

import httpx
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.model_catalog import ModelCatalogEntry

logger = get_logger(__name__)

# LiveBench leaderboard API/data URLs
LIVEBENCH_LEADERBOARD_URL = "https://livebench.ai/api/leaderboard"
LIVEBENCH_RESULTS_HF = "https://huggingface.co/datasets/livebench/model-outputs/resolve/main"


@dataclass
class LiveBenchScore:
    """Benchmark scores for a single model from LiveBench."""
    model_name: str
    provider: str | None
    reasoning: float | None
    coding: float | None
    agentic_coding: float | None
    mathematics: float | None
    data_analysis: float | None
    language: float | None
    instruction_following: float | None
    global_avg: float | None


# Static benchmark data based on LiveBench leaderboard
# This is updated periodically or can be refreshed via API
# Scores are on 0-100 scale
LIVEBENCH_STATIC_DATA: list[dict[str, Any]] = [
    # OpenAI Models
    {
        "model_name": "gpt-4o",
        "provider": "openai",
        "aliases": ["gpt-4o-2024-05-13", "gpt-4o-2024-08-06"],
        "reasoning": 72.5,
        "coding": 68.3,
        "agentic_coding": 65.0,
        "mathematics": 70.1,
        "data_analysis": 69.8,
        "language": 74.2,
        "instruction_following": 71.5,
        "global_avg": 70.2,
    },
    {
        "model_name": "gpt-4o-mini",
        "provider": "openai",
        "aliases": ["gpt-4o-mini-2024-07-18"],
        "reasoning": 58.2,
        "coding": 55.1,
        "agentic_coding": 48.5,
        "mathematics": 56.8,
        "data_analysis": 54.3,
        "language": 62.1,
        "instruction_following": 59.7,
        "global_avg": 56.4,
    },
    {
        "model_name": "gpt-4-turbo",
        "provider": "openai",
        "aliases": ["gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview"],
        "reasoning": 68.9,
        "coding": 64.2,
        "agentic_coding": 58.3,
        "mathematics": 66.5,
        "data_analysis": 65.1,
        "language": 70.8,
        "instruction_following": 68.2,
        "global_avg": 66.0,
    },
    {
        "model_name": "gpt-3.5-turbo",
        "provider": "openai",
        "aliases": ["gpt-3.5-turbo-0125"],
        "reasoning": 42.1,
        "coding": 38.5,
        "agentic_coding": 28.0,
        "mathematics": 40.2,
        "data_analysis": 39.8,
        "language": 48.3,
        "instruction_following": 45.1,
        "global_avg": 40.3,
    },
    {
        "model_name": "o1",
        "provider": "openai",
        "aliases": ["o1-2024-12-17", "o1-preview"],
        "reasoning": 85.2,
        "coding": 82.1,
        "agentic_coding": 78.5,
        "mathematics": 88.3,
        "data_analysis": 79.6,
        "language": 76.8,
        "instruction_following": 74.2,
        "global_avg": 80.7,
    },
    {
        "model_name": "o1-mini",
        "provider": "openai",
        "aliases": ["o1-mini-2024-09-12"],
        "reasoning": 75.8,
        "coding": 72.4,
        "agentic_coding": 68.2,
        "mathematics": 78.5,
        "data_analysis": 70.1,
        "language": 68.9,
        "instruction_following": 66.5,
        "global_avg": 71.5,
    },
    {
        "model_name": "o3-mini",
        "provider": "openai",
        "aliases": ["o3-mini-2025-01-31"],
        "reasoning": 82.1,
        "coding": 79.5,
        "agentic_coding": 76.8,
        "mathematics": 85.2,
        "data_analysis": 77.3,
        "language": 74.6,
        "instruction_following": 72.1,
        "global_avg": 78.2,
    },
    # Anthropic Models
    {
        "model_name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "aliases": ["claude-3.5-sonnet", "claude-3-5-sonnet"],
        "reasoning": 74.8,
        "coding": 76.2,
        "agentic_coding": 72.5,
        "mathematics": 71.3,
        "data_analysis": 72.8,
        "language": 78.5,
        "instruction_following": 75.2,
        "global_avg": 74.5,
    },
    {
        "model_name": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "aliases": ["claude-3.5-haiku", "claude-3-5-haiku"],
        "reasoning": 58.5,
        "coding": 60.1,
        "agentic_coding": 52.3,
        "mathematics": 55.8,
        "data_analysis": 56.2,
        "language": 64.3,
        "instruction_following": 61.8,
        "global_avg": 58.4,
    },
    {
        "model_name": "claude-3-opus-20240229",
        "provider": "anthropic",
        "aliases": ["claude-3-opus"],
        "reasoning": 72.1,
        "coding": 70.5,
        "agentic_coding": 65.8,
        "mathematics": 68.9,
        "data_analysis": 70.2,
        "language": 76.8,
        "instruction_following": 73.5,
        "global_avg": 71.1,
    },
    {
        "model_name": "claude-3-sonnet-20240229",
        "provider": "anthropic",
        "aliases": ["claude-3-sonnet"],
        "reasoning": 62.3,
        "coding": 58.9,
        "agentic_coding": 52.1,
        "mathematics": 58.5,
        "data_analysis": 60.1,
        "language": 66.8,
        "instruction_following": 64.2,
        "global_avg": 60.4,
    },
    {
        "model_name": "claude-3-haiku-20240307",
        "provider": "anthropic",
        "aliases": ["claude-3-haiku"],
        "reasoning": 48.2,
        "coding": 45.8,
        "agentic_coding": 38.5,
        "mathematics": 44.1,
        "data_analysis": 46.5,
        "language": 54.2,
        "instruction_following": 52.1,
        "global_avg": 47.1,
    },
    # Google Models (via Vertex AI Global)
    {
        "model_name": "gemini-2.5-flash",
        "provider": "vertex-global",
        "aliases": ["gemini-2.5-flash", "gemini-2-flash"],
        "reasoning": 72.5,
        "coding": 70.2,
        "agentic_coding": 68.8,
        "mathematics": 71.1,
        "data_analysis": 72.5,
        "language": 74.2,
        "instruction_following": 73.8,
        "global_avg": 71.9,
    },
    {
        "model_name": "gemini-2.5-pro",
        "provider": "vertex-global",
        "aliases": ["gemini-2.5-pro", "gemini-2-pro"],
        "reasoning": 76.8,
        "coding": 74.5,
        "agentic_coding": 72.2,
        "mathematics": 75.3,
        "data_analysis": 76.1,
        "language": 77.5,
        "instruction_following": 76.2,
        "global_avg": 75.5,
    },
    {
        "model_name": "gemini-1.5-pro-001",
        "provider": "vertex-global",
        "aliases": ["gemini-1.5-pro", "gemini-1.5-pro-latest"],
        "reasoning": 66.8,
        "coding": 62.5,
        "agentic_coding": 58.2,
        "mathematics": 64.3,
        "data_analysis": 68.1,
        "language": 71.5,
        "instruction_following": 67.2,
        "global_avg": 65.5,
    },
    {
        "model_name": "gemini-1.5-flash-001",
        "provider": "vertex-global",
        "aliases": ["gemini-1.5-flash", "gemini-1.5-flash-latest"],
        "reasoning": 56.2,
        "coding": 52.8,
        "agentic_coding": 48.5,
        "mathematics": 54.1,
        "data_analysis": 55.8,
        "language": 60.3,
        "instruction_following": 58.5,
        "global_avg": 55.2,
    },
    # Groq-hosted Models (Llama, Mixtral)
    {
        "model_name": "llama-3.3-70b-versatile",
        "provider": "groq",
        "aliases": ["llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct"],
        "reasoning": 62.5,
        "coding": 58.2,
        "agentic_coding": 52.8,
        "mathematics": 60.1,
        "data_analysis": 58.5,
        "language": 65.2,
        "instruction_following": 62.8,
        "global_avg": 60.0,
    },
    {
        "model_name": "llama-3.1-8b-instant",
        "provider": "groq",
        "aliases": ["llama-3.1-8b", "meta-llama/llama-3.1-8b-instruct"],
        "reasoning": 42.8,
        "coding": 38.5,
        "agentic_coding": 32.1,
        "mathematics": 40.2,
        "data_analysis": 39.5,
        "language": 48.1,
        "instruction_following": 45.8,
        "global_avg": 41.0,
    },
    {
        "model_name": "mixtral-8x7b-32768",
        "provider": "groq",
        "aliases": ["mixtral-8x7b", "mistralai/mixtral-8x7b-instruct"],
        "reasoning": 52.1,
        "coding": 48.5,
        "agentic_coding": 42.3,
        "mathematics": 50.8,
        "data_analysis": 49.2,
        "language": 56.5,
        "instruction_following": 54.1,
        "global_avg": 50.5,
    },
    # DeepSeek Models
    {
        "model_name": "deepseek-chat",
        "provider": "deepseek",
        "aliases": ["deepseek-v3", "deepseek-chat-v3"],
        "reasoning": 70.2,
        "coding": 72.8,
        "agentic_coding": 68.5,
        "mathematics": 74.1,
        "data_analysis": 68.3,
        "language": 66.5,
        "instruction_following": 64.8,
        "global_avg": 69.3,
    },
    {
        "model_name": "deepseek-reasoner",
        "provider": "deepseek",
        "aliases": ["deepseek-r1", "deepseek-reasoner-r1"],
        "reasoning": 82.5,
        "coding": 78.2,
        "agentic_coding": 74.8,
        "mathematics": 86.1,
        "data_analysis": 75.5,
        "language": 72.1,
        "instruction_following": 70.5,
        "global_avg": 77.1,
    },
]


class LiveBenchService:
    """
    Service for fetching and managing LiveBench benchmark data.
    
    This service:
    - Provides access to LiveBench benchmark scores
    - Maps model names to catalog entries
    - Updates the model catalog with benchmark data
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def get_static_benchmarks(self) -> list[LiveBenchScore]:
        """
        Get benchmark scores from static data.
        
        This is the fallback/default when live API is unavailable.
        """
        scores = []
        for data in LIVEBENCH_STATIC_DATA:
            scores.append(LiveBenchScore(
                model_name=data["model_name"],
                provider=data.get("provider"),
                reasoning=data.get("reasoning"),
                coding=data.get("coding"),
                agentic_coding=data.get("agentic_coding"),
                mathematics=data.get("mathematics"),
                data_analysis=data.get("data_analysis"),
                language=data.get("language"),
                instruction_following=data.get("instruction_following"),
                global_avg=data.get("global_avg"),
            ))
        return scores

    async def fetch_live_benchmarks(self) -> list[LiveBenchScore] | None:
        """
        Attempt to fetch live benchmark data from LiveBench API.
        
        Returns None if the API is unavailable.
        """
        try:
            client = await self._get_http_client()
            response = await client.get(LIVEBENCH_LEADERBOARD_URL)
            
            if response.status_code != 200:
                logger.warning(
                    "LiveBench API returned non-200 status",
                    status_code=response.status_code,
                )
                return None
            
            data = response.json()
            scores = []
            
            # Parse the leaderboard response
            for entry in data.get("models", []):
                scores.append(LiveBenchScore(
                    model_name=entry.get("model"),
                    provider=self._infer_provider(entry.get("model", "")),
                    reasoning=entry.get("reasoning"),
                    coding=entry.get("coding"),
                    agentic_coding=entry.get("agentic_coding"),
                    mathematics=entry.get("mathematics"),
                    data_analysis=entry.get("data_analysis"),
                    language=entry.get("language"),
                    instruction_following=entry.get("if"),  # IF = Instruction Following
                    global_avg=entry.get("global_avg"),
                ))
            
            return scores
            
        except Exception as e:
            logger.warning(f"Failed to fetch live benchmarks: {e}")
            return None

    def _infer_provider(self, model_name: str) -> str | None:
        """Infer provider from model name."""
        model_lower = model_name.lower()
        
        if "gpt" in model_lower or model_lower.startswith("o1") or model_lower.startswith("o3"):
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mixtral" in model_lower:
            return "groq"  # or meta, depending on hosting
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "mistral" in model_lower:
            return "mistral"
        
        return None

    def _find_benchmark_for_model(
        self,
        catalog_model: str,
        catalog_provider: str,
        benchmarks: list[LiveBenchScore],
    ) -> LiveBenchScore | None:
        """
        Find matching benchmark data for a catalog model.
        
        Handles variations in model naming between catalog and benchmarks.
        """
        # First try exact match
        for bench in benchmarks:
            if bench.model_name == catalog_model:
                if bench.provider is None or bench.provider == catalog_provider:
                    return bench
        
        # Try matching via aliases in static data
        for data in LIVEBENCH_STATIC_DATA:
            if data["provider"] != catalog_provider:
                continue
            
            aliases = [data["model_name"]] + data.get("aliases", [])
            if catalog_model in aliases:
                # Find the benchmark score for this model
                for bench in benchmarks:
                    if bench.model_name == data["model_name"]:
                        return bench
        
        # Try partial matching for common variations
        catalog_lower = catalog_model.lower()
        for bench in benchmarks:
            bench_lower = bench.model_name.lower()
            
            # Check if one contains the other
            if catalog_lower in bench_lower or bench_lower in catalog_lower:
                if bench.provider is None or bench.provider == catalog_provider:
                    return bench
        
        return None

    async def update_catalog_benchmarks(
        self,
        use_live_data: bool = False,
    ) -> dict[str, Any]:
        """
        Update model catalog with LiveBench benchmark scores.
        
        Args:
            use_live_data: Whether to try fetching live data first
            
        Returns:
            Dict with update statistics
        """
        logger.info("Starting catalog benchmark update")
        
        # Get benchmark data
        benchmarks = None
        if use_live_data:
            benchmarks = await self.fetch_live_benchmarks()
        
        if benchmarks is None:
            logger.info("Using static benchmark data")
            benchmarks = self.get_static_benchmarks()
        
        # Get all catalog models
        result = await self.session.execute(
            select(ModelCatalogEntry).where(ModelCatalogEntry.is_active == True)
        )
        catalog_models = result.scalars().all()
        
        updated = 0
        skipped = 0
        not_found = []
        
        now = datetime.now(timezone.utc)
        
        for model in catalog_models:
            # Find matching benchmark
            bench = self._find_benchmark_for_model(
                model.model,
                model.provider,
                benchmarks,
            )
            
            if bench is None:
                skipped += 1
                not_found.append(f"{model.provider}/{model.model}")
                continue
            
            # Update the model with benchmark scores
            model.livebench_reasoning = bench.reasoning
            model.livebench_coding = bench.coding
            model.livebench_agentic_coding = bench.agentic_coding
            model.livebench_mathematics = bench.mathematics
            model.livebench_data_analysis = bench.data_analysis
            model.livebench_language = bench.language
            model.livebench_instruction_following = bench.instruction_following
            model.livebench_global_avg = bench.global_avg
            model.livebench_last_updated = now
            
            updated += 1
        
        await self.session.commit()
        
        logger.info(
            "Catalog benchmark update complete",
            updated=updated,
            skipped=skipped,
        )
        
        return {
            "updated": updated,
            "skipped": skipped,
            "not_found": not_found[:20],  # Limit to first 20
            "timestamp": now.isoformat(),
        }

    async def get_model_benchmark(
        self,
        provider: str,
        model: str,
    ) -> LiveBenchScore | None:
        """
        Get benchmark scores for a specific model.
        
        First checks the catalog, then falls back to static data.
        """
        # Try catalog first
        result = await self.session.execute(
            select(ModelCatalogEntry)
            .where(ModelCatalogEntry.provider == provider)
            .where(ModelCatalogEntry.model == model)
        )
        catalog_model = result.scalar_one_or_none()
        
        if catalog_model and catalog_model.livebench_global_avg is not None:
            return LiveBenchScore(
                model_name=model,
                provider=provider,
                reasoning=catalog_model.livebench_reasoning,
                coding=catalog_model.livebench_coding,
                agentic_coding=catalog_model.livebench_agentic_coding,
                mathematics=catalog_model.livebench_mathematics,
                data_analysis=catalog_model.livebench_data_analysis,
                language=catalog_model.livebench_language,
                instruction_following=catalog_model.livebench_instruction_following,
                global_avg=catalog_model.livebench_global_avg,
            )
        
        # Fall back to static data
        benchmarks = self.get_static_benchmarks()
        return self._find_benchmark_for_model(model, provider, benchmarks)

    def get_benchmark_summary(self, score: LiveBenchScore) -> str:
        """
        Get a human-readable summary of benchmark scores.
        
        Useful for including in prompts or explanations.
        """
        parts = []
        
        if score.global_avg is not None:
            parts.append(f"Overall: {score.global_avg:.1f}")
        if score.reasoning is not None:
            parts.append(f"Reasoning: {score.reasoning:.1f}")
        if score.coding is not None:
            parts.append(f"Coding: {score.coding:.1f}")
        if score.agentic_coding is not None:
            parts.append(f"Agentic: {score.agentic_coding:.1f}")
        if score.mathematics is not None:
            parts.append(f"Math: {score.mathematics:.1f}")
        if score.data_analysis is not None:
            parts.append(f"Data: {score.data_analysis:.1f}")
        if score.language is not None:
            parts.append(f"Language: {score.language:.1f}")
        if score.instruction_following is not None:
            parts.append(f"IF: {score.instruction_following:.1f}")
        
        return " | ".join(parts)
