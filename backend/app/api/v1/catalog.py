"""
Model Catalog API endpoints.

Provides access to the LLM model catalog with pricing, capabilities, and LiveBench benchmarks.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.services.catalog.model_catalog_service import ModelCatalogService
from app.services.benchmark.livebench_service import LiveBenchService
from app.models.model_catalog import ModelCatalogEntry

logger = get_logger(__name__)
router = APIRouter(prefix="/catalog", tags=["Model Catalog"])


@router.get("/models")
async def list_models(
    provider: str | None = Query(None, description="Filter by provider"),
    active_only: bool = Query(True, description="Only return active models"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    List all models in the catalog.
    
    Returns models with their pricing and capabilities.
    """
    service = ModelCatalogService(db)
    models = await service.get_all_models(provider=provider, active_only=active_only)
    
    return {
        "models": [m.to_dict() for m in models],
        "total": len(models),
        "providers": list(set(m.provider for m in models)),
    }


@router.get("/models/{provider}/{model:path}")
async def get_model(
    provider: str,
    model: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get a specific model from the catalog.
    
    If not in local catalog, fetches from Portkey's public API.
    """
    service = ModelCatalogService(db)
    
    # Try local catalog first
    entry = await service.get_model(provider, model)
    
    if entry:
        return entry.to_dict()
    
    # Try fetching from Portkey
    pricing = await service.fetch_model_pricing(provider, model)
    
    if pricing:
        # Sync to local catalog
        entry = await service.sync_model(provider, model)
        await db.commit()
        if entry:
            return entry.to_dict()
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model {provider}/{model} not found",
    )


@router.post("/models/{provider}/{model:path}/sync")
async def sync_model(
    provider: str,
    model: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Sync a model's pricing from Portkey's public API.
    
    Creates or updates the model in the local catalog.
    """
    service = ModelCatalogService(db)
    
    entry = await service.sync_model(provider, model)
    await db.commit()
    
    if entry:
        return {
            "status": "synced",
            "model": entry.to_dict(),
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Could not fetch pricing for {provider}/{model}",
    )


@router.post("/initialize")
async def initialize_catalog(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Initialize the model catalog with default models.
    
    Fetches pricing from Portkey for common models and stores locally.
    This should be called once after deployment or when refreshing the catalog.
    """
    service = ModelCatalogService(db)
    
    stats = await service.initialize_catalog()
    
    return {
        "status": "initialized",
        "stats": stats,
    }


@router.post("/refresh")
async def refresh_pricing(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Refresh pricing for all models in the catalog.
    
    Updates prices from Portkey's public API.
    """
    service = ModelCatalogService(db)
    
    stats = await service.refresh_pricing()
    
    return {
        "status": "refreshed",
        "stats": stats,
    }


@router.get("/providers")
async def list_providers(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    List all available providers with model counts.
    """
    service = ModelCatalogService(db)
    models = await service.get_all_models(active_only=True)
    
    # Count models per provider
    provider_counts: dict[str, int] = {}
    for model in models:
        provider_counts[model.provider] = provider_counts.get(model.provider, 0) + 1
    
    return {
        "providers": [
            {"provider": p, "model_count": c}
            for p, c in sorted(provider_counts.items())
        ],
        "total_providers": len(provider_counts),
        "total_models": len(models),
    }


@router.get("/tiers")
async def list_tiers(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    List all model tiers with descriptions and model counts.
    
    Tiers help categorize models by cost/quality trade-off:
    - budget: Cheapest options for simple tasks
    - standard: Balanced price/performance
    - premium: High quality for complex tasks
    - enterprise: Top-tier for critical applications
    """
    service = ModelCatalogService(db)
    models = await service.get_all_models(active_only=True)
    
    tier_descriptions = {
        "budget": "Cheapest options, ideal for simple tasks and high-volume workloads",
        "standard": "Balanced price/performance for most production use cases",
        "premium": "High quality models for complex reasoning and generation",
        "enterprise": "Top-tier models for critical applications requiring best accuracy",
    }
    
    # Count and get models per tier
    tier_data: dict[str, dict] = {
        tier: {"description": desc, "models": [], "count": 0}
        for tier, desc in tier_descriptions.items()
    }
    
    for model in models:
        tier = model.tier or "unknown"
        if tier in tier_data:
            tier_data[tier]["models"].append({
                "provider": model.provider,
                "model": model.model,
                "display_name": model.display_name,
                "quality_score": model.quality_score,
                "speed_score": model.speed_score,
            })
            tier_data[tier]["count"] += 1
    
    return {
        "tiers": [
            {
                "tier": tier,
                "description": data["description"],
                "model_count": data["count"],
                "models": data["models"][:5],  # Top 5 per tier
            }
            for tier, data in tier_data.items()
            if data["count"] > 0
        ],
    }


@router.get("/tiers/{tier}")
async def get_models_by_tier(
    tier: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get all models in a specific tier.
    
    Valid tiers: budget, standard, premium, enterprise
    """
    valid_tiers = ["budget", "standard", "premium", "enterprise"]
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Must be one of: {valid_tiers}",
        )
    
    service = ModelCatalogService(db)
    models = await service.get_models_by_tier(tier)
    
    return {
        "tier": tier,
        "models": [m.to_dict() for m in models],
        "total": len(models),
    }


@router.get("/use-cases")
async def list_use_cases(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    List all use cases with descriptions and model counts.
    
    Use cases help identify which models excel at specific tasks.
    """
    use_case_descriptions = {
        "general": "General-purpose assistant tasks",
        "coding": "Code generation, review, and debugging",
        "reasoning": "Complex reasoning, math, and logic",
        "creative": "Creative writing and brainstorming",
        "fast": "Low-latency requirements",
        "multimodal": "Vision, audio, and multi-modal tasks",
        "agentic": "Tool use, function calling, and agents",
    }
    
    service = ModelCatalogService(db)
    models = await service.get_all_models(active_only=True)
    
    # Count models per use case
    use_case_counts: dict[str, list] = {uc: [] for uc in use_case_descriptions}
    
    for model in models:
        if model.use_cases:
            for uc in model.use_cases:
                if uc in use_case_counts:
                    use_case_counts[uc].append({
                        "provider": model.provider,
                        "model": model.model,
                        "tier": model.tier,
                        "quality_score": model.quality_score,
                    })
    
    return {
        "use_cases": [
            {
                "use_case": uc,
                "description": use_case_descriptions.get(uc, ""),
                "model_count": len(models_list),
                "top_models": sorted(models_list, key=lambda x: -(x.get("quality_score") or 0))[:5],
            }
            for uc, models_list in use_case_counts.items()
            if models_list
        ],
    }


@router.get("/use-cases/{use_case}")
async def get_models_by_use_case(
    use_case: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get all models that excel at a specific use case.
    
    Valid use cases: general, coding, reasoning, creative, fast, multimodal, agentic
    """
    valid_use_cases = ["general", "coding", "reasoning", "creative", "fast", "multimodal", "agentic"]
    if use_case not in valid_use_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid use case. Must be one of: {valid_use_cases}",
        )
    
    service = ModelCatalogService(db)
    models = await service.get_models_for_use_case(use_case)
    
    return {
        "use_case": use_case,
        "models": [m.to_dict() for m in models],
        "total": len(models),
    }


@router.get("/recommend")
async def get_recommendations(
    agent_purpose: str | None = Query(None, description="What the agent does"),
    cost_sensitivity: str = Query("medium", description="low, medium, or high"),
    min_quality: int | None = Query(None, ge=1, le=10, description="Minimum quality score (1-10)"),
    min_speed: int | None = Query(None, ge=1, le=10, description="Minimum speed score (1-10)"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get model recommendations based on project requirements.
    
    This endpoint infers use cases from the agent purpose and returns
    suitable models sorted by relevance.
    """
    service = ModelCatalogService(db)
    
    # Infer use cases from purpose
    use_cases = service.infer_use_cases_from_project(
        agent_purpose=agent_purpose,
        success_criteria=None,
    )
    
    # Get tier recommendations
    preferred_tiers = service.get_tier_recommendations(cost_sensitivity)
    
    # Get recommendations
    models = await service.get_models_for_recommendation(
        use_cases=use_cases,
        preferred_tiers=preferred_tiers,
        min_quality_score=min_quality,
        min_speed_score=min_speed,
    )
    
    return {
        "inferred_use_cases": use_cases,
        "preferred_tiers": preferred_tiers,
        "recommendations": [m.to_dict() for m in models[:10]],  # Top 10
        "total_matches": len(models),
    }


# =============================================================================
# LiveBench Benchmark Endpoints
# =============================================================================

@router.post("/benchmarks/refresh")
async def refresh_livebench_benchmarks(
    use_live_data: bool = Query(False, description="Try to fetch live data from LiveBench API"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Refresh LiveBench benchmark scores for all models in the catalog.
    
    This updates the livebench_* columns for each model based on their
    benchmark performance across dimensions:
    - Reasoning
    - Coding
    - Agentic Coding
    - Mathematics
    - Data Analysis
    - Language
    - Instruction Following (IF)
    
    By default, uses static benchmark data. Set use_live_data=true to attempt
    fetching from the LiveBench API (may fail if API is unavailable).
    """
    service = LiveBenchService(db)
    
    try:
        stats = await service.update_catalog_benchmarks(use_live_data=use_live_data)
        return {
            "status": "success",
            "stats": stats,
        }
    finally:
        await service.close()


@router.get("/benchmarks/models/{provider}/{model:path}")
async def get_model_benchmarks(
    provider: str,
    model: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get LiveBench benchmark scores for a specific model.
    
    Returns scores for each dimension (0-100 scale):
    - reasoning: Logical reasoning and problem-solving
    - coding: Code generation, review, debugging
    - agentic_coding: Autonomous coding with tool use
    - mathematics: Mathematical problem-solving
    - data_analysis: Data interpretation and analysis
    - language: Language understanding and generation
    - instruction_following: Following complex instructions
    - global_avg: Overall average score
    """
    service = LiveBenchService(db)
    
    try:
        benchmark = await service.get_model_benchmark(provider, model)
        
        if not benchmark:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No benchmark data found for {provider}/{model}",
            )
        
        return {
            "provider": provider,
            "model": model,
            "benchmarks": {
                "reasoning": benchmark.reasoning,
                "coding": benchmark.coding,
                "agentic_coding": benchmark.agentic_coding,
                "mathematics": benchmark.mathematics,
                "data_analysis": benchmark.data_analysis,
                "language": benchmark.language,
                "instruction_following": benchmark.instruction_following,
                "global_avg": benchmark.global_avg,
            },
            "summary": service.get_benchmark_summary(benchmark),
        }
    finally:
        await service.close()


@router.get("/benchmarks/leaderboard")
async def get_benchmark_leaderboard(
    dimension: str = Query("global_avg", description="Dimension to sort by"),
    min_score: float | None = Query(None, ge=0, le=100, description="Minimum score filter"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get a leaderboard of models sorted by benchmark score.
    
    Available dimensions:
    - global_avg (default): Overall average
    - reasoning
    - coding
    - agentic_coding
    - mathematics
    - data_analysis
    - language
    - instruction_following
    """
    valid_dimensions = [
        "global_avg", "reasoning", "coding", "agentic_coding",
        "mathematics", "data_analysis", "language", "instruction_following"
    ]
    
    if dimension not in valid_dimensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dimension. Must be one of: {valid_dimensions}",
        )
    
    # Get all models with benchmark data
    catalog_service = ModelCatalogService(db)
    models = await catalog_service.get_all_models(active_only=True)
    
    # Filter to models with benchmark data and build leaderboard
    leaderboard = []
    score_column = f"livebench_{dimension}"
    
    for model in models:
        score = getattr(model, score_column, None)
        if score is None:
            continue
        if min_score is not None and score < min_score:
            continue
        
        leaderboard.append({
            "provider": model.provider,
            "model": model.model,
            "display_name": model.display_name or model.model,
            "tier": model.tier,
            "score": score,
            "global_avg": model.livebench_global_avg,
            "scores": model.get_livebench_scores(),
        })
    
    # Sort by score descending
    leaderboard.sort(key=lambda x: -(x["score"] or 0))
    
    return {
        "dimension": dimension,
        "min_score": min_score,
        "leaderboard": leaderboard,
        "total": len(leaderboard),
    }
