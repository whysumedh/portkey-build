"""
Model Catalog Service.

Fetches and caches LLM model information from Portkey's open-source pricing database.
https://portkey.ai/docs/product/model-catalog/portkey-models
"""

import httpx
from typing import Any
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.model_catalog import ModelCatalogEntry

logger = get_logger(__name__)

# Portkey's public pricing API (no auth required)
PORTKEY_PRICING_API = "https://api.portkey.ai/model-configs/pricing"
PORTKEY_PROVIDER_PRICING_API = "https://configs.portkey.ai/pricing"

# Model tiers for cost/quality classification
# - budget: Cheapest options, good for simple tasks
# - standard: Balanced price/performance for most use cases
# - premium: High quality, best for complex tasks
# - enterprise: Top-tier models for critical applications

# Use cases that models excel at
# - general: General-purpose assistant tasks
# - coding: Code generation, review, debugging
# - reasoning: Complex reasoning, math, logic
# - creative: Creative writing, brainstorming
# - fast: Low-latency requirements
# - multimodal: Vision, audio, multi-modal tasks
# - agentic: Tool use, function calling, agents

# Default models to pre-populate the catalog with tier and use case data
DEFAULT_MODELS = [
    # ============== OPENAI ==============
    # Premium Tier
    {"provider": "openai", "model": "gpt-4o", "display_name": "GPT-4o", "model_family": "gpt-4", 
     "context_window": 128000, "supports_vision": True, "supports_function_calling": True, "supports_json_mode": True,
     "tier": "premium", "use_cases": ["general", "coding", "reasoning", "multimodal", "agentic"],
     "quality_score": 9, "speed_score": 7, "recommended_for": "Best all-around model for complex tasks requiring vision and reasoning"},
    
    # Standard Tier
    {"provider": "openai", "model": "gpt-4o-mini", "display_name": "GPT-4o Mini", "model_family": "gpt-4", 
     "context_window": 128000, "supports_vision": True, "supports_function_calling": True, "supports_json_mode": True,
     "tier": "standard", "use_cases": ["general", "coding", "fast", "agentic"],
     "quality_score": 7, "speed_score": 8, "recommended_for": "Cost-effective choice for most tasks, great balance of speed and quality"},
    
    {"provider": "openai", "model": "gpt-4-turbo", "display_name": "GPT-4 Turbo", "model_family": "gpt-4", 
     "context_window": 128000, "supports_vision": True, "supports_function_calling": True, "supports_json_mode": True,
     "tier": "premium", "use_cases": ["general", "coding", "reasoning", "multimodal"],
     "quality_score": 9, "speed_score": 6, "recommended_for": "High quality for complex reasoning, slightly slower than GPT-4o"},
    
    # Budget Tier
    {"provider": "openai", "model": "gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo", "model_family": "gpt-3.5", 
     "context_window": 16385, "supports_function_calling": True, "supports_json_mode": True,
     "tier": "budget", "use_cases": ["general", "fast"],
     "quality_score": 5, "speed_score": 9, "recommended_for": "Budget-friendly for simple tasks, fast responses"},
    
    # Enterprise Tier (Reasoning models)
    {"provider": "openai", "model": "o1", "display_name": "O1", "model_family": "o1", 
     "context_window": 200000,
     "tier": "enterprise", "use_cases": ["reasoning", "coding", "creative"],
     "quality_score": 10, "speed_score": 3, "recommended_for": "Best for complex reasoning, math, and scientific problems"},
    
    {"provider": "openai", "model": "o1-mini", "display_name": "O1 Mini", "model_family": "o1", 
     "context_window": 128000,
     "tier": "premium", "use_cases": ["reasoning", "coding"],
     "quality_score": 8, "speed_score": 5, "recommended_for": "Strong reasoning at lower cost than O1"},
    
    {"provider": "openai", "model": "o3-mini", "display_name": "O3 Mini", "model_family": "o3", 
     "context_window": 200000,
     "tier": "premium", "use_cases": ["reasoning", "coding"],
     "quality_score": 9, "speed_score": 5, "recommended_for": "Latest reasoning model, excellent for complex problems"},
    
    # ============== ANTHROPIC ==============
    # Premium Tier
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "model_family": "claude-3.5", 
     "context_window": 200000, "supports_vision": True, "supports_function_calling": True,
     "tier": "premium", "use_cases": ["general", "coding", "reasoning", "creative", "agentic"],
     "quality_score": 9, "speed_score": 7, "recommended_for": "Excellent for coding, analysis, and agentic workflows"},
    
    # Standard Tier
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku", "model_family": "claude-3.5", 
     "context_window": 200000, "supports_vision": True, "supports_function_calling": True,
     "tier": "standard", "use_cases": ["general", "fast", "agentic"],
     "quality_score": 7, "speed_score": 9, "recommended_for": "Fast and capable for everyday tasks, great for agents"},
    
    # Enterprise Tier
    {"provider": "anthropic", "model": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "model_family": "claude-3", 
     "context_window": 200000, "supports_vision": True, "supports_function_calling": True,
     "tier": "enterprise", "use_cases": ["reasoning", "creative", "general"],
     "quality_score": 10, "speed_score": 4, "recommended_for": "Top quality for nuanced reasoning and creative tasks"},
    
    {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "display_name": "Claude 3 Sonnet", "model_family": "claude-3", 
     "context_window": 200000, "supports_vision": True, "supports_function_calling": True,
     "tier": "standard", "use_cases": ["general", "coding", "creative"],
     "quality_score": 7, "speed_score": 7, "recommended_for": "Balanced model for general use"},
    
    # Budget Tier
    {"provider": "anthropic", "model": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku", "model_family": "claude-3", 
     "context_window": 200000, "supports_vision": True, "supports_function_calling": True,
     "tier": "budget", "use_cases": ["general", "fast"],
     "quality_score": 6, "speed_score": 9, "recommended_for": "Fastest Claude, great for high-volume simple tasks"},
    
    # ============== GOOGLE (via Vertex AI Global) ==============
    {"provider": "vertex-global", "model": "gemini-2.5-flash", "display_name": "Gemini 2.5 Flash", "model_family": "gemini-2.5", 
     "context_window": 1048576, "supports_vision": True, "supports_function_calling": True,
     "tier": "standard", "use_cases": ["general", "fast", "multimodal", "agentic"],
     "quality_score": 9, "speed_score": 9, "recommended_for": "Latest fast Gemini model with excellent quality"},
    
    {"provider": "vertex-global", "model": "gemini-2.5-pro", "display_name": "Gemini 2.5 Pro", "model_family": "gemini-2.5", 
     "context_window": 2097152, "supports_vision": True, "supports_function_calling": True,
     "tier": "premium", "use_cases": ["general", "reasoning", "multimodal", "agentic"],
     "quality_score": 9, "speed_score": 7, "recommended_for": "Top-tier Gemini with exceptional reasoning"},
    
    {"provider": "vertex-global", "model": "gemini-1.5-pro-001", "display_name": "Gemini 1.5 Pro", "model_family": "gemini-1.5", 
     "context_window": 2097152, "supports_vision": True, "supports_function_calling": True,
     "tier": "premium", "use_cases": ["general", "reasoning", "multimodal"],
     "quality_score": 8, "speed_score": 6, "recommended_for": "Largest context window available, great for document analysis"},
    
    {"provider": "vertex-global", "model": "gemini-1.5-flash-001", "display_name": "Gemini 1.5 Flash", "model_family": "gemini-1.5", 
     "context_window": 1048576, "supports_vision": True, "supports_function_calling": True,
     "tier": "standard", "use_cases": ["general", "fast", "multimodal"],
     "quality_score": 7, "speed_score": 8, "recommended_for": "Fast and capable with large context"},
    
    
    
    # ============== X.AI ==============
    {"provider": "x-ai", "model": "grok-2-latest", "display_name": "Grok 2", "model_family": "grok-2", 
     "context_window": 131072, "supports_vision": True, "supports_function_calling": True,
     "tier": "premium", "use_cases": ["general", "creative", "reasoning", "multimodal"],
     "quality_score": 8, "speed_score": 7, "recommended_for": "Creative and conversational, real-time knowledge"},
    
    {"provider": "x-ai", "model": "grok-3-beta", "display_name": "Grok Beta", "model_family": "grok", 
     "context_window": 131072,
     "tier": "standard", "use_cases": ["general", "creative"],
     "quality_score": 7, "speed_score": 7, "recommended_for": "Conversational AI with personality"},
]


class ModelCatalogService:
    """
    Service for managing the LLM model catalog.
    
    Fetches pricing from Portkey's public API and caches in the database.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()
    
    async def fetch_model_pricing(self, provider: str, model: str) -> dict[str, Any] | None:
        """
        Fetch pricing for a specific model from Portkey's public API.
        
        API: GET https://api.portkey.ai/model-configs/pricing/{provider}/{model}
        No authentication required.
        
        Returns pricing config or None if not found.
        """
        url = f"{PORTKEY_PRICING_API}/{provider}/{model}"
        
        try:
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"Model not found in Portkey catalog: {provider}/{model}")
                return None
            else:
                logger.warning(
                    f"Portkey pricing API error: {response.status_code}",
                    url=url,
                )
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch model pricing: {e}", provider=provider, model=model)
            return None
    
    async def fetch_provider_models(self, provider: str) -> dict[str, Any] | None:
        """
        Fetch all models for a provider from Portkey's public API.
        
        API: GET https://configs.portkey.ai/pricing/{provider}.json
        No authentication required.
        
        Returns dict of model_id -> pricing_config or None if error.
        """
        url = f"{PORTKEY_PROVIDER_PRICING_API}/{provider}.json"
        
        try:
            response = await self.http_client.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Portkey provider pricing API error: {response.status_code}",
                    provider=provider,
                )
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch provider models: {e}", provider=provider)
            return None
    
    def _extract_pricing(self, pricing_config: dict) -> tuple[float, float]:
        """
        Extract input/output token prices from Portkey pricing config.
        
        Prices are in USD cents per token.
        """
        pay_as_you_go = pricing_config.get("pay_as_you_go", {})
        
        input_price = pay_as_you_go.get("request_token", {}).get("price", 0.0)
        output_price = pay_as_you_go.get("response_token", {}).get("price", 0.0)
        
        return float(input_price), float(output_price)
    
    async def sync_model(
        self,
        provider: str,
        model: str,
        defaults: dict[str, Any] | None = None,
    ) -> ModelCatalogEntry | None:
        """
        Sync a single model's pricing from Portkey API to database.
        
        Creates or updates the model catalog entry.
        """
        # Fetch pricing from Portkey
        pricing_config = await self.fetch_model_pricing(provider, model)
        
        if not pricing_config and not defaults:
            logger.warning(f"No pricing found for {provider}/{model}")
            return None
        
        # Extract pricing
        if pricing_config:
            input_price, output_price = self._extract_pricing(pricing_config)
            cache_read = pricing_config.get("pay_as_you_go", {}).get("cache_read_input_token", {}).get("price")
            cache_write = pricing_config.get("pay_as_you_go", {}).get("cache_write_input_token", {}).get("price")
        else:
            input_price, output_price = 0.0, 0.0
            cache_read, cache_write = None, None
        
        # Check if model exists
        result = await self.session.execute(
            select(ModelCatalogEntry).where(
                ModelCatalogEntry.provider == provider,
                ModelCatalogEntry.model == model,
            )
        )
        entry = result.scalar_one_or_none()
        
        if entry:
            # Update existing entry
            entry.input_price_per_token = input_price
            entry.output_price_per_token = output_price
            entry.cache_read_price_per_token = cache_read
            entry.cache_write_price_per_token = cache_write
            entry.pricing_config = pricing_config
            entry.updated_at = datetime.now(timezone.utc)
            
            # Update defaults if provided
            if defaults:
                for key, value in defaults.items():
                    if hasattr(entry, key) and value is not None:
                        setattr(entry, key, value)
        else:
            # Create new entry
            entry = ModelCatalogEntry(
                provider=provider,
                model=model,
                input_price_per_token=input_price,
                output_price_per_token=output_price,
                cache_read_price_per_token=cache_read,
                cache_write_price_per_token=cache_write,
                pricing_config=pricing_config,
                model_type="chat",
                **(defaults or {}),
            )
            self.session.add(entry)
        
        await self.session.flush()
        logger.info(f"Synced model: {provider}/{model}", input_price=input_price, output_price=output_price)
        
        return entry
    
    async def initialize_catalog(self) -> dict[str, int]:
        """
        Initialize the model catalog with default models.
        
        Fetches pricing from Portkey for each default model.
        Returns stats on models synced.
        """
        stats = {"synced": 0, "failed": 0, "total": len(DEFAULT_MODELS)}
        
        logger.info("Initializing model catalog", total_models=stats["total"])
        
        for model_config in DEFAULT_MODELS:
            provider = model_config.pop("provider")
            model = model_config.pop("model")
            
            try:
                entry = await self.sync_model(provider, model, defaults=model_config)
                if entry:
                    stats["synced"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to sync {provider}/{model}: {e}")
                stats["failed"] += 1
            
            # Re-add provider/model for next iteration reference
            model_config["provider"] = provider
            model_config["model"] = model
        
        await self.session.commit()
        logger.info("Model catalog initialized", **stats)
        
        return stats
    
    async def get_all_models(
        self,
        provider: str | None = None,
        active_only: bool = True,
    ) -> list[ModelCatalogEntry]:
        """
        Get all models from the catalog.
        
        Args:
            provider: Filter by provider
            active_only: Only return active models
        """
        query = select(ModelCatalogEntry)
        
        if provider:
            query = query.where(ModelCatalogEntry.provider == provider)
        
        if active_only:
            query = query.where(ModelCatalogEntry.is_active == True)
        
        query = query.order_by(ModelCatalogEntry.provider, ModelCatalogEntry.model)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_model(self, provider: str, model: str) -> ModelCatalogEntry | None:
        """Get a specific model from the catalog."""
        result = await self.session.execute(
            select(ModelCatalogEntry).where(
                ModelCatalogEntry.provider == provider,
                ModelCatalogEntry.model == model,
            )
        )
        return result.scalar_one_or_none()
    
    async def get_models_for_recommendation(
        self,
        current_provider: str | None = None,
        current_model: str | None = None,
        include_same_family: bool = True,
        use_cases: list[str] | None = None,
        preferred_tiers: list[str] | None = None,
        min_quality_score: int | None = None,
        min_speed_score: int | None = None,
    ) -> list[ModelCatalogEntry]:
        """
        Get models suitable for recommendation based on project requirements.
        
        Args:
            current_provider: The project's current provider
            current_model: The project's current model
            include_same_family: Include models from the same family
            use_cases: Filter by use cases (e.g., ["coding", "reasoning"])
            preferred_tiers: Preferred price tiers (e.g., ["budget", "standard"])
            min_quality_score: Minimum quality score (1-10)
            min_speed_score: Minimum speed score (1-10)
        
        Returns:
            List of models suitable for recommendation, sorted by relevance.
        """
        models = await self.get_all_models(active_only=True)
        
        # Filter by use cases if specified
        if use_cases:
            filtered = []
            for model in models:
                if model.use_cases:
                    # Check if model supports any of the requested use cases
                    if any(uc in model.use_cases for uc in use_cases):
                        filtered.append(model)
                else:
                    # Include models without use case data (they might still be relevant)
                    filtered.append(model)
            models = filtered
        
        # Filter by tier if specified
        if preferred_tiers:
            filtered = []
            for model in models:
                if model.tier and model.tier in preferred_tiers:
                    filtered.append(model)
                elif not model.tier:
                    # Include models without tier data
                    filtered.append(model)
            models = filtered
        
        # Filter by quality score
        if min_quality_score:
            models = [m for m in models if not m.quality_score or m.quality_score >= min_quality_score]
        
        # Filter by speed score
        if min_speed_score:
            models = [m for m in models if not m.speed_score or m.speed_score >= min_speed_score]
        
        # Score and sort models
        def score_model(model: ModelCatalogEntry) -> tuple:
            """Score model for sorting. Lower score = higher priority."""
            score_parts = []
            
            # Priority 1: Match use cases (more matches = better)
            if use_cases and model.use_cases:
                matching_use_cases = len(set(use_cases) & set(model.use_cases))
                score_parts.append(-matching_use_cases)  # Negative so more matches = lower score
            else:
                score_parts.append(0)
            
            # Priority 2: Match preferred tier
            if preferred_tiers and model.tier:
                tier_priority = preferred_tiers.index(model.tier) if model.tier in preferred_tiers else 10
                score_parts.append(tier_priority)
            else:
                score_parts.append(5)
            
            # Priority 3: Quality score (higher = better)
            score_parts.append(-(model.quality_score or 5))
            
            # Priority 4: Same family as current model
            if include_same_family and current_model:
                current = next((m for m in models if m.model == current_model), None)
                if current and current.model_family and model.model_family == current.model_family:
                    score_parts.append(0)
                else:
                    score_parts.append(1)
            else:
                score_parts.append(1)
            
            return tuple(score_parts)
        
        models.sort(key=score_model)
        
        return models
    
    async def get_models_by_tier(self, tier: str) -> list[ModelCatalogEntry]:
        """Get all models of a specific tier."""
        result = await self.session.execute(
            select(ModelCatalogEntry)
            .where(
                ModelCatalogEntry.tier == tier,
                ModelCatalogEntry.is_active == True,
            )
            .order_by(ModelCatalogEntry.quality_score.desc())
        )
        return list(result.scalars().all())
    
    async def get_models_for_use_case(self, use_case: str) -> list[ModelCatalogEntry]:
        """Get models that excel at a specific use case."""
        # Query all active models and filter in Python (JSON containment varies by DB)
        models = await self.get_all_models(active_only=True)
        
        matching = []
        for model in models:
            if model.use_cases and use_case in model.use_cases:
                matching.append(model)
        
        # Sort by quality score
        matching.sort(key=lambda m: -(m.quality_score or 0))
        
        return matching
    
    def get_tier_recommendations(self, cost_sensitivity: str) -> list[str]:
        """
        Get recommended tiers based on cost sensitivity.
        
        Args:
            cost_sensitivity: "low", "medium", or "high"
            
        Returns:
            List of recommended tiers in priority order.
        """
        if cost_sensitivity == "high":
            return ["budget", "standard"]
        elif cost_sensitivity == "low":
            return ["premium", "enterprise", "standard"]
        else:  # medium
            return ["standard", "budget", "premium"]
    
    def infer_use_cases_from_project(
        self,
        agent_purpose: str | None,
        success_criteria: dict | None,
    ) -> list[str]:
        """
        Infer appropriate use cases from project configuration.
        
        Analyzes the agent's purpose and success criteria to determine
        what capabilities are needed.
        """
        use_cases = []
        
        if not agent_purpose:
            return ["general"]
        
        purpose_lower = agent_purpose.lower()
        
        # Infer from keywords in agent purpose
        keyword_mappings = {
            "coding": ["code", "programming", "developer", "software", "debug", "refactor"],
            "reasoning": ["reasoning", "math", "logic", "analysis", "problem", "calculate"],
            "creative": ["creative", "writing", "story", "content", "marketing", "copy"],
            "fast": ["fast", "quick", "realtime", "latency", "speed", "instant"],
            "multimodal": ["image", "vision", "picture", "visual", "photo", "screenshot"],
            "agentic": ["agent", "tool", "function", "action", "automation", "workflow"],
        }
        
        for use_case, keywords in keyword_mappings.items():
            if any(kw in purpose_lower for kw in keywords):
                use_cases.append(use_case)
        
        # Check success criteria for hints
        if success_criteria:
            accuracy = success_criteria.get("accuracy_target") or success_criteria.get("min_accuracy") or 0
            if accuracy > 0.95:
                if "reasoning" not in use_cases:
                    use_cases.append("reasoning")
            
            latency = success_criteria.get("latency_target_ms") or success_criteria.get("max_latency_ms") or 10000
            if latency < 2000:
                if "fast" not in use_cases:
                    use_cases.append("fast")
        
        # Default to general if nothing specific found
        if not use_cases:
            use_cases = ["general"]
        
        return use_cases
    
    async def refresh_pricing(self) -> dict[str, int]:
        """
        Refresh pricing for all models in the catalog from Portkey API.
        """
        models = await self.get_all_models(active_only=False)
        stats = {"updated": 0, "failed": 0, "total": len(models)}
        
        for model in models:
            try:
                pricing = await self.fetch_model_pricing(model.provider, model.model)
                if pricing:
                    input_price, output_price = self._extract_pricing(pricing)
                    model.input_price_per_token = input_price
                    model.output_price_per_token = output_price
                    model.pricing_config = pricing
                    model.updated_at = datetime.now(timezone.utc)
                    stats["updated"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to refresh {model.provider}/{model.model}: {e}")
                stats["failed"] += 1
        
        await self.session.commit()
        logger.info("Pricing refresh complete", **stats)
        
        return stats
