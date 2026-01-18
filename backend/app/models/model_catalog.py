"""
Model Catalog database model.

Stores LLM model information including pricing, capabilities, and parameters.
Data sourced from Portkey's open-source pricing database.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import String, Float, Boolean, JSON, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelCatalogEntry(Base):
    """
    Catalog entry for an LLM model.
    
    Stores pricing, capabilities, and configuration for models
    across all supported providers.
    """
    __tablename__ = "model_catalog"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    
    # Model identification
    provider: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    
    # Pricing (in USD cents per token, as per Portkey API)
    input_price_per_token: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    output_price_per_token: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_read_price_per_token: Mapped[float | None] = mapped_column(Float, nullable=True)
    cache_write_price_per_token: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Context and limits
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Capabilities
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_function_calling: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=True)
    supports_json_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Model type and category
    model_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # chat, completion, embedding, image
    model_family: Mapped[str | None] = mapped_column(String(100), nullable=True)  # gpt-4, claude-3, etc.
    
    # Tier classification (for cost/quality trade-off recommendations)
    tier: Mapped[str | None] = mapped_column(String(20), nullable=True)  # budget, standard, premium, enterprise
    
    # Use case strengths (JSON array of use cases this model excels at)
    # e.g., ["general", "coding", "reasoning", "creative", "fast", "multimodal"]
    use_cases: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Quality/speed characteristics (1-10 scale)
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Overall quality rating
    speed_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Latency/speed rating
    
    # LiveBench benchmark scores (0-100 scale)
    # These scores are populated from the LiveBench leaderboard
    livebench_reasoning: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_coding: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_agentic_coding: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_mathematics: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_data_analysis: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_language: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_instruction_following: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_global_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    livebench_last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Recommended for specific scenarios
    recommended_for: Mapped[str | None] = mapped_column(Text, nullable=True)  # Human-readable recommendation
    
    # Full pricing config from Portkey (for reference)
    pricing_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Model parameters schema
    parameters: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    
    # Unique constraint on provider + model
    __table_args__ = (
        {"extend_existing": True},
    )
    
    def __repr__(self) -> str:
        return f"<ModelCatalogEntry {self.provider}/{self.model}>"
    
    @property
    def full_name(self) -> str:
        """Get full model identifier."""
        return f"{self.provider}/{self.model}"
    
    @property
    def cost_per_1k_input(self) -> float:
        """Get cost per 1K input tokens in USD."""
        return (self.input_price_per_token * 1000) / 100  # Convert cents to dollars
    
    @property
    def cost_per_1k_output(self) -> float:
        """Get cost per 1K output tokens in USD."""
        return (self.output_price_per_token * 1000) / 100  # Convert cents to dollars
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "provider": self.provider,
            "model": self.model,
            "display_name": self.display_name or self.model,
            "input_price_per_token": self.input_price_per_token,
            "output_price_per_token": self.output_price_per_token,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_vision": self.supports_vision,
            "supports_function_calling": self.supports_function_calling,
            "supports_streaming": self.supports_streaming,
            "supports_json_mode": self.supports_json_mode,
            "model_type": self.model_type,
            "model_family": self.model_family,
            "tier": self.tier,
            "use_cases": self.use_cases,
            "quality_score": self.quality_score,
            "speed_score": self.speed_score,
            "livebench_reasoning": self.livebench_reasoning,
            "livebench_coding": self.livebench_coding,
            "livebench_agentic_coding": self.livebench_agentic_coding,
            "livebench_mathematics": self.livebench_mathematics,
            "livebench_data_analysis": self.livebench_data_analysis,
            "livebench_language": self.livebench_language,
            "livebench_instruction_following": self.livebench_instruction_following,
            "livebench_global_avg": self.livebench_global_avg,
            "recommended_for": self.recommended_for,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
        }

    def get_livebench_scores(self) -> dict[str, float | None]:
        """Get all LiveBench benchmark scores as a dictionary."""
        return {
            "reasoning": self.livebench_reasoning,
            "coding": self.livebench_coding,
            "agentic_coding": self.livebench_agentic_coding,
            "mathematics": self.livebench_mathematics,
            "data_analysis": self.livebench_data_analysis,
            "language": self.livebench_language,
            "instruction_following": self.livebench_instruction_following,
            "global_avg": self.livebench_global_avg,
        }

    def has_livebench_scores(self) -> bool:
        """Check if this model has any LiveBench benchmark scores."""
        return self.livebench_global_avg is not None

    def meets_expectations(self, expectations: dict[str, float | None]) -> tuple[bool, list[str]]:
        """
        Check if this model meets the given capability expectations.
        
        Args:
            expectations: Dict of dimension -> minimum score (0-100)
            
        Returns:
            Tuple of (meets_all, list of failed dimensions)
        """
        failed = []
        dimension_mapping = {
            "reasoning": self.livebench_reasoning,
            "coding": self.livebench_coding,
            "agentic_coding": self.livebench_agentic_coding,
            "mathematics": self.livebench_mathematics,
            "data_analysis": self.livebench_data_analysis,
            "language": self.livebench_language,
            "instruction_following": self.livebench_instruction_following,
        }
        
        for dimension, min_score in expectations.items():
            if min_score is None:
                continue
            model_score = dimension_mapping.get(dimension)
            if model_score is None or model_score < min_score:
                failed.append(dimension)
        
        return len(failed) == 0, failed
