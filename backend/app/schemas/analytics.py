"""Pydantic schemas for Analytics API."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


class DistributionParams(BaseModel):
    """Parameters for distribution analysis."""
    field: str
    bins: int = Field(default=20, ge=5, le=100)
    normalize: bool = False


class PercentileParams(BaseModel):
    """Parameters for percentile analysis."""
    metric: str
    percentiles: list[float] = Field(default=[50, 75, 90, 95, 99])
    
    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, v: list[float]) -> list[float]:
        for p in v:
            if not 0 <= p <= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {p}")
        return v


class CorrelationParams(BaseModel):
    """Parameters for correlation analysis."""
    x: str
    y: str
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class AggregationParams(BaseModel):
    """Parameters for aggregation analysis."""
    metric: str
    group_by: str | None = None
    aggregations: list[Literal["count", "sum", "mean", "min", "max", "std"]] = Field(
        default=["count", "mean", "std"]
    )


class ClusteringParams(BaseModel):
    """Parameters for clustering analysis."""
    features: list[str]
    n_clusters: int = Field(default=5, ge=2, le=20)
    method: Literal["kmeans", "hierarchical"] = "kmeans"


class SampleParams(BaseModel):
    """Parameters for sampling."""
    n: int = Field(default=100, ge=1, le=10000)
    stratify_by: str | None = None
    random_seed: int = Field(default=42)


class AnalysisRequest(BaseModel):
    """
    Structured analysis request for the analytics engine.
    
    This is the ONLY interface for requesting statistical analysis.
    LLMs submit these structured requests instead of arbitrary code.
    """
    project_id: UUID
    type: Literal["distribution", "percentile", "correlation", "aggregation", "clustering", "sample"]
    params: dict[str, Any]
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    filters: dict[str, Any] | None = None

    @field_validator("params")
    @classmethod
    def validate_params_for_type(cls, v: dict, info) -> dict:
        """Validate params match the analysis type."""
        # Note: Full validation happens in the analytics engine
        return v


class DistributionResult(BaseModel):
    """Result of a distribution analysis."""
    field: str
    bins: list[float]
    counts: list[int]
    percentages: list[float] | None = None
    total_count: int
    missing_count: int
    statistics: dict[str, float]  # min, max, mean, std, median


class PercentileResult(BaseModel):
    """Result of a percentile analysis."""
    metric: str
    percentiles: dict[float, float]  # percentile -> value
    count: int
    missing_count: int


class CorrelationResult(BaseModel):
    """Result of a correlation analysis."""
    x: str
    y: str
    method: str
    coefficient: float
    p_value: float
    sample_size: int
    interpretation: str  # "strong positive", "weak negative", etc.


class AggregationResult(BaseModel):
    """Result of an aggregation analysis."""
    metric: str
    group_by: str | None
    results: list[dict[str, Any]]
    total_count: int


class ClusteringResult(BaseModel):
    """Result of a clustering analysis."""
    features: list[str]
    n_clusters: int
    cluster_sizes: list[int]
    cluster_centers: list[dict[str, float]]
    inertia: float | None = None


class SampleResult(BaseModel):
    """Result of a sampling operation."""
    sample_size: int
    stratified: bool
    strata_counts: dict[str, int] | None = None
    sample_ids: list[UUID]


class AnalysisResponse(BaseModel):
    """Response from an analysis request."""
    model_config = ConfigDict(from_attributes=True)
    
    request_id: UUID
    project_id: UUID
    analysis_type: str
    status: Literal["completed", "failed", "cached"]
    cached: bool = False
    cache_key: str | None = None
    execution_time_ms: float
    result: (
        DistributionResult 
        | PercentileResult 
        | CorrelationResult 
        | AggregationResult 
        | ClusteringResult 
        | SampleResult 
        | None
    )
    error: str | None = None
    created_at: datetime
