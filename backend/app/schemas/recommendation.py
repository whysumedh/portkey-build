"""Pydantic schemas for Recommendation API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CandidateModelResponse(BaseModel):
    """Schema for candidate model response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    model: str
    provider: str
    is_current: bool
    rank: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_per_request: float
    success_rate: float
    refusal_rate: float
    quality_score: float
    correctness_score: float
    helpfulness_score: float
    safety_score: float
    overall_score: float
    trade_offs: dict | None
    selection_reason: str | None
    exclusion_reason: str | None
    created_at: datetime


class RecommendationResponse(BaseModel):
    """Schema for recommendation response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    project_id: UUID
    evaluation_run_id: UUID | None
    status: str
    recommended_model: str | None
    recommended_provider: str | None
    current_model: str | None
    current_provider: str | None
    confidence_score: float
    risk_level: str
    risk_notes: list[str]
    trade_off_summary: dict | None
    reasoning: str | None
    key_insights: list[str]
    example_improvements: list[dict]
    example_regressions: list[dict]
    judge_disagreements: list[dict]
    version: int
    created_at: datetime
    acknowledged: bool
    acknowledged_at: datetime | None
    acknowledged_by: str | None
    action_taken: str | None
    candidates: list[CandidateModelResponse]


class RecommendationListResponse(BaseModel):
    """Schema for paginated recommendation list."""
    items: list[RecommendationResponse]
    total: int
    page: int
    page_size: int
    pages: int


class RecommendationAcknowledge(BaseModel):
    """Schema for acknowledging a recommendation."""
    acknowledged_by: str
    action_taken: str | None = None
    notes: str | None = None
