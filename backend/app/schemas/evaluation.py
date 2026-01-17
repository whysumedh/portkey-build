"""Pydantic schemas for Evaluation API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class JudgeResultResponse(BaseModel):
    """Schema for judge result response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    judge_type: str
    judge_version: int
    score: float
    confidence: float
    rationale: str | None
    details: dict | None
    agrees_with_baseline: bool | None
    created_at: datetime


class ReplayResultResponse(BaseModel):
    """Schema for replay result response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    original_log_id: UUID
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error_message: str | None
    refusal: bool
    judge_results: list[JudgeResultResponse]
    created_at: datetime


class ReplayRunResponse(BaseModel):
    """Schema for replay run response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    evaluation_run_id: UUID
    model: str
    provider: str
    is_baseline: bool
    status: str
    error_message: str | None
    total_prompts: int
    successful_completions: int
    failed_completions: int
    refusals: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class EvaluationRunCreate(BaseModel):
    """Schema for creating an evaluation run."""
    project_id: UUID
    trigger_type: str = Field(default="manual")
    sample_size: int = Field(default=100, ge=10, le=1000)
    candidate_models: list[str] = Field(
        ..., 
        min_length=1,
        description="List of model identifiers to evaluate"
    )
    time_range_days: int = Field(
        default=30, 
        ge=1, 
        le=90,
        description="Number of days of logs to analyze"
    )


class EvaluationRunResponse(BaseModel):
    """Schema for evaluation run response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    project_id: UUID
    trigger_type: str
    sample_size: int
    candidate_models: list[str]
    logs_start_date: datetime
    logs_end_date: datetime
    total_logs_analyzed: int
    status: str
    error_message: str | None
    replays_completed: int
    replays_total: int
    judgments_completed: int
    judgments_total: int
    started_at: datetime | None
    completed_at: datetime | None
    version: int
    created_at: datetime
    replay_runs: list[ReplayRunResponse] | None = None


class EvaluationRunListResponse(BaseModel):
    """Schema for paginated evaluation run list."""
    items: list[EvaluationRunResponse]
    total: int
    page: int
    page_size: int
    pages: int


class EvaluationProgress(BaseModel):
    """Schema for evaluation progress updates."""
    evaluation_run_id: UUID
    status: str
    replays_completed: int
    replays_total: int
    judgments_completed: int
    judgments_total: int
    current_model: str | None = None
    estimated_completion: datetime | None = None
