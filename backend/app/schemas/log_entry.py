"""Pydantic schemas for Log Entry API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class LogEntryResponse(BaseModel):
    """Schema for log entry response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    project_id: UUID
    portkey_log_id: str
    trace_id: str | None
    span_id: str | None
    timestamp: datetime
    endpoint: str
    prompt_hash: str
    # Note: prompt may be None if privacy mode is enabled
    prompt: str | None
    system_prompt: str | None
    context: dict | None
    tool_calls: list | None
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error_message: str | None
    error_code: str | None
    refusal: bool
    cache_status: str | None
    retry_count: int
    fallback_used: bool
    metadata: dict | None
    ingested_at: datetime


class LogEntryListResponse(BaseModel):
    """Schema for paginated log entry list."""
    items: list[LogEntryResponse]
    total: int
    page: int
    page_size: int
    pages: int


class LogSyncRequest(BaseModel):
    """Schema for requesting log sync from Portkey."""
    start_date: datetime | None = None
    end_date: datetime | None = None
    limit: int = Field(default=1000, ge=1, le=10000)
    include_prompt: bool = Field(
        default=True, 
        description="Whether to store raw prompts (disable for privacy)"
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional Portkey filters (e.g., model, status)"
    )


class LogSyncResponse(BaseModel):
    """Schema for log sync response."""
    project_id: UUID
    logs_synced: int
    logs_skipped: int
    sync_start: datetime
    sync_end: datetime
    oldest_log: datetime | None
    newest_log: datetime | None
    errors: list[str] = Field(default_factory=list)


class LogStatsResponse(BaseModel):
    """Schema for log statistics."""
    project_id: UUID
    total_logs: int
    date_range_start: datetime | None
    date_range_end: datetime | None
    models_used: list[str]
    providers_used: list[str]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float
    success_rate: float
    refusal_rate: float
    error_rate: float
