"""Pydantic schemas for Project API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class SuccessCriteriaCreate(BaseModel):
    """Schema for creating success criteria."""
    min_accuracy: float = Field(default=0.8, ge=0, le=1)
    min_quality_score: float = Field(default=0.7, ge=0, le=1)
    max_latency_ms: float = Field(default=5000, gt=0)
    max_latency_p95_ms: float = Field(default=10000, gt=0)
    max_cost_per_request_usd: float = Field(default=0.10, ge=0)
    max_monthly_cost_usd: float | None = None
    max_refusal_rate: float = Field(default=0.05, ge=0, le=1)
    max_safety_violations: int = Field(default=0, ge=0)
    safety_categories_blocked: list[str] = Field(default_factory=list)


class SuccessCriteriaResponse(SuccessCriteriaCreate):
    """Schema for success criteria response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    project_id: UUID
    created_at: datetime
    updated_at: datetime


class ToleranceLevelsCreate(BaseModel):
    """Schema for creating tolerance levels."""
    cost_sensitivity: Literal["low", "medium", "high"] = "medium"
    latency_tolerance_pct: float = Field(default=0.20, ge=0, le=1)
    cost_tolerance_pct: float = Field(default=0.10, ge=0, le=1)
    quality_tolerance_pct: float = Field(default=0.05, ge=0, le=1)
    absolute_max_latency_ms: float = Field(default=30000, gt=0)
    absolute_max_refusal_rate: float = Field(default=0.10, ge=0, le=1)


class ToleranceLevelsResponse(ToleranceLevelsCreate):
    """Schema for tolerance levels response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    project_id: UUID
    created_at: datetime
    updated_at: datetime


class ProjectCreate(BaseModel):
    """Schema for creating a project."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    agent_purpose: str = Field(..., min_length=1)
    portkey_virtual_key: str | None = None
    portkey_config_id: str | None = None
    current_model: str | None = None
    current_provider: str | None = None
    success_criteria: SuccessCriteriaCreate | None = None
    tolerance_levels: ToleranceLevelsCreate | None = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    agent_purpose: str | None = None
    portkey_virtual_key: str | None = None
    portkey_config_id: str | None = None
    current_model: str | None = None
    current_provider: str | None = None
    is_active: bool | None = None
    success_criteria: SuccessCriteriaCreate | None = None
    tolerance_levels: ToleranceLevelsCreate | None = None


class ProjectResponse(BaseModel):
    """Schema for project response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    description: str | None
    agent_purpose: str
    portkey_virtual_key: str | None
    portkey_config_id: str | None
    current_model: str | None
    current_provider: str | None
    is_active: bool
    last_log_sync: datetime | None
    last_evaluation: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime
    success_criteria: SuccessCriteriaResponse | None
    tolerance_levels: ToleranceLevelsResponse | None


class ProjectListResponse(BaseModel):
    """Schema for paginated project list."""
    items: list[ProjectResponse]
    total: int
    page: int
    page_size: int
    pages: int
