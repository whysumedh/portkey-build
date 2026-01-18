"""Project model - represents a single AI agent being optimized."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.log_entry import LogEntry
    from app.models.evaluation import EvaluationRun, AggregatedMetrics
    from app.models.recommendation import Recommendation


class CostSensitivity(str, Enum):
    """Cost sensitivity levels for project optimization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SuccessCriteria(Base):
    """Success criteria for a project - what defines good performance."""

    __tablename__ = "success_criteria"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )

    # Quality metrics
    min_accuracy: Mapped[float] = mapped_column(Float, default=0.8)
    min_quality_score: Mapped[float] = mapped_column(Float, default=0.7)

    # Performance metrics
    max_latency_ms: Mapped[float] = mapped_column(Float, default=5000.0)
    max_latency_p95_ms: Mapped[float] = mapped_column(Float, default=10000.0)

    # Cost metrics
    max_cost_per_request_usd: Mapped[float] = mapped_column(Float, default=1.00)
    max_monthly_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Safety metrics
    max_refusal_rate: Mapped[float] = mapped_column(Float, default=0.05)
    max_safety_violations: Mapped[int] = mapped_column(Integer, default=0)
    safety_categories_blocked: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ToleranceLevels(Base):
    """Tolerance levels for project constraints - how much deviation is acceptable."""

    __tablename__ = "tolerance_levels"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )

    # How sensitive is this project to cost increases?
    cost_sensitivity: Mapped[str] = mapped_column(
        String(20), default=CostSensitivity.MEDIUM.value
    )

    # Acceptable deviation from targets (as percentages)
    latency_tolerance_pct: Mapped[float] = mapped_column(Float, default=0.20)
    cost_tolerance_pct: Mapped[float] = mapped_column(Float, default=0.10)
    quality_tolerance_pct: Mapped[float] = mapped_column(Float, default=0.05)

    # Hard limits that cannot be exceeded
    absolute_max_latency_ms: Mapped[float] = mapped_column(Float, default=30000.0)
    absolute_max_refusal_rate: Mapped[float] = mapped_column(Float, default=0.10)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CapabilityExpectations(Base):
    """
    LiveBench-aligned capability expectations for model selection.
    
    Users specify minimum benchmark scores (0-100) for each capability dimension.
    Models that don't meet these minimums will be filtered out during selection.
    
    Dimensions align with LiveBench benchmarks:
    - reasoning: Logical reasoning and problem-solving
    - coding: Code generation, review, and debugging
    - agentic_coding: Autonomous coding with tool use
    - mathematics: Mathematical problem-solving
    - data_analysis: Data interpretation and analysis
    - language: Language understanding and generation
    - instruction_following: Following complex instructions
    """

    __tablename__ = "capability_expectations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )

    # LiveBench-aligned dimensions (scores 0-100, minimum expectations)
    reasoning: Mapped[float | None] = mapped_column(Float, nullable=True)
    coding: Mapped[float | None] = mapped_column(Float, nullable=True)
    agentic_coding: Mapped[float | None] = mapped_column(Float, nullable=True)
    mathematics: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_analysis: Mapped[float | None] = mapped_column(Float, nullable=True)
    language: Mapped[float | None] = mapped_column(Float, nullable=True)
    instruction_following: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def has_any_expectation(self) -> bool:
        """Check if any capability expectation is set."""
        return any([
            self.reasoning is not None,
            self.coding is not None,
            self.agentic_coding is not None,
            self.mathematics is not None,
            self.data_analysis is not None,
            self.language is not None,
            self.instruction_following is not None,
        ])

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary for API responses and prompt building."""
        return {
            "reasoning": self.reasoning,
            "coding": self.coding,
            "agentic_coding": self.agentic_coding,
            "mathematics": self.mathematics,
            "data_analysis": self.data_analysis,
            "language": self.language,
            "instruction_following": self.instruction_following,
        }


class Project(Base):
    """
    A Project represents exactly ONE AI agent being optimized.
    
    Each project:
    - Has a clear purpose and success criteria
    - Ingests production logs from that agent
    - Receives model recommendations based on analysis
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_purpose: Mapped[str] = mapped_column(Text, nullable=False)

    # Portkey integration
    portkey_virtual_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    portkey_config_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Selected Portkey log IDs (from log exports)
    selected_log_ids: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    
    # Log filter criteria (for automatic log association)
    log_filter_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Current model configuration
    current_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_provider: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(default=True)
    last_log_sync: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_evaluation: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    success_criteria: Mapped["SuccessCriteria | None"] = relationship(
        "SuccessCriteria",
        foreign_keys="SuccessCriteria.project_id",
        primaryjoin="Project.id == SuccessCriteria.project_id",
        uselist=False,
        lazy="joined",
    )
    tolerance_levels: Mapped["ToleranceLevels | None"] = relationship(
        "ToleranceLevels",
        foreign_keys="ToleranceLevels.project_id",
        primaryjoin="Project.id == ToleranceLevels.project_id",
        uselist=False,
        lazy="joined",
    )
    capability_expectations: Mapped["CapabilityExpectations | None"] = relationship(
        "CapabilityExpectations",
        foreign_keys="CapabilityExpectations.project_id",
        primaryjoin="Project.id == CapabilityExpectations.project_id",
        uselist=False,
        lazy="joined",
    )
    log_entries: Mapped[list["LogEntry"]] = relationship(
        "LogEntry", back_populates="project", lazy="dynamic"
    )
    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun", back_populates="project", lazy="dynamic"
    )
    aggregated_metrics: Mapped[list["AggregatedMetrics"]] = relationship(
        "AggregatedMetrics", back_populates="project", lazy="dynamic"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        "Recommendation", back_populates="project", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}')>"
