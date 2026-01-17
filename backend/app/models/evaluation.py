"""Evaluation models - replay runs, judge results, and aggregated metrics."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.recommendation import Recommendation


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JudgeType(str, Enum):
    """Types of AI judges."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    TASK_SUCCESS = "task_success"
    REFUSAL = "refusal"


class TimeWindow(str, Enum):
    """Time windows for aggregation."""
    DAY_7 = "7d"
    DAY_30 = "30d"
    DAY_90 = "90d"


class EvaluationRun(Base):
    """
    A complete evaluation run for a project.
    
    An evaluation run:
    - Samples historical prompts
    - Replays them on candidate models
    - Runs AI judges on the outputs
    - Produces aggregated metrics
    """

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_project_status", "project_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    # Run configuration
    trigger_type: Mapped[str] = mapped_column(String(50), default="scheduled")
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_models: Mapped[list] = mapped_column(JSON, nullable=False)
    
    # Time range of logs analyzed
    logs_start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    logs_end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_logs_analyzed: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    status: Mapped[str] = mapped_column(String(20), default=EvaluationStatus.PENDING.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Progress tracking
    replays_completed: Mapped[int] = mapped_column(Integer, default=0)
    replays_total: Mapped[int] = mapped_column(Integer, default=0)
    judgments_completed: Mapped[int] = mapped_column(Integer, default=0)
    judgments_total: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="evaluation_runs")
    replay_runs: Mapped[list["ReplayRun"]] = relationship(
        "ReplayRun", back_populates="evaluation_run", lazy="dynamic"
    )
    recommendation: Mapped["Recommendation | None"] = relationship(
        "Recommendation", back_populates="evaluation_run", uselist=False
    )

    def __repr__(self) -> str:
        return f"<EvaluationRun(id={self.id}, status='{self.status}')>"


class ReplayRun(Base):
    """
    A single model replay within an evaluation run.
    
    Replays the same prompts through a specific model to compare performance.
    """

    __tablename__ = "replay_runs"
    __table_args__ = (
        Index("ix_replay_runs_evaluation_model", "evaluation_run_id", "model"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )

    # Model being tested
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Configuration
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    replay_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), default=EvaluationStatus.PENDING.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Aggregated metrics
    total_prompts: Mapped[int] = mapped_column(Integer, default=0)
    successful_completions: Mapped[int] = mapped_column(Integer, default=0)
    failed_completions: Mapped[int] = mapped_column(Integer, default=0)
    refusals: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p50_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p99_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    evaluation_run: Mapped["EvaluationRun"] = relationship(
        "EvaluationRun", back_populates="replay_runs"
    )
    results: Mapped[list["ReplayResult"]] = relationship(
        "ReplayResult", back_populates="replay_run", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<ReplayRun(id={self.id}, model='{self.model}')>"


class ReplayResult(Base):
    """
    Result of replaying a single prompt through a model.
    
    Contains performance metrics but NOT the actual completion
    (for deterministic replay, we only store inputs).
    """

    __tablename__ = "replay_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    replay_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("replay_runs.id", ondelete="CASCADE"), nullable=False
    )
    
    # Reference to original log
    original_log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("log_entries.id"), nullable=False
    )
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Metrics
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    refusal: Mapped[bool] = mapped_column(Boolean, default=False)

    # Completion hash for verification (not the actual completion)
    completion_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    replay_run: Mapped["ReplayRun"] = relationship("ReplayRun", back_populates="results")
    judge_results: Mapped[list["JudgeResult"]] = relationship(
        "JudgeResult", back_populates="replay_result", lazy="joined"
    )

    def __repr__(self) -> str:
        return f"<ReplayResult(id={self.id}, success={self.success})>"


class JudgeResult(Base):
    """
    Result from a single AI judge evaluating a replay result.
    
    Each judge has a narrow responsibility and outputs structured scores.
    """

    __tablename__ = "judge_results"
    __table_args__ = (
        Index("ix_judge_results_type_score", "judge_type", "score"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    replay_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("replay_results.id", ondelete="CASCADE"), nullable=False
    )

    # Judge info
    judge_type: Mapped[str] = mapped_column(String(50), nullable=False)
    judge_version: Mapped[int] = mapped_column(Integer, default=1)
    judge_prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Scores
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Explanation
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # For tracking disagreements
    agrees_with_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    replay_result: Mapped["ReplayResult"] = relationship(
        "ReplayResult", back_populates="judge_results"
    )

    def __repr__(self) -> str:
        return f"<JudgeResult(id={self.id}, type='{self.judge_type}', score={self.score})>"


class AggregatedMetrics(Base):
    """
    Aggregated metrics for a project-model combination over a time window.
    
    Used for trend analysis, drift detection, and recommendations.
    """

    __tablename__ = "aggregated_metrics"
    __table_args__ = (
        Index("ix_aggregated_metrics_project_model_window", "project_id", "model", "time_window"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    # Model info
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)

    # Time window
    time_window: Mapped[str] = mapped_column(String(10), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Volume metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    refusals: Mapped[int] = mapped_column(Integer, default=0)

    # Latency metrics
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p50_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p99_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    latency_std_dev: Mapped[float] = mapped_column(Float, default=0.0)

    # Cost metrics
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cost_per_request: Mapped[float] = mapped_column(Float, default=0.0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Quality metrics (from evaluations)
    avg_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_correctness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_helpfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_safety_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Derived metrics
    success_rate: Mapped[float] = mapped_column(Float, default=1.0)
    refusal_rate: Mapped[float] = mapped_column(Float, default=0.0)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Variance and stability
    quality_variance: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_disagreement_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Drift detection
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    drift_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="aggregated_metrics")

    def __repr__(self) -> str:
        return f"<AggregatedMetrics(project={self.project_id}, model='{self.model}', window='{self.time_window}')>"
