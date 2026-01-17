"""Recommendation models - model recommendations and candidate comparisons."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
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
    from app.models.evaluation import EvaluationRun


class RecommendationStatus(str, Enum):
    """Status of a recommendation."""
    RECOMMENDED = "recommended"
    NO_RECOMMENDATION = "no_recommendation"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_DATA = "insufficient_data"


class CandidateModel(Base):
    """
    A candidate model evaluated for a recommendation.
    
    Contains all metrics and trade-offs for comparison.
    """

    __tablename__ = "candidate_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendations.id", ondelete="CASCADE"), nullable=False
    )

    # Model info
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    is_current: Mapped[bool] = mapped_column(default=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    # Performance metrics
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cost_per_request: Mapped[float] = mapped_column(Float, default=0.0)
    success_rate: Mapped[float] = mapped_column(Float, default=1.0)
    refusal_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Quality metrics
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    correctness_score: Mapped[float] = mapped_column(Float, default=0.0)
    helpfulness_score: Mapped[float] = mapped_column(Float, default=0.0)
    safety_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Composite score
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Trade-offs vs current/baseline
    trade_offs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    """
    Example trade_offs:
    {
        "cost_change_pct": -15.5,
        "latency_change_pct": 10.2,
        "quality_change_pct": -2.1,
        "pros": ["Lower cost", "Faster response"],
        "cons": ["Slightly lower quality"]
    }
    """

    # Why this model was selected/rejected
    selection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    exclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    recommendation: Mapped["Recommendation"] = relationship(
        "Recommendation", back_populates="candidates"
    )

    def __repr__(self) -> str:
        return f"<CandidateModel(model='{self.model}', rank={self.rank})>"


class Recommendation(Base):
    """
    A model recommendation for a project.
    
    Recommendations are:
    - Advisory only (no auto-switching)
    - Confidence-gated
    - Fully explainable
    """

    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_recommendations_project_created", "project_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    evaluation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=True
    )

    # Recommendation status
    status: Mapped[str] = mapped_column(
        String(30), default=RecommendationStatus.NO_RECOMMENDATION.value
    )

    # Recommended model (if any)
    recommended_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recommended_provider: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Current model for comparison
    current_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    current_provider: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Confidence and risk
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(20), default="medium")
    risk_notes: Mapped[list] = mapped_column(JSON, default=list)
    """
    Example risk_notes:
    [
        "Limited sample size (n=50)",
        "High judge disagreement on edge cases",
        "New model with less production history"
    ]
    """

    # Summary trade-offs
    trade_off_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    """
    Example trade_off_summary:
    {
        "cost": {"current": 0.05, "recommended": 0.03, "change_pct": -40},
        "latency": {"current": 500, "recommended": 600, "change_pct": 20},
        "quality": {"current": 0.85, "recommended": 0.82, "change_pct": -3.5}
    }
    """

    # Explainability
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_insights: Mapped[list] = mapped_column(JSON, default=list)
    """
    Example key_insights:
    [
        "Recommended model shows 40% cost reduction with minimal quality impact",
        "Current model has high refusal rate (8%) on customer support queries",
        "Recommended model handles longer prompts better (correlation: 0.72)"
    ]
    """

    # Example cases for explainability
    example_improvements: Mapped[list] = mapped_column(JSON, default=list)
    example_regressions: Mapped[list] = mapped_column(JSON, default=list)
    judge_disagreements: Mapped[list] = mapped_column(JSON, default=list)

    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # User actions
    acknowledged: Mapped[bool] = mapped_column(default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    action_taken: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="recommendations")
    evaluation_run: Mapped["EvaluationRun | None"] = relationship(
        "EvaluationRun", back_populates="recommendation"
    )
    candidates: Mapped[list["CandidateModel"]] = relationship(
        "CandidateModel", back_populates="recommendation", lazy="joined",
        order_by="CandidateModel.rank"
    )

    def __repr__(self) -> str:
        return f"<Recommendation(id={self.id}, status='{self.status}', confidence={self.confidence_score})>"
