"""LogEntry model - represents a single LLM request/response from Portkey."""

import hashlib
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


class LogStatus(str, Enum):
    """Status of a log entry."""
    SUCCESS = "success"
    ERROR = "error"
    REFUSED = "refused"
    TIMEOUT = "timeout"


class LogEntry(Base):
    """
    Immutable log entry from Portkey.
    
    Represents a single LLM request/response with all associated metadata.
    Logs are append-only and versioned for deterministic replay.
    """

    __tablename__ = "log_entries"
    __table_args__ = (
        Index("ix_log_entries_project_timestamp", "project_id", "timestamp"),
        Index("ix_log_entries_model", "model"),
        Index("ix_log_entries_status", "status"),
        Index("ix_log_entries_prompt_hash", "prompt_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # project_id is nullable to allow storing workspace-level logs (log pool)
    # Logs without a project_id are in the "pool" and can be assigned to projects later
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )

    # Portkey identifiers
    portkey_log_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    span_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Request data
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    endpoint: Mapped[str] = mapped_column(String(255), default="/chat/completions")
    
    # Prompt data (stored only if privacy allows)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Context and tool calls
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Full request/response data from Portkey (for detailed view)
    request_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Response data (completion not stored for deterministic replay)
    completion_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Model info
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Metrics
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    status: Mapped[str] = mapped_column(String(20), default=LogStatus.SUCCESS.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    refusal: Mapped[bool] = mapped_column(Boolean, default=False)

    # Portkey-specific metadata
    cache_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Custom metadata from Portkey
    log_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Versioning
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships (project is optional for workspace-level logs)
    project: Mapped["Project | None"] = relationship("Project", back_populates="log_entries")

    @staticmethod
    def compute_hash(content: str | list | dict | None) -> str:
        """
        Compute SHA-256 hash of content.
        
        Handles various content types:
        - str: hashes directly
        - list: converts to JSON string (for multi-modal messages)
        - dict: converts to JSON string
        - None: returns hash of empty string
        """
        import json
        
        if content is None:
            content_str = ""
        elif isinstance(content, str):
            content_str = content
        elif isinstance(content, (list, dict)):
            # Handle multi-modal content (list of content parts) or dict
            content_str = json.dumps(content, sort_keys=True, default=str)
        else:
            content_str = str(content)
        
        return hashlib.sha256(content_str.encode()).hexdigest()

    @classmethod
    def from_portkey_log(cls, project_id: uuid.UUID, portkey_data: dict) -> "LogEntry":
        """Create a LogEntry from Portkey log data."""
        prompt = portkey_data.get("request", {}).get("messages", [])
        prompt_str = str(prompt) if prompt else ""
        
        return cls(
            project_id=project_id,
            portkey_log_id=portkey_data.get("id", str(uuid.uuid4())),
            trace_id=portkey_data.get("trace_id"),
            span_id=portkey_data.get("span_id"),
            timestamp=datetime.fromisoformat(
                portkey_data.get("created_at", datetime.utcnow().isoformat())
            ),
            endpoint=portkey_data.get("endpoint", "/chat/completions"),
            prompt=prompt_str if portkey_data.get("store_prompt", True) else None,
            prompt_hash=cls.compute_hash(prompt_str),
            system_prompt=portkey_data.get("request", {}).get("system"),
            context=portkey_data.get("context"),
            tool_calls=portkey_data.get("request", {}).get("tools"),
            model=portkey_data.get("model", "unknown"),
            provider=portkey_data.get("provider", "unknown"),
            input_tokens=portkey_data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=portkey_data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=portkey_data.get("usage", {}).get("total_tokens", 0),
            latency_ms=portkey_data.get("response_time", 0),
            cost_usd=portkey_data.get("cost", 0.0),
            status=portkey_data.get("status", LogStatus.SUCCESS.value),
            error_message=portkey_data.get("error", {}).get("message"),
            error_code=portkey_data.get("error", {}).get("code"),
            refusal=portkey_data.get("refusal", False),
            cache_status=portkey_data.get("cache_status"),
            retry_count=portkey_data.get("retry_count", 0),
            fallback_used=portkey_data.get("fallback_used", False),
            log_metadata=portkey_data.get("metadata"),
        )

    def __repr__(self) -> str:
        return f"<LogEntry(id={self.id}, model='{self.model}', status='{self.status}')>"
