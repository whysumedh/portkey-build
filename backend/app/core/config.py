"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Agent Model Optimization Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_optimizer"
    )
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Portkey
    portkey_api_key: str = Field(default="")
    portkey_base_url: str = "https://api.portkey.ai"
    
    # Model Selector Agent (uses Claude 3.7 Sonnet via Portkey for intelligent model selection)
    # Uses @provider/model format - no virtual keys needed
    model_selector_model: str = "@bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    model_selector_max_candidates: int = 3  # Max candidate models to select
    
    # Replay Engine Settings
    replay_max_concurrent: int = 5  # Max concurrent replay requests
    replay_timeout_seconds: float = 120.0  # Timeout per replay request
    replay_temperature: float = 0.0  # Deterministic replay by default
    
    # AI Judge Settings (uses @provider/model format)
    judge_model: str = "@openai/gpt-4o"  # Strong model for quality evaluation
    judge_max_concurrent: int = 3  # Max concurrent judge evaluations

    # Evaluation
    default_replay_sample_size: int = 100
    max_replay_sample_size: int = 1000
    confidence_threshold: float = 0.7
    judge_disagreement_threshold: float = 0.3

    # Scheduler
    evaluation_schedule_cron: str = "0 2 * * 0"  # Weekly at 2 AM on Sunday
    enable_scheduler: bool = True

    # Security
    secret_key: str = Field(default="change-me-in-production-use-a-long-random-string")
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def async_database_url(self) -> str:
        """Get async database URL."""
        return str(self.database_url)

    @property
    def sync_database_url(self) -> str:
        """Get sync database URL for Alembic migrations."""
        return str(self.database_url).replace("+asyncpg", "")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
