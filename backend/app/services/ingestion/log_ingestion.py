"""Log ingestion service for syncing logs from Portkey."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.project import Project
from app.services.ingestion.portkey_client import PortkeyClient, get_portkey_client

logger = get_logger(__name__)


class LogIngestionError(Exception):
    """Error during log ingestion."""
    pass


class LogIngestionService:
    """
    Service for ingesting logs from Portkey into the local database.
    
    Features:
    - Incremental sync (only new logs)
    - Deduplication via portkey_log_id
    - Privacy mode (option to not store raw prompts)
    - Batch processing for efficiency
    """

    def __init__(
        self,
        session: AsyncSession,
        portkey_client: PortkeyClient | None = None,
    ):
        self.session = session
        self.portkey = portkey_client or get_portkey_client()

    async def sync_project_logs(
        self,
        project: Project,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        include_prompt: bool = True,
    ) -> dict[str, Any]:
        """
        Sync logs from Portkey for a specific project.
        
        Args:
            project: The project to sync logs for
            start_date: Start of time range (defaults to last sync or 30 days ago)
            end_date: End of time range (defaults to now)
            limit: Maximum number of logs to sync
            include_prompt: Whether to store raw prompts
        
        Returns:
            Sync statistics
        """
        logger.info(
            "Starting log sync",
            project_id=str(project.id),
            project_name=project.name,
        )

        # Determine time range
        if not start_date:
            start_date = project.last_log_sync or (
                datetime.now(timezone.utc) - timedelta(days=30)
            )
        if not end_date:
            end_date = datetime.now(timezone.utc)

        sync_stats = {
            "project_id": project.id,
            "logs_synced": 0,
            "logs_skipped": 0,
            "sync_start": start_date,
            "sync_end": end_date,
            "oldest_log": None,
            "newest_log": None,
            "errors": [],
        }

        try:
            # Fetch logs from Portkey
            logs = await self.portkey.fetch_all_logs(
                virtual_key=project.portkey_virtual_key,
                start_date=start_date,
                end_date=end_date,
                max_logs=limit,
            )

            if not logs:
                logger.info("No new logs found", project_id=str(project.id))
                return sync_stats

            # Get existing log IDs to avoid duplicates
            existing_ids = await self._get_existing_log_ids(
                project.id,
                [log.get("id", "") for log in logs],
            )

            # Process logs in batches
            batch_size = 100
            for i in range(0, len(logs), batch_size):
                batch = logs[i:i + batch_size]
                synced, skipped = await self._process_log_batch(
                    project_id=project.id,
                    logs=batch,
                    existing_ids=existing_ids,
                    include_prompt=include_prompt,
                )
                sync_stats["logs_synced"] += synced
                sync_stats["logs_skipped"] += skipped

            # Update timestamps
            if logs:
                timestamps = [
                    datetime.fromisoformat(log.get("created_at", ""))
                    for log in logs
                    if log.get("created_at")
                ]
                if timestamps:
                    sync_stats["oldest_log"] = min(timestamps)
                    sync_stats["newest_log"] = max(timestamps)

            # Update project's last sync time
            project.last_log_sync = datetime.now(timezone.utc)
            await self.session.commit()

            logger.info(
                "Log sync completed",
                project_id=str(project.id),
                synced=sync_stats["logs_synced"],
                skipped=sync_stats["logs_skipped"],
            )

        except Exception as e:
            logger.error(
                "Log sync failed",
                project_id=str(project.id),
                error=str(e),
            )
            sync_stats["errors"].append(str(e))
            await self.session.rollback()

        return sync_stats

    async def _get_existing_log_ids(
        self,
        project_id: uuid.UUID,
        portkey_log_ids: list[str],
    ) -> set[str]:
        """Get set of Portkey log IDs that already exist in the database."""
        if not portkey_log_ids:
            return set()

        result = await self.session.execute(
            select(LogEntry.portkey_log_id).where(
                LogEntry.project_id == project_id,
                LogEntry.portkey_log_id.in_(portkey_log_ids),
            )
        )
        return set(row[0] for row in result.all())

    async def _process_log_batch(
        self,
        project_id: uuid.UUID,
        logs: list[dict[str, Any]],
        existing_ids: set[str],
        include_prompt: bool,
    ) -> tuple[int, int]:
        """Process a batch of logs, returning (synced_count, skipped_count)."""
        synced = 0
        skipped = 0

        for log_data in logs:
            portkey_log_id = log_data.get("id", "")
            
            if portkey_log_id in existing_ids:
                skipped += 1
                continue

            try:
                # Convert Portkey log to our model
                log_entry = self._convert_portkey_log(
                    project_id=project_id,
                    portkey_data=log_data,
                    include_prompt=include_prompt,
                )
                self.session.add(log_entry)
                synced += 1
                existing_ids.add(portkey_log_id)  # Prevent duplicates within batch
                
            except Exception as e:
                logger.warning(
                    "Failed to process log",
                    portkey_log_id=portkey_log_id,
                    error=str(e),
                )
                skipped += 1

        await self.session.flush()
        return synced, skipped

    def _convert_portkey_log(
        self,
        project_id: uuid.UUID,
        portkey_data: dict[str, Any],
        include_prompt: bool,
    ) -> LogEntry:
        """Convert Portkey log data to LogEntry model."""
        # Extract prompt from messages
        messages = portkey_data.get("request", {}).get("messages", [])
        prompt_str = ""
        system_prompt = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                prompt_str = msg.get("content", "")

        # Parse timestamp
        created_at = portkey_data.get("created_at")
        if isinstance(created_at, str):
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            timestamp = datetime.now(timezone.utc)

        # Determine status
        status = "success"
        if portkey_data.get("error"):
            status = "error"
        elif portkey_data.get("refusal"):
            status = "refused"

        # Calculate cost if not provided
        cost = portkey_data.get("cost", 0.0)
        if cost == 0:
            # Estimate based on tokens (rough estimate)
            usage = portkey_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # Very rough estimate: $0.001 per 1K tokens
            cost = (input_tokens + output_tokens) * 0.000001

        return LogEntry(
            id=uuid.uuid4(),
            project_id=project_id,
            portkey_log_id=portkey_data.get("id", str(uuid.uuid4())),
            trace_id=portkey_data.get("trace_id"),
            span_id=portkey_data.get("span_id"),
            timestamp=timestamp,
            endpoint=portkey_data.get("endpoint", "/chat/completions"),
            prompt=prompt_str if include_prompt else None,
            prompt_hash=LogEntry.compute_hash(prompt_str),
            system_prompt=system_prompt if include_prompt else None,
            context=portkey_data.get("context"),
            tool_calls=portkey_data.get("request", {}).get("tools"),
            model=portkey_data.get("model", "unknown"),
            provider=portkey_data.get("provider", "unknown"),
            input_tokens=portkey_data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=portkey_data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=portkey_data.get("usage", {}).get("total_tokens", 0),
            latency_ms=portkey_data.get("response_time", 0),
            cost_usd=cost,
            status=status,
            error_message=portkey_data.get("error", {}).get("message") if portkey_data.get("error") else None,
            error_code=portkey_data.get("error", {}).get("code") if portkey_data.get("error") else None,
            refusal=portkey_data.get("refusal", False),
            cache_status=portkey_data.get("cache_status"),
            retry_count=portkey_data.get("retry_count", 0),
            fallback_used=portkey_data.get("fallback_used", False),
            metadata=portkey_data.get("metadata"),
        )

    async def get_log_stats(self, project_id: uuid.UUID) -> dict[str, Any]:
        """Get statistics about ingested logs for a project."""
        result = await self.session.execute(
            select(
                func.count(LogEntry.id).label("total"),
                func.min(LogEntry.timestamp).label("oldest"),
                func.max(LogEntry.timestamp).label("newest"),
                func.sum(LogEntry.cost_usd).label("total_cost"),
                func.sum(LogEntry.input_tokens).label("total_input_tokens"),
                func.sum(LogEntry.output_tokens).label("total_output_tokens"),
                func.avg(LogEntry.latency_ms).label("avg_latency"),
            ).where(LogEntry.project_id == project_id)
        )
        row = result.one()

        # Get success/failure counts
        status_result = await self.session.execute(
            select(
                LogEntry.status,
                func.count(LogEntry.id).label("count"),
            )
            .where(LogEntry.project_id == project_id)
            .group_by(LogEntry.status)
        )
        status_counts = {r.status: r.count for r in status_result.all()}

        # Get models used
        models_result = await self.session.execute(
            select(LogEntry.model)
            .where(LogEntry.project_id == project_id)
            .distinct()
        )
        models = [r[0] for r in models_result.all()]

        # Get providers used
        providers_result = await self.session.execute(
            select(LogEntry.provider)
            .where(LogEntry.project_id == project_id)
            .distinct()
        )
        providers = [r[0] for r in providers_result.all()]

        total = row.total or 0
        success_count = status_counts.get("success", 0)
        refused_count = status_counts.get("refused", 0)
        error_count = status_counts.get("error", 0)

        return {
            "project_id": project_id,
            "total_logs": total,
            "date_range_start": row.oldest,
            "date_range_end": row.newest,
            "models_used": models,
            "providers_used": providers,
            "total_cost_usd": float(row.total_cost or 0),
            "total_input_tokens": int(row.total_input_tokens or 0),
            "total_output_tokens": int(row.total_output_tokens or 0),
            "avg_latency_ms": float(row.avg_latency or 0),
            "success_rate": success_count / total if total > 0 else 0,
            "refusal_rate": refused_count / total if total > 0 else 0,
            "error_rate": error_count / total if total > 0 else 0,
        }
