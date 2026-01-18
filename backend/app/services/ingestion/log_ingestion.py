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


# Model pricing per million tokens (USD)
# Updated pricing as of 2026 - prices in $ per 1M tokens
MODEL_PRICING = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o4-mini-deep-research": {"input": 1.10, "output": 4.40},
    # Anthropic models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},  # Claude Opus 4.5
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    # Google models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = {"input": 5.00, "output": 15.00}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost based on model pricing and token counts.
    
    Args:
        model: Model name/identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Cost in USD
    """
    # Find pricing for this model (check for partial matches)
    pricing = None
    model_lower = model.lower()
    
    for model_key, model_pricing in MODEL_PRICING.items():
        if model_key.lower() in model_lower or model_lower in model_key.lower():
            pricing = model_pricing
            break
    
    if pricing is None:
        pricing = DEFAULT_PRICING
        logger.debug(f"Using default pricing for unknown model: {model}")
    
    # Calculate cost (pricing is per million tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def parse_portkey_timestamp(time_str: str | None) -> datetime:
    """
    Parse timestamp from Portkey API.
    
    Portkey returns timestamps in JavaScript Date format:
    "Sat Jan 17 2026 06:20:43 GMT+0000 (Coordinated Universal Time)"
    
    This function handles multiple formats:
    - JavaScript Date format (e.g., "Sat Jan 17 2026 06:20:43 GMT+0000 (Coordinated Universal Time)")
    - ISO format (e.g., "2026-01-17T06:20:43Z" or "2026-01-17T06:20:43+00:00")
    - Unix timestamp (milliseconds or seconds)
    
    Args:
        time_str: Timestamp string from Portkey
        
    Returns:
        Parsed datetime in UTC, or current UTC time if parsing fails
    """
    if not time_str:
        return datetime.now(timezone.utc)
    
    # Handle integer timestamps (Unix epoch)
    if isinstance(time_str, (int, float)):
        try:
            # Check if milliseconds (> year 2100 in seconds)
            if time_str > 4102444800:
                time_str = time_str / 1000
            return datetime.fromtimestamp(time_str, tz=timezone.utc)
        except (ValueError, OSError):
            return datetime.now(timezone.utc)
    
    time_str = str(time_str).strip()
    
    # Try ISO format first (most efficient)
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass
    
    # Try JavaScript Date format: "Sat Jan 17 2026 06:20:43 GMT+0000 (Coordinated Universal Time)"
    # Extract the core part before any parenthetical timezone name
    js_date_str = time_str.split("(")[0].strip()
    
    # Try parsing without day name
    js_formats = [
        "%a %b %d %Y %H:%M:%S GMT%z",  # "Sat Jan 17 2026 06:20:43 GMT+0000"
        "%b %d %Y %H:%M:%S GMT%z",     # "Jan 17 2026 06:20:43 GMT+0000"
        "%a %b %d %Y %H:%M:%S %Z",     # With timezone name
        "%Y-%m-%d %H:%M:%S",           # Simple format
        "%Y-%m-%dT%H:%M:%S",           # ISO without timezone
    ]
    
    for fmt in js_formats:
        try:
            parsed = datetime.strptime(js_date_str, fmt)
            # Ensure timezone aware
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue
    
    # Last resort: try to extract date/time components manually
    # Pattern: "... Jan 17 2026 06:20:43 ..."
    import re
    match = re.search(
        r'(\w{3})\s+(\d{1,2})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})',
        time_str
    )
    if match:
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month_str, day, year, hour, minute, second = match.groups()
        month = month_map.get(month_str, 1)
        try:
            return datetime(
                int(year), month, int(day),
                int(hour), int(minute), int(second),
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    
    logger.warning(f"Failed to parse timestamp: {time_str}, using current time")
    return datetime.now(timezone.utc)


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
        # Extract prompt from messages (handle multi-modal content)
        messages = portkey_data.get("request", {}).get("messages", [])
        prompt_str = ""
        system_prompt = None
        
        for msg in messages:
            content = msg.get("content", "")
            # Handle multi-modal content (list of content parts)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content_str = "\n".join(text_parts) if text_parts else str(content)
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content) if content else ""
            
            if msg.get("role") == "system":
                system_prompt = content_str
            elif msg.get("role") == "user":
                prompt_str = content_str

        # Parse timestamp using robust parser that handles multiple formats
        created_at = portkey_data.get("created_at") or portkey_data.get("time_of_generation")
        timestamp = parse_portkey_timestamp(created_at)

        # Determine status
        status = "success"
        if portkey_data.get("error"):
            status = "error"
        elif portkey_data.get("refusal"):
            status = "refused"

        # Get token counts
        usage = portkey_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        model = portkey_data.get("model", "unknown")
        
        # Calculate cost based on token counts and model pricing
        cost = calculate_cost(model, input_tokens, output_tokens)

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
            model=model,
            provider=portkey_data.get("provider", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=portkey_data.get("response_time", 0),
            cost_usd=cost,
            status=status,
            error_message=portkey_data.get("error", {}).get("message") if portkey_data.get("error") else None,
            error_code=portkey_data.get("error", {}).get("code") if portkey_data.get("error") else None,
            refusal=portkey_data.get("refusal", False),
            cache_status=portkey_data.get("cache_status"),
            retry_count=portkey_data.get("retry_count", 0),
            fallback_used=portkey_data.get("fallback_used", False),
            log_metadata=portkey_data.get("metadata"),
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

    async def import_logs_by_ids(
        self,
        project_id: uuid.UUID,
        log_ids: list[str],
        include_prompt: bool = True,
    ) -> dict[str, Any]:
        """
        Assign specific logs to a project.
        
        This first tries to update existing logs in the database (from the workspace pool).
        Only fetches from Portkey if logs are not found locally.
        
        Args:
            project_id: The project ID to associate logs with
            log_ids: List of Portkey log IDs to import
            include_prompt: Whether to store raw prompts
            
        Returns:
            Import statistics
        """
        logger.info(
            "Assigning logs to project",
            project_id=str(project_id),
            log_count=len(log_ids),
        )
        
        import_stats = {
            "project_id": str(project_id),
            "requested": len(log_ids),
            "assigned": 0,
            "imported": 0,
            "skipped": 0,
            "errors": [],
        }
        
        if not log_ids:
            return import_stats
        
        try:
            # Step 1: Try to update existing logs in the database (from workspace pool)
            # These are logs that were fetched via "Refresh from Portkey" in the logs tab
            from sqlalchemy import update
            
            # Update logs to assign to this project (allow re-assignment from other projects)
            update_result = await self.session.execute(
                update(LogEntry)
                .where(LogEntry.portkey_log_id.in_(log_ids))
                .values(project_id=project_id)
            )
            assigned_count = update_result.rowcount
            import_stats["assigned"] = assigned_count
            
            logger.info(
                "Updated existing logs with project_id",
                assigned=assigned_count,
            )
            
            # Step 1b: Recalculate costs for assigned logs (in case they were stored with cost=0)
            logs_to_fix = await self.session.execute(
                select(LogEntry)
                .where(
                    LogEntry.portkey_log_id.in_(log_ids),
                    LogEntry.project_id == project_id,
                )
            )
            fixed_cost_count = 0
            for log in logs_to_fix.scalars().all():
                # Recalculate cost based on token counts
                new_cost = calculate_cost(log.model, log.input_tokens or 0, log.output_tokens or 0)
                if new_cost != log.cost_usd:
                    log.cost_usd = new_cost
                    fixed_cost_count += 1
            
            if fixed_cost_count > 0:
                logger.info(f"Recalculated costs for {fixed_cost_count} logs")
            
            # Step 2: Check which logs are still missing
            existing_result = await self.session.execute(
                select(LogEntry.portkey_log_id)
                .where(LogEntry.portkey_log_id.in_(log_ids))
            )
            existing_ids = {row[0] for row in existing_result.all()}
            missing_ids = set(log_ids) - existing_ids
            
            if missing_ids:
                logger.info(
                    "Some logs not in database, fetching from Portkey",
                    missing_count=len(missing_ids),
                )
                
                # Step 3: Fetch missing logs from Portkey
                try:
                    all_logs = await self.portkey.get_logs(hours=168)  # Last 7 days
                    
                    # Filter to only the missing log IDs
                    logs_to_import = [
                        log for log in all_logs
                        if (log.get("id") in missing_ids or log.get("log_id") in missing_ids)
                    ]
                    
                    for log_data in logs_to_import:
                        log_id = log_data.get("id") or log_data.get("log_id", "")
                        
                        try:
                            log_entry = self._convert_portkey_export_log(
                                project_id=project_id,
                                portkey_data=log_data,
                                include_prompt=include_prompt,
                            )
                            self.session.add(log_entry)
                            import_stats["imported"] += 1
                            
                        except Exception as e:
                            logger.warning(
                                "Failed to import log",
                                log_id=log_id,
                                error=str(e),
                            )
                            import_stats["errors"].append(f"{log_id}: {str(e)}")
                            import_stats["skipped"] += 1
                    
                except Exception as e:
                    logger.warning(
                        "Failed to fetch missing logs from Portkey",
                        error=str(e),
                    )
                    import_stats["skipped"] = len(missing_ids)
            
            await self.session.flush()
            
            logger.info(
                "Log assignment completed",
                project_id=str(project_id),
                assigned=import_stats["assigned"],
                imported=import_stats["imported"],
                skipped=import_stats["skipped"],
            )
            
        except Exception as e:
            logger.error(
                "Log assignment failed",
                project_id=str(project_id),
                error=str(e),
            )
            import_stats["errors"].append(str(e))
        
        return import_stats

    def _convert_portkey_export_log(
        self,
        project_id: uuid.UUID,
        portkey_data: dict[str, Any],
        include_prompt: bool,
    ) -> LogEntry:
        """
        Convert Portkey export format log data to LogEntry model.
        
        Export format uses different field names:
        - ai_model instead of model
        - ai_org instead of provider
        - req_units, res_units, total_units instead of usage.prompt_tokens etc.
        - time_of_generation instead of created_at
        - response_time instead of latency_ms
        - is_success instead of status
        """
        # Get log ID
        log_id = portkey_data.get("id") or portkey_data.get("log_id") or str(uuid.uuid4())
        
        # Parse timestamp using robust parser that handles multiple formats
        # Portkey export uses JavaScript Date format: "Sat Jan 17 2026 06:20:43 GMT+0000 (...)"
        time_str = portkey_data.get("time_of_generation") or portkey_data.get("created_at")
        timestamp = parse_portkey_timestamp(time_str)
        
        # Get model and provider (export format uses ai_model, ai_org)
        model = portkey_data.get("ai_model") or portkey_data.get("model") or "unknown"
        provider = portkey_data.get("ai_org") or portkey_data.get("ai_provider") or portkey_data.get("provider") or "unknown"
        
        # Get tokens (export format uses req_units, res_units, total_units)
        input_tokens = portkey_data.get("req_units") or portkey_data.get("usage", {}).get("prompt_tokens", 0) or 0
        output_tokens = portkey_data.get("res_units") or portkey_data.get("usage", {}).get("completion_tokens", 0) or 0
        total_tokens = portkey_data.get("total_units") or portkey_data.get("usage", {}).get("total_tokens", 0) or 0
        
        # Get latency
        latency = portkey_data.get("response_time") or portkey_data.get("latency_ms") or 0
        
        # Determine status
        is_success = portkey_data.get("is_success")
        if is_success is True:
            status = "success"
        elif is_success is False:
            status = "error"
        elif portkey_data.get("error"):
            status = "error"
        elif portkey_data.get("refusal"):
            status = "refused"
        else:
            status = portkey_data.get("status", "success")
        
        # Calculate cost based on token counts and model pricing
        # Note: Portkey's cost values can be unreliable for some models (e.g., Claude Opus)
        # so we calculate based on known pricing tables
        cost = calculate_cost(model, int(input_tokens), int(output_tokens))
        
        # Extract prompt data (handle multi-modal content)
        prompt_str = ""
        system_prompt = None
        messages = portkey_data.get("request", {}).get("messages", [])
        
        if messages:
            for msg in messages:
                content = msg.get("content", "")
                # Handle multi-modal content (list of content parts)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content_str = "\n".join(text_parts) if text_parts else str(content)
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = str(content) if content else ""
                
                if msg.get("role") == "system":
                    system_prompt = content_str
                elif msg.get("role") == "user":
                    prompt_str = content_str
        elif portkey_data.get("prompt"):
            prompt_str = str(portkey_data.get("prompt"))
        
        return LogEntry(
            id=uuid.uuid4(),
            project_id=project_id,
            portkey_log_id=log_id,
            trace_id=portkey_data.get("trace_id"),
            span_id=portkey_data.get("span_id"),
            timestamp=timestamp,
            endpoint=portkey_data.get("endpoint", "/chat/completions"),
            prompt=prompt_str if include_prompt else None,
            prompt_hash=LogEntry.compute_hash(prompt_str),
            system_prompt=system_prompt if include_prompt else None,
            context=portkey_data.get("context"),
            tool_calls=portkey_data.get("request", {}).get("tools"),
            model=model,
            provider=provider,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens),
            latency_ms=float(latency),
            cost_usd=float(cost),
            status=status,
            error_message=portkey_data.get("error", {}).get("message") if isinstance(portkey_data.get("error"), dict) else None,
            error_code=portkey_data.get("error", {}).get("code") if isinstance(portkey_data.get("error"), dict) else None,
            refusal=bool(portkey_data.get("refusal", False)),
            cache_status=portkey_data.get("cache_status"),
            retry_count=int(portkey_data.get("retry_count", 0)),
            fallback_used=bool(portkey_data.get("fallback_used", False)),
            log_metadata=portkey_data.get("metadata"),
        )
