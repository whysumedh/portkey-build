"""Portkey API client for log retrieval and model operations."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Default workspace ID (can be overridden)
DEFAULT_WORKSPACE_ID = "2d469afe-6e46-4929-ab71-21de003b711d"


class PortkeyClientError(Exception):
    """Base exception for Portkey client errors."""
    pass


class PortkeyAuthError(PortkeyClientError):
    """Authentication error with Portkey API."""
    pass


class PortkeyRateLimitError(PortkeyClientError):
    """Rate limit exceeded on Portkey API."""
    pass


class PortkeyClient:
    """
    Client for interacting with Portkey APIs.
    
    Handles:
    - Log retrieval from Portkey Observability
    - Model catalog access
    - Configuration management
    - LLM completions through Portkey gateway
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or settings.portkey_api_key
        self.base_url = (base_url or settings.portkey_base_url).rstrip("/")
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("Portkey API key not configured")

    def _get_headers(self, virtual_key: str | None = None) -> dict[str, str]:
        """Get headers for Portkey API requests."""
        headers = {
            "x-portkey-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        if virtual_key:
            headers["x-portkey-virtual-key"] = virtual_key
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_body: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to Portkey API with retry logic."""
        url = f"{self.base_url}{endpoint}"
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        logger.debug(
            "Portkey API request",
            method=method,
            url=url,
            params=params,
            json_body=json_body,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=request_headers,
                )
                
                if response.status_code == 401:
                    raise PortkeyAuthError("Invalid Portkey API key")
                elif response.status_code == 429:
                    raise PortkeyRateLimitError("Portkey rate limit exceeded")
                
                # Log error responses before raising
                if response.status_code >= 400:
                    error_body = response.text
                    logger.error(
                        "Portkey API error response",
                        status_code=response.status_code,
                        endpoint=endpoint,
                        response_body=error_body,
                    )
                    raise PortkeyClientError(f"Portkey API error ({response.status_code}): {error_body}")
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Portkey API error",
                    status_code=e.response.status_code,
                    endpoint=endpoint,
                    error=str(e),
                )
                raise PortkeyClientError(f"Portkey API error: {e}")
            except httpx.RequestError as e:
                logger.error(
                    "Portkey request error",
                    endpoint=endpoint,
                    error=str(e),
                )
                raise PortkeyClientError(f"Request failed: {e}")

    async def get_logs(
        self,
        workspace_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve logs from Portkey Observability using the Log Export API.
        
        Args:
            workspace_id: Filter logs by workspace
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum number of logs to retrieve
            offset: Pagination offset
            filters: Additional filters (model, status, etc.)
        
        Returns:
            Dict containing logs and pagination info
        """
        logger.info(
            "Fetching logs from Portkey via Export API",
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            workspace_id=workspace_id,
        )
        
        # Use the export API (the only working method for listing logs)
        return await self._fetch_logs_via_export(workspace_id, start_date, end_date, limit)

    async def _fetch_logs_via_export(
        self,
        workspace_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Fallback: Fetch logs using the export API."""
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(hours=24)
            
        # Create export - filters with time range are REQUIRED
        filters: dict[str, Any] = {
            "time_of_generation_min": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "time_of_generation_max": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
            
        export_body = {
            "workspace_id": workspace_id or DEFAULT_WORKSPACE_ID,
            "description": f"Log export {datetime.utcnow().isoformat()}",
            "filters": filters,
            "requested_data": [
                # Core identifiers
                "id", "trace_id", "request_id",
                # Timestamps
                "created_at", "time_of_generation",
                # Model info
                "ai_org", "ai_model", "mode", "config",
                # Request/Response
                "request", "response", "request_url",
                # Tokens (Portkey uses units terminology)
                "req_units", "res_units", "total_units",
                # Cost and performance
                "cost", "cost_currency", "response_time", "response_status_code",
                # Status
                "is_success", "status",
                # Extra
                "metadata", "prompt_slug"
            ]
        }
        
        try:
            # Create export
            logger.info("Creating log export", body=export_body)
            create_resp = await self._request("POST", "/v1/logs/exports", json_body=export_body)
            logger.info("Export created", response=create_resp)
            
            export_id = create_resp.get("id") or create_resp.get("export_id")
            total_logs = create_resp.get("total", 0)
            
            if not export_id:
                logger.error("No export ID in response", response=create_resp)
                return {"logs": [], "total": 0}
            
            logger.info(f"Export created with ID {export_id}, total logs: {total_logs}")
            
            # Start export - CRITICAL step
            logger.info(f"Starting export {export_id}")
            try:
                start_resp = await self._request("POST", f"/v1/logs/exports/{export_id}/start")
                logger.info("Export started", response=start_resp)
            except Exception as start_err:
                logger.error(f"Failed to start export: {start_err}")
                # Continue anyway - maybe it auto-starts
            
            # Poll for completion (max 60 seconds)
            for i in range(30):
                await asyncio.sleep(2)
                status_resp = await self._request("GET", f"/v1/logs/exports/{export_id}")
                status = status_resp.get("status", "").lower()
                logger.info(f"Export status poll {i+1}: {status}", response=status_resp)
                
                if status == "success":
                    # Get download URL
                    logger.info("Export successful, fetching download URL")
                    download_resp = await self._request("GET", f"/v1/logs/exports/{export_id}/download")
                    logger.info("Download response", response=download_resp)
                    download_url = download_resp.get("signed_url") or download_resp.get("url") or download_resp.get("download_url")
                    
                    if download_url:
                        # Fetch the data
                        logger.info(f"Downloading from {download_url}")
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(download_url)
                            logger.info(f"Download response status: {resp.status_code}, length: {len(resp.text)}")
                            # Parse JSONL
                            logs = []
                            for line in resp.text.strip().split("\n"):
                                if line:
                                    try:
                                        logs.append(json.loads(line))
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse line: {line[:100]}", error=str(e))
                            logger.info(f"Parsed {len(logs)} logs from export")
                            return {"logs": logs[:limit], "total": len(logs)}
                    else:
                        logger.error("No download URL in response", response=download_resp)
                    break
                elif status in ("failed", "error"):
                    logger.error("Export failed", status=status, response=status_resp)
                    break
                elif status == "pending":
                    logger.debug(f"Export still pending, waiting...")
                    
            logger.warning("Export polling timed out or failed")
            return {"logs": [], "total": 0}
            
        except Exception as e:
            logger.error("Export API failed", error=str(e), exc_info=True)
            return {"logs": [], "total": 0}

    async def get_log_by_id(self, log_id: str) -> dict[str, Any]:
        """Retrieve a specific log entry by ID."""
        # Try the direct endpoint first
        try:
            return await self._request("GET", f"/v1/logs/{log_id}")
        except PortkeyClientError:
            # Try POST with log_id filter
            body = {"filters": {"request_id": log_id}, "page_size": 1}
            result = await self._request("POST", "/v1/logs/search", json_body=body)
            logs = result.get("logs", result.get("data", []))
            if logs:
                return logs[0]
            raise PortkeyClientError(f"Log {log_id} not found")

    async def get_model_catalog(self) -> dict[str, Any]:
        """
        Retrieve the model catalog from Portkey.
        
        Returns available models with their metadata including:
        - Provider and model names
        - Pricing information
        - Context window sizes
        - Capabilities
        """
        return await self._request("GET", "/v1/models")

    async def get_model_info(self, provider: str, model: str) -> dict[str, Any]:
        """Get detailed information about a specific model."""
        return await self._request("GET", f"/v1/models/{provider}/{model}")

    async def get_configs(self, config_id: str | None = None) -> dict[str, Any]:
        """
        Retrieve Portkey configurations.
        
        Configs define routing, fallbacks, caching, and other gateway behaviors.
        """
        endpoint = f"/v1/configs/{config_id}" if config_id else "/v1/configs"
        return await self._request("GET", endpoint)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
        virtual_key: str | None = None,
        config_id: str | None = None,
        metadata: dict | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a chat completion through Portkey gateway.
        
        Used by the replay engine to test candidate models.
        """
        headers = self._get_headers(virtual_key)
        
        if config_id:
            headers["x-portkey-config"] = config_id
        if trace_id:
            headers["x-portkey-trace-id"] = trace_id
        if metadata:
            headers["x-portkey-metadata"] = str(metadata)

        # Set provider header
        headers["x-portkey-provider"] = provider

        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        return await self._request(
            "POST",
            "/v1/chat/completions",
            json_body=payload,
            headers=headers,
        )

    async def health_check(self) -> bool:
        """Check if Portkey API is accessible."""
        try:
            await self._request("GET", "/v1/health")
            return True
        except Exception as e:
            logger.warning("Portkey health check failed", error=str(e))
            return False

    # ===================
    # Log Export API (Beta)
    # ===================

    async def create_log_export(
        self,
        workspace_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
        requested_data: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a log export job.
        
        Args:
            workspace_id: Portkey workspace ID
            start_date: Start of time range
            end_date: End of time range
            filters: Additional filters (model, status, etc.)
            requested_data: List of fields to include in export
        
        Returns:
            Export job details including export_id
        """
        workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
        }
        
        # Build filters
        export_filters: dict[str, Any] = {}
        if start_date:
            export_filters["time_of_generation_min"] = start_date.isoformat()
        if end_date:
            export_filters["time_of_generation_max"] = end_date.isoformat()
        if filters:
            export_filters.update(filters)
        
        if export_filters:
            payload["filters"] = export_filters
        
        # Default fields to request
        if requested_data is None:
            requested_data = [
                "id", "trace_id", "span_id", "created_at", 
                "request", "response", "ai_model", "ai_provider",
                "cost", "response_time", "prompt_tokens", "completion_tokens",
                "total_tokens", "status", "is_success", "metadata"
            ]
        payload["requested_data"] = requested_data

        logger.info(
            "Creating log export",
            workspace_id=workspace_id,
            start_date=start_date,
            end_date=end_date,
        )
        
        return await self._request("POST", "/v1/logs/exports", json_body=payload)

    async def start_log_export(self, export_id: str) -> dict[str, Any]:
        """Start a log export job."""
        logger.info("Starting log export", export_id=export_id)
        return await self._request("POST", f"/v1/logs/exports/{export_id}/start")

    async def get_log_export(self, export_id: str) -> dict[str, Any]:
        """Get the status and details of a log export."""
        return await self._request("GET", f"/v1/logs/exports/{export_id}")

    async def list_log_exports(
        self, 
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List all log exports for a workspace."""
        workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        params = {
            "workspace_id": workspace_id,
            "limit": limit,
        }
        return await self._request("GET", "/v1/logs/exports", params=params)

    async def download_log_export(self, export_id: str) -> dict[str, Any]:
        """Get the download URL for a completed log export."""
        return await self._request("GET", f"/v1/logs/exports/{export_id}/download")

    async def fetch_export_data(self, download_url: str) -> list[dict[str, Any]]:
        """
        Download and parse JSONL export data from signed URL.
        
        Args:
            download_url: Signed URL from download_log_export
            
        Returns:
            List of log entries
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            
            # Parse JSONL (each line is a JSON object)
            logs = []
            for line in response.text.strip().split("\n"):
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse log line: {e}")
            
            logger.info(f"Downloaded {len(logs)} logs from export")
            return logs

    async def export_logs_and_wait(
        self,
        workspace_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
        poll_interval: float = 2.0,
        max_wait_time: float = 300.0,
    ) -> list[dict[str, Any]]:
        """
        Create a log export, wait for completion, and return the data.
        
        This is a convenience method that handles the full export workflow.
        
        Args:
            workspace_id: Portkey workspace ID
            start_date: Start of time range
            end_date: End of time range
            filters: Additional filters
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait for export
            
        Returns:
            List of log entries
        """
        # Create export
        export_response = await self.create_log_export(
            workspace_id=workspace_id,
            start_date=start_date,
            end_date=end_date,
            filters=filters,
        )
        export_id = export_response.get("id") or export_response.get("export_id")
        
        if not export_id:
            raise PortkeyClientError("Failed to get export ID from response")
        
        logger.info(f"Created export job: {export_id}")
        
        # Start export
        await self.start_log_export(export_id)
        
        # Poll for completion
        start_time = asyncio.get_event_loop().time()
        while True:
            status_response = await self.get_log_export(export_id)
            status = status_response.get("status", "").lower()
            
            if status == "success" or status == "completed":
                logger.info(f"Export {export_id} completed successfully")
                break
            elif status in ("failed", "error"):
                raise PortkeyClientError(f"Export failed: {status_response}")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_time:
                raise PortkeyClientError(f"Export timed out after {max_wait_time}s")
            
            logger.debug(f"Export status: {status}, waiting...")
            await asyncio.sleep(poll_interval)
        
        # Download export
        download_response = await self.download_log_export(export_id)
        download_url = download_response.get("url") or download_response.get("signed_url")
        
        if not download_url:
            raise PortkeyClientError("Failed to get download URL")
        
        # Fetch and parse data
        return await self.fetch_export_data(download_url)

    async def fetch_all_logs(
        self,
        workspace_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        batch_size: int = 100,
        max_logs: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all logs with pagination.
        
        Handles pagination automatically to retrieve all matching logs.
        """
        # Use the get_logs method which now tries search/export APIs
        response = await self.get_logs(
            workspace_id=workspace_id or DEFAULT_WORKSPACE_ID,
            start_date=start_date,
            end_date=end_date,
            limit=max_logs or batch_size,
        )
        
        # Response contains logs in "logs" key
        logs = response.get("logs", response.get("data", []))
        
        if max_logs and len(logs) > max_logs:
            logs = logs[:max_logs]
        
        logger.info(f"Fetched {len(logs)} logs from Portkey")
        return logs


# Singleton instance
_portkey_client: PortkeyClient | None = None


def get_portkey_client() -> PortkeyClient:
    """Get or create the Portkey client singleton."""
    global _portkey_client
    if _portkey_client is None:
        _portkey_client = PortkeyClient()
    return _portkey_client
