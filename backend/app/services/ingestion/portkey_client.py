"""Portkey API client for log retrieval and model operations."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


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
        json: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to Portkey API with retry logic."""
        url = f"{self.base_url}{endpoint}"
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=request_headers,
                )
                
                if response.status_code == 401:
                    raise PortkeyAuthError("Invalid Portkey API key")
                elif response.status_code == 429:
                    raise PortkeyRateLimitError("Portkey rate limit exceeded")
                
                response.raise_for_status()
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
        virtual_key: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve logs from Portkey Observability.
        
        Args:
            virtual_key: Filter logs by virtual key
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum number of logs to retrieve
            offset: Pagination offset
            filters: Additional filters (model, status, etc.)
        
        Returns:
            Dict containing logs and pagination info
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if virtual_key:
            params["virtual_key"] = virtual_key
        if filters:
            params.update(filters)

        logger.info(
            "Fetching logs from Portkey",
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )
        
        return await self._request("GET", "/v1/logs", params=params)

    async def get_log_by_id(self, log_id: str) -> dict[str, Any]:
        """Retrieve a specific log entry by ID."""
        return await self._request("GET", f"/v1/logs/{log_id}")

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
            json=payload,
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

    async def fetch_all_logs(
        self,
        virtual_key: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        batch_size: int = 100,
        max_logs: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all logs with pagination.
        
        Handles pagination automatically to retrieve all matching logs.
        """
        all_logs: list[dict[str, Any]] = []
        offset = 0
        
        while True:
            response = await self.get_logs(
                virtual_key=virtual_key,
                start_date=start_date,
                end_date=end_date,
                limit=batch_size,
                offset=offset,
            )
            
            logs = response.get("data", [])
            if not logs:
                break
                
            all_logs.extend(logs)
            offset += len(logs)
            
            if max_logs and len(all_logs) >= max_logs:
                all_logs = all_logs[:max_logs]
                break
            
            # Check if we've reached the end
            total = response.get("total", 0)
            if offset >= total:
                break
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        logger.info(f"Fetched {len(all_logs)} logs from Portkey")
        return all_logs


# Singleton instance
_portkey_client: PortkeyClient | None = None


def get_portkey_client() -> PortkeyClient:
    """Get or create the Portkey client singleton."""
    global _portkey_client
    if _portkey_client is None:
        _portkey_client = PortkeyClient()
    return _portkey_client
