"""
Analysis Request Handler - Safe interface for analytics requests.

This handler validates and executes structured analysis requests.
It is the ONLY interface for statistical analysis - LLMs cannot
execute arbitrary code.
"""

import uuid
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.schemas.analytics import (
    AnalysisRequest,
    AnalysisResponse,
    DistributionResult,
    PercentileResult,
    CorrelationResult,
    AggregationResult,
    ClusteringResult,
    SampleResult,
)
from app.services.analytics.engine import (
    AnalyticsEngine,
    AnalyticsEngineError,
    InsufficientDataError,
    UnsupportedOperationError,
)

logger = get_logger(__name__)


class AnalysisRequestHandler:
    """
    Handles structured analysis requests from LLMs and users.
    
    This is the safe interface that:
    - Validates request structure
    - Executes only allowed operations
    - Returns structured results
    - Caches results for efficiency
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.engine = AnalyticsEngine(session)
        self._cache: dict[str, AnalysisResponse] = {}

    async def handle_request(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Handle an analysis request and return structured results.
        
        Args:
            request: Structured analysis request
            
        Returns:
            AnalysisResponse with results or error
        """
        request_id = uuid.uuid4()
        start_time = time.time()
        
        logger.info(
            "Processing analysis request",
            request_id=str(request_id),
            project_id=str(request.project_id),
            analysis_type=request.type,
        )

        try:
            result = await self._execute_analysis(request)
            execution_time = (time.time() - start_time) * 1000
            
            return AnalysisResponse(
                request_id=request_id,
                project_id=request.project_id,
                analysis_type=request.type,
                status="completed",
                cached=False,
                execution_time_ms=execution_time,
                result=result,
                error=None,
                created_at=datetime.now(timezone.utc),
            )

        except InsufficientDataError as e:
            logger.warning(
                "Insufficient data for analysis",
                request_id=str(request_id),
                error=str(e),
            )
            return AnalysisResponse(
                request_id=request_id,
                project_id=request.project_id,
                analysis_type=request.type,
                status="failed",
                cached=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                result=None,
                error=f"Insufficient data: {str(e)}",
                created_at=datetime.now(timezone.utc),
            )

        except UnsupportedOperationError as e:
            logger.warning(
                "Unsupported operation requested",
                request_id=str(request_id),
                error=str(e),
            )
            return AnalysisResponse(
                request_id=request_id,
                project_id=request.project_id,
                analysis_type=request.type,
                status="failed",
                cached=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                result=None,
                error=f"Unsupported operation: {str(e)}",
                created_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(
                "Analysis request failed",
                request_id=str(request_id),
                error=str(e),
                exc_info=True,
            )
            return AnalysisResponse(
                request_id=request_id,
                project_id=request.project_id,
                analysis_type=request.type,
                status="failed",
                cached=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                result=None,
                error=f"Analysis failed: {str(e)}",
                created_at=datetime.now(timezone.utc),
            )

    async def _execute_analysis(
        self, request: AnalysisRequest
    ) -> DistributionResult | PercentileResult | CorrelationResult | AggregationResult | ClusteringResult | SampleResult:
        """Execute the appropriate analysis based on request type."""
        
        params = request.params
        common_args = {
            "project_id": request.project_id,
            "start_date": request.time_range_start,
            "end_date": request.time_range_end,
            "filters": request.filters,
        }

        if request.type == "distribution":
            result = await self.engine.distribution(
                field=params.get("field", ""),
                bins=params.get("bins", 20),
                normalize=params.get("normalize", False),
                **common_args,
            )
            return DistributionResult(**result)

        elif request.type == "percentile":
            result = await self.engine.percentile(
                metric=params.get("metric", ""),
                percentiles=params.get("percentiles", [50, 75, 90, 95, 99]),
                **common_args,
            )
            return PercentileResult(**result)

        elif request.type == "correlation":
            result = await self.engine.correlation(
                x=params.get("x", ""),
                y=params.get("y", ""),
                method=params.get("method", "pearson"),
                **common_args,
            )
            return CorrelationResult(**result)

        elif request.type == "aggregation":
            result = await self.engine.aggregation(
                metric=params.get("metric", ""),
                group_by=params.get("group_by"),
                aggregations=params.get("aggregations", ["count", "mean", "std"]),
                **common_args,
            )
            return AggregationResult(**result)

        elif request.type == "clustering":
            result = await self.engine.clustering(
                features=params.get("features", []),
                n_clusters=params.get("n_clusters", 5),
                method=params.get("method", "kmeans"),
                **common_args,
            )
            return ClusteringResult(**result)

        elif request.type == "sample":
            result = await self.engine.sample(
                n=params.get("n", 100),
                stratify_by=params.get("stratify_by"),
                random_seed=params.get("random_seed", 42),
                **common_args,
            )
            return SampleResult(**result)

        else:
            raise UnsupportedOperationError(f"Unknown analysis type: {request.type}")

    async def get_summary(self, project_id: uuid.UUID) -> dict[str, Any]:
        """Get summary statistics for a project."""
        return await self.engine.summary_statistics(project_id)


def validate_analysis_request(request_dict: dict[str, Any]) -> list[str]:
    """
    Validate an analysis request dictionary before processing.
    
    Returns a list of validation errors (empty if valid).
    """
    errors = []
    
    if "type" not in request_dict:
        errors.append("Missing 'type' field")
        return errors
    
    analysis_type = request_dict["type"]
    params = request_dict.get("params", {})
    
    if analysis_type == "distribution":
        if "field" not in params:
            errors.append("Distribution requires 'field' parameter")
            
    elif analysis_type == "percentile":
        if "metric" not in params:
            errors.append("Percentile requires 'metric' parameter")
            
    elif analysis_type == "correlation":
        if "x" not in params:
            errors.append("Correlation requires 'x' parameter")
        if "y" not in params:
            errors.append("Correlation requires 'y' parameter")
            
    elif analysis_type == "aggregation":
        if "metric" not in params:
            errors.append("Aggregation requires 'metric' parameter")
            
    elif analysis_type == "clustering":
        if "features" not in params or not params["features"]:
            errors.append("Clustering requires 'features' parameter with at least one feature")
            
    elif analysis_type == "sample":
        pass  # Sample has sensible defaults
        
    else:
        errors.append(f"Unknown analysis type: {analysis_type}")
    
    return errors
