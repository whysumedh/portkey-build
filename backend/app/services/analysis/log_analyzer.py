"""
Log Analyzer - Statistical analysis of user-selected logs for model selection.

This service analyzes logs selected by the user during project creation
and generates a comprehensive report that informs the Model Selector Agent.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.log_entry import LogEntry
from app.models.project import Project, SuccessCriteria

logger = get_logger(__name__)


@dataclass
class TokenDistribution:
    """Token usage distribution statistics."""
    input_tokens: dict[str, float]  # min, max, mean, p50, p95, p99
    output_tokens: dict[str, float]
    total_tokens: dict[str, float]
    histogram_input: list[tuple[float, int]]  # (bin_edge, count)
    histogram_output: list[tuple[float, int]]


@dataclass
class LatencyMetrics:
    """Latency distribution and percentiles."""
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    histogram: list[tuple[float, int]]


@dataclass
class CostBreakdown:
    """Cost analysis by model and provider."""
    total_cost_usd: float
    avg_cost_per_request: float
    cost_by_model: dict[str, float]
    cost_by_provider: dict[str, float]
    projected_monthly_cost: float  # Based on current usage rate


@dataclass
class PromptComplexity:
    """Prompt complexity analysis."""
    avg_prompt_length: float
    max_prompt_length: int
    min_prompt_length: int
    avg_messages_per_request: float
    tool_usage_rate: float  # Percentage of requests with tool calls
    system_prompt_usage_rate: float  # Percentage with system prompts
    complexity_distribution: dict[str, int]  # simple, moderate, complex


@dataclass
class ErrorPatterns:
    """Error and refusal pattern analysis."""
    total_requests: int
    success_rate: float
    error_rate: float
    refusal_rate: float
    timeout_rate: float
    error_codes: dict[str, int]  # Error code to count
    common_error_messages: list[str]


@dataclass
class ModelPerformance:
    """Performance metrics for each model in the logs."""
    model: str
    provider: str
    request_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_per_request: float
    success_rate: float
    refusal_rate: float
    avg_input_tokens: float
    avg_output_tokens: float


@dataclass
class TimeSeriesTrends:
    """Time-based trends in the data."""
    requests_per_hour: dict[int, int]  # Hour of day to count
    requests_per_day: dict[str, int]  # Date string to count
    latency_trend: list[tuple[str, float]]  # (date, avg_latency)
    cost_trend: list[tuple[str, float]]  # (date, total_cost)
    peak_hours: list[int]  # Hours with highest traffic


@dataclass
class AnalysisReport:
    """
    Comprehensive analysis report for model selection.
    
    This report is generated from user-selected logs and provides
    all the information needed by the Model Selector Agent to make
    informed decisions about candidate models.
    """
    # Metadata
    analysis_id: str
    project_id: str
    analyzed_at: str
    log_count: int
    date_range_start: str | None
    date_range_end: str | None
    
    # Core analyses
    token_distribution: TokenDistribution
    latency_metrics: LatencyMetrics
    cost_breakdown: CostBreakdown
    prompt_complexity: PromptComplexity
    error_patterns: ErrorPatterns
    
    # Per-model analysis
    model_performance: list[ModelPerformance]
    
    # Time-series data
    time_trends: TimeSeriesTrends
    
    # Summary insights (human-readable)
    key_insights: list[str] = field(default_factory=list)
    
    # Criteria comparison
    criteria_assessment: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_id": self.analysis_id,
            "project_id": self.project_id,
            "analyzed_at": self.analyzed_at,
            "log_count": self.log_count,
            "date_range": {
                "start": self.date_range_start,
                "end": self.date_range_end,
            },
            "token_distribution": {
                "input_tokens": self.token_distribution.input_tokens,
                "output_tokens": self.token_distribution.output_tokens,
                "total_tokens": self.token_distribution.total_tokens,
            },
            "latency_metrics": {
                "min_ms": self.latency_metrics.min_ms,
                "max_ms": self.latency_metrics.max_ms,
                "mean_ms": self.latency_metrics.mean_ms,
                "std_ms": self.latency_metrics.std_ms,
                "p50_ms": self.latency_metrics.p50_ms,
                "p75_ms": self.latency_metrics.p75_ms,
                "p90_ms": self.latency_metrics.p90_ms,
                "p95_ms": self.latency_metrics.p95_ms,
                "p99_ms": self.latency_metrics.p99_ms,
            },
            "cost_breakdown": {
                "total_cost_usd": self.cost_breakdown.total_cost_usd,
                "avg_cost_per_request": self.cost_breakdown.avg_cost_per_request,
                "cost_by_model": self.cost_breakdown.cost_by_model,
                "cost_by_provider": self.cost_breakdown.cost_by_provider,
                "projected_monthly_cost": self.cost_breakdown.projected_monthly_cost,
            },
            "prompt_complexity": {
                "avg_prompt_length": self.prompt_complexity.avg_prompt_length,
                "max_prompt_length": self.prompt_complexity.max_prompt_length,
                "min_prompt_length": self.prompt_complexity.min_prompt_length,
                "avg_messages_per_request": self.prompt_complexity.avg_messages_per_request,
                "tool_usage_rate": self.prompt_complexity.tool_usage_rate,
                "system_prompt_usage_rate": self.prompt_complexity.system_prompt_usage_rate,
                "complexity_distribution": self.prompt_complexity.complexity_distribution,
            },
            "error_patterns": {
                "total_requests": self.error_patterns.total_requests,
                "success_rate": self.error_patterns.success_rate,
                "error_rate": self.error_patterns.error_rate,
                "refusal_rate": self.error_patterns.refusal_rate,
                "timeout_rate": self.error_patterns.timeout_rate,
                "error_codes": self.error_patterns.error_codes,
            },
            "model_performance": [
                {
                    "model": mp.model,
                    "provider": mp.provider,
                    "request_count": mp.request_count,
                    "avg_latency_ms": mp.avg_latency_ms,
                    "p95_latency_ms": mp.p95_latency_ms,
                    "avg_cost_per_request": mp.avg_cost_per_request,
                    "success_rate": mp.success_rate,
                    "refusal_rate": mp.refusal_rate,
                    "avg_input_tokens": mp.avg_input_tokens,
                    "avg_output_tokens": mp.avg_output_tokens,
                }
                for mp in self.model_performance
            ],
            "time_trends": {
                "requests_per_hour": self.time_trends.requests_per_hour,
                "peak_hours": self.time_trends.peak_hours,
            },
            "key_insights": self.key_insights,
            "criteria_assessment": self.criteria_assessment,
        }
    
    def to_prompt_context(self) -> str:
        """
        Convert to a formatted string for the Model Selector Agent prompt.
        This provides a concise but comprehensive summary for the LLM.
        """
        lines = [
            "## Log Analysis Summary",
            f"- **Analyzed Logs:** {self.log_count}",
            f"- **Date Range:** {self.date_range_start} to {self.date_range_end}",
            "",
            "### Token Usage",
            f"- Average input tokens: {self.token_distribution.input_tokens.get('mean', 0):.0f}",
            f"- Average output tokens: {self.token_distribution.output_tokens.get('mean', 0):.0f}",
            f"- P95 total tokens: {self.token_distribution.total_tokens.get('p95', 0):.0f}",
            "",
            "### Latency Requirements",
            f"- Current average: {self.latency_metrics.mean_ms:.0f}ms",
            f"- P50: {self.latency_metrics.p50_ms:.0f}ms",
            f"- P95: {self.latency_metrics.p95_ms:.0f}ms",
            f"- P99: {self.latency_metrics.p99_ms:.0f}ms",
            "",
            "### Cost Analysis",
            f"- Total cost: ${self.cost_breakdown.total_cost_usd:.4f}",
            f"- Average per request: ${self.cost_breakdown.avg_cost_per_request:.6f}",
            f"- Projected monthly: ${self.cost_breakdown.projected_monthly_cost:.2f}",
            "",
            "### Reliability",
            f"- Success rate: {self.error_patterns.success_rate * 100:.1f}%",
            f"- Error rate: {self.error_patterns.error_rate * 100:.1f}%",
            f"- Refusal rate: {self.error_patterns.refusal_rate * 100:.1f}%",
            "",
            "### Prompt Complexity",
            f"- Average prompt length: {self.prompt_complexity.avg_prompt_length:.0f} chars",
            f"- Tool usage: {self.prompt_complexity.tool_usage_rate * 100:.1f}%",
            f"- System prompt usage: {self.prompt_complexity.system_prompt_usage_rate * 100:.1f}%",
            "",
            "### Current Models in Use",
        ]
        
        for mp in self.model_performance[:5]:  # Top 5 models
            lines.append(
                f"- {mp.provider}/{mp.model}: {mp.request_count} requests, "
                f"{mp.avg_latency_ms:.0f}ms avg, ${mp.avg_cost_per_request:.6f}/req"
            )
        
        lines.append("")
        lines.append("### Key Insights")
        for insight in self.key_insights[:5]:
            lines.append(f"- {insight}")
        
        if self.criteria_assessment:
            lines.append("")
            lines.append("### Criteria Assessment")
            for key, value in self.criteria_assessment.items():
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)


class LogAnalyzer:
    """
    Analyzes user-selected logs to generate insights for model selection.
    
    This analyzer:
    - Takes specific log IDs selected by the user
    - Generates comprehensive statistical analysis
    - Produces insights formatted for the Model Selector Agent
    - Assesses logs against project success criteria
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def analyze_logs(
        self,
        log_ids: list[uuid.UUID],
        project: Project | None = None,
        success_criteria: SuccessCriteria | None = None,
    ) -> AnalysisReport:
        """
        Analyze selected logs and generate a comprehensive report.
        
        Args:
            log_ids: List of log entry IDs to analyze
            project: Optional project for context
            success_criteria: Optional criteria to assess against
            
        Returns:
            AnalysisReport with all statistical analyses
        """
        logger.info(f"Analyzing {len(log_ids)} logs")
        
        # Load logs into DataFrame
        df = await self._load_logs(log_ids)
        
        if df.empty:
            raise ValueError("No logs found with the provided IDs")
        
        # Run all analyses
        token_dist = self._analyze_tokens(df)
        latency = self._analyze_latency(df)
        cost = self._analyze_cost(df)
        complexity = self._analyze_prompt_complexity(df)
        errors = self._analyze_errors(df)
        model_perf = self._analyze_model_performance(df)
        trends = self._analyze_time_trends(df)
        
        # Generate insights
        insights = self._generate_insights(
            df, token_dist, latency, cost, complexity, errors, model_perf
        )
        
        # Assess against criteria if provided
        criteria_assessment = {}
        if success_criteria:
            criteria_assessment = self._assess_criteria(
                df, latency, cost, errors, success_criteria
            )
        
        # Determine date range
        date_start = df["timestamp"].min()
        date_end = df["timestamp"].max()
        
        return AnalysisReport(
            analysis_id=str(uuid.uuid4()),
            project_id=str(project.id) if project else "unknown",
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            log_count=len(df),
            date_range_start=date_start.isoformat() if pd.notna(date_start) else None,
            date_range_end=date_end.isoformat() if pd.notna(date_end) else None,
            token_distribution=token_dist,
            latency_metrics=latency,
            cost_breakdown=cost,
            prompt_complexity=complexity,
            error_patterns=errors,
            model_performance=model_perf,
            time_trends=trends,
            key_insights=insights,
            criteria_assessment=criteria_assessment,
        )

    async def _load_logs(self, log_ids: list[uuid.UUID]) -> pd.DataFrame:
        """Load logs by IDs into a DataFrame."""
        result = await self.session.execute(
            select(LogEntry).where(LogEntry.id.in_(log_ids))
        )
        logs = result.scalars().all()
        
        if not logs:
            return pd.DataFrame()
        
        data = []
        for log in logs:
            # Extract completion from response_data if available
            completion = None
            if log.response_data:
                choices = log.response_data.get("choices", [])
                if choices:
                    completion = choices[0].get("message", {}).get("content", "")
            
            data.append({
                "id": log.id,
                "timestamp": log.timestamp,
                "model": log.model,
                "provider": log.provider,
                "prompt": log.prompt,
                "system_prompt": log.system_prompt,
                "completion": completion,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "total_tokens": log.total_tokens,
                "latency_ms": log.latency_ms,
                "cost_usd": log.cost_usd,
                "status": log.status,
                "refusal": log.refusal,
                "error_message": log.error_message,
                "error_code": log.error_code,
                "tool_calls": log.tool_calls,
                "request_data": log.request_data,
                "response_data": log.response_data,
            })
        
        return pd.DataFrame(data)

    def _analyze_tokens(self, df: pd.DataFrame) -> TokenDistribution:
        """Analyze token usage distribution."""
        def get_stats(series: pd.Series) -> dict[str, float]:
            s = series.dropna()
            if len(s) == 0:
                return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
            return {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "p50": float(np.percentile(s, 50)),
                "p95": float(np.percentile(s, 95)),
                "p99": float(np.percentile(s, 99)),
            }
        
        def get_histogram(series: pd.Series, bins: int = 20) -> list[tuple[float, int]]:
            s = series.dropna()
            if len(s) == 0:
                return []
            counts, edges = np.histogram(s, bins=bins)
            return [(float(edges[i]), int(counts[i])) for i in range(len(counts))]
        
        return TokenDistribution(
            input_tokens=get_stats(df["input_tokens"]),
            output_tokens=get_stats(df["output_tokens"]),
            total_tokens=get_stats(df["total_tokens"]),
            histogram_input=get_histogram(df["input_tokens"]),
            histogram_output=get_histogram(df["output_tokens"]),
        )

    def _analyze_latency(self, df: pd.DataFrame) -> LatencyMetrics:
        """Analyze latency distribution and percentiles."""
        latency = df["latency_ms"].dropna()
        
        if len(latency) == 0:
            return LatencyMetrics(
                min_ms=0, max_ms=0, mean_ms=0, std_ms=0,
                p50_ms=0, p75_ms=0, p90_ms=0, p95_ms=0, p99_ms=0,
                histogram=[],
            )
        
        counts, edges = np.histogram(latency, bins=20)
        histogram = [(float(edges[i]), int(counts[i])) for i in range(len(counts))]
        
        return LatencyMetrics(
            min_ms=float(latency.min()),
            max_ms=float(latency.max()),
            mean_ms=float(latency.mean()),
            std_ms=float(latency.std()) if len(latency) > 1 else 0.0,
            p50_ms=float(np.percentile(latency, 50)),
            p75_ms=float(np.percentile(latency, 75)),
            p90_ms=float(np.percentile(latency, 90)),
            p95_ms=float(np.percentile(latency, 95)),
            p99_ms=float(np.percentile(latency, 99)),
            histogram=histogram,
        )

    def _analyze_cost(self, df: pd.DataFrame) -> CostBreakdown:
        """Analyze cost breakdown by model and provider."""
        total_cost = float(df["cost_usd"].sum())
        avg_cost = float(df["cost_usd"].mean()) if len(df) > 0 else 0.0
        
        # Cost by model
        cost_by_model = df.groupby("model")["cost_usd"].sum().to_dict()
        cost_by_model = {k: float(v) for k, v in cost_by_model.items()}
        
        # Cost by provider
        cost_by_provider = df.groupby("provider")["cost_usd"].sum().to_dict()
        cost_by_provider = {k: float(v) for k, v in cost_by_provider.items()}
        
        # Project monthly cost based on date range
        if len(df) > 0 and pd.notna(df["timestamp"].min()):
            date_range = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
            if date_range > 0:
                # Requests per second * seconds in month * avg cost
                requests_per_second = len(df) / date_range
                seconds_in_month = 30 * 24 * 3600
                projected_monthly = requests_per_second * seconds_in_month * avg_cost
            else:
                projected_monthly = avg_cost * len(df) * 30  # Assume daily rate
        else:
            projected_monthly = 0.0
        
        return CostBreakdown(
            total_cost_usd=total_cost,
            avg_cost_per_request=avg_cost,
            cost_by_model=cost_by_model,
            cost_by_provider=cost_by_provider,
            projected_monthly_cost=projected_monthly,
        )

    def _analyze_prompt_complexity(self, df: pd.DataFrame) -> PromptComplexity:
        """Analyze prompt complexity metrics."""
        # Prompt lengths
        prompt_lengths = df["prompt"].apply(lambda x: len(x) if x else 0)
        
        # Messages per request (from request_data)
        def count_messages(req_data):
            if not req_data:
                return 1
            messages = req_data.get("messages", [])
            return len(messages) if messages else 1
        
        messages_per_request = df["request_data"].apply(count_messages)
        
        # Tool usage
        tool_usage = df["tool_calls"].apply(lambda x: x is not None and len(x) > 0)
        tool_usage_rate = float(tool_usage.mean()) if len(tool_usage) > 0 else 0.0
        
        # System prompt usage
        system_usage = df["system_prompt"].apply(lambda x: x is not None and len(str(x)) > 0)
        system_usage_rate = float(system_usage.mean()) if len(system_usage) > 0 else 0.0
        
        # Complexity distribution based on token count
        def classify_complexity(tokens):
            if tokens < 500:
                return "simple"
            elif tokens < 2000:
                return "moderate"
            else:
                return "complex"
        
        complexity_classes = df["total_tokens"].apply(classify_complexity)
        complexity_dist = complexity_classes.value_counts().to_dict()
        
        return PromptComplexity(
            avg_prompt_length=float(prompt_lengths.mean()) if len(prompt_lengths) > 0 else 0.0,
            max_prompt_length=int(prompt_lengths.max()) if len(prompt_lengths) > 0 else 0,
            min_prompt_length=int(prompt_lengths.min()) if len(prompt_lengths) > 0 else 0,
            avg_messages_per_request=float(messages_per_request.mean()) if len(messages_per_request) > 0 else 1.0,
            tool_usage_rate=tool_usage_rate,
            system_prompt_usage_rate=system_usage_rate,
            complexity_distribution=complexity_dist,
        )

    def _analyze_errors(self, df: pd.DataFrame) -> ErrorPatterns:
        """Analyze error and refusal patterns."""
        total = len(df)
        
        if total == 0:
            return ErrorPatterns(
                total_requests=0,
                success_rate=0.0,
                error_rate=0.0,
                refusal_rate=0.0,
                timeout_rate=0.0,
                error_codes={},
                common_error_messages=[],
            )
        
        success_count = len(df[df["status"] == "success"])
        error_count = len(df[df["status"] == "error"])
        refusal_count = len(df[df["refusal"] == True])
        timeout_count = len(df[df["status"] == "timeout"])
        
        # Error codes
        error_codes = df[df["error_code"].notna()]["error_code"].value_counts().to_dict()
        
        # Common error messages
        error_msgs = df[df["error_message"].notna()]["error_message"].value_counts().head(5)
        common_errors = error_msgs.index.tolist()
        
        return ErrorPatterns(
            total_requests=total,
            success_rate=success_count / total,
            error_rate=error_count / total,
            refusal_rate=refusal_count / total,
            timeout_rate=timeout_count / total,
            error_codes=error_codes,
            common_error_messages=common_errors,
        )

    def _analyze_model_performance(self, df: pd.DataFrame) -> list[ModelPerformance]:
        """Analyze performance per model."""
        results = []
        
        for (model, provider), group in df.groupby(["model", "provider"]):
            latencies = group["latency_ms"].dropna()
            
            results.append(ModelPerformance(
                model=str(model),
                provider=str(provider),
                request_count=len(group),
                avg_latency_ms=float(latencies.mean()) if len(latencies) > 0 else 0.0,
                p95_latency_ms=float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0,
                avg_cost_per_request=float(group["cost_usd"].mean()),
                success_rate=float((group["status"] == "success").mean()),
                refusal_rate=float(group["refusal"].mean()),
                avg_input_tokens=float(group["input_tokens"].mean()),
                avg_output_tokens=float(group["output_tokens"].mean()),
            ))
        
        # Sort by request count descending
        results.sort(key=lambda x: x.request_count, reverse=True)
        
        return results

    def _analyze_time_trends(self, df: pd.DataFrame) -> TimeSeriesTrends:
        """Analyze time-based trends."""
        if df.empty or df["timestamp"].isna().all():
            return TimeSeriesTrends(
                requests_per_hour={},
                requests_per_day={},
                latency_trend=[],
                cost_trend=[],
                peak_hours=[],
            )
        
        # Requests per hour of day
        df["hour"] = df["timestamp"].dt.hour
        requests_per_hour = df["hour"].value_counts().sort_index().to_dict()
        
        # Requests per day
        df["date"] = df["timestamp"].dt.date.astype(str)
        requests_per_day = df["date"].value_counts().sort_index().to_dict()
        
        # Latency trend by day
        latency_by_day = df.groupby("date")["latency_ms"].mean()
        latency_trend = [(str(k), float(v)) for k, v in latency_by_day.items()]
        
        # Cost trend by day
        cost_by_day = df.groupby("date")["cost_usd"].sum()
        cost_trend = [(str(k), float(v)) for k, v in cost_by_day.items()]
        
        # Peak hours (top 3)
        peak_hours = list(
            df["hour"].value_counts().sort_values(ascending=False).head(3).index
        )
        
        return TimeSeriesTrends(
            requests_per_hour={int(k): int(v) for k, v in requests_per_hour.items()},
            requests_per_day=requests_per_day,
            latency_trend=latency_trend,
            cost_trend=cost_trend,
            peak_hours=[int(h) for h in peak_hours],
        )

    def _generate_insights(
        self,
        df: pd.DataFrame,
        token_dist: TokenDistribution,
        latency: LatencyMetrics,
        cost: CostBreakdown,
        complexity: PromptComplexity,
        errors: ErrorPatterns,
        model_perf: list[ModelPerformance],
    ) -> list[str]:
        """Generate human-readable insights from the analysis."""
        insights = []
        
        # Token insights
        avg_tokens = token_dist.total_tokens.get("mean", 0)
        if avg_tokens > 3000:
            insights.append(f"High average token usage ({avg_tokens:.0f} tokens/request) suggests complex prompts that may benefit from models with larger context windows")
        elif avg_tokens < 500:
            insights.append(f"Low average token usage ({avg_tokens:.0f} tokens/request) indicates simpler tasks suitable for faster, cheaper models")
        
        # Latency insights
        if latency.p95_ms > 5000:
            insights.append(f"P95 latency is {latency.p95_ms:.0f}ms - consider faster models if latency is critical")
        if latency.std_ms > latency.mean_ms:
            insights.append("High latency variance detected - some requests take significantly longer than others")
        
        # Cost insights
        if cost.projected_monthly_cost > 100:
            insights.append(f"Projected monthly cost is ${cost.projected_monthly_cost:.2f} - cost optimization may be beneficial")
        
        # Error insights
        if errors.error_rate > 0.05:
            insights.append(f"Error rate of {errors.error_rate * 100:.1f}% is above 5% threshold - investigate error causes")
        if errors.refusal_rate > 0.02:
            insights.append(f"Refusal rate of {errors.refusal_rate * 100:.1f}% detected - some prompts may be triggering safety filters")
        
        # Complexity insights
        if complexity.tool_usage_rate > 0.3:
            insights.append(f"{complexity.tool_usage_rate * 100:.0f}% of requests use tools - prioritize models with strong function calling")
        
        # Model performance insights
        if len(model_perf) > 1:
            best_latency = min(model_perf, key=lambda x: x.avg_latency_ms)
            best_cost = min(model_perf, key=lambda x: x.avg_cost_per_request)
            if best_latency.model != best_cost.model:
                insights.append(
                    f"Trade-off detected: {best_latency.provider}/{best_latency.model} is fastest, "
                    f"but {best_cost.provider}/{best_cost.model} is cheapest"
                )
        
        return insights

    def _assess_criteria(
        self,
        df: pd.DataFrame,
        latency: LatencyMetrics,
        cost: CostBreakdown,
        errors: ErrorPatterns,
        criteria: SuccessCriteria,
    ) -> dict[str, Any]:
        """Assess current performance against success criteria."""
        assessment = {}
        
        # Latency assessment
        if criteria.max_latency_ms:
            meets_latency = latency.mean_ms <= criteria.max_latency_ms
            assessment["latency"] = {
                "target": criteria.max_latency_ms,
                "current": latency.mean_ms,
                "meets_target": meets_latency,
                "gap_pct": ((latency.mean_ms - criteria.max_latency_ms) / criteria.max_latency_ms * 100)
                    if not meets_latency else 0,
            }
        
        # P95 latency assessment
        if criteria.max_latency_p95_ms:
            meets_p95 = latency.p95_ms <= criteria.max_latency_p95_ms
            assessment["latency_p95"] = {
                "target": criteria.max_latency_p95_ms,
                "current": latency.p95_ms,
                "meets_target": meets_p95,
            }
        
        # Cost assessment
        if criteria.max_cost_per_request_usd:
            meets_cost = cost.avg_cost_per_request <= criteria.max_cost_per_request_usd
            assessment["cost_per_request"] = {
                "target": criteria.max_cost_per_request_usd,
                "current": cost.avg_cost_per_request,
                "meets_target": meets_cost,
            }
        
        # Refusal rate assessment
        if criteria.max_refusal_rate:
            meets_refusal = errors.refusal_rate <= criteria.max_refusal_rate
            assessment["refusal_rate"] = {
                "target": criteria.max_refusal_rate,
                "current": errors.refusal_rate,
                "meets_target": meets_refusal,
            }
        
        # Overall assessment
        all_targets = [v.get("meets_target", True) for v in assessment.values() if isinstance(v, dict)]
        assessment["overall_meets_criteria"] = all(all_targets) if all_targets else True
        
        return assessment
