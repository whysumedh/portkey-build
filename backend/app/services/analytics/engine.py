"""
Analytics Engine - Safe, deterministic statistical analysis.

This engine provides read-only statistical operations on log data.
Key safety properties:
- NO arbitrary code execution
- NO filesystem or network access
- Deterministic operations
- Budget-limited (max rows, timeout)
- Cached results
"""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.log_entry import LogEntry

logger = get_logger(__name__)


class AnalyticsEngineError(Exception):
    """Base exception for analytics engine errors."""
    pass


class UnsupportedOperationError(AnalyticsEngineError):
    """Raised when an unsupported operation is requested."""
    pass


class InsufficientDataError(AnalyticsEngineError):
    """Raised when there's not enough data for analysis."""
    pass


class AnalyticsEngine:
    """
    Safe analytics engine for statistical operations on log data.
    
    This is the ONLY way to perform statistical analysis.
    All operations are:
    - Read-only
    - Deterministic
    - Safe (no code execution)
    - Budget-limited
    """

    # Allowed fields for analysis (safety boundary)
    ALLOWED_FIELDS = {
        "prompt_length": "derived",  # len(prompt)
        "input_tokens": "column",
        "output_tokens": "column",
        "total_tokens": "column",
        "latency_ms": "column",
        "cost_usd": "column",
        "refusal": "column",
        "status": "column",
        "model": "column",
        "provider": "column",
        "timestamp": "column",
        "hour_of_day": "derived",  # timestamp.hour
        "day_of_week": "derived",  # timestamp.dayofweek
    }

    # Maximum rows to load (safety limit)
    MAX_ROWS = 100_000

    def __init__(
        self,
        session: AsyncSession,
        cache: dict[str, Any] | None = None,
    ):
        self.session = session
        self.cache = cache or {}
        self._df_cache: dict[str, pd.DataFrame] = {}

    async def _load_data(
        self,
        project_id: uuid.UUID,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """Load log data into a DataFrame with safety limits."""
        cache_key = self._make_cache_key(project_id, start_date, end_date, filters)
        
        if cache_key in self._df_cache:
            return self._df_cache[cache_key]

        # Build query
        query = select(
            LogEntry.id,
            LogEntry.timestamp,
            LogEntry.prompt,
            LogEntry.model,
            LogEntry.provider,
            LogEntry.input_tokens,
            LogEntry.output_tokens,
            LogEntry.total_tokens,
            LogEntry.latency_ms,
            LogEntry.cost_usd,
            LogEntry.status,
            LogEntry.refusal,
        ).where(LogEntry.project_id == project_id)

        if start_date:
            query = query.where(LogEntry.timestamp >= start_date)
        if end_date:
            query = query.where(LogEntry.timestamp <= end_date)
        
        # Apply filters
        if filters:
            if "model" in filters:
                query = query.where(LogEntry.model == filters["model"])
            if "provider" in filters:
                query = query.where(LogEntry.provider == filters["provider"])
            if "status" in filters:
                query = query.where(LogEntry.status == filters["status"])

        # Limit rows for safety
        query = query.limit(self.MAX_ROWS)

        result = await self.session.execute(query)
        rows = result.all()

        if not rows:
            raise InsufficientDataError("No data found for the specified criteria")

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "id", "timestamp", "prompt", "model", "provider",
            "input_tokens", "output_tokens", "total_tokens",
            "latency_ms", "cost_usd", "status", "refusal",
        ])

        # Add derived fields
        df["prompt_length"] = df["prompt"].apply(lambda x: len(x) if x else 0)
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["refusal"] = df["refusal"].astype(int)

        self._df_cache[cache_key] = df
        logger.info(f"Loaded {len(df)} rows for analysis")
        
        return df

    def _make_cache_key(
        self,
        project_id: uuid.UUID,
        start_date: datetime | None,
        end_date: datetime | None,
        filters: dict | None,
    ) -> str:
        """Create a cache key for the query parameters."""
        key_data = f"{project_id}:{start_date}:{end_date}:{filters}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _validate_field(self, field: str) -> None:
        """Validate that a field is allowed for analysis."""
        if field not in self.ALLOWED_FIELDS:
            raise UnsupportedOperationError(
                f"Field '{field}' is not allowed. Allowed fields: {list(self.ALLOWED_FIELDS.keys())}"
            )

    async def distribution(
        self,
        project_id: uuid.UUID,
        field: str,
        bins: int = 20,
        normalize: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Compute the distribution of a numeric field.
        
        Returns histogram data with bin edges and counts.
        """
        self._validate_field(field)
        df = await self._load_data(project_id, start_date, end_date, filters)

        if field not in df.columns:
            raise UnsupportedOperationError(f"Field '{field}' not found in data")

        series = df[field].dropna()
        
        if len(series) == 0:
            raise InsufficientDataError(f"No valid data for field '{field}'")

        # For categorical fields
        if series.dtype == "object" or field in ["model", "provider", "status"]:
            value_counts = series.value_counts(normalize=normalize)
            return {
                "field": field,
                "bins": value_counts.index.tolist(),
                "counts": value_counts.values.tolist(),
                "percentages": (value_counts.values / len(series) * 100).tolist() if normalize else None,
                "total_count": len(series),
                "missing_count": df[field].isna().sum(),
                "statistics": {
                    "unique": series.nunique(),
                    "mode": series.mode().iloc[0] if len(series.mode()) > 0 else None,
                },
            }

        # For numeric fields
        counts, bin_edges = np.histogram(series, bins=bins)
        
        return {
            "field": field,
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "percentages": (counts / len(series) * 100).tolist() if normalize else None,
            "total_count": len(series),
            "missing_count": df[field].isna().sum(),
            "statistics": {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
            },
        }

    async def percentile(
        self,
        project_id: uuid.UUID,
        metric: str,
        percentiles: list[float] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Compute percentile values for a numeric metric.
        
        Returns the values at specified percentiles (e.g., p50, p95, p99).
        """
        self._validate_field(metric)
        df = await self._load_data(project_id, start_date, end_date, filters)

        if metric not in df.columns:
            raise UnsupportedOperationError(f"Metric '{metric}' not found in data")

        series = df[metric].dropna()
        
        if len(series) == 0:
            raise InsufficientDataError(f"No valid data for metric '{metric}'")

        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]

        # Validate percentiles
        for p in percentiles:
            if not 0 <= p <= 100:
                raise UnsupportedOperationError(f"Percentile must be between 0 and 100, got {p}")

        result_percentiles = {}
        for p in percentiles:
            result_percentiles[p] = float(np.percentile(series, p))

        return {
            "metric": metric,
            "percentiles": result_percentiles,
            "count": len(series),
            "missing_count": df[metric].isna().sum(),
        }

    async def correlation(
        self,
        project_id: uuid.UUID,
        x: str,
        y: str,
        method: str = "pearson",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Compute correlation between two numeric fields.
        
        Supports pearson, spearman, and kendall correlation methods.
        """
        self._validate_field(x)
        self._validate_field(y)
        
        if method not in ["pearson", "spearman", "kendall"]:
            raise UnsupportedOperationError(f"Method must be pearson, spearman, or kendall, got {method}")

        df = await self._load_data(project_id, start_date, end_date, filters)

        if x not in df.columns or y not in df.columns:
            raise UnsupportedOperationError(f"Fields '{x}' or '{y}' not found in data")

        # Drop rows with missing values in either column
        valid_df = df[[x, y]].dropna()
        
        if len(valid_df) < 3:
            raise InsufficientDataError("Need at least 3 data points for correlation")

        # Compute correlation
        if method == "pearson":
            coefficient, p_value = stats.pearsonr(valid_df[x], valid_df[y])
        elif method == "spearman":
            coefficient, p_value = stats.spearmanr(valid_df[x], valid_df[y])
        else:  # kendall
            coefficient, p_value = stats.kendalltau(valid_df[x], valid_df[y])

        # Interpret correlation strength
        abs_coef = abs(coefficient)
        if abs_coef < 0.1:
            interpretation = "negligible"
        elif abs_coef < 0.3:
            interpretation = "weak"
        elif abs_coef < 0.5:
            interpretation = "moderate"
        elif abs_coef < 0.7:
            interpretation = "strong"
        else:
            interpretation = "very strong"
        
        direction = "positive" if coefficient > 0 else "negative"
        interpretation = f"{interpretation} {direction}"

        return {
            "x": x,
            "y": y,
            "method": method,
            "coefficient": float(coefficient),
            "p_value": float(p_value),
            "sample_size": len(valid_df),
            "interpretation": interpretation,
        }

    async def aggregation(
        self,
        project_id: uuid.UUID,
        metric: str,
        group_by: str | None = None,
        aggregations: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Compute aggregations (count, sum, mean, etc.) optionally grouped by a field.
        """
        self._validate_field(metric)
        if group_by:
            self._validate_field(group_by)

        if aggregations is None:
            aggregations = ["count", "mean", "std"]

        # Validate aggregation types
        allowed_aggs = {"count", "sum", "mean", "min", "max", "std", "median"}
        for agg in aggregations:
            if agg not in allowed_aggs:
                raise UnsupportedOperationError(
                    f"Aggregation '{agg}' not supported. Allowed: {allowed_aggs}"
                )

        df = await self._load_data(project_id, start_date, end_date, filters)

        if metric not in df.columns:
            raise UnsupportedOperationError(f"Metric '{metric}' not found in data")

        results = []
        
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by)[metric]
            
            for name, group in grouped:
                row = {"group": name}
                for agg in aggregations:
                    if agg == "median":
                        row[agg] = float(group.median())
                    else:
                        row[agg] = float(getattr(group, agg)())
                results.append(row)
        else:
            row = {"group": "all"}
            series = df[metric].dropna()
            for agg in aggregations:
                if agg == "median":
                    row[agg] = float(series.median())
                elif agg == "count":
                    row[agg] = int(series.count())
                else:
                    row[agg] = float(getattr(series, agg)())
            results.append(row)

        return {
            "metric": metric,
            "group_by": group_by,
            "results": results,
            "total_count": len(df),
        }

    async def clustering(
        self,
        project_id: uuid.UUID,
        features: list[str],
        n_clusters: int = 5,
        method: str = "kmeans",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Perform clustering analysis on log data.
        
        Uses K-means clustering to identify patterns in the data.
        """
        for field in features:
            self._validate_field(field)

        if method not in ["kmeans"]:
            raise UnsupportedOperationError(f"Clustering method '{method}' not supported")

        if not 2 <= n_clusters <= 20:
            raise UnsupportedOperationError("n_clusters must be between 2 and 20")

        df = await self._load_data(project_id, start_date, end_date, filters)

        # Check all features exist
        for f in features:
            if f not in df.columns:
                raise UnsupportedOperationError(f"Feature '{f}' not found in data")

        # Prepare data for clustering
        feature_df = df[features].dropna()
        
        if len(feature_df) < n_clusters:
            raise InsufficientDataError(
                f"Need at least {n_clusters} samples for {n_clusters} clusters"
            )

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Compute cluster statistics
        feature_df = feature_df.copy()
        feature_df["cluster"] = labels
        
        cluster_sizes = feature_df["cluster"].value_counts().sort_index().tolist()
        
        # Get cluster centers in original scale
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers = []
        for i, center in enumerate(centers):
            cluster_centers.append({
                f: float(center[j]) for j, f in enumerate(features)
            })

        return {
            "features": features,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "cluster_centers": cluster_centers,
            "inertia": float(kmeans.inertia_),
        }

    async def sample(
        self,
        project_id: uuid.UUID,
        n: int = 100,
        stratify_by: str | None = None,
        random_seed: int = 42,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Sample log entries, optionally stratified by a field.
        
        Returns a deterministic sample for replay evaluation.
        """
        if stratify_by:
            self._validate_field(stratify_by)

        if not 1 <= n <= 10000:
            raise UnsupportedOperationError("Sample size must be between 1 and 10000")

        df = await self._load_data(project_id, start_date, end_date, filters)

        if len(df) < n:
            # Return all if we have fewer than requested
            sample_df = df
        elif stratify_by and stratify_by in df.columns:
            # Stratified sampling
            from sklearn.model_selection import train_test_split
            
            # Calculate sample fraction
            frac = n / len(df)
            
            # Group by stratification field
            grouped = df.groupby(stratify_by)
            samples = []
            
            for name, group in grouped:
                group_n = max(1, int(len(group) * frac))
                if len(group) <= group_n:
                    samples.append(group)
                else:
                    samples.append(group.sample(n=group_n, random_state=random_seed))
            
            sample_df = pd.concat(samples)
        else:
            # Simple random sampling
            sample_df = df.sample(n=n, random_state=random_seed)

        # Get strata counts if applicable
        strata_counts = None
        if stratify_by and stratify_by in sample_df.columns:
            strata_counts = sample_df[stratify_by].value_counts().to_dict()

        return {
            "sample_size": len(sample_df),
            "stratified": stratify_by is not None,
            "strata_counts": strata_counts,
            "sample_ids": sample_df["id"].tolist(),
        }

    async def summary_statistics(
        self,
        project_id: uuid.UUID,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Get comprehensive summary statistics for a project's logs.
        """
        df = await self._load_data(project_id, start_date, end_date, filters)

        numeric_fields = ["input_tokens", "output_tokens", "total_tokens", "latency_ms", "cost_usd", "prompt_length"]
        
        summary = {
            "total_logs": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if not df["timestamp"].isna().all() else None,
                "end": df["timestamp"].max().isoformat() if not df["timestamp"].isna().all() else None,
            },
            "models": df["model"].unique().tolist(),
            "providers": df["provider"].unique().tolist(),
            "status_distribution": df["status"].value_counts().to_dict(),
            "refusal_rate": float(df["refusal"].mean()),
            "numeric_summaries": {},
        }

        for field in numeric_fields:
            if field in df.columns:
                series = df[field].dropna()
                if len(series) > 0:
                    summary["numeric_summaries"][field] = {
                        "count": int(len(series)),
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "p25": float(series.quantile(0.25)),
                        "p50": float(series.quantile(0.50)),
                        "p75": float(series.quantile(0.75)),
                        "p95": float(series.quantile(0.95)),
                        "max": float(series.max()),
                    }

        return summary
