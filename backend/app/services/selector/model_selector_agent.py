"""
Model Selector Agent - AI-powered model selection using Claude 3.7 Sonnet.

This agent analyzes project criteria, capability expectations, and log statistics
to intelligently select the best candidate models for replay evaluation.

Key features:
- Filters models by LiveBench benchmark scores based on user expectations
- Uses Claude 3.7 Sonnet for intelligent model ranking
- Considers cost, latency, quality trade-offs
- Supports Portkey's @provider/model format for seamless model access
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from portkey_ai import AsyncPortkey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.model_catalog import ModelCatalogEntry
from app.models.project import Project, SuccessCriteria, ToleranceLevels, CapabilityExpectations
from app.services.analysis.log_analyzer import AnalysisReport
from app.services.catalog.model_catalog_service import ModelCatalogService

logger = get_logger(__name__)


# System prompt for the Model Selector Agent
MODEL_SELECTOR_SYSTEM_PROMPT = """You are an expert AI model selection agent. Your task is to analyze project requirements, capability expectations, and usage patterns to recommend the best LLM models for evaluation.

You will receive:
1. Project success criteria (latency, cost, quality targets)
2. User's capability expectations (minimum LiveBench benchmark scores)
3. Statistical analysis of production logs
4. Available models from the catalog with pricing, capabilities, AND LiveBench benchmark scores

Your job is to select 3-5 candidate models that:
- Meet ALL specified minimum benchmark scores (CRITICAL - models not meeting these are pre-filtered)
- Meet the project's hard requirements (latency, cost limits)
- Are well-suited for the detected use cases
- Provide good diversity (e.g., different price/quality trade-offs)
- Have the required capabilities (function calling, vision, etc.)

LIVEBENCH BENCHMARK DIMENSIONS:
- Reasoning: Logical reasoning and problem-solving ability
- Coding: Code generation, review, debugging capability
- Agentic Coding: Autonomous coding with tool use
- Mathematics: Mathematical problem-solving
- Data Analysis: Data interpretation and analysis
- Language: Language understanding and generation
- Instruction Following (IF): Following complex instructions accurately

IMPORTANT CONSIDERATIONS:
- Models have already been filtered to meet minimum benchmark requirements
- Prioritize models with HIGHER scores in the dimensions the user cares about most
- If tool/function calling is detected, only recommend models that support it
- If vision/multimodal is detected, only recommend models that support it
- Balance between cost optimization and quality based on the project's cost sensitivity
- Consider context window requirements based on token usage patterns
- When benchmark scores are similar, consider price/performance ratio
- Explain HOW the benchmark scores influenced your selection

OUTPUT FORMAT:
You must respond with a valid JSON object containing:
{
    "candidates": [
        {
            "provider": "provider_name",
            "model": "model_id",
            "rank": 1,
            "reasoning": "Why this model is recommended, including benchmark strengths",
            "expected_cost_per_request": 0.001,
            "expected_latency_ms": 500,
            "strengths": ["strength1", "strength2"],
            "concerns": ["concern1"],
            "benchmark_highlights": "Key benchmark scores that influenced selection"
        }
    ],
    "selection_reasoning": "Overall explanation of the selection strategy and how benchmarks were considered",
    "key_requirements_identified": ["requirement1", "requirement2"],
    "excluded_models": [
        {
            "model": "model_id",
            "reason": "Why excluded (e.g., benchmark score too low)"
        }
    ]
}"""


@dataclass
class CandidateModel:
    """A candidate model selected for replay evaluation."""
    provider: str
    model: str
    rank: int
    reasoning: str
    expected_cost_per_request: float
    expected_latency_ms: float
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    
    # Additional metadata from catalog
    tier: str | None = None
    use_cases: list[str] = field(default_factory=list)
    quality_score: int | None = None
    speed_score: int | None = None
    supports_function_calling: bool = False
    supports_vision: bool = False
    context_window: int | None = None


@dataclass
class ModelSelectionResult:
    """Result from the Model Selector Agent."""
    selection_id: str
    project_id: str
    selected_at: str
    candidates: list[CandidateModel]
    selection_reasoning: str
    key_requirements: list[str]
    excluded_models: list[dict[str, str]]
    analysis_summary: str
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "selection_id": self.selection_id,
            "project_id": self.project_id,
            "selected_at": self.selected_at,
            "candidates": [
                {
                    "provider": c.provider,
                    "model": c.model,
                    "rank": c.rank,
                    "reasoning": c.reasoning,
                    "expected_cost_per_request": c.expected_cost_per_request,
                    "expected_latency_ms": c.expected_latency_ms,
                    "strengths": c.strengths,
                    "concerns": c.concerns,
                    "tier": c.tier,
                    "use_cases": c.use_cases,
                    "quality_score": c.quality_score,
                    "speed_score": c.speed_score,
                    "supports_function_calling": c.supports_function_calling,
                    "supports_vision": c.supports_vision,
                    "context_window": c.context_window,
                }
                for c in self.candidates
            ],
            "selection_reasoning": self.selection_reasoning,
            "key_requirements": self.key_requirements,
            "excluded_models": self.excluded_models,
            "analysis_summary": self.analysis_summary,
            "confidence": self.confidence,
        }


class ModelSelectorAgent:
    """
    AI-powered model selection using Claude 3.7 Sonnet via Portkey.
    
    This agent:
    - Receives project criteria and log analysis
    - Uses Claude 3.7 Sonnet to reason about best candidates
    - Returns ranked list of models for replay evaluation
    
    Uses Portkey's @provider/model format (e.g., @bedrock/us.anthropic.claude-3-7-sonnet)
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.catalog_service = ModelCatalogService(session)
        
        # Initialize Portkey client - uses @provider/model format, no virtual keys needed
        self.portkey = AsyncPortkey(
            api_key=settings.portkey_api_key,
        )

    async def select_models(
        self,
        project: Project,
        analysis_report: AnalysisReport,
        max_candidates: int | None = None,
    ) -> ModelSelectionResult:
        """
        Select candidate models for replay evaluation.
        
        Args:
            project: The project with success criteria and capability expectations
            analysis_report: Statistical analysis of selected logs
            max_candidates: Maximum number of candidates to select
            
        Returns:
            ModelSelectionResult with ranked candidate models
        """
        max_candidates = max_candidates or settings.model_selector_max_candidates
        
        logger.info(
            "Starting model selection",
            project_id=str(project.id),
            log_count=analysis_report.log_count,
            has_expectations=project.capability_expectations is not None,
        )

        # Get available models from catalog
        catalog_models = await self.catalog_service.get_all_models(active_only=True)
        
        if not catalog_models:
            raise ValueError("No models available in catalog")

        # Filter models by capability expectations (LiveBench benchmarks)
        filtered_models, excluded_by_benchmarks = self._filter_by_expectations(
            catalog_models,
            project.capability_expectations,
        )
        
        if not filtered_models:
            logger.warning(
                "No models meet capability expectations, using all models",
                project_id=str(project.id),
            )
            filtered_models = catalog_models
            excluded_by_benchmarks = []
        else:
            logger.info(
                "Filtered models by expectations",
                original_count=len(catalog_models),
                filtered_count=len(filtered_models),
                excluded_count=len(excluded_by_benchmarks),
            )

        # Build prompt for Claude
        prompt = self._build_selection_prompt(
            project=project,
            analysis=analysis_report,
            models=filtered_models,
            max_candidates=max_candidates,
            excluded_by_benchmarks=excluded_by_benchmarks,
        )

        # Call Claude 3.5 Sonnet via Portkey
        try:
            response = await self.portkey.chat.completions.create(
                model=settings.model_selector_model,
                messages=[
                    {"role": "system", "content": MODEL_SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # Low temperature for more deterministic selection
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            selection_data = json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model selection response: {e}")
            # Fallback to deterministic selection
            return await self._fallback_selection(project, analysis_report, catalog_models, max_candidates)
        except Exception as e:
            logger.error(f"Model selection agent failed: {e}")
            # Fallback to deterministic selection
            return await self._fallback_selection(project, analysis_report, catalog_models, max_candidates)

        # Build result
        candidates = []
        for i, candidate_data in enumerate(selection_data.get("candidates", [])[:max_candidates]):
            # Find catalog entry for this model
            catalog_entry = next(
                (m for m in catalog_models 
                 if m.model == candidate_data["model"] and m.provider == candidate_data["provider"]),
                None
            )
            
            candidate = CandidateModel(
                provider=candidate_data["provider"],
                model=candidate_data["model"],
                rank=candidate_data.get("rank", i + 1),
                reasoning=candidate_data.get("reasoning", ""),
                expected_cost_per_request=candidate_data.get("expected_cost_per_request", 0),
                expected_latency_ms=candidate_data.get("expected_latency_ms", 0),
                strengths=candidate_data.get("strengths", []),
                concerns=candidate_data.get("concerns", []),
            )
            
            # Enrich with catalog data
            if catalog_entry:
                candidate.tier = catalog_entry.tier
                candidate.use_cases = catalog_entry.use_cases or []
                candidate.quality_score = catalog_entry.quality_score
                candidate.speed_score = catalog_entry.speed_score
                candidate.supports_function_calling = catalog_entry.supports_function_calling
                candidate.supports_vision = catalog_entry.supports_vision
                candidate.context_window = catalog_entry.context_window
            
            candidates.append(candidate)

        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(selection_data, candidates, analysis_report)

        return ModelSelectionResult(
            selection_id=str(uuid.uuid4()),
            project_id=str(project.id),
            selected_at=datetime.now(timezone.utc).isoformat(),
            candidates=candidates,
            selection_reasoning=selection_data.get("selection_reasoning", ""),
            key_requirements=selection_data.get("key_requirements_identified", []),
            excluded_models=selection_data.get("excluded_models", []),
            analysis_summary=analysis_report.to_prompt_context()[:500],  # Truncated summary
            confidence=confidence,
        )

    def _filter_by_expectations(
        self,
        models: list[ModelCatalogEntry],
        expectations: CapabilityExpectations | None,
    ) -> tuple[list[ModelCatalogEntry], list[dict[str, str]]]:
        """
        Filter models that don't meet minimum benchmark scores.
        
        Args:
            models: List of catalog models
            expectations: User's capability expectations (minimum scores)
            
        Returns:
            Tuple of (filtered_models, excluded_models_with_reasons)
        """
        if not expectations or not expectations.has_any_expectation():
            return models, []
        
        filtered = []
        excluded = []
        
        expectations_dict = expectations.to_dict()
        
        for model in models:
            meets, failed_dimensions = model.meets_expectations(expectations_dict)
            
            if meets:
                filtered.append(model)
            else:
                # Build reason string
                failed_details = []
                for dim in failed_dimensions:
                    expected = expectations_dict.get(dim)
                    actual = getattr(model, f"livebench_{dim}", None)
                    if actual is not None:
                        failed_details.append(f"{dim}: {actual:.1f} < {expected}")
                    else:
                        failed_details.append(f"{dim}: no score < {expected}")
                
                excluded.append({
                    "model": f"{model.provider}/{model.model}",
                    "reason": f"Below minimum benchmark: {', '.join(failed_details)}",
                })
        
        return filtered, excluded

    def _build_selection_prompt(
        self,
        project: Project,
        analysis: AnalysisReport,
        models: list[ModelCatalogEntry],
        max_candidates: int,
        excluded_by_benchmarks: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the prompt for the model selector agent."""
        # Project criteria section
        criteria_section = self._format_criteria(project)
        
        # Capability expectations section
        expectations_section = self._format_expectations(project.capability_expectations)
        
        # Analysis report section
        analysis_section = analysis.to_prompt_context()
        
        # Available models section (includes benchmark scores)
        models_section = self._format_models(models)
        
        # Pre-excluded models section
        excluded_section = ""
        if excluded_by_benchmarks:
            excluded_lines = ["## Pre-Excluded Models (Did Not Meet Benchmark Requirements)"]
            for exc in excluded_by_benchmarks[:10]:  # Limit to 10
                excluded_lines.append(f"- {exc['model']}: {exc['reason']}")
            excluded_section = "\n".join(excluded_lines) + "\n"
        
        prompt = f"""## Task
Select the {max_candidates} best candidate models for replay evaluation based on the project requirements, capability expectations, and log analysis below.

## Project Information
**Name:** {project.name}
**Purpose:** {project.agent_purpose}
**Current Model:** {project.current_provider}/{project.current_model if project.current_model else 'Not set'}

{criteria_section}

{expectations_section}

{analysis_section}

## Available Models (Meeting Benchmark Requirements)
{models_section}

{excluded_section}
## Instructions
1. Analyze the capability expectations to understand which benchmarks matter most
2. Review the log statistics to understand the workload characteristics
3. Match requirements against model capabilities AND benchmark scores
4. Select {max_candidates} diverse candidates that cover different trade-offs
5. Rank them by overall fit, with higher benchmark scores in key areas preferred
6. Provide clear reasoning that references specific benchmark strengths

Respond with a JSON object following the specified format."""

        return prompt

    def _format_expectations(self, expectations: CapabilityExpectations | None) -> str:
        """Format capability expectations for the prompt."""
        lines = ["## Capability Expectations (Minimum LiveBench Scores)"]
        
        if not expectations or not expectations.has_any_expectation():
            lines.append("- No specific capability expectations defined")
            lines.append("- Select based on general quality and use case fit")
            return "\n".join(lines)
        
        exp_dict = expectations.to_dict()
        dimension_labels = {
            "reasoning": "Reasoning",
            "coding": "Coding",
            "agentic_coding": "Agentic Coding",
            "mathematics": "Mathematics",
            "data_analysis": "Data Analysis",
            "language": "Language",
            "instruction_following": "Instruction Following (IF)",
        }
        
        for key, value in exp_dict.items():
            if value is not None:
                label = dimension_labels.get(key, key.replace("_", " ").title())
                lines.append(f"- **{label}:** >= {value:.0f}")
        
        lines.append("")
        lines.append("**Note:** Models not meeting these minimums have been pre-filtered.")
        lines.append("Prioritize models with HIGHER scores in the dimensions specified above.")
        
        return "\n".join(lines)

    def _format_criteria(self, project: Project) -> str:
        """Format project success criteria for the prompt."""
        lines = ["## Success Criteria"]
        
        if project.success_criteria:
            criteria = project.success_criteria
            if criteria.max_latency_ms:
                lines.append(f"- **Max Latency:** {criteria.max_latency_ms}ms")
            if criteria.max_latency_p95_ms:
                lines.append(f"- **Max P95 Latency:** {criteria.max_latency_p95_ms}ms")
            if criteria.max_cost_per_request_usd:
                lines.append(f"- **Max Cost/Request:** ${criteria.max_cost_per_request_usd}")
            if criteria.max_monthly_cost_usd:
                lines.append(f"- **Max Monthly Cost:** ${criteria.max_monthly_cost_usd}")
            if criteria.min_accuracy:
                lines.append(f"- **Min Accuracy:** {criteria.min_accuracy * 100}%")
            if criteria.min_quality_score:
                lines.append(f"- **Min Quality Score:** {criteria.min_quality_score}")
            if criteria.max_refusal_rate:
                lines.append(f"- **Max Refusal Rate:** {criteria.max_refusal_rate * 100}%")
        else:
            lines.append("- No specific criteria defined (optimize for best value)")
        
        if project.tolerance_levels:
            tolerances = project.tolerance_levels
            lines.append(f"\n**Cost Sensitivity:** {tolerances.cost_sensitivity}")
            lines.append(f"**Latency Tolerance:** {tolerances.latency_tolerance_pct * 100}%")
        
        return "\n".join(lines)

    def _format_models(self, models: list[ModelCatalogEntry]) -> str:
        """Format available models for the prompt, including LiveBench scores."""
        lines = []
        
        # Group by tier
        tiers = {"enterprise": [], "premium": [], "standard": [], "budget": [], "other": []}
        for model in models:
            tier = model.tier or "other"
            tiers[tier].append(model)
        
        for tier_name, tier_models in tiers.items():
            if not tier_models:
                continue
            
            lines.append(f"\n### {tier_name.upper()} Tier")
            for m in tier_models[:10]:  # Limit to 10 per tier
                caps = []
                if m.supports_function_calling:
                    caps.append("functions")
                if m.supports_vision:
                    caps.append("vision")
                if m.supports_json_mode:
                    caps.append("json")
                
                caps_str = f" [{', '.join(caps)}]" if caps else ""
                use_cases_str = f" ({', '.join(m.use_cases[:3])})" if m.use_cases else ""
                
                # Format LiveBench scores if available
                bench_str = self._format_benchmark_scores(m)
                
                lines.append(
                    f"- **{m.provider}/{m.model}**: "
                    f"${m.cost_per_1k_input:.4f}/1K in, ${m.cost_per_1k_output:.4f}/1K out, "
                    f"ctx={m.context_window or '?'}K{caps_str}{use_cases_str}"
                )
                if bench_str:
                    lines.append(f"  LiveBench: {bench_str}")
        
        return "\n".join(lines)

    def _format_benchmark_scores(self, model: ModelCatalogEntry) -> str:
        """Format LiveBench benchmark scores for a model."""
        if not model.has_livebench_scores():
            return ""
        
        scores = []
        
        if model.livebench_global_avg is not None:
            scores.append(f"Avg={model.livebench_global_avg:.0f}")
        if model.livebench_reasoning is not None:
            scores.append(f"R={model.livebench_reasoning:.0f}")
        if model.livebench_coding is not None:
            scores.append(f"C={model.livebench_coding:.0f}")
        if model.livebench_agentic_coding is not None:
            scores.append(f"AC={model.livebench_agentic_coding:.0f}")
        if model.livebench_mathematics is not None:
            scores.append(f"M={model.livebench_mathematics:.0f}")
        if model.livebench_data_analysis is not None:
            scores.append(f"DA={model.livebench_data_analysis:.0f}")
        if model.livebench_language is not None:
            scores.append(f"L={model.livebench_language:.0f}")
        if model.livebench_instruction_following is not None:
            scores.append(f"IF={model.livebench_instruction_following:.0f}")
        
        return " | ".join(scores)

    def _calculate_confidence(
        self,
        selection_data: dict,
        candidates: list[CandidateModel],
        analysis: AnalysisReport,
    ) -> float:
        """Calculate confidence score for the selection."""
        confidence = 0.5  # Base confidence
        
        # More candidates = higher confidence
        if len(candidates) >= 3:
            confidence += 0.1
        if len(candidates) >= 5:
            confidence += 0.1
        
        # Clear reasoning = higher confidence
        if selection_data.get("selection_reasoning"):
            confidence += 0.1
        
        # Key requirements identified = higher confidence
        if len(selection_data.get("key_requirements_identified", [])) >= 2:
            confidence += 0.1
        
        # More logs analyzed = higher confidence
        if analysis.log_count >= 50:
            confidence += 0.05
        if analysis.log_count >= 100:
            confidence += 0.05
        
        return min(confidence, 1.0)

    async def _fallback_selection(
        self,
        project: Project,
        analysis: AnalysisReport,
        catalog_models: list[ModelCatalogEntry],
        max_candidates: int,
    ) -> ModelSelectionResult:
        """
        Fallback deterministic selection when AI agent fails.
        Selects models based on simple heuristics and benchmark scores.
        """
        logger.warning("Using fallback deterministic model selection")
        
        # Determine requirements
        needs_function_calling = analysis.prompt_complexity.tool_usage_rate > 0.1
        needs_vision = False  # Would need to detect from logs
        is_cost_sensitive = (
            project.tolerance_levels 
            and project.tolerance_levels.cost_sensitivity == "high"
        )
        
        # First filter by capability expectations
        filtered_models, excluded_by_benchmarks = self._filter_by_expectations(
            catalog_models,
            project.capability_expectations,
        )
        
        if not filtered_models:
            filtered_models = catalog_models
            excluded_by_benchmarks = []
        
        # Score models
        scored_models = []
        for model in filtered_models:
            # Skip if missing required capabilities
            if needs_function_calling and not model.supports_function_calling:
                continue
            if needs_vision and not model.supports_vision:
                continue
            
            # Score based on tier and cost sensitivity
            score = 0
            if model.quality_score:
                score += model.quality_score
            if model.speed_score:
                score += model.speed_score * 0.5
            
            # Bonus for LiveBench scores
            if model.livebench_global_avg:
                score += model.livebench_global_avg / 10  # Add up to 10 points for benchmark
            
            # Weight by user's expectations if available
            if project.capability_expectations:
                exp = project.capability_expectations
                # Add bonus for models that excel in requested dimensions
                if exp.reasoning and model.livebench_reasoning:
                    score += (model.livebench_reasoning - exp.reasoning) / 20
                if exp.coding and model.livebench_coding:
                    score += (model.livebench_coding - exp.coding) / 20
                if exp.mathematics and model.livebench_mathematics:
                    score += (model.livebench_mathematics - exp.mathematics) / 20
            
            if is_cost_sensitive:
                if model.tier == "budget":
                    score += 5
                elif model.tier == "standard":
                    score += 3
            else:
                if model.tier == "premium":
                    score += 3
                elif model.tier == "standard":
                    score += 2
            
            scored_models.append((model, score))
        
        # Sort by score and select top candidates
        scored_models.sort(key=lambda x: x[1], reverse=True)
        selected = scored_models[:max_candidates]
        
        candidates = []
        for i, (model, score) in enumerate(selected):
            # Build reasoning including benchmark info
            reasoning_parts = [f"Selected based on {model.tier} tier"]
            if model.livebench_global_avg:
                reasoning_parts.append(f"LiveBench avg={model.livebench_global_avg:.0f}")
            reasoning_parts.append(f"quality={model.quality_score}, speed={model.speed_score}")
            
            candidates.append(CandidateModel(
                provider=model.provider,
                model=model.model,
                rank=i + 1,
                reasoning=", ".join(reasoning_parts),
                expected_cost_per_request=model.cost_per_1k_input * analysis.token_distribution.input_tokens.get("mean", 1) / 1000,
                expected_latency_ms=analysis.latency_metrics.mean_ms,  # Estimate
                strengths=model.use_cases or [],
                concerns=[],
                tier=model.tier,
                use_cases=model.use_cases or [],
                quality_score=model.quality_score,
                speed_score=model.speed_score,
                supports_function_calling=model.supports_function_calling,
                supports_vision=model.supports_vision,
                context_window=model.context_window,
            ))
        
        key_reqs = []
        if needs_function_calling:
            key_reqs.append("function_calling")
        if project.capability_expectations and project.capability_expectations.has_any_expectation():
            key_reqs.append("benchmark_requirements")
        if not key_reqs:
            key_reqs.append("general")
        
        return ModelSelectionResult(
            selection_id=str(uuid.uuid4()),
            project_id=str(project.id),
            selected_at=datetime.now(timezone.utc).isoformat(),
            candidates=candidates,
            selection_reasoning="Fallback selection based on model capabilities, benchmark scores, and project requirements",
            key_requirements=key_reqs,
            excluded_models=excluded_by_benchmarks,
            analysis_summary=f"Analyzed {analysis.log_count} logs",
            confidence=0.4,  # Lower confidence for fallback
        )
