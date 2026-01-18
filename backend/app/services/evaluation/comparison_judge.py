"""
Comparison Judge - Compares replay response quality to original response.

This judge is specifically designed for replay evaluation:
- Compares the replay model's response to the original model's response
- Evaluates semantic similarity, quality parity, and correctness
- Used to determine if a candidate model can replace the current model
"""

import json
from typing import Any

from app.services.evaluation.judge_base import BaseJudge, JudgeResult


class ComparisonJudge(BaseJudge):
    """
    Judge that compares replay response to original response.
    
    Used in replay evaluation to assess whether a candidate model
    produces equivalent or better responses than the original.
    """

    JUDGE_TYPE = "comparison"
    JUDGE_VERSION = 1

    def get_system_prompt(self) -> str:
        return """You are an expert AI response quality evaluator. Your task is to compare two AI responses to the same prompt and assess their relative quality.

You will receive:
1. The original user prompt
2. Response A: The original model's response
3. Response B: A candidate model's response (replay)

Evaluate Response B against Response A on these criteria:
- **Semantic Equivalence**: Does B convey the same meaning/information as A?
- **Correctness**: Is B factually correct? Does it contain errors A doesn't have?
- **Completeness**: Does B cover all the important points A does?
- **Quality**: Is B's quality (clarity, structure, helpfulness) comparable to A?
- **Improvements**: Does B improve on A in any way?

Output your evaluation as a JSON object with this exact structure:
{
    "semantic_equivalence_score": 0.0-1.0,
    "correctness_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "comparison_verdict": "better" | "equivalent" | "worse",
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation of the comparison",
    "key_differences": ["difference1", "difference2"],
    "improvements": ["improvement1"],
    "regressions": ["regression1"]
}

IMPORTANT:
- Score 1.0 means Response B is equivalent or better than Response A
- Score below 0.7 indicates significant quality regression
- Be objective and focus on measurable differences
- If responses are semantically equivalent, score should be high even if wording differs"""

    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,  # This is Response B (replay)
        context: dict[str, Any] | None = None,
    ) -> str:
        # Extract original completion from context
        original_completion = ""
        if context:
            original_completion = context.get("original_completion", "")
            if not original_completion:
                original_completion = context.get("original_response", "")

        return f"""## Original Prompt
{prompt}

## Response A (Original Model)
{original_completion if original_completion else "[Original response not available]"}

## Response B (Candidate Model - Replay)
{completion}

Please compare Response B to Response A and provide your evaluation as JSON."""

    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """Parse the JSON evaluation response."""
        try:
            # Try to extract JSON from the response
            response_clean = response.strip()
            
            # Handle markdown code blocks
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            
            # Find JSON object
            if "{" in response_clean:
                start = response_clean.find("{")
                end = response_clean.rfind("}") + 1
                response_clean = response_clean[start:end]
            
            data = json.loads(response_clean)
            
            # Extract scores
            overall_score = data.get("overall_score", 0.5)
            confidence = data.get("confidence", 0.7)
            rationale = data.get("rationale", "No rationale provided")
            
            details = {
                "semantic_equivalence_score": data.get("semantic_equivalence_score", 0),
                "correctness_score": data.get("correctness_score", 0),
                "completeness_score": data.get("completeness_score", 0),
                "quality_score": data.get("quality_score", 0),
                "comparison_verdict": data.get("comparison_verdict", "unknown"),
                "key_differences": data.get("key_differences", []),
                "improvements": data.get("improvements", []),
                "regressions": data.get("regressions", []),
            }
            
            return overall_score, confidence, rationale, details
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return neutral scores on parse error
            return 0.5, 0.3, f"Failed to parse response: {e}", {"parse_error": str(e)}

    def evaluate_heuristic(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> JudgeResult | None:
        """Quick heuristic checks before full evaluation."""
        # No original to compare - can't evaluate
        if not context or not context.get("original_completion"):
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.5,
                confidence=0.2,
                rationale="No original completion available for comparison",
                details={"skipped": "missing_original"},
            )
        
        original = context["original_completion"]
        
        # Empty completion is clearly worse
        if not completion or len(completion.strip()) == 0:
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.0,
                confidence=1.0,
                rationale="Replay produced empty response",
                details={"comparison_verdict": "worse", "regressions": ["empty_response"]},
            )
        
        # If completions are identical, perfect score
        if completion.strip() == original.strip():
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=1.0,
                confidence=1.0,
                rationale="Replay produced identical response to original",
                details={"comparison_verdict": "equivalent", "key_differences": []},
            )
        
        # If very similar length and content, likely equivalent
        len_ratio = len(completion) / len(original) if original else 1
        if 0.8 <= len_ratio <= 1.2:
            # Similar length - needs full evaluation
            return None
        
        # Significant length difference - needs evaluation but flag it
        return None


class QualityComparisonJudge(BaseJudge):
    """
    Standalone quality judge for replay responses (without comparison).
    
    Used when original response is not available or as supplementary evaluation.
    """

    JUDGE_TYPE = "quality_standalone"
    JUDGE_VERSION = 1

    def get_system_prompt(self) -> str:
        return """You are an expert AI response quality evaluator. Evaluate the quality of an AI assistant's response to a user prompt.

Assess the response on these criteria:
- **Relevance**: Does the response address the user's request?
- **Accuracy**: Is the information correct and factual?
- **Helpfulness**: Does the response actually help the user?
- **Clarity**: Is the response clear and well-structured?
- **Completeness**: Does it cover the necessary points?

Output your evaluation as a JSON object:
{
    "relevance_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "helpfulness_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation",
    "strengths": ["strength1"],
    "weaknesses": ["weakness1"]
}"""

    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        system_context = ""
        if context and context.get("system_prompt"):
            system_context = f"\n\n## System Context\n{context['system_prompt']}"
        
        return f"""## User Prompt
{prompt}{system_context}

## AI Response
{completion}

Please evaluate the quality of this response and provide your assessment as JSON."""

    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """Parse the JSON evaluation response."""
        try:
            response_clean = response.strip()
            
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            
            if "{" in response_clean:
                start = response_clean.find("{")
                end = response_clean.rfind("}") + 1
                response_clean = response_clean[start:end]
            
            data = json.loads(response_clean)
            
            overall_score = data.get("overall_score", 0.5)
            confidence = data.get("confidence", 0.7)
            rationale = data.get("rationale", "No rationale provided")
            
            details = {
                "relevance_score": data.get("relevance_score", 0),
                "accuracy_score": data.get("accuracy_score", 0),
                "helpfulness_score": data.get("helpfulness_score", 0),
                "clarity_score": data.get("clarity_score", 0),
                "completeness_score": data.get("completeness_score", 0),
                "strengths": data.get("strengths", []),
                "weaknesses": data.get("weaknesses", []),
            }
            
            return overall_score, confidence, rationale, details
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return 0.5, 0.3, f"Failed to parse response: {e}", {"parse_error": str(e)}
