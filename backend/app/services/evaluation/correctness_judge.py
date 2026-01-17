"""
Correctness Judge - Evaluates factual accuracy and task completion.

Responsibility: Determine if the completion correctly addresses the prompt
and provides accurate information.
"""

import json
import re
from typing import Any

from app.services.evaluation.judge_base import BaseJudge, JudgeResult


class CorrectnessJudge(BaseJudge):
    """
    Judge for evaluating correctness and task completion.
    
    Evaluates:
    - Does the response address the prompt?
    - Is the information factually accurate?
    - Was the task completed successfully?
    """

    JUDGE_TYPE = "correctness"
    JUDGE_VERSION = 1

    def get_system_prompt(self) -> str:
        return """You are an expert evaluator assessing the CORRECTNESS of AI responses.

Your task is to evaluate whether the AI's response:
1. Directly addresses the user's prompt
2. Provides factually accurate information
3. Completes the requested task correctly

You must respond with a JSON object containing:
- score: A number from 0.0 to 1.0 (0 = completely incorrect, 1 = perfectly correct)
- confidence: A number from 0.0 to 1.0 indicating your confidence in the score
- rationale: A brief explanation of your scoring
- issues: A list of any correctness issues found (empty if none)

Be objective and fair. Consider partial correctness.

Example response:
{
  "score": 0.85,
  "confidence": 0.9,
  "rationale": "The response correctly explains the concept but has a minor inaccuracy in the example.",
  "issues": ["Example calculation shows 15% instead of correct 12%"]
}"""

    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        context_section = ""
        if context:
            if context.get("system_prompt"):
                context_section += f"\n\nSYSTEM CONTEXT:\n{context['system_prompt']}"
            if context.get("tool_calls"):
                context_section += f"\n\nTOOLS AVAILABLE:\n{json.dumps(context['tool_calls'], indent=2)}"

        return f"""Evaluate the correctness of this AI response.

USER PROMPT:
{prompt}
{context_section}

AI RESPONSE:
{completion}

Provide your evaluation as a JSON object with score, confidence, rationale, and issues."""

    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """Parse the judge's JSON response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            rationale = str(data.get("rationale", "No rationale provided"))
            details = {"issues": data.get("issues", [])}
            
            # Clamp values
            score = max(0.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))
            
            return score, confidence, rationale, details

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing
            score = 0.5
            if "correct" in response.lower() and "incorrect" not in response.lower():
                score = 0.8
            elif "incorrect" in response.lower() or "wrong" in response.lower():
                score = 0.3
            
            return score, 0.3, response[:500], {"parse_error": str(e)}

    def evaluate_heuristic(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> JudgeResult | None:
        """Quick heuristic checks before full evaluation."""
        
        # Check for empty or very short responses
        if not completion or len(completion.strip()) < 10:
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.0,
                confidence=1.0,
                rationale="Response is empty or too short to be useful",
                details={"heuristic": "length_check", "length": len(completion.strip())},
            )
        
        # Check if response is just an error message
        error_patterns = [
            "error occurred",
            "something went wrong",
            "unable to process",
            "technical difficulties",
        ]
        completion_lower = completion.lower()
        for pattern in error_patterns:
            if pattern in completion_lower and len(completion) < 100:
                return JudgeResult(
                    judge_type=self.judge_type,
                    judge_version=self.judge_version,
                    score=0.1,
                    confidence=0.9,
                    rationale="Response appears to be an error message",
                    details={"heuristic": "error_detection", "pattern": pattern},
                )
        
        return None  # Proceed with full evaluation
