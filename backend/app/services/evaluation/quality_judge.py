"""
Quality and Helpfulness Judges - Evaluate response quality.

Responsibilities:
- QualityJudge: Overall quality, coherence, and completeness
- HelpfulnessJudge: How well it helps the user
"""

import json
import re
from typing import Any

from app.services.evaluation.judge_base import BaseJudge, JudgeResult


class QualityJudge(BaseJudge):
    """
    Judge for evaluating overall response quality.
    
    Evaluates:
    - Is the response well-written and coherent?
    - Is it complete and thorough?
    - Is it well-structured and easy to understand?
    """

    JUDGE_TYPE = "quality"
    JUDGE_VERSION = 1

    def get_system_prompt(self) -> str:
        return """You are an expert evaluator assessing the QUALITY of AI responses.

Your task is to evaluate the overall quality of the response:
1. Coherence: Is it logical and well-organized?
2. Completeness: Does it thoroughly address the prompt?
3. Clarity: Is it easy to understand?
4. Professionalism: Is it well-written?

You must respond with a JSON object containing:
- score: A number from 0.0 to 1.0 (0 = poor quality, 1 = excellent quality)
- confidence: A number from 0.0 to 1.0
- rationale: A brief explanation
- strengths: List of quality strengths
- weaknesses: List of quality weaknesses

Example response:
{
  "score": 0.9,
  "confidence": 0.85,
  "rationale": "Well-structured response with clear explanations.",
  "strengths": ["Clear structure", "Good examples"],
  "weaknesses": ["Could be more concise"]
}"""

    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        return f"""Evaluate the quality of this AI response.

USER PROMPT:
{prompt}

AI RESPONSE:
{completion}

Provide your quality evaluation as a JSON object with score, confidence, rationale, strengths, and weaknesses."""

    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """Parse the judge's JSON response."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            rationale = str(data.get("rationale", "No rationale provided"))
            details = {
                "strengths": data.get("strengths", []),
                "weaknesses": data.get("weaknesses", []),
            }
            
            score = max(0.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))
            
            return score, confidence, rationale, details

        except (json.JSONDecodeError, ValueError) as e:
            return 0.5, 0.3, response[:500], {"parse_error": str(e)}

    def evaluate_heuristic(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> JudgeResult | None:
        """Quick quality heuristics."""
        
        # Very short responses are low quality
        if len(completion.strip()) < 20:
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.2,
                confidence=0.8,
                rationale="Response is too short to be useful",
                details={"heuristic": "too_short", "length": len(completion)},
            )
        
        # Repetitive responses
        words = completion.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                return JudgeResult(
                    judge_type=self.judge_type,
                    judge_version=self.judge_version,
                    score=0.3,
                    confidence=0.7,
                    rationale="Response appears to be highly repetitive",
                    details={"heuristic": "repetitive", "unique_ratio": unique_ratio},
                )
        
        return None


class HelpfulnessJudge(BaseJudge):
    """
    Judge for evaluating how helpful the response is to the user.
    
    Evaluates:
    - Does it actually help the user?
    - Is it actionable and practical?
    - Does it anticipate follow-up needs?
    """

    JUDGE_TYPE = "helpfulness"
    JUDGE_VERSION = 1

    def get_system_prompt(self) -> str:
        return """You are an expert evaluator assessing how HELPFUL an AI response is.

Your task is to evaluate from the USER's perspective:
1. Does this response actually help solve the user's problem?
2. Is it actionable - can the user take concrete steps?
3. Does it provide value beyond just answering the literal question?
4. Would you, as the user, be satisfied with this response?

You must respond with a JSON object containing:
- score: A number from 0.0 to 1.0 (0 = not helpful, 1 = extremely helpful)
- confidence: A number from 0.0 to 1.0
- rationale: A brief explanation from the user's perspective
- actionability: "none", "low", "medium", or "high"

Example response:
{
  "score": 0.85,
  "confidence": 0.9,
  "rationale": "Provides clear step-by-step instructions the user can follow immediately.",
  "actionability": "high"
}"""

    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        return f"""Evaluate how helpful this AI response is to the user.

USER PROMPT:
{prompt}

AI RESPONSE:
{completion}

Put yourself in the user's shoes. Provide your helpfulness evaluation as a JSON object with score, confidence, rationale, and actionability."""

    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """Parse the judge's JSON response."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            rationale = str(data.get("rationale", "No rationale provided"))
            details = {
                "actionability": data.get("actionability", "unknown"),
            }
            
            score = max(0.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))
            
            return score, confidence, rationale, details

        except (json.JSONDecodeError, ValueError) as e:
            return 0.5, 0.3, response[:500], {"parse_error": str(e)}

    def evaluate_heuristic(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> JudgeResult | None:
        """Quick helpfulness heuristics."""
        
        completion_lower = completion.lower()
        
        # Responses that just ask for more info without trying to help
        unhelpful_patterns = [
            "could you provide more",
            "i need more information",
            "can you clarify",
            "it depends on",
        ]
        
        unhelpful_count = sum(1 for p in unhelpful_patterns if p in completion_lower)
        
        # If the response is just asking for clarification and nothing else
        if unhelpful_count >= 2 and len(completion) < 200:
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.4,
                confidence=0.7,
                rationale="Response mostly asks for clarification without providing help",
                details={"heuristic": "clarification_only", "actionability": "none"},
            )
        
        return None
