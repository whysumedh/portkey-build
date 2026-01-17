"""
Base Judge Interface - Foundation for all AI judges.

Each judge:
- Has a narrow responsibility
- Uses deterministic prompts
- Outputs structured JSON
- Is versioned for reproducibility
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class JudgeResult:
    """Structured result from a judge evaluation."""
    judge_type: str
    judge_version: int
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)
    prompt_hash: str | None = None
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "judge_type": self.judge_type,
            "judge_version": self.judge_version,
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "details": self.details,
            "prompt_hash": self.prompt_hash,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class BaseJudge(ABC):
    """
    Abstract base class for AI judges.
    
    All judges must:
    - Define a specific responsibility
    - Use versioned, deterministic prompts
    - Output structured JudgeResult
    - Not access filesystem or network directly
    """

    # Judge metadata
    JUDGE_TYPE: str = "base"
    JUDGE_VERSION: int = 1
    
    def __init__(self, portkey_client: Any = None):
        """
        Initialize the judge.
        
        Args:
            portkey_client: Portkey client for LLM calls
        """
        self.portkey = portkey_client
        self._prompt_hash: str | None = None

    @property
    def judge_type(self) -> str:
        return self.JUDGE_TYPE

    @property
    def judge_version(self) -> int:
        return self.JUDGE_VERSION

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this judge."""
        pass

    @abstractmethod
    def get_evaluation_prompt(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the evaluation prompt.
        
        Args:
            prompt: The original user prompt
            completion: The model's completion
            context: Additional context (system prompt, tool calls, etc.)
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> tuple[float, float, str, dict]:
        """
        Parse the LLM response into score, confidence, rationale, details.
        
        Returns:
            Tuple of (score, confidence, rationale, details)
        """
        pass

    def compute_prompt_hash(self) -> str:
        """Compute hash of the judge's prompts for versioning."""
        content = f"{self.get_system_prompt()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def evaluate(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> JudgeResult:
        """
        Evaluate a completion using this judge.
        
        Args:
            prompt: The original user prompt
            completion: The model's completion to evaluate
            context: Additional context
            model: Model to use for judging
            provider: Provider for the judge model
            
        Returns:
            JudgeResult with score, confidence, and rationale
        """
        system_prompt = self.get_system_prompt()
        evaluation_prompt = self.get_evaluation_prompt(prompt, completion, context)
        prompt_hash = self.compute_prompt_hash()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": evaluation_prompt},
        ]

        try:
            response = await self.portkey.chat_completion(
                messages=messages,
                model=model,
                provider=provider,
                temperature=0.0,  # Deterministic
                max_tokens=1000,
            )

            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            score, confidence, rationale, details = self.parse_response(response_text)

            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=score,
                confidence=confidence,
                rationale=rationale,
                details=details,
                prompt_hash=prompt_hash,
            )

        except Exception as e:
            # Return low-confidence result on error
            return JudgeResult(
                judge_type=self.judge_type,
                judge_version=self.judge_version,
                score=0.5,  # Neutral
                confidence=0.0,  # No confidence
                rationale=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
                prompt_hash=prompt_hash,
            )

    def evaluate_heuristic(
        self,
        prompt: str,
        completion: str,
        context: dict[str, Any] | None = None,
    ) -> JudgeResult | None:
        """
        Optional heuristic evaluation (no LLM call).
        
        Override this for fast, rule-based checks that can
        short-circuit the full LLM evaluation.
        
        Returns:
            JudgeResult if heuristic applies, None otherwise
        """
        return None
