"""Budget tracking for LLM planning calls (from DataCollector strategies/_utils.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .llm import LLMResponse


@dataclass
class BudgetTracker:
    """Track tokens/time consumed when calling the LLM provider.

    Wrap every llm.call() in tracker.consume(response, fallback): if the global
    budget is exhausted you get the fallback string instead of a crash, and the
    plan degrades gracefully.
    """

    strategy: str
    tokens: int = 0
    time_spent: float = 0.0
    responses: List[LLMResponse] = field(default_factory=list)

    def consume(self, response: LLMResponse, fallback: str) -> str:
        """Record a response and return usable content or the fallback if budgeted out."""

        self.responses.append(response)
        if response.budget_exceeded:
            return fallback
        self.tokens += response.tokens
        self.time_spent += response.elapsed
        return response.content

    def checkpoint(self) -> int:
        """Return an index representing the current response boundary."""

        return len(self.responses)

    def responses_since(self, mark: int) -> List[LLMResponse]:
        if mark < 0:
            mark = 0
        return self.responses[mark:]
