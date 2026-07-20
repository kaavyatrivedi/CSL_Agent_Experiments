"""Contract-based subtask dispatch.

Drop-in upgrades for CSL_Agent_Experiments/scripts/orchestrators.py, built on the
AgentContract loader. Three dispatchers with one shared signature
(subtask, contracts) -> agent_id:

- dispatch_random:      B1 baseline, no capability checking.
- dispatch_keyword:     CSL keyword pre-check + highest satisfaction rate
                        (same idea as orchestrators.dispatch_CSL, but the keyword
                        table is injectable instead of hardcoded).
- dispatch_llm:         replaces the keyword table with a single small LLM call —
                        the "in production, this calls a classifier" upgrade the
                        orchestrators.py comment anticipates. Cached by csl_decomp.llm,
                        so repeated benchmark runs are free.

assign_plan() applies a dispatcher to every unassigned subtask in a
DecompositionPlan (contract-aware decomposers may have pre-assigned some).
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable, Dict, List, Optional

from . import llm
from .contracts import AgentContract
from .interfaces import DecompositionPlan, Subtask

Dispatcher = Callable[[Subtask, Dict[str, AgentContract]], str]


def dispatch_random(subtask: Subtask, contracts: Dict[str, AgentContract]) -> str:
    """B1 baseline: random assignment, no capability checking."""

    return random.choice(list(contracts))


def make_keyword_dispatcher(predicate_keywords: Dict[str, List[str]]) -> Dispatcher:
    """CSL-style dispatch: pre-condition keyword check, then highest reliability.

    predicate_keywords maps '@predicate_name' -> task-description keywords that
    signal the predicate holds (empty list = not checkable, treated as holding).
    """

    def _pre_compatible(description: str, contract: AgentContract) -> bool:
        text = description.lower()
        for pred in contract.pre_semantic:
            keywords = predicate_keywords.get(pred, [])
            if keywords and not any(kw in text for kw in keywords):
                return False
        return True

    def _dispatch(subtask: Subtask, contracts: Dict[str, AgentContract]) -> str:
        candidates = [
            (aid, c.satisfaction_rate)
            for aid, c in contracts.items()
            if _pre_compatible(subtask.description, c)
        ]
        if not candidates:
            return random.choice(list(contracts))
        return max(candidates, key=lambda pair: pair[1])[0]

    return _dispatch


_LLM_DISPATCH_PROMPT = (
    "You are a dispatcher in a multi-agent system. Pick the single best agent for "
    "this subtask. An agent qualifies only if the subtask satisfies its "
    "pre-conditions and matches its accepted inputs; among qualifying agents "
    "prefer higher reliability.\n"
    "Agents:\n{cards}\n\n"
    "Subtask: {description}\n\n"
    'Return ONLY JSON: {{"agent": "<agent id>", "reason": "..."}}'
)


def dispatch_llm(subtask: Subtask, contracts: Dict[str, AgentContract]) -> str:
    """LLM-classifier dispatch: one small call, falls back to reliability order."""

    cards = "\n".join(f"- {c.capability_summary()}" for c in contracts.values())
    response = llm.call(
        _LLM_DISPATCH_PROMPT.format(cards=cards, description=subtask.description),
        max_tokens=120,
        temperature=0.0,
        caller="dispatch_llm",
    )
    match = re.search(r"\{.*\}", response.content, flags=re.DOTALL)
    if match:
        try:
            choice = json.loads(match.group(0))
            agent = str(choice.get("agent", ""))
            if agent in contracts:
                return agent
        except json.JSONDecodeError:
            pass
    # Fallback: most reliable agent overall.
    return max(contracts.items(), key=lambda pair: pair[1].satisfaction_rate)[0]


def assign_plan(
    plan: DecompositionPlan,
    contracts: Dict[str, AgentContract],
    dispatcher: Dispatcher,
    respect_existing: bool = True,
) -> DecompositionPlan:
    """Assign an agent to every subtask in the plan's graph.

    With respect_existing=True, subtasks already assigned by a contract-aware
    decomposer keep their assignment; only empty slots are dispatched.
    """

    for subtask in plan.subtask_graph:
        if respect_existing and subtask.assigned_agent in contracts:
            continue
        subtask.assigned_agent = dispatcher(subtask, contracts)
    return plan
