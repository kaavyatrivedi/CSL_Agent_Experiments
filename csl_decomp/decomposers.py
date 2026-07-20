"""LLM-driven task decomposers.

Two decomposers, both implementing the TaskDecompositionStrategy protocol:

- RoleDecomposer: adapted from DataCollector's role_decomposed strategy —
  an Architect LLM call proposes a phased plan, a Critic call refines it.
  Agent-agnostic: produces subtasks for a downstream dispatcher to assign.

- ContractAwareDecomposer: the new piece for CSL-Agent. It puts the available
  agents' contract capability cards INTO the decomposition prompt and asks the
  LLM to split the task into subtasks that each satisfy some agent's
  pre-conditions, returning JSON with dependency edges and a suggested agent
  per subtask. Decomposition and dispatch become one contract-constrained step
  instead of decompose-then-hope.

Both degrade gracefully: on LLM budget exhaustion or unparseable output they
fall back to a canned localize->patch->review->verify pipeline and record why
in plan.diagnostics.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from . import llm
from .budget import BudgetTracker
from .contracts import AgentContract
from .interfaces import DecompositionContext, DecompositionPlan, Subtask

_FALLBACK_PIPELINE = [
    ("T1", "Identify the relevant file(s) and location for the task", []),
    ("T2", "Produce the code change (patch or new code) for the task", ["T1"]),
    ("T3", "Review the produced change for correctness and style", ["T2"]),
    ("T4", "Verify the change passes the relevant tests", ["T3"]),
]


def _fallback_subtasks(reason: str) -> List[Subtask]:
    return [
        Subtask(id=sid, description=desc, depends_on=list(deps), rationale=f"fallback: {reason}")
        for sid, desc, deps in _FALLBACK_PIPELINE
    ]


def _extract_json_array(text: str) -> Optional[List[Dict[str, object]]]:
    """Pull the first JSON array out of an LLM response, tolerating code fences."""

    cleaned = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict)]


def _parse_subtask_items(items: List[Dict[str, object]], task_id: str) -> List[Subtask]:
    subtasks: List[Subtask] = []
    for idx, item in enumerate(items, start=1):
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        sid = str(item.get("id") or f"T{idx}")
        deps = item.get("depends_on") or []
        if not isinstance(deps, list):
            deps = [deps]
        agent = item.get("agent") or item.get("assigned_agent")
        subtasks.append(
            Subtask(
                id=sid,
                description=description,
                depends_on=[str(d) for d in deps],
                assigned_agent=str(agent) if agent else None,
                rationale=str(item.get("rationale", "")),
            )
        )
    return subtasks


class RoleDecomposer:
    """Architect proposes a phased plan; Critic refines it. Agent-agnostic."""

    name = "role_decomposed"

    ARCHITECT_PROMPT = (
        "You are the Architect in a software team. Decompose the following task into "
        "3-6 concrete, sequential subtasks that different specialist agents could each "
        "perform (e.g. localization, patching, review, testing).\n"
        "Task: {statement}\n\n"
        "Return ONLY a JSON array, each element: "
        '{{"id": "T1", "description": "...", "depends_on": ["..."]}}'
    )

    CRITIC_PROMPT = (
        "You are the Critic. Here is a proposed decomposition of the task "
        '"{statement}":\n{plan_json}\n\n'
        "Fix at most two weaknesses (missing verification step, subtask too broad for a "
        "single agent, wrong dependency order). Return the corrected FULL decomposition "
        "as ONLY a JSON array with the same schema."
    )

    def __init__(self, use_critic: bool = True, max_tokens: int = 600) -> None:
        self.use_critic = use_critic
        self.max_tokens = max_tokens

    def decompose(self, ctx: DecompositionContext) -> DecompositionPlan:
        tracker = BudgetTracker(f"{self.name}:plan")
        architect_raw = tracker.consume(
            llm.call(
                self.ARCHITECT_PROMPT.format(statement=ctx.problem_statement[:1200]),
                max_tokens=self.max_tokens,
                temperature=0.2,
                caller=self.name,
            ),
            fallback="",
        )
        items = _extract_json_array(architect_raw)
        role_messages = [architect_raw]

        if items and self.use_critic:
            critic_raw = tracker.consume(
                llm.call(
                    self.CRITIC_PROMPT.format(
                        statement=ctx.problem_statement[:400],
                        plan_json=json.dumps(items, indent=1),
                    ),
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    caller=self.name,
                ),
                fallback="",
            )
            role_messages.append(critic_raw)
            revised = _extract_json_array(critic_raw)
            if revised:
                items = revised

        if items:
            subtask_graph = _parse_subtask_items(items, ctx.task_id)
            diagnostics = {"source": "llm"}
        else:
            subtask_graph = _fallback_subtasks("unparseable or budgeted-out LLM output")
            diagnostics = {"source": "fallback"}
        diagnostics.update(
            {
                "planning_tokens": str(tracker.tokens),
                "planning_time": f"{tracker.time_spent:.3f}",
            }
        )
        return DecompositionPlan(
            strategy_name=self.name,
            subtasks=[st.description for st in subtask_graph],
            subtask_graph=subtask_graph,
            role_messages=role_messages,
            diagnostics=diagnostics,
        )


class ContractAwareDecomposer:
    """Decompose WITH the agent contracts in view, so every subtask is dispatchable.

    The prompt embeds one capability card per agent (id, description, accepted
    inputs, pre-condition predicates, measured reliability) and requires each
    subtask to name the agent it targets. This inverts the usual pipeline:
    instead of decomposing blindly and then hoping a dispatcher finds a
    compatible agent, the contract set constrains the decomposition itself.
    """

    name = "contract_aware"

    PROMPT = (
        "You are a task-decomposition planner for a multi-agent system. Available "
        "agents and their contracts:\n{capability_cards}\n\n"
        "Decompose the following task into subtasks such that EVERY subtask matches "
        "one listed agent: its description must satisfy that agent's pre-conditions "
        "and its inputs must be producible from the original task or from earlier "
        "subtasks' outputs (respect each agent's accepted inputs). Prefer agents "
        "with higher reliability when several qualify.\n"
        "Task: {statement}\n\n"
        "Return ONLY a JSON array, each element:\n"
        '{{"id": "T1", "description": "...", "agent": "A03", '
        '"depends_on": ["..."], "rationale": "why this agent\'s pre-conditions hold"}}'
    )

    def __init__(self, contracts: Dict[str, AgentContract], max_tokens: int = 900) -> None:
        self.contracts = contracts
        self.max_tokens = max_tokens

    def decompose(self, ctx: DecompositionContext) -> DecompositionPlan:
        tracker = BudgetTracker(f"{self.name}:plan")
        cards = "\n".join(f"- {c.capability_summary()}" for c in self.contracts.values())
        raw = tracker.consume(
            llm.call(
                self.PROMPT.format(capability_cards=cards, statement=ctx.problem_statement[:1200]),
                max_tokens=self.max_tokens,
                temperature=0.1,
                caller=self.name,
            ),
            fallback="",
        )
        items = _extract_json_array(raw)
        if items:
            subtask_graph = _parse_subtask_items(items, ctx.task_id)
            # Reject hallucinated agent ids so dispatch never routes to a
            # non-existent agent; the dispatcher re-assigns cleared slots.
            unknown = []
            for st in subtask_graph:
                if st.assigned_agent and st.assigned_agent not in self.contracts:
                    unknown.append(st.assigned_agent)
                    st.assigned_agent = None
            diagnostics = {"source": "llm", "unknown_agents": ",".join(unknown)}
        else:
            subtask_graph = _fallback_subtasks("unparseable or budgeted-out LLM output")
            diagnostics = {"source": "fallback", "unknown_agents": ""}
        diagnostics.update(
            {
                "num_agents_in_prompt": str(len(self.contracts)),
                "planning_tokens": str(tracker.tokens),
                "planning_time": f"{tracker.time_spent:.3f}",
            }
        )
        return DecompositionPlan(
            strategy_name=self.name,
            subtasks=[st.description for st in subtask_graph],
            subtask_graph=subtask_graph,
            role_messages=[raw],
            diagnostics=diagnostics,
        )
