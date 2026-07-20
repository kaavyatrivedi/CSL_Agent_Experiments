"""Shared interfaces and dataclasses for task decomposition.

Extracted from the DataCollector research platform (src/decomposition/interfaces.py),
with one extension for CSL-Agent: a `Subtask` dataclass carrying an id, dependency
edges, and a dispatch slot (`assigned_agent`), so decomposition output plugs directly
into contract-based orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class DecompositionContext:
    """Problem context passed into every decomposer."""

    task_id: str
    problem_statement: str
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    constraints: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    nearest_neighbors: List[Dict[str, str]] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    historical_stats: Optional[Dict[str, float]] = None


@dataclass
class Subtask:
    """One node of a decomposition, ready for contract-based dispatch.

    `depends_on` lists ids of subtasks whose output this one consumes —
    a sequential pipeline is simply each subtask depending on the previous one.
    `assigned_agent` is filled by a dispatcher (e.g. CSL pre-condition matching),
    not by the decomposer, unless the decomposer is contract-aware.
    """

    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": list(self.depends_on),
            "assigned_agent": self.assigned_agent,
            "rationale": self.rationale,
        }


@dataclass
class DecompositionPlan:
    """Structured plan produced by every decomposer.

    `subtasks` (flat strings) is kept for compatibility with simple consumers;
    `subtask_graph` is the richer form with ids, dependencies, and dispatch slots.
    When both are set they describe the same plan.
    """

    strategy_name: str
    contract_items: List[Dict[str, object]] = field(default_factory=list)
    contract: Dict[str, str] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    subtask_graph: List[Subtask] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    simulation_traces: List[str] = field(default_factory=list)
    role_messages: List[str] = field(default_factory=list)
    diagnostics: Dict[str, str] = field(default_factory=dict)

    def ordered_subtasks(self) -> List[Subtask]:
        """Return subtask_graph in a valid topological order (raises on cycles)."""

        remaining = {st.id: st for st in self.subtask_graph}
        resolved: List[Subtask] = []
        resolved_ids: set = set()
        while remaining:
            progressed = False
            for sid in list(remaining):
                st = remaining[sid]
                if all(dep in resolved_ids or dep not in remaining for dep in st.depends_on):
                    resolved.append(st)
                    resolved_ids.add(sid)
                    del remaining[sid]
                    progressed = True
            if not progressed:
                raise ValueError(f"Dependency cycle among subtasks: {sorted(remaining)}")
        return resolved


@dataclass
class StrategyResult:
    """Full result bundle from executing a plan."""

    plan: DecompositionPlan
    solution_code: str = ""
    tests_run: List[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)
    round_traces: List[Dict[str, object]] = field(default_factory=list)


class TaskDecompositionStrategy(Protocol):
    """Protocol that all decomposition strategies implement."""

    name: str

    def decompose(self, ctx: DecompositionContext) -> DecompositionPlan:
        ...
