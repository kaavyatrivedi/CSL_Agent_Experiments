"""csl_decomp — multi-source task decomposition toolkit for CSL-Agent experiments.

Extracted and adapted from the DataCollector agentic-AI research platform
(task decomposition strategies, unified LLM provider, budget guardrails),
plus new contract-aware decomposition and dispatch built around the
CSL-Agent contract YAML format.
"""
from . import llm
from .budget import BudgetTracker
from .contracts import AgentContract, load_contracts
from .decomposers import ContractAwareDecomposer, RoleDecomposer
from .dispatch import assign_plan, dispatch_llm, dispatch_random, make_keyword_dispatcher
from .interfaces import (
    DecompositionContext,
    DecompositionPlan,
    StrategyResult,
    Subtask,
    TaskDecompositionStrategy,
)

__all__ = [
    "llm",
    "BudgetTracker",
    "AgentContract",
    "load_contracts",
    "ContractAwareDecomposer",
    "RoleDecomposer",
    "assign_plan",
    "dispatch_llm",
    "dispatch_random",
    "make_keyword_dispatcher",
    "DecompositionContext",
    "DecompositionPlan",
    "StrategyResult",
    "Subtask",
    "TaskDecompositionStrategy",
]
