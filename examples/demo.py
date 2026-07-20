"""End-to-end demo: load contracts -> decompose a pipeline task -> dispatch subtasks.

Run from the repo root. Offline by default (mock LLM, no API key needed) so you
can verify the plumbing works:

    python examples/demo.py

It picks up ground_truth/ contracts automatically. Point it at a real provider
with env vars:

    LLM_PROVIDER=openai LLM_MODEL=gpt-4o OPENAI_API_KEY=sk-... python examples/demo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from csl_decomp import (
    AgentContract,
    ContractAwareDecomposer,
    DecompositionContext,
    RoleDecomposer,
    assign_plan,
    dispatch_llm,
    llm,
    load_contracts,
)

TOY_CONTRACTS = {
    "A01": AgentContract(
        agent_id="A01",
        agent_name="Localiser",
        description="Identifies the buggy file from a GitHub issue description",
        pre_semantic=["@input_is_github_issue"],
        satisfaction_rate=0.93,
    ),
    "A02": AgentContract(
        agent_id="A02",
        agent_name="Patcher",
        description="Generates a unified-diff patch given a file location and issue",
        pre_semantic=["@input_has_file_location"],
        satisfaction_rate=0.71,
    ),
    "A03": AgentContract(
        agent_id="A03",
        agent_name="Verifier",
        description="Verifies that a patch passes the provided tests",
        pre_semantic=["@input_is_patch_and_tests"],
        satisfaction_rate=0.88,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contracts",
        default=str(REPO_ROOT / "ground_truth"),
        help="Directory of contract YAMLs (default: ground_truth/)",
    )
    parser.add_argument(
        "--task",
        default="Fix bug: the cache module has swapped TTL parameters, causing entries to expire immediately.",
        help="Top-level task to decompose",
    )
    args = parser.parse_args()

    contracts_dir = Path(args.contracts)
    contracts = load_contracts(contracts_dir) if contracts_dir.is_dir() else {}
    if not contracts:
        print(f"(no contracts found in {contracts_dir}; using 3 bundled toy contracts)")
        contracts = TOY_CONTRACTS
    print(f"Loaded {len(contracts)} agent contracts:")
    for contract in contracts.values():
        print(f"  {contract.capability_summary()}")

    ctx = DecompositionContext(task_id="demo-01", problem_statement=args.task)

    for decomposer in (RoleDecomposer(), ContractAwareDecomposer(contracts)):
        print(f"\n=== {decomposer.name} ===")
        plan = decomposer.decompose(ctx)
        print(f"plan source: {plan.diagnostics.get('source')} "
              f"({plan.diagnostics.get('planning_tokens')} planning tokens)")
        assign_plan(plan, contracts, dispatch_llm)
        for subtask in plan.ordered_subtasks():
            deps = ",".join(subtask.depends_on) or "-"
            print(f"  [{subtask.id}] -> {subtask.assigned_agent}  (deps: {deps})  {subtask.description}")

    usage = llm.get_usage()
    print(f"\nLLM usage: {usage['totals']}")


if __name__ == "__main__":
    main()
