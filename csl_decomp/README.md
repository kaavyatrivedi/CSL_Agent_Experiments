# csl_decomp — multi-source task decomposition toolkit for CSL-Agent

A self-contained package (extracted from Karan's DataCollector research
platform and adapted for this repo) that adds the pieces between your
contracts and your orchestration experiments:

1. **Automated decomposition** — `pipeline_tasks.py` currently hand-writes
   subtasks. The decomposers here generate them with an LLM, including
   dependency edges, and can do it *contract-aware* (contracts constrain the
   decomposition itself, not just dispatch afterwards).
2. **Cached, budget-guarded, provider-agnostic LLM calls** — a drop-in
   replacement for the raw `openai` calls in `scripts/extract_contract.py`
   and `scripts/collect_traces.py`. Reruns are free (disk cache keyed by the
   full request) and switching providers is an env var.

Everything runs offline out of the box (mock provider), so you can verify the
plumbing before spending a single token. Dependencies (`pyyaml`, `openai`) are
already in the repo's `requirements.txt` — nothing new to install.

## How to run

From the repo root:

```bash
# Offline smoke test — mock LLM, no API key, uses ground_truth/ contracts:
python examples/demo.py

# Real model:
LLM_PROVIDER=openai LLM_MODEL=gpt-4o OPENAI_API_KEY=sk-... python examples/demo.py

# Multi-source extraction sweep (S1 / S1+S2 / S1+S2+traces) with caching,
# writing to extracted/<model>/ so per-model runs never overwrite each other:
LLM_PROVIDER=openai LLM_MODEL=gpt-4o OPENAI_API_KEY=sk-... \
  python examples/extract_contracts_cached.py

# Same sweep on a different model family = a free extraction ablation:
LLM_PROVIDER=gemini LLM_MODEL=gemini-2.5-flash GEMINI_API_KEY=... \
  python examples/extract_contracts_cached.py --agents A01 A02 --configs S1 S1S2
```

`demo.py` loads every contract in `ground_truth/`, decomposes a sample bug-fix
task with both decomposers, dispatches every subtask to an agent, and prints
the resulting pipeline in dependency order.

## What's inside

| Module | What it gives you |
|---|---|
| `csl_decomp/llm.py` | One `llm.call(prompt, ...)` for 11 providers (openai, anthropic, gemini, groq, openrouter, cerebras, sambanova, nvidia-nim, azure, ollama, mock). SHA-256 disk caching (default `.llm_cache/`, override with `LLM_CACHE_DIR`). Global call/token/dollar budgets via `llm.set_config(...)`. Per-caller usage accounting via `llm.get_usage()`. Retry/backoff on 429/5xx. Strips `<think>` blocks from reasoning models. |
| `csl_decomp/interfaces.py` | `DecompositionContext` / `DecompositionPlan` dataclasses plus `Subtask`: id, description, `depends_on` edges, `assigned_agent` slot, and `plan.ordered_subtasks()` (topological order with cycle detection). |
| `csl_decomp/contracts.py` | Loads `ground_truth/*.yaml` (or `extracted/*.yaml`) into a typed `AgentContract` with a `capability_summary()` card used in prompts. Works with any extraction config (S1 / S1+S2 / S1+S2+T / human). |
| `csl_decomp/decomposers.py` | The two decomposers, below. |
| `csl_decomp/dispatch.py` | Three dispatchers with one shared signature, plus `assign_plan()`. |
| `csl_decomp/budget.py` | `BudgetTracker` — budget exhaustion degrades to a fallback plan instead of crashing mid-experiment. |
| `examples/demo.py` | End-to-end: contracts → decompose → dispatch → ordered pipeline. |
| `examples/extract_contracts_cached.py` | Your RQ1 multi-source sweep on the cached LLM layer. |

## The two decomposers

**`RoleDecomposer`** — an Architect LLM call proposes 3–6 subtasks as JSON with
dependency edges; a Critic call fixes up to two weaknesses (missing
verification step, over-broad subtask, wrong ordering). Agent-agnostic:
dispatch happens afterwards. Useful for drafting the remaining pipeline tasks
(P03–P10) that `pipeline_tasks.py` doesn't define yet.

**`ContractAwareDecomposer`** — embeds every agent's capability card (id,
description, accepted inputs, pre-condition predicates, measured
`satisfaction_rate`) into the decomposition prompt, and requires each subtask
to (a) satisfy some listed agent's pre-conditions and (b) consume inputs
producible from earlier subtasks. The LLM returns
`{id, description, agent, depends_on, rationale}` per subtask; hallucinated
agent ids are stripped and re-dispatched. This turns contracts from a
*post-hoc routing filter* into a *constraint on decomposition itself* — a
natural third condition for the orchestration experiment:

- **B1**: random dispatch (no contracts)
- **B2**: typed/keyword dispatch (contracts used *after* decomposition)
- **B3**: contract-aware decomposition (contracts used *during* decomposition)

In testing on the `ground_truth/` contracts with gpt-4.1-mini, the blind
decomposer produced six subtasks with some odd routing (CodeSummarizer
assigned to analyze a bug), while the contract-aware one produced exactly
localize → patch → verify, routed A01 → A02 → A03 with correct dependencies.

## The three dispatchers

All share `(subtask, contracts) -> agent_id`, so they slot into one results
table:

- `dispatch_random` — the B1 baseline.
- `make_keyword_dispatcher(predicate_keywords)` — `dispatch_CSL` from
  `scripts/orchestrators.py`, but the `@predicate -> keywords` table is
  injected instead of hardcoded, so it can be ablated or generated from
  contracts.
- `dispatch_llm` — the "Tier 2 classifier" the `orchestrators.py` comment
  anticipates: one small LLM call chooses among qualifying agents by
  pre-conditions + reliability. Cached, so a full benchmark rerun costs
  nothing after the first pass.

`assign_plan(plan, contracts, dispatcher)` fills `assigned_agent` for every
subtask, respecting assignments a contract-aware decomposer already made.

## Minimal usage

```python
from csl_decomp import (
    llm, load_contracts, ContractAwareDecomposer,
    DecompositionContext, assign_plan, dispatch_llm,
)

llm.set_config(provider="openai", model="gpt-4o", budget_usd=5.0)

contracts = load_contracts("ground_truth")
decomposer = ContractAwareDecomposer(contracts)

ctx = DecompositionContext(task_id="P01", problem_statement="Fix bug: cache module has swapped TTL parameters")
plan = decomposer.decompose(ctx)
assign_plan(plan, contracts, dispatch_llm)

for st in plan.ordered_subtasks():
    print(st.id, "->", st.assigned_agent, ":", st.description)
```

Swapping the raw OpenAI call in `scripts/extract_contract.py` is three lines:

```python
from csl_decomp import llm
llm.set_config(provider="openai", model="gpt-4o")
yaml_text = llm.call(full_prompt, max_tokens=800, caller=f"extract:{cfg_name}").content
```

(and the `time.sleep(1)` can go — retry/backoff is built in). Or just use
`examples/extract_contracts_cached.py`, which is that swap already done.

## Compatibility

Python 3.10+. `pyyaml` required; `openai` / `requests` only for the providers
you use. The mock provider needs neither and is the default. The cache
directory `.llm_cache/` and `extracted/<model>/` outputs may be worth adding
to `.gitignore` depending on whether you want cached responses versioned.
