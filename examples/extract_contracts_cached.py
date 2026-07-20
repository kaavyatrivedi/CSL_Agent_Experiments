"""Multi-source contract extraction on the cached, provider-agnostic LLM layer.

Same experiment as scripts/extract_contract.py (S1 system prompt only ->
S1+S2 docs -> S1+S2+traces), with three differences:

1. Every call goes through csl_decomp.llm: responses are cached on disk keyed
   by the full request, so re-running the sweep after a crash or for RQ1
   re-measurement costs zero API credits and is deterministic.
2. Provider/model come from env vars, so a cross-model extraction ablation
   is a one-line change:
       LLM_PROVIDER=openai    LLM_MODEL=gpt-4o           python examples/extract_contracts_cached.py
       LLM_PROVIDER=anthropic LLM_MODEL=claude-sonnet-5  python examples/extract_contracts_cached.py
       LLM_PROVIDER=gemini    LLM_MODEL=gemini-2.5-flash python examples/extract_contracts_cached.py
   Outputs are written per-model (extracted/<model>/A01_S1.yaml, ...), so runs
   never overwrite each other.
3. The satisfaction rate is computed inline from corpus/<aid>_traces.json
   (fraction of post_satisfied), with 0.0 when traces are missing.

Run from the repo root. No sleep(1) needed — retry/backoff on rate limits is
built into the provider layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from csl_decomp import llm

EXTRACTION_PROMPT = '''
You are a contract extraction expert. Given documentation about an LLM agent,
produce a CSL-Agent contract in YAML format with these exact fields:

contract_id: (agent name + version)
tin:
  type: object
  properties: (list all input fields with types)
  required: (list required fields)
pre:
  structural: (boolean expression over input fields)
  semantic: (list of @predicate_names -- use descriptive names)
post:
  structural: (boolean expression over output fields)
  semantic: (list of @predicate_names)
tout:
  type: object
  properties: (list all output fields)
prob:
  satisfaction_rate: {sat_rate}
  confidence_delta: 0.05
  min_sample_k: 30
  latency_lognormal: {{mu: 0.0, sigma: 0.0}}
comp:
  type: (one of: SELECT, CONCAT, MERGE, CONSENSUS, RESOLVE)

Return ONLY the YAML. No explanation.
'''

CONFIGS = [
    ("S1", False, False),
    ("S1S2", True, False),
    ("S1S2T", True, True),
]


def satisfaction_rate(agent_id: str) -> float:
    trace_file = REPO_ROOT / "corpus" / f"{agent_id}_traces.json"
    if not trace_file.exists():
        return 0.0
    try:
        traces = json.loads(trace_file.read_text(encoding="utf-8"))
        return round(sum(t.get("post_satisfied", False) for t in traces) / max(1, len(traces)), 2)
    except Exception:
        return 0.0


def build_context(agent_id: str, use_docs: bool, use_traces: bool) -> str:
    context = ""
    prompt_file = REPO_ROOT / "corpus" / f"{agent_id}_system_prompt.txt"
    context += f"SYSTEM PROMPT:\n{prompt_file.read_text(encoding='utf-8')}\n\n"
    if use_docs:
        doc_file = REPO_ROOT / "corpus" / f"{agent_id}_docs.txt"
        if doc_file.exists():
            context += f"DOCUMENTATION:\n{doc_file.read_text(encoding='utf-8')}\n\n"
    if use_traces:
        trace_file = REPO_ROOT / "corpus" / f"{agent_id}_traces.json"
        if trace_file.exists():
            traces = json.loads(trace_file.read_text(encoding="utf-8"))[:5]  # first 5 examples only
            context += "EXAMPLE INPUTS/OUTPUTS:\n"
            for t in traces:
                context += f"Input: {json.dumps(t['input'])}\nOutput: {str(t['output'])[:200]}\n---\n"
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="*", help="Agent ids to extract (default: A01..A20)")
    parser.add_argument("--configs", nargs="*", default=[c[0] for c in CONFIGS],
                        help="Which source configs to run (S1 S1S2 S1S2T)")
    parser.add_argument("--out", default="extracted", help="Output root directory")
    args = parser.parse_args()

    agents = args.agents or [f"A{i:02d}" for i in range(1, 21)]
    out_dir = REPO_ROOT / args.out / llm.CONFIG.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"provider={llm.CONFIG.provider} model={llm.CONFIG.model} -> {out_dir}")

    for agent_id in agents:
        if not (REPO_ROOT / "corpus" / f"{agent_id}_system_prompt.txt").exists():
            print(f"{agent_id}: no system prompt in corpus, skipping")
            continue
        sat = satisfaction_rate(agent_id)
        for cfg_name, use_docs, use_traces in CONFIGS:
            if cfg_name not in args.configs:
                continue
            context = build_context(agent_id, use_docs, use_traces)
            response = llm.call(
                f"{EXTRACTION_PROMPT.format(sat_rate=sat)}\n\nAgent documentation:\n\n{context}",
                max_tokens=800,
                temperature=0.0,
                caller=f"extract:{cfg_name}",
            )
            yaml_text = response.content.replace("```yaml", "").replace("```", "").strip()
            target = out_dir / f"{agent_id}_{cfg_name}.yaml"
            target.write_text(yaml_text, encoding="utf-8")
            hit = " (cache hit)" if response.meta.get("cache_hit") else ""
            print(f"{agent_id} {cfg_name}: {len(yaml_text)} chars, {response.tokens} tokens{hit}")

    usage = llm.get_usage()
    print(f"\nDone. LLM usage this run: {usage['totals']}")


if __name__ == "__main__":
    main()
