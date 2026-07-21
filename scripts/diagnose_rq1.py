"""
Diagnostic companion to measure_rq1.py.

For every (agent, config) pair, records a side-by-side comparison of
ground-truth vs extracted tout property names and pre/post predicates
-- including, for every ground-truth predicate, the single best-matching
extracted predicate and its cosine similarity, whether or not it
cleared MATCH_THRESHOLD.

This does NOT recompute or change any Table 2 score. It's for figuring
out *why* a score is low: genuine semantic mismatch vs. threshold
miscalibration vs. structural extraction differences (e.g. properties
nested inside an array vs flat).

Run from the same directory as measure_rq1.py (scripts/), after
measure_rq1.py has been run at least once -- reuses its
embedding_cache.json so this makes no new API calls unless new
predicates showed up since your last run.

Usage:
    python scripts/diagnose_rq1.py            # all agents, all configs
    python scripts/diagnose_rq1.py A01         # just A01, all configs
    python scripts/diagnose_rq1.py A01 S1S2T   # just A01, one config
"""

import sys
import json
import yaml
from pathlib import Path

from measure_rq1 import (
    AGENT_RANGE, CONFIGS, MATCH_THRESHOLD,
    get_prop_names, get_semantic, sanitize_yaml_text,
    load_embed_cache, fetch_missing_embeddings, cosine,
)


def nested_array_props(schema):
    """
    Diagnostic-only: if `properties` contains a field whose type is
    'array' with object items, also surface *those* nested property
    names. get_prop_names() in measure_rq1.py deliberately does NOT do
    this (tout scoring is top-level-only) -- this lets you see when a
    low tout score is actually 'right fields, wrong nesting level'
    rather than genuinely different fields.
    """
    if not isinstance(schema, dict):
        return {}
    props = schema.get('properties', {})
    nested = {}
    if isinstance(props, dict):
        for name, spec in props.items():
            if isinstance(spec, dict) and spec.get('type') == 'array':
                items = spec.get('items', {})
                if isinstance(items, dict) and isinstance(items.get('properties'), dict):
                    nested[name] = set(items['properties'].keys())
    return nested


def best_match(pred, other_preds, cache):
    """Best (similarity, other_pred) for `pred` against a list of candidates."""
    best_sim, best_p = -1.0, None
    for other in other_preds:
        if pred not in cache or other not in cache:
            continue
        sim = cosine(cache[pred], cache[other])
        if sim > best_sim:
            best_sim, best_p = sim, other
    return best_sim, best_p


def load_contract(path):
    if not Path(path).exists():
        return None
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return yaml.safe_load(sanitize_yaml_text(text))


def diagnose_pair(aid, cfg, cache):
    gt = load_contract(f'ground_truth/{aid}_contract_R1.yaml')
    ex = load_contract(f'extracted/{aid}_{cfg}.yaml')
    if gt is None or ex is None:
        return None

    entry = {'agent': aid, 'config': cfg}

    # --- tout diagnostics ---
    gt_tout_props = get_prop_names(gt.get('tout'))
    ex_tout_props = get_prop_names(ex.get('tout'))
    ex_tout_nested = nested_array_props(ex.get('tout'))
    entry['tout'] = {
        'gt_properties': sorted(gt_tout_props),
        'ex_properties': sorted(ex_tout_props),
        'top_level_overlap': sorted(gt_tout_props & ex_tout_props),
        'ex_nested_array_properties': {k: sorted(v) for k, v in ex_tout_nested.items()},
    }
    if not (gt_tout_props & ex_tout_props):
        nested_union = set().union(*ex_tout_nested.values()) if ex_tout_nested else set()
        if gt_tout_props & nested_union:
            entry['tout']['note'] = (
                'Zero top-level overlap, but matching field name(s) exist inside a '
                'nested array in the extracted schema (' +
                ', '.join(sorted(gt_tout_props & nested_union)) +
                ') -- looks like a structural (nesting) mismatch, not a naming mismatch.'
            )

    # --- pre/post diagnostics ---
    for field in ('pre', 'post'):
        gt_preds = get_semantic(gt, field)
        ex_preds = get_semantic(ex, field)
        matches = []
        for g in gt_preds:
            sim, best_ex = best_match(g, ex_preds, cache)
            matches.append({
                'gt_predicate': g,
                'best_extracted_match': best_ex,
                'similarity': round(sim, 3) if best_ex else None,
                'cleared_threshold': bool(best_ex and sim >= MATCH_THRESHOLD),
            })
        entry[field] = {
            'gt_predicates': gt_preds,
            'extracted_predicates': ex_preds,
            'matches': matches,
        }

    return entry


def main():
    args = sys.argv[1:]
    agents = [args[0]] if args else [f'A{i:02d}' for i in AGENT_RANGE]
    configs = [args[1]] if len(args) > 1 else CONFIGS

    cache = load_embed_cache()

    all_preds = set()
    for aid in agents:
        for cfg in configs:
            gt = load_contract(f'ground_truth/{aid}_contract_R1.yaml')
            ex = load_contract(f'extracted/{aid}_{cfg}.yaml')
            if gt:
                all_preds.update(get_semantic(gt, 'pre'))
                all_preds.update(get_semantic(gt, 'post'))
            if ex:
                all_preds.update(get_semantic(ex, 'pre'))
                all_preds.update(get_semantic(ex, 'post'))
    fetch_missing_embeddings(all_preds, cache)

    report = []
    for aid in agents:
        for cfg in configs:
            entry = diagnose_pair(aid, cfg, cache)
            if entry:
                report.append(entry)

    Path('results').mkdir(exist_ok=True)
    with open('results/rq1_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Wrote {len(report)} agent/config diagnostics to results/rq1_diagnostics.json\n')

    structural_tout_flags = sum(1 for e in report if 'note' in e['tout'])
    print(f'tout: {structural_tout_flags}/{len(report)} pairs show zero top-level overlap '
          f'but matching field name(s) nested in an array (structural mismatch pattern)')

    for field in ('pre', 'post'):
        sims = [m['similarity'] for e in report for m in e[field]['matches']
                if m['similarity'] is not None]
        if sims:
            near_miss = sum(1 for s in sims if MATCH_THRESHOLD - 0.15 <= s < MATCH_THRESHOLD)
            print(f'{field}: {len(sims)} gt-predicate comparisons, '
                  f'mean best-match similarity {sum(sims) / len(sims):.3f}, '
                  f'{near_miss} within 0.15 of threshold ({MATCH_THRESHOLD}) but not clearing it')
        else:
            print(f'{field}: no comparable predicate pairs found')


if __name__ == '__main__':
    main()