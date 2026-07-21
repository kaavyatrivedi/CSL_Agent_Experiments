"""
Measures RQ1: extraction accuracy by field.

Compares extracted contracts (extracted/AXX_{cfg}.yaml) to ground-truth
contracts (ground_truth/AXX_contract_R1.yaml), producing numbers for
Table 2.

Changes from the original version:
  - get_prop_names() no longer crashes on list-of-dict `properties` blocks
    (root cause of the A04_S1S2 / A20_S1S2T crashes).
  - tout is now scored with Jaccard similarity over property-name sets,
    instead of the required-subset check (which was vacuously 100%,
    since ground-truth tout blocks never set `required`).
  - pre/post predicate matching is now embedding-based (OpenAI
    text-embedding-3-small, cosine similarity, greedy one-to-one
    matching, threshold 0.75) instead of exact lowercase string
    matching, so e.g. @valid_github_issue and @issue_is_wellformed can
    match even though they don't share characters.
  - Each field (tin/tout/pre/post/comp) is scored in its own try/except,
    so one field crashing no longer discards the other four for that
    (agent, config) pair.
  - Every printed cell reports its own N, since failures/missing files
    can make N differ column to column.
  - A structured failure log is written to results/rq1_failures.json.

Ground truth source: ground_truth/AXX_contract_R1.yaml (Researcher 1
only -- not the 3-annotator FINAL consensus). Agent range: A01-A20 only.
These match your current data on disk; revisit if you re-run Phase 2's
disagreement resolution or extend the corpus to A21-A25.
"""

import os
import re
import json
import yaml
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI()

CONFIGS = ['S1', 'S1S2', 'S1S2T']
FIELDS = ['tin', 'tout', 'pre', 'post', 'comp']
AGENT_RANGE = range(1, 21)  # A01-A20

EMBED_MODEL = 'text-embedding-3-small'
MATCH_THRESHOLD = 0.75
EMBED_CACHE_PATH = 'embedding_cache.json'


# ---------- YAML helpers ----------

def sanitize_yaml_text(text):
    """Quote bare @predicate tokens so they parse as valid YAML."""
    return re.sub(r"(?<!['\"])(@[\w\-]+)(?!['\"])", r"'\1'", text)


def get_prop_names(schema):
    """
    Extract the set of property names from a tin/tout `properties` block.
    Handles both the documented mapping style:
        properties: {file: {type: string}, line: {type: integer}}
    and the list style GPT-4o sometimes produces instead:
        properties: [{name: file, type: string}, {name: line, type: integer}]
    which used to crash with "cannot use 'dict' as a set element".
    """
    if not isinstance(schema, dict):
        return set()
    props = schema.get('properties', {})
    if isinstance(props, dict):
        return set(props.keys())
    if isinstance(props, list):
        names = set()
        for p in props:
            if isinstance(p, dict):
                name = p.get('name') or p.get('field') or next(iter(p), None)
                if name:
                    names.add(name)
            elif isinstance(p, str):
                names.add(p)
        return names
    return set()


def get_semantic(block, key):
    """Safely pull pre.semantic / post.semantic as a list, regardless of
    malformed input (missing keys, wrong types)."""
    if not isinstance(block, dict):
        return []
    val = block.get(key, {})
    if not isinstance(val, dict):
        return []
    sem = val.get('semantic', [])
    return sem if isinstance(sem, list) else []


# ---------- tin: required-subset check (unchanged logic) ----------

def tin_score(extracted_schema, gt_schema):
    """1.0 if every gt-required field name appears in the extracted
    properties, else 0.0. Meaningful because ground-truth tin blocks do
    set `required`."""
    if not isinstance(extracted_schema, dict) or not isinstance(gt_schema, dict):
        return 0.0
    ex_props = get_prop_names(extracted_schema)
    gt_required = gt_schema.get('required', [])
    gt_required = set(gt_required) if isinstance(gt_required, list) else set()
    if not gt_required:
        return 1.0
    return float(gt_required.issubset(ex_props))


# ---------- tout: property-set Jaccard (replaces the vacuous check) ----------

def tout_score(extracted_schema, gt_schema):
    """
    Jaccard similarity between extracted and ground-truth tout property
    name sets. Ground-truth tout blocks never populate `required`, which
    made the old required-subset check unconditionally True; this
    replaces it with a real comparison of what properties were named.
    """
    if not isinstance(extracted_schema, dict) or not isinstance(gt_schema, dict):
        return 0.0
    ex_props = get_prop_names(extracted_schema)
    gt_props = get_prop_names(gt_schema)
    if not gt_props and not ex_props:
        return 1.0
    if not gt_props or not ex_props:
        return 0.0
    return len(gt_props & ex_props) / len(gt_props | ex_props)


# ---------- pre/post: embedding-based predicate matching (Fix 3) ----------

def clean_predicate(p):
    return p.replace('@', '').replace('_', ' ').strip().lower()


def load_embed_cache():
    if Path(EMBED_CACHE_PATH).exists():
        with open(EMBED_CACHE_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_embed_cache(cache):
    with open(EMBED_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f)


def fetch_missing_embeddings(predicates, cache):
    """Batch-fetch embeddings for any predicate not already in cache.
    Cached by the raw '@predicate_name' string; the embedding itself is
    computed from the cleaned/de-symbolized text."""
    missing = sorted({p for p in predicates if p not in cache})
    if not missing:
        return
    print(f'Fetching {len(missing)} new predicate embeddings...')
    BATCH = 100
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i + BATCH]
        cleaned = [clean_predicate(p) for p in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=cleaned)
        for raw, item in zip(batch, resp.data):
            cache[raw] = item.embedding
    save_embed_cache(cache)


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def predicate_soundness_semantic(extracted_preds, gt_preds, cache):
    """
    Recall-style score: fraction of gt predicates that have a
    semantically matching extracted predicate (cosine similarity >=
    MATCH_THRESHOLD), under greedy one-to-one matching (so one
    extracted predicate can't 'cover' multiple gt predicates).
    """
    if not extracted_preds:
        return 0.0
    if not gt_preds:
        return 1.0

    pairs = []
    for i, g in enumerate(gt_preds):
        for j, e in enumerate(extracted_preds):
            if g not in cache or e not in cache:
                continue  # embedding fetch failed for this one; treat as no match
            pairs.append((cosine(cache[g], cache[e]), i, j))
    pairs.sort(reverse=True, key=lambda t: t[0])

    matched_gt, matched_ex, matches = set(), set(), 0
    for sim, i, j in pairs:
        if sim < MATCH_THRESHOLD:
            break
        if i in matched_gt or j in matched_ex:
            continue
        matched_gt.add(i)
        matched_ex.add(j)
        matches += 1

    return matches / len(gt_preds)


# ---------- main ----------

def main():
    os.makedirs('results', exist_ok=True)

    # --- Pass 1: load everything, recording load/parse failures ---
    gt_contracts = {}   # aid -> parsed gt dict
    ex_contracts = {}   # (aid, cfg) -> parsed ex dict
    failures = []       # [{agent, config, field, error}, ...]

    for i in AGENT_RANGE:
        aid = f'A{i:02d}'
        gt_path = f'ground_truth/{aid}_contract_R1.yaml'
        if not Path(gt_path).exists():
            failures.append({'agent': aid, 'config': None, 'field': 'ALL',
                              'error': f'ground truth file not found: {gt_path}'})
            continue
        try:
            with open(gt_path, encoding='utf-8') as f:
                gt_contracts[aid] = yaml.safe_load(f)
        except Exception as e:
            failures.append({'agent': aid, 'config': None, 'field': 'ALL',
                              'error': f'ground truth parse error: {e}'})
            continue

        for cfg in CONFIGS:
            ex_path = f'extracted/{aid}_{cfg}.yaml'
            if not Path(ex_path).exists():
                continue  # not yet extracted -- expected mid-run, not a failure
            try:
                with open(ex_path, encoding='utf-8') as f:
                    ex_contracts[(aid, cfg)] = yaml.safe_load(sanitize_yaml_text(f.read()))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'ALL',
                                  'error': f'extracted YAML parse error: {e}'})

    # --- Collect every predicate string that will need an embedding ---
    all_predicates = set()
    for gt in gt_contracts.values():
        all_predicates.update(get_semantic(gt, 'pre'))
        all_predicates.update(get_semantic(gt, 'post'))
    for ex in ex_contracts.values():
        all_predicates.update(get_semantic(ex, 'pre'))
        all_predicates.update(get_semantic(ex, 'post'))

    embed_cache = load_embed_cache()
    try:
        fetch_missing_embeddings(all_predicates, embed_cache)
    except Exception as e:
        print(f'WARNING: embedding fetch failed ({e}). '
              f'pre/post scores will treat unembedded predicates as non-matches.')

    # --- Pass 2: score each field independently ---
    results = {c: {f: [] for f in FIELDS} for c in CONFIGS}

    for i in AGENT_RANGE:
        aid = f'A{i:02d}'
        if aid not in gt_contracts:
            continue
        gt = gt_contracts[aid]

        for cfg in CONFIGS:
            ex = ex_contracts.get((aid, cfg))
            if not isinstance(ex, dict):
                continue  # missing or failed to parse -- already logged in pass 1

            try:
                results[cfg]['tin'].append(tin_score(ex.get('tin'), gt.get('tin')))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'tin', 'error': str(e)})

            try:
                results[cfg]['tout'].append(tout_score(ex.get('tout'), gt.get('tout')))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'tout', 'error': str(e)})

            try:
                results[cfg]['pre'].append(predicate_soundness_semantic(
                    get_semantic(ex, 'pre'), get_semantic(gt, 'pre'), embed_cache))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'pre', 'error': str(e)})

            try:
                results[cfg]['post'].append(predicate_soundness_semantic(
                    get_semantic(ex, 'post'), get_semantic(gt, 'post'), embed_cache))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'post', 'error': str(e)})

            try:
                gt_comp = gt.get('comp', {}).get('type', '?') if isinstance(gt.get('comp'), dict) else '?'
                ex_comp = ex.get('comp', {}).get('type', '?') if isinstance(ex.get('comp'), dict) else '?'
                results[cfg]['comp'].append(float(gt_comp == ex_comp))
            except Exception as e:
                failures.append({'agent': aid, 'config': cfg, 'field': 'comp', 'error': str(e)})

    # --- Print Table 2, with N per cell ---
    print('\n===== RQ1 RESULTS (Table 2) =====')
    print(f'{"Field":<10} {"S1 only":>16} {"S1+S2":>16} {"S1+S2+T":>16}')
    for field in FIELDS:
        row = [f'{field:<10}']
        for cfg in CONFIGS:
            vals = results[cfg][field]
            cell = f'{np.mean(vals) * 100:.1f}% (n={len(vals)})' if vals else 'n/a (n=0)'
            row.append(f'{cell:>16}')
        print(' '.join(row))

    # --- Partial-failure report ---
    print(f'\n===== FAILURES ({len(failures)} total) =====')
    if failures:
        by_field = {}
        for f in failures:
            by_field[f['field']] = by_field.get(f['field'], 0) + 1
        for field, count in sorted(by_field.items()):
            print(f'  {field}: {count}')
        print('  Full details written to results/rq1_failures.json')
    else:
        print('  none')

    with open('results/rq1_failures.json', 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=2)


if __name__ == '__main__':
    main()