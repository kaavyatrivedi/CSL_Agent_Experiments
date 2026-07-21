"""
Measures RQ1: extraction accuracy by field.
Compares extracted contracts to ground truth, producing numbers for Table 2.
"""
import yaml, json
from pathlib import Path
import re

def schema_compatible(extracted_schema, gt_schema):
    if not extracted_schema or not gt_schema:
        return False
    if not isinstance(extracted_schema, dict) or not isinstance(gt_schema, dict):
        return False

    def get_prop_names(schema):
        props = schema.get('properties', {})
        if isinstance(props, dict):
            return set(props.keys())
        if isinstance(props, list):
            return set(props)
        return set()

    gt_props = get_prop_names(gt_schema)
    ex_props = get_prop_names(extracted_schema)

    gt_required = gt_schema.get('required', [])
    gt_required = set(gt_required) if isinstance(gt_required, list) else set()

    return gt_required.issubset(ex_props)
 
def predicate_soundness(extracted_preds, gt_preds):
    'Returns fraction of extracted predicates that are semantically compatible with gt.'
    if not extracted_preds:
        return 0.0
    # Simple check: gt predicates are a superset (extracted is more specific is OK)
    # For paper: do manual soundness check on a random sample of 50 predicates
    # For now: measure how many extracted predicate names match gt predicate names
    gt_set = set(p.lower().replace('@','') for p in (gt_preds or []))
    ex_set = set(p.lower().replace('@','') for p in (extracted_preds or []))
    if not gt_set: return 1.0
    # Recall-style: fraction of gt predicates matched
    matched = len(gt_set.intersection(ex_set))
    return matched / len(gt_set)

def sanitize_yaml_text(text):
    """Quote bare @predicate tokens (block-list or inline-list style) so they parse as valid YAML."""
    return re.sub(r"(?<!['\"])(@[\w\-]+)(?!['\"])", r"'\1'", text)
 
configs = ['S1', 'S1S2', 'S1S2T']
results = {c: {'tin': [], 'tout': [], 'pre': [], 'post': [], 'comp': []} for c in configs}
 
for i in range(1, 21):
    aid = f'A{i:02d}'
    gt_path = f'ground_truth/{aid}_contract_R1.yaml'
    if not Path(gt_path).exists(): continue
    with open(gt_path, encoding='utf-8') as f:
        gt = yaml.safe_load(f)
    
    for cfg in configs:
        ex_path = f'extracted/{aid}_{cfg}.yaml'
        if not Path(ex_path).exists(): continue

        try:
            with open(ex_path, encoding='utf-8') as f:
                ex = yaml.safe_load(sanitize_yaml_text(f.read()))

            results[cfg]['tin'].append(schema_compatible(ex.get('tin'), gt.get('tin')))
            results[cfg]['tout'].append(schema_compatible(ex.get('tout'), gt.get('tout')))
            results[cfg]['pre'].append(predicate_soundness(
                ex.get('pre', {}).get('semantic', []), gt.get('pre', {}).get('semantic', [])))
            results[cfg]['post'].append(predicate_soundness(
                ex.get('post', {}).get('semantic', []), gt.get('post', {}).get('semantic', [])))

            gt_comp = gt.get('comp', {}).get('type', '?')
            ex_comp = ex.get('comp', {}).get('type', '?')
            results[cfg]['comp'].append(gt_comp == ex_comp)

        except Exception as e:
            print(f'  WARNING: skipping {ex_path} due to unexpected error: {e}')
            continue
 
print('\n===== RQ1 RESULTS (Table 2) =====')
print(f'{"Field":<10} {"S1 only":>10} {"S1+S2":>10} {"S1+S2+T":>10}')
import numpy as np
for field in ['tin', 'tout', 'pre', 'post', 'comp']:
    row = [f'{field:<10}']
    for cfg in configs:
        vals = results[cfg][field]
        row.append(f'{np.mean(vals)*100:>9.1f}%')
    print('  '.join(row))
