# Save as scripts/orchestrators.py
import yaml, random
from pathlib import Path
 
# Load all ground truth contracts
contracts = {}
for i in range(1, 26):
    aid = f'A{i:02d}'
    path = f'ground_truth/{aid}_contract_FINAL.yaml'
    if Path(path).exists():
        with open(path) as f:
            contracts[aid] = yaml.safe_load(f)
 
# ── Baseline B1: Random assignment (no contracts) ──────────────────
def dispatch_B1(subtask):
    'Assign to a random available agent -- no capability checking.'
    return random.choice(list(contracts.keys()))
 
# ── Baseline B2: Type matching only (DSPy-style) ────────────────────
def dispatch_B2(subtask):
    'Assign to first agent whose tin type seems compatible (keyword match).'
    desc = subtask['description'].lower()
    for aid, c in contracts.items():
        agent_desc = c.get('description', '').lower()
        # Simple keyword overlap
        if any(word in agent_desc for word in desc.split()):
            return aid
    return random.choice(list(contracts.keys()))  # fallback
 
# ── CSL-Agent: Pre-condition check ──────────────────────────────────
def check_pre_compatibility(subtask_desc, contract):
    '''
    Returns True if this subtask is compatible with the agent's Pre.
    In production, this calls a classifier. Here we use keyword rules
    based on the semantic predicate names.
    '''
    pre_semantics = contract.get('pre', {}).get('semantic', [])
    task_desc = subtask_desc.lower()
    
    # Map predicate names to expected task keywords
    predicate_keywords = {
        '@valid_github_issue': ['issue', 'bug', 'error', 'fix'],
        '@input_is_python_code': ['code', 'function', 'class', 'module'],
        '@input_is_patch': ['patch', 'diff', 'hunk', 'review'],
        '@input_is_test_description': ['test', 'unit test', 'coverage'],
    }
    for pred in pre_semantics:
        keywords = predicate_keywords.get(pred, [])
        if keywords and not any(kw in task_desc for kw in keywords):
            return False  # predicate violated
    return True
 
def dispatch_CSL(subtask):
    'Assign using CSL-Agent pre-condition check + reliability filter.'
    candidates = []
    for aid, c in contracts.items():
        if check_pre_compatibility(subtask['description'], c):
            p = c.get('prob', {}).get('satisfaction_rate', 0)
            candidates.append((aid, p))
    if not candidates:
        return random.choice(list(contracts.keys()))
    # Select highest-reliability compatible agent
    return max(candidates, key=lambda x: x[1])[0]
 
# Export dispatchers
DISPATCHERS = {
    'B1_NoContracts': dispatch_B1,
    'B2_TypedIO':     dispatch_B2,
    'CSL_Agent':      dispatch_CSL,
}
