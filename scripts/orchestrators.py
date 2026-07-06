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
        # Pre-condition predicates:
        # These are checked by check_pre_compatibility before dispatch

        '@input_is_github_issue': 
            ['issue', 'bug', 'error', 'fix', 'problem', 'report'],           # A01, A13

        '@input_has_file_location': 
            ['file', 'localise', 'localize', 'location', 'line', 'path'],    # A02, A10

        '@input_is_patch_and_tests': 
            ['verify', 'verification', 'validate', 'patch', 'test', 'pass'], # A03

        '@input_is_code_specification': 
            ['write', 'generate', 'implement', 'create', 'function',
            'class', 'module', 'spec'],                                      # A04, A06

        '@input_is_file_search_query': 
            ['find', 'locate', 'search', 'relevant', 'files', 'where'],      # A05

        '@input_is_code_and_instructions': 
            ['refactor', 'rewrite', 'rename', 'restructure', 'edit',
            'update', 'docstring', 'comment'],                               # A07, A08

        '@input_is_stack_trace': 
            ['stack trace', 'traceback', 'exception', 'crash',
            'root cause', 'stacktrace'],                                     # A09

        '@input_is_multiple_patches': 
            ['merge', 'combine', 'conflict', 'patches', 'parallel'],         # A11

        '@input_is_test_suite': 
            ['run tests', 'test suite', 'execute tests', 'test results',
            'pytest', 'unittest'],                                           # A12

        '@input_is_code_function': 
            ['summarise', 'summarize', 'explain', 'describe',
            'what does', 'document'],                                        # A14

        '@input_is_ui_description': 
            ['component', 'react', 'tsx', 'jsx', 'ui', 'interface',
            'button', 'form', 'page'],                                       # A15

        '@input_is_app_description': 
            ['web app', 'application', 'build', 'full stack',
            'frontend', 'backend'],                                          # A16

        '@input_is_runnable_app_spec': 
            ['sandbox', 'run', 'deploy', 'replit', 'executable',
            'runnable'],                                                     # A17

        '@input_is_ui_design': 
            ['clone', 'replicate', 'copy', 'screenshot', 'design',
            'mockup', 'looks like'],                                         # A18

        '@input_is_feature_description': 
            ['spec', 'requirements', 'shall', 'system', 'feature',
            'kiro', 'specification'],                                        # A19

        '@input_is_terminal_task': 
            ['terminal', 'bash', 'shell', 'script', 'command',
            'cli', 'warp'],                                                  # A20


        # Post-condition predicates
        # These are NOT used by check_pre_compatibility. They go to the
        # Tier 2 GPT-4o-mini classifier, which converts them to natural
        # language automatically. Listed here for completeness/documentation
        # but predicate_keywords lookups for these will simply return []
        # and correctly not interfere with pre-condition checking.

        '@output_is_localisation_result':   [],   # A01
        '@output_is_valid_patch':           [],   # A02, A10, A11
        '@output_is_verification_result':   [],   # A03
        '@output_is_code_file':             [],   # A04, A06
        '@output_is_file_list':             [],   # A05
        '@output_is_search_replace_block':  [],   # A07, A08
        '@output_is_root_cause_analysis':   [],   # A09
        '@output_is_merged_patch':          [],   # A11
        '@output_is_test_run_report':       [],   # A12
        '@output_is_parsed_issue':          [],   # A13
        '@output_is_code_summary':          [],   # A14
        '@output_is_react_component':       [],   # A15
        '@output_is_web_app_code':          [],   # A16
        '@output_is_runnable_code':         [],   # A17
        '@output_is_ui_clone':              [],   # A18
        '@output_is_requirements_spec':     [],   # A19
        '@output_is_shell_script':          [],   # A20
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
