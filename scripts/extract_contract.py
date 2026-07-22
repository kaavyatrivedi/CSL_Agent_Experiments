"""
Extraction pipeline: given agent documentation, produces a CSL-Agent contract.

Runs 3 configurations: S1 only, S1+S2, S1+S2+traces.
(The 4th, +human, is done manually in Step 3.3 — not part of this script.)

If adapting the extraction prompt, make sure to change the directory to which extracted 
YAML files are written (search 'extracted') so that the new prompt's results don't 
overwrite the old ones. The old results are used in the RQ1 analysis, so they must be preserved.

"""

import openai, json, yaml, os, time, re
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

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
  semantic: (list of quoted predicate strings like '@predicate_name' -- always wrap in single quotes)
post:
  structural: (boolean expression over output fields)
  semantic: (list of quoted predicate strings like '@predicate_name' -- always wrap in single quotes)
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


def sanitize_yaml_text(text):
    """
    Quote bare @predicate tokens so they parse as valid YAML.
    '@' is a reserved YAML character -- an unquoted list item like
    '- @valid_github_issue' fails to parse. GPT-4o follows the prompt's
    '@predicate_names' phrasing literally and produces exactly this
    pattern, so this is a safety net even with the updated prompt
    wording above, since compliance with the quoting instruction isn't
    guaranteed on every call.
    """
    return re.sub(r"^(\s*-\s*)(@[\w\-]+)\s*$", r"\1'\2'", text, flags=re.MULTILINE)


def call_with_retry(**kwargs):
    """
    Wraps the OpenAI call with exponential backoff, per the Appendix's
    'OpenAI API rate limit errors' fix. Retries up to 5 times, doubling
    the wait each time (1s, 2s, 4s, 8s, 16s) before giving up and
    re-raising whatever the final error was.
    """
    last_error = None
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            last_error = e
            wait = 2 ** attempt
            print(f'  Rate limited, waiting {wait}s (attempt {attempt + 1}/5)...')
            time.sleep(wait)
    # All 5 attempts failed — surface the error instead of silently
    # returning nothing, so a bad extraction doesn't get treated as valid.
    raise last_error


def extract_contract(agent_id, use_docs=False, use_traces=False, sat_rate=0.0):
    # Build context
    context = ''
    with open(f'corpus/{agent_id}_system_prompt.txt', encoding='utf-8') as f:
        context += f'SYSTEM PROMPT:\n{f.read()}\n\n'

    if use_docs:
        doc_file = f'corpus/{agent_id}_docs.txt'
        if os.path.exists(doc_file):
            with open(doc_file, encoding='utf-8') as f:
                context += f'DOCUMENTATION:\n{f.read()}\n\n'
        else:
            # Flag this instead of silently degrading to an S1-only run
            # while still being labeled as an S1S2/S1S2T result.
            print(f'  WARNING: {agent_id} has no docs file — this config '
                  f'will effectively run as S1-only, but will still be '
                  f'saved under a docs-included config name.')

    if use_traces:
        with open(f'corpus/{agent_id}_traces.json', encoding='utf-8') as f:
            traces = json.load(f)[:5]  # first 5 examples only
        context += 'EXAMPLE INPUTS/OUTPUTS:\n'
        for t in traces:
            context += f'Input: {json.dumps(t["input"])}\nOutput: {str(t["output"])[:200]}\n---\n'

    prompt = EXTRACTION_PROMPT.format(sat_rate=sat_rate)

    response = call_with_retry(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'Agent documentation:\n\n{context}'}
        ],
        max_tokens=800
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    # Make sure the output directory exists instead of assuming it does.
    os.makedirs('extracted_original', exist_ok=True)

    from scripts.compute_prob import compute_prob_field

    for i in range(1, 21):
        aid = f'A{i:02d}'
        prob = compute_prob_field(aid)
        sat = prob['satisfaction_rate']

        configs = [
            ('S1', False, False),
            ('S1S2', True, False),
            ('S1S2T', True, True),
        ]

        for cfg_name, use_docs, use_traces in configs:
            out_path = f'extracted_original/{aid}_{cfg_name}.yaml'
            if os.path.exists(out_path):
                print(f'Skipping {aid} config {cfg_name} (already extracted)')
                continue

            print(f'Extracting {aid} config {cfg_name}...')
            yaml_text = extract_contract(aid, use_docs, use_traces, sat)
            # Clean up GPT response (sometimes adds ```yaml fences)
            yaml_text = yaml_text.replace('```yaml', '').replace('```', '').strip()
            # Quote bare @predicate tokens so measure_rq1.py can parse this file
            yaml_text = sanitize_yaml_text(yaml_text)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(yaml_text)
            time.sleep(1)

    print('All extractions complete')