"""
Extraction pipeline: given agent documentation, produces a CSL-Agent contract.
Runs 4 configurations: S1 only, S1+S2, S1+S2+traces, S1+S2+traces+human.
"""
import openai, json, yaml, os
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
 
def extract_contract(agent_id, use_docs=False, use_traces=False, sat_rate=0.0):
    # Build context
    context = ''
    with open(f'corpus/{agent_id}_system_prompt.txt') as f:
        context += f'SYSTEM PROMPT:\n{f.read()}\n\n'
    if use_docs:
        doc_file = f'corpus/{agent_id}_docs.txt'
        if os.path.exists(doc_file):
            with open(doc_file) as f:
                context += f'DOCUMENTATION:\n{f.read()}\n\n'
    if use_traces:
        with open(f'corpus/{agent_id}_traces.json') as f:
            traces = json.load(f)[:5]   # first 5 examples only
        context += 'EXAMPLE INPUTS/OUTPUTS:\n'
        for t in traces:
            context += f'Input: {json.dumps(t["input"])}\nOutput: {str(t["output"])[:200]}\n---\n'
    
    prompt = EXTRACTION_PROMPT.format(sat_rate=sat_rate)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'Agent documentation:\n\n{context}'}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content
 
# Run all 4 configurations for all 25 agents
import subprocess, time
from scripts.compute_prob import compute_prob_field
 
for i in range(1, 26):
    aid = f'A{i:02d}'
    prob = compute_prob_field(aid)
    sat = prob['satisfaction_rate']
    
    configs = [
        ('S1',       False, False),
        ('S1S2',     True,  False),
        ('S1S2T',    True,  True),
    ]
    for cfg_name, use_docs, use_traces in configs:
        print(f'Extracting {aid} config {cfg_name}...')
        yaml_text = extract_contract(aid, use_docs, use_traces, sat)
        # Clean up GPT response (sometimes adds ```yaml fences)
        yaml_text = yaml_text.replace('```yaml','').replace('```','').strip()
        with open(f'extracted/{aid}_{cfg_name}.yaml', 'w') as f:
            f.write(yaml_text)
        time.sleep(1)
 
print('All extractions complete')
