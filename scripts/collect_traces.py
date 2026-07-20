"""
Runs a simple GPT-4o agent on SWE-Bench tasks and records traces.
Each trace captures: input, output, and whether output is well-formed.
Reminder: When collecting traces on a new agent, just change AGENT_ID below —
file paths and the output validator are looked up from it automatically.
(Only the alternate prompt-building blocks for A10/A12/A14 still need
un-commenting, see the notes inside the loop.)
"""

import openai, json, time, re
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI()

AGENT_ID = 'A20'

# Structural postcondition check per agent: is the output well-formed?
# (Same rules as before, previously kept as commented-out is_valid_output lines.)
VALIDATORS = {
    'A01': lambda o: o.strip().startswith('LOCALISATION RESULT:') and 'File:' in o and 'Reason:' in o,
    'A02': lambda o: o.strip().startswith('---') or '@@' in o,  # tests if valid patch
    'A03': lambda o: 'VERIFICATION RESULT:' in o and ('Verdict: PASS' in o or 'Verdict: FAIL' in o),
    'A04': lambda o: o.strip().startswith('FILE:') and '```' in o,
    'A05': lambda o: o.strip().startswith('RELEVANT FILES:'),
    'A06': lambda o: o.strip().startswith('FILE:') and '```' in o,
    'A07': lambda o: bool(re.search(r'<{3,} SEARCH', o)) and bool(re.search(r'>{3,} REPLACE', o)),
    'A08': lambda o: bool(re.search(r'<{3,} SEARCH', o)) and bool(re.search(r'>{3,} REPLACE', o)),
    'A09': lambda o: o.strip().startswith('ROOT CAUSE ANALYSIS:') and 'Root cause location:' in o,
    'A10': lambda o: o.strip().startswith('---') or '@@' in o,  # tests if valid patch
    'A11': lambda o: (o.strip().startswith('---') or '@@' in o) and 'CONFLICT:' not in o or ('CONFLICT:' in o and 'Resolution required: manual' in o),
    'A12': lambda o: o.strip().startswith('TEST RUN REPORT:') and 'Overall verdict:' in o and 'Results:' in o,
    'A13': lambda o: o.strip().startswith('ISSUE PARSE RESULT:') and 'Affected components:' in o and 'Expected behaviour:' in o,
    'A14': lambda o: o.strip().startswith('CODE SUMMARY:') and 'Purpose:' in o and 'Inputs:' in o and 'Outputs:' in o,
    'A15': lambda o: '```' in o and ('tsx' in o or 'jsx' in o or 'typescript' in o),
    'A16': lambda o: '```' in o and ('function' in o or 'const ' in o or 'import ' in o),
    'A17': lambda o: '```' in o and ('import ' in o or 'require(' in o),
    'A18': lambda o: '```' in o and ('className' in o or 'import ' in o),
    'A19': lambda o: 'THE SYSTEM SHALL' in o or ('requirements' in o.lower() and '##' in o),
    'A20': lambda o: '```' in o and ('bash' in o or '```sh' in o or '$' in o),
}

# Load your system prompt for whichever agent is inputted
# Many modern system prompts require UTF-8 decoding because they use Markdown format
with open(f'corpus/{AGENT_ID}_system_prompt.txt', encoding='utf-8') as f:
    system_prompt = f.read()

# Load tasks ( good practice to add it here too)
with open('corpus/swebench_tasks_30.json', encoding='utf-8') as f:
    tasks = json.load(f)

traces = []
for task in tasks[:30]:   # start with 30
    issue_text = task['problem_statement']
    repo = task['repo']

    # Call the agent - basic response
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Issue: {issue_text}\nRepo: {repo}'}
        ],
        max_tokens=1000
    )

    # A10, 14: uses SWE-Bench patch field as code to refer to
    # patch_text = task.get('patch', 'No patch provided')
    # response = client.chat.completions.create(
    #     model='gpt-4o',
    #     messages=[
    #         {'role': 'system', 'content': system_prompt},
    #         {'role': 'user', 'content': f'Issue: {issue_text}\nRepo: {repo}\nPatch:\n{patch_text}'}
    #     ],
    #     max_tokens=1000
    # )

    # A12: uses SWE-Bench patch and test file references
    # patch_text = task.get('patch', 'No patch provided')
    # test_text = task.get('test_patch', 'No test suite provided')
    # response = client.chat.completions.create(
    #     model='gpt-4o',
    #     messages=[
    #         {'role': 'system', 'content': system_prompt},
    #         {'role': 'user', 'content': f'Issue: {issue_text}\nRepo: {repo}\nPatch:\n{patch_text}\nTest suite:\n{test_text}'}
    #     ],
    #     max_tokens=1000
    # )

    output = response.choices[0].message.content

    # Check if output is well-formed (basic structural check). This varies based on the agent.
    is_valid_output = VALIDATORS[AGENT_ID](output)

    traces.append({
        'agent_id': AGENT_ID,
        'task_id': task['instance_id'],
        'input': {'issue': issue_text, 'repo': repo},
        'output': output,
        'post_satisfied': is_valid_output,   # True = postcondition met
    })
    time.sleep(1)  # avoid rate limits
    print(f"Trace {len(traces)}/30 collected")

# Save traces
with open(f'corpus/{AGENT_ID}_traces.json', 'w') as f:
    json.dump(traces, f, indent=2)
print(f'Saved {len(traces)} traces')
print(f'Success rate: {sum(t["post_satisfied"] for t in traces)/len(traces):.2%}')
