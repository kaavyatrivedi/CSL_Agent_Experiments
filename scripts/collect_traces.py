"""
Runs a simple GPT-4o agent on SWE-Bench tasks and records traces.
Each trace captures: input, output, and whether output is well-formed.
Reminder: When collecting  traces on a new agent, replace its system 
prompt and trace files' titles in the code so it reads from and writes
to the correct places. (Ctrl+F "A0", "A1", and "A2" to find all instances.)
"""

import openai, json, time, re
from dotenv import load_dotenv
load_dotenv()
 
client = openai.OpenAI()

# Load your system prompt for whichever agent is inputted
# Many modern system prompts require UTF-8 decoding because they use Markdown format
with open('corpus/A20_system_prompt.txt', encoding='utf-8') as f:
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

    # is_valid_output = output.strip().startswith('LOCALISATION RESULT:') and 'File:' in output and 'Reason:' in output #A01
    # is_valid_output = output.strip().startswith('---') or '@@' in output #A02, A10, tests if valid patch
    # is_valid_output = 'VERIFICATION RESULT:' in output and ('Verdict: PASS' in output or 'Verdict: FAIL' in output) #for A03, test cases
    # is_valid_output = output.strip().startswith('FILE:') and '```' in output #A04, A06
    # is_valid_output = output.strip().startswith('RELEVANT FILES:') #A05
    # is_valid_output = bool(re.search(r'<{3,} SEARCH', output)) and bool(re.search(r'>{3,} REPLACE', output)) #A07, 08
    # is_valid_output = output.strip().startswith('ROOT CAUSE ANALYSIS:') and 'Root cause location:' in output #A09
    # is_valid_output = (output.strip().startswith('---') or '@@' in output) and 'CONFLICT:' not in output or ('CONFLICT:' in output and 'Resolution required: manual' in output) #A11
    # is_valid_output = output.strip().startswith('TEST RUN REPORT:') and 'Overall verdict:' in output and 'Results:' in output #A12
    # is_valid_output = output.strip().startswith('ISSUE PARSE RESULT:') and 'Affected components:' in output and 'Expected behaviour:' in output #A13
    # is_valid_output = output.strip().startswith('CODE SUMMARY:') and 'Purpose:' in output and 'Inputs:' in output and 'Outputs:' in output #A14
    # is_valid_output = '```' in output and ('tsx' in output or 'jsx' in output or 'typescript' in output) #A15
    # is_valid_output = '```' in output and ('function' in output or 'const ' in output or 'import ' in output) #A16
    # is_valid_output = '```' in output and ('import ' in output or 'require(' in output) #A17
    # is_valid_output = '```' in output and ('className' in output or 'import ' in output) #A18
    # is_valid_output = 'THE SYSTEM SHALL' in output or ('requirements' in output.lower() and '##' in output) #A19
    # is_valid_output = '```' in output and ('bash' in output or '```sh' in output or '$' in output) #A20



    traces.append({
        'agent_id': 'A20',
        'task_id': task['instance_id'],
        'input': {'issue': issue_text, 'repo': repo},
        'output': output,
        'post_satisfied': is_valid_output,   # True = postcondition met
    })
    time.sleep(1)  # avoid rate limits
    print(f"Trace {len(traces)}/30 collected")
 
# Save traces
with open(f'corpus/A20_traces.json', 'w') as f:
    json.dump(traces, f, indent=2)
print(f'Saved {len(traces)} traces')
print(f'Success rate: {sum(t["post_satisfied"] for t in traces)/len(traces):.2%}')