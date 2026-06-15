"""
Runs a simple GPT-4o agent on SWE-Bench tasks and records traces.
Each trace captures: input, output, and whether output is well-formed.
Reminder: When collecting  traces on a new agent, replace its system 
prompt and trace files' titles in the code so it reads from and writes
to the correct places.
"""

import openai, json, time
from dotenv import load_dotenv
load_dotenv()
 
client = openai.OpenAI()
 
# Load your system prompt for whichever agent's prompt is inputted
with open('corpus/A01_system_prompt.txt') as f:
    system_prompt = f.read()
 
# Load tasks
with open('corpus/swebench_tasks_30.json') as f:
    tasks = json.load(f)
 
traces = []
for task in tasks[:30]:   # start with 30
    issue_text = task['problem_statement']
    repo = task['repo']
    
    # Call the agent
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Issue: {issue_text}\nRepo: {repo}'}
        ],
        max_tokens=1000
    )
    output = response.choices[0].message.content
    
    # Check if output is well-formed (basic structural check)
    is_valid_patch = output.strip().startswith('---') or '@@' in output
    
    traces.append({
        'agent_id': 'A01',
        'task_id': task['instance_id'],
        'input': {'issue': issue_text, 'repo': repo},
        'output': output,
        'post_satisfied': is_valid_patch,   # True = postcondition met
    })
    time.sleep(1)  # avoid rate limits
    print(f"Trace {len(traces)}/30 collected")
 
# Save traces
with open(f'corpus/A02_traces.json', 'w') as f:
    json.dump(traces, f, indent=2)
print(f'Saved {len(traces)} traces')
print(f'Success rate: {sum(t["post_satisfied"] for t in traces)/len(traces):.2%}')