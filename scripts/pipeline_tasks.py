# Save as scripts/pipeline_tasks.py
# 10 pipeline task definitions
 
PIPELINE_TASKS = [
    {
        'task_id': 'P01',
        'description': 'Fix bug: cache module has swapped TTL parameter',
        'subtasks': [
            {'id': 'P01-T1', 'description': 'Identify buggy file from issue'},
            {'id': 'P01-T2', 'description': 'Generate a patch for the bug'},
            {'id': 'P01-T3', 'description': 'Review the generated patch'},
            {'id': 'P01-T4', 'description': 'Verify patch passes tests'},
        ]
    },
    {
        'task_id': 'P02',
        'description': 'Add unit tests for the authentication module',
        'subtasks': [
            {'id': 'P02-T1', 'description': 'Find existing auth functions'},
            {'id': 'P02-T2', 'description': 'Generate test cases for each function'},
            {'id': 'P02-T3', 'description': 'Write the test file'},
        ]
    },
    # ... define 8 more similar tasks
]
