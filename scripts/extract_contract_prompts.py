EXTRACTION_PROMPT_ORIGINAL = '''
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

"""
Two EXTRACTION_PROMPT variants for scripts/extract_contract.py, built from
the diagnosis of A01-A20's ground truth:

  - pre.semantic / post.semantic in your R1 ground truth is NOT open-ended
    predicate naming -- it's a small, closed, reused taxonomy (16 pre
    categories, 19 post categories across 20 agents, several shared
    verbatim across agents with identical keyword lists). The original
    prompt asked the model to freely generate predicate strings, which is
    a much harder and noisier task than the ground truth actually calls
    for, and explains why pre/post predicate matching came out so low
    even after switching to embedding-based similarity.

Pick ONE of these two prompts to replace EXTRACTION_PROMPT with, run it,
and re-measure with measure_rq1.py -- do not merge them, since they test
different things:

  EXTRACTION_PROMPT_CLASSIFICATION
    Hands the model the exact closed taxonomy and asks it to select from
    it. This should push pre/post scores up substantially, but changes
    what RQ1 measures: it becomes "can the model disambiguate among 16-19
    known categories" rather than "can the model discover the right
    concept from scratch." If your S1 vs S1+S2 vs S1+S2+T comparison is
    meant to show how much extra context (docs, traces) helps concept
    discovery, this framing may compress that comparison, since even
    S1-only keyword matching could get partway there.

  EXTRACTION_PROMPT_FREE_GENERATION
    Keeps free-form generation (no closed list given) but fixes the
    concrete framing bug from the original prompt: it now explicitly
    asks for a domain CATEGORY label ('@input_is_<category>' /
    '@output_is_<category>'), not an operational-readiness or
    output-quality judgment, since that mismatch (not just wording) was
    the dominant driver of near-zero similarity scores in the diagnostic
    sample (A01: '@input_is_github_issue' vs extracted
    '@working_directory_accessible' -- different concepts entirely, not
    synonyms). This variant is closer to a fair test of whether more
    context (S2/T) helps the model discover the right concept
    unassisted, since it isn't handed the answer set.

Both preserve the {sat_rate} / {{mu: 0.0, sigma: 0.0}} format placeholders
used by extract_contract.py's `.format(sat_rate=sat_rate)` call -- only
the tin/tout/pre/post instructions changed.
"""


EXTRACTION_PROMPT_CLASSIFICATION = '''
You are a contract extraction expert. Given documentation about an LLM agent,
produce a CSL-Agent contract in YAML format with these exact fields:

contract_id: (agent name + version)

tin:
  type: object
  properties: (list all input fields with types)
  required: (list required fields)

pre:
  structural: (boolean expression over input fields)
  semantic: (list of quoted predicate strings -- see closed list below)

post:
  structural: (boolean expression over output fields)
  semantic: (list of quoted predicate strings -- see closed list below)

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

IMPORTANT -- pre.semantic and post.semantic must be chosen from the FIXED,
CLOSED lists below. Do NOT invent new predicate names. Select every
category that applies to this agent's input/output -- usually exactly
one, occasionally two (e.g. an agent that both validates a patch and
merges it would get both '@output_is_valid_patch' and
'@output_is_merged_patch').

Choose a category if the agent's documented purpose matches its
description, or its typical input/output contains language similar to
its keywords.

PRE CATEGORIES (choose from these for pre.semantic):
  '@input_is_github_issue'          -- issue, bug, error, fix, problem, report
  '@input_has_file_location'        -- file, localise, localize, location, line, path
  '@input_is_patch_and_tests'       -- verify, verification, validate, patch, test, pass
  '@input_is_code_specification'    -- write, generate, implement, create, function, class, module, spec
  '@input_is_file_search_query'     -- find, locate, search, relevant, files, where
  '@input_is_code_and_instructions' -- refactor, rewrite, rename, restructure, edit, update, docstring, comment
  '@input_is_stack_trace'           -- stack trace, traceback, exception, crash, root cause, stacktrace
  '@input_is_multiple_patches'      -- merge, combine, conflict, patches, parallel
  '@input_is_test_suite'            -- run tests, test suite, execute tests, test results, pytest, unittest
  '@input_is_code_function'         -- summarise, summarize, explain, describe, what does, document
  '@input_is_ui_description'        -- component, react, tsx, jsx, ui, interface, button, form, page
  '@input_is_app_description'       -- web app, application, build, full stack, frontend, backend
  '@input_is_runnable_app_spec'     -- sandbox, run, deploy, replit, executable, runnable
  '@input_is_ui_design'             -- clone, replicate, copy, screenshot, design, mockup, looks like
  '@input_is_feature_description'   -- spec, requirements, shall, system, feature, kiro, specification
  '@input_is_terminal_task'         -- terminal, bash, shell, script, command, cli, warp

If truly none of the above applies, use '@input_is_other'.

POST CATEGORIES (choose from these for post.semantic):
  '@output_is_localisation_result'
  '@output_is_valid_patch'
  '@output_preserves_test_files'
  '@output_is_verification_result'
  '@output_is_code_file'
  '@output_is_file_list'
  '@output_is_search_replace_block'
  '@output_has_filepath_before_block'
  '@output_is_root_cause_analysis'
  '@output_is_merged_patch'
  '@output_is_test_run_report'
  '@output_is_parsed_issue'
  '@output_is_code_summary'
  '@output_is_react_component'
  '@output_is_web_app_code'
  '@output_is_runnable_code'
  '@output_is_ui_clone'
  '@output_is_requirements_spec'
  '@output_is_shell_script'

If truly none of the above applies, use '@output_is_other'.

Return ONLY the YAML. No explanation.
'''


EXTRACTION_PROMPT_FREE_GENERATION = '''
You are a contract extraction expert. Given documentation about an LLM agent,
produce a CSL-Agent contract in YAML format with these exact fields:

contract_id: (agent name + version)

tin:
  type: object
  properties: (list all input fields with types)
  required: (list required fields)

pre:
  structural: (boolean expression over input fields)
  semantic: (list of quoted predicate strings)

post:
  structural: (boolean expression over output fields)
  semantic: (list of quoted predicate strings)

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

IMPORTANT -- how to write pre.semantic and post.semantic:

A semantic predicate names the DOMAIN CATEGORY the input (or output)
belongs to -- NOT a check on whether the agent or environment is ready to
run, and NOT a judgment of whether the agent did a good job.

Use the naming pattern '@input_is_<category>' for pre.semantic and
'@output_is_<category>' for post.semantic, where <category> is the type
of object being described, in domain terms. Include more than one entry
only if the input/output genuinely belongs to multiple distinct
categories at once.

Correct examples (illustrative -- do not reuse these exact category
names unless they genuinely apply):
  pre.semantic:  ['@input_is_sql_query']
  post.semantic: ['@output_is_query_result_table']
  pre.semantic:  ['@input_is_support_ticket']
  post.semantic: ['@output_is_priority_label']

INCORRECT -- do not produce predicates like these. They describe agent
readiness or output quality, not the domain category of the input/output:
  ['@working_directory_accessible']    -- environment readiness, not input type
  ['@problem_statement_understood']    -- agent comprehension, not input type
  ['@correct_file_identification']     -- output quality judgment, not output type

If unsure what category to name, ask: "what single label would this
input/output be filed under if sorting a pile of examples into labeled
folders?" The folder label is the predicate.

Return ONLY the YAML. No explanation.
'''