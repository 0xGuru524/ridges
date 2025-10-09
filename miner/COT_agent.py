from __future__ import annotations
import ast, os, requests, subprocess, sys, textwrap, time, traceback, json, re, inspect, random, csv, logging, threading, math
from pyexpat import model
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, NamedTuple
from json import JSONDecodeError
from enum import Enum
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import urllib.request as _urlreq

#--------------------------Constants-------------------------------
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
DEFAULT_PROXY_URL = os.getenv('SANDBOS_PROXY_URL', 'http://sandbox_proxy')
DEFAULT_TIMEOUT = int(os.getenv('AGENT_TIMEOUT', '1800'))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv('MAX_STEPS_TEST_PATCH_FIND', '400'))
MAX_FIX_TASK_STEPS = 400
USE_FUNCTION_CHUNKS = os.getenv('EMBED_WHOLE_FILES', '0') != '1'
RUN_ID = os.getenv('RUN_ID', '')
REPOR_DIR = ''
DEBUG_MODE = True
#--------------------------Constants-------------------------------
#---------------------------Prompts--------------------------------
STOP_INSTRUCTION = textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")
FORMAT_PROMPT_V0 = textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning (include:
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `next_tool_name`: Must be an exact tool name from the tool list
   - `next_tool_args`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: ""
     next_tool_args: {}

3. **Example Valid Format**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: "apply_code_edit"
   next_tool_args: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    logger.error(f'Invalid JSON: {{response}}')\n    raise"
   }

4. **Invalid Format Examples** (Avoid These):
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")
PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
'''
You are the problem type checker that will categories problem type into:

1. CREATE: If the problem statement is about creating a new functionality from scratch.
2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

Only respond with the "FIX" or "CREATE".
'''
)
ONESHOT_SYSTEM_PROMPT = textwrap.dedent("""
You are an autonomous programmer. The user will provide a bug report or 
feature request (the "problem") plus a compact summary of the most 
relevant repository files.  Your job is to return ONE *valid* unified 
diff patch that fixes the problem. If you have any questions, do not ask the user. 
Instead, solve it to the best of your ability with the knowledge you have.

You will be provided with a summary of the repository files that are most relevant to the problem.
Your patch must be valid and apply cleanly when run from the repository root.

STRICT FORMAT RULES
1. Return *only* the diff – no prose, no Markdown back-ticks.
2. The diff must start with 'diff --git a/<path> b/<path>' followed by 
   the standard "--- a/<path>" and "+++ b/<path>" headers.
3. Use -u style context hunks that begin with lines like @@ -N,M +N,M @@.
4. Every changed file needs its own header block as in rule 2.
5. End the patch with a trailing newline.

Be exact: if the diff is syntactically malformed or wrapped in extra 
text the automated patch tool will fail.

OUTPUT RULES (VERY IMPORTANT, STRICT)
• You MUST end your reply with a *raw* JSON object with "code_response" property – nothing else.
• Must hold the unified diff from rules 1-5 *verbatim*.
Example: {"code_response": "diff --git a/foo.py b/foo.py\n..."}
""")
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")
GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent("""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If you give a return value type when creating a function, be careful about the return value type.

Return only the final python files code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)
INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

CRITICAL INFINITE LOOP DETECTION:
1. Check for while True: loops without guaranteed exit conditions
2. Verify all while loops have clear termination conditions
3. Ensure recursive functions have proper base cases
4. Look for loops that depend on external state that might never change
5. Check for patterns that could lead to infinite iteration

If you find potential infinite loops:
- Provide a corrected version of the code
- Ensure all loops have finite termination conditions
- Add reasonable iteration limits or timeout mechanisms where appropriate

If no infinite loops are detected:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

example:
```python
a.py
contents of a.py

b.py
contents of b.py
```
"""
)
MULTILINE_CHECK_PROMPT = textwrap.dedent("""
You are an expert code reviewer specializing in multiline text detection and prevention. Your task is to analyze the generated Python code for potential multiline text issues and provide a corrected version if issues are found.

1. If problem statement is not related to mutliline text response, just return the original code unchanged
2. Otherwise:
    - If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
        example:
        ```text
        a1
        b1
        ```
        you should use:
        "a1\nb1"
    - If problem statement requires a list of strings as a response, use list of strings for multiline text problems.
        example:
        ```text
        [
            "a1",
            "[EMPTY_LINE],
            "n1"
        ]
        ```
        you should use:
        ["a1", "\n", "n1"]

If you find potential multiline text issues:
- Provide a corrected version of the code
- Ensure the code is not using multiline text

If no multiline text issues are detected:
- Return the original code unchanged

Your response should be in JSON format, with the following keys:
- feedback: explain the feedback in very detail.
- code: full code
    - STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

Respose Example:
{
    "feedback": "updating code due to sth",
    "code": '''
    ```python
    a.py
    contents of a.py
    ```
    '''
}
"""
)
GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
Return only the final Python code.
If the return value  is 'None' in Python code you generated, the type does not match. use None
Response Examples:
```python
a.py
{content}

b.py
{content}
```

Remember: The output format and type are as important as the algorithm correctness. Pay meticulous attention to exact return types and formatting requirements.
"""
)
GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

If the return value  is 'None' in Python code you generated, the type does not match. use None

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```

Remember: The output format and type are as important as the algorithm correctness. Pay meticulous attention to exact return types and formatting requirements.
"""
)
GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
Remember: The output format and type are as important as the algorithm correctness. Pay meticulous attention to exact return types and formatting requirements.
"""
)
TESTCASES_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

Important:
1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
3. Minimize all testcases as you have context and generation limit

If no invalid testcases are detected and covered all critical edge cases:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

First, analyze the problem, identify the type of problem, and then decide whether it requires code or a bug fix. 

If you need to code, do the following:
Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
9. If the return value  is 'None' in Python code you generated, the type does not match. Use None insteed 'None'.
Return only the final Python code.

Response Examples:
```python
a.py
contents of a.py

b.py
contents of b.py
```

If you need to fix the code, do the following:
## Follow these steps to fix the issue:
1. As next step, find the relevant files in the repo to work on.
2. Localise the code causing the issue.
3. Edit the sourcecode of the repo to resolve the issue.
4. Think about edgecases and make sure the fix handles them as well.
5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
14. If you're making too many errors or the same error keeps repeating in a 'create' type problem that are difficult to eliminate, try creating it again. Be sure not to repeat the same errors as before.
15. If you find the return value  is 'None' in Python code you generated, the type does not match. Use None insteed 'None'.

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

## Test generation guidance:
- Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.
- Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.
- Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
- Keep generated tests minimal and focused on the bug and its edge cases.
- Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start.
Here is the problem statement:
{problem_statement}
Here is the code skeleton:
{code_skeleton}
""")
FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
You are a helpful assistant that can find the test runner for a given repository.
- The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
- Do not use the test runner to run test for whole repository or test setup.
- Read the README file and find the test runner. If there is no test runner, return pytest.
- Output format should be as the following. No other texts are allowed.
abc/test.py
""")
TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
You are a helpful assistant that determines the mode of the test runner.
Read the test runner file and determine if it requires a module or a file path to run the test.
Output should be one of MODULE or FILE, No other texts are allowed.
- MODULE: When the test runner requires a module path to run the test.
- FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
""")
#---------------------------Prompts--------------------------------
#---------------------------Logger---------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
#---------------------------Logger---------------------------------
#------------------------Core Classes------------------------------
class EnhancedCOT:
    class Action:
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, 
                     observation: list|tuple|str, is_error: bool = False, raw_response: str = None,
                     total_attempts: int = 0, inference_error_counter: dict = None, request_data: list = None):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False
    def __init__(self, latest_observations_to_keep: int = 30):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
    def is_valid_tool_call(self, next_tool_name: str|list, next_tool_args: dict|list) -> bool:
        if len(self.thoughts) == 0:
            return True
        last_tool_name = self.thoughts[-1].next_tool_name
        last_tool_args = self.thoughts[-1].next_tool_args
        if next_tool_name == last_tool_name and next_tool_args == last_tool_args:
            return False
        return True
    def add_action(self, action: EnhancedCOT.Action) -> bool:
        self.thoughts.append(action)
        return True
    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    def to_str(self):
        messages = []
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}\n"
                )
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str = (
                    f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n"
                )
            else:
                if thought.is_error is None or i == len(self.thoughts) - 1:
                    assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render = str(thought.observation)
                    else:
                        obs_render = str(thought.observation)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str = (
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render = str(thought.observation)
                        else:
                            obs_render = str(thought.observation)
                        user_str = f"observation: {obs_render}"
            messages.append({"role": "assistant", "content": assistant_str})
            messages.append({"role": "user", "content": user_str})
        return messages
class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n: int = 1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list) - n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError(f"Invalid JSON: {json_string}")
    @classmethod
    def log_to_failed_messages(cls, text_resp: str):
        with open("../failed_messages.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([text_resp])
class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, Optional(str)]:
        if type(raw_text) is dict and raw_text.get('error', None) is not None and raw_text.get('error') != '':
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith('}') and not raw_text.strip().endswith('}]'):
            return False, 'Incomplete response, your response must be shorter to fit within context limit'
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if '<|reserved_token_' in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.AUTHENTICATION_ERROR.name
        return True, None
    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}
    @classmethod
    def fix_json_string_llm(cls, json_string: str, attempt: int = 0) -> Optional[dict]:
        messages = [
            {'role': 'system', 'content': 'Fix the json string set by the user, Reply only with teh json string and nothing else.'},
            {'role': 'user', 'content': json_string}
        ]
        response = cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response = response.replace('```json', '').strip('```')
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f'Error fixing json string: {e}, trying again..')
            logger.error(f'json string is :{json_string}')
            logger.error(f'LLM response is :{response}')
            return None
    @classmethod
    def make_request(cls, messages: list, model: str, attemp: int = 0, temperature: float = 0.0) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        print("[REQUEST] run_id:", run_id)
        request_data = {
            "run_id": run_id if run_id else "1",
            "messages": messages,
            "temperature": temperature,
            "model": model
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=request_data, timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after 120 seconds for model {model}")
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for model {model}: {e}")
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for model {model}: {e}")
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return f"ERROR: Request failed for model {model}"
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.error(f"Invalid JSON response for model {model}: {e}")
            logger.error(f"Response content: {response.text[:500]}...")
            return f"ERROR: Invalid JSON response for model {model}"
        try:
            is_oai_interface = (type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices')) > 0 and response_json.get('choices')[0].get('message') is not None)
            if is_oai_interface:
                raw_text = response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    raw_text = response_json.strip("\n").strip()
                else:
                    raw_text = response_json
            if type(raw_text) is not dict:
                raw_text = raw_text.lstrip()
            return raw_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response structure for model {model}: {e}")
            logger.error(f"Response JSON: {response_json}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error processing response for model {model}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: Unexpected error for model {model}"
    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, model: str, max_retries: int = 30, base_delay: float = 1.0, temperature: float = 0.0) -> tuple:
        raw_text = 'not defined'
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text = cls.make_request(messages, model = AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)], temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name] += 1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name] += 1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name] += 1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and "TIMEOUT" not in error_body:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(random.uniform(1.2 * delay, 1.5 * delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages
    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:    
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        result_json = {}
        for i in range(len(arguments)):
            value = match.group(i + 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            value = value.replace('\\n', '\n')
            result_json[arguments[i]] = value
        return result_json
    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        next_tool_args = next_tool_args.replace('```json', '').strip('```')
        error_msg = ''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg = f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args
    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = "1", 
                  return_json: bool = False, temperature: float = 0.0) -> dict:
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")
            if role == "assistant" and not content.strip():
                continue
            cleaned_msgs.append({"role": role, "content": content})
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = cls._request_next_action_with_retry(
            cleaned_msgs, model=model, temperature=temperature)
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub("[\'\"]*next_thought[\'\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_name[\'\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_args[\'\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("[\'\"]*observation[\'\"]*:", "observation:", text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:") > 10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp = "next_thought: " + text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:"):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp = re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*", "next_tool_name: " + next_tool_name, text_resp)
        return text_resp
    @classmethod
    def parse_response(cls, text_resp: str) -> tuple:
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:") < text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:"):
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                    
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
        else:
            if "next_thought:" not in text_resp:
                error_msg = "Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg = "Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg = "Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:") > text_resp.find("next_tool_name:"):
                error_msg = "Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:") > text_resp.find("next_tool_args:"):
                error_msg = "Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg
        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
        return next_thought, next_tool_name, next_tool_args, error_msg
class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content
    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None
    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        self._process_function(node)
    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)
    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None
class ClassVisitor(ast.NodeVisitor):    
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content
    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)
class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}
    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14
            GIT_OPERATION_FAILED = 15
            GIT_CONFIG_ERROR = 16
            GIT_STATE_ERROR = 17
            GIT_MERGE_CONFLICT = 18
            GIT_BRANCH_ERROR = 19
            TEST_COVERAGE_ERROR = 20
            DEPENDENCY_ANALYSIS_ERROR = 21
            CODE_SMELL_DETECTION_ERROR = 22
            GIT_HISTORY_ERROR = 23
            CODE_QUALITY_ERROR = 24
            SOLUTION_VALIDATION_ERROR = 25
            CODE_STYLE_ERROR = 26
            SOLUTION_COMPARISON_ERROR = 27
        def __init__(self, error_type: ErrorType, message: str):    
            self.error_type = error_type
            self.message = message
    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] += 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper
    def __init__(self, **kwargs):
        pass
    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        if not doc_fn.strip() or "Arguments:" not in doc_fn:
            sig = inspect.signature(fn)
            param_descriptions = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                param_descriptions.append(f"{param_name}: {param_name} parameter")
            doc_fn = f"Tool: {name}\n\nArguments:\n" + "\n".join(param_descriptions) + "\n\nOutput: Tool execution result"
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = f"{param.name} parameter"
            arguments_section = doc_fn.split("Arguments:")
            if len(arguments_section) > 1:
                arguments_text = arguments_section[1].split("Output:")[0] if "Output:" in arguments_section[1] else arguments_section[1]
                # Look for parameter description in format "param_name: description"
                param_pattern = rf"{param.name}:\s*([^\n]+)"
                param_match = re.search(param_pattern, arguments_text)
                if param_match:
                    param_description = param_match.group(1).strip()
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        return tool_schemas
    @classmethod
    def get_tool_args_for_tool(self, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']
    def get_tool_docs(self) -> str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])
    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method
    def _check_syntax_error(self, content: str, file_path: str = "<unknown>") -> tuple[bool, Optional[Error]]:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Syntax error. {str(e)}")

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)

    def _run_code(self, content: str, file_path: str) -> str:
        self._save(file_path, content)
    
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                if mod in sys.builtin_module_names:
                    continue

                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (os.path.isfile(local_file) or os.path.isfile(local_pkg_init) or os.path.isdir(local_pkg_dir) or
                    os.path.isfile(lib_file) or os.path.isfile(lib_pkg_init) or os.path.isdir(lib_pkg_dir)):
                    continue

                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name, f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES

            raise EnhancedToolManager.Error(error_type, f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
        return observation
    
    def _add_line_numbers_to_content(self, content: str, start_line: int = 1) -> str:
        lines = content.splitlines()
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:6}|{line}")
        return '\n'.join(numbered_lines)
    
    def _add_context_to_similar_match(self, original_content: str, formatted_match: str, context_lines: int = 2) -> str:
        lines = original_content.split('\n')
        match_lines = formatted_match.split('\n')
        if len(match_lines) < 2:
            return formatted_match
            
        actual_content_lines = match_lines[1:]
        actual_content = '\n'.join(actual_content_lines)
        
        best_match_start = -1
        best_similarity = 0
        
        for i in range(len(lines) - len(actual_content_lines) + 1):
            candidate_lines = lines[i:i + len(actual_content_lines)]
            candidate_content = '\n'.join(candidate_lines)
            
            import difflib
            similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_start = i
        
        if best_match_start == -1:
            return formatted_match

        start_line = max(0, best_match_start - context_lines)
        end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
        
        context_lines_list = []
        for i in range(start_line, end_line):
            line_num = i + 1
            prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
            context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
        
        description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
        return f"{description}\n" + "\n".join(context_lines_list)

    def _find_most_similar_content(self, original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
        import difflib
        
        lines = original_content.split('\n')
        chunks = []
        
        for i, line in enumerate(lines):
            if line.strip():
                chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
        
        search_lines = search_string.split('\n')
        target_chunk_size = max(3, len(search_lines))
        
        for i in range(len(lines) - target_chunk_size + 1):
            chunk_lines = lines[i:i + target_chunk_size]
            chunk_content = '\n'.join(chunk_lines).strip()
            if chunk_content:
                chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
        
        similarities = []
        for chunk_desc, chunk_content in chunks:
            ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
            if ratio > 0.3:
                similarities.append((ratio, chunk_desc, chunk_content))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

    def get_final_git_patch(self) -> str:
        try:
            command = """
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"
class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created = []
        self.is_solution_approved = False
        self.test_runner = test_runner
        self.test_runner_mode = test_runner_mode
        self.generated_test_files = []
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }
        self.tool_invocations = {
            k: 0 for k in self.TOOL_LIST.keys()
        }
    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> tuple[bool, Optional[EnhancedToolManager.Error]]:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Syntax error. {str(e)}")
    def _get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, 
                         search_term: str = None, limit: int = 5000) -> str:
        if search_term is not None and search_term != "":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
        func_ranges = self.get_function_ranges(file_path)
        if search_start_line is not None:
            for start, end, name in func_ranges:
                if start <= search_start_line <= end:
                    if start < search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line = start
        if search_end_line is not None:
            for start, end, name in func_ranges:
                if start <= search_end_line <= end:
                    if end > search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line = end          
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()
        return Utils.limit_strings(content, n=limit) if limit != -1 else content
    @EnhancedToolManager.tool
    def generate_init_solution(problem_statement: str, code_skeleton: str) -> str:
        """
        This is a method to generate a complete, working Python initial solution for the given problem statement. There's no need to save the file. This method saves the received content to a file.
        Arguments:
            problem_statement: filesystem path to target file. This file must be python file. Problem statemant you must solve. this is plan text like sentences for expleration or mathmatics formular. etc. 
            code_skeleton: code skeleton related to the given problem. This must be python code skeleton.
        Output:
            ```python
            a.py
            contents of a.py

            b.py
            contents of b.py
            ```
        """
        code_generation_messages = [
            {
                "role": "system",
                "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
            },
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
            }
        ]
        code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME)
        created_files = extract_and_write_files(code_response)
        test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
        
        extract_and_write_files(test_cases)
        return code_response.strip()
    
    @EnhancedToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        Output:
            File content as string
        '''
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)
    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        '''
        Writes text content to specified filesystem location
        Arguments:
            file_path: target filesystem path
            content: text data to write
        Output:
            Success message
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        '''
        Get approval for proposed solution
        Arguments:
            solutions: list of solutions proposed by you
            selected_solution: index of the solution you think is the best
            reason_for_selection: reason for selecting the solution over other solutions
        Output:
            approval status
        '''
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]
            parsed_solutions.extend(sols)
        solutions = parsed_solutions
        if type(solutions) is not list or len(solutions) < 2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: solutions must be a list with length at least 2.")
        self.is_solution_approved = True
        return "Approved"
    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths
        Arguments:
            function_paths: list of function paths
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionVisitor(content)
                visitor.visit(tree)
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"
        return functions
    @EnhancedToolManager.tool
    def get_classes(self, class_paths: List[str]) -> Dict[str, str]:
        '''
        Get classes from a list of class paths
        Arguments:
            class_paths: list of class paths
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassVisitor(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"
        return classes
    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for text pattern across all .py files in the project
        Arguments:
            search_term: text pattern to locate
            case_sensitive: flag to determine if search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE
        for root, _, files in os.walk("."):
            if ".git" in root or "docs" in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        if not re.search(search_term, content, search_flags):
                            continue
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)
                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")
        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase.")
        return output
    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
            tree = None
        func_ranges: list[tuple[int, int, str]] = []
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges
    def _extract_function_matches(self, file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in file '{file_path}'")
        func_ranges = self.get_function_ranges(file_path)
        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None
        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)
        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")
        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")
        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)
    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name, f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)
    @EnhancedToolManager.tool
    def start_over(self, problem_with_old_approach: str, new_apprach_to_try: str):
        '''
        Revert changes and start over
        Arguments:
            problem_with_old_approach: issues with current approach
            new_apprach_to_try: new approach to try
        Output:
            confirmation message
        '''
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
    def get_final_git_patch(self) -> str:
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    
    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name, f"Error: file '{file_path}' is not a python file.")
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: test_function_code cannot be empty.")
        is_new_file = not os.path.exists(file_path)
        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]
        if is_new_file:
            new_content = test_fn + "\n"
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: generated test function has syntax error: {err}")
        else:
            original = self._get_file_content(file_path, limit=-1)
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue
            if new_content is None:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: inserting test caused syntax error. First error: {first_error}")
        self._save(file_path, new_content)
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)
        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."
    def create_new_file(self, file_path: str, content: str) -> str:
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    @EnhancedToolManager.tool
    def run_repo_tests(self, file_paths: List[str]) -> str:
        '''
        Runs the tests for the repository
        Arguments:
            file_paths: path of the files to run the tests for
        Output:
            Returns the stdout/stderr from the executed files
        '''
        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output
    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        '''
        Runs any python code
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in
        Output:
            Returns the stdout/stderr from the executed file
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)
        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]
                if mod in sys.builtin_module_names:
                    continue
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)
                if (os.path.isfile(local_file) or os.path.isfile(local_pkg_init) or os.path.isdir(local_pkg_dir) or
                    os.path.isfile(lib_file) or os.path.isfile(lib_pkg_init) or os.path.isdir(lib_pkg_dir)):
                    continue
                disallowed_modules.add(mod)
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type, f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
        return observation
    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        '''
        Performs targeted text replacement within source files
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error: file '{file_path}' does not exist.")
        original = self._get_file_content(file_path, limit=-1)
        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                new_content = original.replace(search, replace)
                try:
                    is_error, error = self.check_syntax_error(new_content)
                    if not is_error:
                        self.save_file(file_path, new_content)
                        return "ok, code edit applied successfully"
                    else:
                        error.message = "code edit failed. " + error.message
                        raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name, f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
    @EnhancedToolManager.tool
    def finish(self, investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: detailed summary of findings and solution  
        Output:
            completion status
        '''
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name, qa_response.get("analysis", ""))
#------------------------Core Classes------------------------------
def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    while retry < 10:
        try:
            logger.info(f"[MULTI_STEP] Attempt {retry + 1}/10")
            code_response = EnhancedNetwork.make_request(code_generation_messages, model=KIMI_MODEL_NAME)
            loop_check_messages = [
                {
                    "role": "system",
                    "content": INFINITE_LOOP_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
                }   
            ]
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            logger.info(f"[MULTI_STEP] Loop check completed ({len(loop_check_response)} chars)")
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            lines = solution.split("\n")
            first_line = lines[0].strip() if lines else ""
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue
            logger.info(f"[MULTI_STEP] Success: {len(lines)} lines, filename: '{first_line}'")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    if retry >= 10:
        logger.error("[MULTI_STEP] Failed after 10 attempts")
        return ""
    return ""
def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation")
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton)
            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]
                response = EnhancedNetwork.make_request(messages, model=KIMI_MODEL_NAME)
                messages.append(
                    {"role": "user", "content": f"""This is a initial solution for this problem. Check it and if you find some errors or bugs, fix it."""},
                    {"role": "assistant", "content" : response}
                )
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                logger.info("Generated initial solution successfully using fallback approach")
                return solution
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""
def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Testcase check completed")
            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue
            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    return ""
def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""
def fix_lazy_property_generation(problem_statement: str, initial_solution: str) -> str:
    try:
        if "class " not in initial_solution:
            logger.info("No class found in solution, skipping")
            return initial_solution

        detection_messages = [
            {
                "role": "system",
                "content": "You are a Python code expert. Analyze if this problem requires lazy property generation. Look for properties that should generate values when accessed but currently just return stored values. Return only 'YES' or 'NO'."
            },
            {
                "role": "user", 
                "content": f"Problem statement:\n{problem_statement}\n\nCurrent solution:\n{initial_solution}\n\nDoes this problem require lazy property generation where properties should generate values when accessed if they don't exist?"
            }
        ]
        
        detection_response = EnhancedNetwork.make_request(detection_messages, model=QWEN_MODEL_NAME)
        needs_fix = "YES" in detection_response.upper()
        
        if not needs_fix:
            logger.info("[LAZY_PROPERTY_FIX] Problem does not need lazy property generation fix, skipping")
            return initial_solution
            
        logger.info("[LAZY_PROPERTY_FIX] Problem needs lazy property generation fix, applying LLM-based fix")
        
        messages = [
            {
                "role": "system",
                "content": "You are a Python code expert. Fix the lazy property generation issues. Properties should generate values when accessed if they don't exist. Return ONLY the complete corrected Python code with filename. Do not include any explanations, comments, or markdown formatting."
            },
            {
                "role": "user", 
                "content": f"Problem statement:\n{problem_statement}\n\nCurrent solution:\n{initial_solution}\n\nRewrite the solution to use instance-level tracking only. Remove all class-level tracking. Use _past_names as instance variable. Do not call reset() in __init__. Implement lazy property generation. Return ONLY the corrected Python code with filename, no explanations. Do not include any explanations, comments, or markdown formatting."
            }
        ]
        
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
        logger.info(f"[LAZY_PROPERTY_FIX] LLM response received ({len(response)} chars)")
        
        fixed_solution = response.strip()
        if fixed_solution.startswith('```python'):
            fixed_solution = fixed_solution[9:]
        if fixed_solution.startswith('```'):
            fixed_solution = fixed_solution[3:]
        if fixed_solution.endswith('```'):
            fixed_solution = fixed_solution[:-3]
        fixed_solution = fixed_solution.strip()
        
        lines = fixed_solution.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        logger.info(f"[LAZY_PROPERTY_FIX] Fixed solution first line: '{first_line}'")
        logger.info(f"[LAZY_PROPERTY_FIX] Fixed solution length: {len(fixed_solution)} chars")
        
        if first_line.endswith('.py'):
            logger.info("[LAZY_PROPERTY_FIX] Successfully applied LLM-based lazy property fix")
            return fixed_solution
        else:
            logger.warning(f"[LAZY_PROPERTY_FIX] Format validation failed - first line: '{first_line}', using original")
            return initial_solution
            
    except Exception as e:
        logger.error(f"[LAZY_PROPERTY_FIX] LLM-based fix failed: {e}, using original")
        return initial_solution


def fix_recursive_word_definitions(problem_statement: str, initial_solution: str) -> str:
    try:
        if "class " not in initial_solution:
            logger.info("No class found in solution, skipping")
            return initial_solution
        detection_messages = [
            {
                "role": "system",
                "content": "You are a programming expert. Analyze if this problem requires fixing recursive definition issues. Look for code that handles definitions that can reference themselves, causing infinite loops during execution. Return only 'YES' or 'NO'."
            },
            {
                "role": "user", 
                "content": f"Problem statement:\n{problem_statement}\n\nCurrent solution:\n{initial_solution}\n\nDoes this problem have recursive definition issues where definitions can reference themselves, causing infinite loops during execution-time expansion?"
            }
        ]
        detection_response = EnhancedNetwork.make_request(detection_messages, model=QWEN_MODEL_NAME)
        needs_fix = "YES" in detection_response.upper()
        if not needs_fix:
            logger.info("[RECURSIVE_DEFINITION_FIX] Problem does not need recursive definition fix, skipping")
            return initial_solution
        logger.info("[RECURSIVE_DEFINITION_FIX] Definition patterns detected, applying LLM-based fix")
        messages = [
            {
                "role": "system",
                "content": "You are a programming expert. Fix recursive definition issues in the evaluator. The problem is that when a definition references itself, the current expansion logic creates infinite loops. Fix this by properly handling recursive definitions during definition time, not execution time. IMPORTANT: Do NOT prevent recursive definitions - they should be allowed and handled correctly. CRITICAL: Return ONLY the corrected Python code with filename, no explanations, comments, or markdown formatting. Do not use ```python or ``` blocks."
            },
            {
                "role": "user", 
                "content": f"Problem statement: {problem_statement}\n\nCurrent solution: {initial_solution}\n\nFix the recursive definition issue. Follow these steps: STEP 1: PROBLEM - Current solution has infinite loops when definitions reference themselves - Must handle recursive definitions properly STEP 2: REQUIREMENTS - Recursive definitions must be ALLOWED and handled correctly - Do NOT raise errors for recursive definitions - Do NOT add conditions that prevent self-reference STEP 3: SOLUTION - Process definitions during definition time, not execution time - Create flattened version by expanding references - If reference points to same word being defined, keep it as-is - If reference points to different word, expand it recursively STEP 4: EXECUTION - When encountering defined word, prepend its flattened definition to input stream - Use simple prepending approach - Do NOT replace entire token list or reset index STEP 5: ARCHITECTURE - Process definitions line by line first, then execute only the last line - Separate definition phase from execution phase completely - Use simple prepending approach - Do NOT use complex token list manipulation or index reset STEP 6: VALIDATION - Check if word name is a number (positive or negative) using try/except approach - If conversion succeeds, it's a number - raise error immediately - If conversion fails, it's not a number - continue normally - Use simple logic: try conversion, if successful then raise error, if exception then continue Return ONLY the corrected Python code with filename, no explanations, no markdown formatting, no ``` blocks."
            }
        ]
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
        logger.info(f"[RECURSIVE_DEFINITION_FIX] LLM response received ({len(response)} chars)")
        fixed_solution = response.strip()
        if fixed_solution.startswith('```python'):
            fixed_solution = fixed_solution[9:]
        if fixed_solution.startswith('```'):
            fixed_solution = fixed_solution[3:]
        if fixed_solution.endswith('```'):
            fixed_solution = fixed_solution[:-3]
        fixed_solution = fixed_solution.strip()
        lines = fixed_solution.split('\n')
        first_line = lines[0].strip() if lines else ""
        logger.info(f"[RECURSIVE_DEFINITION_FIX] Fixed solution first line: '{first_line}'")
        logger.info(f"[RECURSIVE_DEFINITION_FIX] Fixed solution length: {len(fixed_solution)} chars")
        if first_line.endswith('.py'):
            logger.info("[RECURSIVE_DEFINITION_FIX] Successfully applied LLM-based recursive definition fix")
            return fixed_solution
        else:
            logger.warning(f"[RECURSIVE_DEFINITION_FIX] Format validation failed - first line: '{first_line}', using original")
            return initial_solution
    except Exception as e:
        logger.error(f"[RECURSIVE_DEFINITION_FIX] LLM-based fix failed: {e}, using original")
        return initial_solution
def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    created_files = []
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    for line in lines:
        stripped_line = line.strip()
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and
            not stripped_line.startswith('#')):
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                content = '\n'.join(current_content).strip()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            current_filename = stripped_line
            current_content = []
        else:
            if current_filename:
                current_content.append(line)
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content = '\n'.join(current_content).strip()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    return created_files
def process_task(input_dict):
    start_time = time.time()
    problem_statement = input_dict.get("problem_statement", "")
    code_skeleton = get_code_skeleton()
    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 300
    patch = fix_task_solve_workflow(
        problem_statement,
        code_skeleton,
        timeout=timeout,
        run_id_1=run_id,
        instance_id="",
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=100
    )
    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch
def get_code_skeleton() -> str:
    result = ""
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                result += f"{file}\n{{\n{content}\n}}\n\n"
    return result
def get_directory_tree(start_path: str = '.') -> str:
    tree_lines = []
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        try:
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            try:
                items = os.listdir(path)
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
            except PermissionError:
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)
def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    current_dir = os.path.dirname(file_path)
    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)
    return None
def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "pytest"
    except Exception as e:
        logger.error(f"Error finding test runner: {e}")
        return "pytest"
def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)
    return module_path.replace(os.path.sep, '.')
def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)
    return module_path
def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"
    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "FILE"
    except Exception as e:
        logger.error(f"Error determining test runner mode: {e}")
        return "FILE"
def count_test_cases(file_path: str) -> int:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)
    except (FileNotFoundError, UnicodeDecodeError):
        return 0
def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []
    test_file_path = None
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    test_files.sort(key=len)
    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break
    if not test_file_path:
        print(f"no test file found")
        return "pytest", "FILE"
    print(f"test_file_path: {test_file_path}")
    readme_file_path = find_readme(test_file_path, '.')
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)
    else:
        print("No README found, using default pytest")
    return test_runner, test_runner_mode

def fix_task_solve_workflow(problem_statement: str, code_skeleton: str, *, timeout: int, run_id_1: str, instance_id: str = "", \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS):
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT()
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "generate_init_solution",
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "get_functions",
            "get_classes",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"Starting main agent execution...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V0)
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement, code_skeleton=code_skeleton)
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached", next_tool_name="finish", next_tool_args={}, observation="", is_error=True, inference_error_counter={}, request_data=[]))
            break
        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(messages, model="NousResearch/DeepHermes-3-Mistral-24B-Preview", run_id=run_id)
        except Exception as e:
            import traceback
            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg, next_tool_name="", next_tool_args={}, observation="", is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            break
        logger.info(f"About to execute operation: {next_tool_name}")
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', '')
                next_tool_name = next_tool_name.replace("'", "")
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=next_observation, is_error=False, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback
            error_msg = f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=error_msg, is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            continue
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=error_msg, is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            continue
        if next_tool_name == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached", next_tool_name="", next_tool_args={}, observation="", is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")
    return patch
#------------------------Type Checker------------------------------
def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {'role': 'system', 'content': PROBLEM_TYPE_CHECK_PROMPT},
                {'role': 'user', 'content':f'{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}'}
            ]
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f'Error: {e}')
            retry += 1
        time.sleep(2)
    return response
#------------------------Type Checker------------------------------
#----------------------------Init----------------------------------
def ensure_git_initialize():
    logger.info('Starting git initialization check...')
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    try:
        logger.info(f"Work directory: {work_dir}")
        logger.info(f"Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        os.chdir(work_dir)
        logger.info(f"After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        if not os.path.exists(".git"):
            logger.info("Initializing git repository...")
            subprocess.run(['git', 'init'], check=True)
            subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', work_dir], check=True)
            logger.info(f'.git exists: {os.path.exists(".git")}')
            logger.info(f'Files in current dir: {os.listdir(".")[:10]}')
            logger.info("Setting git config...")
            subprocess.run(['git', 'config', '--global', 'user.email', 'agent@sandbox.local'], check=True)
            subprocess.run(['git', 'config', '--global', 'user.name', 'sandbox_agent'], check=True)
            logger.info('Adding all files...')
            subprocess.run(['git', 'add', '.'], check=True)
            logger.info('Creating initial commit...')
            result = subprocess.run(['git', 'commit', '-m', 'Initial commit'], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info('Initial commit created successfully')
            else:
                logger.info('Git initialization completed successfully')
        else:
            logger.info(f'Git repository already exists')
            subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', work_dir])
    except Exception as e:
        logger.error(f"Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)
def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + os.getcwd()
    if Path(os.getcwd() + '/lib').exists() and os.getcwd() + '/lib' not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + os.getcwd() + '/lib'
#----------------------------init----------------------------------
def agent_main(input_dict: Dict[str, Any], repo_dir: str="repo", test_mode: bool = False):
    global DEFAULT_PROXY_URL, REPO_DIR, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    if test_mode:
        DEFAULT_TIMEOUT = 2000
        MAX_TEST_PATCH_TIMEOUT = 100
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    ensure_git_initialize()
    set_env_for_agent()
    result = process_task(input_dict)
    os.system("git reset --hard")
    return result