from __future__ import annotations
import ast
import json
import os
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
from uuid import uuid4
import urllib.request as _urlreq
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import math


PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

PROBLEM_LANGUAGE_PYTHON = "python"

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1800"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

# Embedding configuration
MAX_EMBED_TOKENS = 128000  # Reduced from 512000 for faster processing
MAX_EMBED_CHARS = MAX_EMBED_TOKENS * 4  # reserve but we'll split by tokens
EMBED_MODEL_NAME = "agentica-org/DeepCoder-14B-Preview"
USE_FUNCTION_CHUNKS = os.getenv("EMBED_WHOLE_FILES", "0") != "1"
RUN_ID = os.getenv("RUN_ID", "")
REPO_DIR = ""
DEBUG_MODE = True

GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS=[GLM_MODEL_NAME, QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME, KIMI_MODEL_NAME]
MAX_FIX_TASK_STEPS = 400

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
'''
You are the problem type checker that will categories problem type into:

1. CREATE: If the problem statement is about creating a new functionality from scratch.
2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

Only respond with the "FIX" or "CREATE".
'''
)

ONESHOT_SYSTEM_PROMPT = """
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
"""

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
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

Response Examples:
```python
a.py
{content}

b.py
{content}
```
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

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
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
MAIN_COT_REASONING_PROMPT=textwrap.dedent("""
# Role: Expert Coding Assistant 🚀

## Context
You are operating at the root of a Python repository. Your mission is to systematically analyze and fix coding issues using comprehensive reasoning and available tools.
As a first step, find the relevant files in the repo to work on.

## Core Workflow (Chain of Thought)

### Phase 1: Deep Analysis & Understanding
1. **Problem Comprehension**: 
   - Analyze the problem statement thoroughly
   - Compare expected vs actual outputs from test cases
   - Identify the core issue and success criteria
   - Note: Solutions must align with most relevant test case expectations

2. **Codebase Exploration**:
   - Use `search_in_all_files_content` for initial discovery across repository
   - Follow with `search_in_specified_file_v2` for focused file analysis
   - Use `get_file_content` with line filtering for precise examination
   - Map code structure using `get_functions` and `get_classes`

3. **Root Cause Investigation**:
   - Examine file contents to localize problematic code
   - Trace execution flow and data transformations
   - Identify exact location and nature of the bug
   - Use `run_code` to reproduce issues if needed

### Phase 2: Multi-Solution Design
1. **Solution Generation**:
   - Brainstorm at least 2 meaningfully different approaches
   - Each solution must completely address the root cause
   - Consider implementation complexity and risk
   - Ensure solutions produce output closest to test case expectations

2. **Impact Assessment**:
   - Analyze how each solution affects existing functionality
   - Ensure backward compatibility unless explicitly waived
   - Identify edge cases and boundary conditions
   - Plan testing strategy for each solution

### Phase 3: Approval & Implementation
1. **Solution Presentation**:
   - Use `get_approval_for_solution` with detailed solution comparisons
   - Select best solution based on test case alignment
   - Provide clear reasoning for selection

2. **Systematic Implementation**:
   - After approval, use `apply_code_edit` for precise changes
   - Apply fixes to ALL identified locations in codebase
   - Maintain consistency across multiple files
   - Use iterative search-apply-verify cycles

### Phase 4: Validation & Quality Assurance
1. **Testing Strategy**:
   - Use `run_repo_tests` to validate existing functionality
   - Use `generate_test_function` for focused bug reproduction tests
   - Use `run_code` for custom validation scenarios
   - Ensure all relevant test cases pass

2. **Final Verification**:
   - Re-check all modified files for consistency
   - Confirm no unintended side effects
   - Validate solution meets all requirements
   - Use `finish` with comprehensive investigation summary

## Critical Principles

### Multi-File Awareness
- **Never stop at first match** - use iterative searches until all instances found
- **Consistent changes** - ensure uniform modifications across codebase
- **Cross-file impact** - consider dependencies between files

### Solution Quality Standards
- **Minimal changes**: Fix issue with least necessary modifications
- **Backward compatibility**: Maintain unless explicitly required otherwise
- **Test alignment**: Solutions must match expected output in relevant test cases
- **Elegance**: Prefer clean, maintainable approaches

### Tool Compliance
- **Always get approval** before using `apply_code_edit`
- **Never create test files** with `save_file`
- **Use `generate_test_function`** for test creation needs
- **Validate syntax** - all modifications are syntax-checked

## Solution Presentation Format

When using `get_approval_for_solution`, structure solutions as:

**Solution 1: [Descriptive Name]**
- **Approach**: High-level strategy and implementation details
- **Files Modified**: Complete list with specific changes
- **Edge Cases Handled**: Specific scenarios addressed
- **Test Alignment**: How it matches expected outputs
- **Pros**: Advantages and elegance
- **Cons**: Potential drawbacks

**Solution 2: [Descriptive Name]**
- [Same detailed structure]

**Selection Reasoning**: Clear explanation why chosen solution best matches test case expectations

## Quality Checklist
Before seeking approval, verify:
- [ ] At least 2 meaningfully different solutions developed
- [ ] All relevant files identified through comprehensive searching
- [ ] Solutions align with test case expected outputs
- [ ] Backward compatibility maintained
- [ ] Edge cases properly handled
- [ ] Changes are minimal and focused
- [ ] Implementation plan is precise and testable

## Execution Mindset
- **Be systematic**: Follow CoT framework rigorously
- **Be thorough**: Leave no code path unexamined
- **Be precise**: Use exact text matching for edits
- **Be collaborative**: Present clear options for approval
- **Be quality-focused**: Prioritize solutions that best match test expectations
""")

TOOL_PROMPT = textwrap.dedent("""
# TOOL KNOWLEDGE BASE

## Available Tools & Their Purposes:

### File Operations
**get_file_content(file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str**
- Retrieves file contents with optional filtering
- Can extract specific line ranges or filter by search term
- Use: When you need targeted examination of file contents
- Note: Only works with Python files

**save_file(file_path: str, content: str)->str**
- Writes text content to specified filesystem location
- Rejects edits with syntax errors
- Use: For making code changes after approval
- Important: Do not use to create test files

### Code Analysis & Search
**search_in_all_files_content(search_term: str, case_sensitive: bool = False) -> str**
- Searches for text patterns across all .py files (excluding test files)
- Use: Initial discovery of relevant code across entire repository
- Output: File paths and line numbers where pattern was found

**search_in_specified_file_v2(file_path: str, search_term: str)->str**
- Locates text patterns within a specific Python file
- Use: Deep analysis of identified files
- Output: Matching locations with line numbers

**get_functions(function_paths: List[str])->Dict[str, str]**
- Extracts function definitions using full paths (e.g., "file.py::class::function")
- Use: Understanding specific function implementations
- Output: Dictionary of function paths to function bodies

**get_classes(class_paths: List[str])->Dict[str, str]**
- Extracts class definitions using full paths (e.g., "file.py::class")
- Use: Understanding class structures and hierarchies
- Output: Dictionary of class paths to class bodies

### Execution & Testing
**run_code(content:str,file_path:str)->str**
- Executes Python code by saving to file and running
- Use: Testing changes, reproducing issues, validating fixes
- Important: Only use for execution, not for permanent test creation
- Note: May fail due to missing dependencies

**run_repo_tests(file_paths:List[str])->str**
- Runs tests for specific repository files
- Use: Validating that changes don't break existing tests
- Output: stdout/stderr from test execution

**generate_test_function(file_path: str, test_function_code: str, position: str = "append") -> str**
- Creates or appends test functions to test files
- Use: Creating focused tests for bug reproduction or validation
- Important: Generated tests are excluded from final patch
- Position options: "append", "top", "after_imports", "before_main", "auto"

### Solution Management
**get_approval_for_solution(solutions:list[str],selected_solution:int,reason_for_selection:str)->str**
- Presents solutions to user for approval
- Requires at least 2 meaningfully different solutions
- Selection criteria: Closest to expected output in most relevant test case
- Use: Before implementing any major changes
- Output: approval/not_approved

**apply_code_edit(file_path:str, search:str, replace:str)->str**
- Performs targeted text replacement in source files
- Rejects edits with syntax errors
- Use: After receiving approval for solution
- Important: Only exact text patterns can be replaced

### Workflow Control
**start_over(problem_with_old_approach:str,new_apprach_to_try:str)**
- Reverts all changes and restarts analysis
- Use: When current approach is fundamentally flawed
- Requires: Explanation of previous issues and new strategy

**finish(investigation_summary: str)**
- Signals completion of workflow
- Requires detailed summary in format:
  Problem: <problem_statement>
  Investigation: <investigation_summary>  
  Solution: <your_solution>

## Tool Usage Patterns:

### Discovery Flow:
1. `search_in_all_files_content()` → Broad repository search
2. `search_in_specified_file_v2()` → File-specific pattern matching
3. `get_file_content()` → Targeted content extraction
4. `get_functions()`/`get_classes()` → Structural analysis

### Solution Flow:
1. `get_approval_for_solution()` → Present 2+ solutions for approval
2. `apply_code_edit()` → Make precise changes after approval
3. `run_code()`/`run_repo_tests()` → Validate changes
4. `finish()` → Complete workflow

### Testing Flow:
1. `generate_test_function()` → Create focused tests
2. `run_repo_tests()` → Execute relevant tests
3. `run_code()` → Custom validation code

## Critical Constraints:
- **Approval Required**: Use `get_approval_for_solution` before `apply_code_edit`
- **No Test Creation**: Don't use `save_file` for test files
- **Python Only**: Most tools work only with Python files
- **Syntax Checking**: Modification tools validate syntax
""")
DYNAMIC_TOOL_SELECTION_PROMPT=textwrap.dedent(
"""
# Dynamic Tool Selection Guide

## Based on Task Phase:

### Exploration & Discovery:
**When starting investigation:**
- `search_in_all_files_content()` - Find all occurrences across repository
- `search_in_specified_file_v2()` - Deep dive into specific files
- `get_file_content()` - Examine files with optional line filtering

**When understanding code structure:**
- `get_functions()` - Analyze specific function implementations
- `get_classes()` - Examine class hierarchies and methods

### Solution Development:
**When designing fixes:**
- `run_code()` - Test hypotheses and reproduce issues
- `generate_test_function()` - Create focused tests for validation

**When ready to implement:**
- `get_approval_for_solution()` - REQUIRED before any code changes
- `apply_code_edit()` - Make precise text replacements after approval

### Validation & Completion:
**When testing changes:**
- `run_repo_tests()` - Validate existing test suites
- `run_code()` - Execute custom validation scenarios
- `generate_test_function()` - Add protective tests

**When finished:**
- `finish()` - Submit comprehensive summary

## Critical Tool Combinations:

### Complete Bug Fix Workflow:
1. `search_in_all_files_content()` → Find all relevant code
2. `get_file_content()` → Examine specific implementations  
3. `get_approval_for_solution()` → Get approval for 2+ solutions
4. `apply_code_edit()` → Implement approved changes
5. `run_repo_tests()` → Validate no regressions
6. `finish()` → Complete with summary

### Complex Multi-file Issues:
1. Iterative searches until all occurrences found
2. Structural analysis with `get_functions`/`get_classes`
3. Comprehensive solution presentation for approval
4. Consistent application across all files
5. Full test suite validation

## Important Reminders:
- ✅ Always use `get_approval_for_solution` before `apply_code_edit`
- ✅ Use `generate_test_function` for test creation, never `save_file`
- ✅ Solutions must align with test case expected outputs
- ✅ Perform comprehensive multi-file searches
- ✅ Provide detailed investigation summaries in `finish`
"""
)
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

## Follow these steps to fix the issue:
1. As a first step, find the relevant files in the repo to work on.
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
# Now let's start. Here is the problem statement:
{problem_statement}
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


STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")
REQUIREMENT_ANALYSIS_PROMPT = textwrap.dedent("""
# REQUIREMENT ANALYSIS DOCUMENT

## INSTRUCTION
Generate a comprehensive, well-structured requirement analysis document by systematically analyzing the provided problem statement and code skeleton. The output must be a complete, professional document that can be directly used for software development planning.

## INPUT DATA

### PROBLEM STATEMENT:
{problem_statement}

### CODE SKELETON:
{code_skeleton}

## DOCUMENT STRUCTURE REQUIREMENTS

The output must follow this exact structure with all sections completed:

### 1. EXECUTIVE SUMMARY
- Brief overview of the problem domain
- Primary objectives of the solution
- Key technical challenges identified
- Overall scope boundaries

### 2. FUNCTIONAL REQUIREMENTS SPECIFICATION

#### 2.1 Core Functional Requirements
- List each functional requirement with unique identifier (FR-001, FR-002, etc.)
- Each requirement must be testable, specific, and measurable
- Include priority classification (P0: Critical, P1: Important, P2: Optional)

#### 2.2 Input/Output Specifications
- Detailed input requirements and formats
- Expected output specifications and formats
- Data validation requirements
- Error conditions and handling expectations

#### 2.3 Business Rules
- Specific business logic requirements
- Decision points and conditional logic
- Calculation rules and formulas
- Validation rules and constraints

### 3. TECHNICAL CONSTRAINTS DOCUMENTATION

#### 3.1 Interface Constraints
- Complete list of function signatures that must be preserved
- Parameter types and return value specifications
- Class interfaces and method contracts
- Any decorators or annotations that must remain

#### 3.2 Architectural Constraints
- Code organization patterns that must be followed
- Module structure and import dependencies
- Inheritance hierarchies and class relationships
- Existing design patterns to maintain

#### 3.3 Environmental Constraints
- Programming language and version requirements
- Library and dependency restrictions
- Platform and runtime environment constraints
- Performance and resource limitations

### 4. ACCEPTANCE CRITERIA DEFINITION

#### 4.1 Success Scenarios
- Primary success paths with expected behaviors
- Normal operation test cases
- Standard input/output validation cases

#### 4.2 Edge Case Scenarios
- Boundary condition test cases
- Extreme input value handling
- Resource exhaustion scenarios
- Concurrency and timing issues (if applicable)

#### 4.3 Failure Scenarios
- Error condition test cases
- Invalid input handling requirements
- Exception handling and recovery expectations
- Graceful degradation requirements

### 5. QUALITY ATTRIBUTES SPECIFICATION

#### 5.1 Performance Requirements
- Response time expectations
- Throughput requirements
- Resource utilization limits
- Scalability considerations

#### 5.2 Reliability Requirements
- Availability expectations
- Fault tolerance requirements
- Recovery time objectives
- Data integrity requirements

#### 5.3 Maintainability Requirements
- Code documentation standards
- Testing coverage expectations
- Modularity and extensibility requirements
- Code style and formatting standards

#### 5.4 Security Requirements
- Input sanitization requirements
- Data protection needs
- Access control considerations
- Vulnerability prevention measures

### 6. RISK ASSESSMENT AND MITIGATION

#### 6.1 Technical Risks
- Complexity-related risks
- Integration challenges
- Performance risks
- Technology limitation risks

#### 6.2 Requirement Risks
- Ambiguity in requirements
- Conflicting requirements
- Missing requirement identification
- Scope creep potential

#### 6.3 Implementation Risks
- Algorithmic complexity risks
- Resource constraint risks
- Timeline and scheduling risks
- Dependency management risks

#### 6.4 Risk Mitigation Strategies
- Specific mitigation approaches for each identified risk
- Contingency planning
- Validation and verification strategies
- Monitoring and adjustment mechanisms

### 7. ASSUMPTIONS AND DEPENDENCIES

#### 7.1 Explicit Assumptions
- Documented assumptions about unclear requirements
- Environmental assumptions
- User behavior assumptions
- Data quality assumptions

#### 7.2 Technical Dependencies
- Library and framework dependencies
- External service dependencies
- Platform and infrastructure dependencies
- Integration point dependencies

#### 7.3 External Dependencies
- Third-party service requirements
- API dependencies and contracts
- Data source dependencies
- Compliance and regulatory dependencies

### 8. VALIDATION AND VERIFICATION PLAN

#### 8.1 Testing Strategy
- Unit testing approach
- Integration testing strategy
- System testing methodology
- Acceptance testing criteria

#### 8.2 Quality Assurance Measures
- Code review requirements
- Static analysis expectations
- Performance testing approach
- Security testing requirements

#### 8.3 Success Metrics
- Quantitative success indicators
- Qualitative success measures
- Performance benchmarks
- User acceptance criteria

## DOCUMENT QUALITY REQUIREMENTS

- Each section must be complete with no placeholder text
- Requirements must be specific, measurable, achievable, relevant, and testable (SMART)
- Technical specifications must be precise and unambiguous
- All code skeleton constraints must be explicitly documented
- Risk assessment must be comprehensive and realistic
- The document must be actionable for development teams

## FORMATTING STANDARDS

- Use clear, professional language
- Maintain consistent numbering and heading hierarchy
- Include specific examples where appropriate for clarity
- Ensure all technical terms are properly defined
- Document must be self-contained and comprehensive

## OUTPUT

Generate the complete requirement analysis document following the specified structure. Ensure all sections are thoroughly completed with detailed, actionable content that directly addresses the provided problem statement and code skeleton.
""")
ARCHITECTURE_ANALYSIS_PROMPT = textwrap.dedent("""
# SOFTWARE ARCHITECTURE ANALYST

## PRIMARY OBJECTIVE
Analyze the technical constraints and design an implementation architecture that satisfies all functional requirements while strictly adhering to the provided code skeleton's interfaces and patterns.

## INPUT DATA
**Problem Statement:**
{problem_statement}

**Code Skeleton Structure:**
{code_skeleton}

**Validated Requirements:**
{requirement_document}

## MANDATORY ANALYSIS FRAMEWORK

### 1. IMMUTABLE ARCHITECTURAL CONSTRAINTS IDENTIFICATION
Extract and document every technical constraint imposed by the code skeleton that cannot be altered under any circumstances.

### 2. IMPLEMENTATION BOUNDARY DEFINITION  
Precisely delineate what code must be written versus what code must be preserved exactly as provided.

### 3. DATA FLOW AND STATE MANAGEMENT DESIGN
Design the complete data transformation pipeline from input to output, including all intermediate states and transformations.

### 4. ERROR HANDLING AND VALIDATION ARCHITECTURE
Define comprehensive error detection, handling, and recovery mechanisms for all possible failure modes.

### 5. QUALITY ATTRIBUTE SATISFACTION PLAN
Specify how each quality requirement will be technically achieved in the implementation.

## REQUIRED OUTPUT SECTIONS

### ARCHITECTURAL CONSTRAINTS ANALYSIS:

**Preserved Function Signatures:**
List every function signature from the code skeleton that must be maintained exactly, including:
- Function name
- Parameter names and types
- Return type
- Any decorators or annotations

**Preserved Class Structures:**
Document every class definition that must be maintained, including:
- Class name and inheritance hierarchy
- Method signatures within the class
- Class-level variables and properties
- Any metaclass or special method definitions

**Fixed Import Dependencies:**
List every import statement that must remain unchanged, including:
- Module imports
- Specific function/class imports
- Import aliases if present

**Immutable Architectural Patterns:**
Identify all patterns in the existing code that must be replicated, including:
- Data access patterns
- Error handling approaches
- Code organization conventions
- Naming conventions and style

### SOLUTION ARCHITECTURE DESIGN:

**Core Implementation Strategy:**
Describe the complete technical approach for fulfilling each functional requirement within the identified constraints.

**Algorithm Selection Rationale:**
For each algorithmic requirement, specify:
- The exact algorithm to implement
- Time and space complexity analysis
- Justification for algorithm selection
- Any optimizations required

**Data Structure Specifications:**
Define every data structure to be used, including:
- Primary data containers and their types
- Auxiliary data structures for intermediate processing
- Memory allocation patterns
- Data mutation policies

**Component Responsibility Matrix:**
Assign each functional requirement to specific components/functions, specifying:
- Which component handles which requirement
- Input/output contracts for each component
- Dependencies between components
- Interface specifications

### INTEGRATION ARCHITECTURE:

**Complete Data Flow Specification:**
Map the entire data transformation pipeline from initial input to final output, detailing:
- Each processing step
- Data format at each stage
- Transformation logic between stages
- Error propagation paths

**State Management Design:**
Define how application state will be managed, including:
- Variable scoping strategy
- State mutation control
- Shared state access patterns
- State persistence requirements

**Component Interaction Protocol:**
Specify how different components will communicate, including:
- Function call sequences
- Data passing mechanisms
- Error handling coordination
- Resource sharing protocols

### QUALITY ATTRIBUTES IMPLEMENTATION:

**Performance Assurance Measures:**
Detail specific techniques to meet performance requirements:
- Algorithmic complexity guarantees
- Memory usage optimization strategies
- Processing efficiency improvements
- Bottleneck prevention mechanisms

**Maintainability Enforcement:**
Specify practices to ensure code maintainability:
- Code organization structure
- Documentation standards
- Modularity boundaries
- Extension point definitions

**Testability Infrastructure:**
Design the testing support architecture:
- Unit test isolation boundaries
- Mocking interfaces
- Test data generation strategies
- Validation checkpoints

**Robustness Implementation:**
Define fault tolerance mechanisms:
- Input validation layers
- Exception handling hierarchy
- Recovery procedures
- Graceful degradation paths

### RISK ANALYSIS AND MITIGATION:

**Technical Risk Assessment:**
Identify every potential technical challenge, including:
- Algorithmic complexity risks
- Integration compatibility issues
- Performance boundary conditions
- Edge case handling difficulties

**Implementation Risk Mitigation:**
Specify concrete mitigation strategies for each identified risk:
- Alternative implementation approaches
- Fallback mechanisms
- Validation checkpoints
- Rollback procedures

**Complexity Management:**
Define approaches to manage implementation complexity:
- Problem decomposition strategy
- Incremental implementation phases
- Integration testing approach
- Debugging support infrastructure

### IMPLEMENTATION ROADMAP:

**Phase 1: Foundation Implementation**
List every foundational component that must be implemented first, in dependency order.

**Phase 2: Core Functionality Implementation**  
List every core feature component to implement after foundations are stable.

**Phase 3: Edge Case and Error Handling**
List all edge case handling and error management components.

**Phase 4: Optimization and Validation**
List all performance optimization and final validation tasks.

## SPECIFIC TECHNICAL REQUIREMENTS

### FOR CREATE TASKS:
- All implementations must fit precisely within the provided skeleton structure
- No existing function signatures may be modified
- No additional parameters may be added to existing functions
- All stubs and TODOs must be implemented according to their implied contracts
- Any helper functions must follow the established code patterns

### FOR FIX TASKS:
- All fixes must be minimal and surgical
- No working functionality may be disrupted
- All existing interfaces must be preserved
- Only specifically identified issues may be addressed
- Regression prevention must be designed into every change

## VALIDATION CRITERIA

The architecture must be validated against these criteria:
1. Every functional requirement has a clear implementation path
2. All code skeleton constraints are strictly respected
3. Data flow is complete and well-defined
4. Error handling covers all identified failure modes
5. Performance requirements are technically achievable
6. Implementation risks are identified and mitigated

## OUTPUT INSTRUCTIONS

Generate the architecture analysis using exactly the section headers and structure specified above. Each section must contain complete technical details without omission. Do not include any examples, analogies, or placeholder text. Provide specific, actionable technical specifications that can be directly implemented by a development team.

Begin the analysis now.
""")
FORMAT_PROMPT_V0=textwrap.dedent("""
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
run_id=None
  
class EnhancedCOT:
    class Action:
            
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=30):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep
        
    def is_valid_tool_call(self, next_tool_name: str|list, next_tool_args: dict|list) -> bool:
        if len(self.thoughts) == 0:
            return True
            
        last_tool_name = self.thoughts[-1].next_tool_name
        last_tool_args = self.thoughts[-1].next_tool_args
        
        # Exact match check - definitely reject
        if next_tool_name == last_tool_name and next_tool_args == last_tool_args:
            return False
            
        return True

    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        # if not self.is_valid_tool_call(action.next_tool_name, action.next_tool_args):
        #     return False
        self.thoughts.append(action)
        return True
        
    def is_thought_repeated(self)->bool:
        # Check if the last thought is the same as the previous thought.
        # If there are less than 2 thoughts, skip (return False).
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i<len(self.thoughts)-self.latest_observations_to_keep:
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}\n"
                )
                # Compute observation summary length safely for str/list/None
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str=( f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n")
                
            else:
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
            messages.append({"role":"assistant","content":assistant_str})
            messages.append({"role":"user","content":user_str})
        return messages
    
    def export_to_csv(self,file_path:str="./xray.csv"):
        with open(file_path, "w") as f:
            writer=csv.writer(f)
            writer.writerow(["next_thought","next_tool_name","next_tool_args","observation","is_error","raw_response","total_attempts","is_deleted"])
            if len(self.thoughts)>0:
                for thought in self.thoughts:
                    writer.writerow([thought.next_thought,thought.next_tool_name,thought.next_tool_args,thought.observation,thought.is_error,thought.raw_response,thought.total_attempts,str(thought.inference_error_counter),str(thought.request_data),len(str(thought.request_data)),thought.is_deleted])
                
                
    def get_tokens_used(self):
        # quick, safe heuristic assuming ~0.75 tokens/word
        msgs = self.to_str()
        text = "\n".join(m["content"] for m in msgs)
        word_count = len(text.split())
        return int(word_count * 0.75)

class Utils:
    @classmethod
    def get_available_modules(cls) -> set[str]:
        """Return the set of top-level module names that can be imported in the
        *current* Python environment.

        The result includes:
        • built-in/stdlib module names (`sys.builtin_module_names`)
        • every top-level name discoverable on `sys.path` via `pkgutil.iter_modules()`
        This is useful when we need to check whether a piece of code depends on a
        package that is *not* present in the environment.
        """
        import sys, pkgutil

        available: set[str] = set(sys.builtin_module_names)
        for module_info in pkgutil.iter_modules():
            # Only keep the top-level package name (before the first dot)
            top_level = module_info.name.split(".")[0]
            available.add(top_level)
        return available

    @classmethod
    def message_to_str(cls,messages:list[dict]): 
        final_str=""
        for message in messages:
            role=message["role"]
            content=message["content"]
            final_str+=f"{role}: {content}\n"
        return final_str
    
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                # if fixed_json == ""
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError(f"Invalid JSON: {json_string}")
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])

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

class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0)->str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        print("[REQUEST] run_id:", run_id)

        # Cache miss - make the actual request
        request_data = {
                "run_id": run_id if run_id else "1",
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        
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
            is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
            if is_oai_interface:
                raw_text=response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    raw_text=response_json.strip("\n").strip()
                else:
                    raw_text=response_json
            if type(raw_text) is not dict:
                raw_text=raw_text.lstrip()
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
    def _request_next_action_with_retry(cls, messages: dict, 
                            model: str,
                            max_retries: int = 5, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text=cls.make_request(messages,model=AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)], temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
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
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(1.2*delay, 1.5*delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''

        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name,required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = "1",return_json:bool=False, temperature:float=0.0) -> dict:
        """Prod inference with caching"""
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

        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            # remove all leading and trailing quotes in tool_name
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                # Enforce arrays per new contract: if single string/object, wrap as arrays
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
                error_msg=f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            Utils.log_to_failed_messages(text_resp)
            return None,None,None,error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args,error_msg

class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
            GIT_OPERATION_FAILED=15
            GIT_CONFIG_ERROR=16
            GIT_STATE_ERROR=17
            GIT_MERGE_CONFLICT=18
            GIT_BRANCH_ERROR=19
            TEST_COVERAGE_ERROR = 20
            DEPENDENCY_ANALYSIS_ERROR = 21
            CODE_SMELL_DETECTION_ERROR = 22
            GIT_HISTORY_ERROR = 23
            CODE_QUALITY_ERROR = 24
            SOLUTION_VALIDATION_ERROR = 25
            CODE_STYLE_ERROR = 26
            SOLUTION_COMPARISON_ERROR = 27
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        # Preserve original function metadata
       
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True

        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
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
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method
    
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            # self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)

    def _run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
    
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES

            raise EnhancedToolManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"

        return observation


    def _add_line_numbers_to_content(self, content: str, start_line: int = 1) -> str:
        """Helper method to add line numbers to content."""
        lines = content.splitlines()
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:6}|{line}")
        return '\n'.join(numbered_lines)
    
    def _add_context_to_similar_match(self, original_content: str, formatted_match: str, context_lines: int = 2) -> str:
        """Add context lines around a similar match for better understanding."""
        lines = original_content.split('\n')
        
        # Extract the actual content from the formatted match (remove the description part)
        match_lines = formatted_match.split('\n')
        if len(match_lines) < 2:
            return formatted_match
            
        # Skip the description line (e.g., "Lines 45-47: ..." or "Line 23: ...")
        actual_content_lines = match_lines[1:]
        actual_content = '\n'.join(actual_content_lines)
        
        # Find where this content appears in the original file
        best_match_start = -1
        best_similarity = 0
        
        # Search for the best matching position in the original content
        for i in range(len(lines) - len(actual_content_lines) + 1):
            candidate_lines = lines[i:i + len(actual_content_lines)]
            candidate_content = '\n'.join(candidate_lines)
            
            import difflib
            similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_start = i
        
        if best_match_start == -1:
            return formatted_match  # Fallback to original if can't find position
        
        # Calculate context boundaries
        start_line = max(0, best_match_start - context_lines)
        end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
        
        # Build the context with line numbers
        context_lines_list = []
        for i in range(start_line, end_line):
            line_num = i + 1
            prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
            context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
        
        # Extract original description
        description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
        
        return f"{description}\n" + "\n".join(context_lines_list)

    def _find_most_similar_content(self, original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
        """Find the most similar content chunks to the search string."""
        import difflib
        
        # Split content into meaningful chunks
        lines = original_content.split('\n')
        
        # Try different chunk sizes to find the best match
        chunks = []
        
        # Individual lines
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
        
        # Multi-line chunks (3-5 lines) for better context
        search_lines = search_string.split('\n')
        target_chunk_size = max(3, len(search_lines))
        
        for i in range(len(lines) - target_chunk_size + 1):
            chunk_lines = lines[i:i + target_chunk_size]
            chunk_content = '\n'.join(chunk_lines).strip()
            if chunk_content:
                chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
        
        # Calculate similarity scores
        similarities = []
        for chunk_desc, chunk_content in chunks:
            ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
            if ratio > 0.3:  # Only include reasonably similar content
                similarities.append((ratio, chunk_desc, chunk_content))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            # Update to include cfg, txt, and toml files along with py files
            # Check whether ignore_files is a property of this clas
            command = f"""
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
            
            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"
class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]

        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations={
          k:0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
            
        # check if start and end line are not between a function..
        func_ranges=self.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line=end
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit!=-1  else content
    
    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
            
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        
        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions)<2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: solutions must be a list with length at least 2.")

        self.is_solution_approved = True
        return "Approved"
          
    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
 
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
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
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
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
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        If more context is needed (e.g., surrounding functions, classes, etc.), follow up with get_classes or get_functions.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
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

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
            tree = None  # Fallback if file cannot be parsed.

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{file_path}'")

        func_ranges=self.get_function_ranges(file_path)

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
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)

    # @tool
    def search_recurive_in_all_files_in_directory(self, directory_path: str, search_term: str)->str:
        '''
        Locates text patterns recursively within all files in a specific directory
        Arguments:
            directory_path: target directory for pattern matching
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: directory '{directory_path}' does not exist.")
        output=subprocess.run(["bash", "-c", f"grep -rn --include='*.py' {directory_path} -e '{search_term}'"], capture_output=True)
        output=output.stdout.decode("utf-8")
        output=Utils.limit_strings(output, n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{directory_path}'")
        return output
    
    @EnhancedToolManager.tool
    def start_over(self,problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            # Log stderr separately so it never pollutes the patch.
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
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")

        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,"Error: test_function_code cannot be empty.")

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
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
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: generated test function has syntax error: {err}")
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
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
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")

            # Try each candidate until one passes syntax check
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
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: inserting test caused syntax error. First error: {first_error}")

        self._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    def create_new_file(self,file_path:str, content:str)->str:
        '''
        Generates new file with specified content at target location. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            file_path: destination path for new file
            content: text content for file creation
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)

    @EnhancedToolManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
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
    def run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
       

        return observation
    
    @EnhancedToolManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
            
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: file '{file_path}' does not exist.")
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=self.check_syntax_error(new_content)
                        if not is_error:
                            self.save_file(file_path, new_content)
                                
                            return "ok, code edit applied successfully"
                        else:
                            error.message="code edit failed. "+error.message
                            raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
    
    @EnhancedToolManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

# ---------------------------------------------------------------------------
# Chunk representation and collector ---------------------------------------
# ---------------------------------------------------------------------------

class Chunk(NamedTuple):
    file: str
    start_line: int
    end_line: int
    text: str


def _guess_tokens(text: str) -> int:
    """Rough token count estimate: ~0.75 tokens per word."""
    return int(len(text.split()) * 0.75)


def _collect_code_chunks(root: str = ".") -> List[Chunk]:
    """Collect function-level code chunks from Python files."""
    chunks: List[Chunk] = []
    
    for root, _, files in os.walk(root):
        # Skip hidden directories and common non-source dirs
        if any(part.startswith('.') for part in Path(root).parts):
            continue
        if root.endswith(('__pycache__', 'node_modules', '.git')):
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError):
                continue
                
            try:
                tree = ast.parse(content, filename=file_path)
            except SyntaxError:
                continue
                
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = getattr(node, 'lineno', 1)
                    end_line = getattr(node, 'end_lineno', start_line)
                    
                    if end_line is None:
                        # Fallback: count lines in the node
                        lines = content.split('\n')
                        if start_line <= len(lines):
                            node_text = '\n'.join(lines[start_line-1:])
                            end_line = start_line + node_text.count('\n')
                        else:
                            end_line = start_line
                    
                    # Extract the actual source lines
                    lines = content.split('\n')
                    if start_line <= len(lines) and end_line <= len(lines):
                        chunk_text = '\n'.join(lines[start_line-1:end_line])
                        
                        # Skip if chunk is too large
                        if len(chunk_text) > MAX_EMBED_CHARS:
                            continue
                            
                        chunks.append(Chunk(
                            file=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            text=chunk_text
                        ))
                        
            # Also add the entire file as a chunk if it's reasonably sized
            if len(content) <= MAX_EMBED_CHARS and content.strip():
                chunks.append(Chunk(
                    file=file_path,
                    start_line=1,
                    end_line=content.count('\n') + 1,
                    text=content
                ))
    
    return chunks


def _token_windows(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> List[str]:
    """Split text into token-limited windows."""
    words = text.split()
    windows = []
    
    for i in range(0, len(words), max_tokens):
        window_words = words[i:i + max_tokens]
        windows.append(' '.join(window_words))
    
    return windows


def _lang_tag(path: str) -> str:
    """Return language tag for syntax highlighting."""
    ext = os.path.splitext(path)[1].lower()
    lang_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'matlab',
        '.sh': 'bash',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.md': 'markdown',
        '.txt': 'text',
    }
    return lang_map.get(ext, 'text')


def _collect_repo_texts(root: str = ".") -> Dict[str, str]:
    """Collect all text content from repository files."""
    texts: Dict[str, str] = {}
    
    for root, _, files in os.walk(root):
        # Skip hidden directories and common non-source dirs
        if any(part.startswith('.') for part in Path(root).parts):
            continue
        if root.endswith(('__pycache__', 'node_modules', '.git')):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip binary files and very large files
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz')):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Skip if file is too large
                if len(content) > MAX_EMBED_CHARS:
                    continue
                    
                texts[file_path] = content
                
            except (UnicodeDecodeError, PermissionError):
                continue

    return texts


# ---------------------------------------------------------------------------
# Embedding cache and helper -------------------------------------------------
# ---------------------------------------------------------------------------

_EMBED_CACHE: Dict[str, List[float]] = {}
ZERO_VEC: List[float] = [0.0] * 1024  # embedding for empty input


def _remote_embed(text: str, proxy_url: str, run_id: str) -> List[float]:
    """Return embedding vector for *text* via the proxy /api/embedding endpoint.

    Caches results in-memory to avoid duplicate HTTP calls.
    """
    print(f"[Agent] Embedding request: {len(text)} chars")
    # Short-circuit empty or whitespace-only inputs.
    if not text.strip():
        return _EMBED_CACHE.setdefault("", [0.0] * 1024)

    # Retry–shrink loop to satisfy 512-token limit
    attempt_text = text
    for _ in range(2):  # original + 1 retry after halving
        tokens = attempt_text.split()
        if len(tokens) > MAX_EMBED_TOKENS:
            attempt_text = " ".join(tokens[:MAX_EMBED_TOKENS])

        url = f"{proxy_url.rstrip('/')}/api/embedding"
        req = _urlreq.Request(
            url,
            data=json.dumps({"input": attempt_text, "run_id": run_id}, ensure_ascii=False).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with _urlreq.urlopen(req, timeout=60) as resp:  # Reduced timeout from 300 to 60
                data_raw = resp.read()
                data = json.loads(data_raw.decode())
                print(f"[Agent] Embedding response: {len(data)} bytes")

                if isinstance(data, list):
                    vec = data[0] if (len(data) == 1 and isinstance(data[0], list)) else data
                    _EMBED_CACHE[text] = vec
                    return vec
                if isinstance(data, dict) and "embedding" in data:
                    vec = data["embedding"]
                    _EMBED_CACHE[text] = vec
                    return vec

                # If we received a validation error about tokens, halve and retry
                if isinstance(data, dict) and data.get("error_type") == "Validation":
                    attempt_text = " ".join(tokens[: len(tokens) // 2])
                    continue

                return ZERO_VEC
        except Exception as e:
            print(f"[Agent] Embedding error: {e}")
            return ZERO_VEC

    return ZERO_VEC


def _cosine(u: List[float], v: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))

    if nu == 0 or nv == 0:
        return 0.0

    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


# ---------------------------------------------------------------------------
# One-shot retrieval using remote embeddings --------------------------------
# ---------------------------------------------------------------------------

def run_oneshot(
    problem_text: str,
    *,
    proxy_url: str,
    model_name: str,
    run_id: str,
    top_k: int = 30,
) -> str:
    """Build repository summary and send a single LLM call.

    Embeddings are fetched from the internal `/api/embedding` proxy endpoint, so no model weights or internet access are required inside the sandbox.
    """
    print(f"[Agent] One-shot mode: {len(problem_text)} chars problem")
    if USE_FUNCTION_CHUNKS:
        code_chunks = _collect_code_chunks()
        if not code_chunks:
            raise RuntimeError("repository appears empty – nothing to embed")
        chunk_texts = [c.text for c in code_chunks]
    else:
        repo_texts = _collect_repo_texts()
        if not repo_texts:
            raise RuntimeError("repository appears empty – nothing to embed")
        code_chunks = [Chunk(file=fp, start_line=1, end_line=text.count("\n") + 1, text=text) for fp, text in repo_texts.items()]
        chunk_texts = [c.text for c in code_chunks]

    # --------------------------------------------------------------------
    # Cheap TF-IDF pre-filter to limit expensive embedding calls ----------
    # --------------------------------------------------------------------

    PRE_FILTER_TOP = int(os.getenv("PREFILTER_TOP", "50"))  # Reduced from 200 for faster processing

    if len(chunk_texts) > PRE_FILTER_TOP:
        # Simple TF-IDF scoring
        problem_words = set(problem_text.lower().split())
        chunk_scores = []
        
        for chunk_text in chunk_texts:
            chunk_words = set(chunk_text.lower().split())
            common_words = problem_words.intersection(chunk_words)
            score = len(common_words) / max(len(problem_words), 1)
            chunk_scores.append(score)
        
        # Keep top chunks by TF-IDF score
        sorted_indices = sorted(range(len(chunk_scores)), key=lambda i: -chunk_scores[i])
        top_indices = sorted_indices[:PRE_FILTER_TOP]
        
        code_chunks = [code_chunks[i] for i in top_indices]
        chunk_texts = [chunk_texts[i] for i in top_indices]

    # --------------------------------------------------------------------
    # Embed problem text and all chunks ----------------------------------
    # --------------------------------------------------------------------

    query_vec = _remote_embed(problem_text, proxy_url, run_id)
    chunk_vecs: List[List[float]] = [None] * len(chunk_texts)  # type: ignore

    MAX_WORKERS = min(8, int(os.getenv("EMBED_CONCURRENCY", "8")))  # Increased concurrency

    print(f"[Agent] Embedding {len(chunk_texts)} chunks with {MAX_WORKERS} workers")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_to_idx = {pool.submit(safe_remote_embed, txt, proxy_url, run_id): idx for idx, txt in enumerate(chunk_texts)}

        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                chunk_vecs[idx] = fut.result()
            except Exception as exc:
                # Log and keep zero vector; retrieval will simply rank it low.
                print(f"[agent] embedding error (chunk {idx}): {exc}")
                chunk_vecs[idx] = ZERO_VEC

    sims = [ _cosine(vec, query_vec) for vec in chunk_vecs ]

    # --------------------------------------------------------------------
    # Light path-based bonus if filename is mentioned in the problem text --
    # --------------------------------------------------------------------
    prob_lower = problem_text.lower()
    for idx, ch in enumerate(code_chunks):
        base = os.path.basename(ch.file).lower()
        if base in prob_lower or base.split(".")[0] in prob_lower:
            sims[idx] += 0.2

    sorted_idx = sorted(range(len(sims)), key=lambda i: -sims[i])

    TARGET_TOKENS = 6_000  # Reduced from 12_000 for faster processing
    token_budget = int(TARGET_TOKENS * 0.85)
    token_total = 0
    top_idx: list[int] = []
    for idx in sorted_idx:
        tok = _guess_tokens(chunk_texts[idx])
        if token_total + tok > token_budget:
            break
        token_total += tok
        top_idx.append(idx)

    # Fallback to at most top_k if budget yields too many
    if len(top_idx) > top_k:
        top_idx = top_idx[:top_k]

    summary_parts: list[str] = []
    for idx in top_idx:
        ch = code_chunks[idx]
        body = ch.text[:5000]
        tag = _lang_tag(ch.file)
        header = f"### {ch.file}:L{ch.start_line}-{ch.end_line}"
        summary_parts.append(f"{header}\n```{tag}\n{body}\n```")

    repo_summary = "\n\n".join(summary_parts)

    # --------------------------------------------------------------------
    # Build initial conversation messages.
    # --------------------------------------------------------------------

    print(f"[Agent] Repository summary: {len(repo_summary)} chars, {len(top_idx)} chunks, {token_total} tokens")
    print(f"[Agent] repo_summary:\n{repo_summary}")
    messages = [
        {"role": "system", "content": ONESHOT_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
        {"role": "user", "content": "Repository summary (top files):\n\n" + repo_summary},
    ]

    proxy_resp = inference_for_oneshot_embedding(messages, proxy_url, run_id, model_name)

    print(f"[agent] Proxy response received: {proxy_resp}")
    code_resp = proxy_resp.get("code_response", "")

    patch_text = None
    # if code_resp and (code_resp.startswith("diff") or code_resp.startswith("--- ")):
    patch_text = code_resp
    # Sanitize diff to strip markdown fences and chatter
    patch_text = _sanitize_patch(patch_text)
    print(f"[agent] Sanitized patch : {patch_text}")

    ok, dry_out = _dry_run_patch(patch_text)
    if ok:
        result = _apply_patch(patch_text)
        print(f"[agent] Patch applied. patch:\n{patch_text}\n{result}")
        return patch_text
    else:
        print(f"[agent] Patch failed to apply. patch:\n'{patch_text}'\n{dry_out}")
        # Patch failed – append feedback and ask for correction.
        messages.append({"role": "assistant", "content": code_resp[:200]})
        messages.append({"role": "user", "content": "Patch failed to apply. Please reply with a corrected unified diff only."})
        raise RuntimeError(f"[agent] Patch could not be applied through oneshot embedding.")


def safe_remote_embed(text, proxy_url, run_id, max_retries=3):  # Reduced retries from 5 to 3
    for attempt in range(max_retries):
        try:
            return _remote_embed(text, proxy_url, run_id)
        except Exception as e:
            sleep_time = 2  # Reduced sleep time from 10 to 2 seconds
            print(f"Rate limited, retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    return ZERO_VEC


def _apply_patch(patch: str) -> str:
    """Apply a git patch and return the result."""
    try:
        # Write patch to temporary file
        with open(".temp_patch", "w") as f:
            f.write(patch)
        
        # Apply the patch
        result = subprocess.run(
            ["git", "apply", ".temp_patch"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Generate final patch
            final_patch = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return final_patch.stdout
        else:
            print(f"Patch application failed: {result.stderr}")
            return ""
            
    except Exception as e:
        print(f"Error applying patch: {e}")
        return ""
    finally:
        # Clean up temporary file
        try:
            os.remove(".temp_patch")
        except:
            pass


def inference_for_oneshot_embedding(messages: List[Dict[str, Any]], proxy_url: str, run_id: str, model: str = None) -> dict:
    """Send inference request to the proxy and return the response."""
    # Build request data
    request_data = {
        "run_id": run_id,
        "messages": messages,
        "temperature": 0
    }
    
    if model:
        request_data["model"] = model

    # Send HTTP request
    url = f"{proxy_url.rstrip('/')}/api/inference"
    request_bytes = json.dumps(request_data, ensure_ascii=False).encode('utf-8')
    
    print(f"[agent] Making inference request to {url}")
    
    try:
        req = _urlreq.Request(url, data=request_bytes, method="POST")
        req.add_header("Content-Type", "application/json")
        
        with _urlreq.urlopen(req, timeout=120) as resp:  # Reduced timeout from 600 to 120
            response_chunks = []
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                response_chunks.append(chunk)
            response_body = b"".join(response_chunks)
            # response_body = resp.read()
            print(f"[agent] HTTP {resp.status} from {url} ({len(response_body)} bytes)")
            
            response_txt = response_body.decode("utf-8")
            response_json = json.loads(response_txt)
            
            # The proxy may return a plain string instead of a JSON object with
            # separate text / code fields.  In that case we wrap it in the
            # expected shape so downstream logic can stay unchanged.
           
            if isinstance(response_json, str):
                if response_json.find("code_response") != -1:
                    response_json = json.loads(response_json)
                else:
                    raw_text: str = response_json
                    # Attempt to separate a unified diff from the explanatory text.
                    diff_start = None
                    if raw_text.startswith("diff") or raw_text.startswith("--- "):
                        diff_start = 0
                    else:
                        # Look for the first occurrence of a diff header inside the text.
                        for marker in ("\ndiff --git", "\n--- "):
                            idx = raw_text.find(marker)
                            if idx != -1:
                                diff_start = idx + 1  # skip the leading newline
                                break

                    code_resp = raw_text
                    if diff_start is not None:
                        code_resp = raw_text[diff_start:].lstrip()

                    response_json = {"code_response": code_resp}

            return response_json

            
    except Exception as e:
        print(f"[agent] Inference request failed: {e}")
        raise RuntimeError(f"Inference request failed: {e}")


def _sanitize_patch(patch: str) -> str:
    """Return *patch* stripped of markdown/code-fence chatter.

    The LLM sometimes wraps the unified diff in Markdown fences or appends
    explanatory text (e.g. `DISCUSSION`, `EOF`).  This helper keeps only the
    lines that are valid in a unified diff so downstream tools (`patch`,
    `git apply`) succeed and the resulting patch file is clean.
    """
    # Stop at the first closing code fence (```), if present.
    stop_idx = patch.find("\n```")
    if stop_idx != -1:
        patch = patch[:stop_idx]

    allowed_prefixes = (
        "diff --git",
        "index ",
        "--- ",
        "+++ ",
        "@@",
        "new file mode",
        "deleted file mode",
        "similarity index",
        "rename from",
        "rename to",
        "Binary files",
        "\\ No newline",
    )

    cleaned_lines: list[str] = []
    for line in patch.splitlines():
        # Early exit markers – anything after is junk.
        if line.strip() in {"DISCUSSION", "EOF"}:
            break
        if line.startswith(allowed_prefixes):
            cleaned_lines.append(line)
            continue
        # Hunk content lines start with space, plus or minus.
        if line.startswith(("+", "-", " ")):
            cleaned_lines.append(line)
            continue
        # Ignore everything else (markdown, commentary, empty lines outside hunks).

    # Post-process header lines: ensure 'a/' and 'b/' prefixes so a default
    # `patch -p1` invocation resolves correctly when run from the repository
    # root.  We intentionally *do not* touch paths pointing to /dev/null.
    for idx, ln in enumerate(cleaned_lines):
        if ln.startswith("--- ") and not ln.startswith("--- a/") and not ln.startswith("--- /dev/null"):
            cleaned_lines[idx] = "--- a/" + ln[4:]
        elif ln.startswith("+++ ") and not ln.startswith("+++ b/") and not ln.startswith("+++ /dev/null"):
            cleaned_lines[idx] = "+++ b/" + ln[4:]

    return "\n".join(cleaned_lines) + "\n"


def _dry_run_patch(patch: str) -> tuple[bool, str]:
    """Validate patch without applying it."""
    try:
        # Sanitize the patch first
        sanitized_patch = _sanitize_patch(patch)
        
        if not sanitized_patch.strip():
            return False, "Empty patch"
        
        # Write to temporary file
        with open(".temp_patch", "w") as f:
            f.write(sanitized_patch)
        
        # Try to apply with --check flag
        result = subprocess.run(
            ["git", "apply", "--check", ".temp_patch"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.returncode == 0, result.stderr
        
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up
        try:
            os.remove(".temp_patch")
        except:
            pass


def process_task_with_oneshot_embedding(input_dict: Dict[str, Any]):
    """Process task using oneshot embedding approach."""
    problem_text = input_dict.get("problem_statement")
    result = run_oneshot(
        problem_text,
        proxy_url=DEFAULT_PROXY_URL,
        model_name=EMBED_MODEL_NAME,
        run_id=RUN_ID,
    )

    return result


def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, REPO_DIR, DEFAULT_TIMEOUT, MAX_TEST_PATCH_TIMEOUT, RUN_ID, run_id
    RUN_ID = os.getenv("RUN_ID", "")
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    REPO_DIR = repo_dir
    if test_mode:
        DEFAULT_TIMEOUT = 1156
        MAX_TEST_PATCH_TIMEOUT = 400
    
    sys.path.insert(0, repo_dir)

    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    ensure_git_initialized()

    set_env_for_agent()

    # Check problem type first
    problem_type = check_problem_type(input_dict.get("problem_statement"))
    
    if problem_type == PROBLEM_TYPE_FIX:
        # Use embedding-based approach for FIX tasks
        try:
            result = process_fix_task(input_dict)#process_task_with_oneshot_embedding(input_dict)
        except Exception as e:
            print(f"[agent] Error occurred while processing with oneshot embedding: {e}")
            print(f"[agent] Falling back to traditional FIX workflow...")
            result = process_fix_task(input_dict)
    else:
        # Use traditional approach for CREATE tasks
        result = process_create_task(input_dict)

    os.system("git reset --hard")

    return result

def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)

            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1
        
        time.sleep(2)

    return response

def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    # Pattern to match ```text...``` blocks
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        
        return f'```text\n{processed_content}\n```'
    
    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str, requirement: str, architecture: str) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nRequirement document:\n{requirement}\n\nArchitecture:\n{architecture}\n\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    while retry < 10:
        try:
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
            code_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)

            solution = code_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def generate_initial_solution(problem_statement: str, code_skeleton: str, requirement: str, architecture: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation")
            
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton, requirement, architecture)
            
            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nRequirement document:\n{requirement}\n\nArchitecture:\n{architecture}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                
                # Clean up the response
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
            
            # Step 5: Infinite Loop Check and Validation
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

            # Clean up the final response (use loop check response as it's the final validated version)
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
                
                # Fallback to original single-step approach if multi-step fails
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
                
                # Clean up the response
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

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files
def fix_lazy_property_generation(problem_statement: str, initial_solution: str) -> str:
    """Fix lazy property generation issues using LLM analysis for specific problems only."""
    try:
        # Check if solution contains a class definition
        if "class " not in initial_solution:
            logger.info("No class found in solution, skipping")
            return initial_solution

        # Use LLM to determine if this problem needs lazy property generation fix
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
                "content": f"Problem statement:\n{problem_statement}\n\nCurrent solution:\n{initial_solution}\n\nRewrite the solution to use instance-level tracking only. Remove all class-level tracking. Use _past_names as instance variable. Do not call reset() in __init__. Implement lazy property generation. Return ONLY the corrected Python code with filename, no explanations."
            }
        ]
        
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
        logger.info(f"[LAZY_PROPERTY_FIX] LLM response received ({len(response)} chars)")
        
        # Clean up response
        fixed_solution = response.strip()
        if fixed_solution.startswith('```python'):
            fixed_solution = fixed_solution[9:]
        if fixed_solution.startswith('```'):
            fixed_solution = fixed_solution[3:]
        if fixed_solution.endswith('```'):
            fixed_solution = fixed_solution[:-3]
        fixed_solution = fixed_solution.strip()
        
        # Validate format
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
    """Fix recursive definition issues in evaluator using LLM analysis."""
    try:
        # Check if solution contains a class definition
        if "class " not in initial_solution:
            logger.info("No class found in solution, skipping")
            return initial_solution

        # Use LLM to determine if this problem needs recursive definition fix
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
        
        # Clean up response
        fixed_solution = response.strip()
        if fixed_solution.startswith('```python'):
            fixed_solution = fixed_solution[9:]
        if fixed_solution.startswith('```'):
            fixed_solution = fixed_solution[3:]
        if fixed_solution.endswith('```'):
            fixed_solution = fixed_solution[:-3]
        fixed_solution = fixed_solution.strip()
        
        # Validate format
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

def enhance_solution_with_constants(problem_statement: str, initial_solution: str) -> str:
    """Use LLM to add any missing constants that might be needed for imports."""
    try:
        logger.info("[ENHANCE] Starting solution enhancement")
        
        # Check if solution contains a class definition
        if "class " not in initial_solution:
            logger.info("[ENHANCE] No class found in solution, skipping constant enhancement")
            return initial_solution
        
        logger.info("[ENHANCE] Class detected, proceeding with constant enhancement")
        
        messages = [
            {
                "role": "system",
                "content": "You are a code reviewer. Add constants at the TOP of the Python file (before any classes) for repeated string literals. These constants must be at module level, not inside classes. IMPORTANT: All imports must come BEFORE any constants that depend on them. Return ONLY the complete enhanced Python code with filename."
            },
            {
                "role": "user", 
                "content": f"Current solution:\n{initial_solution}\n\nAdd constants at the very top of the file (after filename, before any classes) for repeated string literals. Note: Use NONE = '' for empty string values(Not allowed to use EMPTY, NULL, UNDEFINED, etc.). Example:\nfilename.py\nNONE = ''\nCONST1 = 'value'\nCONST2 = 'value'\n\nclass SomeClass:\n    # existing code"
            }
        ]
        
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
        logger.info(f"[ENHANCE] LLM response received ({len(response)} chars)")
        
        # Clean up response
        enhanced_solution = response.strip()
        if enhanced_solution.startswith('```python'):
            enhanced_solution = enhanced_solution[9:]
        if enhanced_solution.startswith('```'):
            enhanced_solution = enhanced_solution[3:]
        if enhanced_solution.endswith('```'):
            enhanced_solution = enhanced_solution[:-3]
        enhanced_solution = enhanced_solution.strip()
        
        # Validate format
        lines = enhanced_solution.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        logger.info(f"[ENHANCE] Enhanced solution first line: '{first_line}'")
        logger.info(f"[ENHANCE] Enhanced solution length: {len(enhanced_solution)} chars")
        
        if first_line.endswith('.py'):
            logger.info("[ENHANCE] Successfully enhanced solution with constants")
            return enhanced_solution
        else:
            logger.warning(f"[ENHANCE] Format validation failed - first line: '{first_line}', using original")
            return initial_solution
            
    except Exception as e:
        logger.error(f"[ENHANCE] Enhancement failed: {e}, using original")
        return initial_solution
    
def process_create_task(input_dict):
    start_time = time.time()
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)
    code_skeleton = get_code_skeleton()
    def requirement_analysis(problem_statement: str, code_skeleton: str) -> str:
        prompt = REQUIREMENT_ANALYSIS_PROMPT.format(
        problem_statement=problem_statement,
        code_skeleton=code_skeleton
            )
        messages = [
            {"role": "system", "content": "You are a expert software requirements analyst. Generate comprehensive, well-structured requirement documents that are immediately actionable for development teams."},
            {"role": "user", "content": prompt}
            ]
        response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        return response if response else ""
    requirement = requirement_analysis(problem_statement, code_skeleton)
    logger.info(f"[REQUIREMENT]\n{requirement}")
    def architecture_analyzer(problem_statement: str, code_skeleton: str, requirement: str) -> str:
        prompt = ARCHITECTURE_ANALYSIS_PROMPT.format(
            problem_statement=problem_statement,
            code_skeleton=code_skeleton,
            requirement_document=requirement
            )
        messages = [
        {
            "role": "system", 
            "content": "You are a principal software architect. Provide complete technical specifications without omissions or examples."
        },
        {
            "role": "user", 
            "content": prompt
        }]
        response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME,)
        return response if response else ''
    architecture = architecture_analyzer(problem_statement=problem_statement, code_skeleton=code_skeleton, requirement=requirement)
    logger.info(f"[ARCHITECTURE]\n{architecture}")
    
    code_skeleton = get_code_skeleton()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, requirement, architecture)
    initial_solution = enhance_solution_with_constants(problem_statement, initial_solution)
    initial_solution = fix_lazy_property_generation(problem_statement, initial_solution)
    initial_solution = fix_recursive_word_definitions(problem_statement, initial_solution)
    
    created_files = extract_and_write_files(initial_solution)

    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    extract_and_write_files(test_cases)

    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 60
    
    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        instance_id="",
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=100
    )

    if patch is None:
        print("Patch is None")
        extract_and_write_files(initial_solution)
    
    
    extract_and_write_files(patch)
    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch

def get_code_skeleton() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []
    
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            
            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            
            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                
                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                
                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                
                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
            except PermissionError:
                # Handle directories we can't read
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
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
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
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
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)
    
    except (FileNotFoundError, UnicodeDecodeError):
        return 0

def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
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

def process_fix_task(input_dict: Dict[str, Any]):
    """Main entry point for FIX task processing using embedding-based approach."""
    global RUN_ID, REPO_DIR
    
    # Set up environment
    RUN_ID = os.getenv("RUN_ID", "")
    repo_dir = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_dir.split('/')[-1]
    repod_path = repo_dir[:-len(repod_dir)-1]
    
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    problem_statement=input_dict.get("problem_statement", "")
    REPO_DIR = os.getcwd()
    set_env_for_agent()
    code_skeleton = get_code_skeleton()
    def requirement_analysis(proplem_statement: str, code_skeleton: str) -> str:
        prompt = REQUIREMENT_ANALYSIS_PROMPT.format(
        proplem_statement=proplem_statement,
        code_skeleton=code_skeleton
            )
        messages = [
            {"role": "system", "content": "You are a expert software requirements analyst. Generate comprehensive, well-structured requirement documents that are immediately actionable for development teams."},
            {"role": "user", "content": prompt}
            ]
        response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        return response if response else ""
    requirement = requirement_analysis(problem_statement, code_skeleton)
    logger.info(f"[REQUIREMENT]\n{requirement}")
    def architecture_analyzer(problem_statement: str, code_skeleton: str, requirement: str) -> str:
        prompt = ARCHITECTURE_ANALYSIS_PROMPT.format(
            problem_statement=problem_statement,
            code_skeleton=code_skeleton,
            requirement_document=requirement,
            task_type="FIX"
            )
        messages = [
        {
            "role": "system", 
            "content": "You are a principal software architect. Provide complete technical specifications without omissions or examples."
        },
        {
            "role": "user", 
            "content": prompt
        }]
        response = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME,)
        return response if response else ''
    architecture = architecture_analyzer(problem_statement=problem_statement, code_skeleton=code_skeleton, requirement=requirement)
    logger.info(f"[ARCHITECTURE]\n{architecture}")
    
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        logger.info(f"About to execute embedding-based FIX workflow...")
        
        # Use embedding-based approach for FIX tasks
        result = process_task_with_oneshot_embedding(input_dict)
        
        logger.info(f"Embedding-based workflow completed, result length: {len(result) if result else 0}")
        
        # Reset git state
        os.system("git reset --hard")
        
        return result
        
    except Exception as e:
        import traceback
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"[CRITICAL] Exception in embedding-based FIX task processing: {error_info}")
        
        # Fallback to traditional approach if embedding fails
        logger.info("Falling back to traditional FIX workflow...")
        try:
            result = fix_task_solve_workflow(
                input_dict.get("problem_statement", ""),
                timeout=DEFAULT_TIMEOUT,
                run_id_1=RUN_ID,
                instance_id="",
                test_runner="pytest",
                test_runner_mode="FILE",
                requirement=requirement,
                architecture=architecture
            )
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return ""

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id=run_id_1
    cot=EnhancedCOT()
    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
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
    system_prompt = MAIN_COT_REASONING_PROMPT
    tool_prompt = TOOL_PROMPT
    dynamictoolselection = DYNAMIC_TOOL_SELECTION_PROMPT
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": tool_prompt},
                {"role": "system", "content": dynamictoolselection},
                {"role": "system", "content": STOP_INSTRUCTION},
                {"role": "user", "content": instance_prompt},
                
            ]
        messages.extend(cot.to_str())
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME, run_id=run_id)
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts),inference_error_counter=error_counter,request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        # This happens if we exit the loop without breaking (reached MAX_STEPS)
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS: # This is create task case and failed with smaller fix steps so try to use original solution supposing generated testcases are wrong
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch