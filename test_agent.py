#!/usr/bin/env python3
"""
Simple test agent for local validation testing.
This agent just returns a basic solution for testing purposes.
"""

import json
import os
import sys
from typing import Dict, Any

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False) -> str:
    """
    Simple test agent that returns a basic solution.
    
    Args:
        input_dict: Dictionary containing problem information
        repo_dir: Directory where the problem files are located
        test_mode: Whether running in test mode
    
    Returns:
        String containing the solution diff
    """
    
    problem_statement = input_dict.get("problem_statement", "")
    test_cases = input_dict.get("test_cases", [])
    
    print(f"[TEST_AGENT] Problem statement: {problem_statement[:100]}...")
    print(f"[TEST_AGENT] Test cases: {len(test_cases)}")
    
    # For testing purposes, return a simple diff that adds a basic function
    # This is just a placeholder - in reality, the agent would analyze the problem
    # and generate appropriate code
    
    solution_diff = """--- a/main.py
+++ b/main.py
@@ -1,3 +1,10 @@
 def main():
-    pass
+    # Basic implementation for testing
+    return "Hello, World!"
+
+def solve():
+    # Placeholder solution
+    return True
 """
    
    print(f"[TEST_AGENT] Generated solution diff with {len(solution_diff)} characters")
    
    return solution_diff

if __name__ == "__main__":
    # For testing
    test_input = {
        "problem_statement": "Implement a simple function that returns 'Hello, World!'",
        "test_cases": [{"input": "main()", "expected_output": "Hello, World!"}]
    }
    
    result = agent_main(test_input)
    print(f"Result: {result}")




