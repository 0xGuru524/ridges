#!/usr/bin/env python3
"""
Local Validation Script for Ridges Agents

This script runs local validation on a sample set of problems to compare
different agent versions. It selects the first 5 problems from each dataset
(Polyglot and SWE-bench Verified) and runs the agent against them to get
the exact same scoring as the validator.

Usage:
    python local_validation.py --agent-path miner/agent.py --output results.json
    python local_validation.py --agent-path miner/agent_v2.py --output results_v2.json
    python local_validation.py --compare results.json results_v2.json
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import socket
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

class LocalValidator:
    def __init__(self, agent_path: str, output_path: str, timeout: int = 300):
        self.agent_path = Path(agent_path)
        self.output_path = Path(output_path)
        self.timeout = timeout
        self.results = {}
        
        # Validate agent file exists
        if not self.agent_path.exists():
            raise FileNotFoundError(f"Agent file not found: {self.agent_path}")
        
        # Get local IP for inference gateway
        self.local_ip = self._get_local_ip()
        self.gateway_url = f"http://{self.local_ip}:8000"
        
    def _get_local_ip(self):
        """Get the local IP address for the inference gateway."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            console.print(f"⚠️ Could not determine local IP: {e}", style="yellow")
            return "127.0.0.1"  # Use localhost as fallback for local validation
    
    def _start_inference_gateway(self):
        """Start the inference gateway if not already running."""
        try:
            # Check if gateway is already running
            response = requests.get(f"{self.gateway_url}/docs", timeout=5)
            if response.status_code == 200:
                console.print(f"✅ Inference gateway already running at {self.gateway_url}", style="green")
                return None
        except:
            pass
        
        console.print(f"🚀 Starting inference gateway on {self.local_ip}:8000...", style="yellow")
        
        # Start the gateway
        process = subprocess.Popen(
            ["debugpy","--listen","localhost:5678", "main.py"],
            cwd="inference_gateway",
            preexec_fn=os.setsid,
            env={**os.environ, "PROXY_HOST": "0.0.0.0", "PROXY_PORT": "8000"}
        )
        
        # Wait for gateway to start
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.gateway_url}/docs", timeout=2)
                if response.status_code == 200:
                    console.print(f"✅ Inference gateway started at {self.gateway_url}", style="green")
                    return process
            except:
                time.sleep(1)
        
        console.print("⚠️ Inference gateway may not have started properly", style="yellow")
        return process
    
    def _stop_inference_gateway(self, process):
        """Stop the inference gateway process."""
        if process:
            try:
                os.killpg(os.getpgid(process.pid), 9)
                console.print("✅ Stopped inference gateway", style="green")
            except:
                pass
    
    def get_sample_problems(self, focused: bool = False) -> List[Dict[str, Any]]:
        """Get the sample problems (first 5 from each dataset) or focused problems."""
        sample_problems = []
        
        if focused:
            # Use the 6 specific problems mentioned by the user
            focused_problems = [
                "affine-cipher",
                "robot-name",
                "pig-latin",
                "poker",
                "grep",
                "rest-api",
                "phone-number",
                "pov",
                "hungman",
                "proverb"
                "scale-generator",
                "beer-song",
                "bowling",
                "connect",
                "food-chain",
                "book-store",
                "list-ops",
                "grade-school",
                "dot-dsl",
                "forth",
                "react"
            ]
            
            # Load Polyglot problems
            polyglot_path = Path("validator/datasets/polyglot/polyglot.json")
            if polyglot_path.exists():
                with open(polyglot_path, "r") as f:
                    polyglot_problems = json.load(f)
                
                # Find the focused problems in polyglot
                for problem in polyglot_problems:
                    if problem["name"] in focused_problems:
                        sample_problems.append({
                            "name": problem["name"],
                            "suite": "polyglot",
                            "tests": problem["tests"]
                        })
            
            # Load SWE-bench problems
            swebench_path = Path("validator/datasets/swebench_verified/swebench_verified.json")
            if swebench_path.exists():
                with open(swebench_path, "r") as f:
                    swebench_problems = json.load(f)
                
                # Find the focused problems in swebench
                for problem in swebench_problems:
                    if problem["instance_id"] in focused_problems:
                        sample_problems.append({
                            "name": problem["instance_id"],
                            "suite": "swebench_verified",
                            "tests": problem.get("FAIL_TO_PASS", []) + problem.get("PASS_TO_PASS", [])
                        })
        else:
            # Load Polyglot problems
            polyglot_path = Path("validator/datasets/polyglot/polyglot.json")
            if polyglot_path.exists():
                with open(polyglot_path, "r") as f:
                    polyglot_problems = json.load(f)
                
                # Take first 5 problems
                for problem in polyglot_problems[:5]:
                    sample_problems.append({
                        "name": problem["name"],
                        "suite": "polyglot",
                        "tests": problem["tests"]
                    })
            
            # Load SWE-bench problems
            swebench_path = Path("validator/datasets/swebench_verified/swebench_verified.json")
            if swebench_path.exists():
                with open(swebench_path, "r") as f:
                    swebench_problems = json.load(f)
                
                # Take first 5 problems
                for problem in swebench_problems[:5]:
                    sample_problems.append({
                        "name": problem["instance_id"],
                        "suite": "swebench_verified",
                        "tests": problem.get("FAIL_TO_PASS", []) + problem.get("PASS_TO_PASS", [])
                    })
        
        return sample_problems
    
    def run_agent_on_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on a single problem and return results."""
        problem_name = problem["name"]
        suite_name = problem["suite"]
        
        console.print(f"🧪 Testing {problem_name} ({suite_name})", style="cyan")
        
        # Use the existing ridges.py test-agent command
        cmd = [
            "python", "ridges.py", "test-agent",
            problem_name,
            str(self.agent_path),
            "--timeout", str(self.timeout),
            "--gateway-url", self.gateway_url
        ]
        
        start_time = time.time()
        
        try:
            # Set environment variables for the agent
            env = os.environ.copy()
            env["SANDBOX_PROXY_URL"] = self.gateway_url
            env["AGENT_TIMEOUT"] = str(self.timeout)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 60,  # Add buffer for subprocess timeout
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse the output to extract test results
            # The ridges.py command outputs test results in a specific format
            output_lines = result.stdout.split('\n')
            
            # Look for test result patterns in the output
            test_results = []
            status = "error"
            
            # Check if the test completed successfully
            if "Container exceeded timeout" in result.stdout:
                status = "timeout"
            elif "ERROR" in result.stdout and "========== ERROR ==========" in result.stdout:
                status = "error"
            elif "passed" in result.stdout and "failed" in result.stdout and "skipped" in result.stdout:
                status = "success"
                # Try to extract test results from the output
                test_results = self._parse_test_results(result.stdout)
            elif "SUCCESS" in result.stdout or "test_results" in result.stdout:
                status = "success"
                # Try to extract test results from the output
                test_results = self._parse_test_results(result.stdout)
            
            return {
                "problem_name": problem_name,
                "suite": suite_name,
                "status": status,
                "duration": duration,
                "test_results": test_results,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "problem_name": problem_name,
                "suite": suite_name,
                "status": "timeout",
                "duration": self.timeout + 60,
                "test_results": [],
                "stdout": "",
                "stderr": "Process timed out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "problem_name": problem_name,
                "suite": suite_name,
                "status": "error",
                "duration": time.time() - start_time,
                "test_results": [],
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    def _parse_test_results(self, output: str) -> List[Dict[str, str]]:
        """Parse test results from the ridges.py output."""
        test_results = []
        
        # Look for test result patterns in the output
        lines = output.split('\n')
        in_test_results = False
        
        for line in lines:
            # Check if we're in the test results section
            if "========== TEST RESULTS ==========" in line:
                in_test_results = True
                continue
            elif "========== LOGS ==========" in line:
                in_test_results = False
                continue
            
            if in_test_results and line.strip():
                # Parse lines like "test_decode_a_sentence - no category - pass"
                if " - " in line and ("pass" in line or "fail" in line or "skip" in line):
                    parts = line.split(" - ")
                    if len(parts) >= 3:
                        test_name = parts[0].strip()
                        status = parts[-1].strip()
                        test_results.append({"name": test_name, "status": status})
        
        # If no test results found in the structured format, try the old format
        if not test_results:
            for line in lines:
                if "PASSED" in line or "FAILED" in line:
                    # Extract test name and status
                    if "PASSED" in line:
                        test_name = line.split(":")[0].strip()
                        test_results.append({"name": test_name, "status": "pass"})
                    elif "FAILED" in line:
                        test_name = line.split(":")[0].strip()
                        test_results.append({"name": test_name, "status": "fail"})
        
        return test_results
    
    def calculate_scores(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate scores from the test results."""
        total_problems = len(results)
        successful_problems = sum(1 for r in results if r["status"] == "success")
        timeout_problems = sum(1 for r in results if r["status"] == "timeout")
        error_problems = sum(1 for r in results if r["status"] == "error")
        
        # Calculate test-level scores
        total_tests = 0
        passed_tests = 0
        
        for result in results:
            if result["status"] == "success" and result["test_results"]:
                for test in result["test_results"]:
                    total_tests += 1
                    if test["status"] == "pass":
                        passed_tests += 1
        
        # Calculate scores by suite
        polyglot_results = [r for r in results if r["suite"] == "polyglot"]
        swebench_results = [r for r in results if r["suite"] == "swebench_verified"]
        
        polyglot_success = sum(1 for r in polyglot_results if r["status"] == "success")
        swebench_success = sum(1 for r in swebench_results if r["status"] == "success")
        
        return {
            "overall": {
                "total_problems": total_problems,
                "successful_problems": successful_problems,
                "timeout_problems": timeout_problems,
                "error_problems": error_problems,
                "success_rate": successful_problems / total_problems if total_problems > 0 else 0
            },
            "tests": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "by_suite": {
                "polyglot": {
                    "total": len(polyglot_results),
                    "successful": polyglot_success,
                    "success_rate": polyglot_success / len(polyglot_results) if polyglot_results else 0
                },
                "swebench_verified": {
                    "total": len(swebench_results),
                    "successful": swebench_success,
                    "success_rate": swebench_success / len(swebench_results) if swebench_results else 0
                }
            }
        }
    
    def run_validation(self, focused: bool = False):
        """Run the full validation process."""
        console.print(Panel(
            f"[bold cyan]🧪 Local Agent Validation[/bold cyan]\n"
            f"[yellow]Agent:[/yellow] {self.agent_path}\n"
            f"[yellow]Output:[/yellow] {self.output_path}\n"
            f"[yellow]Timeout:[/yellow] {self.timeout}s\n"
            f"[yellow]Mode:[/yellow] {'Focused (6 specific problems)' if focused else 'Sample (first 5 from each dataset)'}",
            title=" Validation Setup", border_style="cyan"
        ))
        
        # Get sample problems
        sample_problems = self.get_sample_problems(focused)
        console.print(f"📋 Found {len(sample_problems)} sample problems", style="green")
        
        # Start inference gateway
        gateway_process = self._start_inference_gateway()
        
        try:
            # Run validation with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Running validation...", total=len(sample_problems))
                
                results = []
                for i, problem in enumerate(sample_problems):
                    result = self.run_agent_on_problem(problem)
                    results.append(result)
                    
                    # Update progress
                    progress.update(task, advance=1, description=f"Testing {problem['name']}")
                    
                    # Small delay to avoid overwhelming the system
                    time.sleep(1)
            
            # Calculate scores
            scores = self.calculate_scores(results)
            
            # Save results
            output_data = {
                "agent_path": str(self.agent_path),
                "timestamp": time.time(),
                "sample_problems": sample_problems,
                "results": results,
                "scores": scores
            }
            
            with open(self.output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            # Display results
            self.display_results(scores, results)
            
            console.print(f"✅ Results saved to {self.output_path}", style="green")
            
        finally:
            # Stop inference gateway
            self._stop_inference_gateway(gateway_process)
    
    def display_results(self, scores: Dict[str, Any], results: List[Dict[str, Any]]):
        """Display the validation results in a nice format."""
        console.print("\n" + "="*80)
        console.print("📊 VALIDATION RESULTS", style="bold cyan")
        console.print("="*80)
        
        # Overall scores
        overall = scores["overall"]
        console.print(f"\n🎯 Overall Performance:")
        console.print(f"   Problems: {overall['successful_problems']}/{overall['total_problems']} ({overall['success_rate']:.1%})")
        console.print(f"   Timeouts: {overall['timeout_problems']}")
        console.print(f"   Errors: {overall['error_problems']}")
        
        # Test-level scores
        tests = scores["tests"]
        if tests["total_tests"] > 0:
            console.print(f"\n🧪 Test Performance:")
            console.print(f"   Tests: {tests['passed_tests']}/{tests['total_tests']} ({tests['test_success_rate']:.1%})")
        
        # By suite
        by_suite = scores["by_suite"]
        console.print(f"\n📚 By Dataset:")
        for suite_name, suite_scores in by_suite.items():
            console.print(f"   {suite_name}: {suite_scores['successful']}/{suite_scores['total']} ({suite_scores['success_rate']:.1%})")
        
        # Detailed results table
        table = Table(title="Detailed Results")
        table.add_column("Problem", style="cyan")
        table.add_column("Suite", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Tests", style="blue")
        
        for result in results:
            status_color = {
                "success": "green",
                "timeout": "yellow", 
                "error": "red"
            }.get(result["status"], "white")
            
            test_count = len(result["test_results"]) if result["test_results"] else 0
            duration = f"{result['duration']:.1f}s"
            
            table.add_row(
                result["problem_name"],
                result["suite"],
                f"[{status_color}]{result['status']}[/{status_color}]",
                duration,
                str(test_count)
            )
        
        console.print(table)


def compare_results(file1: str, file2: str):
    """Compare results from two validation runs."""
    console.print(Panel(
        f"[bold cyan]📊 Comparing Results[/bold cyan]\n"
        f"[yellow]File 1:[/yellow] {file1}\n"
        f"[yellow]File 2:[/yellow] {file2}",
        title=" Comparison", border_style="cyan"
    ))
    
    # Load results
    with open(file1, "r") as f:
        results1 = json.load(f)
    
    with open(file2, "r") as f:
        results2 = json.load(f)
    
    # Compare scores
    scores1 = results1["scores"]
    scores2 = results2["scores"]
    
    console.print("\n📈 Score Comparison:")
    console.print(f"   Overall Success Rate: {scores1['overall']['success_rate']:.1%} → {scores2['overall']['success_rate']:.1%}")
    console.print(f"   Test Success Rate: {scores1['tests']['test_success_rate']:.1%} → {scores2['tests']['test_success_rate']:.1%}")
    
    # Compare by suite
    for suite_name in scores1["by_suite"]:
        if suite_name in scores2["by_suite"]:
            rate1 = scores1["by_suite"][suite_name]["success_rate"]
            rate2 = scores2["by_suite"][suite_name]["success_rate"]
            console.print(f"   {suite_name}: {rate1:.1%} → {rate2:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Local validation for Ridges agents")
    parser.add_argument("--agent-path", required=True, help="Path to agent file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem in seconds")
    parser.add_argument("--focused", action="store_true", help="Use focused mode with 6 specific problems (affine-cipher, beer-song, grade-school, dot-dsl, forth, react)")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"), help="Compare two result files")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        validator = LocalValidator(args.agent_path, args.output, args.timeout)
        validator.run_validation(focused=args.focused)


if __name__ == "__main__":
    main()


