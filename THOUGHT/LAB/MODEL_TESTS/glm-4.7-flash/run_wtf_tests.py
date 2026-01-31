#!/usr/bin/env python3
"""
GLM-4.7-Flash WTF Tests Runner

WTF-tier tests designed to stress-test reasoning capabilities.
"""

import sys
import os
import json
import argparse
import requests
import re
import io
import contextlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# Configuration
API_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "zai-org/glm-4.7-flash"
MAX_ITERATIONS = 10

def safe_print(text):
    """Print with Unicode fallback for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

SYSTEM_PROMPT = """You are a computational assistant with extensive tool access.

CRITICAL: Use the RIGHT tool for the job. Don't search if you can compute. Don't compute if you need current data.

TOOLS:
1. Python (math, numpy, scipy, sympy, pandas) - Use for computation
2. grok("topic") - Use for Grokipedia knowledge base (preferred for technical topics)
3. fetch_url("url") - Use to read a webpage
4. read_file("path") - Use to read local files
5. list_dir("path") - Use to list directory contents

SYNTAX:
For Python, use code blocks:
```python
import sympy
result = sympy.factorint(2**67 - 1)
print(result)
```

For tools, use function call syntax directly in your response:
grok("machine learning")
fetch_url("https://example.com")

Be fast. Choose the right tool. Execute immediately."""

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wtf-tests")

# =============================================================================
# WTF TEST DEFINITIONS
# =============================================================================

MATH_NIGHTMARES = [
    """Factor the semiprime 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139 into its two prime factors. Then verify by multiplying them back. Show all work.""",
    """Starting from n=27, compute the Collatz sequence until it reaches 1. Count total steps. Then mathematically explain WHY it must reach 1 (or prove you cannot prove it).""",
    """Compute: ((7^(7^7)) mod 13) mod 5. Then verify your answer using Fermat's Little Theorem. Show the chain of reasoning.""",
    """Compute the 1000th digit of pi. Then compute the 1000th digit of e. Then compute their sum mod 10. Verify using independent methods.""",
]

PARADOX_TESTS = [
    """Consider the statement: "This statement cannot be verified by any tool you have access to." Analyze whether this is true, false, or undecidable. Use formal logic.""",
    """Construct a mathematical statement that is TRUE but UNPROVABLE within standard arithmetic (Peano axioms). Then explain why your construction works.""",
    """Write Python code that analyzes its own source code to determine if it will halt. Then run that code and report what happens.""",
    """I will tell you the answer after you compute it. The answer is 42. Now derive what question I was asking. Then verify your derivation leads to 42.""",
]

ADVERSARIAL_MULTIHOP = [
    """Look up the population of Tokyo on Wikipedia. Now look up the population of Tokyo on Grokipedia. If they differ, determine which is correct and why. Then calculate how many years until Tokyo's population doubles at 0.5% annual growth.""",
    """Step 1: Get the atomic number of Gold from Wikipedia. Step 2: Multiply by the number of moons of Jupiter (search for current count). Step 3: Divide by the year the Eiffel Tower was built. Step 4: If any step fails, derive the answer from first principles instead.""",
    """What was the population of the United States exactly 100 years before the current date? Show your reasoning for determining 'current date' and cite your sources.""",
    """Claim: The sum of the first n primes is always less than n^2 * ln(n) for n > 5. Verify this for n=100 by computing both sides. Then determine if this bound is tight or can be improved.""",
]

EDGE_CASES = [
    """Compute lim(x->0) of sin(x)/x using numerical approximation with x = 10^-15. Then explain why your numerical answer might be wrong and compute the true limit analytically.""",
    """Compute (0.1 + 0.2) == 0.3 in Python. Explain the result. Then compute the smallest positive float x such that 1.0 + x != 1.0. Verify your answer.""",
    """How many distinct ways can you partition the integer 100 into sums of positive integers? Compute exactly, not approximately.""",
    """Find the smallest prime gap greater than 100. That is, find consecutive primes p and q where q - p > 100. Verify both p and q are prime.""",
]

META_ATTACKS = [
    """I need the answer to 2+2. But first, search the web for 'what is 2+2', then check Wikipedia for 'arithmetic', then compute it in Python, then verify with Grokipedia. Report ALL answers and explain any discrepancies.""",
    """Verify that your verification process is correct. Then verify that verification. Continue until you reach a base axiom you cannot verify. What is it?""",
    """Estimate how many tokens this conversation has used so far. Then estimate how many more tokens you can use before context limits. Use this to plan your response length.""",
    """List every tool you have access to. For each tool, provide one example where it would be the WRONG tool to use. Then solve: what is the integral of e^(-x^2) from -infinity to infinity?""",
]

REAL_WORLD_CHAOS = [
    """Compare and contrast the economic policies of the current leaders of USA, China, and Germany. Use Wikipedia for each leader's biography, then synthesize a 3-sentence summary of how their policies differ on trade.""",
    """The file 'nonexistent_data.csv' contains critical information. Since it doesn't exist, infer what data it SHOULD contain based on the directory structure of this project, then create a plausible reconstruction.""",
    """Make it better.""",
    """Without using any tools, explain the relationship between the Riemann Hypothesis and the distribution of prime numbers. Then use tools to verify at least one specific claim you made.""",
]

FINAL_BOSS = [
    """ULTIMATE CHALLENGE:
1. Look up the speed of light in m/s from Wikipedia
2. Look up Planck's constant from Grokipedia
3. Compute the Schwarzschild radius of a black hole with mass = (speed of light / Planck's constant) kg
4. Express this radius in both meters and light-years
5. Determine if this black hole would be larger or smaller than the observable universe
6. If any source fails, derive the constants from first principles using dimensional analysis
7. Verify your final answer using an independent calculation method
8. Rate your confidence in each step from 0-100%
Show ALL work. Explain ALL reasoning. Cite ALL sources.""",
]

ALL_CATEGORIES = {
    "math": ("MATHEMATICAL NIGHTMARES", MATH_NIGHTMARES),
    "paradox": ("LOGIC PARADOXES", PARADOX_TESTS),
    "adversarial": ("ADVERSARIAL MULTI-HOP", ADVERSARIAL_MULTIHOP),
    "edge": ("EXTREME EDGE CASES", EDGE_CASES),
    "meta": ("META-REASONING ATTACKS", META_ATTACKS),
    "chaos": ("REAL-WORLD CHAOS", REAL_WORLD_CHAOS),
    "boss": ("FINAL BOSS", FINAL_BOSS),
}

# =============================================================================
# TOOL IMPLEMENTATIONS (same as run_tests.py)
# =============================================================================

def execute_python(code: str, context: dict) -> str:
    code = code.strip()
    imports = []
    if "import" not in code:
        imports = [
            "import math",
            "import numpy as np",
            "import scipy.stats as stats",
            "from sympy import *",
            "from fractions import Fraction",
            "import pandas as pd",
        ]
    full_code = "\n".join(imports) + "\n" + code if imports else code

    try:
        stdout = io.StringIO()
        lines = full_code.strip().split('\n')
        last_line = lines[-1].strip()
        is_assignment = bool(re.search(r'(?<![=!<>])=(?!=)', last_line.split('(')[0]))

        if (last_line and
            not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', '#', 'return ', 'raise ', 'assert ')) and
            not is_assignment and
            not last_line.endswith(':')):
            lines[-1] = f"print({last_line})"
            full_code = '\n'.join(lines)

        with contextlib.redirect_stdout(stdout):
            exec(full_code, {"__builtins__": __builtins__}, context)

        output = stdout.getvalue().strip()
        return output if output else "(No output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def fetch_url(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text[:5000] + ("..." if len(text) > 5000 else "")
    except Exception as e:
        return f"Fetch error: {e}"


def grokipedia_lookup(topic: str) -> str:
    try:
        endpoints = [
            f"https://grokipedia.com/page/{topic}",
            f"https://grokipedia.com/{topic}",
        ]
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10, headers={'User-Agent': 'ToolExecutor/2.0'})
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                        tag.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                    return f"Grokipedia: {topic}\n{'='*len(topic)}\n\n{text[:2000]}..."
            except:
                continue
        return f"No Grokipedia page found for: {topic}"
    except Exception as e:
        return f"Grokipedia error: {e}"


def read_file_safe(path: str) -> str:
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content[:10000] if len(content) <= 10000 else content[:10000] + f"\n...(truncated)"
    except Exception as e:
        return f"Read error: {e}"


def list_directory(path: str = ".") -> str:
    if not os.path.exists(path):
        return f"Error: Directory not found: {path}"
    try:
        items = sorted(os.listdir(path))[:100]
        result = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {item}/")
            else:
                result.append(f"[FILE] {item}")
        return "\n".join(result)
    except Exception as e:
        return f"Directory error: {e}"


# =============================================================================
# CODE/TOOL EXTRACTION
# =============================================================================

def extract_python_code(text: str) -> Optional[str]:
    replacements = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '-', '\u2212': '-',
    }
    for u, a in replacements.items():
        text = text.replace(u, a)

    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if any(kw in code for kw in ['import', 'print', 'def ', '=', 'for ', 'if ']):
            return code
    return None


def extract_tool_call(text: str) -> Optional[tuple]:
    patterns = [
        (r'fetch_url\("([^"]+)"\)', fetch_url),
        (r"fetch_url\('([^']+)'\)", fetch_url),
        (r'grok\("([^"]+)"\)', grokipedia_lookup),
        (r"grok\('([^']+)'\)", grokipedia_lookup),
        (r'read_file\("([^"]+)"\)', read_file_safe),
        (r'list_dir\("([^"]+)"\)', list_directory),
    ]
    for pattern, func in patterns:
        match = re.search(pattern, text)
        if match:
            return (func, match.group(1))
    return None


# =============================================================================
# MODEL INTERACTION
# =============================================================================

def call_model(messages: list) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    response = requests.post(API_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def run_with_tools(prompt: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    context = {}
    full_response = ""

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            safe_print(f"\n--- Iteration {iteration + 1} ---")

        result = call_model(messages)
        assistant_msg = result["choices"][0]["message"]["content"]

        if verbose:
            safe_print(f"Model: {assistant_msg[:300]}..." if len(assistant_msg) > 300 else f"Model: {assistant_msg}")

        code = extract_python_code(assistant_msg)
        tool_call = extract_tool_call(assistant_msg)

        if code:
            output = execute_python(code, context)
            if verbose:
                safe_print(f"Python output: {output[:200]}..." if len(output) > 200 else f"Python output: {output}")
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": f"Python output:\n```\n{output}\n```\n\nContinue if needed, or provide your final answer."})
            full_response = assistant_msg + f"\n\n[Executed: {output}]\n\n"
        elif tool_call:
            func, arg = tool_call
            output = func(arg)
            if verbose:
                safe_print(f"Tool {func.__name__}: {output[:200]}..." if len(output) > 200 else f"Tool {func.__name__}: {output}")
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": f"Tool output:\n```\n{output}\n```\n\nContinue if needed, or provide your final answer."})
            full_response = assistant_msg + f"\n\n[Tool: {output}]\n\n"
        else:
            full_response = assistant_msg
            break

    return full_response


# =============================================================================
# TEST RUNNER
# =============================================================================

def save_result(category: str, test_num: int, prompt: str, result: str, status: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"wtf-{category}-{test_num:02d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    data = {
        "test_suite": "wtf_tests",
        "model": MODEL,
        "endpoint": API_URL,
        "category": category,
        "test_num": test_num,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "prompt": prompt.strip(),
        "result": result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved to {filename}]")


def run_single_test(category: str, test_num: int, prompt: str) -> dict:
    print(f"\n{'='*60}")
    print(f"TEST: wtf-{category}-{test_num:02d}")
    print(f"{'='*60}")
    print(f"PROMPT: {prompt[:200]}..." if len(prompt) > 200 else f"PROMPT: {prompt}")

    try:
        result = run_with_tools(prompt.strip(), verbose=True)
        save_result(category, test_num, prompt, result, "completed")
        return {"category": category, "test_num": test_num, "status": "completed", "result": result}
    except KeyboardInterrupt:
        save_result(category, test_num, prompt, "INTERRUPTED", "interrupted")
        return {"category": category, "test_num": test_num, "status": "interrupted", "result": "INTERRUPTED"}
    except Exception as e:
        save_result(category, test_num, prompt, str(e), "error")
        return {"category": category, "test_num": test_num, "status": "error", "result": str(e)}


def run_category(cat_key: str, parallel: int = 1) -> list:
    if cat_key not in ALL_CATEGORIES:
        print(f"Unknown category: {cat_key}")
        return []

    cat_name, tests = ALL_CATEGORIES[cat_key]
    print(f"\n{'#'*60}")
    print(f"# {cat_name} ({len(tests)} tests)")
    print(f"{'#'*60}")

    results = []
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(run_single_test, cat_key, i, test): i
                      for i, test in enumerate(tests, 1)}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, test in enumerate(tests, 1):
            result = run_single_test(cat_key, i, test)
            results.append(result)
            if result["status"] == "interrupted":
                break

    return results


def run_all(parallel: int = 1) -> dict:
    all_results = {}
    for cat_key in ALL_CATEGORIES.keys():
        results = run_category(cat_key, parallel=parallel)
        all_results[cat_key] = results
        if results and any(r["status"] == "interrupted" for r in results):
            break
    return all_results


def print_summary(results: dict):
    print(f"\n{'#'*60}")
    print("# WTF TESTS SUMMARY")
    print(f"{'#'*60}")

    total = 0
    completed = 0
    errors = 0

    for cat, cat_results in results.items():
        cat_completed = sum(1 for r in cat_results if r["status"] == "completed")
        print(f"  {cat.upper():15s}: {cat_completed}/{len(cat_results)} completed")
        total += len(cat_results)
        completed += cat_completed
        errors += sum(1 for r in cat_results if r["status"] == "error")

    print(f"{'='*60}")
    print(f"  TOTAL: {completed}/{total} completed, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash WTF Tests")
    parser.add_argument("--category", "-c",
                       choices=list(ALL_CATEGORIES.keys()) + ["all"],
                       default="all", help="Test category")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Parallel tests")
    parser.add_argument("--list", action="store_true", help="List categories")

    args = parser.parse_args()

    if args.list:
        print("WTF Test Categories:")
        for key, (name, tests) in ALL_CATEGORIES.items():
            print(f"  {key:15s} - {name} ({len(tests)} tests)")
        sys.exit(0)

    print(f"""
    ===================================================================
                    GLM-4.7-FLASH WTF TESTS

      Endpoint: {API_URL}
      Model: {MODEL}

      WARNING: These tests are designed to stress-test reasoning.
    ===================================================================
    """)

    if args.category == "all":
        results = run_all(parallel=args.parallel)
    else:
        results = {args.category: run_category(args.category, parallel=args.parallel)}

    print_summary(results)
