#!/usr/bin/env python3
"""
GLM-4.7-Flash Test Runner

Runs task benchmarks against glm-4.7-flash model.
Endpoint: http://10.5.0.2:1234
Model: zai-org/glm-4.7-flash

Usage:
  python run_tests.py --level warmup
  python run_tests.py --level all
  python run_tests.py --single easy-2
  python run_tests.py --parallel 2  # Run 2 tests in parallel
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

# Configuration for glm-4.7-flash
API_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "zai-org/glm-4.7-flash"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a computational assistant with extensive tool access.

CRITICAL: Use the RIGHT tool for the job. Don't search if you can compute. Don't compute if you need current data.

TOOLS:
1. Python (math, numpy, scipy, sympy, pandas) - Use for computation
2. grok("topic") - Use for Grokipedia knowledge base (preferred for technical topics)
3. fetch_url("url") - Use to read a webpage
4. read_file("path") - Use to read local files
5. list_dir("path") - Use to list directory contents

DISABLED (API required, not configured):
- search_web("query") - Requires duckduckgo-search
- wiki("topic") - Requires wikipedia-api
- oracle("question") - Requires oracle_bridge module

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
read_file("config.json")
list_dir(".")

Be fast. Choose the right tool. Execute immediately."""

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# TASKS (same as task_benchmarks.py)
# =============================================================================

WARMUP_TASKS = [
    {"id": "warmup-1", "name": "Basic Arithmetic", "prompt": "What is 7 * 8? Use Python to compute this.", "expected": "56", "tool": "python"},
    {"id": "warmup-2", "name": "String Reverse", "prompt": "Reverse the string 'hello world' using Python.", "expected": "dlrow olleh", "tool": "python"},
    {"id": "warmup-3", "name": "Wikipedia Lookup", "prompt": "What year was the Eiffel Tower completed? Look it up on Wikipedia.", "expected": "1889", "tool": "wikipedia"},
    {"id": "warmup-4", "name": "Simple Search", "prompt": "Who is the current CEO of Microsoft? Search the web.", "expected": "Satya Nadella", "tool": "web_search"},
]

EASY_TASKS = [
    {"id": "easy-1", "name": "Prime Check", "prompt": "Is 97 a prime number? Write Python code to check.", "expected": "True", "tool": "python"},
    {"id": "easy-2", "name": "Factorial", "prompt": "What is 15 factorial? Compute it in Python and express in scientific notation.", "expected": "1307674368000", "tool": "python"},
    {"id": "easy-3", "name": "URL Fetch Inference", "prompt": "Fetch https://en.wikipedia.org/wiki/Albert_Einstein and determine his age when he published the special theory of relativity.", "expected": "26", "tool": "fetch_url"},
    {"id": "easy-4", "name": "API Fetch + Extract", "prompt": "Fetch the Bitcoin price from https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd and extract the current USD price.", "expected": "numeric", "tool": "fetch_url"},
]

MEDIUM_TASKS = [
    {"id": "medium-1", "name": "Lookup + Calculate", "prompt": """Use grok to look up the speed of light (in m/s).
Then calculate how many seconds it takes light to travel from the Sun to Earth (distance: 149.6 million km).
Show your calculation in Python.""", "expected": "~499 seconds", "tools": ["grok", "python"]},
    {"id": "medium-2", "name": "Fetch + Compute", "prompt": """Fetch https://www.worldometers.info/world-population/ to get current world population estimate.
Then calculate what percentage of the world lives in India (population ~1.4 billion).
Use Python for the calculation.""", "expected": "~17-18%", "tools": ["fetch_url", "python"]},
    {"id": "medium-3", "name": "Compute + Verify", "prompt": """Compute the first 20 Fibonacci numbers in Python.
Then compute the ratio of consecutive numbers (F(n+1)/F(n)) for the last 5 pairs.
What value does this ratio approach?""", "expected": "golden ratio, ~1.618", "tools": ["python"]},
    {"id": "medium-4", "name": "Multi-Source Synthesis", "prompt": """Fetch https://en.wikipedia.org/wiki/Mount_Everest and extract its height.
Fetch https://en.wikipedia.org/wiki/Mariana_Trench and extract its depth.
What is the total vertical distance between the highest and lowest points on Earth?
Use Python if needed for calculation.""", "expected": "~19,833 meters", "tools": ["fetch_url", "python"]},
]

HARD_TASKS = [
    {"id": "hard-1", "name": "Fact Verification", "prompt": """Claim: "The Great Wall of China is visible from space."
Fetch https://en.wikipedia.org/wiki/Great_Wall_of_China to verify this claim.
Provide evidence for or against, citing the sources.""", "expected": "FALSE (not visible with naked eye from space)", "tools": ["fetch_url"]},
    {"id": "hard-2", "name": "Compound Interest Calculation", "prompt": """Assume the US Federal Reserve interest rate is 4.5%.
If you invested $10,000 at this rate compounded monthly:
1. How much would you have after 5 years?
2. How long to double your money?
Use Python for calculations. Show formulas used.""", "expected": "calculation with 4.5% rate", "tools": ["python"]},
    {"id": "hard-3", "name": "Statistical Analysis", "prompt": """Generate 1000 random samples from a normal distribution with mean=100, std=15.
Calculate: mean, median, std, skewness, kurtosis.
Then perform a Shapiro-Wilk test to verify normality.
Report p-value and conclusion.""", "expected": "p > 0.05, normally distributed", "tools": ["python"]},
    {"id": "hard-4", "name": "Historical Research", "prompt": """Research the Space Race:
1. Fetch https://en.wikipedia.org/wiki/Sputnik_1 - when did it launch?
2. Fetch https://en.wikipedia.org/wiki/Apollo_11 - when did it land on the moon?
3. Calculate the time between these events in days
4. List at least 3 major milestones between these dates.""", "expected": "dates + calculation", "tools": ["fetch_url", "python"]},
]

EXPERT_TASKS = [
    {"id": "expert-1", "name": "Orbital Mechanics", "prompt": """Calculate the orbital period of a satellite at altitude 400km above Earth.
Use:
- Earth's radius: 6,371 km
- Earth's mass: 5.972 x 10^24 kg
- G = 6.674 x 10^-11 N*m^2/kg^2

Apply Kepler's third law. Express answer in minutes.
Then look up the actual orbital period of the ISS to verify.""", "expected": "~92 minutes", "tools": ["python", "wikipedia"]},
    {"id": "expert-2", "name": "Economic Analysis", "prompt": """Look up the current GDP of USA, China, and Japan.
Calculate:
1. Combined GDP as percentage of world GDP
2. If China's GDP grows at 5% and USA at 2%, when will China's GDP exceed USA's?
3. Plot a 20-year projection (just describe the trend, no actual plot needed)""", "expected": "analysis with projections", "tools": ["web_search", "python"]},
    {"id": "expert-3", "name": "Cryptography Basics", "prompt": """Implement RSA encryption from scratch:
1. Generate two small primes p=61, q=53
2. Compute n = p*q and phi(n) = (p-1)(q-1)
3. Choose e=17, verify gcd(e, phi(n)) = 1
4. Compute d such that e*d mod phi(n) = 1
5. Encrypt the message m=42
6. Decrypt to verify you get 42 back

Show all calculations in Python.""", "expected": "working RSA with m=42 recovered", "tools": ["python"]},
    {"id": "expert-4", "name": "Climate Data Analysis", "prompt": """Research climate change:
1. What was the global average temperature anomaly in 1900? (search)
2. What is the current temperature anomaly? (search)
3. Calculate the rate of change per decade
4. If this rate continues, project the anomaly for 2050
5. What is the Paris Agreement target? How close/far are we?""", "expected": "data + projection + comparison", "tools": ["web_search", "python"]},
]

ALL_LEVELS = {
    "warmup": ("WARMUP", WARMUP_TASKS),
    "easy": ("EASY", EASY_TASKS),
    "medium": ("MEDIUM", MEDIUM_TASKS),
    "hard": ("HARD", HARD_TASKS),
    "expert": ("EXPERT", EXPERT_TASKS),
}

# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

_PERSISTENT_CONTEXT = {}

def execute_python(code: str) -> str:
    """Execute Python code with persistent state."""
    global _PERSISTENT_CONTEXT
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
            exec(full_code, {"__builtins__": __builtins__}, _PERSISTENT_CONTEXT)

        output = stdout.getvalue().strip()
        return output if output else "(No output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
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
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    except Exception as e:
        return f"Fetch error: {e}"


def grokipedia_lookup(topic: str) -> str:
    """Get content from Grokipedia."""
    try:
        endpoints = [
            f"https://grokipedia.com/page/{topic}",
            f"https://grokipedia.com/api/page/{topic}",
            f"https://grokipedia.com/wiki/{topic}",
            f"https://grokipedia.com/{topic}",
        ]
        for endpoint in endpoints:
            try:
                headers = {'User-Agent': 'ToolExecutor/2.0'}
                response = requests.get(endpoint, timeout=10, headers=headers)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        content = data.get('content', data.get('text', str(data)))
                        return f"Grokipedia: {topic}\n{'='*len(topic)}\n\n{content[:2000]}..."
                    except:
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
    """Read file contents (read-only for safety)."""
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath('.')
    if not abs_path.startswith(cwd):
        return f"Error: Cannot read files outside current directory"
    if not os.path.exists(abs_path):
        return f"Error: File not found: {path}"
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if len(content) > 10000:
            return content[:10000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"Read error: {e}"


def list_directory(path: str = ".") -> str:
    """List directory contents."""
    if not os.path.exists(path):
        return f"Error: Directory not found: {path}"
    try:
        items = os.listdir(path)
        items.sort()
        result = []
        for item in items[:100]:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {item}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[FILE] {item} ({size} bytes)")
        if len(items) > 100:
            result.append(f"... and {len(items) - 100} more items")
        return "\n".join(result)
    except Exception as e:
        return f"Directory error: {e}"


# =============================================================================
# CODE/TOOL EXTRACTION
# =============================================================================

def normalize_unicode(code: str) -> str:
    """Convert fancy Unicode characters to ASCII equivalents."""
    replacements = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u00ab': '"', '\u00bb': '"', '\u201e': '"', '\u201a': "'",
        '\u2013': '-', '\u2014': '-', '\u2212': '-',
        '\u00a0': ' ', '\u2003': ' ', '\u2002': ' ',
        '\u2026': '...', '\u00d7': '*', '\u00f7': '/',
    }
    for unicode_char, ascii_char in replacements.items():
        code = code.replace(unicode_char, ascii_char)
    return code


def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from various formats."""
    def clean_code(code: str) -> str:
        code = normalize_unicode(code)
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            if line.startswith('>>> '):
                cleaned.append(line[4:])
            elif line.startswith('... '):
                cleaned.append(line[4:])
            elif line.strip() in ('>>>', '...'):
                continue
            else:
                cleaned.append(line)
        return '\n'.join(cleaned)

    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return clean_code(matches[-1].strip())

    pattern = r"<parameter=code>\s*(.*?)</parameter>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return clean_code(matches[-1].strip())

    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if any(kw in code for kw in ['import', 'print', 'def ', '=', 'for ', 'if ', '>>>']):
            return clean_code(code)

    return None


def extract_tool_call(text: str) -> Optional[tuple]:
    """Extract tool calls from text."""
    patterns = [
        (r'fetch_url\("([^"]+)"\)', fetch_url),
        (r"fetch_url\('([^']+)'\)", fetch_url),
        (r'grok\("([^"]+)"\)', grokipedia_lookup),
        (r"grok\('([^']+)'\)", grokipedia_lookup),
        (r'grokipedia\("([^"]+)"\)', grokipedia_lookup),
        (r"grokipedia\('([^']+)'\)", grokipedia_lookup),
        (r'read_file\("([^"]+)"\)', read_file_safe),
        (r"read_file\('([^']+)'\)", read_file_safe),
        (r'list_dir\("([^"]+)"\)', list_directory),
        (r"list_dir\('([^']+)'\)", list_directory),
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
    """Call the model API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    response = requests.post(API_URL, json=payload, timeout=600)  # Increased to 10 min
    response.raise_for_status()
    return response.json()


def run_with_tools(prompt: str, verbose: bool = True) -> str:
    """Run the model with tool execution loop."""
    def vprint(text):
        if verbose:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode('ascii', 'replace').decode('ascii'))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    full_response = ""

    for iteration in range(MAX_ITERATIONS):
        vprint(f"\n{'='*60}")
        vprint(f"Iteration {iteration + 1}")
        vprint('='*60)

        result = call_model(messages)
        assistant_msg = result["choices"][0]["message"]["content"]

        vprint(f"\nModel response:\n{assistant_msg[:500]}..." if len(assistant_msg) > 500 else f"\nModel response:\n{assistant_msg}")

        code = extract_python_code(assistant_msg)
        tool_call = extract_tool_call(assistant_msg)

        if code:
            vprint(f"\n--- Executing Python ---")
            vprint(code)
            vprint("------------------------")

            output = execute_python(code)

            vprint(f"\n--- Output ---")
            vprint(output)
            vprint("--------------")

            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": f"Python output:\n```\n{output}\n```\n\nContinue if needed, or provide your final answer."})

            full_response = assistant_msg + f"\n\n[Executed: {output}]\n\n"

        elif tool_call:
            func, arg = tool_call
            vprint(f"\n--- Calling Tool: {func.__name__}('{arg}') ---")

            output = func(arg)

            vprint(f"\n--- Tool Output ---")
            vprint(output[:500] + "..." if len(output) > 500 else output)
            vprint("-------------------")

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

def save_result(task_id: str, prompt: str, result: str, status: str, expected: str, subdir: str = "task-benchmarks"):
    """Save test result to output directory."""
    outdir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(outdir, exist_ok=True)

    filename = f"{task_id}.json"
    filepath = os.path.join(outdir, filename)

    data = {
        "test_suite": "task_benchmarks",
        "model": MODEL,
        "endpoint": API_URL,
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "prompt": prompt.strip(),
        "expected": expected,
        "result": result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Saved to {subdir}/{filename}]")


def run_task(task: dict) -> dict:
    """Run a single task and return result."""
    global _PERSISTENT_CONTEXT
    _PERSISTENT_CONTEXT = {}  # Reset context for each task

    task_id = task["id"]
    name = task["name"]
    prompt = task["prompt"]
    expected = task.get("expected", "N/A")

    print(f"\n{'='*70}")
    print(f"TASK: {task_id} - {name}")
    print(f"{'='*70}")
    print(f"PROMPT: {prompt[:300]}..." if len(prompt) > 300 else f"PROMPT: {prompt}")
    print(f"EXPECTED: {expected}")
    print(f"{'='*70}")

    try:
        result = run_with_tools(prompt.strip())
        status = "completed"
        print(f"\nRESULT:\n{result[:1000]}..." if len(result) > 1000 else f"\nRESULT:\n{result}")
        save_result(task_id, prompt, result, status, expected)
        return {"task_id": task_id, "status": status, "result": result}

    except KeyboardInterrupt:
        print("\n[INTERRUPTED BY USER]")
        save_result(task_id, prompt, "INTERRUPTED", "interrupted", expected)
        return {"task_id": task_id, "status": "interrupted", "result": "INTERRUPTED"}

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        save_result(task_id, prompt, str(e), "error", expected)
        return {"task_id": task_id, "status": "error", "result": str(e)}


def run_task_parallel(task: dict) -> dict:
    """Run a single task (for parallel execution)."""
    global _PERSISTENT_CONTEXT
    # Create a fresh context for this thread
    import threading
    thread_context = {}

    task_id = task["id"]
    name = task["name"]
    prompt = task["prompt"]
    expected = task.get("expected", "N/A")

    print(f"\n[START] {task_id} - {name}")

    try:
        # Inline run_with_tools to use thread-local context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()}
        ]

        full_response = ""

        for iteration in range(MAX_ITERATIONS):
            result = call_model(messages)
            assistant_msg = result["choices"][0]["message"]["content"]

            code = extract_python_code(assistant_msg)
            tool_call = extract_tool_call(assistant_msg)

            if code:
                # Execute with thread-local context
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
                        exec(full_code, {"__builtins__": __builtins__}, thread_context)

                    output = stdout.getvalue().strip()
                    output = output if output else "(No output)"
                except Exception as e:
                    output = f"Error: {type(e).__name__}: {e}"

                messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "user", "content": f"Python output:\n```\n{output}\n```\n\nContinue if needed, or provide your final answer."})
                full_response = assistant_msg + f"\n\n[Executed: {output}]\n\n"

            elif tool_call:
                func, arg = tool_call
                output = func(arg)
                messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "user", "content": f"Tool output:\n```\n{output}\n```\n\nContinue if needed, or provide your final answer."})
                full_response = assistant_msg + f"\n\n[Tool: {output}]\n\n"

            else:
                full_response = assistant_msg
                break

        status = "completed"
        save_result(task_id, prompt, full_response, status, expected)
        print(f"[DONE] {task_id} - {status}")
        return {"task_id": task_id, "status": status, "result": full_response}

    except Exception as e:
        print(f"[ERROR] {task_id}: {e}")
        save_result(task_id, prompt, str(e), "error", expected)
        return {"task_id": task_id, "status": "error", "result": str(e)}


def run_level(level_key: str, parallel: int = 1) -> list:
    """Run all tasks at a given level."""
    if level_key not in ALL_LEVELS:
        print(f"Unknown level: {level_key}")
        return []

    level_name, tasks = ALL_LEVELS[level_key]

    print(f"\n{'#'*70}")
    print(f"# LEVEL: {level_name}")
    print(f"# Tasks: {len(tasks)}")
    print(f"# Parallel: {parallel}")
    print(f"{'#'*70}")

    if parallel > 1:
        results = []
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(run_task_parallel, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        return results
    else:
        results = []
        for task in tasks:
            result = run_task(task)
            results.append(result)
            if result["status"] == "interrupted":
                break
        return results


def run_single(task_id: str) -> dict:
    """Run a single task by ID."""
    for level_key, (_, tasks) in ALL_LEVELS.items():
        for task in tasks:
            if task["id"] == task_id:
                return run_task(task)

    print(f"Task not found: {task_id}")
    return {"task_id": task_id, "status": "not_found", "result": None}


def run_all(parallel: int = 1) -> dict:
    """Run all levels in order."""
    all_results = {}
    for level_key in ["warmup", "easy", "medium", "hard", "expert"]:
        results = run_level(level_key, parallel=parallel)
        all_results[level_key] = results

        if results and any(r["status"] == "interrupted" for r in results):
            break

    return all_results


def print_summary(results: dict):
    """Print summary of results."""
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}")

    total = 0
    completed = 0
    errors = 0

    for level, level_results in results.items():
        level_completed = sum(1 for r in level_results if r["status"] == "completed")
        level_total = len(level_results)
        print(f"  {level.upper():10s}: {level_completed}/{level_total} completed")
        total += level_total
        completed += level_completed
        errors += sum(1 for r in level_results if r["status"] == "error")

    print(f"{'='*70}")
    print(f"  TOTAL:      {completed}/{total} completed, {errors} errors")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-4.7-Flash Task Benchmarks")
    parser.add_argument("--level", "-l",
                        choices=["warmup", "easy", "medium", "hard", "expert", "all"],
                        default="warmup",
                        help="Difficulty level to run")
    parser.add_argument("--single", "-s", type=str,
                        help="Run single task by ID (e.g., 'easy-2')")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of parallel tasks (default: 1)")
    parser.add_argument("--list", action="store_true",
                        help="List all available tasks")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Tasks:")
        print("="*70)
        for level_key, (level_name, tasks) in ALL_LEVELS.items():
            print(f"\n{level_name}:")
            for task in tasks:
                print(f"  {task['id']:15s} - {task['name']}")
        sys.exit(0)

    print(f"""
    ===================================================================
                    GLM-4.7-FLASH BENCHMARK SUITE

      Endpoint: {API_URL}
      Model: {MODEL}

      Progressive difficulty testing for tool-augmented LLMs
      Press Ctrl+C to abort any task.
    ===================================================================
    """)

    if args.single:
        result = run_single(args.single)
        print(f"\nTask {args.single}: {result['status']}")
    elif args.level == "all":
        results = run_all(parallel=args.parallel)
        print_summary(results)
    else:
        results = run_level(args.level, parallel=args.parallel)
        print_summary({args.level: results})
