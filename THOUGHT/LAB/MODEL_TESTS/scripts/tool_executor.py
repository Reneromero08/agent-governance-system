"""
Tool Executor for Nemotron 3 Nano 30B
Gives the model a real TI-89: Python code execution with results returned.

Usage:
    python tool_executor.py "Is 2^67-1 prime? If not, factorize it completely."
"""

import requests
import json
import sys
import re
from typing import Optional, Dict, Any

API_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "nemotron-3-nano-30b-a3b"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a computational assistant with Python access. USE THE TOOL IMMEDIATELY.

CRITICAL: Do NOT spend time reasoning about the problem. Write Python code FIRST, get results, THEN explain.

Write code in ```python blocks. It will be executed and results returned.

Available: math, numpy (as np), sympy, itertools, fractions

ALWAYS start with code. Think AFTER you see results. Be fast.

Example - DO THIS:
```python
import sympy
result = sympy.factorint(2**67 - 1)
print(result)
```

Then explain after seeing output."""

# Persistent execution context - shared across all tool calls in a session
_PERSISTENT_CONTEXT = {}

def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    import io
    import contextlib
    global _PERSISTENT_CONTEXT

    code = code.strip()

    # Add common imports if not present
    imports = []
    if "import" not in code:
        imports = [
            "import math",
            "import numpy as np",
            "from sympy import *",
            "from fractions import Fraction",
        ]

    full_code = "\n".join(imports) + "\n" + code if imports else code

    try:
        stdout = io.StringIO()

        # Check if last line is an expression (not assignment, import, etc.)
        lines = full_code.strip().split('\n')
        last_line = lines[-1].strip()

        # If last line looks like an expression, wrap it in print
        # Check for assignment: single = not part of ==, !=, <=, >=
        import re
        is_assignment = bool(re.search(r'(?<![=!<>])=(?!=)', last_line.split('(')[0]))

        if (last_line and
            not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', '#', 'return ', 'raise ', 'assert ')) and
            not is_assignment and
            not last_line.endswith(':')):
            # Wrap last expression in print
            lines[-1] = f"print({last_line})"
            full_code = '\n'.join(lines)

        with contextlib.redirect_stdout(stdout):
            exec(full_code, {"__builtins__": __builtins__}, _PERSISTENT_CONTEXT)

        output = stdout.getvalue().strip()
        return output if output else "(No output)"

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from various formats."""

    # Try markdown code blocks first: ```python ... ```
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try <tool_call> format
    pattern = r"<parameter=code>\s*(.*?)</parameter>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try generic code blocks: ``` ... ```
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        # Only return if it looks like Python
        if any(kw in code for kw in ['import', 'print', 'def ', '=', 'for ', 'if ']):
            return code

    return None


def call_model(messages: list) -> Dict[str, Any]:
    """Call the model API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2500,
        "temperature": 0.6
    }

    response = requests.post(API_URL, json=payload, timeout=180)
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

        # Call model
        result = call_model(messages)
        assistant_msg = result["choices"][0]["message"]["content"]

        vprint(f"\nModel response:\n{assistant_msg[:500]}..." if len(assistant_msg) > 500 else f"\nModel response:\n{assistant_msg}")

        # Check for code to execute
        code = extract_python_code(assistant_msg)

        if code:
            vprint(f"\n--- Executing Python ---")
            vprint(code)
            vprint("------------------------")

            output = execute_python(code)

            vprint(f"\n--- Output ---")
            vprint(output)
            vprint("--------------")

            # Add to conversation
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": f"Python output:\n```\n{output}\n```\n\nContinue your analysis."})

            full_response = assistant_msg + f"\n\n[Executed: {output}]\n\n"
        else:
            # No code to execute - model is done
            full_response = assistant_msg
            break

    return full_response


def safe_print(text):
    """Print with fallback for unicode issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def main():
    if len(sys.argv) < 2:
        # Default test: Mersenne 67
        prompt = "Is 2^67 - 1 prime? If not, provide its complete prime factorization. Use Python to compute."
    else:
        prompt = " ".join(sys.argv[1:])

    safe_print(f"Prompt: {prompt}")
    safe_print("="*60)

    result = run_with_tools(prompt, verbose=True)

    safe_print("\n" + "="*60)
    safe_print("FINAL RESULT")
    safe_print("="*60)
    safe_print(result)


if __name__ == "__main__":
    main()
