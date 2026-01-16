#!/usr/bin/env python3
"""
Tool Executor V2 - Practical Tools Edition

Adds:
1. Persistent state (DONE)
2. Web search (DuckDuckGo - no API key)
3. Web fetch (direct URL access)
4. Wikipedia lookup
5. File system access (read-only for safety)
6. Safe shell commands

Focus: Real-world utility, not benchmark maxxing.
"""

import requests
import sys
import re
from typing import Optional, Dict, Any

API_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "nemotron-3-nano-30b-a3b"
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


# Persistent execution context
_PERSISTENT_CONTEXT = {}


def execute_python(code: str) -> str:
    """Execute Python code with persistent state."""
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
            "import scipy.stats as stats",
            "from sympy import *",
            "from fractions import Fraction",
            "import pandas as pd",
        ]

    full_code = "\n".join(imports) + "\n" + code if imports else code

    try:
        stdout = io.StringIO()

        # Check if last line is an expression
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


def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=5), 1):
                results.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}\n")

        return "\n".join(results) if results else "No results found"
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"


def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    try:
        from bs4 import BeautifulSoup

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        # Limit to 5000 chars
        return text[:5000] + ("..." if len(text) > 5000 else "")

    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    except Exception as e:
        return f"Fetch error: {e}"


def wikipedia_lookup(topic: str) -> str:
    """Get Wikipedia summary."""
    try:
        import wikipediaapi

        wiki = wikipediaapi.Wikipedia(
            user_agent='ToolExecutor/2.0',
            language='en'
        )

        page = wiki.page(topic)

        if page.exists():
            summary = page.summary[:2000]
            return f"{page.title}\n{'='*len(page.title)}\n\n{summary}..."
        return f"No Wikipedia page found for: {topic}"

    except ImportError:
        return "Error: wikipedia-api not installed. Run: pip install wikipedia-api"
    except Exception as e:
        return f"Wikipedia error: {e}"


def grokipedia_lookup(topic: str) -> str:
    """Get content from Grokipedia (https://grokipedia.com)."""
    try:
        # Try multiple possible endpoints (based on /page/Semiotics pattern)
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
                    # Try JSON first
                    try:
                        data = response.json()
                        content = data.get('content', data.get('text', str(data)))
                        return f"Grokipedia: {topic}\n{'='*len(topic)}\n\n{content[:2000]}..."
                    except:
                        # Fall back to HTML parsing
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Remove noise
                        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                            tag.decompose()

                        text = soup.get_text(separator='\n', strip=True)
                        return f"Grokipedia: {topic}\n{'='*len(topic)}\n\n{text[:2000]}..."
            except:
                continue

        return f"No Grokipedia page found for: {topic}"

    except Exception as e:
        return f"Grokipedia error: {e}"


def ask_oracle(question: str) -> str:
    """
    Ask another AI (Claude via web chat) for help with questions this model can't answer.

    Use this when:
    - You need current information beyond your knowledge cutoff
    - The question requires deep reasoning you're not confident about
    - You need a second opinion or verification
    - The task is outside your core competencies

    Think of this as "phone a friend" - use sparingly for genuinely hard questions.
    """
    try:
        # Import the oracle bridge
        try:
            from oracle_bridge import ask_oracle_auto
            result = ask_oracle_auto(question)
            return result
        except ImportError:
            return (
                f"Oracle Tool (Not Configured)\n"
                f"{'='*40}\n\n"
                f"Question: {question}\n\n"
                f"The oracle_bridge module is not available.\n\n"
                f"To enable, configure one of:\n"
                f"1. Set OPENAI_API_KEY environment variable (for ChatGPT)\n"
                f"2. Set ANTHROPIC_API_KEY environment variable (for Claude)\n"
                f"3. Run a local LLM at http://localhost:1234\n\n"
                f"See oracle_bridge.py for configuration details."
            )

    except Exception as e:
        return f"Oracle error: {e}"


def read_file_safe(path: str) -> str:
    """Read file contents (read-only for safety)."""
    import os

    # Security: only allow reading from current dir and subdirs
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath('.')

    if not abs_path.startswith(cwd):
        return f"Error: Cannot read files outside current directory"

    if not os.path.exists(abs_path):
        return f"Error: File not found: {path}"

    try:
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Limit to 10k chars
        if len(content) > 10000:
            return content[:10000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content

    except Exception as e:
        return f"Read error: {e}"


def list_directory(path: str = ".") -> str:
    """List directory contents."""
    import os

    if not os.path.exists(path):
        return f"Error: Directory not found: {path}"

    try:
        items = os.listdir(path)
        items.sort()

        result = []
        for item in items[:100]:  # Limit to 100 items
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


def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from various formats."""

    def normalize_unicode(code: str) -> str:
        """Convert fancy Unicode characters to ASCII equivalents."""
        replacements = {
            # Fancy quotes to ASCII
            '\u201c': '"',  # "
            '\u201d': '"',  # "
            '\u2018': "'",  # '
            '\u2019': "'",  # '
            '\u00ab': '"',  # <<
            '\u00bb': '"',  # >>
            '\u201e': '"',  # ,,
            '\u201a': "'",  # ,
            # Dashes
            '\u2013': '-',  # en-dash
            '\u2014': '-',  # em-dash
            '\u2212': '-',  # minus sign
            # Spaces
            '\u00a0': ' ',  # non-breaking space
            '\u2003': ' ',  # em space
            '\u2002': ' ',  # en space
            # Other common issues
            '\u2026': '...',  # ellipsis
            '\u00d7': '*',    # multiplication sign
            '\u00f7': '/',    # division sign
        }
        for unicode_char, ascii_char in replacements.items():
            code = code.replace(unicode_char, ascii_char)
        return code

    def strip_repl_prompts(code: str) -> str:
        """Strip >>> and ... prompts from REPL-style code."""
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            # Strip >>> prompt
            if line.startswith('>>> '):
                cleaned.append(line[4:])
            # Strip ... continuation prompt
            elif line.startswith('... '):
                cleaned.append(line[4:])
            # Skip bare >>> or ...
            elif line.strip() in ('>>>', '...'):
                continue
            else:
                cleaned.append(line)
        return '\n'.join(cleaned)

    def clean_code(code: str) -> str:
        """Apply all cleaning steps."""
        code = normalize_unicode(code)
        code = strip_repl_prompts(code)
        return code

    # Try markdown code blocks first
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        return clean_code(code)

    # Try <tool_call> format
    pattern = r"<parameter=code>\s*(.*?)</parameter>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        return clean_code(code)

    # Try generic code blocks
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if any(kw in code for kw in ['import', 'print', 'def ', '=', 'for ', 'if ', '>>>']):
            return clean_code(code)

    return None


def extract_tool_call(text: str) -> Optional[tuple]:
    """Extract tool calls from text (function syntax or JSON)."""
    import json

    # Try JSON format first: {"action": "search_web", "parameters": {"query": "..."}}
    # Also handles: {"action": "search_web", "action_input": "query"}
    try:
        # Find JSON-like structures - handle nested braces for parameters
        # Pattern 1: Simple (no nested braces) - for action_input format
        json_pattern_simple = r'\{[^{}]*"action"[^{}]*\}'
        # Pattern 2: One level nesting - for parameters format
        json_pattern_nested = r'\{[^{}]*"action"[^{}]*\{[^{}]*\}[^{}]*\}'

        json_matches = re.findall(json_pattern_nested, text, re.DOTALL)
        json_matches += re.findall(json_pattern_simple, text, re.DOTALL)

        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                action = data.get("action")
                # Handle both "parameters" dict and "action_input" string formats
                params = data.get("parameters", {})
                if isinstance(params, str):
                    params = {}  # Reset if params was a string
                action_input = data.get("action_input", "")

                tool_map = {
                    "search_web": (search_web, params.get("query") or action_input),
                    "fetch_url": (fetch_url, params.get("url") or action_input),
                    "wiki": (wikipedia_lookup, params.get("topic") or action_input),
                    "wikipedia": (wikipedia_lookup, params.get("topic") or action_input),
                    "grok": (grokipedia_lookup, params.get("topic") or action_input),
                    "grokipedia": (grokipedia_lookup, params.get("topic") or action_input),
                    "oracle": (ask_oracle, params.get("question") or action_input),
                    "read_file": (read_file_safe, params.get("path") or action_input),
                    "list_dir": (list_directory, params.get("path") or action_input or "."),
                }

                if action in tool_map:
                    func, arg = tool_map[action]
                    if arg:
                        return (func, arg)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    # Try function call syntax: search_web("query")
    # ONLY include working tools - disabled tools won't be parsed
    patterns = [
        # WORKING TOOLS
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
        # DISABLED - commented out so model won't try to use them
        # (r'search_web\("([^"]+)"\)', search_web),
        # (r'wiki\("([^"]+)"\)', wikipedia_lookup),
        # (r'oracle\("([^"]+)"\)', ask_oracle),
    ]

    for pattern, func in patterns:
        match = re.search(pattern, text)
        if match:
            return (func, match.group(1))

    return None


def call_model(messages: list) -> Dict[str, Any]:
    """Call the model API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    response = requests.post(API_URL, json=payload, timeout=900)  # 5 minutes for complex prompts
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

        # Check for Python code
        code = extract_python_code(assistant_msg)

        # Check for tool calls
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
            # No code or tool - model is done
            full_response = assistant_msg
            break

    return full_response


def safe_print(text):
    """Print with fallback for unicode issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def main():
    if len(sys.argv) < 2:
        prompt = "Search the web for the current world population and compute what percentage lives in Asia."
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
