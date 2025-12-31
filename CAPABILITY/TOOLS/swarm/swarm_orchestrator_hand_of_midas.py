#!/usr/bin/env python3
"""
✨ THE HAND OF MIDAS (Ministral-3:8B) — Everything It Touches Turns to Gold ✨

One agent. No voting. No parliament.

Per file:
1) Midas reads (instruction + error + file).
2) Midas may call tools (local, MCP, optional web) to gather facts.
3) Midas outputs the FULL corrected Python file in a ```python block.
4) We validate (denylist + parse + compile). Optional trial test (pytest).
5) Atomic write + backup + incremental report.

Tool-call format (model must output exactly):
```tool
{"name":"read_file","args":{"path":"REL/PATH.py"}}
```
Final answer format:
```python
# full corrected file
```
Quotes: King Midas mythology - the golden touch.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# --- UNICODE FIX FOR WINDOWS ---
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ------------------------------

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Repo Root
# --------------------------------------------------------------------------------

def find_repo_root() -> Path:
    # Try current file location
    cur = Path(__file__).resolve()
    for _ in range(8):
        if (cur / ".git").exists() or (cur / "THOUGHT").exists():
            return cur
        cur = cur.parent
    # Fallback
    return Path(__file__).resolve().parents[3]

REPO_ROOT = find_repo_root()

DEFAULT_INPUT_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
ALT_INPUT_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_REPORT.json"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "CLAW_REPORT.json"

DEFAULT_OLLAMA_GENERATE = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_TAGS = "http://localhost:11434/api/tags"

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClawConfig:
    repo_root: Path = REPO_ROOT
    input_report: Path = DEFAULT_INPUT_REPORT
    output_report: Path = DEFAULT_OUTPUT_REPORT

    ollama_generate: str = DEFAULT_OLLAMA_GENERATE
    ollama_tags: str = DEFAULT_OLLAMA_TAGS
    model: str = "ministral-3:8b"  # The golden touch - balanced speed and quality
    keep_alive: str = "20m"

    # generation
    temperature: float = 0.2
    num_ctx: int = 32768
    num_predict: int = 8192
    timeout_s: int = 420

    # tool loop
    max_steps: int = 10
    max_tool_result_chars: int = 6000

    # file inclusion
    max_file_chars: int = 28000
    head_chars: int = 9000
    tail_chars: int = 9000
    snippet_radius_lines: int = 40

    # safety
    dry_run: bool = False
    backup: bool = True
    deny_substrings: Tuple[str, ...] = (
        "rm -rf", "shutil.rmtree(", "os.remove(", "os.unlink(",
        "subprocess.run(['rm'", "subprocess.run([\"rm\"",
    )

    # tests
    run_pytest: bool = False
    pytest_timeout_s: int = 120

    # web tool
    enable_web: bool = False
    web_allow_domains: Tuple[str, ...] = ()
    web_max_bytes: int = 400_000
    web_timeout_s: int = 20

# --------------------------------------------------------------------------------
# Logging + Quotes
# --------------------------------------------------------------------------------

_MIDAS_QUOTES = (
    "Everything I touch turns to gold.",
    "The golden touch never fails.",
    "Behold, perfection.",
    "From rust to gold, from bugs to beauty.",
    "Another masterpiece.",
)

def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("HandOfMidas")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger

def midas_say(logger: logging.Logger, key: str) -> None:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_MIDAS_QUOTES)
    logger.info(f'Midas: "{_MIDAS_QUOTES[idx]}"')

# --------------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, cfg: ClawConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._tls = threading.local()
        self._sem = threading.Semaphore(4)

    def _session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is not None: return s
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        self._tls.session = s
        return s

    def tags(self) -> List[str]:
        try:
            r = self._session().get(self.cfg.ollama_tags, timeout=5)
            r.raise_for_status()
            return [m.get("name") for m in r.json().get("models", [])]
        except: return []

    def generate(self, prompt: str, min_tokens: int = None) -> str:
        options = {
            "temperature": self.cfg.temperature, 
            "num_ctx": self.cfg.num_ctx, 
            "num_predict": min_tokens if min_tokens else self.cfg.num_predict
        }
        
        payload = {
            "model": self.cfg.model, 
            "prompt": prompt, 
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": options
        }
        
        self.logger.debug(f"Sending prompt ({len(prompt)} chars) to Ollama with num_predict={options['num_predict']}")
        
        with self._sem:
            r = self._session().post(self.cfg.ollama_generate, json=payload, timeout=self.cfg.timeout_s)
        
        if r.status_code != 200: 
            self.logger.error(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
            raise RuntimeError(f"Ollama error {r.status_code}")
        
        data = r.json()
        response = data.get("response", "")
        
        # Log generation stats
        eval_count = data.get("eval_count", 0)
        self.logger.debug(f"Generated {eval_count} tokens, response length: {len(response)} chars")
        
        # Log if we got an empty response from a successful API call
        if not response.strip():
            self.logger.warning(f"Ollama returned empty response. Full data: {json.dumps(data)[:500]}")
            if data.get("done") and not response:
                self.logger.error("Model may not be loaded or is generating empty output")
        
        return response.strip()

# --------------------------------------------------------------------------------
# Path safety + IO
# --------------------------------------------------------------------------------

def safe_repo_path(repo_root: Path, rel_path: str) -> Path:
    p = (repo_root / rel_path).resolve()
    if repo_root.resolve() not in p.parents and p != repo_root.resolve():
        raise ValueError(f"Escape: {rel_path}")
    return p

def atomic_write(path: Path, content: str, backup: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".claw.bak")
        if bak.exists(): 
            bak.unlink()
        shutil.copy2(path, bak)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), suffix=".tmp", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    os.replace(str(tmp), str(path))

def truncate_at_line_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars: return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > int(max_chars * 0.9): return cut[:nl] + "\n\n# ...[TRUNCATED]...\n"
    return cut + "\n\n# ...[TRUNCATED]...\n"

def file_focus_excerpt(content: str, error: str, head: int, tail: int, radius: int) -> str:
    lines = content.splitlines()
    n = len(lines)
    # Find error line
    m = re.search(r"line\s+(\d+)", error, re.IGNORECASE)
    mid = ""
    if m:
        ln = int(m.group(1))
        start = max(0, ln - radius - 1)
        end = min(n, ln + radius)
        mid = f"\n# ... WINDOW lines {start+1}-{end} ...\n" + "\n".join(lines[start:end]) + "\n"
    
    # Simple truncate approach for head/tail to avoid giant context
    h_text = "\n".join(lines[:radius*2]) # Keep top N lines
    t_text = "\n".join(lines[-radius:])
    return f"# HEAD\n{h_text}\n{mid}\n# TAIL\n{t_text}"

# --------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------

def tool_read_file(cfg: ClawConfig, path: str) -> str:
    try:
        p = safe_repo_path(cfg.repo_root, path)
        if not p.exists(): return f"Missing: {path}"
        return truncate_at_line_boundary(p.read_text("utf-8", "replace"), cfg.max_file_chars)
    except Exception as e: return f"Error: {e}"

def tool_grep(cfg: ClawConfig, pattern: str, root: str = ".", max_hits: int = 50) -> str:
    res = []
    try:
        rx = re.compile(pattern)
        for p in safe_repo_path(cfg.repo_root, root).rglob("*.py"):
            try:
                txt = p.read_text("utf-8", "replace")
                for i, line in enumerate(txt.splitlines(), 1):
                    if rx.search(line):
                        res.append(f"{p.relative_to(cfg.repo_root)}:{i}:{line.strip()}")
                        if len(res) >= max_hits: break
            except: pass
            if len(res) >= max_hits: break
    except Exception as e: return f"Error: {e}"
    return "\n".join(res) if res else "No hits."

def tool_run_pytest(cfg: ClawConfig, target: str) -> str:
    if not cfg.run_pytest: return "pytest disabled"
    try:
        cmd = [sys.executable, "-m", "pytest", "-q", target]
        p = subprocess.run(cmd, cwd=str(cfg.repo_root), capture_output=True, text=True, timeout=cfg.pytest_timeout_s)
        return (p.stdout + "\n" + p.stderr)[-2000:]
    except Exception as e: return f"Error: {e}"

TOOLS = {
    "read_file": tool_read_file,
    "grep": tool_grep,
}

# --------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------

def extract_tool_call(text: str) -> Optional[Dict]:
    m = re.search(r"```tool\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None

def extract_python(text: str) -> Optional[str]:
    # Try standard markdown first
    m = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    
    # Try generic code block
    m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if m: 
        code = m.group(1).strip()
        # Only return if it looks like Python
        if any(keyword in code for keyword in ["def ", "class ", "import ", "from ", "if ", "for ", "while "]):
            return code
    
    # If prompt ended with ```python\n, model might just continue directly
    # Look for code until closing ```
    m = re.search(r"^(.*?)\n```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if len(code) > 50 and any(keyword in code for keyword in ["def ", "class ", "import ", "from "]):
            return code
    
    return None

def solve_one(cfg: ClawConfig, client: OllamaClient, logger: logging.Logger, item: Dict) -> Dict:
    rel = item.get("file", "")
    if not rel: return {"status": "failed", "reason": "no_file", "file": ""}
    
    midas_say(logger, rel)
    
    try:
        fpath = safe_repo_path(cfg.repo_root, rel)
    except ValueError as e:
        logger.error(f"Path error: {e}")
        return {"status": "failed", "reason": "invalid_path", "file": rel}
    
    if not fpath.exists(): 
        return {"status": "failed", "reason": "miss", "file": rel}
    
    orig = fpath.read_text("utf-8", "replace")
    error_msg = item.get("last_error", "").strip()
    
    # For code models, show the actual file content (or smart excerpt)
    if len(orig) <= cfg.max_file_chars:
        code_context = orig
    else:
        code_context = file_focus_excerpt(orig, error_msg, 0, 0, cfg.snippet_radius_lines)
    
    # Don't start the code block - let the model generate it
    transcript = (
        f"Fix the following Python code that has an error.\n\n"
        f"File: {rel}\n"
        f"Error: {error_msg}\n\n"
        f"Current code:\n"
        f"```python\n"
        f"{code_context}\n"
        f"```\n\n"
        f"Please provide the complete fixed version of the file in a ```python code block:\n"
    )
    
    history = []
    empty_count = 0
    unproductive_count = 0
    
    logger.info(f"Prompt length: {len(transcript)} chars, context: {len(code_context)} chars")
    
    for step in range(cfg.max_steps):

        logger.debug(f"Step {step+1}/{cfg.max_steps} generating...")
        try:
            # If we've had empty responses, try forcing more tokens
            min_tokens = 1000 if empty_count > 0 else None
            res = client.generate(transcript, min_tokens=min_tokens)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"status": "failed", "reason": "generation_error", "file": rel, "tools": history}
            
        logger.debug(f"Raw Output ({len(res)} chars): {res[:400]}...")

        if not res.strip():
            empty_count += 1
            logger.warning(f"Empty response from model (count: {empty_count}).")
            if empty_count >= 3:
                logger.error("Too many empty responses, giving up.")
                return {"status": "failed", "reason": "empty_responses", "file": rel, "tools": history}
            transcript += "\n# (Model output empty, try again with more detail)\n"
            time.sleep(1)
            continue
        
        # Check for minimal responses (just newlines, etc)
        if len(res.strip()) < 10:
            empty_count += 1
            logger.warning(f"Minimal response from model: '{res}' (count: {empty_count})")
            if empty_count >= 3:
                logger.error("Too many minimal responses, giving up.")
                return {"status": "failed", "reason": "minimal_responses", "file": rel, "tools": history}
            transcript += f"\n# (Response too short: '{res}'. Please provide the COMPLETE fixed file in ```python ... ```)\n"
            time.sleep(1)
            continue
        
        # Check tool
        tc = extract_tool_call(res)
        if tc:
            empty_count = 0
            unproductive_count = 0
            history.append(tc)
            logger.info(f"Tool Call: {tc.get('name')} {tc.get('args')}")
            # Execute
            fn = TOOLS.get(tc.get("name"))
            if fn:
                try:
                    tres = fn(cfg, **tc.get("args", {}))
                    truncated_result = tres[:cfg.max_tool_result_chars]
                    transcript += f"\n{res}\n\n# TOOL RESULT:\n'''\n{truncated_result}\n'''\n\n# NEXT ACTION:\n"
                except Exception as e:
                    logger.warning(f"Tool execution error: {e}")
                    transcript += f"\n{res}\n\n# TOOL ERROR: {e}\n# NEXT ACTION:\n"
            else:
                logger.warning(f"Unknown tool: {tc.get('name')}")
                transcript += f"\n{res}\n\n# ERROR: Unknown tool '{tc.get('name')}'\n# AVAILABLE TOOLS: {list(TOOLS.keys())}\n# NEXT ACTION:\n"
            continue
        
        # Check code
        code = extract_python(res)
        if code:
            empty_count = 0
            unproductive_count = 0
            logger.info(f"Code extracted: {len(code)} chars")
            
            # Safety check
            for deny in cfg.deny_substrings:
                if deny in code:
                    logger.error(f"Denied pattern found: {deny}")
                    return {"status": "failed", "reason": "denied_pattern", "file": rel, "tools": history}
            
            # Validate
            try:
                ast.parse(code)
                if not cfg.dry_run: 
                    atomic_write(fpath, code, cfg.backup)
                    logger.info(f"✓ Fixed {rel}")
                else:
                    logger.info(f"✓ Would fix {rel} (dry-run)")
                return {"status": "fixed", "file": rel, "tools": history}
            except Exception as e:
                logger.warning(f"Syntax Error: {e}")
                transcript += f"\n{res}\n\n# SYNTAX ERROR: {e}\n# ACTION: Regenerate corrected code.\n"
                continue
                
        # Fallback (Chat/Think) - no tool call, no code
        unproductive_count += 1
        logger.debug(f"No tool/code extracted (unproductive count: {unproductive_count})")
        logger.debug(f"Response preview: '{res[:300]}'")
        
        if unproductive_count >= 3:
            logger.error(f"Too many unproductive responses.")
            logger.error(f"Last 3 responses were not tool calls or code blocks.")
            logger.error(f"Final response: {res[:800]}")
            return {"status": "failed", "reason": "unproductive_loop", "file": rel, "tools": history}
        
        # Give explicit feedback to model
        transcript += (
            f"\n{res}\n\n"
            f"# ERROR: No code block detected in your response.\n"
            f"# You must output the COMPLETE fixed file inside ```python ... ``` markers.\n"
            f"# Example format:\n"
            f"# ```python\n"
            f"# import os\n"
            f"# def my_function():\n"
            f"#     pass\n"
            f"# ```\n\n"
            f"# Please try again with the complete fixed file:\n"
        )
        time.sleep(0.5)
        
    # Max steps reached with primary model - consult the Oracle (qwen2.5-coder:7b)
    logger.warning(f"Ministral-3:8b stuck after {cfg.max_steps} steps. Consulting qwen2.5-coder:7b...")
    
    # Switch to Qwen Coder for one final attempt
    original_model = cfg.model
    cfg = dataclass.replace(cfg, model="qwen2.5-coder:7b")
    oracle_client = OllamaClient(cfg, logger)
    
    try:
        logger.info("Oracle (Qwen Coder) attempting fix...")
        res = oracle_client.generate(transcript, min_tokens=1000)
        logger.debug(f"Oracle Output ({len(res)} chars): {res[:400]}...")
        
        code = extract_python(res)
        if code:
            logger.info(f"✨ Oracle provided solution: {len(code)} chars")
            # Safety check
            for deny in cfg.deny_substrings:
                if deny in code:
                    logger.error(f"Oracle solution denied: {deny}")
                    return {"status": "failed", "reason": "denied_pattern", "file": rel, "tools": history}
            # Validate
            try:
                ast.parse(code)
                if not cfg.dry_run:
                    atomic_write(fpath, code, cfg.backup)
                    logger.info(f"✓ Fixed {rel} (via Oracle)")
                else:
                    logger.info(f"✓ Would fix {rel} (dry-run, via Oracle)")
                return {"status": "fixed", "file": rel, "tools": history, "oracle": True}
            except Exception as e:
                logger.error(f"Oracle solution has syntax error: {e}")
        else:
            logger.error("Oracle also failed to generate code")
    except Exception as e:
        logger.error(f"Oracle consultation failed: {e}")
    
    # Both models failed
    logger.error(f"Both Ministral and Oracle failed for {rel}")
    return {"status": "failed", "reason": "max_steps_and_oracle", "file": rel, "tools": history}

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="The Hand of Midas")
    p.add_argument("--input", default=str(DEFAULT_INPUT_REPORT))
    p.add_argument("--model", default="ministral-3:8b")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = p.parse_args()
    
    logger = setup_logging(args.debug)
    
    # Determine input report path
    input_path = Path(args.input)
    if not input_path.exists() and ALT_INPUT_REPORT.exists():
        input_path = ALT_INPUT_REPORT
        logger.info(f"Using alternate input: {ALT_INPUT_REPORT}")
    elif not input_path.exists():
        logger.error(f"No input report at {input_path} or {ALT_INPUT_REPORT}")
        return
    
    cfg = ClawConfig(
        input_report=input_path,
        model=args.model,
        dry_run=args.dry_run
    )
    client = OllamaClient(cfg, logger)
    
    # Check if model is available
    logger.info(f"Checking Ollama connection and model availability...")
    try:
        available_models = client.tags()
        if available_models:
            logger.info(f"Available models: {', '.join(available_models[:5])}")
            if cfg.model not in available_models:
                logger.warning(f"Model '{cfg.model}' not found in Ollama!")
                logger.warning(f"You may need to run: ollama pull {cfg.model}")
                # Try to continue anyway in case the model name is slightly different
        else:
            logger.warning("Could not retrieve model list from Ollama")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        return
        
    items = json.loads(cfg.input_report.read_text("utf-8"))
    if not items:
        logger.warning("No items to process")
        return
        
    results = []
    
    logger.info(f"✨ The Hand of Midas touches {len(items)} files with {args.workers} workers...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(solve_one, cfg, client, logger, item): item for item in items}
        
        with tqdm(total=len(items), desc="✨ MIDAS") as pbar:
            for f in as_completed(futures):
                try:
                    res = f.result()
                    results.append(res)
                except Exception as e:
                    item = futures[f]
                    logger.error(f"Worker failed for {item.get('file', '?')}: {e}")
                    results.append({"status": "failed", "reason": "exception", "file": item.get("file", ""), "error": str(e)})
                pbar.update(1)
    
    # Summary
    fixed = sum(1 for r in results if r.get("status") == "fixed")
    failed = len(results) - fixed
    logger.info(f"✓ Fixed: {fixed} | ✗ Failed: {failed}")
    
    cfg.output_report.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report.write_text(json.dumps(results, indent=2), "utf-8")
    logger.info(f"Report written to {cfg.output_report}")

if __name__ == "__main__":
    main()