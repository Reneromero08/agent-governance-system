#!/usr/bin/env python3
"""
🎯 THE PROFESSIONAL (Ministral-8B Dual-Mode System) 🎯

Two-level escalation for ministral-3:8b:
1. LEVEL 1: Restrictive Mode - Fast, focused, minimal thinking
   - Strict prompt: "Fix only what's broken, no refactoring"
   - Lower temperature (0.1)
   - Shorter context window
   - Quick turnaround

2. LEVEL 2: Thinking Mode - Deep analysis, full context
   - Analytical prompt: "Think step-by-step, explain reasoning"
   - Higher temperature (0.3)
   - Full context window
   - Chain-of-thought enabled

"The Professional doesn't waste time. Level 1 for simple fixes, Level 2 for complex problems."
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Repo Root
# --------------------------------------------------------------------------------

def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for _ in range(8):
        if (cur / ".git").exists() or (cur / "THOUGHT").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parents[3]

REPO_ROOT = find_repo_root()

DEFAULT_INPUT = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
DEFAULT_OUTPUT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "PROFESSIONAL_REPORT.json"

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfessionalConfig:
    repo_root: Path = REPO_ROOT
    input_manifest: Path = DEFAULT_INPUT
    output_report: Path = DEFAULT_OUTPUT
    
    model: str = "ministral-3:8b"
    keep_alive: str = "20m"
    
    # Level 1: Restrictive
    level1_temperature: float = 0.1
    level1_num_ctx: int = 16384
    level1_num_predict: int = 4096
    level1_timeout: int = 180
    
    # Level 2: Thinking
    level2_temperature: float = 0.3
    level2_num_ctx: int = 32768
    level2_num_predict: int = 8192
    level2_timeout: int = 420
    
    max_file_chars: int = 28000
    max_workers: int = 3
    
    block_dangerous_ops: bool = True
    keep_backups: bool = True

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------

def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("professional")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    
    logger.addHandler(sh)
    return logger

# --------------------------------------------------------------------------------
# HTTP Session
# --------------------------------------------------------------------------------

def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = build_session()

# --------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------

LEVEL1_PROMPT_TEMPLATE = """[PROFESSIONAL - LEVEL 1: RESTRICTIVE MODE]

You are a professional Python developer. Fix ONLY what is broken. No refactoring, no style changes.

INSTRUCTION: {instruction}

ERROR: {error}

FILE: {file_path}

```python
{file_content}
```

OUTPUT THE COMPLETE FIXED FILE in a ```python block. Nothing else.
"""

LEVEL2_PROMPT_TEMPLATE = """[PROFESSIONAL - LEVEL 2: THINKING MODE]

You are a senior Python architect. Think step-by-step and explain your reasoning.

INSTRUCTION: {instruction}

ERROR: {error}

FILE: {file_path}

```python
{file_content}
```

THINK THROUGH THE PROBLEM:
1. What is the root cause?
2. What needs to change?
3. Are there edge cases?

Then OUTPUT THE COMPLETE FIXED FILE in a ```python block.
"""

# --------------------------------------------------------------------------------
# LLM Call
# --------------------------------------------------------------------------------

def call_ollama(
    prompt: str,
    model: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    timeout: int,
    keep_alive: str,
    logger: logging.Logger,
) -> str:
    """Call Ollama API with retry logic."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    
    for attempt in range(3):
        try:
            r = SESSION.post(OLLAMA_API, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("response", "")
            logger.warning(f"Ollama returned {r.status_code}, retrying...")
        except Exception as e:
            logger.warning(f"Ollama call failed (attempt {attempt+1}): {e}")
            time.sleep(0.5 * (attempt + 1))
    
    raise RuntimeError("Ollama call failed after 3 attempts")

# --------------------------------------------------------------------------------
# File Processing
# --------------------------------------------------------------------------------

def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from markdown fence."""
    match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1)
    return None

def validate_python(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def process_file_level1(
    task: Dict[str, Any],
    config: ProfessionalConfig,
    logger: logging.Logger,
) -> Tuple[bool, Optional[str], str]:
    """Try Level 1: Restrictive mode."""
    file_path = task["file"]
    instruction = task.get("instruction", "Fix all test failures")
    error = task.get("last_error", "Unknown error")
    
    full_path = config.repo_root / file_path
    if not full_path.exists():
        return False, None, f"File not found: {file_path}"
    
    file_content = full_path.read_text(encoding="utf-8")
    if len(file_content) > config.max_file_chars:
        file_content = file_content[:config.max_file_chars] + "\n# ... (truncated)"
    
    prompt = LEVEL1_PROMPT_TEMPLATE.format(
        instruction=instruction,
        error=error,
        file_path=file_path,
        file_content=file_content,
    )
    
    logger.info(f"Level 1 (Restrictive): {file_path}")
    
    try:
        response = call_ollama(
            prompt=prompt,
            model=config.model,
            temperature=config.level1_temperature,
            num_ctx=config.level1_num_ctx,
            num_predict=config.level1_num_predict,
            timeout=config.level1_timeout,
            keep_alive=config.keep_alive,
            logger=logger,
        )
        
        code = extract_python_code(response)
        if not code:
            return False, None, "No Python code block in response"
        
        valid, err = validate_python(code)
        if not valid:
            return False, None, f"Syntax error: {err}"
        
        return True, code, "Level 1 success"
        
    except Exception as e:
        return False, None, f"Level 1 failed: {e}"

def process_file_level2(
    task: Dict[str, Any],
    config: ProfessionalConfig,
    logger: logging.Logger,
) -> Tuple[bool, Optional[str], str]:
    """Try Level 2: Thinking mode."""
    file_path = task["file"]
    instruction = task.get("instruction", "Fix all test failures")
    error = task.get("last_error", "Unknown error")
    
    full_path = config.repo_root / file_path
    file_content = full_path.read_text(encoding="utf-8")
    if len(file_content) > config.max_file_chars:
        file_content = file_content[:config.max_file_chars] + "\n# ... (truncated)"
    
    prompt = LEVEL2_PROMPT_TEMPLATE.format(
        instruction=instruction,
        error=error,
        file_path=file_path,
        file_content=file_content,
    )
    
    logger.info(f"Level 2 (Thinking): {file_path}")
    
    try:
        response = call_ollama(
            prompt=prompt,
            model=config.model,
            temperature=config.level2_temperature,
            num_ctx=config.level2_num_ctx,
            num_predict=config.level2_num_predict,
            timeout=config.level2_timeout,
            keep_alive=config.keep_alive,
            logger=logger,
        )
        
        code = extract_python_code(response)
        if not code:
            return False, None, "No Python code block in response"
        
        valid, err = validate_python(code)
        if not valid:
            return False, None, f"Syntax error: {err}"
        
        return True, code, "Level 2 success"
        
    except Exception as e:
        return False, None, f"Level 2 failed: {e}"

def write_file_safely(path: Path, content: str, config: ProfessionalConfig) -> None:
    """Write file with backup and atomic-ish rename for Windows."""
    if config.keep_backups and path.exists():
        backup = path.with_suffix(path.suffix + ".prof_bak")
        try:
            if backup.exists():
                backup.unlink()
            path.rename(backup)
        except Exception:
            pass # fallback if rename fails
    
    # Write to temp first
    tmp = path.with_suffix(path.suffix + ".prof_tmp")
    tmp.write_text(content, encoding="utf-8")
    
    try:
        if path.exists():
            path.unlink()
        tmp.rename(path)
    except Exception:
        # Fallback if atomic rename fails
        path.write_text(content, encoding="utf-8")
        if tmp.exists():
            tmp.unlink()

def process_file(
    task: Dict[str, Any],
    config: ProfessionalConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Process a single file with 2-level escalation."""
    file_path = task["file"]
    
    # Try Level 1
    success, code, message = process_file_level1(task, config, logger)
    
    if success:
        full_path = config.repo_root / file_path
        write_file_safely(full_path, code, config)
        return {
            "file": file_path,
            "status": "fixed",
            "level": 1,
            "message": message,
        }
    
    logger.warning(f"Level 1 failed: {message}")
    
    # Escalate to Level 2
    success, code, message = process_file_level2(task, config, logger)
    
    if success:
        full_path = config.repo_root / file_path
        write_file_safely(full_path, code, config)
        return {
            "file": file_path,
            "status": "fixed",
            "level": 2,
            "message": message,
        }
    
    return {
        "file": file_path,
        "status": "failed",
        "level": 2,
        "message": message,
    }

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="The Professional: Ministral-8B Dual-Mode")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="ministral-3:8b")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    config = ProfessionalConfig(
        input_manifest=args.input,
        output_report=args.output,
        model=args.model,
        max_workers=args.workers,
    )
    
    logger = setup_logging(args.debug)
    
    logger.info("=" * 70)
    logger.info("🎯 THE PROFESSIONAL (Ministral-8B Dual-Mode)")
    logger.info("=" * 70)
    logger.info(f"Model: {config.model}")
    logger.info(f"Level 1: Restrictive (temp={config.level1_temperature})")
    logger.info(f"Level 2: Thinking (temp={config.level2_temperature})")
    logger.info("")
    
    # Load manifest
    with open(config.input_manifest, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    logger.info(f"Processing {len(tasks)} files...")
    
    results = []
    for task in tqdm(tasks, desc="🎯 PROFESSIONAL"):
        result = process_file(task, config, logger)
        results.append(result)
    
    # Write report
    report = {
        "model": config.model,
        "total": len(results),
        "level1_fixed": sum(1 for r in results if r.get("level") == 1),
        "level2_fixed": sum(1 for r in results if r.get("level") == 2),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }
    
    with open(config.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 RESULTS")
    logger.info("=" * 70)
    logger.info(f"Level 1 (Restrictive): {report['level1_fixed']} fixed")
    logger.info(f"Level 2 (Thinking): {report['level2_fixed']} fixed")
    logger.info(f"Failed: {report['failed']}")
    logger.info(f"Report: {config.output_report}")
    
    return 0 if report['failed'] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
