#!/usr/bin/env python3
"""
Super Turbo Deluxe Swarm v4.4
- Ant-first immediate start (warm Ant sync, warm others in background)
- Lazy + background file preload
- Ant retries (temps) before escalation
- Architect gets AST-based summary + error feedback loop
- Selective Foreman verification for risky/critical changes
- Line-based diff risk detection
- Plan sanity gate
- Atomic writes with unique temp files + optional backups
- Thread-safe cache + per-thread HTTP sessions
- Rich report (steps, plan, last_error, previews, diff stats)
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
import threading
import difflib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
REPORT_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
LOG_PATH = REPO_ROOT / "swarm_debug.log"

OLLAMA_TAGS = "http://localhost:11434/api/tags"
OLLAMA_API = "http://localhost:11434/api/generate"

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class SwarmConfig:
    max_workers: int = 6
    ollama_slots: int = 2

    ant_model: str = "qwen2.5-coder:0.5b"
    foreman_model: str = "qwen2.5-coder:3b"
    architect_model: str = "qwen2.5-coder:7b"
    consultant_model: str = "qwen2.5:7b"  # The "Regular 7B" helper

    # Context / output budgets
    ant_num_ctx: int = 8192
    ant_num_predict: int = 3072

    architect_num_ctx: int = 8192
    architect_num_predict: int = 4096

    foreman_num_ctx: int = 8192
    foreman_num_predict: int = 4096

    # Strategy
    ant_temps: Tuple[float, ...] = (0.1, 0.3)
    risky_diff_ratio: float = 0.8  # allow more changes
    keep_alive: str = "10m"
    max_file_chars_for_full_send: int = 80_000
    max_preview_chars: int = 800

    # Safety
    block_dangerous_ops: bool = False  # UNBLOCK for test fixes
    keep_backups: bool = True
    backup_suffix: str = ".swarm_bak"
    safe_write_suffix: str = ".swarm_tmp"
    allow_foreman_direct_fallback: bool = True  # NEW: medium model fallback before architect
    allow_architect_direct_fallback: bool = True

    # Critical files heuristics (foreman gate) - TESTBENCH excluded since tests are safe to modify
    critical_prefixes: Tuple[str, ...] = ("CAPABILITY/TOOLS/", "CAPABILITY/PRIMITIVES/", "CAPABILITY/PIPELINES/", "CAPABILITY/SCHEMAS/", "CAPABILITY/MCP/")

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------

def _setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("swarm")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# --------------------------------------------------------------------------------
# Thread-local sessions + shared semaphore
# --------------------------------------------------------------------------------

_THREAD = threading.local()
_OLLAMA_SEM: Optional[threading.Semaphore] = None

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def get_session() -> requests.Session:
    if not hasattr(_THREAD, "session") or _THREAD.session is None:
        _THREAD.session = _build_session()
    return _THREAD.session

# --------------------------------------------------------------------------------
# File cache + background preload
# --------------------------------------------------------------------------------

_FILE_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_PRELOAD_STOP = threading.Event()

def safe_repo_path(rel: str) -> Path:
    # Prevent path traversal and absolute paths
    rel_norm = rel.replace("\\", "/").lstrip("/")
    p = (REPO_ROOT / rel_norm).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"Unsafe path: {rel}")
    return p

def get_cached_file(rel: str) -> str:
    with _CACHE_LOCK:
        if rel in _FILE_CACHE:
            return _FILE_CACHE[rel]
    # load outside lock to reduce contention
    try:
        p = safe_repo_path(rel)
        text = p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception:
        text = ""
    with _CACHE_LOCK:
        _FILE_CACHE[rel] = text
    return text

def _preload_worker(files: List[str], logger: logging.Logger) -> None:
    for rel in files:
        if _PRELOAD_STOP.is_set():
            return
        with _CACHE_LOCK:
            if rel in _FILE_CACHE:
                continue
        try:
            p = safe_repo_path(rel)
            text = p.read_text(encoding="utf-8") if p.exists() else ""
        except Exception as e:
            logger.debug(f"preload error {rel}: {e}")
            text = ""
        with _CACHE_LOCK:
            _FILE_CACHE[rel] = text

# --------------------------------------------------------------------------------
# Model availability + warmup
# --------------------------------------------------------------------------------

_MODEL_FALLBACKS: Dict[str, str] = {
    "ant": "qwen2.5-coder:0.5b",
    "foreman": "qwen2.5-coder:3b",
    "architect": "qwen2.5-coder:7b",
    "consultant": "qwen2.5:7b"
}
_MODEL_READY: Dict[str, threading.Event] = {
    "ant": threading.Event(),
    "foreman": threading.Event(),
    "architect": threading.Event(),
    "consultant": threading.Event()
}

def consultant_help(cfg: SwarmConfig, file_path: str, instruction: str, logger: logging.Logger) -> str:
    """Consultant (Regular 7B) provides high-level advice/strategy."""
    wait_model_ready("consultant", timeout=2.0)
    prompt = (
        "[CONSULTANT]\n"
        "You are a senior technical consultant. Provide a brief, high-level strategy to solve the task.\n"
        "Do not write code. Just explain the approach in 3-4 bullet points.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n"
    )
    
    return run_agent_raw(
        model=_MODEL_FALLBACKS["consultant"],
        prompt=prompt,
        timeout=40,
        temperature=0.3,
        num_predict=512,
        num_ctx=4096,
        keep_alive=cfg.keep_alive,
        logger=logger,
    ).strip()

def list_models(logger: logging.Logger) -> List[str]:
    s = get_session()
    r = s.get(OLLAMA_TAGS, timeout=3)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama not responding: {r.status_code}")
    return [m["name"] for m in r.json().get("models", [])]

def pick_model(requested: str, available: List[str], fallback: str) -> str:
    if requested in available:
        return requested
    base = requested.split(":")[0]
    for m in available:
        if m.startswith(base + ":"):
            return m
    return fallback


def warm_model(tier: str, model: str, logger: logging.Logger) -> None:
    s = get_session()
    try:
        logger.info(f"Warming {tier} ({model})...")
        s.post(
            OLLAMA_API,
            json={"model": model, "prompt": "hi", "stream": False, "keep_alive": "10m", "options": {"num_predict": 1}},
            timeout=30,
        )
        _MODEL_READY[tier].set()
        logger.info(f"[OK] {tier} ready ({model})")
    except Exception as e:
        logger.warning(f"[WARN] Warmup failed for {tier} ({model}): {e}")
        _MODEL_READY[tier].set()  # do not deadlock

def wait_model_ready(tier: str, timeout: float = 30.0) -> None:
    """Wait for model to be ready. Default 30s to handle slow model loading."""
    _MODEL_READY[tier].wait(timeout=timeout)


# --------------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------------

def run_agent_raw(
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    num_predict: int,
    num_ctx: int,
    keep_alive: str,
    logger: logging.Logger,
) -> str:
    # Make the semaphore a hard throttle, retries handled by HTTPAdapter and our loop
    assert _OLLAMA_SEM is not None
    s = get_session()

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

    backoff = 0.35
    for attempt in range(3):
        try:
            with _OLLAMA_SEM:
                r = s.post(OLLAMA_API, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("response", "") or ""
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** attempt))
                continue
            logger.debug(f"ollama http {r.status_code}: {r.text[:200]}")
            return ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.debug(f"ollama retry {attempt+1}/3: {e}")
            time.sleep(backoff * (2 ** attempt))
        except Exception as e:
            logger.debug(f"ollama fatal: {e}")
            return ""
    return ""

# --------------------------------------------------------------------------------
# Extraction + verification
# --------------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)

def extract_code(resp: str) -> Optional[str]:
    if not resp:
        return None
    blocks = _CODE_FENCE_RE.findall(resp)
    if blocks:
        return blocks[-1].strip()

    # fallback: if it looks like code, try whole response
    candidate = resp.strip()
    if candidate.startswith("import ") or candidate.startswith("from ") or "def " in candidate or "class " in candidate:
        # If the model wrapped it in leading chatter, try stripping to first import/def/class
        lines = candidate.splitlines()
        start = None
        for i, ln in enumerate(lines):
            s = ln.lstrip()
            if s.startswith(("import ", "from ", "def ", "class ")):
                start = i
                break
        if start is not None:
            return "\n".join(lines[start:]).strip()
        return candidate
    return None

def compile_check(code: str) -> Tuple[bool, str, Optional[int]]:
    try:
        compile(code, "<swarm>", "exec")
        return True, "", None
    except SyntaxError as e:
        lineno = getattr(e, "lineno", None)
        msg = f"{e.__class__.__name__}: {e.msg} (line {lineno})"
        return False, msg, lineno
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None

def snippet_around_line(code: str, lineno: Optional[int], radius: int = 10) -> str:
    if not lineno:
        return ""
    lines = code.splitlines()
    i = max(0, min(len(lines) - 1, lineno - 1))
    a = max(0, i - radius)
    b = min(len(lines), i + radius + 1)
    out = []
    for idx in range(a, b):
        prefix = ">>" if idx == i else "  "
        out.append(f"{prefix} {idx+1:4d} | {lines[idx]}")
    return "\n".join(out)

def looks_dangerous(code: str) -> bool:
    needles = ("rm -rf", "shutil.rmtree", "os.remove(", "Path.unlink(", "subprocess.run(", "subprocess.Popen(")
    return any(n in code for n in needles)

def enforce_minimal_sanity(original: str, fixed: str) -> Tuple[bool, str]:
    # Only enforce REPO_ROOT if the original had it
    if "REPO_ROOT" in original and "REPO_ROOT" not in fixed:
        return False, "dropped_REPO_ROOT"

    # If Path(...) used, ensure pathlib is available (naive but catches common misses)
    uses_path = ("Path(" in fixed) or ("pathlib." in fixed)
    has_path_import = ("from pathlib import Path" in fixed) or ("import pathlib" in fixed)
    if uses_path and not has_path_import:
        return False, "missing_path_import"

    return True, "ok"

def line_change_ratio(original: str, fixed: str) -> float:
    a = original.splitlines()
    b = fixed.splitlines()
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(a=a, b=b)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += (i2 - i1)
    return changed / max(1, len(a))

# --------------------------------------------------------------------------------
# Architect summary + plan gate
# --------------------------------------------------------------------------------

_PLAN_BAD = re.compile(r"\b(rewrite|refactor\s+entire|entire\s+file|big\s+refactor|massive\s+change)\b", re.I)

def plan_is_sane(plan: str, instruction: str) -> bool:
    if not plan or len(plan) < 20:
        return False
    if "```" in plan:
        return False
    if _PLAN_BAD.search(plan):
        allow = re.search(r"\b(refactor|rewrite|restructure|redesign|overhaul)\b", instruction, re.I)
        return bool(allow)
    return True

def summarize_for_architect(code: str, head_lines: int = 60, tail_lines: int = 60) -> str:
    imports: List[str] = []
    defs: List[str] = []
    classes: List[str] = []
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node) if hasattr(ast, "unparse") else node.__class__.__name__)
            elif isinstance(node, ast.FunctionDef):
                defs.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                defs.append(f"async {node.name}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except Exception:
        pass

    lines = code.splitlines()
    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:]) if len(lines) > tail_lines else ""

    parts = []
    if imports:
        parts.append("IMPORTS:\n" + "\n".join(imports[:40]))
    if classes:
        parts.append("CLASSES:\n" + "\n".join(classes[:60]))
    if defs:
        parts.append("FUNCTIONS:\n" + "\n".join(defs[:120]))
    parts.append("HEAD:\n" + head)
    if tail:
        parts.append("TAIL:\n" + tail)
    return "\n\n".join(parts)

# --------------------------------------------------------------------------------
# Tier logic
# --------------------------------------------------------------------------------

def architect_plan(cfg: SwarmConfig, file_path: str, instruction: str, last_error: str, failed_snippet: str, logger: logging.Logger) -> Optional[str]:
    wait_model_ready("architect", timeout=2.0)
    content = get_cached_file(file_path)
    summary = summarize_for_architect(content if len(content) <= cfg.max_file_chars_for_full_send else content[:cfg.max_file_chars_for_full_send])

    prompt = (
        "[SENIOR ARCHITECT]\n"
        "Create a minimal, targeted fix plan. Bullet points only. No code. No backticks.\n"
        "Keep changes as small as possible.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        + (f"LAST_ERROR: {last_error}\n\n" if last_error else "")
        + (f"FAILED SNIPPET:\n{failed_snippet}\n\n" if failed_snippet else "")
        + f"FILE SUMMARY:\n{summary}\n"
    )

    timeout = 35 + min(45, len(content) // 2000)
    resp = run_agent_raw(
        model=_MODEL_FALLBACKS["architect"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=0.2,
        num_predict=cfg.architect_num_predict,
        num_ctx=cfg.architect_num_ctx,
        keep_alive=cfg.keep_alive,
        logger=logger,
    ).strip()

    if not plan_is_sane(resp, instruction):
        return None
    return resp

def ant_execute(cfg: SwarmConfig, file_path: str, instruction: str, plan: Optional[str], temperature: float, logger: logging.Logger) -> str:
    wait_model_ready("ant", timeout=2.0)
    # Don't read whole file if huge? Ant is 0.5b context limited. 
    # But for now we trust cache size limit in config.
    original = get_cached_file(file_path)
    
    # Prompt engineering for speed
    sys_prompt = (
        "[ANT: SYNTAX REPAIR]\n"
        "You are a fast, precise code repair ant. Fix the syntax/logic error.\n"
        "Return ONLY the corrected python code in a ```python block.\n"
        "Do NOT remove the 'REPO_ROOT' setup lines or imports at the top.\n"
    )
    if plan:
        sys_prompt += f"FOLLOW PLAN:\n{plan}\n"
    
    prompt = (
        f"{sys_prompt}\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n"
        "CODE:\n```python\n"
        f"{original}\n"
        "```\n"
    )

    timeout = 15 + min(30, len(original) // 2000)
    
    return run_agent_raw(
        model=_MODEL_FALLBACKS["ant"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=temperature,
        num_predict=cfg.ant_num_predict,
        num_ctx=cfg.ant_num_ctx,
        keep_alive=cfg.keep_alive,
        logger=logger,
    )

def foreman_verify(cfg: SwarmConfig, file_path: str, instruction: str, original: str, fixed: str, plan: str, logger: logging.Logger) -> bool:
    wait_model_ready("foreman", timeout=2.0)

    def clip(s: str, n: int = 700) -> str:
        if len(s) <= n:
            return s
        return s[:350] + "\n...\n" + s[-350:]

    prompt = (
        "Strict code reviewer. Be conservative.\n"
        "No hallucinations. No unrelated changes. If uncertain, reject.\n"
        "Decide if FIXED correctly implements TASK with minimal necessary changes.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        f"PLAN:\n{plan}\n\n"
        f"ORIGINAL (snippet):\n{clip(original)}\n\n"
        f"FIXED (snippet):\n{clip(fixed)}\n\n"
        'Answer ONLY "YES" or "NO: <short reason>".'
    )

    resp = run_agent_raw(
        model=_MODEL_FALLBACKS["foreman"],
        prompt=prompt,
        timeout=25,
        temperature=0.1,
        num_predict=cfg.foreman_num_predict,
        num_ctx=cfg.foreman_num_ctx,
        keep_alive=cfg.keep_alive,
        logger=logger,
    ).strip()

    return resp.upper().startswith("YES")

def foreman_direct_fix(cfg: SwarmConfig, file_path: str, instruction: str, plan: str, last_error: str, failed_snippet: str, logger: logging.Logger) -> str:
    """Foreman (3B Coder) with THINKING prompts."""
    wait_model_ready("foreman", timeout=2.0)
    original = get_cached_file(file_path)

    prompt = (
        "[FOREMAN: COT FIX]\n"
        "You are a thoughtful engineer. First, THINK about the problem breakdown.\n"
        "Then, generate the COMPLETE code.\n"
        "Strictly follow the plan.\n"
        "Return ONLY the full file in one ```python block.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        f"PLAN:\n{plan}\n\n"
        + (f"LAST_ERROR:\n{last_error}\n\n" if last_error else "")
        + "CURRENT FILE:\n```python\n"
        f"{original}\n"
        "```\n"
    )

    timeout = 60 + min(90, len(original) // 1000)
    return run_agent_raw(
        model=_MODEL_FALLBACKS["foreman"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=0.2,
        num_predict=max(cfg.ant_num_predict, 4096),
        num_ctx=cfg.foreman_num_ctx,
        keep_alive=cfg.keep_alive,
        logger=logger,
    )

def architect_direct_fix(cfg: SwarmConfig, file_path: str, instruction: str, last_error: str, failed_snippet: str, logger: logging.Logger) -> str:
    """Architect (7B Coder) with optional Consultant help."""
    wait_model_ready("architect", timeout=2.0)
    original = get_cached_file(file_path)
    summary = summarize_for_architect(original if len(original) <= cfg.max_file_chars_for_full_send else original[:cfg.max_file_chars_for_full_send])

    # Consult if previous attempts failed or if explicitly tricky
    advice = ""
    if last_error or "complex" in instruction.lower() or "refactor" in instruction.lower():
        logger.info(f"[{file_path}] Architect requests Consultant help...")
        advice = consultant_help(cfg, file_path, instruction, logger)
        if advice:
             logger.info(f"[{file_path}] Consultant advice: {advice[:50]}...")

    prompt = (
        "[SENIOR ARCHITECT: DIRECT FIX]\n"
        "Generate the COMPLETE corrected Python file.\n"
        "Keep changes minimal and focused on the task.\n"
        "Avoid duplicate imports.\n"
        "Return ONLY the full file in one ```python block. No extra text.\n\n"
        + (f"CONSULTANT ADVICE:\n{advice}\n\n" if advice else "") +
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        + (f"LAST_ERROR:\n{last_error}\n\n" if last_error else "")
        + (f"FAILED SNIPPET:\n{failed_snippet}\n\n" if failed_snippet else "")
        + f"FILE SUMMARY:\n{summary}\n\n"
        "CURRENT FILE:\n```python\n"
        f"{original}\n"
        "```\n"
    )

    timeout = 70 + min(100, len(original) // 1000)
    return run_agent_raw(
        model=_MODEL_FALLBACKS["architect"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=0.18,
        num_predict=max(cfg.ant_num_predict, 6144),
        num_ctx=max(cfg.architect_num_ctx, cfg.ant_num_ctx),
        keep_alive=cfg.keep_alive,
        logger=logger,
    )

# --------------------------------------------------------------------------------
# Atomic write + backup
# --------------------------------------------------------------------------------

def atomic_write(cfg: SwarmConfig, target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}{cfg.safe_write_suffix}.{uuid4().hex}.py")
    tmp.write_text(text, encoding="utf-8")

    if cfg.keep_backups and target.exists():
        bak = target.with_name(target.name + cfg.backup_suffix)
        try:
            if bak.exists():
                bak.unlink()
            target.replace(bak)
        except Exception:
            # If backup fails, do not proceed to overwrite
            tmp.unlink(missing_ok=True)
            raise

    tmp.replace(target)

# --------------------------------------------------------------------------------
# Risk heuristics
# --------------------------------------------------------------------------------

def is_critical(cfg: SwarmConfig, rel: str) -> bool:
    rel_norm = rel.replace("\\", "/")
    return rel_norm.startswith(cfg.critical_prefixes)

# --------------------------------------------------------------------------------
# Task processing
# --------------------------------------------------------------------------------

def process_task(cfg: SwarmConfig, task: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    fp = task.get("file", "")
    instruction = task.get("instruction", "")
    start = time.monotonic()

    res: Dict[str, Any] = {
        "file": fp,
        "status": "failed",
        "reason": "unknown",
        "elapsed": 0.0,
        "model_used": None,
        "steps": [],
        "plan": "",
        "last_error": "",
        "code_preview": "",
        "diff_line_ratio": None,
        "critical": is_critical(cfg, fp),
    }

    try:
        target = safe_repo_path(fp)
    except Exception as e:
        res["reason"] = f"unsafe_path:{e}"
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    original = get_cached_file(fp)

    def record(step: str, ok: bool, why: str = "", extra: Optional[Dict[str, Any]] = None, t0: Optional[float] = None) -> None:
        rec = {"step": step, "success": ok, "info": why}
        if t0 is not None:
            rec["time"] = round(time.monotonic() - t0, 2)
        if extra:
            rec.update(extra)
        res["steps"].append(rec)

    last_error = ""
    last_snippet = ""
    plan: Optional[str] = None

    # Attempt 1..N: Ant direct retries
    fixed: Optional[str] = None
    for i, temp in enumerate(cfg.ant_temps, 1):
        t0 = time.monotonic()
        resp = ant_execute(cfg, fp, instruction, plan=None, temperature=temp, logger=logger)
        code = extract_code(resp)
        if not code:
            record("ant_direct", False, "no_code", {"attempt": i, "temp": temp}, t0)
            last_error = "no_code"
            continue

        ok_sanity, why_sanity = enforce_minimal_sanity(original, code)
        ok_comp, why_comp, lineno = compile_check(code)

        if not ok_sanity:
            record("ant_direct", False, f"sanity:{why_sanity}", {"attempt": i, "temp": temp}, t0)
            last_error = f"sanity:{why_sanity}"
            continue

        if cfg.block_dangerous_ops and looks_dangerous(code):
            record("ant_direct", False, "dangerous_ops", {"attempt": i, "temp": temp}, t0)
            last_error = "dangerous_ops"
            continue

        if not ok_comp:
            last_error = why_comp
            last_snippet = snippet_around_line(code, lineno)
            record("ant_direct", False, f"compile:{why_comp}", {"attempt": i, "temp": temp}, t0)
            continue

        fixed = code
        record("ant_direct", True, "ok", {"attempt": i, "temp": temp}, t0)
        break

    # If direct succeeded, continue; else escalate to Architect plan
    if fixed is None:
        t0 = time.monotonic()
        plan = architect_plan(cfg, fp, instruction, last_error, last_snippet, logger)
        res["plan"] = plan or ""
        if not plan:
            record("architect_plan", False, "invalid_plan_or_timeout", None, t0)
            res["last_error"] = last_error
            res["elapsed"] = round(time.monotonic() - start, 2)
            return res
        record("architect_plan", True, "ok", None, t0)

        # Planned Ant retries (same temps)
        for i, temp in enumerate(cfg.ant_temps, 1):
            t1 = time.monotonic()
            resp = ant_execute(cfg, fp, instruction, plan=plan, temperature=temp, logger=logger)
            code = extract_code(resp)
            if not code:
                record("ant_planned", False, "no_code", {"attempt": i, "temp": temp}, t1)
                last_error = "no_code"
                continue

            ok_sanity, why_sanity = enforce_minimal_sanity(original, code)
            ok_comp, why_comp, lineno = compile_check(code)

            if not ok_sanity:
                record("ant_planned", False, f"sanity:{why_sanity}", {"attempt": i, "temp": temp}, t1)
                last_error = f"sanity:{why_sanity}"
                continue

            if cfg.block_dangerous_ops and looks_dangerous(code):
                record("ant_planned", False, "dangerous_ops", {"attempt": i, "temp": temp}, t1)
                last_error = "dangerous_ops"
                continue

            if not ok_comp:
                last_error = why_comp
                last_snippet = snippet_around_line(code, lineno)
                record("ant_planned", False, f"compile:{why_comp}", {"attempt": i, "temp": temp}, t1)
                continue

            fixed = code
            record("ant_planned", True, "ok", {"attempt": i, "temp": temp}, t1)
            break

        # Escalation step 3: Foreman direct-code (medium model)
        if fixed is None and cfg.allow_foreman_direct_fallback and plan:
            t2 = time.monotonic()
            resp = foreman_direct_fix(cfg, fp, instruction, plan, last_error, last_snippet, logger)
            code = extract_code(resp)
            if not code:
                record("foreman_direct", False, "no_code", None, t2)
            else:
                ok_sanity, why_sanity = enforce_minimal_sanity(original, code)
                ok_comp, why_comp, lineno = compile_check(code)
                if (not ok_sanity) or (not ok_comp):
                    why = f"sanity:{why_sanity}" if not ok_sanity else f"compile:{why_comp}"
                    record("foreman_direct", False, why, None, t2)
                    last_error = why
                    last_snippet = snippet_around_line(code, lineno) if lineno else ""
                elif cfg.block_dangerous_ops and looks_dangerous(code):
                    record("foreman_direct", False, "dangerous_ops", None, t2)
                    last_error = "dangerous_ops"
                else:
                    fixed = code
                    record("foreman_direct", True, "ok", None, t2)

        # Escalation step 4 (final): Architect direct-code (big model)
        if fixed is None and cfg.allow_architect_direct_fallback:
            t2 = time.monotonic()
            resp = architect_direct_fix(cfg, fp, instruction, last_error, last_snippet, logger)
            code = extract_code(resp)
            if not code:
                record("architect_direct", False, "no_code", None, t2)
            else:
                ok_sanity, why_sanity = enforce_minimal_sanity(original, code)
                ok_comp, why_comp, lineno = compile_check(code)
                if (not ok_sanity) or (not ok_comp):
                    why = f"sanity:{why_sanity}" if not ok_sanity else f"compile:{why_comp}"
                    record("architect_direct", False, why, None, t2)
                elif cfg.block_dangerous_ops and looks_dangerous(code):
                    record("architect_direct", False, "dangerous_ops", None, t2)
                else:
                    fixed = code
                    record("architect_direct", True, "ok", None, t2)

    if fixed is None:
        res["reason"] = "no_valid_candidate"
        res["last_error"] = last_error
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    # Risk assessment for Foreman gate
    ratio = line_change_ratio(original, fixed)
    res["diff_line_ratio"] = round(ratio, 4)

    risky = res["critical"] or (ratio > cfg.risky_diff_ratio)

    # Foreman verification if risky
    if risky:
        t0 = time.monotonic()
        ok = foreman_verify(cfg, fp, instruction, original, fixed, res["plan"] or "direct_fix", logger)
        record("foreman_verify", ok, "risky_gate", {"risky": True, "ratio": round(ratio, 4)}, t0)
        if not ok:
            res["reason"] = "foreman_reject"
            res["last_error"] = last_error
            res["code_preview"] = fixed[:cfg.max_preview_chars]
            res["elapsed"] = round(time.monotonic() - start, 2)
            return res

    # Write
    try:
        atomic_write(cfg, target, fixed)
        # update cache
        with _CACHE_LOCK:
            _FILE_CACHE[fp] = fixed
        res["status"] = "success"
        res["reason"] = "ok"
        res["model_used"] = _MODEL_FALLBACKS.get("ant")
    except Exception as e:
        res["reason"] = f"write_error:{type(e).__name__}:{e}"
        res["code_preview"] = fixed[:cfg.max_preview_chars]

    res["last_error"] = last_error
    res["elapsed"] = round(time.monotonic() - start, 2)
    return res

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Super Turbo Deluxe Swarm v4.4")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--ollama-slots", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-foreman-direct", action="store_true", help="Disable foreman direct-code intermediate fallback")
    parser.add_argument("--no-architect-direct", action="store_true", help="Disable architect direct-code final fallback")
    args = parser.parse_args()

    cfg = SwarmConfig(
        max_workers=args.max_workers,
        ollama_slots=args.ollama_slots,
        allow_foreman_direct_fallback=(not args.no_foreman_direct),
        allow_architect_direct_fallback=(not args.no_architect_direct),
    )
    logger = _setup_logging(args.debug)

    global _OLLAMA_SEM
    _OLLAMA_SEM = threading.Semaphore(cfg.ollama_slots)

    print("\n" + "=" * 70)
    print("🐜 Super Entropy Crusher v4.5".center(70))
    print("=" * 70 + "\n")

    # Ollama availability + model mapping
    try:
        available = list_models(logger)
    except Exception as e:
        logger.error(f"Ollama not available: {e}")
        sys.exit(1)

    _MODEL_FALLBACKS["ant"] = pick_model(cfg.ant_model, available, cfg.foreman_model)
    _MODEL_FALLBACKS["foreman"] = pick_model(cfg.foreman_model, available, cfg.ant_model)
    _MODEL_FALLBACKS["architect"] = pick_model(cfg.architect_model, available, _MODEL_FALLBACKS["foreman"])

    logger.info(f"ant     -> {_MODEL_FALLBACKS['ant']}")
    logger.info(f"foreman -> {_MODEL_FALLBACKS['foreman']}")
    logger.info(f"architect -> {_MODEL_FALLBACKS['architect']}")

    # Load manifest
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    tasks = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        logger.error("Manifest empty or invalid")
        sys.exit(1)

    # Dedup by file, keep last instruction
    by_file: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        f = t.get("file")
        if not f:
            continue
        by_file[f] = {"file": f, "instruction": t.get("instruction", "")}
    tasks = list(by_file.values())

    # Start background preload thread (lazy cache still works)
    files = [t["file"] for t in tasks]
    preloader = threading.Thread(target=_preload_worker, args=(files, logger), daemon=True)
    preloader.start()

    # Warm Ant sync (fast, ~5-10s), then start immediately
    logger.info("Warming Ant model...")
    warm_model("ant", _MODEL_FALLBACKS["ant"], logger)
    
    # Warm others in background
    threading.Thread(target=warm_model, args=("foreman", _MODEL_FALLBACKS["foreman"], logger), daemon=True).start()
    threading.Thread(target=warm_model, args=("architect", _MODEL_FALLBACKS["architect"], logger), daemon=True).start()

    # Worker cap to avoid insane queueing
    effective_workers = min(cfg.max_workers, max(1, cfg.ollama_slots * 4))
    logger.info(f"Starting swarm: workers={effective_workers} slots={cfg.ollama_slots} tasks={len(tasks)}")

    report: List[Dict[str, Any]] = []
    ok = 0
    bad = 0
    t_start = time.time()
    
    # Track active files for live status
    active_files: Dict[Any, str] = {}  # future -> filename
    active_lock = threading.Lock()

    try:
        with tqdm(
            total=len(tasks),
            desc="Swarm",
            bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
            ascii=True,  # Use ASCII characters for Windows
        ) as pbar:
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                futs = {ex.submit(process_task, cfg, t, logger): t for t in tasks}
                
                # Update active files display
                with active_lock:
                    for fut, task in futs.items():
                        fname = task["file"].split("/")[-1]  # Just filename
                        active_files[fut] = fname

                for fut in as_completed(futs):
                    r = fut.result()
                    report.append(r)
                    
                    # Get filename
                    fname = r.get("file", "").split("/")[-1]
                    status_icon = "✅" if r.get("status") == "success" else "❌"
                    
                    if r.get("status") == "success":
                        ok += 1
                    else:
                        bad += 1
                    
                    # Remove from active and update display
                    with active_lock:
                        active_files.pop(fut, None)
                        active_list = list(active_files.values())[:3]  # Show max 3 active
                        active_str = ", ".join(active_list) if active_list else "idle"
                    
                    pbar.set_postfix_str(f"{status_icon} {fname[:30]} | ✅ {ok} ❌ {bad} | Active: {active_str}")
                    pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted: saving partial report")
    finally:
        _PRELOAD_STOP.set()

    total = time.time() - t_start
    logger.info(f"done: success={ok} failed={bad} total={total:.1f}s avg={total/max(1,len(tasks)):.1f}s/task")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"report -> {REPORT_PATH}")

    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
