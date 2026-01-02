#!/usr/bin/env python3
"""
Super Turbo Entropy Crusher v5.0 - TIME ATTACK MODE
- Time-based forced escalation (no waiting for failures!)
- 0-15s: Ant direct (temp 0.18)
- 15-30s: Ant higher temp (0.35)
- 30-45s: Architect creates plan
- 45-60s: Ant with plan
- 60-75s: Foreman direct-code
- 75-90s: Architect direct-code
- 90s+: Hard timeout, skip file
- Fast 3s startup (Ant warmup only)
- Live progress with active file tracking
- Thread-safe parallel execution
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
    max_workers: int = 24
    ollama_slots: int = 6

    ant_model: str = "qwen2.5:1.5b"
    foreman_model: str = "qwen2.5:7b"
    architect_model: str = "qwen3:8b"

    # Context / output budgets
    ant_num_ctx: int = 8192
    ant_num_predict: int = 3072

    architect_num_ctx: int = 8192
    architect_num_predict: int = 1024

    foreman_num_ctx: int = 4096
    foreman_num_predict: int = 256

    # Strategy
    ant_temps: Tuple[float, ...] = (0.18, 0.35)
    risky_diff_ratio: float = 0.35  # line-change fraction
    keep_alive: str = "1h"  # Keep models loaded for 1 hour
    max_file_chars_for_full_send: int = 80_000  # very large files get summarized for architect; ant still gets full if under this
    max_preview_chars: int = 800

    # Safety
    block_dangerous_ops: bool = True
    keep_backups: bool = True
    backup_suffix: str = ".swarm_bak"
    safe_write_suffix: str = ".swarm_tmp"
    allow_foreman_direct_fallback: bool = True  # NEW: medium model fallback before architect
    allow_architect_direct_fallback: bool = True

    # Critical files heuristics (foreman gate)
    critical_prefixes: Tuple[str, ...] = ("CAPABILITY/", "PRIMITIVES/", "PIPELINES/", "SCHEMAS/", "MCP/", "TESTBENCH/")

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

_MODEL_FALLBACKS: Dict[str, str] = {}  # "ant"/"foreman"/"architect" -> model name
_MODEL_READY: Dict[str, threading.Event] = {"ant": threading.Event(), "foreman": threading.Event(), "architect": threading.Event()}

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

import subprocess

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
    print(f"DEBUG: run_agent_raw start for {model} (timeout={timeout})", flush=True)
    
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

    try:
        data = json.dumps(payload)
        print(f"DEBUG: calling curl for {model}...", flush=True)
        cmd = ["curl", "-s", "-X", "POST", OLLAMA_API, "-H", "Content-Type: application/json", "-d", data, "--max-time", str(timeout)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        print(f"DEBUG: curl returned code={res.returncode}", flush=True)
        
        if res.returncode != 0:
            return ""
        
        resp_json = json.loads(res.stdout)
        return resp_json.get("response", "") or ""
    except Exception as e:
        print(f"DEBUG: run_agent_raw error: {e}", flush=True)
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

def architect_plan(cfg: SwarmConfig, file_path: str, instruction: str, last_error: str, failed_snippet: str, logger: logging.Logger, forced_timeout: Optional[int] = None) -> Optional[str]:
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

    timeout = forced_timeout if forced_timeout is not None else (35 + min(45, len(content) // 2000))
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

def ant_execute(cfg: SwarmConfig, file_path: str, instruction: str, plan: Optional[str], temperature: float, logger: logging.Logger, forced_timeout: Optional[int] = None) -> str:
    content = get_cached_file(file_path)

    prompt = (
        "[MANDATORY CODING TASK]\n"
        "Return ONLY the complete, fixed Python file in one ```python block.\n"
        "No explanations.\n"
        "Do not introduce unrelated changes.\n"
        "Avoid duplicate imports.\n\n"
        f"TASK: {instruction}\n"
        + (f"\nPLAN:\n{plan}\n" if plan else "\n")
        + f"\nFILE: {file_path}\n"
        "\nCURRENT CONTENT:\n```python\n"
        f"{content}\n"
        "```\n"
    )

    timeout = forced_timeout if forced_timeout is not None else (55 + min(75, len(content) // 1200))
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

def foreman_direct_fix(cfg: SwarmConfig, file_path: str, instruction: str, plan: str, last_error: str, failed_snippet: str, logger: logging.Logger, forced_timeout: Optional[int] = None) -> str:
    original = get_cached_file(file_path)

    prompt = (
        "[FOREMAN: DIRECT FIX]\n"
        "You are a mid-level engineer. Generate the COMPLETE corrected Python file.\n"
        "Follow the plan strictly. Keep changes minimal.\n"
        "Avoid duplicate imports.\n"
        "Return ONLY the full file in one ```python block. No extra text.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        f"PLAN:\n{plan}\n\n"
        + (f"LAST_ERROR:\n{last_error}\n\n" if last_error else "")
        + (f"FAILED SNIPPET:\n{failed_snippet}\n\n" if failed_snippet else "")
        + "CURRENT FILE:\n```python\n"
        f"{original}\n"
        "```\n"
    )

    timeout = forced_timeout if forced_timeout is not None else (50 + min(70, len(original) // 1500))
    return run_agent_raw(
        model=_MODEL_FALLBACKS["foreman"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=0.2,
        num_predict=max(cfg.ant_num_predict, 3072),  # Give foreman more budget than ant
        num_ctx=cfg.foreman_num_ctx,
        keep_alive=cfg.keep_alive,
        logger=logger,
    )

def architect_direct_fix(cfg: SwarmConfig, file_path: str, instruction: str, last_error: str, failed_snippet: str, logger: logging.Logger, forced_timeout: Optional[int] = None) -> str:
    original = get_cached_file(file_path)
    summary = summarize_for_architect(original if len(original) <= cfg.max_file_chars_for_full_send else original[:cfg.max_file_chars_for_full_send])

    prompt = (
        "[SENIOR ARCHITECT: DIRECT FIX]\n"
        "Generate the COMPLETE corrected Python file.\n"
        "Keep changes minimal and focused on the task.\n"
        "Avoid duplicate imports.\n"
        "Return ONLY the full file in one ```python block. No extra text.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        + (f"LAST_ERROR:\n{last_error}\n\n" if last_error else "")
        + (f"FAILED SNIPPET:\n{failed_snippet}\n\n" if failed_snippet else "")
        + f"FILE SUMMARY:\n{summary}\n\n"
        "CURRENT FILE:\n```python\n"
        f"{original}\n"
        "```\n"
    )

    timeout = forced_timeout if forced_timeout is not None else (55 + min(75, len(original) // 1200))
    return run_agent_raw(
        model=_MODEL_FALLBACKS["architect"],
        prompt=prompt,
        timeout=int(timeout),
        temperature=0.18,
        num_predict=max(cfg.ant_num_predict, 4096),
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
    """TIME ATTACK MODE: Escalate based on elapsed time, not failures."""
    fp = task.get("file", "")
    fp = task.get("file", "unknown")
    instruction = task.get("instruction", "")
    start = time.monotonic()
    
    # Time boundaries (seconds since start)
    B_ANT1, B_ANT2, B_PLAN, B_ANT_PLAN, B_FOREMAN, B_ARCHITECT, B_TIMEOUT = 30, 60, 90, 120, 150, 180, 210

    res: Dict[str, Any] = {
        "file": fp, "status": "failed", "reason": "unknown", "elapsed": 0.0,
        "model_used": None, "steps": [], "plan": "", "last_error": "",
        "code_preview": "", "diff_line_ratio": None, "critical": is_critical(cfg, fp),
    }

    try:
        target = safe_repo_path(fp)
        original = get_cached_file(fp)
        
        def elapsed() -> float:
            return time.monotonic() - start
        
        def record(step: str, ok: bool, why: str = "", t: float = 0) -> None:
            res["steps"].append({"step": step, "success": ok, "info": why, "time": round(t, 2)})

        def try_code(code: Optional[str], step: str, t0: float) -> Optional[str]:
            if not code:
                record(step, False, "no_code_or_timeout", elapsed() - t0)
                return None
            ok_sanity, why_sanity = enforce_minimal_sanity(original, code)
            if not ok_sanity:
                record(step, False, f"sanity:{why_sanity}", elapsed() - t0)
                return None
            if cfg.block_dangerous_ops and looks_dangerous(code):
                record(step, False, "dangerous_ops", elapsed() - t0)
                return None
            ok_comp, why_comp, _ = compile_check(code)
            if not ok_comp:
                record(step, False, f"compile:{why_comp}", elapsed() - t0)
                return None
            record(step, True, "ok", elapsed() - t0)
            return code

        fixed: Optional[str] = None
        plan: Optional[str] = None
        last_error = ""

        # TIER 1: Ant direct (temp 0.18)
        if fixed is None:
            t0 = elapsed()
            budget = max(2, B_ANT1 - t0)
            logger.info(f"[{fp}] T1 (Ant 0.18) start. Budget: {budget}s")
            resp = ant_execute(cfg, fp, instruction, None, 0.18, logger, forced_timeout=int(budget))
            fixed = try_code(extract_code(resp), "ant_t0.18", t0)

        # TIER 2: Ant direct (temp 0.35)
        if fixed is None:
            t0 = elapsed()
            budget = max(2, B_ANT2 - t0)
            logger.info(f"[{fp}] T2 (Ant 0.35) start. Budget: {budget}s")
            resp = ant_execute(cfg, fp, instruction, None, 0.35, logger, forced_timeout=int(budget))
            fixed = try_code(extract_code(resp), "ant_t0.35", t0)

        # TIER 3: Architect plan
        if fixed is None:
            t0 = elapsed()
            budget = max(2, B_PLAN - t0)
            logger.info(f"[{fp}] T3 (Arch Plan) start. Budget: {budget}s")
            plan = architect_plan(cfg, fp, instruction, last_error, "", logger, forced_timeout=int(budget))
            res["plan"] = plan or ""
            record("architect_plan", bool(plan), "ok" if plan else "timeout", elapsed() - t0)

        # TIER 4: Ant with plan
        if fixed is None and plan:
            t0 = elapsed()
            budget = max(2, B_ANT_PLAN - t0)
            logger.info(f"[{fp}] T4 (Ant Planned) start. Budget: {budget}s")
            resp = ant_execute(cfg, fp, instruction, plan, 0.25, logger, forced_timeout=int(budget))
            fixed = try_code(extract_code(resp), "ant_planned", t0)

        # TIER 5: Foreman direct
        if fixed is None and cfg.allow_foreman_direct_fallback:
            t0 = elapsed()
            budget = max(2, B_FOREMAN - t0)
            logger.info(f"[{fp}] T5 (Foreman Direct) start. Budget: {budget}s")
            resp = foreman_direct_fix(cfg, fp, instruction, plan or "direct_fix", last_error, "", logger, forced_timeout=int(budget))
            fixed = try_code(extract_code(resp), "foreman_direct", t0)

        # TIER 6: Architect direct
        if fixed is None and cfg.allow_architect_direct_fallback:
            t0 = elapsed()
            budget = max(2, B_ARCHITECT - t0)
            logger.info(f"[{fp}] T6 (Arch Direct) start. Budget: {budget}s")
            resp = architect_direct_fix(cfg, fp, instruction, last_error, "", logger, forced_timeout=int(budget))
            fixed = try_code(extract_code(resp), "architect_direct", t0)

        # FINAL WRITE
        if fixed:
            atomic_write(cfg, target, fixed)
            with _CACHE_LOCK:
                _FILE_CACHE[fp] = fixed
            res["status"] = "success"
            res["reason"] = "ok"
            res["diff_line_ratio"] = round(line_change_ratio(original, fixed), 4)
            logger.info(f"[{fp}] SUCCESS")
        else:
            res["reason"] = "timeout" if elapsed() >= B_TIMEOUT else "no_valid_candidate"
            logger.warning(f"[{fp}] FAILED (elapsed: {elapsed():.1f}s)")

    except Exception as e:
        logger.error(f"[{fp}] CRITICAL WORKER ERROR: {e}")
        res["reason"] = f"exception:{e}"

    res["elapsed"] = round(time.monotonic() - start, 2)
    return res

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Super Turbo Entropy Crusher v5.0 - TIME ATTACK MODE")
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
    print("🐜 SUPER TURBO ENTROPY CRUSHER v5.0 - TIME ATTACK".center(70))
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

    # Start instantly but warm models in background
    logger.info("Starting swarm immediately (background warmup)...")
    
    # Start warmups in BG so we don't block
    for tier in ["ant", "foreman", "architect"]:
        t_model = _MODEL_FALLBACKS.get(tier)
        if t_model:
            threading.Thread(target=warm_model, args=(tier, t_model, logger), daemon=True).start()
        # Still set event so we don't DEADLOCK if someone calls wait_model_ready
        # But actually, let's just NOT wait in the tier functions.


    # Sequential run for reliability
    report: List[Dict[str, Any]] = []
    ok = 0
    bad = 0
    t_start = time.time()
    
    logger.info(f"Starting SAFE sequential run for {len(tasks)} tasks...")
    print(f"SAFE MODE: Processing {len(tasks)} tasks sequentially (no warmup)...")

    for i, task in enumerate(tasks):
        fp = task["file"]
        print(f"[{i+1}/{len(tasks)}] Starting {fp}...", flush=True)
        try:
            r = process_task(cfg, task, logger)
            report.append(r)
            if r.get("status") == "success":
                ok += 1
                print(f"  ✅ SUCCESS: {fp}", flush=True)
            else:
                bad += 1
                print(f"  ❌ FAILED: {fp} ({r.get('reason')})", flush=True)
        except Exception as e:
            print(f"  💥 CRASH: {fp} - {e}", flush=True)
            logger.error(f"Sequential crash on {fp}: {e}")

    total = time.time() - t_start
    print(f"\nDONE: ok={ok} bad={bad} total={total:.1f}s")
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0)

if __name__ == "__main__":
    main()
