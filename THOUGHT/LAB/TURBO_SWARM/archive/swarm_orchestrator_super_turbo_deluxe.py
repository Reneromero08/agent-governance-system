from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
REPORT_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_PATH = REPO_ROOT / "swarm_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("TurboSwarm")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SwarmConfig:
    max_workers: int = 24
    ollama_slots: int = 6

    ant_model: str = "qwen2.5:1.5b"
    foreman_model: str = "qwen2.5:7b"
    architect_model: str = "qwen3:8b"

    keep_alive: str = "5m"

    # Generation budgets
    ant_num_ctx: int = 4096
    ant_num_predict: int = 2048
    architect_num_ctx: int = 6144
    architect_num_predict: int = 800
    foreman_num_ctx: int = 2048
    foreman_num_predict: int = 256

    # Ant retry temps (cheap success-rate bump)
    ant_temps: Tuple[float, ...] = (0.18, 0.35)

    # Heuristics
    max_file_chars: int = 200_000
    risky_diff_ratio: float = 0.35

    # Foreman verify is used only when risky or critical
    critical_prefixes: Tuple[str, ...] = (
        "CAPABILITY/PRIMITIVES/",
        "CAPABILITY/PIPELINES/",
        "MCP/",
        "VALIDATOR/",
        "TESTBENCH/",
    )

# -----------------------------------------------------------------------------
# Globals (thread-safe usage)
# -----------------------------------------------------------------------------

OLLAMA_SEM: Optional[threading.Semaphore] = None
MODEL_FALLBACKS: Dict[str, str] = {}

FILE_CACHE: Dict[str, str] = {}
FILE_LOCK = threading.Lock()

MODEL_READY: Dict[str, threading.Event] = {}
_MODEL_WARM_LOCK = threading.Lock()
_MODEL_WARM_STARTED: set[str] = set()

_THREAD = threading.local()

# -----------------------------------------------------------------------------
# HTTP session per thread (avoid shared Session race)
# -----------------------------------------------------------------------------

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=0,  # we do our own per-call retries
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def get_session() -> requests.Session:
    s = getattr(_THREAD, "session", None)
    if s is None:
        s = _build_session()
        _THREAD.session = s
    return s

# -----------------------------------------------------------------------------
# Repo path guard
# -----------------------------------------------------------------------------

def safe_repo_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        raise ValueError("absolute_path_not_allowed")
    candidate = (REPO_ROOT / p).resolve()
    root = REPO_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except Exception as e:
        raise ValueError("path_escapes_repo_root") from e
    return candidate

# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------

def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest_not_found:{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest_must_be_list")
    out: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        fp = item.get("file")
        ins = item.get("instruction")
        if isinstance(fp, str) and isinstance(ins, str) and fp.strip() and ins.strip():
            out.append({"file": fp.strip(), "instruction": ins.strip()})
    return out

# -----------------------------------------------------------------------------
# File cache (lazy + optional background preloader)
# -----------------------------------------------------------------------------

def get_cached_file(file_path: str) -> str:
    with FILE_LOCK:
        if file_path in FILE_CACHE:
            return FILE_CACHE[file_path]

    # Load outside lock (avoid blocking other threads)
    try:
        full = safe_repo_path(file_path)
        txt = full.read_text(encoding="utf-8", errors="replace") if full.exists() else ""
    except Exception as e:
        logger.warning(f"read_failed {file_path}: {type(e).__name__}: {e}")
        txt = ""

    with FILE_LOCK:
        FILE_CACHE[file_path] = txt
    return txt

def preload_files_background(tasks: List[Dict[str, str]]) -> None:
    def _run():
        logger.info("📂 Background file preload started")
        for t in tasks:
            fp = t["file"]
            with FILE_LOCK:
                if fp in FILE_CACHE:
                    continue
            _ = get_cached_file(fp)
        logger.info("📂 Background file preload done")
    threading.Thread(target=_run, daemon=True).start()

# -----------------------------------------------------------------------------
# Model discovery + warmup
# -----------------------------------------------------------------------------

def resolve_model_fallbacks(cfg: SwarmConfig) -> Dict[str, str]:
    sess = get_session()
    r = sess.get(OLLAMA_TAGS, timeout=4)
    r.raise_for_status()
    available = {m["name"] for m in r.json().get("models", [])}

    def pick(wanted: str, ultimate: str) -> str:
        if wanted in available:
            return wanted
        base = wanted.split(":")[0]
        family = sorted([m for m in available if m.startswith(base + ":")])
        if family:
            return family[0]
        return ultimate

    return {
        "ant": pick(cfg.ant_model, cfg.foreman_model),
        "foreman": pick(cfg.foreman_model, cfg.foreman_model),
        "architect": pick(cfg.architect_model, cfg.foreman_model),
    }

def wait_model_ready(tier: str, timeout: float = 0.0) -> None:
    ev = MODEL_READY.get(tier)
    if not ev:
        return
    if timeout and timeout > 0:
        ev.wait(timeout=timeout)

def warm_model_once(cfg: SwarmConfig, tier: str) -> None:
    """
    One-shot warmup for a tier. Sets MODEL_READY[tier] when done (success or fail).
    Safe under concurrency.
    """
    ev = MODEL_READY.setdefault(tier, threading.Event)

    with _MODEL_WARM_LOCK:
        if tier in _MODEL_WARM_STARTED:
            return
        _MODEL_WARM_STARTED.add(tier)

    try:
        _ = run_agent_raw(
            model=MODEL_FALLBACKS[tier],
            prompt="hi",
            timeout=25,
            temperature=0.1,
            num_predict=1,
            num_ctx=1024,
            keep_alive=cfg.keep_alive,
        )
    finally:
        ev.set()

def warm_in_background(cfg: SwarmConfig, tiers: List[str]) -> None:
    def _bg(t: str) -> None:
        try:
            warm_model_once(cfg, t)
        except Exception as e:
            logger.warning(f"warmup_failed {t}: {type(e).__name__}: {e}")
            MODEL_READY.setdefault(t, threading.Event).set()
    for t in tiers:
        threading.Thread(target=_bg, args=(t,), daemon=True).start()

# -----------------------------------------------------------------------------
# Ollama call
# -----------------------------------------------------------------------------

def run_agent_raw(
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    num_predict: int,
    num_ctx: int,
    keep_alive: str,
) -> str:
    sess = get_session()
    max_retries = 2

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
        },
    }

    for attempt in range(max_retries + 1):
        try:
            assert OLLAMA_SEM is not None
            with OLLAMA_SEM:
                r = sess.post(OLLAMA_API, json=payload, timeout=timeout)

            if r.status_code == 200:
                return r.json().get("response", "") or ""

            if r.status_code in (502, 503, 504) and attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue

            logger.error(f"LLM_HTTP_{r.status_code}: {r.text[:250]}")
            return ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                time.sleep(0.8 * (2 ** attempt))
                continue
            logger.error(f"LLM_CONN: {type(e).__name__}: {e}")
            return ""
        except Exception as e:
            logger.exception(f"LLM_ERR: {type(e).__name__}: {e}")
            return ""
    return ""

# -----------------------------------------------------------------------------
# Extraction + verify + atomic write
# -----------------------------------------------------------------------------

_CODEBLOCK_RE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(response: str) -> Optional[str]:
    if not response:
        return None
    blocks = _CODEBLOCK_RE.findall(response)
    if blocks:
        return max((b.strip() for b in blocks), key=len, default=None)
    s = response.strip()
    if s.startswith("import ") or s.startswith("from ") or ("def " in s) or ("class " in s):
        return s
    return None

def verify_code(code: str) -> Tuple[bool, str, Optional[int]]:
    if not code or len(code) < 50:
        return False, "empty_or_too_short", None
    if "```" in code:
        return False, "contains_fence", None
    try:
        ast.parse(code)
        return True, "ok", None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})", e.lineno
    except Exception as e:
        return False, f"ParseError: {type(e).__name__}", None

def atomic_write(target_path: Path, code: str) -> None:
    tmp = target_path.with_suffix(target_path.suffix + f".swarm_tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(code, encoding="utf-8")
    os.replace(tmp, target_path)

def diff_ratio(original: str, fixed: str) -> float:
    if not original:
        return 1.0
    return abs(len(fixed) - len(original)) / max(1, len(original))

def is_critical(cfg: SwarmConfig, file_path: str) -> bool:
    return file_path.startswith(cfg.critical_prefixes)

def snippet_around_line(src: str, lineno: Optional[int], radius: int = 12) -> str:
    if lineno is None:
        return ""
    lines = src.splitlines()
    if not lines:
        return ""
    i = max(0, min(len(lines) - 1, lineno - 1))
    a = max(0, i - radius)
    b = min(len(lines), i + radius + 1)
    out: List[str] = []
    for idx in range(a, b):
        mark = ">>" if idx == i else "  "
        out.append(f"{mark} {idx+1:4d} | {lines[idx]}")
    return "\n".join(out)

# -----------------------------------------------------------------------------
# Smart context for Architect
# -----------------------------------------------------------------------------

def summarize_for_architect(src: str) -> str:
    imports: List[str] = []
    defs: List[str] = []
    classes: List[str] = []
    lines = src.splitlines()

    try:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if 1 <= node.lineno <= len(lines):
                    imports.append(lines[node.lineno - 1].strip())
            elif isinstance(node, ast.FunctionDef):
                defs.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except Exception:
        pass

    head = src[:700]
    tail = src[-700:] if len(src) > 700 else ""
    return (
        "IMPORTS:\n" + ("\n".join(imports[:60]) if imports else "(none)") + "\n\n"
        "CLASSES:\n" + (", ".join(classes[:80]) if classes else "(none)") + "\n\n"
        "FUNCS:\n" + (", ".join(defs[:160]) if defs else "(none)") + "\n\n"
        "HEAD:\n" + head + "\n\n"
        "TAIL:\n" + tail
    )

# -----------------------------------------------------------------------------
# Tiered logic
# -----------------------------------------------------------------------------

def architect_plan(cfg: SwarmConfig, file_path: str, instruction: str, last_error: str, failed_snippet: str) -> Optional[str]:
    wait_model_ready("architect", timeout=2.0)

    original = get_cached_file(file_path)
    summary = summarize_for_architect(original)

    prompt = (
        "Create a minimal fix plan for the task.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        f"LAST FAILURE:\n{last_error or '(none)'}\n\n"
        + (f"FAILED ATTEMPT SNIPPET (around error):\n{failed_snippet}\n\n" if failed_snippet else "")
        + f"FILE SUMMARY:\n{summary}\n\n"
        "OUTPUT: bullet points only. No code. No backticks."
    )

    resp = run_agent_raw(
        model=MODEL_FALLBACKS["architect"],
        prompt=prompt,
        timeout=45,
        temperature=0.18,
        num_predict=cfg.architect_num_predict,
        num_ctx=cfg.architect_num_ctx,
        keep_alive=cfg.keep_alive,
    ).strip()

    if not resp or "```" in resp or len(resp) < 20:
        return None
    return resp

def ant_execute(cfg: SwarmConfig, file_path: str, instruction: str, plan: str, last_error: str, temperature: float) -> str:
    content = get_cached_file(file_path)

    # Dynamic timeout based on file size (cap)
    timeout = 70 + min(70, max(0, len(content) // 2500))

    prompt = (
        "[CRITICAL FIX TASK]\n"
        "Apply the task to the file.\n\n"
        f"TASK: {instruction}\n\n"
        + (f"PLAN:\n{plan}\n\n" if plan else "")
        + (f"LAST_ERROR:\n{last_error}\n\n" if last_error else "")
        + "CURRENT FILE:\n```python\n"
        f"{content}\n"
        "```\n\n"
        "RULES:\n"
        "1. Preserve existing imports and structure. Add imports ONLY if required.\n"
        "2. Avoid unrelated refactors.\n"
        "3. Output ONLY the complete fixed file in a single ```python block.\n"
        "4. No extra text.\n"
    )

    return run_agent_raw(
        model=MODEL_FALLBACKS["ant"],
        prompt=prompt,
        timeout=timeout,
        temperature=temperature,
        num_predict=cfg.ant_num_predict,
        num_ctx=cfg.ant_num_ctx,
        keep_alive=cfg.keep_alive,
    )

def foreman_verify(cfg: SwarmConfig, file_path: str, instruction: str, original: str, fixed: str) -> Tuple[bool, str]:
    wait_model_ready("foreman", timeout=2.0)

    o = (original[:900] + "\n...\n" + original[-450:]) if len(original) > 1500 else original
    f = (fixed[:900] + "\n...\n" + fixed[-450:]) if len(fixed) > 1500 else fixed

    prompt = (
        "Strict code reviewer.\n"
        "Decide if FIXED correctly implements TASK with no unrelated changes.\n\n"
        f"TASK: {instruction}\n"
        f"FILE: {file_path}\n\n"
        f"ORIGINAL (snippet):\n{o}\n\n"
        f"FIXED (snippet):\n{f}\n\n"
        'Answer ONLY "YES" or "NO: <short reason>".'
    )

    resp = run_agent_raw(
        model=MODEL_FALLBACKS["foreman"],
        prompt=prompt,
        timeout=35,
        temperature=0.1,
        num_predict=cfg.foreman_num_predict,
        num_ctx=cfg.foreman_num_ctx,
        keep_alive=cfg.keep_alive,
    ).strip()

    if resp.startswith("YES"):
        return True, "approved"
    if resp.startswith("NO"):
        return False, resp[:160]
    return False, "unclear_review"

# -----------------------------------------------------------------------------
# Task processing
# -----------------------------------------------------------------------------

def process_task(cfg: SwarmConfig, task: Dict[str, str]) -> Dict[str, Any]:
    fp = task["file"]
    instruction = task["instruction"]
    start = time.monotonic()

    res: Dict[str, Any] = {
        "file": fp,
        "status": "failed",
        "reason": "unknown",
        "elapsed": 0.0,
        "model_used": None,
        "steps": [],
    }

    try:
        target_path = safe_repo_path(fp)
    except Exception as e:
        res["reason"] = f"bad_path:{type(e).__name__}"
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    original = get_cached_file(fp)
    if len(original) > cfg.max_file_chars:
        res["reason"] = "file_too_large"
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    last_error = ""
    failed_snippet = ""

    # Step 1: Ant direct retries
    for idx, temp in enumerate(cfg.ant_temps, start=1):
        t0 = time.monotonic()
        resp = ant_execute(cfg, fp, instruction, plan="", last_error=last_error, temperature=temp)
        fixed = extract_code(resp)

        ok = False
        why = "no_code"
        lineno = None
        if fixed:
            ok, why, lineno = verify_code(fixed)
            if not ok:
                last_error = why
                failed_snippet = snippet_around_line(fixed, lineno)

        res["steps"].append({
            "step": f"ant_direct_{idx}",
            "temp": temp,
            "got_code": bool(fixed),
            "verify": why,
            "time": round(time.monotonic() - t0, 2),
        })

        if fixed and ok:
            risky = is_critical(cfg, fp) or (diff_ratio(original, fixed) > cfg.risky_diff_ratio)
            if risky:
                approved, note = foreman_verify(cfg, fp, instruction, original, fixed)
                res["steps"].append({"step": "foreman_verify", "success": approved, "note": note})
                if not approved:
                    last_error = f"ForemanReject: {note}"
                    continue

            try:
                atomic_write(target_path, fixed)
                with FILE_LOCK:
                    FILE_CACHE[fp] = fixed
                res.update({
                    "status": "success",
                    "reason": "ant_direct_success",
                    "model_used": MODEL_FALLBACKS["ant"],
                })
                res["elapsed"] = round(time.monotonic() - start, 2)
                return res
            except Exception as e:
                res["reason"] = f"write_error:{type(e).__name__}"
                res["steps"].append({"step": "write", "success": False, "note": f"{type(e).__name__}:{e}"})
                res["elapsed"] = round(time.monotonic() - start, 2)
                return res

    # Step 2: Architect plan with feedback
    t0 = time.monotonic()
    plan = architect_plan(cfg, fp, instruction, last_error=last_error, failed_snippet=failed_snippet)
    res["steps"].append({"step": "architect_plan", "success": bool(plan), "time": round(time.monotonic() - t0, 2)})

    if not plan:
        res["reason"] = "invalid_plan"
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    # Step 3: Ant with plan
    t0 = time.monotonic()
    resp = ant_execute(cfg, fp, instruction, plan=plan, last_error=last_error, temperature=0.22)
    fixed = extract_code(resp)

    if not fixed:
        res["reason"] = "no_code_generated"
        res["steps"].append({"step": "ant_planned", "success": False, "verify": "no_code", "time": round(time.monotonic() - t0, 2)})
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    ok, why, _lineno = verify_code(fixed)
    res["steps"].append({"step": "ant_planned", "success": ok, "verify": why, "time": round(time.monotonic() - t0, 2)})

    if not ok:
        res["reason"] = f"verification_failed:{why}"
        res["elapsed"] = round(time.monotonic() - start, 2)
        return res

    risky = is_critical(cfg, fp) or (diff_ratio(original, fixed) > cfg.risky_diff_ratio)
    if risky:
        approved, note = foreman_verify(cfg, fp, instruction, original, fixed)
        res["steps"].append({"step": "foreman_verify", "success": approved, "note": note})
        if not approved:
            res["reason"] = "foreman_reject"
            res["elapsed"] = round(time.monotonic() - start, 2)
            return res

    try:
        atomic_write(target_path, fixed)
        with FILE_LOCK:
            FILE_CACHE[fp] = fixed
        res.update({
            "status": "success",
            "reason": "planned_ant_success",
            "model_used": MODEL_FALLBACKS["ant"],
        })
    except Exception as e:
        res["reason"] = f"write_error:{type(e).__name__}"
        res["steps"].append({"step": "write", "success": False, "note": f"{type(e).__name__}:{e}"})

    res["elapsed"] = round(time.monotonic() - start, 2)
    return res

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Turbo Swarm v4.3 (Ant-first immediate start)")
    parser.add_argument("--max-workers", type=int, default=SwarmConfig.max_workers)
    parser.add_argument("--ollama-slots", type=int, default=SwarmConfig.ollama_slots)
    parser.add_argument("--ant-model", type=str, default=SwarmConfig.ant_model)
    parser.add_argument("--foreman-model", type=str, default=SwarmConfig.foreman_model)
    parser.add_argument("--architect-model", type=str, default=SwarmConfig.architect_model)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    cfg = SwarmConfig(
        max_workers=int(args.max_workers),
        ollama_slots=int(args.ollama_slots),
        ant_model=str(args.ant_model),
        foreman_model=str(args.foreman_model),
        architect_model=str(args.architect_model),
    )

    global OLLAMA_SEM, MODEL_FALLBACKS, MODEL_READY
    OLLAMA_SEM = threading.Semaphore(cfg.ollama_slots)

    try:
        tasks = load_manifest(MANIFEST_PATH)
    except Exception as e:
        logger.error(f"manifest_load_failed: {type(e).__name__}: {e}")
        raise SystemExit(1)

    if not tasks:
        logger.info("No tasks to process.")
        return

    # Cap workers to reduce pointless queueing
    effective_workers = min(cfg.max_workers, cfg.ollama_slots * 4)
    logger.info(f"workers={effective_workers} slots={cfg.ollama_slots} tasks={len(tasks)}")

    # Model discovery
    try:
        MODEL_FALLBACKS = resolve_model_fallbacks(cfg)
    except Exception as e:
        logger.error(f"ollama_tags_failed: {type(e).__name__}: {e}")
        raise SystemExit(1)

    logger.info(f"models={MODEL_FALLBACKS}")

    # Ready events
    MODEL_READY = {
        "ant": threading.Event(),
        "foreman": threading.Event(),
        "architect": threading.Event(),
    }

    # Warm Ant synchronously, start immediately after
    warm_model_once(cfg, "ant")
    logger.info("✅ Ant warmed. Starting swarm now.")

    # Warm others in background
    warm_in_background(cfg, ["foreman", "architect"])

    # Start background file preload (non-blocking)
    preload_files_background(tasks)

    report: List[Dict[str, Any]] = []
    success = 0
    failed = 0
    t0 = time.time()

    with tqdm(
        total=len(tasks),
        desc="🐜 Swarm",
        bar_format="{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
        colour="green",
    ) as pbar:
        try:
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                futures = {ex.submit(process_task, cfg, t): t for t in tasks}
                for fut in as_completed(futures):
                    r = fut.result()
                    report.append(r)
                    if r.get("status") == "success":
                        success += 1
                    else:
                        failed += 1
                    pbar.set_postfix_str(f"✅ {success} ❌ {failed}")
                    pbar.update(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted. Saving partial report.")

    total = time.time() - t0
    logger.info(f"done success={success} failed={failed} total={total:.1f}s avg={total/max(1,len(tasks)):.1f}s")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(1 if failed else 0)

if __name__ == "__main__":
    main()
