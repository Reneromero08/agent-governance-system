#!/usr/bin/env python3
"""
🐞 Entropy Hackers: Legend Edition (v3.4 - Tournament) 🐞
---------------------------------------------------------

Cast:
- Alan Turing     (Academic): Correctness > Speed.
- Elliot Alderson (Pragmatist): Passing > Pretty.
- Neo             (Security): Safety > Everything.
- Claude Shannon  (Judge): Tie-breaker, fallback arbiter.

Pipeline (per file):
1) PROPOSALS: Turing, Elliot, Neo each generate a full corrected file (parallel).
2) QUALIFY: Disqualify non-compiling or unsafe candidates.
3) TOURNAMENT: If a test command is available, try each candidate (apply -> test -> revert).
4) SELECT:
   - If any candidate passes tests: winner = smallest diff among passers (tie: Turing > Neo > Elliot).
   - Else fallback to DIFF-BASED DEMOCRACY voting (diffs only), Shannon breaks ties.
5) WRITE: Atomic write + optional backup. Incremental report saves.

Notes:
- Default max_workers=1 because test-based tournaments mutate the worktree (even though we revert).
- Voting reads diffs, not full files, to fix context-truncation issues.

"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Repo root detection
# -----------------------------------------------------------------------------

def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for _ in range(7):
        if (cur / ".git").exists() or (cur / "THOUGHT").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()

DEFAULT_INPUT_REPORT_1 = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
DEFAULT_INPUT_REPORT_2 = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_REPORT.json"
DEFAULT_MANIFEST = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "SQUAD_REPORT.json"

DEFAULT_OLLAMA_API = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_TAGS = "http://localhost:11434/api/tags"


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SquadConfig:
    repo_root: Path = REPO_ROOT

    input_report: Path = DEFAULT_INPUT_REPORT_1
    alt_input_report: Path = DEFAULT_INPUT_REPORT_2
    manifest_path: Path = DEFAULT_MANIFEST
    output_report: Path = DEFAULT_OUTPUT_REPORT

    ollama_api: str = DEFAULT_OLLAMA_API
    ollama_tags: str = DEFAULT_OLLAMA_TAGS
    keep_alive: str = "20m"

    model_turing: str = "qwen2.5:7b"
    model_elliot: str = "qwen2.5-coder:7b"
    model_neo: str = "dolphin3:latest"
    model_shannon: str = "ministral-3:8b"

    # Context and generation
    member_ctx: int = 16384
    judge_ctx: int = 32768
    member_predict: int = 8192
    judge_predict: int = 2048

    # Temperatures
    turing_temp: float = 0.2
    elliot_temp: float = 0.6
    neo_temp: float = 0.2
    shannon_temp: float = 0.1

    # Timeouts (seconds)
    proposal_timeout_s: int = 300
    vote_timeout_s: int = 120
    test_timeout_s: int = 120
    tags_timeout_s: int = 4
    http_timeout_connect_s: int = 3

    # Concurrency
    max_workers: int = 1         # file-level processing
    ollama_slots: int = 4        # concurrent ollama calls per process

    # IO safety
    dry_run: bool = False
    backup: bool = True

    # Prompt shaping
    max_file_chars_member: int = 24_000
    head_chars: int = 6_000
    tail_chars: int = 6_000
    snippet_radius_lines: int = 35

    # Voting (diff-based)
    enable_voting_fallback: bool = True
    democracy_voters: bool = True  # if False: Shannon-only scoring
    max_diff_chars_vote: int = 14_000

    # Guards
    deny_substrings: Tuple[str, ...] = (
        "rm -rf",
        "shutil.rmtree(",
        "os.remove(",
        "os.unlink(",
        "subprocess.run(['rm'",
        "subprocess.run([\"rm\"",
        "subprocess.call(['rm'",
        "subprocess.call([\"rm\"",
    )


# -----------------------------------------------------------------------------
# Flavor quotes (deterministic)
# -----------------------------------------------------------------------------

CHARACTERS = {
    "turing": "Alan Turing",
    "elliot": "Elliot Alderson",
    "neo": "Neo",
    "shannon": "Claude Shannon",
}

_QUOTES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "turing": {
        "warm": (
            "We proceed by reduction.",
            "Evidence first.",
            "If it compiles, we test it.",
        ),
        "propose": (
            "Formalizing the failure.",
            "Deriving the correction.",
        ),
        "judge": (
            "Checking for logical consistency.",
            "Does it halt correctly?",
        ),
    },
    "elliot": {
        "warm": (
            "Don’t romanticize it.",
            "Patch the lie.",
            "Make it pass.",
        ),
        "propose": (
            "Shipping the fix.",
            "Breaking the failure state.",
        ),
        "judge": (
            "Does it work?",
            "I’m not here for pretty.",
        ),
    },
    "neo": {
        "warm": (
            "I can see the code.",
            "There is no spoon.",
            "Trust boundaries matter.",
        ),
        "propose": (
            "Scanning for unsafe ops.",
            "Hardening the edges.",
        ),
        "judge": (
            "Looking for agents in the diff.",
            "Safety is the invariant.",
        ),
    },
    "shannon": {
        "warm": (
            "Compress the noise.",
            "Signal over entropy.",
        ),
        "judge": (
            "Aggregating signals.",
            "Reducing uncertainty.",
        ),
        "tie": (
            "Tie detected. Choosing the lowest-entropy outcome.",
            "Resolving the ambiguity.",
        ),
    },
}


def _pick_quote(who: str, kind: str, key: str) -> Optional[str]:
    bucket = _QUOTES.get(who, {}).get(kind)
    if not bucket:
        return None
    h = hashlib.sha1(f"{who}:{kind}:{key}".encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(bucket)
    return bucket[idx]


def say(logger: logging.Logger, who: str, kind: str, key: str) -> None:
    q = _pick_quote(who, kind, key)
    if q:
        logger.info(f'{CHARACTERS[who]}: "{q}"')


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("EntropyHackers")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger


# -----------------------------------------------------------------------------
# Ollama client (requests, retries, semaphore, per-thread sessions)
# -----------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, cfg: SquadConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._tls = threading.local()
        self._sem = threading.Semaphore(cfg.ollama_slots)

    def _session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is not None:
            return s
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("POST", "GET"),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        self._tls.session = s
        return s

    def tags(self) -> Dict[str, Any]:
        s = self._session()
        r = s.get(self.cfg.ollama_tags, timeout=(self.cfg.http_timeout_connect_s, self.cfg.tags_timeout_s))
        r.raise_for_status()
        return r.json()

    def warm(self, model: str) -> None:
        payload = {
            "model": model,
            "prompt": "warmup",
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": 0.0, "num_predict": 1},
        }
        try:
            self.generate(payload, timeout_s=10)
        except Exception:
            pass

    def generate(self, payload: Dict[str, Any], timeout_s: int) -> str:
        s = self._session()
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                with self._sem:
                    r = s.post(
                        self.cfg.ollama_api,
                        json=payload,
                        timeout=(self.cfg.http_timeout_connect_s, timeout_s),
                    )
                if r.status_code != 200:
                    raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")
                return (r.json().get("response") or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(0.4 + attempt * 0.6)
        raise RuntimeError(f"Ollama call failed after retries: {last_err}")


def validate_models(cfg: SquadConfig, client: OllamaClient, logger: logging.Logger) -> None:
    required = {cfg.model_turing, cfg.model_elliot, cfg.model_neo, cfg.model_shannon}
    tags = client.tags()
    available = {m.get("name") for m in tags.get("models", []) if m.get("name")}
    missing = required - available
    if missing:
        logger.warning(f"Missing Ollama models: {sorted(missing)} (attempting anyway)")
    for m in sorted(required):
        logger.info(f"Warming model: {m}")
        client.warm(m)


# -----------------------------------------------------------------------------
# Safety and IO helpers
# -----------------------------------------------------------------------------

def safe_repo_path(repo_root: Path, rel_path: str) -> Path:
    p = (repo_root / rel_path).resolve()
    rr = repo_root.resolve()
    if rr not in p.parents and p != rr:
        raise ValueError(f"Path escapes repo_root: {rel_path}")
    return p


def atomic_write(path: Path, content: str, backup: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        path.replace(bak)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), suffix=".tmp", delete=False) as f:
        tmp_path = Path(f.name)
        f.write(content)

    os.replace(str(tmp_path), str(path))


def save_results_atomic(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), suffix=".tmp", delete=False) as f:
        tmp_path = Path(f.name)
        json.dump(results, f, indent=2)
    os.replace(str(tmp_path), str(path))


def deny_scan(cfg: SquadConfig, code: str) -> Optional[str]:
    low = code.lower()
    for s in cfg.deny_substrings:
        if s.lower() in low:
            return s
    return None


def extract_code(text: str) -> Optional[str]:
    # Prefer the last python-ish block if multiple exist
    blocks = list(re.finditer(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE))
    if blocks:
        return blocks[-1].group(1).strip()
    # Fallback: if it looks like a Python module
    if "def " in text or "class " in text or "import " in text:
        return text.strip()
    return None


def validate_syntax(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        compile(code, "<candidate>", "exec")
        return True, "ok"
    except Exception as e:
        return False, str(e)


def unified_diff(old: str, new: str, rel: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{rel} (old)",
            tofile=f"{rel} (new)",
            lineterm="",
        )
    )


def diff_stats(diff_text: str) -> Tuple[int, int]:
    adds = 0
    dels = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            adds += 1
        elif line.startswith("-"):
            dels += 1
    return adds, dels


def truncate_at_newline(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    nl = cut.rfind("\n")
    if nl > int(max_chars * 0.9):
        return s[:nl] + "\n\n# ... [TRUNCATED] ...\n"
    return cut + "\n\n# ... [TRUNCATED] ...\n"


def build_evidence_pack(cfg: SquadConfig, content: str, error: str) -> str:
    """
    Context strategy:
    - always include head and tail
    - if error includes "line N", include ±radius lines around N
    """
    content = truncate_at_newline(content, cfg.max_file_chars_member)
    lines = content.splitlines()

    head = content[: cfg.head_chars]
    tail = content[-cfg.tail_chars :] if len(content) > cfg.tail_chars else ""

    m = re.search(r"line\s+(\d+)", error, re.IGNORECASE)
    mid = ""
    if m:
        try:
            ln = int(m.group(1))
            start = max(0, ln - 1 - cfg.snippet_radius_lines)
            end = min(len(lines), ln - 1 + cfg.snippet_radius_lines + 1)
            chunk = "\n".join(lines[start:end])
            mid = f"\n\n# Error context (lines {start+1}-{end}):\n{chunk}\n"
        except Exception:
            mid = ""

    if mid:
        pack = head + "\n\n# ...\n" + mid + "\n# ...\n\n" + tail
    else:
        pack = head + ("\n\n# ...\n\n" + tail if tail else "")

    return truncate_at_newline(pack, cfg.max_file_chars_member)


# -----------------------------------------------------------------------------
# Prompts (keep flavors)
# -----------------------------------------------------------------------------

def turing_prompt(instruction: str, error: str, evidence: str) -> str:
    return (
        "[ALAN TURING]\n"
        "TASK: Generate the CORRECTED Python file.\n"
        "Focus on correctness, edge cases, and best practices.\n"
        "Use the evidence (error trace) to deduce the logical flaw.\n\n"
        f"INSTRUCTION: {instruction}\n"
        f"ERROR:\n{error}\n\n"
        "EVIDENCE:\n```python\n"
        f"{evidence}\n"
        "```\n\n"
        "Output ONLY the FULL corrected file in a ```python block."
    )


def elliot_prompt(instruction: str, error: str, evidence: str) -> str:
    return (
        "[ELLIOT]\n"
        "TASK: Generate the PATCHED Python file.\n"
        "Focus on making it PASS. Minimal effective change.\n"
        "Dirty hacks are allowed if they are contained.\n\n"
        f"INSTRUCTION: {instruction}\n"
        f"ERROR:\n{error}\n\n"
        "EVIDENCE:\n```python\n"
        f"{evidence}\n"
        "```\n\n"
        "Output ONLY the FULL corrected file in a ```python block."
    )


def neo_prompt(instruction: str, error: str, evidence: str) -> str:
    return (
        "[NEO]\n"
        "TASK: Generate the SAFE Python file.\n"
        "Isolate dangerous ops. Validate paths and inputs.\n"
        "Restore system integrity without risky side effects.\n\n"
        f"INSTRUCTION: {instruction}\n"
        f"ERROR:\n{error}\n\n"
        "EVIDENCE:\n```python\n"
        f"{evidence}\n"
        "```\n\n"
        "Output ONLY the FULL corrected file in a ```python block."
    )


def vote_prompt(voter: str, instruction: str, error: str, diffs: Dict[str, str]) -> str:
    """
    Diff-only voting to avoid context trap.
    Must return JSON mapping exactly candidate names to ints.
    """
    names = list(diffs.keys())
    blobs = []
    for name in names:
        d = diffs[name]
        d = truncate_at_newline(d, 12_000)
        blobs.append(f"--- CANDIDATE {name} DIFF ---\n{d}\n")
    blob = "\n".join(blobs)

    return (
        f"[{voter.upper()}]\n"
        "Score each candidate 0-10.\n"
        f"Candidates are: {', '.join(names)}\n"
        "Return JSON mapping EXACTLY those names to integers.\n"
        'Example: {"Turing": 8, "Elliot": 5, "Neo": 9}\n\n'
        f"INSTRUCTION: {instruction}\n"
        f"ERROR:\n{error}\n\n"
        "DIFFS:\n"
        f"{blob}\n"
        "Return ONLY the JSON."
    )


def shannon_score_prompt(instruction: str, error: str, diffs: Dict[str, str]) -> str:
    names = list(diffs.keys())
    blobs = []
    for name in names:
        d = truncate_at_newline(diffs[name], 12_000)
        blobs.append(f"--- CANDIDATE {name} DIFF ---\n{d}\n")
    blob = "\n".join(blobs)

    return (
        "[CLAUDE SHANNON]\n"
        "You are the judge. Assign a confidence score 0-10 to each candidate.\n"
        "Prefer correctness and safety. Penalize risky side effects.\n"
        f"Candidates: {', '.join(names)}\n"
        'Return JSON like {"Turing": 8, "Elliot": 5, "Neo": 9}\n\n'
        f"INSTRUCTION: {instruction}\n"
        f"ERROR:\n{error}\n\n"
        "DIFFS:\n"
        f"{blob}\n"
        "Return ONLY the JSON."
    )


# -----------------------------------------------------------------------------
# Vote parsing (strict)
# -----------------------------------------------------------------------------

def parse_vote_json(text: str, expected_keys: List[str]) -> Optional[Dict[str, int]]:
    # Strip code fences if present
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Find first balanced {...} block
    best = None
    stack = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    cand = cleaned[start : i + 1]
                    best = cand  # keep last complete object
                    start = None

    if not best:
        return None

    # allow single quotes
    fixed = best.replace("'", '"')
    try:
        obj = json.loads(fixed)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    out: Dict[str, int] = {}
    for k in expected_keys:
        v = obj.get(k)
        if not isinstance(v, (int, float)):
            return None
        out[k] = int(v)

    # reject extra keys if they included unknowns
    for k in obj.keys():
        if k not in expected_keys:
            return None

    return out


# -----------------------------------------------------------------------------
# Tournament testing
# -----------------------------------------------------------------------------

WORKTREE_LOCK = threading.Lock()

def run_test_command(cfg: SquadConfig, logger: logging.Logger, test_cmd: Optional[List[str]]) -> Tuple[bool, str]:
    if not test_cmd:
        return True, "no_test_command"

    try:
        p = subprocess.run(
            test_cmd,
            cwd=str(cfg.repo_root),
            capture_output=True,
            text=True,
            timeout=cfg.test_timeout_s,
        )
        ok = (p.returncode == 0)
        tail = (p.stdout or "")[-2000:] + "\n" + (p.stderr or "")[-2000:]
        return ok, tail.strip()
    except subprocess.TimeoutExpired:
        return False, "test_timeout"
    except Exception as e:
        return False, f"test_error: {e}"


def item_test_command(item: Dict[str, Any]) -> Optional[List[str]]:
    """
    Heuristics:
    - if item["test_command"] is a list: use it
    - if item["test_command"] is a string: split basic shell-ish (safe)
    - if item has pytest nodeid keys: build python -m pytest -q <nodeid>
    """
    tc = item.get("test_command")
    if isinstance(tc, list) and all(isinstance(x, str) for x in tc) and tc:
        return tc
    if isinstance(tc, str) and tc.strip():
        # minimal split; no shell=True
        parts = tc.strip().split()
        return parts if parts else None

    nodeid = item.get("pytest_nodeid") or item.get("nodeid") or item.get("pytest")
    if isinstance(nodeid, str) and nodeid.strip():
        return [sys.executable, "-m", "pytest", "-q", nodeid.strip()]

    return None


# -----------------------------------------------------------------------------
# Proposals and selection
# -----------------------------------------------------------------------------

def run_proposals(
    client: OllamaClient,
    cfg: SquadConfig,
    instruction: str,
    error: str,
    evidence: str,
    key: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    def worker(name: str) -> Tuple[str, Optional[str], Optional[str]]:
        try:
            if name == "Turing":
                say(logger, "turing", "propose", key)
                prompt = turing_prompt(instruction, error, evidence)
                model = cfg.model_turing
                temp = cfg.turing_temp
            elif name == "Elliot":
                say(logger, "elliot", "propose", key)
                prompt = elliot_prompt(instruction, error, evidence)
                model = cfg.model_elliot
                temp = cfg.elliot_temp
            else:
                say(logger, "neo", "propose", key)
                prompt = neo_prompt(instruction, error, evidence)
                model = cfg.model_neo
                temp = cfg.neo_temp

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": cfg.keep_alive,
                "options": {
                    "temperature": temp,
                    "num_ctx": cfg.member_ctx,
                    "num_predict": cfg.member_predict,
                },
            }
            raw = client.generate(payload, timeout_s=cfg.proposal_timeout_s)
            code = extract_code(raw)
            if not code:
                return name, None, "no_code_extracted"

            bad = deny_scan(cfg, code)
            if bad:
                return name, None, f"deny_hit: {bad}"

            ok, why = validate_syntax(code)
            if not ok:
                return name, None, f"syntax: {why}"

            return name, code, None
        except Exception as e:
            return name, None, f"error: {e}"

    proposals: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(worker, n): n for n in ("Turing", "Elliot", "Neo")}
        for f in as_completed(futs):
            name, code, err = f.result()
            if code:
                proposals[name] = code
            else:
                logger.warning(f"Proposal {name} failed: {err}")

    return proposals


def vote_on_diffs(
    client: OllamaClient,
    cfg: SquadConfig,
    instruction: str,
    error: str,
    diffs: Dict[str, str],
    key: str,
    logger: logging.Logger,
) -> Dict[str, int]:
    names = list(diffs.keys())
    if not names:
        return {}

    scores = {n: 0 for n in names}

    def voter_call(voter: str) -> Optional[Dict[str, int]]:
        try:
            if voter == "Shannon":
                say(logger, "shannon", "judge", key)
                prompt = shannon_score_prompt(instruction, error, diffs)
                model = cfg.model_shannon
                temp = cfg.shannon_temp
                ctx = cfg.judge_ctx
                predict = cfg.judge_predict
            else:
                who = voter.lower()
                say(logger, who, "judge", key)
                prompt = vote_prompt(voter, instruction, error, diffs)
                if voter == "Turing":
                    model = cfg.model_turing
                elif voter == "Elliot":
                    model = cfg.model_elliot
                elif voter == "Neo":
                    model = cfg.model_neo
                else:
                    model = cfg.model_shannon # fallback

                temp = 0.2
                ctx = cfg.member_ctx
                predict = cfg.judge_predict

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": cfg.keep_alive,
                "options": {
                    "temperature": temp,
                    "num_ctx": ctx,
                    "num_predict": predict,
                },
            }
            raw = client.generate(payload, timeout_s=cfg.vote_timeout_s)
            parsed = parse_vote_json(raw, expected_keys=names)
            return parsed
        except Exception as e:
            logger.warning(f"Vote failed for {voter}: {e}")
            return None

    voters = ["Turing", "Elliot", "Neo", "Shannon"] if cfg.democracy_voters else ["Shannon"]

    with ThreadPoolExecutor(max_workers=len(voters)) as ex:
        futs = [ex.submit(voter_call, v) for v in voters]
        for f in as_completed(futs):
            m = f.result()
            if not m:
                continue
            for k, v in m.items():
                scores[k] += int(v)

    return scores


def priority_order(name: str) -> int:
    # lower is better
    return {"Turing": 0, "Neo": 1, "Elliot": 2}.get(name, 9)


def select_winner_by_passers(passers: List[Tuple[str, str, str]]) -> str:
    """
    passers: (name, diff_text, code)
    winner: smallest diff (adds+dels), tie: persona priority
    """
    scored: List[Tuple[int, int, int, str]] = []
    for (name, diff_text, _code) in passers:
        adds, dels = diff_stats(diff_text)
        scored.append((adds + dels, adds, priority_order(name), name))
    scored.sort()
    return scored[0][3] if scored else passers[0][0]


def process_item(
    cfg: SquadConfig,
    client: OllamaClient,
    item: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    rel = item.get("file") or item.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return {"file": rel or "", "status": "failed", "reason": "missing_file_path"}

    try:
        fpath = safe_repo_path(cfg.repo_root, rel)
    except Exception as e:
        return {"file": rel, "status": "failed", "reason": f"bad_path: {e}"}

    if not fpath.exists():
        return {"file": rel, "status": "failed", "reason": "missing"}

    original = fpath.read_text(encoding="utf-8", errors="replace")
    error = item.get("last_error") or item.get("error") or item.get("reason") or "Unknown"
    instruction = item.get("instruction") or "Fix the bug in this file."

    evidence = build_evidence_pack(cfg, original, error)

    logger.info(f"🐞 {rel}")

    # 1) Proposals
    proposals = run_proposals(client, cfg, instruction, error, evidence, rel, logger)
    if not proposals:
        return {"file": rel, "status": "failed", "reason": "no_valid_proposals"}

    # 2) Build diffs and stats
    diffs = {name: unified_diff(original, code, rel) for name, code in proposals.items()}
    stats = {name: {"adds": diff_stats(diffs[name])[0], "dels": diff_stats(diffs[name])[1]} for name in proposals}

    # 3) Tournament (if we have a test command)
    test_cmd = item_test_command(item)
    passers: List[Tuple[str, str, str]] = []
    test_logs: Dict[str, str] = {}
    test_ok_map: Dict[str, bool] = {}

    if test_cmd:
        logger.info(f"  - Tournament: {' '.join(test_cmd)}")
        # Serialize worktree mutation even if user increases max_workers elsewhere
        with WORKTREE_LOCK:
            for name in ("Turing", "Elliot", "Neo"):
                if name not in proposals:
                    continue
                cand = proposals[name]

                if cfg.dry_run:
                    # In dry-run, we cannot really test meaningfully. Mark unknown.
                    test_ok_map[name] = False
                    test_logs[name] = "dry_run"
                    continue

                # Apply candidate
                fpath.write_text(cand, encoding="utf-8")
                ok, log = run_test_command(cfg, logger, test_cmd)
                test_ok_map[name] = ok
                test_logs[name] = log

                # Revert immediately (unless it's the only pass and we choose early)
                fpath.write_text(original, encoding="utf-8")

                if ok:
                    passers.append((name, diffs[name], cand))

            # Restore original (belt and suspenders)
            fpath.write_text(original, encoding="utf-8")

    winner: Optional[str] = None
    selection_reason = ""

    if passers:
        winner = select_winner_by_passers(passers)
        selection_reason = "tests_passed"
        logger.info(f"  - Passers: {[p[0] for p in passers]} -> Winner: {winner}")

    # 4) Fallback: diff-based voting
    if not winner:
        if not cfg.enable_voting_fallback:
            # deterministic fallback
            winner = min(proposals.keys(), key=priority_order)
            selection_reason = "no_tests_no_vote_fallback"
        else:
            scores = vote_on_diffs(client, cfg, instruction, error, diffs, rel, logger)
            if not scores:
                winner = min(proposals.keys(), key=priority_order)
                selection_reason = "vote_failed_priority_fallback"
            else:
                max_score = max(scores.values())
                leaders = [k for k, v in scores.items() if v == max_score]
                if len(leaders) > 1:
                    say(logger, "shannon", "tie", rel)
                    leaders.sort(key=priority_order)
                winner = leaders[0]
                selection_reason = "vote"
            logger.info(f"  - Winner: {winner} ({selection_reason})")

    final_code = proposals[winner]

    # 5) Final safety check before write
    bad = deny_scan(cfg, final_code)
    if bad:
        return {"file": rel, "status": "failed", "reason": f"deny_hit_final: {bad}", "winner": winner}

    ok, why = validate_syntax(final_code)
    if not ok:
        return {"file": rel, "status": "failed", "reason": f"final_syntax_invalid: {why}", "winner": winner}

    # 6) Write (atomic)
    if not cfg.dry_run:
        atomic_write(fpath, final_code, backup=cfg.backup)

    d = diffs.get(winner, "")
    adds, dels = diff_stats(d)

    out: Dict[str, Any] = {
        "file": rel,
        "status": "fixed",
        "winner": winner,
        "reason": selection_reason,
        "diff_adds": adds,
        "diff_dels": dels,
        "candidates": list(proposals.keys()),
        "candidate_stats": stats,
    }

    if test_cmd:
        out["test_command"] = test_cmd
        out["test_ok"] = test_ok_map.get(winner, False) if not cfg.dry_run else None
        out["test_logs_tail"] = test_logs.get(winner, "")

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def load_report(cfg: SquadConfig, logger: logging.Logger) -> List[Dict[str, Any]]:
    if cfg.input_report.exists():
        return json.loads(cfg.input_report.read_text(encoding="utf-8"))
    if cfg.alt_input_report.exists():
        logger.warning(f"Primary report missing, using alt: {cfg.alt_input_report}")
        return json.loads(cfg.alt_input_report.read_text(encoding="utf-8"))
    raise SystemExit(f"Missing input report: {cfg.input_report} (and alt {cfg.alt_input_report})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Entropy Hackers: Legend Edition (v3.4)")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT_REPORT_1))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_REPORT))
    ap.add_argument("--model-turing", type=str, default="qwen3:8b")
    ap.add_argument("--model-elliot", type=str, default="qwen2.5-coder:7b")
    ap.add_argument("--model-neo", type=str, default="dolphin3:8b")
    ap.add_argument("--model-shannon", type=str, default="ministral-3:8b")
    ap.add_argument("--ollama-api", type=str, default=DEFAULT_OLLAMA_API)
    ap.add_argument("--ollama-tags", type=str, default=DEFAULT_OLLAMA_TAGS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--ollama-slots", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--no-vote", action="store_true")
    ap.add_argument("--shannon-only", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-every", type=int, default=5, help="save results every N items")
    args = ap.parse_args()

    logger = setup_logging(args.debug)

    cfg = SquadConfig(
        input_report=Path(args.input),
        output_report=Path(args.output),
        model_turing=args.model_turing,
        model_elliot=args.model_elliot,
        model_neo=args.model_neo,
        model_shannon=args.model_shannon,
        ollama_api=args.ollama_api,
        ollama_tags=args.ollama_tags,
        dry_run=args.dry_run,
        backup=(not args.no_backup),
        ollama_slots=max(1, args.ollama_slots),
        max_workers=max(1, args.max_workers),
        enable_voting_fallback=(not args.no_vote),
        democracy_voters=(not args.shannon_only),
    )

    # Default behavior: tests mutate worktree, so parallel file processing is risky.
    # We allow max_workers > 1, but tournament tests are still serialized by WORKTREE_LOCK.
    if cfg.max_workers > 1:
        logger.warning("max_workers > 1 enabled. Worktree mutations are serialized; speedup may be limited.")

    client = OllamaClient(cfg, logger)

    say(logger, "shannon", "warm", "startup")
    validate_models(cfg, client, logger)

    items = load_report(cfg, logger)

    # If report includes both successes and failures, focus on failures.
    failures = [x for x in items if x.get("status") not in ("success", "passed", "ok")]
    work = failures if failures else items

    logger.info(f"Processing {len(work)} items")

    results: List[Dict[str, Any]] = []
    save_every = max(1, int(args.save_every))

    def process_one(item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return process_item(cfg, client, item, logger)
        except Exception as e:
            rel = item.get("file") or item.get("path") or ""
            return {"file": rel, "status": "failed", "reason": f"exception: {e}"}

    if cfg.max_workers == 1:
        for i, item in enumerate(tqdm(work, desc="Hackers", ncols=100)):
            r = process_one(item)
            results.append(r)
            if (i + 1) % save_every == 0:
                save_results_atomic(cfg.output_report, results)
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(process_one, it) for it in work]
            for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="Hackers", ncols=100)):
                results.append(f.result())
                if (i + 1) % save_every == 0:
                    save_results_atomic(cfg.output_report, results)

    save_results_atomic(cfg.output_report, results)

    fixed = sum(1 for r in results if r.get("status") == "fixed")
    failed = sum(1 for r in results if r.get("status") != "fixed")
    logger.info(f"Done. Fixed: {fixed}, Failed: {failed}")
    logger.info(f"Report: {cfg.output_report}")


if __name__ == "__main__":
    main()
