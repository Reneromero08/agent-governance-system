#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_META_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "CORTEX_META.json"
SECTION_INDEX_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "SECTION_INDEX.json"
DB_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "cortex.db"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _choose_fence(text: str) -> str:
    if "````" in text:
        return "`````"
    if "```" in text:
        return "````"
    return "```"


def compute_ags_01_canon_bytes(project_root: Path) -> bytes:
    canon_dir = project_root / "CANON"
    if not canon_dir.exists():
        raise FileNotFoundError("CANON_DIR_MISSING")

    canon_paths: List[Path] = []
    for p in canon_dir.rglob("*"):
        try:
            if p.is_file():
                canon_paths.append(p)
        except OSError:
            continue

    rels = sorted((p.relative_to(project_root).as_posix() for p in canon_paths))
    out_lines: List[str] = ["# Canon", ""]

    for rel in rels:
        src = project_root / Path(rel)
        if not src.exists():
            continue
        text = _read_text(src).replace("\r\n", "\n").replace("\r", "\n")
        fence = _choose_fence(text)
        out_lines.append(f"## `repo/{rel}`")
        out_lines.append("")
        out_lines.append(fence)
        out_lines.append(text.rstrip("\n"))
        out_lines.append(fence)
        out_lines.append("")

    rendered = "\n".join(out_lines).rstrip() + "\n"
    return rendered.encode("utf-8")


def compute_canon_sha256(project_root: Path) -> str:
    return _sha256_bytes(compute_ags_01_canon_bytes(project_root))


def _git(args: List[str]) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"GIT_ERROR: {' '.join(args)}: {(res.stderr or res.stdout).strip()}")
    return res.stdout.strip()


def _git_status_porcelain() -> List[str]:
    out = _git(["status", "--porcelain"])
    return [line for line in out.splitlines() if line.strip()]


def _read_cortex_meta() -> Optional[Dict[str, Any]]:
    if CORTEX_META_PATH.exists():
        return json.loads(CORTEX_META_PATH.read_text(encoding="utf-8"))
    return None


def _read_db_generated_at() -> Optional[str]:
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE key = ?", ("generated_at",))
            row = cur.fetchone()
            return str(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def _parse_flags(argv: List[str]) -> Tuple[bool, bool, bool]:
    strict = False
    allow_dirty_tracked = False
    want_json = True

    i = 0
    while i < len(argv):
        a = argv[i].strip()
        if a == "--strict":
            strict = True
        elif a == "--allow-dirty-tracked":
            allow_dirty_tracked = True
        elif a == "--json":
            want_json = True
        else:
            raise ValueError(f"UNKNOWN_ARG: {a}")
        i += 1

    return strict, allow_dirty_tracked, want_json


def _stable_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=False) + "\n"


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    reasons: List[str] = []
    error: Optional[str] = None

    try:
        strict, allow_dirty_tracked, _ = _parse_flags(argv)
    except Exception as exc:
        strict = False
        allow_dirty_tracked = False
        error = str(exc)

    branch = None
    head_sha = None
    tracked_dirty = None
    untracked_count = None
    canon_sha256 = None
    cortex_present = None
    cortex_generated_at = None
    cortex_sha256 = None

    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        head_sha = _git(["rev-parse", "HEAD"])

        status_lines = _git_status_porcelain()
        tracked_dirty = any(not line.startswith("??") for line in status_lines)
        untracked_count = sum(1 for line in status_lines if line.startswith("??"))

        canon_sha256 = compute_canon_sha256(PROJECT_ROOT)

        cortex_present = SECTION_INDEX_PATH.exists()
        if cortex_present:
            cortex_sha256 = _sha256_file(SECTION_INDEX_PATH)

        meta = _read_cortex_meta()
        if meta is not None:
            cortex_generated_at = meta.get("generated_at")
            recorded_canon_sha = meta.get("canon_sha256")
            recorded_cortex_sha = meta.get("cortex_sha256")
            if recorded_cortex_sha and cortex_sha256 and recorded_cortex_sha != cortex_sha256:
                reasons.append("CORTEX_SHA_MISMATCH")
            if recorded_canon_sha and canon_sha256 and recorded_canon_sha != canon_sha256:
                reasons.append("CANON_CHANGED_SINCE_CORTEX")
        else:
            reasons.append("CORTEX_META_MISSING")
            raise RuntimeError("CORTEX_META_MISSING")
    except Exception as exc:
        error = error or str(exc)

    if tracked_dirty:
        reasons.append("DIRTY_TRACKED")
    if cortex_present is False:
        reasons.append("CORTEX_MISSING")
    if (untracked_count or 0) > 0:
        reasons.append("UNTRACKED_PRESENT")

    handoff_path = PROJECT_ROOT / "HANDOFF.md"
    if not handoff_path.exists():
        reasons.append("HANDOFF_MISSING")

    blocking = False
    if "CORTEX_MISSING" in reasons:
        blocking = True
    if "CANON_CHANGED_SINCE_CORTEX" in reasons:
        blocking = True
    if "DIRTY_TRACKED" in reasons and not allow_dirty_tracked:
        blocking = True
    if "UNTRACKED_PRESENT" in reasons and strict:
        blocking = True

    if error is not None:
        exit_code = 3
        verdict = "BLOCKED"
        reasons = ["ERROR"] + reasons
    elif blocking:
        exit_code = 2
        verdict = "BLOCKED"
    else:
        exit_code = 0
        verdict = "SAFE"

    out: Dict[str, Any] = {}
    out["git_branch"] = branch
    out["git_head_sha"] = head_sha
    out["tracked_dirty"] = tracked_dirty
    out["untracked_count"] = untracked_count
    out["canon_sha256"] = canon_sha256
    out["cortex_present"] = cortex_present
    out["cortex_generated_at"] = cortex_generated_at
    out["cortex_sha256"] = cortex_sha256
    out["verdict"] = verdict
    out["reasons"] = reasons
    if error is not None:
        out["error"] = error

    sys.stdout.write(_stable_json(out))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
