#!/usr/bin/env python3

from __future__ import annotations

import json
import argparse
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple


POLICY_VERSION = "1"

# Default artifacts root (repo-relative POSIX path).
ARTIFACTS_ROOT = "CONTRACTS/_runs"

_VALID_MODES = {"read-only", "artifact-only", "repo-write"}


def _stable_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=False) + "\n"


def _is_windows_abs(raw: str) -> bool:
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return True
    return False


def normalize_repo_relpath(raw: str) -> str:
    if raw is None:
        raise ValueError("PATH_INVALID")
    value = str(raw).strip()
    if not value:
        raise ValueError("PATH_INVALID")

    value = value.replace("\\", "/")
    if value.startswith("/"):
        raise ValueError("PATH_INVALID")
    if _is_windows_abs(value):
        raise ValueError("PATH_INVALID")

    p = PurePosixPath(value)
    if p.is_absolute():
        raise ValueError("PATH_INVALID")

    # Normalize (collapse '.'), but reject any traversal.
    parts: List[str] = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("PATH_INVALID")
        parts.append(part)

    if not parts:
        raise ValueError("PATH_INVALID")

    norm = PurePosixPath(*parts).as_posix()
    if ".." in PurePosixPath(norm).parts:
        raise ValueError("PATH_INVALID")
    return norm


def _sort_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out)


def _normalize_paths(paths_obj: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(paths_obj, dict):
        raise ValueError("PATH_INVALID")

    read_raw = paths_obj.get("read", [])
    write_raw = paths_obj.get("write", [])
    if not isinstance(read_raw, list) or not isinstance(write_raw, list):
        raise ValueError("PATH_INVALID")

    read_norm: List[str] = []
    write_norm: List[str] = []

    for item in read_raw:
        read_norm.append(normalize_repo_relpath(item))
    for item in write_raw:
        write_norm.append(normalize_repo_relpath(item))

    return _sort_unique(read_norm), _sort_unique(write_norm)


def _is_under(root: str, rel: str) -> bool:
    root_norm = normalize_repo_relpath(root)
    rel_norm = normalize_repo_relpath(rel)
    if rel_norm == root_norm:
        return True
    return rel_norm.startswith(root_norm + "/")


def admit_intent(intent: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    reasons: List[str] = []
    warnings: List[str] = []

    if not isinstance(intent, dict):
        return 3, {
            "verdict": "BLOCK",
            "reasons": ["INTENT_INVALID"],
            "normalized_paths": {"read": [], "write": []},
            "policy_version": POLICY_VERSION,
        }

    mode = intent.get("mode")
    if mode not in _VALID_MODES:
        reasons.append("MODE_INVALID")

    allow_repo_write = bool(intent.get("allow_repo_write", False))
    max_writes = intent.get("max_writes", None)
    if max_writes is not None:
        try:
            max_writes_int = int(max_writes)
            if max_writes_int < 0:
                raise ValueError()
            max_writes = max_writes_int
        except Exception:
            reasons.append("LIMIT_INVALID")
            max_writes = None

    paths_obj = intent.get("paths")
    try:
        read_norm, write_norm = _normalize_paths(paths_obj)
    except Exception:
        read_norm, write_norm = [], []
        reasons.append("PATH_INVALID")

    if max_writes is not None and len(write_norm) > int(max_writes):
        reasons.append("MAX_WRITES_EXCEEDED")

    # Policy rules
    if mode == "read-only":
        if write_norm:
            reasons.append("WRITE_NOT_ALLOWED")

    if mode == "artifact-only":
        for w in write_norm:
            if not _is_under(ARTIFACTS_ROOT, w):
                reasons.append("WRITE_OUTSIDE_ARTIFACTS")
            else:
                # Warn if writing to an untracked artifact location isn't available here.
                pass
        # If write is outside artifacts, it also touches repo.
        if any(not _is_under(ARTIFACTS_ROOT, w) for w in write_norm):
            reasons.append("WRITE_TOUCHES_REPO")

    if mode == "repo-write":
        if not allow_repo_write:
            reasons.append("REPO_WRITE_FLAG_REQUIRED")

    verdict = "ALLOW" if not reasons else "BLOCK"
    exit_code = 0 if verdict == "ALLOW" else 2

    result: Dict[str, Any] = {
        "verdict": verdict,
        "reasons": sorted(reasons),
        "normalized_paths": {"read": read_norm, "write": write_norm},
        "policy_version": POLICY_VERSION,
    }
    if warnings:
        result["warnings"] = sorted(warnings)
    return exit_code, result


def run_cli(intent_path: str) -> int:
    raw = json.loads(Path(intent_path).read_text(encoding="utf-8"))
    rc, result = admit_intent(raw)
    print(_stable_json(result), end="")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="admission", add_help=True)
    parser.add_argument("--intent", required=True, help="Path to intent.json")
    args = parser.parse_args(argv)
    return run_cli(str(args.intent))


if __name__ == "__main__":
    raise SystemExit(main())
