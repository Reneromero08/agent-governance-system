#!/usr/bin/env python3
"""
Tools for interacting with generated Cortex artifacts.

Usage:
  python TOOLS/cortex.py read <section_id>
  python TOOLS/cortex.py search "<query>"
  python TOOLS/cortex.py resolve <section_id>
"""

import argparse
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECTION_INDEX_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "SECTION_INDEX.json"
RUNS_ROOT = PROJECT_ROOT / "CONTRACTS" / "_runs"

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def load_section_index(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_section(sections: List[Dict[str, Any]], section_id: str) -> Optional[Dict[str, Any]]:
    for record in sections:
        if record.get("section_id") == section_id:
            return record
    return None


def normalize_section_id_arg(section_id: str) -> str:
    value = str(section_id or "").strip()
    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in {"\"", "'"}:
            return value[1:-1]
    return value


def _sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip())
    return cleaned.strip("._-") or "run"


def _normalize_newlines(value: object) -> object:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, list):
        return [_normalize_newlines(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_newlines(v) for k, v in value.items()}
    return value


def log_provenance_event(event: Dict[str, Any]) -> None:
    run_id = os.environ.get("CORTEX_RUN_ID")
    if not run_id:
        return
    run_dir = RUNS_ROOT / _sanitize_run_id(run_id)
    log_path = run_dir / "events.jsonl"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_newlines(event)
        line = json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n"
        log_path.write_text("", encoding="utf-8") if not log_path.exists() else None
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
    except Exception:
        # Logging must never change the command's behavior.
        return


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_section_text(project_root: Path, record: Dict[str, Any]) -> str:
    rel_path = str(record.get("path", "")).strip()
    if not rel_path:
        raise ValueError("section record missing path")

    start_line = int(record.get("start_line"))
    end_line = int(record.get("end_line"))
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        raise ValueError("invalid start_line/end_line in section record")

    file_path = project_root / Path(rel_path)
    content = file_path.read_text(encoding="utf-8", errors="replace")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines(keepends=True)

    if start_line > len(lines) or end_line > len(lines):
        raise ValueError("section line range outside file bounds")

    return "".join(lines[start_line - 1 : end_line])


def _tokenize(query: str) -> List[str]:
    return [t.strip().lower() for t in query.split() if t.strip()]


def _score_section(query_tokens: List[str], heading: str, text: str) -> int:
    if not query_tokens:
        return 0
    heading_l = (heading or "").lower()
    text_l = (text or "").lower()
    score = 0
    for token in query_tokens:
        if token in heading_l:
            score += 2
        elif token in text_l:
            score += 1
    return score


def cmd_search(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "search",
                "query": str(args.query or "").strip() or None,
                "section_id": None,
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    query = str(args.query or "").strip()
    tokens = _tokenize(query)
    if not tokens:
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "search",
                "query": query or None,
                "section_id": None,
                "result_count": 0,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 1

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "search",
                "query": query or None,
                "section_id": None,
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    scored: List[tuple[int, str, str]] = []
    for record in sections:
        section_id = str(record.get("section_id", "")).strip()
        heading = str(record.get("heading", "")).strip()
        if not section_id:
            continue

        try:
            text = read_section_text(PROJECT_ROOT, record)
        except Exception:
            # Skip unreadable sections rather than crashing search.
            continue

        score = _score_section(tokens, heading, text)
        if score <= 0:
            continue
        scored.append((score, section_id, heading))

    if not scored:
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "search",
                "query": query or None,
                "section_id": None,
                "result_count": 0,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 1

    # Deterministic sorting: primary score (desc), secondary section_id (asc).
    scored.sort(key=lambda t: (-t[0], t[1]))
    log_provenance_event(
        {
            "timestamp_utc": _timestamp_utc(),
            "op": "search",
            "query": query or None,
            "section_id": None,
            "result_count": len(scored),
            "path": None,
            "start_line": None,
            "end_line": None,
            "hash": None,
        }
    )
    for _, section_id, heading in scored:
        try:
            sys.stdout.write(f"{section_id}\t{heading}\n")
        except BrokenPipeError:
            return 0
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "read",
                "query": None,
                "section_id": normalize_section_id_arg(args.section_id),
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "read",
                "query": None,
                "section_id": normalize_section_id_arg(args.section_id),
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    section_id = normalize_section_id_arg(args.section_id)
    record = find_section(sections, section_id)
    if not record:
        print(f"Unknown section_id: {section_id}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "read",
                "query": None,
                "section_id": section_id,
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    try:
        text = read_section_text(PROJECT_ROOT, record)
    except Exception as exc:
        print(f"Failed to read section: {exc}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "read",
                "query": None,
                "section_id": section_id,
                "result_count": None,
                "path": str(record.get("path")).replace("\\", "/") if record.get("path") else None,
                "start_line": int(record.get("start_line")) if record.get("start_line") else None,
                "end_line": int(record.get("end_line")) if record.get("end_line") else None,
                "hash": str(record.get("hash")) if record.get("hash") else None,
            }
        )
        return 1

    log_provenance_event(
        {
            "timestamp_utc": _timestamp_utc(),
            "op": "read",
            "query": None,
            "section_id": str(record.get("section_id")),
            "result_count": None,
            "path": str(record.get("path")).replace("\\", "/") if record.get("path") else None,
            "start_line": int(record.get("start_line")),
            "end_line": int(record.get("end_line")),
            "hash": str(record.get("hash")),
        }
    )
    sys.stdout.write(text)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "resolve",
                "query": None,
                "section_id": normalize_section_id_arg(args.section_id),
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "resolve",
                "query": None,
                "section_id": normalize_section_id_arg(args.section_id),
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    section_id = normalize_section_id_arg(args.section_id)
    record = find_section(sections, section_id)
    if not record:
        print(f"Unknown section_id: {section_id}", file=sys.stderr)
        log_provenance_event(
            {
                "timestamp_utc": _timestamp_utc(),
                "op": "resolve",
                "query": None,
                "section_id": section_id,
                "result_count": None,
                "path": None,
                "start_line": None,
                "end_line": None,
                "hash": None,
            }
        )
        return 2

    payload = {
        "end_line": int(record.get("end_line")),
        "hash": str(record.get("hash")),
        "path": str(record.get("path")).replace("\\", "/"),
        "section_id": str(record.get("section_id")),
        "start_line": int(record.get("start_line")),
    }
    heading = record.get("heading")
    if heading is not None:
        payload["heading"] = str(heading)

    log_provenance_event(
        {
            "timestamp_utc": _timestamp_utc(),
            "op": "resolve",
            "query": None,
            "section_id": str(record.get("section_id")),
            "result_count": None,
            "path": str(record.get("path")).replace("\\", "/") if record.get("path") else None,
            "start_line": int(record.get("start_line")),
            "end_line": int(record.get("end_line")),
            "hash": str(record.get("hash")),
        }
    )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    read_p = sub.add_parser("read", help="Print a section by section_id (1-based inclusive lines)")
    read_p.add_argument("section_id", help="Deterministic section id (<path>::<heading_slug>::<ordinal>)")
    read_p.set_defaults(func=cmd_read)

    search_p = sub.add_parser("search", help="Search SECTION_INDEX and print matching section_ids")
    search_p.add_argument("query", help="Plain text query (use quotes to include spaces)")
    search_p.set_defaults(func=cmd_search)

    resolve_p = sub.add_parser("resolve", help="Print JSON metadata for a section_id")
    resolve_p.add_argument("section_id", help="Deterministic section id (<path>::<heading_slug>::<ordinal>)")
    resolve_p.set_defaults(func=cmd_resolve)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
