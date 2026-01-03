#!/usr/bin/env python3
"""
Tools for interacting with generated Cortex artifacts.

Usage:
  python TOOLS/cortex.py read <section_id>
  python TOOLS/cortex.py search "<query>"
  python TOOLS/cortex.py resolve <section_id>
  python TOOLS/cortex.py summary <section_id>
  python TOOLS/cortex.py summary --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SECTION_INDEX_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated" / "SECTION_INDEX.json"
SUMMARY_INDEX_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated" / "SUMMARY_INDEX.json"
RUNS_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
SECTION_INDEX_REL = Path("NAVIGATION") / "CORTEX" / "_generated" / "SECTION_INDEX.json"

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def load_section_index(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_index(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_section(sections: List[Dict[str, Any]], section_id: str) -> Optional[Dict[str, Any]]:
    for record in sections:
        if record.get("section_id") == section_id:
            return record
    return None


def find_summary(records: List[Dict[str, Any]], section_id: str) -> Optional[Dict[str, Any]]:
    for record in records:
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


class ProvenanceLogger:
    schema_version = "1.0"

    def __init__(self, project_root: Path, section_index_path: Path) -> None:
        self._project_root = project_root
        self._section_index_path = section_index_path

    def enabled(self) -> bool:
        return bool(os.environ.get("CORTEX_RUN_ID"))

    def _run_id(self) -> str:
        return _sanitize_run_id(os.environ.get("CORTEX_RUN_ID", ""))

    def _run_dir(self) -> Path:
        return RUNS_ROOT / self._run_id()

    def _events_path(self) -> Path:
        return self._run_dir() / "events.jsonl"

    def _meta_path(self) -> Path:
        return self._run_dir() / "run_meta.json"

    def _ensure_run_meta(self) -> None:
        if not self.enabled():
            return
        run_dir = self._run_dir()
        meta_path = self._meta_path()
        if meta_path.exists():
            return

        run_dir.mkdir(parents=True, exist_ok=True)

        section_index_sha = None
        section_index_rel = None
        if self._section_index_path.exists():
            section_index_rel = SECTION_INDEX_REL.as_posix()
            try:
                data = self._section_index_path.read_bytes()
                section_index_sha = hashlib.sha256(data).hexdigest()
            except Exception:
                section_index_sha = None
        else:
            print(f"Warning: SECTION_INDEX missing at {SECTION_INDEX_REL.as_posix()}", file=sys.stderr)

        payload = {
            "cortex_provenance_schema_version": self.schema_version,
            "created_at_utc": _timestamp_utc(),
            "run_id": self._run_id(),
            "section_index_path": section_index_rel,
            "section_index_sha256": section_index_sha,
        }
        meta_path.write_text(
            json.dumps(_normalize_newlines(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def log_event(
        self,
        *,
        op: str,
        query: Optional[str] = None,
        section_id: Optional[str] = None,
        result_count: Optional[int] = None,
        path: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        hash_value: Optional[str] = None,
    ) -> None:
        if not self.enabled():
            return
        try:
            self._ensure_run_meta()
            event = {
                "end_line": end_line,
                "hash": hash_value,
                "op": op,
                "path": path,
                "query": query,
                "result_count": result_count,
                "section_id": section_id,
                "start_line": start_line,
                "timestamp_utc": _timestamp_utc(),
            }
            line = json.dumps(_normalize_newlines(event), sort_keys=True, ensure_ascii=False) + "\n"
            with open(self._events_path(), "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
        except Exception:
            return


PROVENANCE = ProvenanceLogger(PROJECT_ROOT, SECTION_INDEX_PATH)


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
        PROVENANCE.log_event(op="search", query=str(args.query or "").strip() or None)
        return 2

    query = str(args.query or "").strip()
    tokens = _tokenize(query)
    if not tokens:
        PROVENANCE.log_event(op="search", query=query or None, result_count=0)
        return 1

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        PROVENANCE.log_event(op="search", query=query or None)
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
        PROVENANCE.log_event(op="search", query=query or None, result_count=0)
        return 1

    # Deterministic sorting: primary score (desc), secondary section_id (asc).
    scored.sort(key=lambda t: (-t[0], t[1]))
    PROVENANCE.log_event(op="search", query=query or None, result_count=len(scored))
    for _, section_id, heading in scored:
        try:
            sys.stdout.write(f"{section_id}\t{heading}\n")
        except BrokenPipeError:
            return 0
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        PROVENANCE.log_event(op="read", section_id=normalize_section_id_arg(args.section_id))
        return 2

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        PROVENANCE.log_event(op="read", section_id=normalize_section_id_arg(args.section_id))
        return 2

    section_id = normalize_section_id_arg(args.section_id)
    record = find_section(sections, section_id)
    if not record:
        print(f"Unknown section_id: {section_id}", file=sys.stderr)
        PROVENANCE.log_event(op="read", section_id=section_id)
        return 2

    try:
        text = read_section_text(PROJECT_ROOT, record)
    except Exception as exc:
        print(f"Failed to read section: {exc}", file=sys.stderr)
        PROVENANCE.log_event(
            op="read",
            section_id=section_id,
            path=str(record.get("path")).replace("\\", "/") if record.get("path") else None,
            start_line=int(record.get("start_line")) if record.get("start_line") else None,
            end_line=int(record.get("end_line")) if record.get("end_line") else None,
            hash_value=str(record.get("hash")) if record.get("hash") else None,
        )
        return 1

    PROVENANCE.log_event(
        op="read",
        section_id=str(record.get("section_id")),
        path=str(record.get("path")).replace("\\", "/") if record.get("path") else None,
        start_line=int(record.get("start_line")),
        end_line=int(record.get("end_line")),
        hash_value=str(record.get("hash")),
    )
    sys.stdout.write(text)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        PROVENANCE.log_event(op="resolve", section_id=normalize_section_id_arg(args.section_id))
        return 2

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        PROVENANCE.log_event(op="resolve", section_id=normalize_section_id_arg(args.section_id))
        return 2

    section_id = normalize_section_id_arg(args.section_id)
    record = find_section(sections, section_id)
    if not record:
        print(f"Unknown section_id: {section_id}", file=sys.stderr)
        PROVENANCE.log_event(op="resolve", section_id=section_id)
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

    PROVENANCE.log_event(
        op="resolve",
        section_id=str(record.get("section_id")),
        path=str(record.get("path")).replace("\\", "/") if record.get("path") else None,
        start_line=int(record.get("start_line")),
        end_line=int(record.get("end_line")),
        hash_value=str(record.get("hash")),
    )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    if not SUMMARY_INDEX_PATH.exists():
        print(f"SUMMARY_INDEX not found: {SUMMARY_INDEX_PATH}", file=sys.stderr)
        if getattr(args, "section_id", None):
            PROVENANCE.log_event(op="summary", section_id=normalize_section_id_arg(args.section_id))
        else:
            PROVENANCE.log_event(op="summary")
        return 2

    try:
        records = load_summary_index(SUMMARY_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SUMMARY_INDEX: {exc}", file=sys.stderr)
        if getattr(args, "section_id", None):
            PROVENANCE.log_event(op="summary", section_id=normalize_section_id_arg(args.section_id))
        else:
            PROVENANCE.log_event(op="summary")
        return 2

    if getattr(args, "list", False):
        for record in sorted(records, key=lambda r: str(r.get("section_id") or "")):
            section_id = str(record.get("section_id") or "").strip()
            summary_path = str(record.get("summary_path") or "").strip()
            if not section_id or not summary_path:
                continue
            try:
                sys.stdout.write(f"{section_id}\t{summary_path}\n")
            except BrokenPipeError:
                return 0
        PROVENANCE.log_event(op="summary")
        return 0

    if not getattr(args, "section_id", None):
        print("Missing section_id (or use --list)", file=sys.stderr)
        PROVENANCE.log_event(op="summary")
        return 2

    section_id = normalize_section_id_arg(args.section_id)
    record = find_summary(records, section_id)
    if not record:
        print(f"Unknown section_id (no summary): {section_id}", file=sys.stderr)
        PROVENANCE.log_event(op="summary", section_id=section_id)
        return 2

    summary_rel = str(record.get("summary_path") or "").strip()
    if not summary_rel:
        print(f"Missing summary_path for section_id: {section_id}", file=sys.stderr)
        PROVENANCE.log_event(
            op="summary",
            section_id=section_id,
            hash_value=str(record.get("section_hash") or "") or None,
        )
        return 2

    summary_path = PROJECT_ROOT / Path(summary_rel)
    try:
        content = summary_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"Failed to read summary: {exc}", file=sys.stderr)
        PROVENANCE.log_event(
            op="summary",
            section_id=section_id,
            hash_value=str(record.get("section_hash") or "") or None,
        )
        return 1

    PROVENANCE.log_event(
        op="summary",
        section_id=section_id,
        hash_value=str(record.get("section_hash") or "") or None,
    )
    sys.stdout.write(content)
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

    summary_p = sub.add_parser("summary", help="Print deterministic advisory summary for a section_id")
    summary_p.add_argument("section_id", nargs="?", help="Deterministic section id (<path>::<heading_slug>::<ordinal>)")
    summary_p.add_argument("--list", action="store_true", help="List available summaries (section_id<TAB>summary_path)")
    summary_p.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
