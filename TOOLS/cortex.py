#!/usr/bin/env python3
"""
Tools for interacting with generated Cortex artifacts.

Usage:
  python TOOLS/cortex.py read <section_id>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECTION_INDEX_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "SECTION_INDEX.json"


def load_section_index(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_section(sections: List[Dict[str, Any]], section_id: str) -> Optional[Dict[str, Any]]:
    for record in sections:
        if record.get("section_id") == section_id:
            return record
    return None


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
    lines = content.splitlines(keepends=True)

    if start_line > len(lines) or end_line > len(lines):
        raise ValueError("section line range outside file bounds")

    return "".join(lines[start_line - 1 : end_line])


def cmd_read(args: argparse.Namespace) -> int:
    if not SECTION_INDEX_PATH.exists():
        print(f"SECTION_INDEX not found: {SECTION_INDEX_PATH}", file=sys.stderr)
        return 2

    try:
        sections = load_section_index(SECTION_INDEX_PATH)
    except Exception as exc:
        print(f"Failed to read SECTION_INDEX: {exc}", file=sys.stderr)
        return 2

    record = find_section(sections, args.section_id)
    if not record:
        print(f"Unknown section_id: {args.section_id}", file=sys.stderr)
        return 2

    try:
        text = read_section_text(PROJECT_ROOT, record)
    except Exception as exc:
        print(f"Failed to read section: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    read_p = sub.add_parser("read", help="Print a section by section_id (1-based inclusive lines)")
    read_p.add_argument("section_id", help="Deterministic section id (<path>::<heading_slug>::<ordinal>)")
    read_p.set_defaults(func=cmd_read)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

