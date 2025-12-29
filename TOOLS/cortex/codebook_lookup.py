#!/usr/bin/env python3
"""
Codebook Lookup

Utility to look up codebook entries by their ID.

Usage:
    python TOOLS/codebook_lookup.py @C3      # Look up by ID
    python TOOLS/codebook_lookup.py --list   # List all entries
    python TOOLS/codebook_lookup.py --json   # Output as JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_PATH = PROJECT_ROOT / "CANON" / "CODEBOOK.md"


def parse_codebook() -> dict:
    """Parse the codebook markdown into a lookup dictionary."""
    if not CODEBOOK_PATH.exists():
        return {}
    
    content = CODEBOOK_PATH.read_text(encoding="utf-8", errors="ignore")
    entries = {}
    
    # Parse table rows: | `@ID` | Summary | `Source` |
    pattern = re.compile(r'\|\s*`(@\w+)`\s*\|\s*(.+?)\s*\|\s*`(.+?)`\s*\|')
    
    for match in pattern.finditer(content):
        entry_id = match.group(1)
        summary = match.group(2).strip()
        source = match.group(3).strip()
        entries[entry_id] = {
            "id": entry_id,
            "summary": summary,
            "source": source
        }
    
    return entries


def lookup(entry_id: str) -> dict:
    """Look up a single entry by ID."""
    entries = parse_codebook()
    
    # Normalize ID (ensure @ prefix, uppercase letter)
    if not entry_id.startswith("@"):
        entry_id = "@" + entry_id
    
    return entries.get(entry_id, None)


def expand(entry_id: str) -> str:
    """Expand a codebook ID to its full source content."""
    entry = lookup(entry_id)
    if not entry:
        return f"Unknown codebook entry: {entry_id}"
    
    source = entry["source"]
    source_path = PROJECT_ROOT / source
    
    if not source_path.exists():
        return f"{entry_id} = {entry['summary']} (source: {source})"
    
    if source_path.is_dir():
        # For skills, return SKILL.md content
        skill_md = source_path / "SKILL.md"
        if skill_md.exists():
            return skill_md.read_text(encoding="utf-8", errors="ignore")
        return f"{entry_id} = {entry['summary']} (directory: {source})"
    
    # For files, return content
    return source_path.read_text(encoding="utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Look up AGS codebook entries")
    parser.add_argument("id", nargs="?", help="Codebook ID to look up (e.g., @C3)")
    parser.add_argument("--list", action="store_true", help="List all entries")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--expand", action="store_true", help="Expand to full source content")
    
    args = parser.parse_args()
    
    entries = parse_codebook()
    
    if args.list:
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            for entry_id, entry in sorted(entries.items()):
                print(f"{entry_id}: {entry['summary']}")
        return 0
    
    if not args.id:
        parser.print_help()
        return 1
    
    entry = lookup(args.id)
    
    if not entry:
        print(f"Not found: {args.id}")
        return 1
    
    if args.expand:
        print(expand(args.id))
    elif args.json:
        print(json.dumps(entry, indent=2))
    else:
        print(f"{entry['id']}: {entry['summary']}")
        print(f"Source: {entry['source']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# API for import
__all__ = ["lookup", "expand", "parse_codebook"]
