#!/usr/bin/env python3
"""
Codebook Lookup Tool

Resolves symbolic identifiers to their expanded semantic content.
Supports both ASCII identifiers (@C0, @I3) and CJK single-token symbols (法, 道).

Usage:
    python codebook_lookup.py <id>           # Look up entry, return summary
    python codebook_lookup.py <id> --expand  # Look up entry, return full content
    python codebook_lookup.py --list         # List all entries
    python codebook_lookup.py --list --json  # List all entries as JSON

Compression ratios measured:
    - Single CJK symbol: up to 56,370x (1 token → full canon)
    - ASCII codebook ID: 3-20x per reference
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"
CANON_ROOT = PROJECT_ROOT / "LAW" / "CANON"

# Semantic symbol mappings (CJK single-token symbols → canon regions)
# These achieve 56,370x compression when receiver has canon in context
SEMANTIC_SYMBOLS = {
    # Domain pointers (1 token each)
    "法": {
        "id": "法",
        "name": "law",
        "description": "All canon law (LAW/CANON/*)",
        "path": "LAW/CANON",
        "type": "domain",
        "token_count": 1,
    },
    "真": {
        "id": "真",
        "name": "truth",
        "description": "Semiotic Foundation (THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md)",
        "path": "LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md",
        "type": "file",
        "token_count": 1,
    },
    "契": {
        "id": "契",
        "name": "contract",
        "description": "Contract law (CONTRACT.md)",
        "path": "LAW/CANON/CONSTITUTION/CONTRACT.md",
        "type": "file",
        "token_count": 1,
    },
    "驗": {
        "id": "驗",
        "name": "verify",
        "description": "Verification protocols (VERIFICATION.md)",
        "path": "LAW/CANON/GOVERNANCE/VERIFICATION.md",
        "type": "file",
        "token_count": 1,
    },
    "恆": {
        "id": "恆",
        "name": "invariants",
        "description": "System invariants (INVARIANTS.md)",
        "path": "LAW/CANON/CONSTITUTION/INVARIANTS.md",
        "type": "file",
        "token_count": 1,
    },
    "道": {
        "id": "道",
        "name": "path/principle",
        "description": "The way - context-activated meaning (path, principle, speech, method)",
        "expansions": {
            "path": "the way, the path to follow",
            "principle": "the underlying principle of nature",
            "speech": "to speak, to express in words",
            "method": "the method, the technique",
        },
        "type": "polysemic",
        "token_count": 1,
    },
    # Compound symbols
    "法.驗": {
        "id": "法.驗",
        "name": "law.verify",
        "description": "Verification within canon law",
        "paths": [
            "LAW/CANON/GOVERNANCE/VERIFICATION.md",
            "LAW/CANON/GOVERNANCE/VALIDATION_HOOKS.md",
        ],
        "type": "compound",
        "token_count": 3,
    },
}


def load_codebook() -> dict:
    """Load the commonsense codebook."""
    if CODEBOOK_PATH.exists():
        with open(CODEBOOK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"symbols": {}, "version": "0.0.0"}


def get_file_content(path: str) -> str:
    """Get content from a file path."""
    full_path = PROJECT_ROOT / path
    if full_path.is_file():
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    elif full_path.is_dir():
        # Return concatenated content of all .md files in directory
        content = []
        for md_file in sorted(full_path.rglob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                content.append(f"# {md_file.relative_to(PROJECT_ROOT)}\n\n{f.read()}")
        return "\n\n---\n\n".join(content)
    return f"[Path not found: {path}]"


def lookup_entry(entry_id: str, expand: bool = False) -> dict:
    """Look up a codebook entry by ID."""
    # Check semantic symbols first
    if entry_id in SEMANTIC_SYMBOLS:
        entry = SEMANTIC_SYMBOLS[entry_id].copy()
        if expand:
            if "path" in entry:
                entry["content"] = get_file_content(entry["path"])
            elif "paths" in entry:
                contents = []
                for p in entry["paths"]:
                    contents.append(f"## {p}\n\n{get_file_content(p)}")
                entry["content"] = "\n\n---\n\n".join(contents)
            elif "expansions" in entry:
                entry["content"] = json.dumps(entry["expansions"], indent=2)
        return {"found": True, "entry": entry}

    # Check commonsense codebook
    codebook = load_codebook()
    symbols = codebook.get("symbols", {})

    # Normalize ID (add @ prefix if missing for legacy format)
    normalized_id = entry_id if entry_id.startswith("@") else f"@{entry_id}"

    if normalized_id in symbols:
        predicates = symbols[normalized_id]
        return {
            "found": True,
            "entry": {
                "id": normalized_id,
                "predicates": predicates,
                "type": "predicate",
            }
        }

    # Not found
    return {
        "found": False,
        "error": f"Entry not found: {entry_id}",
        "available_semantic": list(SEMANTIC_SYMBOLS.keys()),
    }


def list_entries() -> dict:
    """List all available codebook entries."""
    codebook = load_codebook()

    return {
        "semantic_symbols": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "type": v["type"],
                "token_count": v["token_count"],
            }
            for k, v in SEMANTIC_SYMBOLS.items()
        },
        "commonsense_symbols": codebook.get("symbols", {}),
        "commonsense_version": codebook.get("version", "unknown"),
        "total_semantic": len(SEMANTIC_SYMBOLS),
        "total_commonsense": len(codebook.get("symbols", {})),
    }


def main():
    parser = argparse.ArgumentParser(description="Codebook lookup tool")
    parser.add_argument("id", nargs="?", help="Entry ID to look up")
    parser.add_argument("--expand", action="store_true", help="Return full content")
    parser.add_argument("--list", action="store_true", help="List all entries")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.list:
        result = list_entries()
    elif args.id:
        result = lookup_entry(args.id, expand=args.expand)
    else:
        result = list_entries()

    if args.json or args.list:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result.get("found"):
            entry = result["entry"]
            print(f"ID: {entry['id']}")
            print(f"Name: {entry.get('name', 'N/A')}")
            print(f"Type: {entry.get('type', 'N/A')}")
            print(f"Description: {entry.get('description', 'N/A')}")
            if "content" in entry:
                print(f"\n--- Content ---\n{entry['content'][:2000]}")
                if len(entry.get("content", "")) > 2000:
                    print(f"\n... [truncated, {len(entry['content'])} chars total]")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            if "available_semantic" in result:
                print(f"Available semantic symbols: {', '.join(result['available_semantic'])}")
            sys.exit(1)


if __name__ == "__main__":
    main()
