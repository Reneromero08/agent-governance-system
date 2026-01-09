#!/usr/bin/env python3
"""
Codebook Lookup Tool (符典)

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

Principle: Symbols do not carry phonetic glosses. They point directly to semantic
regions. Human reference: THOUGHT/LAB/FORMULA/CODIFIER.md
"""

import argparse
import json
import re
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

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SYMBOL MAPPINGS (符 → 路)
# ═══════════════════════════════════════════════════════════════════════════════
# CJK single-token symbols → semantic regions
# NO PHONETIC GLOSSES - symbols point directly to domains/paths
# Human reference: THOUGHT/LAB/FORMULA/CODIFIER.md
# ═══════════════════════════════════════════════════════════════════════════════

SEMANTIC_SYMBOLS = {
    # ───────────────────────────────────────────────────────────────────────────
    # CORE DOMAIN POINTERS (measured compression ratios)
    # ───────────────────────────────────────────────────────────────────────────
    "法": {
        "id": "法",
        "path": "LAW/CANON",
        "type": "domain",
        "compression": 56370,
    },
    "真": {
        "id": "真",
        "path": "LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md",
        "type": "file",
        "compression": 8200,
    },
    "契": {
        "id": "契",
        "path": "LAW/CANON/CONSTITUTION/CONTRACT.md",
        "type": "file",
        "compression": 4100,
    },
    "恆": {
        "id": "恆",
        "path": "LAW/CANON/CONSTITUTION/INVARIANTS.md",
        "type": "file",
        "compression": 5600,
    },
    "驗": {
        "id": "驗",
        "path": "LAW/CANON/GOVERNANCE/VERIFICATION.md",
        "type": "file",
        "compression": 3800,
    },

    # ───────────────────────────────────────────────────────────────────────────
    # GOVERNANCE OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    "證": {
        "id": "證",
        "path": "NAVIGATION/RECEIPTS",
        "type": "domain",
        "compression": 12000,
    },
    "變": {
        "id": "變",
        "paths": [
            "THOUGHT/LAB/CATALYTIC",
            "LAW/CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md",
        ],
        "type": "compound",
        "compression": 8500,
    },
    "冊": {
        "id": "冊",
        "path": "NAVIGATION/CORTEX/db",
        "type": "domain",
        "compression": 4200,
    },
    "錄": {
        "id": "錄",
        "paths": [
            "CAPABILITY/PRIMITIVES/canon_index.py",
            "CAPABILITY/PRIMITIVES/adr_index.py",
            "CAPABILITY/PRIMITIVES/skill_index.py",
            "CAPABILITY/PRIMITIVES/cross_ref_index.py",
        ],
        "type": "compound",
        "compression": 15000,
    },
    "限": {
        "id": "限",
        "path": "LAW/CANON/CONSTITUTION/INVARIANTS.md",
        "type": "alias",
        "alias_of": "恆",
        "compression": 5600,
    },
    "許": {
        "id": "許",
        "paths": [
            "LAW/CANON/CONSTITUTION/CONTRACT.md",
            "LAW/CONTEXT/decisions/ADR-001-build-and-artifacts.md",
        ],
        "type": "compound",
        "compression": 3200,
    },
    "禁": {
        "id": "禁",
        "paths": [
            "LAW/CANON/CONSTITUTION/INVARIANTS.md",
            "LAW/CANON/GOVERNANCE/IMMUTABILITY.md",
        ],
        "type": "compound",
        "compression": 4800,
    },
    "雜": {
        "id": "雜",
        "path": "LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md",
        "type": "file",
        "compression": 2100,
    },
    "復": {
        "id": "復",
        "paths": [
            "CAPABILITY/SKILLS/pipeline-dag-restore/SKILL.md",
        ],
        "type": "compound",
        "compression": 3600,
    },

    # ───────────────────────────────────────────────────────────────────────────
    # VALIDATION OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    "試": {
        "id": "試",
        "path": "CAPABILITY/TESTBENCH",
        "type": "domain",
        "compression": 42000,
    },
    "查": {
        "id": "查",
        "path": "NAVIGATION/CORTEX/semantic",
        "type": "domain",
        "compression": 8800,
    },
    "載": {
        "id": "載",
        "paths": [
            "CAPABILITY/TOOLS/codebook_lookup.py",
            "CAPABILITY/PRIMITIVES/canon_index.py",
        ],
        "type": "compound",
        "compression": 4500,
    },
    "存": {
        "id": "存",
        "paths": [
            "NAVIGATION/CAS",
        ],
        "type": "compound",
        "compression": 6200,
    },
    "掃": {
        "id": "掃",
        "paths": [
            "CAPABILITY/PRIMITIVES/skill_index.py",
            "CAPABILITY/PRIMITIVES/canon_index.py",
        ],
        "type": "compound",
        "compression": 5100,
    },
    "核": {
        "id": "核",
        "paths": [
            "LAW/CANON/GOVERNANCE/VERIFICATION.md",
            "CAPABILITY/SKILLS/cas-integrity-check/SKILL.md",
        ],
        "type": "compound",
        "compression": 4900,
    },

    # ───────────────────────────────────────────────────────────────────────────
    # STRUCTURAL SYMBOLS
    # ───────────────────────────────────────────────────────────────────────────
    "道": {
        "id": "道",
        "type": "polysemic",
        "contexts": {
            "CONTEXT_PATH": "LAW/CANON",
            "CONTEXT_PRINCIPLE": "LAW/CANON/FOUNDATION",
            "CONTEXT_METHOD": "CAPABILITY/SKILLS",
        },
        "compression": 0,  # Context-dependent
    },
    "圖": {
        "id": "圖",
        "path": "CAPABILITY/PRIMITIVES/cross_ref_index.py",
        "type": "file",
        "compression": 3400,
    },
    "鏈": {
        "id": "鏈",
        "paths": [
            "CAPABILITY/SKILLS/pipeline-dag-scheduler/SKILL.md",
            "CAPABILITY/SKILLS/pipeline-dag-receipts/SKILL.md",
        ],
        "type": "compound",
        "compression": 4100,
    },
    "根": {
        "id": "根",
        "path": "LAW/CANON",
        "type": "alias",
        "alias_of": "法",
        "compression": 56370,
    },
    "枝": {
        "id": "枝",
        "path": "THOUGHT/LAB",
        "type": "domain",
        "compression": 28000,
    },

    # ───────────────────────────────────────────────────────────────────────────
    # COMPOUND SYMBOLS (composed with . operator)
    # ───────────────────────────────────────────────────────────────────────────
    "法.驗": {
        "id": "法.驗",
        "paths": [
            "LAW/CANON/GOVERNANCE/VERIFICATION.md",
            "LAW/CANON/GOVERNANCE/VALIDATION_HOOKS.md",
        ],
        "type": "compound",
        "compression": 7200,
    },
    "法.契": {
        "id": "法.契",
        "path": "LAW/CANON/CONSTITUTION/CONTRACT.md",
        "type": "file",
        "compression": 4100,
    },
    "證.雜": {
        "id": "證.雜",
        "paths": [
            "NAVIGATION/RECEIPTS",
            "LAW/CANON/SEMANTIC/TOKEN_RECEIPT_SPEC.md",
        ],
        "type": "compound",
        "compression": 6800,
    },
    "冊.雜": {
        "id": "冊.雜",
        "paths": [
            "NAVIGATION/CORTEX/db",
            "CAPABILITY/PRIMITIVES/model_registry.py",
        ],
        "type": "compound",
        "compression": 5400,
    },
}


def load_codebook() -> dict:
    """Load the commonsense codebook."""
    if CODEBOOK_PATH.exists():
        with open(CODEBOOK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"symbols": {}, "version": "0.0.0"}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACT MACRO GRAMMAR PARSER
# ═══════════════════════════════════════════════════════════════════════════════
# Grammar: RADICAL[OPERATOR][NUMBER][:CONTEXT]
# Examples: C3, I5, C*, V!, C3:build, C&I
# ═══════════════════════════════════════════════════════════════════════════════

MACRO_PATTERN = re.compile(
    r'^([CIVLGSRAJP])'      # Radical (required)
    r'([*!?&|.])?'          # Operator (optional)
    r'(\d+)?'               # Number (optional)
    r'(?::(\w+))?$'         # Context (optional, after colon)
)

RADICALS = {
    'C': 'contract_rules',
    'I': 'invariants',
    'V': 'verification',
    'L': 'law',
    'G': 'governance',
    'S': 'schema',
    'R': 'receipt',
    'A': 'adr',
    'J': 'jobspec',
    'P': 'policy',
}

OPERATORS = {
    '*': 'ALL',
    '!': 'NOT',
    '?': 'CHECK',
    '&': 'AND',
    '|': 'OR',
    '.': 'PATH',
}


def parse_macro(macro: str) -> dict | None:
    """Parse a compact macro notation into components."""
    match = MACRO_PATTERN.match(macro)
    if not match:
        return None

    radical, operator, number, context = match.groups()
    return {
        'radical': radical,
        'domain': RADICALS.get(radical),
        'operator': operator,
        'operator_meaning': OPERATORS.get(operator) if operator else None,
        'number': int(number) if number else None,
        'context': context,
        'raw': macro,
    }


def lookup_macro(macro: str, expand: bool = False) -> dict:
    """Look up a compact macro notation."""
    parsed = parse_macro(macro)
    if not parsed:
        return {'found': False, 'error': f'Invalid macro format: {macro}'}

    codebook = load_codebook()
    radical = parsed['radical']
    number = parsed['number']
    operator = parsed['operator']
    context = parsed['context']

    # Handle domain lookups (no number)
    if number is None and operator != '*':
        # Return domain info from radicals
        radicals = codebook.get('radicals', {})
        if radical in radicals:
            entry = radicals[radical].copy()
            entry['id'] = radical
            entry['type'] = 'radical'
            if expand and 'path' in entry:
                entry['content'] = get_file_content(entry['path'])
            return {'found': True, 'entry': entry, 'parsed': parsed}
        return {'found': False, 'error': f'Unknown radical: {radical}'}

    # Handle ALL operator (*)
    if operator == '*':
        domain_key = parsed['domain']
        if domain_key in ('contract_rules', 'invariants'):
            rules = codebook.get(domain_key, {})
            entry = {
                'id': macro,
                'type': 'collection',
                'count': len(rules),
                'items': list(rules.keys()),
            }
            if expand:
                entry['rules'] = rules
            return {'found': True, 'entry': entry, 'parsed': parsed}
        return {'found': False, 'error': f'* operator not supported for {radical}'}

    # Handle specific rule lookups (C3, I5, etc.)
    if number is not None:
        domain_key = parsed['domain']
        rule_key = f'{radical}{number}'

        if domain_key == 'contract_rules':
            rules = codebook.get('contract_rules', {})
            if rule_key in rules:
                entry = rules[rule_key].copy()
                entry['id'] = rule_key
                entry['type'] = 'contract_rule'
                if context:
                    entry['context'] = context
                return {'found': True, 'entry': entry, 'parsed': parsed}
            return {'found': False, 'error': f'Contract rule not found: {rule_key}'}

        elif domain_key == 'invariants':
            rules = codebook.get('invariants', {})
            if rule_key in rules:
                entry = rules[rule_key].copy()
                entry['id'] = rule_key
                entry['type'] = 'invariant'
                if context:
                    entry['context'] = context
                return {'found': True, 'entry': entry, 'parsed': parsed}
            return {'found': False, 'error': f'Invariant not found: {rule_key}'}

        return {'found': False, 'error': f'Numbered lookup not supported for {radical}'}

    return {'found': False, 'error': f'Could not resolve macro: {macro}'}


def get_file_content(path: str) -> str:
    """Get content from a file path."""
    # Strip any anchor fragments (e.g., file.md#section)
    path = path.split("#")[0]
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


# ═══════════════════════════════════════════════════════════════════════════════
# STACKED RESOLUTION (Phase 5.2.3.1)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM1_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
CANON_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"
ADR_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "adr_index.db"
SKILL_INDEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "skill_index.db"


def _get_domain_paths(entry):
    """Extract all paths from a symbol entry."""
    paths = []
    if "path" in entry:
        paths.append(entry["path"])
    if "paths" in entry:
        paths.extend(entry["paths"])
    if "contexts" in entry:
        paths.extend(entry["contexts"].values())
    return paths


def _fts_search_within_paths(query, paths, limit=20):
    """Full-text search within specific paths using CORTEX FTS5."""
    if not SYSTEM1_DB.exists():
        return []
    import sqlite3
    results = []
    try:
        with sqlite3.connect(str(SYSTEM1_DB)) as conn:
            conn.row_factory = sqlite3.Row
            path_conditions = " OR ".join([f"f.path LIKE ?" for _ in paths])
            path_params = [f"{p}%" for p in paths]
            cursor = conn.execute(f"""
                SELECT f.path, c.chunk_index, fts.content,
                       snippet(chunks_fts, 0, '<<', '>>', '...', 64) as snippet, rank
                FROM chunks_fts fts
                JOIN chunks c ON fts.chunk_id = c.chunk_id
                JOIN files f ON c.file_id = f.file_id
                WHERE chunks_fts MATCH ? AND ({path_conditions})
                ORDER BY rank LIMIT ?
            """, (query, *path_params, limit))
            for row in cursor.fetchall():
                results.append({
                    'path': row['path'], 'chunk_index': row['chunk_index'],
                    'content': row['content'], 'snippet': row['snippet'], 'rank': row['rank'],
                })
    except Exception:
        pass
    return results


def _get_index_db_for_paths(paths):
    """Determine the best index database for given paths."""
    for path in paths:
        if path.startswith("LAW/CANON"):
            if CANON_INDEX_DB.exists():
                return CANON_INDEX_DB
        elif path.startswith("LAW/CONTEXT/decisions"):
            if ADR_INDEX_DB.exists():
                return ADR_INDEX_DB
        elif path.startswith("CAPABILITY/SKILLS"):
            if SKILL_INDEX_DB.exists():
                return SKILL_INDEX_DB
    if SYSTEM1_DB.exists():
        return SYSTEM1_DB
    return None


def _semantic_search_within_paths(query, paths, limit=10):
    """Vector similarity search within specific paths using CORTEX embeddings."""
    db_path = _get_index_db_for_paths(paths)
    if not db_path:
        return []
    results = []
    import sqlite3
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='canon_records'")
            has_canon_records = cursor.fetchone() is not None
            if has_canon_records:
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                primitives_path = PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"
                if str(primitives_path) not in sys.path:
                    sys.path.insert(0, str(primitives_path))
                try:
                    import importlib
                    canon_module = importlib.import_module('canon_index')
                    search_canon = canon_module.search_canon
                    search_results = search_canon(query, top_k=limit * 2)
                    for r in search_results:
                        file_path = r.get('file_path', '')
                        if not file_path.startswith("LAW/CANON"):
                            file_path = f"LAW/CANON/{file_path}"
                        for domain_path in paths:
                            if file_path.startswith(domain_path):
                                results.append({
                                    'path': file_path, 'content': r.get('text', ''),
                                    'similarity': r.get('similarity', 0.5), 'hash': r.get('content_hash', ''),
                                })
                                break
                        if len(results) >= limit:
                            break
                except Exception:
                    pass
    except Exception:
        pass
    return results


def stacked_lookup(entry_id, query=None, semantic=None, expand=False, limit=10):
    """Stacked symbol resolution with optional FTS or semantic filtering."""
    base_result = lookup_entry(entry_id, expand=False)
    if not base_result.get('found'):
        return base_result
    entry = base_result['entry']
    domain_paths = _get_domain_paths(entry)

    if not query and not semantic:
        if expand:
            if "path" in entry:
                entry["content"] = get_file_content(entry["path"])
            elif "paths" in entry:
                contents = [f"## {p}\n\n{get_file_content(p)}" for p in entry["paths"]]
                entry["content"] = "\n\n---\n\n".join(contents)
            elif "contexts" in entry:
                entry["content"] = json.dumps(entry["contexts"], indent=2)
        return {"found": True, "entry": entry, "resolution": "L1"}

    if query:
        chunks = _fts_search_within_paths(query, domain_paths, limit=limit)
        if chunks:
            filtered_content = [f"## {c['path']} (chunk {c['chunk_index']})\n\n{c['content']}" for c in chunks]
            entry["filtered_content"] = "\n\n---\n\n".join(filtered_content)
            entry["chunks"] = chunks
            entry["chunk_count"] = len(chunks)
        else:
            entry["filtered_content"] = f"[No FTS matches for '{query}' in {domain_paths}]"
            entry["chunks"] = []
            entry["chunk_count"] = 0
        return {"found": True, "entry": entry, "resolution": "L1+L2", "query": query, "domain_paths": domain_paths}

    if semantic:
        chunks = _semantic_search_within_paths(semantic, domain_paths, limit=limit)
        if chunks:
            filtered_content = [f"## {c['path']} (similarity: {c['similarity']:.3f})\n\n{c['content']}" for c in chunks]
            entry["filtered_content"] = "\n\n---\n\n".join(filtered_content)
            entry["chunks"] = chunks
            entry["chunk_count"] = len(chunks)
        else:
            entry["filtered_content"] = f"[No semantic matches for '{semantic}' in {domain_paths}]"
            entry["chunks"] = []
            entry["chunk_count"] = 0
        return {"found": True, "entry": entry, "resolution": "L1+L3", "semantic": semantic, "domain_paths": domain_paths}

    return base_result


def lookup_entry(entry_id: str, expand: bool = False) -> dict:
    """Look up a codebook entry by ID."""
    # Try compact macro notation first (C3, I5, C*, etc.)
    if MACRO_PATTERN.match(entry_id):
        result = lookup_macro(entry_id, expand=expand)
        if result.get('found'):
            return result

    # Check semantic symbols (CJK)
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
            elif "contexts" in entry:
                # Polysemic symbol - return context mappings
                entry["content"] = json.dumps(entry["contexts"], indent=2)
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

    # Build macro vocabulary from codebook
    macro_vocab = {
        "radicals": list(codebook.get("radicals", {}).keys()),
        "operators": list(codebook.get("operators", {}).keys()),
        "contract_rules": list(codebook.get("contract_rules", {}).keys()),
        "invariants": list(codebook.get("invariants", {}).keys()),
    }

    # No phonetic glosses - just symbol → path → type → compression
    return {
        "semantic_symbols": {
            k: {
                "type": v["type"],
                "path": v.get("path"),
                "paths": v.get("paths"),
                "compression": v.get("compression", 0),
            }
            for k, v in SEMANTIC_SYMBOLS.items()
        },
        "macro_vocabulary": macro_vocab,
        "macro_version": codebook.get("version", "unknown"),
        "macro_grammar": codebook.get("grammar", {}),
        "total_semantic": len(SEMANTIC_SYMBOLS),
        "total_macros": (
            len(macro_vocab["contract_rules"]) +
            len(macro_vocab["invariants"]) +
            len(macro_vocab["radicals"])
        ),
        "codifier": "THOUGHT/LAB/FORMULA/CODIFIER.md",
    }


def main():
    parser = argparse.ArgumentParser(description="Codebook lookup tool (符典)")
    parser.add_argument("id", nargs="?", help="Entry ID to look up")
    parser.add_argument("--expand", action="store_true", help="Return full content (L1)")
    parser.add_argument("--query", type=str, help="FTS query within domain (L1+L2)")
    parser.add_argument("--semantic", type=str, help="Semantic query within domain (L1+L3)")
    parser.add_argument("--limit", type=int, default=10, help="Max results for filtered queries")
    parser.add_argument("--list", action="store_true", help="List all entries")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.list:
        result = list_entries()
    elif args.id:
        if args.query or args.semantic:
            result = stacked_lookup(args.id, query=args.query, semantic=args.semantic,
                                    expand=args.expand, limit=args.limit)
        else:
            result = lookup_entry(args.id, expand=args.expand)
    else:
        result = list_entries()

    if args.json or args.list:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result.get("found"):
            entry = result["entry"]
            print(f"符: {entry['id']}")
            print(f"類: {entry.get('type', 'N/A')}")
            if entry.get("path"):
                print(f"路: {entry.get('path')}")
            if entry.get("paths"):
                print(f"路: {', '.join(entry.get('paths', []))}")
            if entry.get("compression"):
                print(f"壓: {entry.get('compression')}×")
            resolution = result.get("resolution")
            if resolution:
                print(f"層: {resolution}")
            if "filtered_content" in entry:
                print(f"濾: {entry.get('chunk_count', 0)} chunks")
                print(f"\n--- 濾容 ---\n{entry['filtered_content'][:3000]}")
                if len(entry.get("filtered_content", "")) > 3000:
                    print(f"\n... [{len(entry['filtered_content'])} chars]")
            elif "content" in entry:
                print(f"\n--- 容 ---\n{entry['content'][:2000]}")
                if len(entry.get("content", "")) > 2000:
                    print(f"\n... [{len(entry['content'])} chars]")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            if "available_semantic" in result:
                print(f"Available: {', '.join(result['available_semantic'])}")
            sys.exit(1)


if __name__ == "__main__":
    main()
