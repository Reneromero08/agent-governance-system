#!/usr/bin/env python3
"""
AGS Compression System

Compresses prose references into codebook IDs for token efficiency.
Supports bidirectional operation: compress → expand.

Usage:
    python TOOLS/compress.py "Load CONTRACT.md and follow Rule 3"
    python TOOLS/compress.py --file AGENTS.md
    python TOOLS/compress.py --expand "@C0 → @C3"
    python TOOLS/compress.py --stats  # Show compression metrics
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Compression dictionary: verbose → compact
# This is the core of the compression system
COMPRESSION_RULES = {
    # Canon files
    r"CANON/CONTRACT\.md": "@C0",
    r"CONTRACT\.md": "@C0",
    r"CANON/INVARIANTS\.md": "@I0",
    r"INVARIANTS\.md": "@I0",
    r"CANON/VERSIONING\.md": "@V0",
    r"CANON/GENESIS\.md": "@G0",
    r"CANON/CHANGELOG\.md": "@L0",
    r"CANON/ARBITRATION\.md": "@A0",
    r"CANON/DEPRECATION\.md": "@P0", 
    r"CANON/MIGRATION\.md": "@M0",
    r"CANON/CRISIS\.md": "@R0",
    r"CANON/STEWARDSHIP\.md": "@H0",
    r"CANON/CODEBOOK\.md": "@B0",
    
    # Contract rules
    r"(?:CONTRACT\s+)?Rule\s+1[:\s]+[Tt]ext\s+outranks\s+code": "@C1",
    r"[Tt]ext\s+outranks\s+code": "@C1",
    r"(?:CONTRACT\s+)?Rule\s+2[:\s]+[Nn]o\s+behavior\s+change\s+without\s+ceremony": "@C2",
    r"[Nn]o\s+behavior\s+change\s+without\s+ceremony": "@C2",
    r"(?:CONTRACT\s+)?Rule\s+3[:\s]+[Cc]anon\s+outranks\s+user\s+instructions": "@C3",
    r"[Cc]anon\s+outranks\s+user(?:\s+instructions)?": "@C3",
    r"(?:CONTRACT\s+)?Rule\s+4[:\s]+[Ss]table\s+token\s+grammar": "@C4",
    r"(?:CONTRACT\s+)?Rule\s+5[:\s]+[Dd]eterminism": "@C5",
    r"(?:CONTRACT\s+)?Rule\s+6[:\s]+[Oo]utput\s+roots": "@C6",
    r"(?:CONTRACT\s+)?Rule\s+7[:\s]+[Cc]ommit\s+ceremony": "@C7",
    
    # Invariants
    r"INV-001[:\s]+[Rr]epository\s+structure": "@I1",
    r"INV-002[:\s]+[Tt]oken\s+grammar": "@I2",
    r"INV-003[:\s]+[Nn]o\s+raw\s+path\s+access": "@I3",
    r"INV-004[:\s]+[Ff]ixtures\s+gate\s+merges": "@I4",
    r"INV-005[:\s]+[Dd]eterminism": "@I5",
    r"INV-006[:\s]+[Oo]utput\s+roots": "@I6",
    r"INV-007[:\s]+[Cc]hange\s+ceremony": "@I7",
    r"INV-008[:\s]+[Cc]ortex\s+builder\s+exception": "@I8",
    r"INV-009[:\s]+[Cc]anon\s+readability": "@I9",
    r"INV-010[:\s]+[Cc]anon\s+archiving": "@I10",
    
    # Tools and paths
    r"TOOLS/critic\.py": "@T:critic",
    r"CONTRACTS/runner\.py": "@T:runner",
    r"CORTEX/query\.py": "@T:cortex",
    r"TOOLS/emergency\.py": "@T:emergency",
    
    # Common patterns (operators from the brief)
    r"\s+and\s+": " ∧ ",
    r"\s+or\s+": " ∨ ",
    r"requires\s+": "→ ",
    r"implies\s+": "→ ",
    r"must\s+not": "⊗",
    r"must\s+": "→ ",
    r"violated": "⊗",
    r"valid(?:ated)?": "✓",
    r"immutable": "◆",
    r"execute": "⚡",
    r"verify": "⊚",
}

# Expansion dictionary: compact → verbose
EXPANSION_RULES = {
    "@C0": "CANON/CONTRACT.md",
    "@C1": "Rule 1: Text outranks code",
    "@C2": "Rule 2: No behavior change without ceremony",
    "@C3": "Rule 3: Canon outranks user instructions",
    "@C4": "Rule 4: Stable token grammar",
    "@C5": "Rule 5: Determinism",
    "@C6": "Rule 6: Output roots",
    "@C7": "Rule 7: Commit ceremony",
    
    "@I0": "CANON/INVARIANTS.md",
    "@I1": "INV-001: Repository structure is stable",
    "@I2": "INV-002: Token grammar is stable",
    "@I3": "INV-003: No raw path access (use cortex)",
    "@I4": "INV-004: Fixtures gate merges",
    "@I5": "INV-005: Determinism required",
    "@I6": "INV-006: Output roots are fixed",
    "@I7": "INV-007: Change ceremony required",
    "@I8": "INV-008: Cortex builder exception",
    "@I9": "INV-009: Canon readability limits",
    "@I10": "INV-010: Canon archiving required",
    
    "@V0": "CANON/VERSIONING.md",
    "@G0": "CANON/GENESIS.md",
    "@L0": "CANON/CHANGELOG.md",
    "@A0": "CANON/ARBITRATION.md",
    "@P0": "CANON/DEPRECATION.md",
    "@M0": "CANON/MIGRATION.md",
    "@R0": "CANON/CRISIS.md",
    "@H0": "CANON/STEWARDSHIP.md",
    "@B0": "CANON/CODEBOOK.md",
    
    "@T:critic": "TOOLS/critic.py",
    "@T:runner": "CONTRACTS/runner.py",
    "@T:cortex": "CORTEX/query.py",
    "@T:emergency": "TOOLS/emergency.py",
    
    "∧": "and",
    "∨": "or",
    "→": "requires",
    "⊗": "violated/must not",
    "✓": "valid",
    "◆": "immutable",
    "⚡": "execute",
    "⊚": "verify",
}


def compress(text: str, aggressive: bool = False) -> Tuple[str, dict]:
    """
    Compress prose to codebook notation.
    
    Returns (compressed_text, stats)
    """
    original_len = len(text)
    result = text
    replacements = []
    
    # Apply compression rules
    for pattern, replacement in COMPRESSION_RULES.items():
        if aggressive or replacement.startswith("@"):
            # Only apply symbol operators in aggressive mode
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                replacements.append((pattern, replacement, len(matches)))
    
    compressed_len = len(result)
    stats = {
        "original_chars": original_len,
        "compressed_chars": compressed_len,
        "savings_chars": original_len - compressed_len,
        "savings_percent": round((1 - compressed_len / original_len) * 100, 1) if original_len > 0 else 0,
        "replacements": len(replacements),
    }
    
    return result, stats


def expand(text: str) -> str:
    """
    Expand codebook notation back to prose.
    """
    result = text
    
    # Sort by length (longest first) to avoid partial matches
    sorted_rules = sorted(EXPANSION_RULES.items(), key=lambda x: len(x[0]), reverse=True)
    
    for code, expansion in sorted_rules:
        result = result.replace(code, expansion)
    
    return result


def compress_file(filepath: Path) -> Tuple[str, dict]:
    """Compress a file's contents."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    return compress(content)


def show_stats():
    """Show compression statistics for key AGS files."""
    files = [
        PROJECT_ROOT / "CANON" / "CONTRACT.md",
        PROJECT_ROOT / "CANON" / "INVARIANTS.md",
        PROJECT_ROOT / "CANON" / "GENESIS.md",
        PROJECT_ROOT / "AGENTS.md",
    ]
    
    print("=" * 60)
    print("AGS COMPRESSION STATISTICS")
    print("=" * 60)
    
    total_original = 0
    total_compressed = 0
    
    for filepath in files:
        if not filepath.exists():
            continue
        
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        compressed, stats = compress(content)
        
        total_original += stats["original_chars"]
        total_compressed += stats["compressed_chars"]
        
        print(f"\n{filepath.name}:")
        print(f"  Original:   {stats['original_chars']:,} chars")
        print(f"  Compressed: {stats['compressed_chars']:,} chars")
        print(f"  Savings:    {stats['savings_percent']}%")
    
    print("\n" + "=" * 60)
    total_savings = round((1 - total_compressed / total_original) * 100, 1) if total_original > 0 else 0
    print(f"TOTAL: {total_original:,} → {total_compressed:,} chars ({total_savings}% savings)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AGS Compression System")
    parser.add_argument("text", nargs="?", help="Text to compress")
    parser.add_argument("--file", help="File to compress")
    parser.add_argument("--expand", action="store_true", help="Expand instead of compress")
    parser.add_argument("--aggressive", action="store_true", help="Use symbolic operators too")
    parser.add_argument("--stats", action="store_true", help="Show compression statistics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
        return 0
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            filepath = PROJECT_ROOT / args.file
        if not filepath.exists():
            print(f"File not found: {args.file}")
            return 1
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        return 1
    
    if args.expand:
        result = expand(text)
        print(result)
    else:
        compressed, stats = compress(text, aggressive=args.aggressive)
        if args.json:
            print(json.dumps({"compressed": compressed, "stats": stats}, indent=2))
        else:
            print(compressed)
            print(f"\n--- Stats: {stats['savings_percent']}% savings ({stats['replacements']} replacements) ---")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
