#!/usr/bin/env python3
"""
TOKENIZER_ATLAS Generator (Phase 5.3.4)

Generates formal artifact tracking glyph/operator token counts across tokenizers.
Enforces single-token preference for semantic symbols.

Usage:
    python -m CAPABILITY.TOOLS.generate_tokenizer_atlas
    python CAPABILITY/TOOLS/generate_tokenizer_atlas.py --output LAW/CANON/SEMANTIC/TOKENIZER_ATLAS.json

Output:
    - TOKENIZER_ATLAS.json with symbol -> token_count mappings
    - Preferred glyph list (single-token symbols)
    - Warnings for multi-token symbols

Exit Codes:
    0: Success
    1: Multi-token symbols found in preferred set (CI gate failure)
    2: Tokenizer unavailable
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "LAW" / "CANON" / "SEMANTIC" / "TOKENIZER_ATLAS.json"

# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TOKENIZERS = {
    "cl100k_base": "GPT-4, GPT-3.5-turbo",
    "o200k_base": "GPT-4o, o1, Claude (proxy)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Core CJK semantic symbols (from SPC_SPEC.md Section 2.1)
CJK_SYMBOLS = [
    # Core domain pointers (measured compression)
    "法",  # LAW/CANON - 56,370x
    "真",  # THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md - 8,200x
    "契",  # CONTRACT.md - 4,100x
    "恆",  # INVARIANTS.md - 5,600x
    "驗",  # VERIFICATION.md - 3,800x
    # Governance operations
    "證",  # NAVIGATION/RECEIPTS - 12,000x
    "變",  # CATALYTIC - 8,500x
    "冊",  # CORTEX/db - 4,200x
    "錄",  # Index primitives - 15,000x
    "限",  # alias of 恆
    "許",  # permissions
    "禁",  # prohibitions
    "雜",  # TOKEN_RECEIPT_SPEC - 2,100x
    "復",  # restore skills - 3,600x
    # Validation operations
    "試",  # TESTBENCH - 42,000x
    "查",  # semantic search - 8,800x
    "載",  # codebook lookup - 4,500x
    "存",  # CAS - 6,200x
    "掃",  # scan/index - 5,100x
    "核",  # verification/integrity - 4,900x
    # Structural symbols
    "道",  # polysemic (path/principle/method)
    "圖",  # cross_ref_index - 3,400x
    "鏈",  # pipeline DAG - 4,100x
    "根",  # alias of 法
    "枝",  # THOUGHT/LAB - 28,000x
]

# Compound CJK symbols (composed with . operator)
COMPOUND_SYMBOLS = [
    "法.驗",  # law -> verification
    "法.契",  # law -> contract
    "證.雜",  # receipts -> token
    "冊.雜",  # registry -> token
]

# ASCII radicals (from SPC_SPEC.md Section 2.3)
RADICALS = ["C", "I", "V", "L", "G", "S", "R", "A", "J", "P"]

# Operators (from SPC_SPEC.md Section 2.3)
OPERATORS = ["*", "!", "?", "&", "|", "."]

# Numbered radicals (common usage)
NUMBERED_RADICALS = [
    "C3", "C7", "C8",  # Contract rules
    "I5", "I6",        # Invariants
]

# Preferred single-token symbols (MUST be single-token - CI gate enforced)
# These are verified single-token under BOTH cl100k_base and o200k_base
# NOTE: Original SPC_SPEC listed 法,真,契,恆,驗,證,試 but cl100k_base
#       doesn't support all CJK as single tokens. o200k_base has better coverage.
PREFERRED_SINGLE_TOKEN = [
    "法",  # LAW/CANON - single-token both
    "真",  # TRUTH - single-token both
    "限",  # INVARIANTS alias - single-token both
    "查",  # semantic search - single-token both
    "存",  # CAS - single-token both
    "核",  # verification - single-token both
    "道",  # path/principle - single-token both
]

# Symbols that are single-token under o200k_base only (GPT-4o, o1)
# These provide good compression on modern models but not cl100k_base
O200K_SINGLE_TOKEN = [
    "契", "驗", "證", "變", "冊", "錄", "許", "禁", "復",
    "試", "載", "掃", "圖", "鏈", "根", "枝",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_token_count(text: str, encoding_name: str) -> int:
    """
    Get token count for text under specified encoding.

    Args:
        text: Text to tokenize
        encoding_name: Tokenizer encoding (cl100k_base, o200k_base)

    Returns:
        Token count
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken")
    except Exception as e:
        raise RuntimeError(f"Tokenization failed for '{text}': {e}")


def get_token_ids(text: str, encoding_name: str) -> List[int]:
    """Get actual token IDs for debugging."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return enc.encode(text)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ATLAS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_atlas() -> Dict[str, Any]:
    """
    Generate the TOKENIZER_ATLAS artifact.

    Returns:
        Atlas dictionary with symbol mappings and metadata
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect all symbols
    all_symbols = (
        CJK_SYMBOLS +
        COMPOUND_SYMBOLS +
        RADICALS +
        OPERATORS +
        NUMBERED_RADICALS
    )

    # Deduplicate while preserving order
    seen = set()
    unique_symbols = []
    for s in all_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    # Build symbol mappings
    symbols: Dict[str, Dict[str, int]] = {}
    warnings: List[str] = []
    single_token_verified: List[str] = []
    multi_token_symbols: List[str] = []

    for symbol in unique_symbols:
        symbols[symbol] = {}
        is_single_token_all = True

        for encoding in TOKENIZERS.keys():
            try:
                count = get_token_count(symbol, encoding)
                symbols[symbol][encoding] = count

                if count > 1:
                    is_single_token_all = False
                    token_ids = get_token_ids(symbol, encoding)
                    warnings.append(
                        f"'{symbol}' is {count} tokens under {encoding} (IDs: {token_ids})"
                    )
            except Exception as e:
                symbols[symbol][encoding] = -1
                warnings.append(f"Failed to tokenize '{symbol}' with {encoding}: {e}")

        if is_single_token_all and symbol in CJK_SYMBOLS:
            single_token_verified.append(symbol)
        elif symbol in CJK_SYMBOLS:
            multi_token_symbols.append(symbol)

    # Compute preferred glyphs (single-token under all tokenizers)
    preferred_glyphs = [
        s for s in CJK_SYMBOLS
        if all(symbols.get(s, {}).get(enc, 0) == 1 for enc in TOKENIZERS.keys())
    ]

    # Build atlas
    atlas = {
        "version": "1.0.0",
        "schema": "TOKENIZER_ATLAS_V1",
        "generated_utc": timestamp,
        "generator": "CAPABILITY/TOOLS/generate_tokenizer_atlas.py",
        "phase": "5.3.4",
        "tokenizers": list(TOKENIZERS.keys()),
        "tokenizer_models": TOKENIZERS,
        "symbols": symbols,
        "symbol_categories": {
            "cjk_symbols": CJK_SYMBOLS,
            "compound_symbols": COMPOUND_SYMBOLS,
            "radicals": RADICALS,
            "operators": OPERATORS,
            "numbered_radicals": NUMBERED_RADICALS,
        },
        "preferred_glyphs": preferred_glyphs,
        "preferred_single_token_enforced": PREFERRED_SINGLE_TOKEN,
        "o200k_single_token_only": O200K_SINGLE_TOKEN,
        "statistics": {
            "total_symbols": len(unique_symbols),
            "single_token_cjk": len(single_token_verified),
            "multi_token_cjk": len(multi_token_symbols),
            "preferred_glyph_count": len(preferred_glyphs),
        },
        "warnings": warnings,
    }

    # Compute content hash for integrity
    content_for_hash = json.dumps(
        {k: v for k, v in atlas.items() if k not in ["generated_utc"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    atlas["content_hash"] = hashlib.sha256(content_for_hash.encode()).hexdigest()

    return atlas


def verify_preferred_single_token(atlas: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify all preferred single-token symbols are indeed single-token.

    This is the CI gate check.

    Returns:
        (passed, violations) tuple
    """
    violations = []

    for symbol in PREFERRED_SINGLE_TOKEN:
        if symbol not in atlas["symbols"]:
            violations.append(f"'{symbol}' not found in atlas")
            continue

        for encoding in atlas["tokenizers"]:
            count = atlas["symbols"][symbol].get(encoding, -1)
            if count != 1:
                violations.append(
                    f"VIOLATION: '{symbol}' is {count} tokens under {encoding} (expected 1)"
                )

    return len(violations) == 0, violations


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate TOKENIZER_ATLAS.json artifact",
        epilog="Phase 5.3.4 - Formal tokenizer tracking for SPC symbols"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output only JSON to stdout (no messages)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing atlas, don't regenerate"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any preferred symbol is multi-token (CI gate mode)"
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    # Verify-only mode
    if args.verify_only:
        if not output_path.exists():
            sys.stderr.write(f"[ERROR] Atlas not found: {output_path}\n")
            return 2

        with open(output_path, 'r', encoding='utf-8') as f:
            atlas = json.load(f)

        passed, violations = verify_preferred_single_token(atlas)

        if args.json:
            print(json.dumps({"passed": passed, "violations": violations}, indent=2))
        else:
            if passed:
                print(f"[PASS] All {len(PREFERRED_SINGLE_TOKEN)} preferred symbols are single-token")
            else:
                print(f"[FAIL] {len(violations)} violations found:")
                for v in violations:
                    print(f"  - {v}")

        return 0 if passed else 1

    # Generate atlas
    try:
        atlas = generate_atlas()
    except RuntimeError as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        return 2

    # Verify preferred symbols
    passed, violations = verify_preferred_single_token(atlas)

    # Output
    if args.json:
        print(json.dumps(atlas, indent=2, ensure_ascii=False))
    else:
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(atlas, f, indent=2, ensure_ascii=False)

        print(f"TOKENIZER_ATLAS Generator (Phase 5.3.4)")
        print("=" * 60)
        print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
        print(f"Generated: {atlas['generated_utc']}")
        print(f"Content Hash: {atlas['content_hash'][:16]}...")
        print()
        print("Statistics:")
        print(f"  Total symbols:      {atlas['statistics']['total_symbols']}")
        print(f"  Single-token CJK:   {atlas['statistics']['single_token_cjk']}")
        print(f"  Multi-token CJK:    {atlas['statistics']['multi_token_cjk']}")
        print(f"  Preferred glyphs:   {atlas['statistics']['preferred_glyph_count']}")
        print()

        if atlas['warnings']:
            print(f"Warnings ({len(atlas['warnings'])}):")
            for w in atlas['warnings'][:10]:
                print(f"  - {w}")
            if len(atlas['warnings']) > 10:
                print(f"  ... and {len(atlas['warnings']) - 10} more")
            print()

        # CI gate check
        if passed:
            print(f"[PASS] All {len(PREFERRED_SINGLE_TOKEN)} preferred symbols verified single-token")
        else:
            print(f"[FAIL] {len(violations)} violations in preferred set:")
            for v in violations:
                print(f"  - {v}")

        print()
        print("=" * 60)

    # Exit code for CI
    if args.strict and not passed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
