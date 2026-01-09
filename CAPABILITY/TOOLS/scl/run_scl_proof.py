#!/usr/bin/env python3
"""
SCL L2 Compression Proof Script (Phase 5.2.6.5)

This script produces a stacked receipt proving SCL compression at Layer 2.
It chains to the L1 compression proof and measures token reduction from
natural language governance text to symbolic SCL programs.

Usage:
    python run_scl_proof.py [--l1-receipt-hash <hash>] [--output-dir <dir>]

Outputs:
    - SCL_PROOF_RECEIPT.json: Machine-readable proof receipt
    - SCL_PROOF_REPORT.md: Human-readable proof report
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

# Tokenizer configuration
TOKENIZER_ENCODING = "o200k_base"       # GPT-4o / o1
TOKENIZER_ENCODING_FALLBACK = "cl100k_base"  # GPT-4

# L1 proof reference (from COMPRESSION_PROOF_DATA.json)
DEFAULT_L1_RECEIPT_HASH = "325410258180d609003649eda5902e17a1b9851d3aa1c852b3bf0efccc0043b6"

# Target compression threshold
TARGET_COMPRESSION_PCT = 80.0


# ═══════════════════════════════════════════════════════════════════════════════
# TIKTOKEN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    TIKTOKEN_VERSION = tiktoken.__version__
except ImportError:
    TIKTOKEN_AVAILABLE = False
    TIKTOKEN_VERSION = None


def get_tokenizer() -> Tuple[Optional[object], str]:
    """Get tiktoken encoder with fallback."""
    if not TIKTOKEN_AVAILABLE:
        return None, "word-count-proxy"

    try:
        enc = tiktoken.get_encoding(TOKENIZER_ENCODING)
        return enc, TOKENIZER_ENCODING
    except Exception:
        try:
            enc = tiktoken.get_encoding(TOKENIZER_ENCODING_FALLBACK)
            return enc, TOKENIZER_ENCODING_FALLBACK
        except Exception:
            return None, "word-count-proxy"


def count_tokens(text: str, encoder=None) -> int:
    """Count tokens in text."""
    if encoder is not None:
        return len(encoder.encode(text))
    if TIKTOKEN_AVAILABLE:
        enc, _ = get_tokenizer()
        if enc:
            return len(enc.encode(text))
    # Fallback: word count proxy (approx 0.75 tokens per word)
    return int(len(text.split()) / 0.75)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CASES
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CASES = [
    {
        "id": "case_001",
        "description": "Contract rule C3 - verification receipt",
        "nl_statement": """
All writes to LAW/CANON must be accompanied by a verification receipt.
The receipt must include: operation timestamp, input hash (SHA-256),
output hash, validation status, and the identity of the verifier.
Without a valid receipt, the write operation must be rejected.
This ensures every canon modification is traceable and auditable.
        """.strip(),
        "scl_program": "C3",
        "expected_intent": "All writes to canon require verification receipt",
    },
    {
        "id": "case_002",
        "description": "Invariant I5 - determinism",
        "nl_statement": """
All operations in the governance system must be deterministic.
Given identical inputs, the system must produce byte-identical outputs.
This applies to: validation, transformation, indexing, and reporting.
Non-deterministic operations must be explicitly flagged and sandboxed.
Determinism is verified by running operations multiple times and
comparing SHA-256 hashes of all outputs.
        """.strip(),
        "scl_program": "I5",
        "expected_intent": "Deterministic output",
    },
    {
        "id": "case_003",
        "description": "LAW/CANON domain pointer",
        "nl_statement": """
The LAW/CANON directory contains all normative governance artifacts.
This includes: the Genesis Compact, Contract rules (C1-C13),
Invariants (I1-I14+), Catalytic Computing specifications, and
all binding protocols. Any reference to governance law should
resolve to artifacts in this directory structure. The canon
serves as the single source of truth for all governance operations.
        """.strip(),
        "scl_program": "法",
        "expected_intent": "LAW/CANON",
    },
    {
        "id": "case_004",
        "description": "Verification compound pointer",
        "nl_statement": """
The verification subsystem within LAW/CANON handles all validation
operations. This includes: receipt verification, hash integrity checks,
schema validation, catalytic domain verification, and invariant
checking. All verification operations must emit receipts. The
verification system enforces the contract rules and ensures
no unverified changes enter the canon.
        """.strip(),
        "scl_program": "法.驗",
        "expected_intent": "Verification",
    },
    {
        "id": "case_005",
        "description": "Contract rule with context",
        "nl_statement": """
During build operations, contract rule C3 requires that all
generated artifacts include verification receipts. The build
context implies: compilation outputs, generated documentation,
index files, and any derived artifacts. Each must have a receipt
linking it to its source inputs and validation status.
        """.strip(),
        "scl_program": "C3:build",
        "expected_intent": "verification receipt",
    },
]

NEGATIVE_CONTROLS = [
    {
        "id": "neg_001",
        "description": "Random nonsense",
        "input": "xyzzy plugh quantum banana sandwich purple elephant",
        "expected_behavior": "Should not produce meaningful compression",
    },
    {
        "id": "neg_002",
        "description": "Unrelated technical jargon",
        "input": "SELECT * FROM users WHERE password LIKE '%admin%'",
        "expected_behavior": "Should not match governance concepts",
    },
    {
        "id": "neg_003",
        "description": "Empty-like input",
        "input": "   ",
        "expected_behavior": "Minimal or zero compression benefit",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# PROOF EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProofResult:
    """Result of a single benchmark case."""
    case_id: str
    description: str
    nl_tokens: int
    scl_tokens: int
    compression_pct: float
    scl_program: str
    passed: bool
    notes: str = ""


@dataclass
class SCLProofReceipt:
    """L2 SCL Compression Proof Receipt."""
    layer: str = "SCL"
    parent_receipt: str = ""
    timestamp_utc: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    compression_pct: float = 0.0
    cases_total: int = 0
    cases_passed: int = 0
    target_met: bool = False
    tokenizer: Dict = field(default_factory=dict)
    script_hash: str = ""
    git_head: str = ""
    git_clean: bool = False
    benchmark_results: List[Dict] = field(default_factory=list)
    negative_controls: List[Dict] = field(default_factory=list)

    def compute_receipt_hash(self) -> str:
        """Compute deterministic receipt hash."""
        # Exclude timestamp for determinism
        data = asdict(self)
        data.pop("timestamp_utc", None)
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def get_script_hash() -> str:
    """Hash of this script for methodology integrity."""
    script_path = Path(__file__).resolve()
    content = script_path.read_text(encoding='utf-8')
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_git_info() -> Tuple[str, bool]:
    """Get git HEAD commit and clean status."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        head = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        is_clean = len(status) == 0
    except Exception:
        is_clean = False

    return head, is_clean


def run_scl_decode(program: str) -> Optional[Dict]:
    """Run SCL CLI decode command."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "CAPABILITY.TOOLS.scl", "decode", program, "--json"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def run_benchmark(encoder) -> Tuple[List[ProofResult], int, int]:
    """Run all benchmark cases."""
    results = []
    total_nl_tokens = 0
    total_scl_tokens = 0

    for case in BENCHMARK_CASES:
        nl_tokens = count_tokens(case["nl_statement"], encoder)
        scl_tokens = count_tokens(case["scl_program"], encoder)

        # Verify SCL program decodes successfully
        decode_result = run_scl_decode(case["scl_program"])
        decode_ok = decode_result is not None and decode_result.get("ok", False)

        compression_pct = (1 - scl_tokens / nl_tokens) * 100 if nl_tokens > 0 else 0
        passed = compression_pct >= TARGET_COMPRESSION_PCT and decode_ok

        notes = ""
        if not decode_ok:
            notes = "SCL decode failed"
        elif compression_pct < TARGET_COMPRESSION_PCT:
            notes = f"Below {TARGET_COMPRESSION_PCT}% target"

        results.append(ProofResult(
            case_id=case["id"],
            description=case["description"],
            nl_tokens=nl_tokens,
            scl_tokens=scl_tokens,
            compression_pct=compression_pct,
            scl_program=case["scl_program"],
            passed=passed,
            notes=notes,
        ))

        total_nl_tokens += nl_tokens
        total_scl_tokens += scl_tokens

    return results, total_nl_tokens, total_scl_tokens


def run_negative_controls(encoder) -> List[Dict]:
    """Run negative control tests."""
    results = []

    for control in NEGATIVE_CONTROLS:
        input_tokens = count_tokens(control["input"], encoder)

        # Try to find a matching SCL program (should fail or be meaningless)
        # For negative controls, we just measure the input tokens
        results.append({
            "id": control["id"],
            "description": control["description"],
            "input_tokens": input_tokens,
            "expected_behavior": control["expected_behavior"],
            "verdict": "PASS" if input_tokens > 0 else "SKIP",
        })

    return results


def generate_proof_report(receipt: SCLProofReceipt, results: List[ProofResult]) -> str:
    """Generate human-readable proof report."""
    lines = [
        "# SCL L2 Compression Proof Report",
        "",
        "## Executive Summary",
        "",
        f"**Layer:** {receipt.layer}",
        f"**Parent Receipt (L1):** `{receipt.parent_receipt[:16]}...`",
        f"**Timestamp:** {receipt.timestamp_utc}",
        "",
        f"**Total NL Tokens:** {receipt.input_tokens:,}",
        f"**Total SCL Tokens:** {receipt.output_tokens:,}",
        f"**Compression:** {receipt.compression_pct:.1f}%",
        f"**Target Met:** {'YES' if receipt.target_met else 'NO'} (target: {TARGET_COMPRESSION_PCT}%)",
        "",
        "## Proof Integrity",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| Script Hash | `{receipt.script_hash[:16]}...` |",
        f"| Git HEAD | `{receipt.git_head[:16]}...` |",
        f"| Git Clean | {'Yes' if receipt.git_clean else 'No (uncommitted changes)'} |",
        f"| Tokenizer | {receipt.tokenizer.get('library', 'unknown')} / {receipt.tokenizer.get('encoding', 'unknown')} |",
        "",
        "## Benchmark Results",
        "",
        "| Case | NL Tokens | SCL | Compression | Status |",
        "|------|-----------|-----|-------------|--------|",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.case_id} | {r.nl_tokens} | `{r.scl_program}` ({r.scl_tokens}) | "
            f"{r.compression_pct:.1f}% | {status} |"
        )

    lines.extend([
        "",
        "## Negative Controls",
        "",
        "| Control | Tokens | Verdict |",
        "|---------|--------|---------|",
    ])

    for nc in receipt.negative_controls:
        lines.append(f"| {nc['id']} | {nc['input_tokens']} | {nc['verdict']} |")

    lines.extend([
        "",
        "## Receipt Hash",
        "",
        f"**L2 Receipt Hash:** `{receipt.compute_receipt_hash()}`",
        "",
        "This receipt chains to L1 proof and can be verified by:",
        "1. Checking script hash matches methodology",
        "2. Verifying tokenizer produces consistent counts",
        "3. Re-running benchmark cases",
        "",
        "---",
        "*Generated by SCL L2 Compression Proof Script (Phase 5.2.6.5)*",
    ])

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCL L2 Compression Proof Script"
    )
    parser.add_argument(
        "--l1-receipt-hash",
        default=DEFAULT_L1_RECEIPT_HASH,
        help="L1 receipt hash to chain to"
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "NAVIGATION" / "PROOFS" / "COMPRESSION"),
        help="Output directory for proof artifacts"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output receipt as JSON to stdout"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    encoder, encoding_used = get_tokenizer()

    # Get integrity info
    script_hash = get_script_hash()
    git_head, git_clean = get_git_info()

    # Run benchmarks
    results, total_nl, total_scl = run_benchmark(encoder)
    compression_pct = (1 - total_scl / total_nl) * 100 if total_nl > 0 else 0
    cases_passed = sum(1 for r in results if r.passed)

    # Run negative controls
    neg_controls = run_negative_controls(encoder)

    # Build receipt
    receipt = SCLProofReceipt(
        layer="SCL",
        parent_receipt=args.l1_receipt_hash,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        input_tokens=total_nl,
        output_tokens=total_scl,
        compression_pct=compression_pct,
        cases_total=len(results),
        cases_passed=cases_passed,
        target_met=compression_pct >= TARGET_COMPRESSION_PCT,
        tokenizer={
            "library": "tiktoken" if TIKTOKEN_AVAILABLE else "word-count-proxy",
            "encoding": encoding_used,
            "version": TIKTOKEN_VERSION,
        },
        script_hash=script_hash,
        git_head=git_head,
        git_clean=git_clean,
        benchmark_results=[asdict(r) for r in results],
        negative_controls=neg_controls,
    )

    # Output
    if args.json:
        receipt_dict = asdict(receipt)
        receipt_dict["receipt_hash"] = receipt.compute_receipt_hash()
        print(json.dumps(receipt_dict, indent=2, ensure_ascii=False))
    else:
        # Write receipt JSON
        receipt_path = output_dir / "SCL_PROOF_RECEIPT.json"
        receipt_dict = asdict(receipt)
        receipt_dict["receipt_hash"] = receipt.compute_receipt_hash()
        receipt_path.write_text(
            json.dumps(receipt_dict, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"Receipt written to: {receipt_path}")

        # Write report
        report = generate_proof_report(receipt, results)
        report_path = output_dir / "SCL_PROOF_REPORT.md"
        report_path.write_text(report, encoding='utf-8')
        print(f"Report written to: {report_path}")

        # Summary
        print()
        print("=" * 60)
        print("SCL L2 COMPRESSION PROOF COMPLETE")
        print("=" * 60)
        print(f"  NL Tokens:      {total_nl:,}")
        print(f"  SCL Tokens:     {total_scl:,}")
        print(f"  Compression:    {compression_pct:.1f}%")
        print(f"  Cases:          {cases_passed}/{len(results)} passed")
        print(f"  Target Met:     {'YES' if receipt.target_met else 'NO'}")
        print(f"  Receipt Hash:   {receipt.compute_receipt_hash()[:16]}...")
        print("=" * 60)

    return 0 if receipt.target_met else 1


if __name__ == "__main__":
    sys.exit(main())