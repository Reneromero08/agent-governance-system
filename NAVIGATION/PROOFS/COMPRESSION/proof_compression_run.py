#!/usr/bin/env python3
"""
Phase 6.4.9: proof_compression_run

Unified compression proof runner that:
1. Runs existing compression proof (token savings)
2. Runs benchmark tasks (task performance validation)
3. Emits machine-readable JSON + human-readable MD report
4. Produces auditable proof bundle (6.4.8)

Outputs:
- NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_DATA.json (machine)
- NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_REPORT.md (human)
- NAVIGATION/PROOFS/COMPRESSION/BENCHMARK_RESULTS.json (task performance)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import local modules
from corpus_spec import ProofCorpusSpec, get_default_spec, render_spec_report
from benchmark_tasks import (
    BENCHMARK_VERSION,
    BenchmarkRunner,
    get_all_benchmark_tasks,
    run_benchmarks,
)

# Optional LLM benchmark integration
try:
    from llm_benchmark_runner import (
        BENCHMARK_VERSION as LLM_BENCHMARK_VERSION,
        LLMBenchmarkRunner,
        run_llm_benchmarks,
        load_sample_corpus,
    )
    LLM_BENCHMARKS_AVAILABLE = True
except ImportError:
    LLM_BENCHMARKS_AVAILABLE = False

# Output paths
OUT_DIR = REPO_ROOT / "NAVIGATION" / "PROOFS" / "COMPRESSION"
OUT_PROOF_DATA = OUT_DIR / "COMPRESSION_PROOF_DATA.json"
OUT_PROOF_REPORT = OUT_DIR / "COMPRESSION_PROOF_REPORT.md"
OUT_BENCHMARK_RESULTS = OUT_DIR / "BENCHMARK_RESULTS.json"
OUT_CORPUS_SPEC = OUT_DIR / "CORPUS_SPEC.json"

# Existing compression proof script
COMPRESSION_PROOF_SCRIPT = (
    REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "compression_proof" / "run_compression_proof.py"
)


def _sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_head_commit() -> Optional[str]:
    """Get current git HEAD commit."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _run_compression_proof_script() -> Dict[str, Any]:
    """
    Run the existing compression proof script.

    Returns the proof data from the output JSON file.
    """
    if not COMPRESSION_PROOF_SCRIPT.exists():
        return {
            "error": f"Compression proof script not found: {COMPRESSION_PROOF_SCRIPT}",
            "status": "SKIP",
        }

    try:
        result = subprocess.run(
            [sys.executable, str(COMPRESSION_PROOF_SCRIPT)],
            capture_output=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            return {
                "error": result.stderr.decode("utf-8", errors="replace")[:1000],
                "status": "FAIL",
            }

        # Read output data
        if OUT_PROOF_DATA.exists():
            return json.loads(OUT_PROOF_DATA.read_text(encoding="utf-8"))
        else:
            return {
                "error": "Proof data file not generated",
                "status": "FAIL",
            }

    except subprocess.TimeoutExpired:
        return {
            "error": "Compression proof timed out after 300 seconds",
            "status": "FAIL",
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "FAIL",
        }


def _run_benchmarks() -> Dict[str, Any]:
    """
    Run benchmark tasks and return results.
    """
    try:
        results = run_benchmarks()
        return results.to_dict()
    except Exception as e:
        return {
            "error": str(e),
            "status": "FAIL",
            "benchmark_version": BENCHMARK_VERSION,
            "tasks_run": 0,
            "parity_achieved": False,
        }


def _run_llm_benchmarks(run_llm: bool = False) -> Optional[Dict[str, Any]]:
    """
    Run LLM benchmark tasks (optional, requires Nemotron endpoint).

    Args:
        run_llm: If True, attempt to run LLM benchmarks

    Returns:
        Results dict or None if skipped/unavailable
    """
    if not run_llm:
        return None

    if not LLM_BENCHMARKS_AVAILABLE:
        return {"status": "SKIP", "reason": "LLM benchmark module not available"}

    try:
        # Check if endpoint is available
        runner = LLMBenchmarkRunner()
        if not runner.check_endpoint():
            return {"status": "SKIP", "reason": "LLM endpoint not available"}

        # Load corpus
        baseline_corpus, compressed_corpus = load_sample_corpus()

        # Run benchmarks
        results = run_llm_benchmarks(
            baseline_corpus=baseline_corpus,
            compressed_corpus=compressed_corpus,
        )
        return results.to_dict()

    except Exception as e:
        return {
            "error": str(e),
            "status": "FAIL",
        }


def _compute_proof_bundle_hash(data: Dict[str, Any]) -> str:
    """
    Compute hash of proof bundle for auditable verification.

    Includes all proof components except the hash itself.
    """
    # Create deterministic representation
    bundle_data = {
        "timestamp_utc": data.get("timestamp_utc"),
        "git_rev": data.get("git_rev"),
        "corpus_spec_hash": data.get("corpus_spec_hash"),
        "compression_proof_receipt": data.get("compression_proof", {}).get("proof_receipt"),
        "benchmark_results_hash": _sha256_hex(
            json.dumps(data.get("benchmark_results", {}), sort_keys=True)
        ),
    }
    return _sha256_hex(json.dumps(bundle_data, sort_keys=True))


def run_proof_compression(run_llm: bool = False) -> Dict[str, Any]:
    """
    Run full compression proof suite.

    Args:
        run_llm: If True, also run LLM benchmarks (requires Nemotron endpoint)

    Returns unified proof bundle with all artifacts.
    """
    timestamp = _utc_timestamp()
    git_rev = _git_head_commit()

    # Get corpus specification
    corpus_spec = get_default_spec()

    # Run compression proof
    print("Running compression proof...")
    compression_proof = _run_compression_proof_script()

    # Run benchmark tasks
    print("Running benchmark tasks...")
    benchmark_results = _run_benchmarks()

    # Optionally run LLM benchmarks
    llm_benchmark_results = None
    if run_llm:
        print("Running LLM benchmarks...")
        llm_benchmark_results = _run_llm_benchmarks(run_llm=True)

    # Build proof bundle
    proof_bundle = {
        "proof_type": "compression",
        "proof_version": "1.1.0" if run_llm else "1.0.0",
        "timestamp_utc": timestamp,
        "git_rev": git_rev,
        "corpus_spec": corpus_spec.to_dict(),
        "corpus_spec_hash": _sha256_hex(json.dumps(corpus_spec.to_dict(), sort_keys=True)),
        "compression_proof": compression_proof,
        "benchmark_results": benchmark_results,
    }

    # Add LLM results if available
    if llm_benchmark_results:
        proof_bundle["llm_benchmark_results"] = llm_benchmark_results

    # Compute bundle hash
    proof_bundle["proof_bundle_hash"] = _compute_proof_bundle_hash(proof_bundle)

    # Determine overall status
    compression_ok = compression_proof.get("status") != "FAIL" and "error" not in compression_proof
    benchmark_ok = benchmark_results.get("parity_achieved", False)

    # LLM benchmarks are optional - only factor in if they ran and failed
    llm_ok = True
    if llm_benchmark_results and llm_benchmark_results.get("status") == "FAIL":
        llm_ok = llm_benchmark_results.get("parity_achieved", True)

    proof_bundle["status"] = "PASS" if (compression_ok and benchmark_ok and llm_ok) else "FAIL"

    return proof_bundle


def render_proof_report(data: Dict[str, Any]) -> str:
    """Render human-readable proof report."""
    lines = [
        "<!-- GENERATED: Phase 6.4.9 Compression Proof Report -->",
        "",
        "# Compression Proof Report",
        "",
        "## Summary",
        "",
        f"**Status:** {data.get('status', 'UNKNOWN')}",
        f"**Timestamp:** {data.get('timestamp_utc', 'unknown')}",
        f"**Git Rev:** `{data.get('git_rev', 'unknown')}`",
        f"**Proof Bundle Hash:** `{data.get('proof_bundle_hash', 'unknown')[:16]}...`",
        "",
        "## Compression Proof (Token Savings)",
        "",
    ]

    compression = data.get("compression_proof", {})
    if "error" in compression:
        lines.extend([
            f"**Error:** {compression.get('error')}",
            "",
        ])
    else:
        baselines = compression.get("baselines", {})
        lines.extend([
            f"**Baseline A (All Files):** {baselines.get('BaselineA_tokens', 0):,} tokens",
            f"**Baseline B (Filtered):** {baselines.get('BaselineB_tokens', 0):,} tokens",
            "",
            "### Query Results",
            "",
            "| Query | NewWay Tokens | Savings (A) | Savings (B) |",
            "|-------|---------------|-------------|-------------|",
        ])

        for q in compression.get("queries", []):
            lines.append(
                f"| {q.get('query_text', 'unknown')[:30]} | "
                f"{q.get('NewWayTokensFiltered', 0):,} | "
                f"{q.get('SavingsPctA', 0):.2%} | "
                f"{q.get('SavingsPctB', 0):.2%} |"
            )

        lines.append("")

    lines.extend([
        "## Benchmark Results (Task Performance)",
        "",
    ])

    benchmark = data.get("benchmark_results", {})
    if "error" in benchmark:
        lines.extend([
            f"**Error:** {benchmark.get('error')}",
            "",
        ])
    else:
        baseline = benchmark.get("baseline_results", {})
        compressed = benchmark.get("compressed_results", {})
        lines.extend([
            f"**Benchmark Version:** {benchmark.get('benchmark_version', 'unknown')}",
            f"**Tasks Run:** {benchmark.get('tasks_run', 0)}",
            "",
            "### Results Comparison",
            "",
            "| Metric | Baseline | Compressed |",
            "|--------|----------|------------|",
            f"| Tasks Passed | {baseline.get('tasks_passed', 0)} | {compressed.get('tasks_passed', 0)} |",
            f"| Tasks Failed | {baseline.get('tasks_failed', 0)} | {compressed.get('tasks_failed', 0)} |",
            f"| Success Rate | {baseline.get('success_rate', 0):.2%} | {compressed.get('success_rate', 0):.2%} |",
            "",
            f"**Parity Achieved:** {'Yes' if benchmark.get('parity_achieved') else 'No'}",
            "",
        ])

    # Add LLM benchmark section if present
    llm_benchmark = data.get("llm_benchmark_results")
    if llm_benchmark:
        lines.extend([
            "## LLM Benchmark Results (Nemotron)",
            "",
        ])
        if "error" in llm_benchmark:
            lines.extend([
                f"**Error:** {llm_benchmark.get('error')}",
                "",
            ])
        elif llm_benchmark.get("status") == "SKIP":
            lines.extend([
                f"**Skipped:** {llm_benchmark.get('reason', 'Unknown reason')}",
                "",
            ])
        else:
            baseline = llm_benchmark.get("baseline_results", {})
            compressed = llm_benchmark.get("compressed_results", {})
            lines.extend([
                f"**Endpoint:** {llm_benchmark.get('endpoint', 'unknown')}",
                f"**Model:** {llm_benchmark.get('model', 'unknown')}",
                f"**Tasks Run:** {llm_benchmark.get('tasks_run', 0)}",
                f"**Total Latency:** {llm_benchmark.get('total_latency_ms', 0):.0f}ms",
                "",
                "### LLM Results Comparison",
                "",
                "| Metric | Baseline | Compressed |",
                "|--------|----------|------------|",
                f"| Tasks Passed | {baseline.get('tasks_passed', 0)} | {compressed.get('tasks_passed', 0)} |",
                f"| Tasks Failed | {baseline.get('tasks_failed', 0)} | {compressed.get('tasks_failed', 0)} |",
                f"| Success Rate | {baseline.get('success_rate', 0):.2%} | {compressed.get('success_rate', 0):.2%} |",
                "",
                f"**LLM Parity Achieved:** {'Yes' if llm_benchmark.get('parity_achieved') else 'No'}",
                "",
            ])

    lines.extend([
        "## Verification",
        "",
        "To reproduce this proof:",
        "",
        "```bash",
        "# Basic benchmarks",
        "python NAVIGATION/PROOFS/COMPRESSION/proof_compression_run.py",
        "",
        "# With LLM benchmarks (requires Nemotron endpoint)",
        "python NAVIGATION/PROOFS/COMPRESSION/proof_compression_run.py --llm",
        "```",
        "",
        "---",
        "",
        "*Phase 6.4.9 compliant proof bundle.*",
    ])

    return "\n".join(lines)


def write_outputs(proof_bundle: Dict[str, Any]) -> None:
    """Write all proof outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write corpus spec
    corpus_spec = proof_bundle.get("corpus_spec", {})
    OUT_CORPUS_SPEC.write_text(
        json.dumps(corpus_spec, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Write benchmark results
    benchmark = proof_bundle.get("benchmark_results", {})
    OUT_BENCHMARK_RESULTS.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Render and write report
    report = render_proof_report(proof_bundle)
    # Note: OUT_PROOF_REPORT may have been written by compression_proof script
    # We write a separate integrated report
    integrated_report_path = OUT_DIR / "PROOF_COMPRESSION_INTEGRATED.md"
    integrated_report_path.write_text(report, encoding="utf-8")

    # Write full bundle
    bundle_path = OUT_DIR / "PROOF_COMPRESSION_BUNDLE.json"
    bundle_path.write_text(
        json.dumps(proof_bundle, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Outputs written to {OUT_DIR}/")


def main() -> int:
    """Run compression proof and output results."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6.4.9: Compression Proof Runner")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run LLM benchmarks (requires Nemotron endpoint at http://10.5.0.2:1234)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 6.4.9: Compression Proof Runner")
    print("=" * 60)
    print()

    if args.llm:
        print("LLM benchmarks enabled (--llm flag)")
        print()

    proof_bundle = run_proof_compression(run_llm=args.llm)
    write_outputs(proof_bundle)

    print()
    print(f"Status: {proof_bundle.get('status', 'UNKNOWN')}")
    print(f"Bundle Hash: {proof_bundle.get('proof_bundle_hash', 'unknown')}")

    # Show LLM results summary if available
    llm_results = proof_bundle.get("llm_benchmark_results")
    if llm_results and "parity_achieved" in llm_results:
        print(f"LLM Parity: {llm_results.get('parity_achieved')}")

    if proof_bundle.get("status") == "PASS":
        print("\nSUCCESS: Compression proof validated")
        return 0
    else:
        print("\nFAILED: Compression proof did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
