#!/usr/bin/env python3
"""
Q41 Phase 2: Langlands Test Runner

Orchestrates TIER 2 and TIER 5 tests:
- TIER 2.1: Semantic L-Functions
- TIER 2.2: Ramanujan Bound
- TIER 5.1: Arthur-Selberg Trace Formula

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import sys
import json
import argparse
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))
from q41_shared_utils import (
    TestConfig, TestResult, to_builtin,
    DEFAULT_CORPUS, compute_corpus_hash, load_embeddings
)

# Import individual tests
from q41_tier2_1_l_functions import run_test as run_tier2_1
from q41_tier2_2_ramanujan_bound import run_test as run_tier2_2
from q41_tier5_1_trace_formula import run_test as run_tier5_1

__version__ = "1.0.0"
__suite__ = "Q41_PHASE2_LANGLANDS"


def run_phase2_tests(
    embeddings_dict: Dict[str, Any],
    config: TestConfig,
    verbose: bool = True
) -> List[TestResult]:
    """Run all Phase 2 tests."""
    results = []

    tests = [
        ("TIER 2.1: L-Functions", run_tier2_1),
        ("TIER 2.2: Ramanujan Bound", run_tier2_2),
        ("TIER 5.1: Trace Formula", run_tier5_1),
    ]

    if verbose:
        print("\n" + "="*70)
        print("Q41 PHASE 2: LANGLANDS PROGRAM TESTS")
        print("="*70)

    for test_name, test_fn in tests:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Running: {test_name}")
            print("="*60)

        result = test_fn(embeddings_dict, config, verbose=verbose)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n  >>> {test_name}: {status}")

    return results


def generate_receipt(
    results: List[TestResult],
    config: TestConfig,
    corpus: List[str],
    embeddings_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate JSON receipt for Phase 2."""
    timestamp = datetime.now(timezone.utc)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    receipt = {
        "suite": __suite__,
        "version": __version__,
        "timestamp": timestamp.isoformat(),
        "phase": 2,
        "summary": {
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
            "tests": [
                {"name": r.name, "passed": r.passed}
                for r in results
            ]
        },
        "config": asdict(config),
        "corpus_hash": compute_corpus_hash(corpus),
        "models": list(embeddings_dict.keys()),
        "results": [
            {
                "name": r.name,
                "test_type": r.test_type,
                "passed": r.passed,
                "metrics": to_builtin(r.metrics),
                "thresholds": to_builtin(r.thresholds),
                "controls": to_builtin(r.controls),
                "notes": r.notes
            }
            for r in results
        ],
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine()
        }
    }

    return receipt


def generate_report(results: List[TestResult], config: TestConfig) -> str:
    """Generate markdown report."""
    lines = [
        "# Q41 Phase 2: Langlands Program Tests - Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Suite Version:** {__version__}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    lines.append(f"**Phase 2 Tests:** {passed}/{total} passed")
    lines.append("")
    lines.append("| Test | Result | Key Metric |")
    lines.append("|------|--------|------------|")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        # Extract a key metric
        if "mean_functional_equation_quality" in r.metrics:
            key_metric = f"FE quality: {r.metrics['mean_functional_equation_quality']:.3f}"
        elif "mean_alpha" in r.metrics:
            key_metric = f"α={r.metrics['mean_alpha']:.3f}, R²={r.metrics['mean_r_squared']:.3f}"
        elif "mean_equality_error" in r.metrics:
            key_metric = f"Equality error: {r.metrics['mean_equality_error']:.4f}"
        else:
            key_metric = "-"
        lines.append(f"| {r.name} | {status} | {key_metric} |")

    lines.extend(["", "---", ""])

    # Detailed results
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.extend([
            f"## {r.name}",
            "",
            f"**Result:** {status}",
            "",
            f"**Notes:** {r.notes}",
            "",
            "### Metrics",
            "```json",
        ])

        # Simplified metrics (exclude model_results for brevity)
        simple_metrics = {k: v for k, v in r.metrics.items() if k != "model_results"}
        lines.append(json.dumps(to_builtin(simple_metrics), indent=2))

        lines.extend([
            "```",
            "",
            "### Controls",
            "```json",
            json.dumps(to_builtin(r.controls), indent=2),
            "```",
            "",
            "---",
            "",
        ])

    # What Phase 2 tests
    lines.extend([
        "## What Phase 2 Tests",
        "",
        "| TIER | Test | Description |",
        "|------|------|-------------|",
        "| 2.1 | L-Functions | Euler product, functional equation |",
        "| 2.2 | Ramanujan Bound | Eigenvalue power-law decay |",
        "| 5.1 | Trace Formula | Spectral = Geometric equality |",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Q41 Phase 2: Langlands Program Tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print(f"Q41 PHASE 2: LANGLANDS PROGRAM TESTS v{__version__}")
        print("="*70)
        print()
        print("This suite tests Langlands-related structures:")
        print("  - TIER 2.1: Semantic L-Functions")
        print("  - TIER 2.2: Ramanujan Bound Analog")
        print("  - TIER 5.1: Arthur-Selberg Trace Formula")
        print()

    config = TestConfig(seed=args.seed)
    corpus = DEFAULT_CORPUS
    corpus_hash = compute_corpus_hash(corpus)

    if verbose:
        print(f"Corpus: {len(corpus)} words")
        print(f"Corpus hash: {corpus_hash[:16]}...")
        print()

    if verbose:
        print("Loading embeddings...")
    embeddings = load_embeddings(corpus, verbose=verbose)

    if len(embeddings) < 2:
        print("ERROR: Need at least 2 embedding models")
        sys.exit(1)

    if verbose:
        print(f"\nLoaded {len(embeddings)} models: {list(embeddings.keys())}")

    # Run tests
    results = run_phase2_tests(embeddings, config, verbose=verbose)

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("PHASE 2 FINAL SUMMARY")
        print("="*70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if verbose:
        print(f"\n  Phase 2 Tests: {passed}/{total} passed")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"    {r.name}: {status}")

    # Generate and save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Receipt
    receipt = generate_receipt(results, config, corpus, embeddings)
    receipt_path = out_dir / f"q41_phase2_receipt_{timestamp_str}.json"
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2, default=to_builtin)

    # Report
    report = generate_report(results, config)
    report_path = out_dir / f"q41_phase2_report_{timestamp_str}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    if verbose:
        print(f"\n  Receipt: {receipt_path}")
        print(f"  Report: {report_path}")

    # Overall status
    all_pass = passed == total
    if verbose:
        print(f"\n  {'ALL PHASE 2 TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
