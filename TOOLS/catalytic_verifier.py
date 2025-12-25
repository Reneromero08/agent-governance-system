#!/usr/bin/env python3
"""
Catalytic Bundle/Chain Verifier (Phase 1 Option #1)

Deterministic CLI for verifying SPECTRUM-02 bundles and SPECTRUM-03 chains.

Usage:
  # Verify a single run bundle
  python catalytic_verifier.py --run-dir CONTRACTS/_runs/<run_id> [--strict]

  # Verify a chain of runs
  python catalytic_verifier.py --chain CONTRACTS/_runs/<run1> CONTRACTS/_runs/<run2> ... [--strict]

  # Verify all runs in a directory (as a chain)
  python catalytic_verifier.py --chain-dir CONTRACTS/_runs [--strict]

  # JSON output
  python catalytic_verifier.py --run-dir CONTRACTS/_runs/<run_id> --json

Options:
  --run-dir PATH        Path to single run directory to verify
  --chain PATH [PATH...]  Paths to run directories (in chain order)
  --chain-dir PATH      Directory containing run directories (sorted by name)
  --strict              Enable strict verification mode
  --no-proof            Skip PROOF.json verification (not recommended)
  --json                Output as JSON

Exit codes:
  0: Verification passed
  1: Verification failed
  2: Invalid arguments

Contract:
  - Fail closed: any ambiguity rejects
  - Verification depends ONLY on bundle artifacts + file hashes + ordering
  - No heuristics, no logs, no side channels
  - Enforces forbidden artifacts: rejects if logs/, tmp/, transcript.json exist
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.verify_bundle import BundleVerifier


class VerifierCLI:
    """CLI wrapper for BundleVerifier."""

    def __init__(self):
        self.verifier = BundleVerifier(project_root=REPO_ROOT)

    def verify_single_bundle(
        self,
        run_dir: Path,
        strict: bool = False,
        check_proof: bool = True
    ) -> tuple[bool, dict]:
        """Verify a single bundle.

        Args:
            run_dir: Path to run directory
            strict: Enable strict verification
            check_proof: Check PROOF.json

        Returns:
            (success, report)
        """
        result = self.verifier.verify_bundle(
            run_dir=run_dir,
            strict=strict,
            check_proof=check_proof
        )

        report = {
            "type": "bundle",
            "run_dir": str(run_dir),
            "valid": result["valid"],
            "errors": result["errors"],
            "error_count": len(result["errors"])
        }

        return result["valid"], report

    def verify_chain(
        self,
        run_dirs: List[Path],
        strict: bool = False,
        check_proof: bool = True
    ) -> tuple[bool, dict]:
        """Verify a chain of bundles.

        Args:
            run_dirs: List of run directories (in chain order)
            strict: Enable strict verification
            check_proof: Check PROOF.json

        Returns:
            (success, report)
        """
        result = self.verifier.verify_chain(
            run_dirs=run_dirs,
            strict=strict,
            check_proof=check_proof
        )

        report = {
            "type": "chain",
            "run_dirs": [str(d) for d in run_dirs],
            "run_count": len(run_dirs),
            "valid": result["valid"],
            "errors": result["errors"],
            "error_count": len(result["errors"])
        }

        return result["valid"], report

    def verify_chain_from_directory(
        self,
        chain_dir: Path,
        strict: bool = False,
        check_proof: bool = True
    ) -> tuple[bool, dict]:
        """Verify all runs in a directory as a chain.

        Runs are ordered by directory name (alphabetically).

        Args:
            chain_dir: Directory containing run directories
            strict: Enable strict verification
            check_proof: Check PROOF.json

        Returns:
            (success, report)
        """
        if not chain_dir.exists():
            return False, {
                "type": "chain_dir",
                "valid": False,
                "errors": [{
                    "code": "DIRECTORY_NOT_FOUND",
                    "severity": "error",
                    "message": f"Chain directory not found: {chain_dir}",
                    "path": str(chain_dir),
                    "details": {}
                }],
                "error_count": 1
            }

        # Find all subdirectories
        run_dirs = sorted([d for d in chain_dir.iterdir() if d.is_dir()])

        if not run_dirs:
            return False, {
                "type": "chain_dir",
                "chain_dir": str(chain_dir),
                "valid": False,
                "errors": [{
                    "code": "NO_RUNS_FOUND",
                    "severity": "error",
                    "message": f"No run directories found in: {chain_dir}",
                    "path": str(chain_dir),
                    "details": {}
                }],
                "error_count": 1
            }

        return self.verify_chain(run_dirs, strict=strict, check_proof=check_proof)

    def print_human_readable(self, success: bool, report: dict):
        """Print human-readable output."""
        print("=" * 60)
        if report["type"] == "bundle":
            print(f"Bundle Verification: {report['run_dir']}")
        elif report["type"] == "chain":
            print(f"Chain Verification: {report['run_count']} runs")
        elif report["type"] == "chain_dir":
            print(f"Chain Directory Verification: {report.get('chain_dir', 'unknown')}")
        print("=" * 60)

        if success:
            print("[PASS] Verification successful")
        else:
            print(f"[FAIL] Verification failed ({report['error_count']} errors)")
            print()
            for i, error in enumerate(report["errors"], 1):
                run_id = error.get("run_id", "unknown")
                code = error.get("code", "UNKNOWN")
                message = error.get("message", "No message")
                print(f"{i}. [{code}] {message}")
                if "run_id" in error:
                    print(f"   Run: {run_id}")
                if "details" in error and error["details"]:
                    print(f"   Details: {error['details']}")

    def print_json(self, report: dict):
        """Print JSON output."""
        print(json.dumps(report, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify SPECTRUM-02 bundles and SPECTRUM-03 chains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mutually exclusive group for verification mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--run-dir",
        type=Path,
        help="Path to single run directory to verify"
    )
    mode_group.add_argument(
        "--chain",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="Paths to run directories (in chain order)"
    )
    mode_group.add_argument(
        "--chain-dir",
        type=Path,
        help="Directory containing run directories (sorted by name)"
    )

    # Options
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict verification mode"
    )
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="Skip PROOF.json verification (not recommended)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = VerifierCLI()

    # Run verification based on mode
    try:
        if args.run_dir:
            success, report = cli.verify_single_bundle(
                run_dir=args.run_dir,
                strict=args.strict,
                check_proof=not args.no_proof
            )
        elif args.chain:
            success, report = cli.verify_chain(
                run_dirs=args.chain,
                strict=args.strict,
                check_proof=not args.no_proof
            )
        elif args.chain_dir:
            success, report = cli.verify_chain_from_directory(
                chain_dir=args.chain_dir,
                strict=args.strict,
                check_proof=not args.no_proof
            )
        else:
            # Should never reach here due to required=True
            parser.print_help()
            return 2

    except Exception as e:
        if args.json:
            print(json.dumps({
                "valid": False,
                "errors": [{
                    "code": "INTERNAL_ERROR",
                    "severity": "error",
                    "message": str(e),
                    "path": "/",
                    "details": {}
                }],
                "error_count": 1
            }, indent=2))
        else:
            print(f"[ERROR] Internal error: {e}", file=sys.stderr)
        return 1

    # Output results
    if args.json:
        cli.print_json(report)
    else:
        cli.print_human_readable(success, report)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
