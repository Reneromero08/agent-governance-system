#!/usr/bin/env python3
"""
CLI Output Helper Module (Phase 6.14)

Provides standardized exit codes, JSON output formatting, and quiet mode handling
for CI-friendly command execution.
"""

import sys
from typing import Dict, Any, List, Optional


# Standardized exit codes
# Documented in code comments as required by Phase 6.14
EXIT_OK = 0                      # OK
EXIT_VERIFICATION_FAILED = 1       # Verification failed (policy/attestation/hash/order/bounds)
EXIT_INVALID_INPUT = 2             # Invalid input (missing file, bad JSON, schema invalid)
EXIT_INTERNAL_ERROR = 3            # Internal error (unexpected exception)


def format_json_report(
    ok: bool,
    command: str,
    errors: Optional[List[Dict[str, str]]] = None,
    bundle_id: Optional[str] = None,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    merkle_root: Optional[str] = None,
    counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Format machine-readable JSON report for verifier commands.

    Args:
        ok: True if verification passed, False otherwise
        command: Command name (e.g., "bundle_verify", "bundle_run", "trust_verify")
        errors: Optional list of error dicts with "code" and "message" keys
        bundle_id: Optional bundle identifier
        run_id: Optional run identifier
        job_id: Optional job identifier
        merkle_root: Optional Merkle root hex string
        counts: Optional dict with counts (e.g., {"steps": N, "artifacts": M, "receipts": K})

    Returns:
        JSON-serializable dictionary with all fields sorted deterministically
    """
    report = {
        "ok": ok,
        "command": command
    }

    if errors:
        report["errors"] = errors

    if bundle_id is not None:
        report["bundle_id"] = bundle_id

    if run_id is not None:
        report["run_id"] = run_id

    if job_id is not None:
        report["job_id"] = job_id

    if merkle_root is not None:
        report["merkle_root"] = merkle_root

    if counts is not None:
        report["counts"] = counts

    return report


def write_json_report(report: Dict[str, Any], quiet: bool = False) -> None:
    """Write JSON report to stdout (only JSON + trailing newline).

    When quiet is False, human logs go to stderr.
    When quiet is True, only errors go to stderr.

    Args:
        report: JSON-serializable report dictionary
        quiet: If True, suppress non-error stderr output
    """
    from catalytic_chat.receipt import canonical_json_bytes

    report_bytes = canonical_json_bytes(report)
    sys.stdout.buffer.write(report_bytes)
    sys.stdout.buffer.flush()


def write_info(message: str, quiet: bool = False) -> None:
    """Write informational message to stderr (unless quiet mode).

    Args:
        message: Message to write
        quiet: If True, suppress output
    """
    if not quiet:
        sys.stderr.write(message)
        sys.stderr.write("\n")


def write_error(message: str) -> None:
    """Write error message to stderr (always displayed, even in quiet mode).

    Args:
        message: Error message to write
    """
    sys.stderr.write(message)
    sys.stderr.write("\n")


def classify_exit_code(
    is_verification_failure: bool = False,
    is_invalid_input: bool = False,
    is_internal_error: bool = False
) -> int:
    """Classify and return appropriate exit code.

    Priority order (first matching condition wins):
    1. Internal error → EXIT_INTERNAL_ERROR (3)
    2. Invalid input → EXIT_INVALID_INPUT (2)
    3. Verification failure → EXIT_VERIFICATION_FAILED (1)
    4. Success → EXIT_OK (0)

    Args:
        is_verification_failure: True if verification failed
        is_invalid_input: True if input is invalid (missing file, bad JSON, schema error)
        is_internal_error: True if unexpected exception occurred

    Returns:
        Standardized exit code (0-3)
    """
    if is_internal_error:
        return EXIT_INTERNAL_ERROR

    if is_invalid_input:
        return EXIT_INVALID_INPUT

    if is_verification_failure:
        return EXIT_VERIFICATION_FAILED

    return EXIT_OK
