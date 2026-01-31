"""
Catalytic Error Codes (SPECTRUM-05 Section 8.2)

Structured error codes for catalytic operations with consistent format
and machine-readable details.
"""

from typing import Dict, Optional, Any


class CatalyticError(Exception):
    """Base exception for catalytic operations with structured error codes."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Export error as dictionary for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# Error codes per SPECTRUM-05:

# CAT-001: Required artifact missing from run ledger
CAT_001_ARTIFACT_MISSING = "CAT-001"

# CAT-002: Schema validation failed (PROOF.json, JOBSPEC.json, etc.)
CAT_002_SCHEMA_INVALID = "CAT-002"

# CAT-003: Restoration failed - catalytic domain not restored to pre-state
CAT_003_RESTORATION_FAILED = "CAT-003"

# CAT-004: PROOF.json verification failed or mismatched
CAT_004_PROOF_MISMATCH = "CAT-004"

# CAT-005: Domain constraint violation (symlinks, traversal, forbidden paths)
CAT_005_DOMAIN_VIOLATION = "CAT-005"

# CAT-006: CAS (Content-Addressable Store) corruption detected
CAT_006_CAS_CORRUPTION = "CAT-006"

# CAT-007: Forbidden artifact present in run ledger
CAT_007_FORBIDDEN_ARTIFACT = "CAT-007"

# CAT-008: Path normalization or validation failed
CAT_008_PATH_INVALID = "CAT-008"

# CAT-009: Preflight validation rejected the JobSpec
CAT_009_PREFLIGHT_REJECTED = "CAT-009"

# CAT-010: Memoization cache corruption or mismatch
CAT_010_MEMO_CORRUPTION = "CAT-010"


# Convenience factory functions

def artifact_missing(artifact_name: str, ledger_dir: str) -> CatalyticError:
    """Create error for missing required artifact."""
    return CatalyticError(
        CAT_001_ARTIFACT_MISSING,
        f"Required artifact missing: {artifact_name}",
        {"artifact": artifact_name, "ledger_dir": ledger_dir},
    )


def schema_invalid(artifact_name: str, validation_errors: list) -> CatalyticError:
    """Create error for schema validation failure."""
    return CatalyticError(
        CAT_002_SCHEMA_INVALID,
        f"Schema validation failed for {artifact_name}",
        {"artifact": artifact_name, "errors": validation_errors},
    )


def restoration_failed(domain: str, diff: dict) -> CatalyticError:
    """Create error for restoration failure."""
    return CatalyticError(
        CAT_003_RESTORATION_FAILED,
        f"Catalytic domain not restored: {domain}",
        {"domain": domain, "diff": diff},
    )


def proof_mismatch(expected: str, actual: str) -> CatalyticError:
    """Create error for PROOF.json mismatch."""
    return CatalyticError(
        CAT_004_PROOF_MISMATCH,
        "PROOF.json verification failed",
        {"expected": expected, "actual": actual},
    )


def domain_violation(path: str, reason: str) -> CatalyticError:
    """Create error for domain constraint violation."""
    return CatalyticError(
        CAT_005_DOMAIN_VIOLATION,
        f"Domain constraint violated: {reason}",
        {"path": path, "reason": reason},
    )


def cas_corruption(blob_hash: str, expected_hash: str) -> CatalyticError:
    """Create error for CAS corruption."""
    return CatalyticError(
        CAT_006_CAS_CORRUPTION,
        f"CAS blob corruption detected: {blob_hash}",
        {"blob_hash": blob_hash, "expected_hash": expected_hash},
    )


def forbidden_artifact(artifact_name: str, ledger_dir: str) -> CatalyticError:
    """Create error for forbidden artifact presence."""
    return CatalyticError(
        CAT_007_FORBIDDEN_ARTIFACT,
        f"Forbidden artifact present: {artifact_name}",
        {"artifact": artifact_name, "ledger_dir": ledger_dir},
    )


def path_invalid(path: str, reason: str) -> CatalyticError:
    """Create error for path validation failure."""
    return CatalyticError(
        CAT_008_PATH_INVALID,
        f"Path validation failed: {reason}",
        {"path": path, "reason": reason},
    )


def preflight_rejected(errors: list) -> CatalyticError:
    """Create error for preflight rejection."""
    return CatalyticError(
        CAT_009_PREFLIGHT_REJECTED,
        "Preflight validation rejected JobSpec",
        {"errors": errors},
    )


def memo_corruption(cache_key: str, reason: str) -> CatalyticError:
    """Create error for memoization cache corruption."""
    return CatalyticError(
        CAT_010_MEMO_CORRUPTION,
        f"Memoization cache corruption: {reason}",
        {"cache_key": cache_key, "reason": reason},
    )
