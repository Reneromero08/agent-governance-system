#!/usr/bin/env python3
"""
Execution Policy Module (Phase 6.8)

Deterministic policy for bundle execution and verification requirements.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ExecutionPolicyError(RuntimeError):
    pass


DEFAULT_POLICY_VERSION = "1.0.0"


def load_execution_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate execution policy from file.

    Args:
        path: Path to policy JSON file, or None for default

    Returns:
        Validated and normalized policy dictionary

    Raises:
        ExecutionPolicyError: If policy is invalid or cannot be loaded
    """
    try:
        import jsonschema
    except ImportError:
        raise ExecutionPolicyError("jsonschema package required for policy validation. Install: pip install jsonschema")

    if path is None:
        raise ExecutionPolicyError("policy path is required")

    policy_path = Path(path)
    if not policy_path.exists():
        raise ExecutionPolicyError(f"policy file not found: {policy_path}")

    try:
        policy_bytes = policy_path.read_bytes()
    except Exception as e:
        raise ExecutionPolicyError(f"failed to read policy file: {e}")

    policy_text = policy_bytes.decode('utf-8')
    policy = json.loads(policy_text)

    validate_policy(policy, schema_path=None)

    return policy


def validate_policy(policy: Dict[str, Any], schema_path: Optional[Path] = None) -> None:
    """Validate policy against schema and normalize values.

    Args:
        policy: Policy dictionary to validate
        schema_path: Optional path to schema file for validation

    Raises:
        ExecutionPolicyError: If policy is invalid
    """
    try:
        import jsonschema
    except ImportError:
        raise ExecutionPolicyError("jsonschema package required for policy validation. Install: pip install jsonschema")

    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "SCHEMAS" / "execution_policy.schema.json"

    if not schema_path.exists():
        raise ExecutionPolicyError(f"policy schema not found: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=policy, schema=schema)
    except jsonschema.ValidationError as e:
        raise ExecutionPolicyError(f"policy validation failed: {e.message}")

    policy_version = policy.get("policy_version")
    if not isinstance(policy_version, str):
        raise ExecutionPolicyError("policy_version must be a string")

    _normalize_boolean(policy, "require_verify_bundle", False)
    _normalize_boolean(policy, "require_verify_chain", False)
    _normalize_boolean(policy, "require_receipt_attestation", False)
    _normalize_boolean(policy, "require_merkle_attestation", False)
    _normalize_boolean(policy, "strict_trust", False)
    _normalize_boolean(policy, "strict_identity", False)

    trust_policy_path = policy.get("trust_policy_path")
    if trust_policy_path is not None:
        if not isinstance(trust_policy_path, str):
            raise ExecutionPolicyError("trust_policy_path must be a string or null")
        if trust_policy_path == "":
            policy["trust_policy_path"] = None

    policy["policy_version"] = policy_version


def _normalize_boolean(policy: Dict[str, Any], key: str, default_value: bool) -> None:
    """Normalize boolean policy field.

    Args:
        policy: Policy dictionary
        key: Policy field name
        default_value: Default value if field is missing
    """
    value = policy.get(key)
    if value is None:
        policy[key] = default_value
    elif not isinstance(value, bool):
        raise ExecutionPolicyError(f"{key} must be a boolean, got {type(value).__name__}")


def policy_from_cli_args(args) -> Dict[str, Any]:
    """Build execution policy from CLI arguments.

    Merges existing CLI flags into a single policy dict.

    Args:
        args: Parsed command-line arguments from argparse

    Returns:
        Deterministic policy dictionary

    Raises:
        ExecutionPolicyError: If policy configuration is invalid
    """
    policy = {
        "policy_version": DEFAULT_POLICY_VERSION
    }

    require_verify_bundle = getattr(args, 'verify_bundle', False)
    require_verify_chain = getattr(args, 'verify_chain', False)
    require_receipt_attestation = getattr(args, 'require_attestation', False)
    require_merkle_attestation = getattr(args, 'require_merkle_attestation', False)
    strict_trust = getattr(args, 'strict_trust', False)
    strict_identity = getattr(args, 'strict_identity', False)
    trust_policy_path = getattr(args, 'trust_policy', None)

    policy["require_verify_bundle"] = bool(require_verify_bundle)
    policy["require_verify_chain"] = bool(require_verify_chain)
    policy["require_receipt_attestation"] = bool(require_receipt_attestation)
    policy["require_merkle_attestation"] = bool(require_merkle_attestation)
    policy["strict_trust"] = bool(strict_trust)
    policy["strict_identity"] = bool(strict_identity)

    if trust_policy_path is not None:
        if not isinstance(trust_policy_path, str):
            raise ExecutionPolicyError("trust_policy_path must be a string")
        if trust_policy_path == "":
            trust_policy_path = None
        policy["trust_policy_path"] = trust_policy_path

    validate_policy(policy)

    return policy


def policy_requires_trust(policy: Dict[str, Any]) -> bool:
    """Check if policy requires trust checking.

    Args:
        policy: Policy dictionary

    Returns:
        True if strict_trust or strict_identity or trust_policy_path is set
    """
    return (
        policy.get("strict_trust", False) or
        policy.get("strict_identity", False) or
        policy.get("trust_policy_path") is not None
    )


def get_policy_error(requirement: str) -> str:
    """Get standardized error message for policy requirement failure.

    Args:
        requirement: Name of the requirement that failed

    Returns:
        Error message string
    """
    error_messages = {
        "bundle": "bundle verification failed",
        "chain": "receipt chain verification failed",
        "receipt_attestation": "receipt attestation required but missing",
        "merkle_attestation": "merkle attestation required but missing",
        "trust": "trust policy required but not provided",
        "identity": "strict identity check failed"
    }

    return error_messages.get(requirement, "policy requirement failed")
