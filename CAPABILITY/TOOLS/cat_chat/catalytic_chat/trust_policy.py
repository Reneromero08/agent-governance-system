#!/usr/bin/env python3
"""
Trust Policy Module (Phase 6.6)

Deterministic policy for pinning validator public keys allowed to attest.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class TrustPolicyError(RuntimeError):
    pass


def load_trust_policy_bytes(path: Path) -> bytes:
    """Read trust policy file as exact bytes.

    Args:
        path: Path to trust policy JSON file

    Returns:
        Exact file bytes

    Raises:
        TrustPolicyError: If file not found or read fails
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise TrustPolicyError(f"trust policy file not found: {path}")

    try:
        return path.read_bytes()
    except Exception as e:
        raise TrustPolicyError(f"failed to read trust policy: {e}")


def parse_trust_policy(policy_bytes: bytes) -> Dict[str, Any]:
    """Parse and validate trust policy from bytes.

    Args:
        policy_bytes: Trust policy file bytes

    Returns:
        Parsed and validated policy dictionary

    Raises:
        TrustPolicyError: If JSON parsing or schema validation fails
    """
    try:
        import jsonschema
    except ImportError:
        raise TrustPolicyError("jsonschema package required for trust policy validation")

    policy_text = policy_bytes.decode('utf-8')
    try:
        policy = json.loads(policy_text)
    except json.JSONDecodeError as e:
        raise TrustPolicyError(f"invalid JSON in trust policy: {e}")

    schema_path = Path(__file__).parent.parent / "SCHEMAS" / "trust_policy.schema.json"
    if not schema_path.exists():
        raise TrustPolicyError(f"trust policy schema not found: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=policy, schema=schema)
    except jsonschema.ValidationError as e:
        raise TrustPolicyError(f"trust policy schema validation failed: {e.message}")

    return policy


def build_trust_index(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Build deterministic trust index from policy.

    Returns index with two maps:
    - public_key_hex (lowercase) -> validator entry
    - validator_id -> validator entry

    Args:
        policy: Validated trust policy dictionary

    Returns:
        Trust index dictionary with 'by_public_key' and 'by_validator_id' maps

    Raises:
        TrustPolicyError: If duplicate validator_id or public_key found
    """
    allow = policy.get("allow", [])
    validator_ids = set()
    public_keys = set()

    by_public_key = {}
    by_validator_id = {}

    for entry in allow:
        validator_id = entry.get("validator_id")
        public_key = entry.get("public_key")

        if not isinstance(validator_id, str):
            raise TrustPolicyError(f"invalid validator_id in allow list")

        if not isinstance(public_key, str):
            raise TrustPolicyError(f"invalid public_key in allow list")

        if validator_id in validator_ids:
            raise TrustPolicyError(f"duplicate validator_id: {validator_id}")

        public_key_lower = public_key.lower()
        if public_key_lower in public_keys:
            raise TrustPolicyError(f"duplicate public_key: {public_key}")

        validator_ids.add(validator_id)
        public_keys.add(public_key_lower)

        by_public_key[public_key_lower] = entry
        by_validator_id[validator_id] = entry

    return {
        "by_public_key": by_public_key,
        "by_validator_id": by_validator_id
    }


def get_validator_by_id(index: Dict[str, Any], validator_id: str) -> Optional[Dict[str, Any]]:
    """Get validator entry by validator_id from trust index.

    Args:
        index: Trust index from build_trust_index
        validator_id: Validator identifier string

    Returns:
        Validator entry dict or None if not found
    """
    return index.get("by_validator_id", {}).get(validator_id)


def get_validator_by_public_key(index: Dict[str, Any], public_key_hex: str) -> Optional[Dict[str, Any]]:
    """Get validator entry by public_key from trust index.

    Args:
        index: Trust index from build_trust_index
        public_key_hex: Public key hex string

    Returns:
        Validator entry dict or None if not found
    """
    if not isinstance(public_key_hex, str):
        return None
    return index.get("by_public_key", {}).get(public_key_hex.lower())


def is_key_allowed(index: Dict[str, Any], public_key_hex: str, scope: str, scheme: str = "ed25519") -> bool:
    """Check if public key is allowed for given scope and scheme.

    Args:
        index: Trust index from build_trust_index
        public_key_hex: Public key hex string
        scope: Either "RECEIPT" or "MERKLE"
        scheme: Signature scheme (default "ed25519")

    Returns:
        True if key is allowed, False otherwise
    """
    if not isinstance(public_key_hex, str):
        return False

    public_key_lower = public_key_hex.lower()
    entry = index.get("by_public_key", {}).get(public_key_lower)

    if not entry:
        return False

    if not entry.get("enabled", False):
        return False

    schemes = entry.get("schemes", [])
    if scheme not in schemes:
        return False

    scopes = entry.get("scope", [])
    if scope not in scopes:
        return False

    return True
