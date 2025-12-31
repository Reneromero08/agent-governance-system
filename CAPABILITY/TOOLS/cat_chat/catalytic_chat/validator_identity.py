#!/usr/bin/env python3
"""
Validator Identity Module (Phase 6.7)

Deterministic validator identity and build fingerprinting.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class ValidatorIdentityError(RuntimeError):
    pass


def get_build_id(repo_root: Optional[Path] = None) -> str:
    """Get deterministic build ID from repository.

    Preferred method: git:<short-sha> from repo root.
    Fallback: file:<sha256-prefix> from fixed file set.

    Args:
        repo_root: Repository root path (defaults to cwd if None)

    Returns:
        Build ID string in format "git:<sha>" or "file:<sha256>"

    Raises:
        ValidatorIdentityError: If unable to determine build ID
    """
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            short_sha = result.stdout.strip()
            if short_sha and len(short_sha) >= 7:
                return f"git:{short_sha}"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    fixed_files = [
        "THOUGHT/LAB/CAT_CHAT/catalytic_chat/executor.py",
        "THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py",
        "THOUGHT/LAB/CAT_CHAT/catalytic_chat/receipt.py",
        "THOUGHT/LAB/CAT_CHAT/catalytic_chat/merkle_attestation.py",
        "THOUGHT/LAB/CAT_CHAT/catalytic_chat/validator_identity.py",
    ]

    hash_strings = []
    for file_path in sorted(fixed_files):
        file_full_path = repo_root / file_path
        if file_full_path.exists():
            file_bytes = file_full_path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            hash_strings.append(f"{file_path}:{file_hash}")

    combined = "\n".join(hash_strings).encode('utf-8')
    combined_hash = hashlib.sha256(combined).hexdigest()
    return f"file:{combined_hash[:16]}"


def get_validator_identity(signing_key: bytes, repo_root: Optional[Path] = None, validator_id: Optional[str] = None) -> Dict[str, str]:
    """Get complete validator identity dict.

    Args:
        signing_key: Ed25519 signing key bytes (32 or 64 bytes)
        repo_root: Repository root path (defaults to cwd if None)
        validator_id: Optional validator ID (defaults to build_id if None)

    Returns:
        Validator identity dict with validator_id, scheme, public_key, build_id

    Raises:
        ValidatorIdentityError: If signing key is invalid or build ID cannot be determined
    """
    try:
        from nacl.signing import SigningKey
    except ImportError:
        raise ValidatorIdentityError("PyNaCl library required. Install: pip install pynacl")

    if not isinstance(signing_key, bytes):
        raise ValidatorIdentityError("signing_key must be bytes")

    if len(signing_key) not in (32, 64):
        raise ValidatorIdentityError("signing_key must be 32 or 64 bytes")

    sk = SigningKey(signing_key)
    vk = sk.verify_key
    public_key_hex = vk.encode().hex()

    build_id = get_build_id(repo_root)

    if validator_id is None:
        validator_id = build_id

    return {
        "validator_id": validator_id,
        "scheme": "ed25519",
        "public_key": public_key_hex,
        "build_id": build_id
    }


def parse_build_id(build_id: str) -> Dict[str, str]:
    """Parse build ID into type and value.

    Args:
        build_id: Build ID string (e.g., "git:abc1234" or "file:sha256prefix")

    Returns:
        Dict with "type" ("git" or "file") and "value"

    Raises:
        ValidatorIdentityError: If build_id format is invalid
    """
    if not isinstance(build_id, str):
        raise ValidatorIdentityError("build_id must be a string")

    parts = build_id.split(":", 1)
    if len(parts) != 2:
        raise ValidatorIdentityError(f"invalid build_id format: {build_id}")

    build_type, value = parts

    if build_type == "git":
        if not value or len(value) < 7:
            raise ValidatorIdentityError(f"git build_id must be at least 7 chars: {build_id}")
    elif build_type == "file":
        if not value or len(value) < 16:
            raise ValidatorIdentityError(f"file build_id must be at least 16 chars: {build_id}")
    else:
        raise ValidatorIdentityError(f"unknown build_id type: {build_type}")

    return {"type": build_type, "value": value}
