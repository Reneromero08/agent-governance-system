#!/usr/bin/env python3
"""
Merkle Attestation Module (Phase 6.5)

Deterministic Ed25519 signing and verification of receipt chain Merkle root.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class MerkleAttestationError(RuntimeError):
    pass


def _hex_to_bytes(s: str) -> bytes:
    try:
        return bytes.fromhex(s)
    except Exception as e:
        raise MerkleAttestationError(f"invalid hex: {e}")


def _bytes_to_hex(b: bytes) -> str:
    return b.hex()


def validate_merkle_root_hex(merkle_root_hex: str) -> None:
    """Validate merkle root hex string.

    Args:
        merkle_root_hex: Merkle root hex string

    Raises:
        MerkleAttestationError: If hex is invalid or wrong length
    """
    if not isinstance(merkle_root_hex, str):
        raise MerkleAttestationError("merkle_root must be a string")
    if len(merkle_root_hex) != 64:
        raise MerkleAttestationError("merkle_root must be 64 hex characters (32 bytes)")
    try:
        bytes.fromhex(merkle_root_hex)
    except ValueError as e:
        raise MerkleAttestationError(f"merkle_root invalid hex: {e}")


def sign_merkle_root(
    merkle_root_hex: str,
    signing_key_hex: str,
    validator_id: Optional[str] = None,
    build_id: Optional[str] = None
) -> Dict[str, str]:
    """Sign Merkle root with Ed25519 private key.

    Args:
        merkle_root_hex: Merkle root hex string (64 hex chars)
        signing_key_hex: Ed25519 signing key hex (64 hex chars for seed, or 128 hex chars for seed+pub)
        validator_id: Optional validator identifier
        build_id: Optional deterministic build fingerprint

    Returns:
        Attestation dict with scheme, merkle_root, public_key, signature, and optional validator identity

    Raises:
        MerkleAttestationError: If validation or signing fails
    """
    validate_merkle_root_hex(merkle_root_hex)

    if not isinstance(signing_key_hex, str):
        raise MerkleAttestationError("signing_key must be a string")
    if len(signing_key_hex) not in (64, 128):
        raise MerkleAttestationError("signing_key must be 64 or 128 hex characters (32 or 64 bytes)")

    try:
        from nacl.signing import SigningKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise MerkleAttestationError("PyNaCl library required for signing. Install: pip install pynacl")

    signing_key_bytes = _hex_to_bytes(signing_key_hex)
    merkle_root_bytes = _hex_to_bytes(merkle_root_hex)

    sk = SigningKey(signing_key_bytes)
    vk = sk.verify_key.encode()
    public_key_bytes = vk

    vid_bytes = validator_id.encode('utf-8') if validator_id is not None else b""
    bid_bytes = build_id.encode('utf-8') if build_id is not None else b""

    msg = b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes + b"|VID:" + vid_bytes + b"|BID:" + bid_bytes + b"|PK:" + public_key_bytes

    sig = sk.sign(msg).signature

    attestation = {
        "scheme": "ed25519",
        "merkle_root": merkle_root_hex,
        "public_key": _bytes_to_hex(vk),
        "signature": _bytes_to_hex(sig),
    }

    if validator_id is not None:
        attestation["validator_id"] = validator_id

    if build_id is not None:
        attestation["build_id"] = build_id

    return attestation


def verify_merkle_attestation(att: Dict[str, Any]) -> None:
    """Verify Merkle attestation signature.

    Args:
        att: Attestation dict with scheme, merkle_root, public_key, signature

    Raises:
        MerkleAttestationError: If validation or verification fails
    """
    if not isinstance(att, dict):
        raise MerkleAttestationError("attestation must be an object")

    scheme = att.get("scheme")
    if scheme != "ed25519":
        raise MerkleAttestationError(f"unsupported scheme: {scheme!r}")

    merkle_root_hex = att.get("merkle_root")
    if not isinstance(merkle_root_hex, str):
        raise MerkleAttestationError("merkle_root must be a string")

    validate_merkle_root_hex(merkle_root_hex)

    pub_hex = att.get("public_key")
    sig_hex = att.get("signature")

    if not isinstance(pub_hex, str) or not isinstance(sig_hex, str):
        raise MerkleAttestationError("public_key and signature must be strings")

    vk_bytes = _hex_to_bytes(pub_hex)
    sig_bytes = _hex_to_bytes(sig_hex)

    if len(vk_bytes) != 32:
        raise MerkleAttestationError("invalid public_key length (must be 32 bytes)")
    if len(sig_bytes) != 64:
        raise MerkleAttestationError("invalid signature length (must be 64 bytes)")

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise MerkleAttestationError("PyNaCl library required for verification. Install: pip install pynacl")

    vk = VerifyKey(vk_bytes)

    validator_id = att.get("validator_id")
    build_id = att.get("build_id")

    vid_bytes = validator_id.encode('utf-8') if validator_id is not None else b""
    bid_bytes = build_id.encode('utf-8') if build_id is not None else b""

    merkle_root_bytes = _hex_to_bytes(merkle_root_hex)
    msg = b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes + b"|VID:" + vid_bytes + b"|BID:" + bid_bytes + b"|PK:" + vk_bytes

    try:
        vk.verify(msg, sig_bytes)
    except BadSignatureError:
        raise MerkleAttestationError("bad signature")


def verify_merkle_attestation_with_trust(
    att: Dict[str, Any],
    merkle_root_hex: str,
    trust_index: Optional[Dict[str, Any]],
    strict: bool,
    strict_identity: bool = False
) -> None:
    """Verify Merkle attestation signature with optional strict trust checking.

    Args:
        att: Attestation dict with scheme, merkle_root, public_key, signature
        merkle_root_hex: Expected Merkle root hex string
        trust_index: Trust index from trust_policy.build_trust_index, or None
        strict: If True, enforce trust policy checking
        strict_identity: If True, enforce build_id pinning in strict mode

    Raises:
        MerkleAttestationError: If validation or verification fails
    """
    if not isinstance(att, dict):
        raise MerkleAttestationError("attestation must be an object")

    scheme = att.get("scheme")
    if scheme != "ed25519":
        raise MerkleAttestationError(f"unsupported scheme: {scheme!r}")

    att_merkle_root = att.get("merkle_root")
    if not isinstance(att_merkle_root, str):
        raise MerkleAttestationError("merkle_root must be a string")

    if att_merkle_root != merkle_root_hex:
        raise MerkleAttestationError("merkle_root mismatch")

    validate_merkle_root_hex(merkle_root_hex)

    pub_hex = att.get("public_key")
    sig_hex = att.get("signature")

    if not isinstance(pub_hex, str) or not isinstance(sig_hex, str):
        raise MerkleAttestationError("public_key and signature must be strings")

    vk_bytes = _hex_to_bytes(pub_hex)
    sig_bytes = _hex_to_bytes(sig_hex)

    if len(vk_bytes) != 32:
        raise MerkleAttestationError("invalid public_key length (must be 32 bytes)")
    if len(sig_bytes) != 64:
        raise MerkleAttestationError("invalid signature length (must be 64 bytes)")

    validator_id = att.get("validator_id")
    build_id = att.get("build_id")

    vid_bytes = validator_id.encode('utf-8') if validator_id is not None else b""
    bid_bytes = build_id.encode('utf-8') if build_id is not None else b""

    merkle_root_bytes = _hex_to_bytes(merkle_root_hex)
    msg = b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes + b"|VID:" + vid_bytes + b"|BID:" + bid_bytes + b"|PK:" + vk_bytes

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise MerkleAttestationError("PyNaCl library required for verification. Install: pip install pynacl")

    vk = VerifyKey(vk_bytes)

    try:
        vk.verify(msg, sig_bytes)
    except BadSignatureError:
        raise MerkleAttestationError("bad signature")

    if strict:
        if trust_index is None:
            raise MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")

        from catalytic_chat.trust_policy import get_validator_by_id, get_validator_by_public_key, is_key_allowed

        if validator_id is not None:
            validator_entry = get_validator_by_id(trust_index, validator_id)
            if validator_entry is None:
                raise MerkleAttestationError(f"VALIDATOR_ID_NOT_FOUND: {validator_id}")

            validator_pub_key = validator_entry.get("public_key")
            if validator_pub_key is None:
                raise MerkleAttestationError("VALIDATOR_ENTRY_MISSING_PUBLIC_KEY")

            validator_pub_key_lower = validator_pub_key.lower()
            pub_key_lower = pub_hex.lower()

            if validator_pub_key_lower != pub_key_lower:
                raise MerkleAttestationError(f"PUBLIC_KEY_MISMATCH: validator_id={validator_id} expects {validator_pub_key_lower}, got {pub_key_lower}")

            if not is_key_allowed(trust_index, pub_hex, "MERKLE", "ed25519"):
                raise MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")

            if strict_identity:
                pinned_build_id = validator_entry.get("build_id")
                if pinned_build_id:
                    attestation_build_id = att.get("build_id")
                    if attestation_build_id != pinned_build_id:
                        raise MerkleAttestationError(f"BUILD_ID_MISMATCH: attestation build_id={attestation_build_id}, pinned={pinned_build_id}")
        else:
            if not is_key_allowed(trust_index, pub_hex, "MERKLE", "ed25519"):
                raise MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")

            if strict_identity:
                from catalytic_chat.trust_policy import get_validator_by_public_key

                entry = get_validator_by_public_key(trust_index, pub_hex)
                if entry:
                    pinned_build_id = entry.get("build_id")
                    if pinned_build_id:
                        attestation_build_id = att.get("build_id")
                        if attestation_build_id != pinned_build_id:
                            raise MerkleAttestationError(f"BUILD_ID_MISMATCH: attestation build_id={attestation_build_id}, pinned={pinned_build_id}")


def validate_merkle_attestation_schema(att: Dict[str, Any]) -> None:
    """Validate attestation against JSON schema.

    Args:
        att: Attestation dict to validate

    Raises:
        MerkleAttestationError: If validation fails
    """
    try:
        import jsonschema
    except ImportError:
        raise MerkleAttestationError("jsonschema package required for schema validation")

    schema_path = Path(__file__).parent.parent / "SCHEMAS" / "merkle_attestation.schema.json"
    if not schema_path.exists():
        raise MerkleAttestationError(f"Merkle attestation schema not found: {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=att, schema=schema)
    except jsonschema.ValidationError as e:
        raise MerkleAttestationError(f"schema validation failed: {e.message}")


def write_merkle_attestation(out_path: Path, att: Dict[str, Any]) -> None:
    """Write merkle attestation to file as canonical JSON with trailing newline.

    Args:
        out_path: Path to write attestation
        att: Attestation dictionary
    """
    from catalytic_chat.receipt import canonical_json_bytes
    att_bytes = canonical_json_bytes(att)
    out_path.write_bytes(att_bytes)


def load_merkle_attestation(att_path: Path) -> Optional[Dict[str, Any]]:
    """Load merkle attestation from file.

    Args:
        att_path: Path to attestation file

    Returns:
        Attestation dictionary or None if file doesn't exist
    """
    if not att_path.exists():
        return None

    att_bytes = att_path.read_bytes()
    att_text = att_bytes.decode('utf-8').rstrip('\n')
    return json.loads(att_text)
