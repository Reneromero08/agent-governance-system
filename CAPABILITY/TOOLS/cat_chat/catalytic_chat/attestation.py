#!/usr/bin/env python3
"""
Attestation Module (Phase 6.2)

Deterministic signing and verification of receipt bytes.
"""

from typing import Dict, Any, Optional


class AttestationError(RuntimeError):
    pass


def _hex_to_bytes(s: str) -> bytes:
    try:
        return bytes.fromhex(s)
    except Exception as e:
        raise AttestationError(f"invalid hex: {e}")


def _bytes_to_hex(b: bytes) -> str:
    return b.hex()


def sign_receipt_bytes(
    receipt_bytes: bytes,
    private_key: bytes,
    validator_id: Optional[str] = None,
    build_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Deterministic signature. Input bytes MUST be canonical receipt bytes with attestation stub.
    private_key may be:
      - 32-byte ed25519 seed, or
      - 64-byte SigningKey bytes (seed+pub). PyNaCl accepts both via SigningKey(...)
    validator_id and build_id are optional validator identity fields.
    Output is canonicalized to hex.
    """
    try:
        from nacl.signing import SigningKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise AttestationError("PyNaCl library required for signing. Install: pip install pynacl")

    sk = SigningKey(private_key)
    sig = sk.sign(receipt_bytes).signature  # 64 bytes
    vk = sk.verify_key.encode()             # 32 bytes

    attestation = {
        "scheme": "ed25519",
        "public_key": _bytes_to_hex(vk),
        "signature": _bytes_to_hex(sig),
    }

    if validator_id is not None:
        attestation["validator_id"] = validator_id

    if build_id is not None:
        attestation["build_id"] = build_id

    return attestation


def sign_receipt(
    receipt: Dict[str, Any],
    private_key: bytes,
    validator_id: Optional[str] = None,
    build_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sign a receipt with validator identity fields.

    This is the preferred method for signing receipts as it ensures
    validator_id and build_id are included in the signed bytes.

    Args:
        receipt: Receipt dictionary to sign
        private_key: Ed25519 signing key bytes (32 or 64 bytes)
        validator_id: Optional validator identifier
        build_id: Optional deterministic build fingerprint

    Returns:
        Receipt dictionary with attestation added

    Raises:
        AttestationError: If signing fails
    """
    try:
        from nacl.signing import SigningKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise AttestationError("PyNaCl library required for signing. Install: pip install pynacl")

    from catalytic_chat.receipt import receipt_signed_bytes

    sk = SigningKey(private_key)
    vk = sk.verify_key.encode()

    attestation_stub = {
        "scheme": "ed25519",
        "public_key": _bytes_to_hex(vk),
        "signature": ""
    }

    if validator_id is not None:
        attestation_stub["validator_id"] = validator_id

    if build_id is not None:
        attestation_stub["build_id"] = build_id

    receipt_copy = dict(receipt)
    receipt_copy["attestation"] = attestation_stub

    signed_bytes = receipt_signed_bytes(receipt_copy)

    sig = sk.sign(signed_bytes).signature

    attestation = {
        "scheme": "ed25519",
        "public_key": _bytes_to_hex(vk),
        "signature": _bytes_to_hex(sig),
    }

    if validator_id is not None:
        attestation["validator_id"] = validator_id

    if build_id is not None:
        attestation["build_id"] = build_id

    receipt_copy["attestation"] = attestation
    return receipt_copy


def verify_receipt_bytes(receipt_bytes: bytes, attestation: Dict[str, str]) -> None:
    if attestation is None:
        return
    if not isinstance(attestation, dict):
        raise AttestationError("attestation must be an object")

    import json
    from catalytic_chat.receipt import receipt_canonical_bytes

    receipt_json = json.loads(receipt_bytes.decode('utf-8'))

    signing_stub = {
        "scheme": attestation.get("scheme"),
        "public_key": attestation.get("public_key", "").lower() if isinstance(attestation.get("public_key"), str) else attestation.get("public_key"),
        "signature": ""
    }

    if attestation.get("validator_id") is not None:
        signing_stub["validator_id"] = attestation["validator_id"]

    if attestation.get("build_id") is not None:
        signing_stub["build_id"] = attestation["build_id"]

    canonical_bytes = receipt_canonical_bytes(receipt_json, attestation_override=signing_stub)

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise AttestationError("PyNaCl library required for verification. Install: pip install pynacl")

    scheme = attestation.get("scheme")
    if scheme != "ed25519":
        raise AttestationError(f"unsupported scheme: {scheme!r}")

    pub_hex = attestation.get("public_key")
    sig_hex = attestation.get("signature")
    if not isinstance(pub_hex, str) or not isinstance(sig_hex, str):
        raise AttestationError("public_key and signature must be strings")

    vk_bytes = _hex_to_bytes(pub_hex)
    sig_bytes = _hex_to_bytes(sig_hex)

    if len(vk_bytes) != 32:
        raise AttestationError("invalid public_key length")
    if len(sig_bytes) != 64:
        raise AttestationError("invalid signature length")

    vk = VerifyKey(vk_bytes)
    try:
        vk.verify(canonical_bytes, sig_bytes)
    except BadSignatureError:
        raise AttestationError("bad signature")


def verify_receipt_attestation(
    receipt: Dict[str, Any],
    trust_index: Optional[Dict[str, Any]],
    strict: bool,
    strict_identity: bool = False
) -> None:
    """Verify receipt attestation with optional strict trust checking.

    Args:
        receipt: Receipt dictionary with attestation field
        trust_index: Trust index from trust_policy.build_trust_index, or None
        strict: If True, enforce trust policy checking
        strict_identity: If True, enforce build_id pinning in strict mode

    Raises:
        AttestationError: If verification or trust check fails
    """
    attestation = receipt.get("attestation")

    if attestation is None:
        return

    if not isinstance(attestation, dict):
        raise AttestationError("attestation must be an object")

    from catalytic_chat.receipt import receipt_signed_bytes

    canonical_bytes = receipt_signed_bytes(receipt)

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        raise AttestationError("PyNaCl library required for verification. Install: pip install pynacl")

    scheme = attestation.get("scheme")
    if scheme != "ed25519":
        raise AttestationError(f"unsupported scheme: {scheme!r}")

    pub_hex = attestation.get("public_key")
    sig_hex = attestation.get("signature")
    if not isinstance(pub_hex, str) or not isinstance(sig_hex, str):
        raise AttestationError("public_key and signature must be strings")

    vk_bytes = _hex_to_bytes(pub_hex)
    sig_bytes = _hex_to_bytes(sig_hex)

    if len(vk_bytes) != 32:
        raise AttestationError("invalid public_key length")
    if len(sig_bytes) != 64:
        raise AttestationError("invalid signature length")

    vk = VerifyKey(vk_bytes)
    try:
        vk.verify(canonical_bytes, sig_bytes)
    except BadSignatureError:
        raise AttestationError("bad signature")

    if strict:
        if trust_index is None:
            raise AttestationError("UNTRUSTED_VALIDATOR_KEY")

        from catalytic_chat.trust_policy import get_validator_by_id, get_validator_by_public_key, is_key_allowed

        validator_id = attestation.get("validator_id")

        if validator_id is not None:
            validator_entry = get_validator_by_id(trust_index, validator_id)
            if validator_entry is None:
                raise AttestationError(f"VALIDATOR_ID_NOT_FOUND: {validator_id}")

            validator_pub_key = validator_entry.get("public_key")
            if validator_pub_key is None:
                raise AttestationError("VALIDATOR_ENTRY_MISSING_PUBLIC_KEY")

            validator_pub_key_lower = validator_pub_key.lower()
            pub_key_lower = pub_hex.lower()

            if validator_pub_key_lower != pub_key_lower:
                raise AttestationError(f"PUBLIC_KEY_MISMATCH: validator_id={validator_id} expects {validator_pub_key_lower}, got {pub_key_lower}")

            if not is_key_allowed(trust_index, pub_hex, "RECEIPT", "ed25519"):
                raise AttestationError("UNTRUSTED_VALIDATOR_KEY")

            if strict_identity:
                pinned_build_id = validator_entry.get("build_id")
                if pinned_build_id:
                    attestation_build_id = attestation.get("build_id")
                    if attestation_build_id != pinned_build_id:
                        raise AttestationError(f"BUILD_ID_MISMATCH: attestation build_id={attestation_build_id}, pinned={pinned_build_id}")
        else:
            if not is_key_allowed(trust_index, pub_hex, "RECEIPT", "ed25519"):
                raise AttestationError("UNTRUSTED_VALIDATOR_KEY")

            if strict_identity:
                entry = get_validator_by_public_key(trust_index, pub_hex)
                if entry:
                    pinned_build_id = entry.get("build_id")
                    if pinned_build_id:
                        attestation_build_id = attestation.get("build_id")
                        if attestation_build_id != pinned_build_id:
                            raise AttestationError(f"BUILD_ID_MISMATCH: attestation build_id={attestation_build_id}, pinned={pinned_build_id}")