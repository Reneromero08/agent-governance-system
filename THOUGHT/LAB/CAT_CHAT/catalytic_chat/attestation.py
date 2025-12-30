#!/usr/bin/env python3
"""
Attestation Module (Phase 6.2)

Deterministic signing and verification of receipt bytes.
"""

from typing import Dict, Any


class AttestationError(RuntimeError):
    pass


def _hex_to_bytes(s: str) -> bytes:
    try:
        return bytes.fromhex(s)
    except Exception as e:
        raise AttestationError(f"invalid hex: {e}")


def _bytes_to_hex(b: bytes) -> str:
    return b.hex()


def sign_receipt_bytes(receipt_bytes: bytes, private_key: bytes) -> Dict[str, str]:
    """
    Deterministic signature. Input bytes MUST be canonical receipt bytes with attestation=null/None.
    private_key may be:
      - 32-byte ed25519 seed, or
      - 64-byte SigningKey bytes (seed+pub). PyNaCl accepts both via SigningKey(...)
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
    
    return {
        "scheme": "ed25519",
        "public_key": _bytes_to_hex(vk),
        "signature": _bytes_to_hex(sig),
    }


def verify_receipt_bytes(receipt_bytes: bytes, attestation: Dict[str, str]) -> None:
    if attestation is None:
        return
    if not isinstance(attestation, dict):
        raise AttestationError("attestation must be an object")

    import json
    from catalytic_chat.receipt import receipt_canonical_bytes

    receipt_json = json.loads(receipt_bytes.decode('utf-8'))

    canonical_bytes = receipt_canonical_bytes(receipt_json, attestation_override=None)

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
