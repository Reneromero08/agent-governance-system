"""
SPECTRUM-04: Ed25519 Signature Primitives

Provides cryptographic signing for PROOF.json and bundle verification.
All signatures use Ed25519 with deterministic key generation.

Usage:
    from CAPABILITY.PRIMITIVES.signature import (
        generate_keypair,
        sign_proof,
        verify_signature,
        SignatureBundle,
    )

    # Generate a keypair
    private_key, public_key = generate_keypair()

    # Sign a proof
    signature_bundle = sign_proof(proof_dict, private_key)

    # Verify a signature
    valid = verify_signature(proof_dict, signature_bundle, public_key)

Key Management:
    - Private keys MUST be stored outside the repository
    - Public keys are embedded in validator identity
    - Key IDs are the first 8 hex chars of sha256(public_key_bytes)

Security Properties:
    - Ed25519 provides 128-bit security level
    - Signatures are deterministic (same message + key = same signature)
    - Verification fails fast on any tampering
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


# =============================================================================
# Key Types and Serialization
# =============================================================================


def _bytes_to_hex(data: bytes) -> str:
    """Convert bytes to lowercase hex string."""
    return data.hex().lower()


def _hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes."""
    return bytes.fromhex(hex_str)


def _compute_key_id(public_key_bytes: bytes) -> str:
    """Compute key ID from public key bytes (first 8 hex chars of sha256)."""
    return hashlib.sha256(public_key_bytes).hexdigest()[:8]


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a new Ed25519 keypair.

    Returns:
        (private_key_bytes, public_key_bytes)
        - private_key_bytes: 32 bytes (seed)
        - public_key_bytes: 32 bytes

    The private key bytes are the seed, not the full 64-byte private key.
    This is the format expected by Ed25519.from_private_bytes().
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    return private_bytes, public_bytes


def load_private_key(private_bytes: bytes) -> ed25519.Ed25519PrivateKey:
    """Load Ed25519 private key from raw bytes (32-byte seed)."""
    return ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)


def load_public_key(public_bytes: bytes) -> ed25519.Ed25519PublicKey:
    """Load Ed25519 public key from raw bytes (32 bytes)."""
    return ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)


# =============================================================================
# Signature Bundle
# =============================================================================


@dataclass
class SignatureBundle:
    """
    A cryptographic signature bundle for a proof.

    Contains the signature and metadata needed for verification.
    """

    signature: str  # 64-byte signature as hex (128 chars)
    public_key: str  # 32-byte public key as hex (64 chars)
    key_id: str  # First 8 hex chars of sha256(public_key)
    algorithm: str  # Always "Ed25519"
    timestamp: str  # ISO 8601 timestamp of signing

    def to_dict(self) -> Dict[str, str]:
        """Serialize to JSON-compatible dict."""
        return {
            "signature": self.signature,
            "public_key": self.public_key,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "SignatureBundle":
        """Deserialize from dict."""
        return cls(
            signature=d["signature"],
            public_key=d["public_key"],
            key_id=d["key_id"],
            algorithm=d["algorithm"],
            timestamp=d["timestamp"],
        )


# =============================================================================
# Signing and Verification
# =============================================================================


def _canonical_json_bytes(obj: Any) -> bytes:
    """
    Compute canonical JSON bytes for signing.

    Rules:
    - UTF-8 encoding
    - Keys sorted lexicographically
    - No extra whitespace (separators are ',' and ':')
    - No trailing newline
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_proof(
    proof: Dict[str, Any],
    private_key: Union[bytes, ed25519.Ed25519PrivateKey],
    timestamp: Optional[str] = None,
) -> SignatureBundle:
    """
    Sign a proof using Ed25519.

    The signature is computed over the canonical JSON bytes of the proof
    (excluding any existing signature field).

    Args:
        proof: The PROOF.json dict to sign (must not contain 'signature' field)
        private_key: 32-byte private key seed, or Ed25519PrivateKey object
        timestamp: Optional signing timestamp (defaults to current UTC time)

    Returns:
        SignatureBundle containing signature and metadata

    Raises:
        ValueError: If proof already contains 'signature' field
    """
    if "signature" in proof:
        raise ValueError("Proof already contains 'signature' field. Remove it before signing.")

    # Ensure we have a key object
    if isinstance(private_key, bytes):
        key_obj = load_private_key(private_key)
    else:
        key_obj = private_key

    # Get public key
    public_key = key_obj.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # Compute canonical bytes to sign
    message = _canonical_json_bytes(proof)

    # Sign
    signature_bytes = key_obj.sign(message)

    # Build bundle
    return SignatureBundle(
        signature=_bytes_to_hex(signature_bytes),
        public_key=_bytes_to_hex(public_bytes),
        key_id=_compute_key_id(public_bytes),
        algorithm="Ed25519",
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


def verify_signature(
    proof: Dict[str, Any],
    signature_bundle: Union[SignatureBundle, Dict[str, str]],
    public_key: Optional[Union[bytes, ed25519.Ed25519PublicKey]] = None,
) -> bool:
    """
    Verify an Ed25519 signature on a proof.

    Args:
        proof: The PROOF.json dict (must not contain 'signature' field)
        signature_bundle: SignatureBundle or dict with signature data
        public_key: Optional public key to verify against.
                    If None, uses the public_key from the bundle.

    Returns:
        True if signature is valid, False otherwise

    Notes:
        - If public_key is provided, it MUST match the bundle's public_key
        - Verification fails if algorithms don't match
    """
    # Normalize bundle
    if isinstance(signature_bundle, dict):
        bundle = SignatureBundle.from_dict(signature_bundle)
    else:
        bundle = signature_bundle

    # Check algorithm
    if bundle.algorithm != "Ed25519":
        return False

    # Get public key
    if public_key is None:
        try:
            key_bytes = _hex_to_bytes(bundle.public_key)
            key_obj = load_public_key(key_bytes)
        except (ValueError, Exception):
            return False
    elif isinstance(public_key, bytes):
        key_obj = load_public_key(public_key)
        # Verify it matches bundle
        if _bytes_to_hex(public_key) != bundle.public_key:
            return False
    else:
        key_obj = public_key
        # Verify it matches bundle
        key_bytes = key_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if _bytes_to_hex(key_bytes) != bundle.public_key:
            return False

    # Extract signature bytes
    try:
        signature_bytes = _hex_to_bytes(bundle.signature)
    except (ValueError, Exception):
        return False

    # Compute message (canonical JSON of proof without signature field)
    proof_copy = {k: v for k, v in proof.items() if k != "signature"}
    message = _canonical_json_bytes(proof_copy)

    # Verify
    try:
        key_obj.verify(signature_bytes, message)
        return True
    except Exception:
        return False


def verify_key_id(public_key: Union[bytes, ed25519.Ed25519PublicKey], expected_key_id: str) -> bool:
    """
    Verify that a public key matches an expected key ID.

    Args:
        public_key: 32-byte public key or Ed25519PublicKey object
        expected_key_id: Expected 8-char hex key ID

    Returns:
        True if key ID matches, False otherwise
    """
    if isinstance(public_key, bytes):
        key_bytes = public_key
    else:
        key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    actual_id = _compute_key_id(key_bytes)
    return actual_id == expected_key_id.lower()


# =============================================================================
# Key File I/O (for development/testing)
# =============================================================================


def save_keypair(
    private_key: bytes,
    public_key: bytes,
    private_path: Path,
    public_path: Path,
) -> None:
    """
    Save keypair to files.

    Args:
        private_key: 32-byte private key seed
        public_key: 32-byte public key
        private_path: Path to save private key (hex encoded)
        public_path: Path to save public key (hex encoded)

    Warning:
        Private key file should be stored securely outside the repo!
    """
    private_path.write_text(_bytes_to_hex(private_key))
    public_path.write_text(_bytes_to_hex(public_key))


def load_keypair(
    private_path: Path,
    public_path: Path,
) -> Tuple[bytes, bytes]:
    """
    Load keypair from files.

    Args:
        private_path: Path to private key file (hex encoded)
        public_path: Path to public key file (hex encoded)

    Returns:
        (private_key_bytes, public_key_bytes)
    """
    private_hex = private_path.read_text().strip()
    public_hex = public_path.read_text().strip()

    return _hex_to_bytes(private_hex), _hex_to_bytes(public_hex)


def load_public_key_file(public_path: Path) -> bytes:
    """Load just the public key from a file."""
    return _hex_to_bytes(public_path.read_text().strip())
