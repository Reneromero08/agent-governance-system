#!/usr/bin/env python3
"""
Sign Proof CLI (SPECTRUM-04)

Sign PROOF.json files using Ed25519 or verify existing signatures.

Usage:
  # Generate a new keypair
  python sign_proof.py keygen --private-key keys/validator.key --public-key keys/validator.pub

  # Sign a proof (adds signature field to PROOF.json)
  python sign_proof.py sign --proof-file CONTRACTS/_runs/<run_id>/PROOF.json \
                            --private-key keys/validator.key

  # Verify a signed proof
  python sign_proof.py verify --proof-file CONTRACTS/_runs/<run_id>/PROOF.json \
                              [--public-key keys/validator.pub]

  # Show key info
  python sign_proof.py keyinfo --public-key keys/validator.pub

Exit codes:
  0: Success
  1: Verification failed
  2: Invalid arguments or file errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.signature import (
    generate_keypair,
    sign_proof,
    verify_signature,
    SignatureBundle,
    save_keypair,
    load_keypair,
    load_public_key_file,
    _compute_key_id,
    _bytes_to_hex,
)


def cmd_keygen(args: argparse.Namespace) -> int:
    """Generate a new Ed25519 keypair."""
    private_path = Path(args.private_key)
    public_path = Path(args.public_key)

    # Check if files exist
    if private_path.exists() and not args.force:
        print(f"ERROR: Private key file already exists: {private_path}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        return 2

    if public_path.exists() and not args.force:
        print(f"ERROR: Public key file already exists: {public_path}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        return 2

    # Create parent directories
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate keypair
    private_key, public_key = generate_keypair()
    save_keypair(private_key, public_key, private_path, public_path)

    key_id = _compute_key_id(public_key)

    if args.json:
        result = {
            "ok": True,
            "private_key_path": str(private_path),
            "public_key_path": str(public_path),
            "key_id": key_id,
            "public_key_hex": _bytes_to_hex(public_key),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Generated Ed25519 keypair:")
        print(f"  Private key: {private_path}")
        print(f"  Public key:  {public_path}")
        print(f"  Key ID:      {key_id}")
        print()
        print("IMPORTANT: Keep the private key file secure and outside the repository!")

    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    """Sign a PROOF.json file."""
    proof_path = Path(args.proof_file)
    private_path = Path(args.private_key)

    if not proof_path.exists():
        print(f"ERROR: Proof file not found: {proof_path}", file=sys.stderr)
        return 2

    if not private_path.exists():
        print(f"ERROR: Private key file not found: {private_path}", file=sys.stderr)
        return 2

    # Load proof
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in proof file: {e}", file=sys.stderr)
        return 2

    # Check if already signed
    if "signature" in proof and not args.force:
        print(f"ERROR: Proof already has a signature.", file=sys.stderr)
        print("Use --force to re-sign.", file=sys.stderr)
        return 2

    # Remove existing signature if re-signing
    proof_to_sign = {k: v for k, v in proof.items() if k != "signature"}

    # Load private key
    try:
        private_key = load_public_key_file(private_path)  # It's stored as hex
    except Exception as e:
        print(f"ERROR: Failed to load private key: {e}", file=sys.stderr)
        return 2

    # Sign
    try:
        bundle = sign_proof(proof_to_sign, private_key)
    except Exception as e:
        print(f"ERROR: Signing failed: {e}", file=sys.stderr)
        return 2

    # Add signature to proof
    proof_to_sign["signature"] = bundle.to_dict()

    # Write back
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = proof_path

    output_path.write_text(json.dumps(proof_to_sign, indent=2), encoding="utf-8")

    if args.json:
        result = {
            "ok": True,
            "proof_file": str(output_path),
            "key_id": bundle.key_id,
            "signature": bundle.signature[:16] + "...",
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Signed proof: {output_path}")
        print(f"  Key ID: {bundle.key_id}")
        print(f"  Timestamp: {bundle.timestamp}")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a signed PROOF.json file."""
    proof_path = Path(args.proof_file)

    if not proof_path.exists():
        print(f"ERROR: Proof file not found: {proof_path}", file=sys.stderr)
        return 2

    # Load proof
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in proof file: {e}", file=sys.stderr)
        return 2

    # Check for signature
    if "signature" not in proof:
        if args.json:
            print(json.dumps({"ok": False, "code": "NO_SIGNATURE", "message": "Proof has no signature"}))
        else:
            print("[FAIL] Proof has no signature field.")
        return 1

    # Extract signature bundle
    try:
        bundle = SignatureBundle.from_dict(proof["signature"])
    except Exception as e:
        if args.json:
            print(json.dumps({"ok": False, "code": "INVALID_SIGNATURE_FORMAT", "message": str(e)}))
        else:
            print(f"[FAIL] Invalid signature format: {e}")
        return 1

    # Load public key if provided
    public_key = None
    if args.public_key:
        try:
            public_key = load_public_key_file(Path(args.public_key))
        except Exception as e:
            print(f"ERROR: Failed to load public key: {e}", file=sys.stderr)
            return 2

    # Verify
    proof_without_sig = {k: v for k, v in proof.items() if k != "signature"}
    valid = verify_signature(proof_without_sig, bundle, public_key)

    if args.json:
        result = {
            "ok": valid,
            "code": "VERIFIED" if valid else "INVALID_SIGNATURE",
            "key_id": bundle.key_id,
            "algorithm": bundle.algorithm,
            "timestamp": bundle.timestamp,
        }
        print(json.dumps(result, indent=2))
    else:
        if valid:
            print(f"[PASS] Signature verified.")
            print(f"  Key ID: {bundle.key_id}")
            print(f"  Signed: {bundle.timestamp}")
        else:
            print(f"[FAIL] Signature verification failed.")
            print(f"  Key ID: {bundle.key_id}")

    return 0 if valid else 1


def cmd_keyinfo(args: argparse.Namespace) -> int:
    """Show information about a public key."""
    public_path = Path(args.public_key)

    if not public_path.exists():
        print(f"ERROR: Public key file not found: {public_path}", file=sys.stderr)
        return 2

    try:
        public_key = load_public_key_file(public_path)
    except Exception as e:
        print(f"ERROR: Failed to load public key: {e}", file=sys.stderr)
        return 2

    key_id = _compute_key_id(public_key)
    key_hex = _bytes_to_hex(public_key)

    if args.json:
        result = {
            "ok": True,
            "path": str(public_path),
            "key_id": key_id,
            "public_key_hex": key_hex,
            "algorithm": "Ed25519",
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Public Key Info:")
        print(f"  File:      {public_path}")
        print(f"  Key ID:    {key_id}")
        print(f"  Algorithm: Ed25519")
        print(f"  Hex:       {key_hex}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sign and verify PROOF.json files using Ed25519 (SPECTRUM-04)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # keygen
    p_keygen = subparsers.add_parser("keygen", help="Generate a new Ed25519 keypair")
    p_keygen.add_argument("--private-key", required=True, help="Path to save private key")
    p_keygen.add_argument("--public-key", required=True, help="Path to save public key")
    p_keygen.add_argument("--force", action="store_true", help="Overwrite existing files")

    # sign
    p_sign = subparsers.add_parser("sign", help="Sign a PROOF.json file")
    p_sign.add_argument("--proof-file", required=True, help="Path to PROOF.json")
    p_sign.add_argument("--private-key", required=True, help="Path to private key file")
    p_sign.add_argument("--output", help="Output path (defaults to overwriting input)")
    p_sign.add_argument("--force", action="store_true", help="Re-sign if already signed")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify a signed PROOF.json")
    p_verify.add_argument("--proof-file", required=True, help="Path to PROOF.json")
    p_verify.add_argument("--public-key", help="Path to public key (optional, uses embedded key if not provided)")

    # keyinfo
    p_keyinfo = subparsers.add_parser("keyinfo", help="Show public key information")
    p_keyinfo.add_argument("--public-key", required=True, help="Path to public key file")

    args = parser.parse_args()

    if args.command == "keygen":
        return cmd_keygen(args)
    elif args.command == "sign":
        return cmd_sign(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "keyinfo":
        return cmd_keyinfo(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
