#!/usr/bin/env python3
"""
Verify File Membership CLI (Phase 4.2.3)

Verify that a specific file was in the manifest at snapshot time using Merkle membership proofs.
This enables selective verification without requiring the full manifest.

Usage:
  # Verify from PROOF.json with embedded membership proofs
  python verify_file.py --proof-file CONTRACTS/_runs/<run_id>/PROOF.json \
                        --path path/to/file.txt \
                        --state pre  # or post

  # Verify with explicit proof and root
  python verify_file.py --proof-json '{"path":"...", "bytes_hash":"...", "steps":[...]}' \
                        --bytes-hash <sha256> \
                        --root <merkle_root>

Exit codes:
  0: Verification passed
  1: Verification failed
  2: Invalid arguments or missing data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.restore_proof import verify_file_membership
from CAPABILITY.PRIMITIVES.merkle import MerkleProof


def verify_from_proof_file(
    proof_file: Path,
    file_path: str,
    state: str,
) -> tuple[bool, dict]:
    """
    Verify file membership from a PROOF.json file.

    Args:
        proof_file: Path to PROOF.json
        file_path: The file path to verify
        state: 'pre' or 'post' (which state to check against)

    Returns:
        (success, report)
    """
    if not proof_file.exists():
        return False, {
            "ok": False,
            "code": "PROOF_FILE_NOT_FOUND",
            "message": f"Proof file not found: {proof_file}",
        }

    try:
        proof_data = json.loads(proof_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, {
            "ok": False,
            "code": "PROOF_FILE_INVALID_JSON",
            "message": f"Invalid JSON in proof file: {e}",
        }

    # Determine which state to use
    if state == "pre":
        state_key = "pre_state"
    elif state == "post":
        state_key = "post_state"
    else:
        return False, {
            "ok": False,
            "code": "INVALID_STATE",
            "message": f"Invalid state '{state}', must be 'pre' or 'post'",
        }

    if state_key not in proof_data:
        return False, {
            "ok": False,
            "code": "STATE_NOT_FOUND",
            "message": f"State '{state_key}' not found in proof file",
        }

    state_data = proof_data[state_key]
    domain_root = state_data.get("domain_root_hash")

    if not domain_root:
        return False, {
            "ok": False,
            "code": "DOMAIN_ROOT_MISSING",
            "message": f"domain_root_hash missing from {state_key}",
        }

    # Check if membership proofs are available
    membership_proofs = state_data.get("membership_proofs")
    if not membership_proofs:
        return False, {
            "ok": False,
            "code": "MEMBERSHIP_PROOFS_MISSING",
            "message": f"No membership_proofs in {state_key}. Run with --full-proofs to generate them.",
        }

    # Find proof for the requested file
    file_proof = membership_proofs.get(file_path)
    if not file_proof:
        return False, {
            "ok": False,
            "code": "FILE_PROOF_NOT_FOUND",
            "message": f"No membership proof for file '{file_path}' in {state_key}",
        }

    # Get the bytes_hash from the proof
    bytes_hash = file_proof.get("bytes_hash")
    if not bytes_hash:
        return False, {
            "ok": False,
            "code": "BYTES_HASH_MISSING",
            "message": f"bytes_hash missing from proof for '{file_path}'",
        }

    # Verify membership
    try:
        valid = verify_file_membership(file_path, bytes_hash, file_proof, domain_root)
    except Exception as e:
        return False, {
            "ok": False,
            "code": "VERIFICATION_ERROR",
            "message": f"Verification error: {e}",
        }

    if valid:
        return True, {
            "ok": True,
            "code": "VERIFIED",
            "message": f"File '{file_path}' verified in {state_key}",
            "path": file_path,
            "bytes_hash": bytes_hash,
            "domain_root": domain_root,
            "state": state,
        }
    else:
        return False, {
            "ok": False,
            "code": "VERIFICATION_FAILED",
            "message": f"File '{file_path}' failed verification against {state_key}",
            "path": file_path,
            "bytes_hash": bytes_hash,
            "domain_root": domain_root,
            "state": state,
        }


def verify_explicit(
    proof_json: str,
    path: str,
    bytes_hash: str,
    root: str,
) -> tuple[bool, dict]:
    """
    Verify file membership with explicit proof data.

    Args:
        proof_json: JSON string containing MerkleProof
        path: The file path
        bytes_hash: The file's SHA-256 hash
        root: The Merkle root to verify against

    Returns:
        (success, report)
    """
    try:
        proof_data = json.loads(proof_json)
    except json.JSONDecodeError as e:
        return False, {
            "ok": False,
            "code": "INVALID_PROOF_JSON",
            "message": f"Invalid proof JSON: {e}",
        }

    try:
        valid = verify_file_membership(path, bytes_hash, proof_data, root)
    except Exception as e:
        return False, {
            "ok": False,
            "code": "VERIFICATION_ERROR",
            "message": f"Verification error: {e}",
        }

    if valid:
        return True, {
            "ok": True,
            "code": "VERIFIED",
            "message": f"File '{path}' verified",
            "path": path,
            "bytes_hash": bytes_hash,
            "root": root,
        }
    else:
        return False, {
            "ok": False,
            "code": "VERIFICATION_FAILED",
            "message": f"File '{path}' failed verification",
            "path": path,
            "bytes_hash": bytes_hash,
            "root": root,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify file membership using Merkle proofs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode 1: From PROOF.json file
    mode1 = parser.add_argument_group("Mode 1: Verify from PROOF.json")
    mode1.add_argument(
        "--proof-file",
        type=Path,
        help="Path to PROOF.json with embedded membership proofs",
    )
    mode1.add_argument(
        "--path",
        type=str,
        help="File path to verify",
    )
    mode1.add_argument(
        "--state",
        choices=["pre", "post"],
        default="post",
        help="Which state to verify against (default: post)",
    )

    # Mode 2: Explicit proof
    mode2 = parser.add_argument_group("Mode 2: Explicit proof verification")
    mode2.add_argument(
        "--proof-json",
        type=str,
        help="MerkleProof as JSON string",
    )
    mode2.add_argument(
        "--bytes-hash",
        type=str,
        help="SHA-256 hash of file contents",
    )
    mode2.add_argument(
        "--root",
        type=str,
        help="Merkle root to verify against",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Determine mode
    if args.proof_file and args.path:
        # Mode 1: From PROOF.json
        success, report = verify_from_proof_file(
            args.proof_file,
            args.path,
            args.state,
        )
    elif args.proof_json and args.path and args.bytes_hash and args.root:
        # Mode 2: Explicit proof
        success, report = verify_explicit(
            args.proof_json,
            args.path,
            args.bytes_hash,
            args.root,
        )
    else:
        parser.print_help()
        print("\nError: Provide either (--proof-file, --path) or (--proof-json, --path, --bytes-hash, --root)")
        return 2

    # Output
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if success:
            print(f"[PASS] {report['message']}")
            if "bytes_hash" in report:
                print(f"  Hash: {report['bytes_hash']}")
            if "domain_root" in report:
                print(f"  Root: {report['domain_root']}")
        else:
            print(f"[FAIL] {report['message']}")
            print(f"  Code: {report['code']}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
