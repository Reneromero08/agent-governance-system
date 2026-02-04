#!/usr/bin/env python3
"""
Seal Release CLI (Crypto Safe - Phase 1)

Create tamper-evident seals for AGS releases to defend CCL v1.4 license
provisions (Sections 3.6, 3.7, 4.4).

Usage:
  # Generate a new keypair (uses default location: LAW/CONTRACTS/_keys/)
  python -m CAPABILITY.TOOLS.catalytic.seal_release keygen

  # Seal the repository (uses default key location)
  python -m CAPABILITY.TOOLS.catalytic.seal_release seal --repo-dir .

  # Seal with custom key path
  python -m CAPABILITY.TOOLS.catalytic.seal_release seal \
      --repo-dir . --private-key /path/to/custom.key

  # Seal with exclusions
  python -m CAPABILITY.TOOLS.catalytic.seal_release seal \
      --repo-dir . --exclude .ags-cas --exclude .git

Exit codes:
  0: Success
  1: Operation failed
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

# Default key location (gitignored for private key, tracked for public key)
DEFAULT_KEY_DIR = REPO_ROOT / "LAW" / "CONTRACTS" / "_keys"
DEFAULT_PRIVATE_KEY = DEFAULT_KEY_DIR / "release.key"
DEFAULT_PUBLIC_KEY = DEFAULT_KEY_DIR / "release.pub"

from CAPABILITY.PRIMITIVES.signature import (
    generate_keypair,
    save_keypair,
    _compute_key_id,
    _bytes_to_hex,
)
from CAPABILITY.PRIMITIVES.release_sealer import seal_repo


def cmd_keygen(args: argparse.Namespace) -> int:
    """Generate a new Ed25519 keypair for release signing."""
    private_path = Path(args.private_key) if args.private_key else DEFAULT_PRIVATE_KEY
    public_path = Path(args.public_key) if args.public_key else DEFAULT_PUBLIC_KEY

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
        print(f"Generated Ed25519 keypair for release signing:")
        print(f"  Private key: {private_path}")
        print(f"  Public key:  {public_path}")
        print(f"  Key ID:      {key_id}")
        print()
        print("IMPORTANT: Keep the private key file secure and outside the repository!")
        print("           Include the public key in your release for verification.")

    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    """Seal a repository by creating a signed manifest."""
    repo_dir = Path(args.repo_dir).resolve()
    private_path = Path(args.private_key).resolve() if args.private_key else DEFAULT_PRIVATE_KEY
    version = args.version

    if not repo_dir.is_dir():
        print(f"ERROR: Repository directory not found: {repo_dir}", file=sys.stderr)
        return 2

    if not private_path.is_file():
        print(f"ERROR: Private key not found: {private_path}", file=sys.stderr)
        return 2

    # Parse exclusions
    exclude_patterns = args.exclude if args.exclude else None

    try:
        receipt = seal_repo(
            repo_dir,
            private_path,
            version,
            exclude_patterns=exclude_patterns,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Sealing failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        result = {
            "ok": True,
            "manifest_path": receipt.manifest_path,
            "signature_path": receipt.signature_path,
            "manifest_hash": receipt.manifest_hash,
            "merkle_root": receipt.merkle_root,
            "file_count": receipt.file_count,
            "total_bytes": receipt.total_bytes,
            "git_commit": receipt.git_commit,
            "sealed_at": receipt.sealed_at,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Repository sealed successfully!")
        print()
        print(f"  Manifest:     {receipt.manifest_path}")
        print(f"  Signature:    {receipt.signature_path}")
        print(f"  Files:        {receipt.file_count:,}")
        print(f"  Total bytes:  {receipt.total_bytes:,}")
        print(f"  Merkle root:  {receipt.merkle_root[:16]}...")
        print(f"  Manifest hash:{receipt.manifest_hash[:16]}...")
        if receipt.git_commit:
            print(f"  Git commit:   {receipt.git_commit[:8]}...")
        print()
        print("To verify this release:")
        print(f"  python -m CAPABILITY.TOOLS.catalytic.verify_release \\")
        print(f"      --repo-dir {repo_dir} --pubkey <public_key_path>")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seal releases with tamper-evident cryptographic signatures (Crypto Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # keygen
    p_keygen = subparsers.add_parser("keygen", help="Generate a new Ed25519 keypair")
    p_keygen.add_argument(
        "--private-key",
        help=f"Path to save private key (default: {DEFAULT_PRIVATE_KEY.relative_to(REPO_ROOT)})",
    )
    p_keygen.add_argument(
        "--public-key",
        help=f"Path to save public key (default: {DEFAULT_PUBLIC_KEY.relative_to(REPO_ROOT)})",
    )
    p_keygen.add_argument("--force", action="store_true", help="Overwrite existing files")

    # seal
    p_seal = subparsers.add_parser("seal", help="Seal a repository")
    p_seal.add_argument("--repo-dir", required=True, help="Path to repository root")
    p_seal.add_argument("--version", required=True, help="Release version (e.g., v3.9.0)")
    p_seal.add_argument(
        "--private-key",
        help=f"Path to private key file (default: {DEFAULT_PRIVATE_KEY.relative_to(REPO_ROOT)})",
    )
    p_seal.add_argument(
        "--exclude",
        action="append",
        help="Path prefix to exclude (can be repeated)",
    )

    args = parser.parse_args()

    if args.command == "keygen":
        return cmd_keygen(args)
    elif args.command == "seal":
        return cmd_seal(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
