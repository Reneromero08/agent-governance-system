#!/usr/bin/env python3
"""
Verify Release CLI (Crypto Safe - Phase 1)

Verify tamper-evident seals on AGS releases to enforce CCL v1.4 license
provisions (Sections 3.6, 3.7, 4.4).

Usage:
  # Verify with embedded public key
  python -m CAPABILITY.TOOLS.catalytic.verify_release --repo-dir .

  # Verify with explicit public key
  python -m CAPABILITY.TOOLS.catalytic.verify_release \
      --repo-dir . --pubkey keys/release.pub

Exit codes:
  0: Verification PASSED - all files match the sealed manifest
  1: Verification FAILED - tampering detected or seal invalid
  2: Invalid arguments or file errors

What verification checks:
  1. RELEASE_MANIFEST.json exists
  2. RELEASE_MANIFEST.json.sig exists
  3. Signature is valid (Ed25519)
  4. All files in manifest exist
  5. All file hashes match manifest

CCL v1.4 Reference:
  Section 3.6: "Any modification to Protected Artifacts MUST be
               accompanied by clear notice of modification."

  If verification FAILS, the seal has been broken, indicating:
  - File tampering (content changed)
  - File removal (protected artifact deleted)
  - Signature tampering (manifest or signature modified)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Default public key location
DEFAULT_PUBLIC_KEY = REPO_ROOT / "LAW" / "CONTRACTS" / "_keys" / "release.pub"

from CAPABILITY.PRIMITIVES.release_sealer import verify_seal
from CAPABILITY.PRIMITIVES.release_manifest import VerificationStatus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify tamper-evident release seals (Crypto Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to repository root containing RELEASE_MANIFEST.json",
    )
    parser.add_argument(
        "--pubkey",
        help=f"Path to public key file (default: {DEFAULT_PUBLIC_KEY.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if args.pubkey:
        public_key_path = Path(args.pubkey).resolve()
    elif DEFAULT_PUBLIC_KEY.exists():
        public_key_path = DEFAULT_PUBLIC_KEY
    else:
        public_key_path = None

    if not repo_dir.is_dir():
        print(f"ERROR: Repository directory not found: {repo_dir}", file=sys.stderr)
        return 2

    if public_key_path and not public_key_path.is_file():
        print(f"ERROR: Public key not found: {public_key_path}", file=sys.stderr)
        return 2

    # Verify
    result = verify_seal(repo_dir, public_key_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.verbose:
        print(result.verbose())
    else:
        if result.passed:
            print(f"[PASS] Verification succeeded")
            print(f"  Files verified: {result.verified_files}")
            if result.manifest_hash:
                print(f"  Manifest hash:  {result.manifest_hash[:16]}...")
        else:
            print(f"[FAIL] {result.status.value}")
            print(f"  {result.message}")
            if result.failed_path:
                print(f"  Failed path:    {result.failed_path}")
            if result.expected_hash:
                print(f"  Expected hash:  {result.expected_hash[:16]}...")
            if result.actual_hash:
                print(f"  Actual hash:    {result.actual_hash[:16]}...")

    # Exit code: 0 for PASS, 1 for any failure
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
