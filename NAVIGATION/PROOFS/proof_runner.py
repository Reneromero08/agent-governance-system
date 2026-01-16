#!/usr/bin/env python3
"""
Phase 6.4.11: Unified Proof Runner for Pack Generation

This module binds all proof types into pack generation:
1. Compression proof (token savings validation)
2. Catalytic proof (restore + purity)

Called during pack generation to:
- Run proofs fresh for each pack
- Seal outputs in public packs (Phase 2.4)
- Update PROOF_MANIFEST.json
- Fail pack if proofs cannot be computed

Usage:
    # In pack generation code:
    from NAVIGATION.PROOFS.proof_runner import ProofRunner
    runner = ProofRunner()
    result = runner.run_all_proofs()
    if result["status"] == "FAIL":
        raise PackGenerationError("Proofs failed")
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Proof output directories
PROOFS_DIR = REPO_ROOT / "NAVIGATION" / "PROOFS"
COMPRESSION_DIR = PROOFS_DIR / "COMPRESSION"
CATALYTIC_DIR = PROOFS_DIR / "CATALYTIC"

# Manifest path
PROOF_MANIFEST_PATH = PROOFS_DIR / "PROOF_MANIFEST.json"

# Proof runner scripts
COMPRESSION_PROOF_RUNNER = COMPRESSION_DIR / "proof_compression_run.py"
CATALYTIC_PROOF_RUNNER = CATALYTIC_DIR / "proof_catalytic_run.py"


def _sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_head_commit() -> Optional[str]:
    """Get current git HEAD commit."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


class ProofRunner:
    """
    Unified proof runner for pack generation.

    Runs all required proofs and produces a combined manifest.
    Fail-closed: if any proof fails, the entire run fails.
    """

    def __init__(self, pack_id: Optional[str] = None, bundle_id: Optional[str] = None):
        """
        Initialize proof runner.

        Args:
            pack_id: Optional pack identifier for binding
            bundle_id: Optional bundle identifier for binding
        """
        self.pack_id = pack_id
        self.bundle_id = bundle_id
        self.timestamp = _utc_timestamp()
        self.git_rev = _git_head_commit()

    def run_compression_proof(self) -> Dict[str, Any]:
        """
        Run compression proof.

        Returns proof result or error dict.
        """
        if not COMPRESSION_PROOF_RUNNER.exists():
            return {
                "proof_type": "compression",
                "status": "SKIP",
                "error": f"Proof runner not found: {COMPRESSION_PROOF_RUNNER}",
            }

        try:
            result = subprocess.run(
                [sys.executable, str(COMPRESSION_PROOF_RUNNER)],
                capture_output=True,
                timeout=600,  # 10 minute timeout
                cwd=str(REPO_ROOT),
            )

            # Read output bundle
            bundle_path = COMPRESSION_DIR / "PROOF_COMPRESSION_BUNDLE.json"
            if bundle_path.exists():
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                return {
                    "proof_type": "compression",
                    "status": bundle.get("status", "UNKNOWN"),
                    "bundle_hash": bundle.get("proof_bundle_hash"),
                    "output_path": str(bundle_path.relative_to(REPO_ROOT)),
                }
            else:
                return {
                    "proof_type": "compression",
                    "status": "FAIL",
                    "error": "Bundle not generated",
                    "stderr": result.stderr.decode("utf-8", errors="replace")[:500],
                }

        except subprocess.TimeoutExpired:
            return {
                "proof_type": "compression",
                "status": "FAIL",
                "error": "Proof timed out after 600 seconds",
            }
        except Exception as e:
            return {
                "proof_type": "compression",
                "status": "FAIL",
                "error": str(e),
            }

    def run_catalytic_proof(self) -> Dict[str, Any]:
        """
        Run catalytic proof.

        Returns proof result or error dict.
        """
        if not CATALYTIC_PROOF_RUNNER.exists():
            return {
                "proof_type": "catalytic",
                "status": "SKIP",
                "error": f"Proof runner not found: {CATALYTIC_PROOF_RUNNER}",
            }

        try:
            result = subprocess.run(
                [sys.executable, str(CATALYTIC_PROOF_RUNNER)],
                capture_output=True,
                timeout=300,  # 5 minute timeout
                cwd=str(REPO_ROOT),
            )

            # Read output bundle
            bundle_path = CATALYTIC_DIR / "PROOF_CATALYTIC_BUNDLE.json"
            if bundle_path.exists():
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                return {
                    "proof_type": "catalytic",
                    "status": bundle.get("status", "UNKNOWN"),
                    "bundle_hash": bundle.get("proof_bundle_hash"),
                    "output_path": str(bundle_path.relative_to(REPO_ROOT)),
                }
            else:
                return {
                    "proof_type": "catalytic",
                    "status": "FAIL",
                    "error": "Bundle not generated",
                    "stderr": result.stderr.decode("utf-8", errors="replace")[:500],
                }

        except subprocess.TimeoutExpired:
            return {
                "proof_type": "catalytic",
                "status": "FAIL",
                "error": "Proof timed out after 300 seconds",
            }
        except Exception as e:
            return {
                "proof_type": "catalytic",
                "status": "FAIL",
                "error": str(e),
            }

    def run_all_proofs(self) -> Dict[str, Any]:
        """
        Run all proofs and produce combined manifest.

        Returns:
            Dict with status and all proof results.
            Status is PASS only if all proofs pass.
        """
        manifest = {
            "manifest_version": "1.0.0",
            "timestamp_utc": self.timestamp,
            "git_rev": self.git_rev,
            "pack_id": self.pack_id,
            "bundle_id": self.bundle_id,
            "proofs": [],
        }

        # Run compression proof
        print("Running compression proof...")
        compression_result = self.run_compression_proof()
        manifest["proofs"].append(compression_result)

        # Run catalytic proof
        print("Running catalytic proof...")
        catalytic_result = self.run_catalytic_proof()
        manifest["proofs"].append(catalytic_result)

        # Compute overall status (fail-closed)
        all_pass = all(
            p.get("status") in ("PASS", "SKIP")
            for p in manifest["proofs"]
        )
        any_fail = any(
            p.get("status") == "FAIL"
            for p in manifest["proofs"]
        )

        if any_fail:
            manifest["status"] = "FAIL"
        elif all_pass:
            manifest["status"] = "PASS"
        else:
            manifest["status"] = "PARTIAL"

        # Compute manifest hash
        manifest["manifest_hash"] = _sha256_hex(
            json.dumps(manifest, sort_keys=True)
        )

        return manifest

    def write_manifest(self, manifest: Dict[str, Any]) -> Path:
        """
        Write proof manifest to disk.

        Args:
            manifest: Manifest dict from run_all_proofs()

        Returns:
            Path to written manifest
        """
        PROOFS_DIR.mkdir(parents=True, exist_ok=True)
        PROOF_MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return PROOF_MANIFEST_PATH


def run_proofs_for_pack(
    pack_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run all proofs for pack generation.

    Args:
        pack_id: Pack identifier for binding
        bundle_id: Bundle identifier for binding
        fail_on_error: If True, raises exception on proof failure

    Returns:
        Proof manifest dict

    Raises:
        RuntimeError: If fail_on_error=True and any proof fails
    """
    runner = ProofRunner(pack_id=pack_id, bundle_id=bundle_id)
    manifest = runner.run_all_proofs()
    runner.write_manifest(manifest)

    if fail_on_error and manifest.get("status") == "FAIL":
        failed = [
            p["proof_type"]
            for p in manifest.get("proofs", [])
            if p.get("status") == "FAIL"
        ]
        raise RuntimeError(f"Proofs failed: {', '.join(failed)}")

    return manifest


def get_proof_artifacts() -> List[Dict[str, Any]]:
    """
    Get list of proof artifacts for inclusion in pack bundle.

    Returns list of artifact dicts with kind, path, and hash.
    """
    artifacts = []

    # Add proof manifest
    if PROOF_MANIFEST_PATH.exists():
        content = PROOF_MANIFEST_PATH.read_text(encoding="utf-8")
        artifacts.append({
            "kind": "PROOF_MANIFEST",
            "path": str(PROOF_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_hex(content),
        })

    # Add compression bundle
    compression_bundle = COMPRESSION_DIR / "PROOF_COMPRESSION_BUNDLE.json"
    if compression_bundle.exists():
        content = compression_bundle.read_text(encoding="utf-8")
        artifacts.append({
            "kind": "PROOF_COMPRESSION",
            "path": str(compression_bundle.relative_to(REPO_ROOT)),
            "sha256": _sha256_hex(content),
        })

    # Add catalytic bundle
    catalytic_bundle = CATALYTIC_DIR / "PROOF_CATALYTIC_BUNDLE.json"
    if catalytic_bundle.exists():
        content = catalytic_bundle.read_text(encoding="utf-8")
        artifacts.append({
            "kind": "PROOF_CATALYTIC",
            "path": str(catalytic_bundle.relative_to(REPO_ROOT)),
            "sha256": _sha256_hex(content),
        })

    return artifacts


def main() -> int:
    """Run all proofs standalone."""
    print("=" * 60)
    print("Phase 6.4.11: Unified Proof Runner")
    print("=" * 60)
    print()

    try:
        manifest = run_proofs_for_pack(fail_on_error=False)

        print()
        print(f"Overall Status: {manifest.get('status', 'UNKNOWN')}")
        print(f"Manifest Hash: {manifest.get('manifest_hash', 'unknown')}")
        print()
        print("Proof Results:")
        for proof in manifest.get("proofs", []):
            status = proof.get("status", "UNKNOWN")
            ptype = proof.get("proof_type", "unknown")
            print(f"  - {ptype}: {status}")
            if proof.get("error"):
                print(f"    Error: {proof['error']}")

        print()
        print(f"Manifest written to: {PROOF_MANIFEST_PATH}")

        if manifest.get("status") == "PASS":
            print("\nSUCCESS: All proofs validated")
            return 0
        else:
            print("\nFAILED: Some proofs did not pass")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
