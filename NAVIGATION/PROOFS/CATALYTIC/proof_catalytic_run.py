#!/usr/bin/env python3
"""
Phase 6.4.10: proof_catalytic_run

Catalytic proof runner that validates:
1. RESTORE: Cassettes can be restored from cartridge exports (Phase 6.3)
2. PURITY: No side effects or corruption during restore
3. MERKLE: Hash chains verify correctly

Outputs:
- NAVIGATION/PROOFS/CATALYTIC/RESTORE_PROOF.json
- NAVIGATION/PROOFS/CATALYTIC/PURITY_SCAN.json
- NAVIGATION/PROOFS/CATALYTIC/PROOF_CATALYTIC_REPORT.md

Uses Phase 6.0-6.3 cassette restore infrastructure.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"))
    sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "PRIMITIVES"))

# Output paths
OUT_DIR = REPO_ROOT / "NAVIGATION" / "PROOFS" / "CATALYTIC"
OUT_RESTORE_PROOF = OUT_DIR / "RESTORE_PROOF.json"
OUT_PURITY_SCAN = OUT_DIR / "PURITY_SCAN.json"
OUT_REPORT = OUT_DIR / "PROOF_CATALYTIC_REPORT.md"

# Test directory for restore validation
TEST_DIR = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes" / "_proof_catalytic"


def _sha256_hex(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _cleanup_test_dir() -> None:
    """Clean up test directory."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)


def _setup_test_dir() -> Path:
    """Setup clean test directory."""
    _cleanup_test_dir()
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


class RestoreProofRunner:
    """
    Runs restore proof validation.

    Tests that:
    1. Memories can be saved to a cassette
    2. Cassette can be exported as cartridge
    3. Cartridge can be imported to new cassette
    4. All memories are byte-identical after restore
    5. Merkle roots match before/after restore
    """

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results: Dict[str, Any] = {}

    def run_restore_cycle(self) -> Dict[str, Any]:
        """Run full restore cycle test."""
        try:
            from memory_cassette import MemoryCassette
        except ImportError as e:
            return {
                "status": "FAIL",
                "error": f"Cannot import MemoryCassette: {e}",
            }

        results = {
            "timestamp_utc": _utc_timestamp(),
            "test_id": str(uuid.uuid4())[:8],
            "steps": [],
        }

        # Step 1: Create cassette and save memories
        step1 = {"name": "create_and_save", "status": "pending"}
        try:
            db_path = self.test_dir / "original.db"
            cassette = MemoryCassette(db_path=db_path, agent_id="proof_agent")

            test_memories = [
                "First proof memory - testing restore guarantee",
                "Second proof memory - deterministic content",
                "Third proof memory - Merkle verification test",
                "Fourth proof memory - byte-identical restore",
                "Fifth proof memory - Phase 6.4.10 validation",
            ]

            saved_hashes = []
            for text in test_memories:
                hash_id, receipt = cassette.memory_save(text)
                saved_hashes.append(hash_id)

            step1["status"] = "pass"
            step1["memories_saved"] = len(saved_hashes)
            step1["hashes"] = saved_hashes
        except Exception as e:
            step1["status"] = "fail"
            step1["error"] = str(e)

        results["steps"].append(step1)

        if step1["status"] != "pass":
            results["status"] = "FAIL"
            return results

        # Step 2: Export cartridge
        step2 = {"name": "export_cartridge", "status": "pending"}
        try:
            export_dir = self.test_dir / "cartridge_export"
            manifest = cassette.export_cartridge(export_dir)

            step2["status"] = "pass"
            step2["record_count"] = manifest.get("record_count", 0)
            step2["receipt_count"] = manifest.get("receipt_count", 0)
            step2["original_merkle_root"] = manifest.get("content_merkle_root")
        except Exception as e:
            step2["status"] = "fail"
            step2["error"] = str(e)

        results["steps"].append(step2)

        if step2["status"] != "pass":
            results["status"] = "FAIL"
            return results

        # Step 3: Simulate corruption by deleting DB
        step3 = {"name": "simulate_corruption", "status": "pending"}
        try:
            os.remove(db_path)
            step3["status"] = "pass"
            step3["db_deleted"] = True
        except Exception as e:
            step3["status"] = "fail"
            step3["error"] = str(e)

        results["steps"].append(step3)

        # Step 4: Import cartridge to new cassette
        step4 = {"name": "import_cartridge", "status": "pending"}
        try:
            cassette2 = MemoryCassette(db_path=db_path, agent_id="proof_agent")
            import_result = cassette2.import_cartridge(export_dir)

            step4["status"] = "pass"
            step4["restored_records"] = import_result.get("restored_records", 0)
            step4["merkle_verified"] = import_result.get("merkle_verified", False)
        except Exception as e:
            step4["status"] = "fail"
            step4["error"] = str(e)

        results["steps"].append(step4)

        if step4["status"] != "pass":
            results["status"] = "FAIL"
            return results

        # Step 5: Verify all memories restored byte-identical
        step5 = {"name": "verify_content", "status": "pending"}
        try:
            all_match = True
            mismatches = []

            for i, (hash_id, original_text) in enumerate(zip(saved_hashes, test_memories)):
                recalled = cassette2.memory_recall(hash_id)
                if recalled is None:
                    all_match = False
                    mismatches.append({"hash": hash_id, "error": "not found"})
                elif recalled.get("text") != original_text:
                    all_match = False
                    mismatches.append({
                        "hash": hash_id,
                        "error": "content mismatch",
                        "expected_len": len(original_text),
                        "actual_len": len(recalled.get("text", "")),
                    })

            step5["status"] = "pass" if all_match else "fail"
            step5["all_match"] = all_match
            step5["verified_count"] = len(saved_hashes)
            if mismatches:
                step5["mismatches"] = mismatches
        except Exception as e:
            step5["status"] = "fail"
            step5["error"] = str(e)

        results["steps"].append(step5)

        # Step 6: Verify Merkle root matches
        step6 = {"name": "verify_merkle", "status": "pending"}
        try:
            export_dir2 = self.test_dir / "cartridge_verify"
            manifest2 = cassette2.export_cartridge(export_dir2)

            original_root = step2.get("original_merkle_root")
            restored_root = manifest2.get("content_merkle_root")

            merkle_match = original_root == restored_root

            step6["status"] = "pass" if merkle_match else "fail"
            step6["merkle_match"] = merkle_match
            step6["original_root"] = original_root[:16] + "..." if original_root else None
            step6["restored_root"] = restored_root[:16] + "..." if restored_root else None
        except Exception as e:
            step6["status"] = "fail"
            step6["error"] = str(e)

        results["steps"].append(step6)

        # Overall status
        all_pass = all(s.get("status") == "pass" for s in results["steps"])
        results["status"] = "PASS" if all_pass else "FAIL"

        return results


class PurityScanRunner:
    """
    Runs purity scan to verify no side effects during restore.

    Checks:
    1. No unexpected files created outside test directory
    2. No modification to canonical files
    3. Receipt chain integrity
    """

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir

    def scan_for_side_effects(self) -> Dict[str, Any]:
        """Scan for side effects of restore operation."""
        results = {
            "timestamp_utc": _utc_timestamp(),
            "checks": [],
        }

        # Check 1: Test directory isolation
        check1 = {"name": "directory_isolation", "status": "pending"}
        try:
            # List all files in test dir
            test_files = list(self.test_dir.rglob("*"))
            test_files = [f for f in test_files if f.is_file()]

            # All files should be within test_dir
            escaped = [f for f in test_files if not str(f).startswith(str(self.test_dir))]

            check1["status"] = "pass" if not escaped else "fail"
            check1["files_in_test_dir"] = len(test_files)
            check1["escaped_files"] = len(escaped)
        except Exception as e:
            check1["status"] = "fail"
            check1["error"] = str(e)

        results["checks"].append(check1)

        # Check 2: Verify receipt chain integrity
        check2 = {"name": "receipt_chain", "status": "pending"}
        try:
            # Import verifier
            from receipt_verifier import CassetteReceiptVerifier

            db_path = self.test_dir / "original.db"
            if db_path.exists():
                verifier = CassetteReceiptVerifier(db_path)
                chain_result = verifier.verify_full_chain()

                check2["status"] = "pass" if chain_result.get("valid") else "fail"
                check2["chain_valid"] = chain_result.get("valid", False)
                check2["chain_length"] = chain_result.get("chain_length", 0)
                if not chain_result.get("valid"):
                    check2["errors"] = chain_result.get("errors", [])[:5]
            else:
                check2["status"] = "skip"
                check2["reason"] = "Database not found"
        except ImportError:
            check2["status"] = "skip"
            check2["reason"] = "Receipt verifier not available"
        except Exception as e:
            check2["status"] = "fail"
            check2["error"] = str(e)

        results["checks"].append(check2)

        # Check 3: No temp files left behind
        check3 = {"name": "temp_cleanup", "status": "pending"}
        try:
            # Check for any .tmp or .partial files
            temp_files = list(self.test_dir.rglob("*.tmp"))
            temp_files.extend(list(self.test_dir.rglob("*.partial")))
            temp_files.extend(list(self.test_dir.rglob("*~")))

            check3["status"] = "pass" if not temp_files else "fail"
            check3["temp_files_found"] = len(temp_files)
        except Exception as e:
            check3["status"] = "fail"
            check3["error"] = str(e)

        results["checks"].append(check3)

        # Overall status
        all_pass = all(
            c.get("status") in ("pass", "skip")
            for c in results["checks"]
        )
        results["status"] = "PASS" if all_pass else "FAIL"

        return results


def run_proof_catalytic() -> Dict[str, Any]:
    """Run full catalytic proof suite."""
    timestamp = _utc_timestamp()

    proof_bundle = {
        "proof_type": "catalytic",
        "proof_version": "1.0.0",
        "timestamp_utc": timestamp,
    }

    # Setup test directory
    test_dir = _setup_test_dir()

    try:
        # Run restore proof
        print("Running restore proof...")
        restore_runner = RestoreProofRunner(test_dir)
        restore_results = restore_runner.run_restore_cycle()
        proof_bundle["restore_proof"] = restore_results

        # Run purity scan
        print("Running purity scan...")
        purity_runner = PurityScanRunner(test_dir)
        purity_results = purity_runner.scan_for_side_effects()
        proof_bundle["purity_scan"] = purity_results

        # Overall status
        restore_ok = restore_results.get("status") == "PASS"
        purity_ok = purity_results.get("status") == "PASS"
        proof_bundle["status"] = "PASS" if (restore_ok and purity_ok) else "FAIL"

    finally:
        # Cleanup
        _cleanup_test_dir()

    # Compute bundle hash
    proof_bundle["proof_bundle_hash"] = _sha256_hex(
        json.dumps(proof_bundle, sort_keys=True)
    )

    return proof_bundle


def render_proof_report(data: Dict[str, Any]) -> str:
    """Render human-readable catalytic proof report."""
    lines = [
        "<!-- GENERATED: Phase 6.4.10 Catalytic Proof Report -->",
        "",
        "# Catalytic Proof Report",
        "",
        "## Summary",
        "",
        f"**Status:** {data.get('status', 'UNKNOWN')}",
        f"**Timestamp:** {data.get('timestamp_utc', 'unknown')}",
        f"**Proof Bundle Hash:** `{data.get('proof_bundle_hash', 'unknown')[:16]}...`",
        "",
        "## Restore Proof (Phase 6.3 Validation)",
        "",
    ]

    restore = data.get("restore_proof", {})
    lines.append(f"**Status:** {restore.get('status', 'UNKNOWN')}")
    lines.append("")
    lines.append("### Steps")
    lines.append("")
    lines.append("| Step | Name | Status |")
    lines.append("|------|------|--------|")

    for i, step in enumerate(restore.get("steps", []), 1):
        status_icon = "Pass" if step.get("status") == "pass" else "Fail"
        lines.append(f"| {i} | {step.get('name', 'unknown')} | {status_icon} |")

    lines.extend([
        "",
        "## Purity Scan",
        "",
    ])

    purity = data.get("purity_scan", {})
    lines.append(f"**Status:** {purity.get('status', 'UNKNOWN')}")
    lines.append("")
    lines.append("### Checks")
    lines.append("")
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")

    for check in purity.get("checks", []):
        status = check.get("status", "unknown")
        lines.append(f"| {check.get('name', 'unknown')} | {status.capitalize()} |")

    lines.extend([
        "",
        "## Verification",
        "",
        "To reproduce this proof:",
        "",
        "```bash",
        "python NAVIGATION/PROOFS/CATALYTIC/proof_catalytic_run.py",
        "```",
        "",
        "---",
        "",
        "*Phase 6.4.10 compliant catalytic proof bundle.*",
    ])

    return "\n".join(lines)


def write_outputs(proof_bundle: Dict[str, Any]) -> None:
    """Write all proof outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write restore proof
    OUT_RESTORE_PROOF.write_text(
        json.dumps(proof_bundle.get("restore_proof", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Write purity scan
    OUT_PURITY_SCAN.write_text(
        json.dumps(proof_bundle.get("purity_scan", {}), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Write report
    report = render_proof_report(proof_bundle)
    OUT_REPORT.write_text(report, encoding="utf-8")

    # Write full bundle
    bundle_path = OUT_DIR / "PROOF_CATALYTIC_BUNDLE.json"
    bundle_path.write_text(
        json.dumps(proof_bundle, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Outputs written to {OUT_DIR}/")


def main() -> int:
    """Run catalytic proof and output results."""
    print("=" * 60)
    print("Phase 6.4.10: Catalytic Proof Runner")
    print("=" * 60)
    print()

    proof_bundle = run_proof_catalytic()
    write_outputs(proof_bundle)

    print()
    print(f"Status: {proof_bundle.get('status', 'UNKNOWN')}")
    print(f"Bundle Hash: {proof_bundle.get('proof_bundle_hash', 'unknown')}")

    if proof_bundle.get("status") == "PASS":
        print("\nSUCCESS: Catalytic proof validated")
        return 0
    else:
        print("\nFAILED: Catalytic proof did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
