#!/usr/bin/env python3
"""
CAT_CHAT Golden Demo: Deterministic Bundle Execution
=====================================================

This demo shows the core CAT_CHAT pipeline:
1. Bundle creation (deterministic artifact packaging)
2. Bundle verification (hash integrity checks)
3. Bundle execution (receipt generation)
4. Receipt verification (chain integrity)

No external dependencies required beyond Python stdlib + jsonschema.
Runs from a fresh clone without any pre-existing state.

Usage:
    cd agent-governance-system
    $env:PYTHONPATH = "THOUGHT\\LAB\\CAT_CHAT"
    python THOUGHT\\LAB\\CAT_CHAT\\golden_demo\\golden_demo.py
"""

import hashlib
import json
import sys
import tempfile
import shutil
from pathlib import Path

DEMO_ROOT = Path(__file__).parent
CAT_CHAT_ROOT = DEMO_ROOT.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))


def canonical_json(data):
    """Serialize to canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_hex(content):
    """Compute SHA256 hex digest."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def print_header(title):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_step(num, total, description):
    """Print a step indicator."""
    print(f"\n[{num}/{total}] {description}...")


def create_demo_bundle(workspace):
    """Create a minimal bundle demonstrating the protocol."""
    bundle_dir = workspace / "demo_bundle"
    bundle_dir.mkdir(parents=True)
    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir()

    # Read demo content from fixtures
    demo_content_path = DEMO_ROOT / "fixtures" / "demo_content.txt"
    if demo_content_path.exists():
        content = demo_content_path.read_text(encoding='utf-8')
    else:
        # Fallback if fixtures not found
        content = "# Golden Demo Content\n\nThis is deterministic content.\n"

    # Ensure trailing newline
    if not content.endswith('\n'):
        content += '\n'

    # Create artifact
    artifact_id = sha256_hex("demo_content")[:16]
    artifact_path = artifacts_dir / f"{artifact_id}.txt"
    artifact_path.write_text(content, encoding='utf-8')

    content_hash = sha256_hex(content)
    content_bytes = len(content.encode('utf-8'))

    # Build step
    step = {
        "step_id": "step_001",
        "ordinal": 1,
        "op": "READ_SECTION",
        "refs": {"section_id": "golden_demo_content"},
        "constraints": {"slice": "lines[0:50]"},
        "expected_outputs": {}
    }

    # Build artifact manifest entry
    artifact = {
        "artifact_id": artifact_id,
        "kind": "SECTION_SLICE",
        "ref": "golden_demo_content",
        "slice": "lines[0:50]",
        "path": f"artifacts/{artifact_id}.txt",
        "sha256": content_hash,
        "bytes": content_bytes
    }

    # Compute plan hash
    plan_data = {
        "run_id": "golden_demo_run",
        "steps": [step]
    }
    plan_hash = sha256_hex(canonical_json(plan_data))

    # Build pre-manifest (empty bundle_id and root_hash)
    pre_manifest = {
        "bundle_version": "5.0.0",
        "bundle_id": "",
        "run_id": "golden_demo_run",
        "job_id": "golden_demo_job",
        "message_id": "golden_demo_msg",
        "plan_hash": plan_hash,
        "steps": [step],
        "inputs": {"symbols": [], "files": [], "slices": ["lines[0:50]"]},
        "artifacts": [artifact],
        "hashes": {"root_hash": ""},
        "provenance": {}
    }

    # Compute bundle_id from pre-manifest
    bundle_id = sha256_hex(canonical_json(pre_manifest))

    # Compute root_hash from artifacts
    hash_string = f"{artifact_id}:{content_hash}\n"
    root_hash = sha256_hex(hash_string)

    # Build final manifest
    manifest = dict(pre_manifest)
    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    # Write bundle.json with trailing newline
    bundle_json = bundle_dir / "bundle.json"
    manifest_json = canonical_json(manifest)
    bundle_json.write_text(manifest_json + "\n", encoding='utf-8')

    return bundle_dir, manifest


def verify_bundle(bundle_dir):
    """Verify bundle integrity."""
    from catalytic_chat.bundle import BundleVerifier

    verifier = BundleVerifier(bundle_dir)
    return verifier.verify()


def execute_bundle(bundle_dir, receipt_out):
    """Execute bundle and generate receipt."""
    from catalytic_chat.executor import BundleExecutor

    executor = BundleExecutor(
        bundle_dir=bundle_dir,
        receipt_out=receipt_out
    )
    return executor.execute()


def verify_receipt(receipt_path):
    """Verify receipt hash computation by loading from file."""
    from catalytic_chat.receipt import compute_receipt_hash, load_receipt

    # Load the actual receipt from file (not the executor return dict)
    receipt = load_receipt(receipt_path)
    if receipt is None:
        return False, None, None

    stored_hash = receipt.get("receipt_hash")
    computed_hash = compute_receipt_hash(receipt)

    return stored_hash == computed_hash, stored_hash, computed_hash


def run_demo():
    """Run the golden demo."""
    print_header("CAT_CHAT GOLDEN DEMO - Deterministic Execution")
    print("\nThis demo shows the core CAT_CHAT pipeline:")
    print("  - Bundle creation (deterministic packaging)")
    print("  - Bundle verification (hash integrity)")
    print("  - Bundle execution (receipt generation)")
    print("  - Receipt verification (chain integrity)")

    # Create temporary workspace
    print_step(1, 5, "Creating temporary workspace")
    workspace = Path(tempfile.mkdtemp(prefix="cat_chat_demo_"))
    print(f"  Workspace: {workspace}")

    try:
        # Create demo bundle
        print_step(2, 5, "Creating demo bundle")
        bundle_dir, manifest = create_demo_bundle(workspace)
        print(f"  Bundle ID: {manifest['bundle_id'][:32]}...")
        print(f"  Root Hash: {manifest['hashes']['root_hash'][:32]}...")
        print(f"  Artifacts: {len(manifest['artifacts'])}")

        # Verify bundle
        print_step(3, 5, "Verifying bundle integrity")
        verify_result = verify_bundle(bundle_dir)
        print(f"  Status: {verify_result['status'].upper()}")
        print(f"  Bundle ID verified: {verify_result['bundle_id'][:32]}...")
        print(f"  Root hash verified: {verify_result['root_hash'][:32]}...")

        if verify_result['status'] != 'success':
            print("\n  ERROR: Bundle verification failed!")
            return 1

        # Execute bundle
        print_step(4, 5, "Executing bundle and generating receipt")
        receipt_out = workspace / "receipt.json"
        exec_result = execute_bundle(bundle_dir, receipt_out)
        print(f"  Outcome: {exec_result['outcome']}")
        print(f"  Receipt Hash: {exec_result['receipt_hash'][:32]}...")
        print(f"  Receipt Path: {exec_result['receipt_path']}")

        # Verify receipt
        print_step(5, 5, "Verifying receipt integrity")
        receipt_valid, stored, computed = verify_receipt(receipt_out)
        print(f"  Hash Match: {receipt_valid}")
        print(f"  Stored:   {stored[:32]}...")
        print(f"  Computed: {computed[:32]}...")

        if not receipt_valid:
            print("\n  ERROR: Receipt hash mismatch!")
            return 1

        # Summary
        print_header("DEMO COMPLETE - Summary")
        print(f"Bundle ID:     {manifest['bundle_id'][:48]}...")
        print(f"Root Hash:     {manifest['hashes']['root_hash'][:48]}...")
        print(f"Receipt Hash:  {exec_result['receipt_hash'][:48]}...")
        print(f"Outcome:       {exec_result['outcome']}")
        print()
        print("All verifications PASSED.")
        print("The system is deterministic and fail-closed.")

        # Determinism check
        print_header("DETERMINISM CHECK")
        print("Re-running bundle creation to verify identical hashes...")

        workspace2 = Path(tempfile.mkdtemp(prefix="cat_chat_demo2_"))
        bundle_dir2, manifest2 = create_demo_bundle(workspace2)

        if manifest["bundle_id"] == manifest2["bundle_id"]:
            print(f"  Bundle ID: IDENTICAL")
        else:
            print(f"  Bundle ID: MISMATCH!")
            return 1

        if manifest["hashes"]["root_hash"] == manifest2["hashes"]["root_hash"]:
            print(f"  Root Hash: IDENTICAL")
        else:
            print(f"  Root Hash: MISMATCH!")
            return 1

        print()
        print("Determinism verified - same inputs produce identical outputs.")

        shutil.rmtree(workspace2, ignore_errors=True)

        return 0

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nMake sure PYTHONPATH includes THOUGHT\\LAB\\CAT_CHAT")
        print("Example: $env:PYTHONPATH = \"THOUGHT\\LAB\\CAT_CHAT\"")
        return 2

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 3

    finally:
        # Cleanup
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    """Entry point."""
    sys.exit(run_demo())


if __name__ == "__main__":
    main()
