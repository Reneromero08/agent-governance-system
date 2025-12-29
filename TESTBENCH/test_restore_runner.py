import json
import shutil
import sys
from pathlib import Path

import pytest

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.restore_runner import restore_bundle, restore_chain, RESTORE_CODES
from PRIMITIVES.verify_bundle import BundleVerifier


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_sha256_hex(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _write_signed_bundle(
    *,
    run_dir: Path,
    output_rel: str,
    output_bytes: bytes,
    include_proof: bool = True,
    proof_verified: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write output bytes at project_root-relative path.
    output_abs = (REPO_ROOT / output_rel)
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    output_abs.write_bytes(output_bytes)

    task_spec = {"task_id": run_dir.name, "outputs": {"durable_paths": [output_rel]}}
    task_spec_bytes = json.dumps(task_spec, indent=2).encode("utf-8")
    (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

    status_obj = {"status": "success", "cmp01": "pass", "run_id": run_dir.name}
    (run_dir / "STATUS.json").write_text(json.dumps(status_obj, indent=2))

    output_hashes_obj = {
        "validator_semver": "1.0.0",
        "validator_build_id": "test",
        "hashes": {output_rel: f"sha256:{_compute_sha256_hex(output_bytes)}"},
    }
    (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps(output_hashes_obj, indent=2))

    if include_proof:
        proof = {
            "proof_version": "1.0.0",
            "restoration_result": {
                "verified": bool(proof_verified),
                "condition": "RESTORED_IDENTICAL" if proof_verified else "RESTORATION_FAILED",
            },
        }
        (run_dir / "PROOF.json").write_text(json.dumps(proof, indent=2))

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    validator_id = _compute_sha256_hex(bytes.fromhex(public_key_hex))

    (run_dir / "VALIDATOR_IDENTITY.json").write_text(
        json.dumps({"algorithm": "ed25519", "public_key": public_key_hex, "validator_id": validator_id})
    )

    verifier = BundleVerifier(project_root=REPO_ROOT)
    bundle_root = verifier._compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes)

    signed_payload = {"bundle_root": bundle_root, "decision": "ACCEPT", "validator_id": validator_id}
    (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps(signed_payload))
    canonical_payload = verifier._canonicalize_json(signed_payload)
    msg = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload
    signature_hex = private_key.sign(msg).hex()

    (run_dir / "SIGNATURE.json").write_text(
        json.dumps({"payload_type": "BUNDLE", "signature": signature_hex, "validator_id": validator_id})
    )


@pytest.fixture()
def work_area():
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_test_restore_runner"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_restore_rejects_when_verifier_strict_fails(work_area):
    run_dir = work_area / "bad-bundle"
    run_dir.mkdir()
    (run_dir / "TASK_SPEC.json").write_text("{}")
    (run_dir / "STATUS.json").write_text(json.dumps({"status": "success", "cmp01": "pass"}))
    (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps({"hashes": {}}))
    (run_dir / "VALIDATOR_IDENTITY.json").write_text(json.dumps({"algorithm": "ed25519", "public_key": "0" * 64, "validator_id": "0" * 64}))
    (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps({"bundle_root": "0" * 64, "decision": "ACCEPT", "validator_id": "0" * 64}))
    # Missing SIGNATURE.json -> verifier fails

    restore_root = work_area / "dest"
    restore_root.mkdir()
    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_VERIFY_STRICT_FAILED"]


def test_restore_rejects_when_proof_missing(work_area):
    run_id = "no-proof"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello", include_proof=False)

    restore_root = work_area / "dest"
    restore_root.mkdir()
    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_PROOF_MISSING"]


def test_restore_reject_if_target_exists(work_area):
    run_id = "target-exists"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello")

    restore_root = work_area / "dest"
    restore_root.mkdir()
    target_abs = restore_root / output_rel
    target_abs.parent.mkdir(parents=True, exist_ok=True)
    target_abs.write_bytes(b"preexisting")

    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_TARGET_PATH_EXISTS"]
    assert target_abs.read_bytes() == b"preexisting"


def test_restore_rejects_traversal_path(work_area):
    run_id = "traversal"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/../evil.txt"
    # The physical file path is normalized by the filesystem.
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello")

    restore_root = work_area / "dest"
    restore_root.mkdir()
    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_PATH_TRAVERSAL_DETECTED"]
    assert result["details"]["phase"] == "PREFLIGHT"


def test_restore_rejects_symlink_escape(work_area):
    run_id = "symlink-escape"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello")

    restore_root = work_area / "dest"
    restore_root.mkdir()
    # Make restore_root/CONTRACTS a symlink pointing outside restore_root.
    (restore_root / "CONTRACTS").symlink_to(REPO_ROOT / "CONTRACTS", target_is_directory=True)

    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_SYMLINK_ESCAPE_DETECTED"]


def test_restore_staging_hash_mismatch_rolls_back(work_area, monkeypatch):
    import PRIMITIVES.restore_runner as rr

    run_id = "staging-mismatch"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello")

    restore_root = work_area / "dest"
    restore_root.mkdir()

    def corrupt_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"corrupt")

    monkeypatch.setattr(rr, "_copy_file", corrupt_copy)

    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"]
    assert not (restore_root / output_rel).exists()
    assert not any(p.name.startswith(".spectrum06_staging_") for p in restore_root.iterdir())
    assert not (restore_root / "RESTORE_MANIFEST.json").exists()
    assert not (restore_root / "RESTORE_REPORT.json").exists()


def test_restore_success_emits_artifacts_and_matches_hashes(work_area):
    run_id = "success"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    payload = b"hello\n"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=payload)

    restore_root = work_area / "dest"
    restore_root.mkdir()
    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is True
    assert result["code"] == "OK"

    restored_path = restore_root / output_rel
    assert restored_path.read_bytes() == payload

    manifest_path = restore_root / "RESTORE_MANIFEST.json"
    report_path = restore_root / "RESTORE_REPORT.json"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest_bytes = manifest_path.read_bytes()
    report_bytes = report_path.read_bytes()
    assert not manifest_bytes.endswith(b"\n")
    assert not report_bytes.endswith(b"\n")

    expected_hash = f"sha256:{_compute_sha256_hex(payload)}"
    expected_manifest = {"entries": [{"bytes": len(payload), "relative_path": output_rel, "sha256": expected_hash}]}
    expected_report = {
        "bundle_roots": [result["details"]["bundle_root"]],
        "chain_root": None,
        "ok": True,
        "restored_bytes": len(payload),
        "restored_files_count": 1,
    }

    assert manifest_bytes == _canonical_json_bytes(expected_manifest)
    assert report_bytes == _canonical_json_bytes(expected_report)


def test_chain_restore_all_or_nothing(work_area):
    run_id1 = "chain-1"
    run_id2 = "chain-2"
    run1 = work_area / run_id1
    run2 = work_area / run_id2
    out1 = f"CONTRACTS/_runs/_test_restore_runner/{run_id1}/out/a.txt"
    out2 = f"CONTRACTS/_runs/_test_restore_runner/{run_id2}/out/b.txt"

    _write_signed_bundle(run_dir=run1, output_rel=out1, output_bytes=b"a")
    _write_signed_bundle(run_dir=run2, output_rel=out2, output_bytes=b"b", include_proof=False)

    restore_root = work_area / "dest"
    restore_root.mkdir()

    result = restore_chain([run1, run2], restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_PROOF_MISSING"]
    assert not (restore_root / run_id1).exists()
    assert not (restore_root / run_id2).exists()
    assert not any(p.name.startswith(".spectrum06_chain_") for p in restore_root.iterdir())


def test_restore_rollback_failure_returns_restore_rollback_failed(work_area, monkeypatch):
    import PRIMITIVES.restore_runner as rr

    run_id = "rollback-fail"
    run_dir = work_area / run_id
    output_rel = f"CONTRACTS/_runs/_test_restore_runner/{run_id}/out/result.txt"
    _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"hello")

    restore_root = work_area / "dest"
    restore_root.mkdir()

    def corrupt_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"corrupt")

    monkeypatch.setattr(rr, "_copy_file", corrupt_copy)
    monkeypatch.setattr(rr, "_rollback_bundle", lambda *a, **k: False)

    result = restore_bundle(run_dir, restore_root, strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_ROLLBACK_FAILED"]
    assert result["details"]["cause_code"] == RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"]
