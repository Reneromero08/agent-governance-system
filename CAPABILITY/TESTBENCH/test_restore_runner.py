import json
import shutil
import sys
from pathlib import Path

import pytest

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.restore_runner import restore_bundle, restore_chain, RESTORE_CODES
from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier


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
    base = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_test_restore_runner"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_restore_rejects_when_verifier_strict_fails(work_area):
    run_dir = work_area / "bad-bundle"
    run_dir.mkdir()
    (run_dir / "TASK_SPEC.json").write_text("{}")
    (run_dir / "STATUS.json").write_text(json.dumps({"status": "success", "cmp01": "pass"}))
    (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps({"hashes": {}}))
    (run_dir / "VALIDATOR_IDENTITY.json").write_text(json.dumps({"algorithm": "ed25519", "public_key": "", "validator_id": ""}))
    (run_dir / "PROOF.json").write_text("")

    (work_area / "dest").mkdir()
    result = restore_bundle(run_dir, work_area / "dest", strict=True)
    assert result["ok"] is False
    assert result["code"] == RESTORE_CODES["RESTORE_VERIFY_STRICT_FAILED"]
    assert not any(p.exists() for p in (work_area / "dest").iterdir())


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


def test_restore_rollback_failure_returns_restore_rollback_failed(work_area, monkeypatch):
    import CAPABILITY.PRIMITIVES.restore_runner as rr

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

