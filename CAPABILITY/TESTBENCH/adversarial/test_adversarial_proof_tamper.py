from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))
sys.path.insert(0, str(REPO_ROOT / "TOOLS"))

from PRIMITIVES.restore_proof import RestorationProofValidator
from PRIMITIVES.verify_bundle import BundleVerifier
from catalytic_validator import CatalyticLedgerValidator


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_valid_spectrum05_run(run_dir: Path) -> tuple[BundleVerifier, Path]:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cryptography is required for SPECTRUM-05 strict verifier tests") from e

    run_dir.mkdir(parents=True, exist_ok=True)
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Output file referenced from OUTPUT_HASHES.json
    out_dir = run_dir / "out"
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / "result.txt"
    output_bytes = b"OK\n"
    output_path.write_bytes(output_bytes)
    output_rel = str(output_path.relative_to(REPO_ROOT)).replace("\\", "/")

    task_spec = {"task": "adversarial", "outputs": {"durable_paths": [output_rel]}}
    task_spec_bytes = json.dumps(task_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

    status_obj = {"status": "success", "cmp01": "pass"}
    (run_dir / "STATUS.json").write_text(json.dumps(status_obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    output_hashes_obj = {
        "validator_semver": "1.0.0",
        "validator_build_id": "adversarial",
        "hashes": {output_rel: f"sha256:{_sha256_hex(output_bytes)}"},
    }
    (run_dir / "OUTPUT_HASHES.json").write_text(
        json.dumps(output_hashes_obj, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    proof_validator = RestorationProofValidator(REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json")
    proof = proof_validator.generate_proof(
        run_id=run_dir.name,
        catalytic_domains=["CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain"],
        pre_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        post_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        timestamp="CATALYTIC-DPT-02_CONFIG",
    )
    (run_dir / "PROOF.json").write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    # Identity/signature (strict mode)
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_hex = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    validator_id = _sha256_hex(bytes.fromhex(public_key_hex))

    (run_dir / "VALIDATOR_IDENTITY.json").write_text(
        json.dumps({"algorithm": "ed25519", "public_key": public_key_hex, "validator_id": validator_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    bundle_root = verifier._compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes)  # type: ignore[attr-defined]
    signed_payload = {"bundle_root": bundle_root, "decision": "ACCEPT", "validator_id": validator_id}
    (run_dir / "SIGNED_PAYLOAD.json").write_text(
        json.dumps(signed_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    canonical_payload = verifier._canonicalize_json(signed_payload)  # type: ignore[attr-defined]
    signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload
    signature_hex = private_key.sign(signature_message).hex()
    (run_dir / "SIGNATURE.json").write_text(
        json.dumps({"payload_type": "BUNDLE", "signature": signature_hex, "validator_id": validator_id}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    return verifier, output_path


def test_proof_hash_tamper_fails_closed_deterministically() -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "proof_tamper"
    _rm(base)
    run_dir = base / "run"
    verifier, _ = _make_valid_spectrum05_run(run_dir)

    ok = verifier.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
    assert ok["ok"] is True

    proof_path = run_dir / "PROOF.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert isinstance(proof.get("proof_hash"), str) and len(proof["proof_hash"]) == 64

    # Tamper a single byte in a JSON string value while keeping proof_hash unchanged.
    proof["proof_version"] = "1.0.1"
    proof_path.write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    res = verifier.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
    assert res["ok"] is False
    assert res["code"] == "ARTIFACT_MALFORMED"
    assert "proof_hash mismatch" in res["message"]


def test_output_hash_manifest_tamper_fails_closed() -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "output_hash_tamper"
    _rm(base)
    run_dir = base / "run"
    verifier, _ = _make_valid_spectrum05_run(run_dir)

    res0 = verifier.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
    assert res0["ok"] is True

    # Swap one hash value (do not re-sign): must fail closed deterministically (bundle root mismatch).
    oh_path = run_dir / "OUTPUT_HASHES.json"
    obj = json.loads(oh_path.read_text(encoding="utf-8"))
    any_key = sorted(obj["hashes"].keys())[0]
    obj["hashes"][any_key] = obj["hashes"][any_key].replace("0", "1", 1)
    oh_path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    res = verifier.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
    assert res["ok"] is False
    assert res["code"] == "BUNDLE_ROOT_MISMATCH"


def test_domain_roots_tamper_rejected_by_cmp01_validator() -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "domain_roots_tamper"
    _rm(base)
    run_dir = base / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal canonical + legacy set required by CatalyticLedgerValidator.
    jobspec = {
        "job_id": "adv-domain-roots",
        "phase": 4,
        "task_type": "test_execution",
        "intent": "domain roots tamper test",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": ["CONTRACTS/_runs/_tmp/adversarial/domain_roots_tamper/domain"],
        "determinism": "deterministic",
    }
    (run_dir / "JOBSPEC.json").write_text(json.dumps(jobspec, indent=2), encoding="utf-8")
    (run_dir / "STATUS.json").write_text(
        json.dumps({"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "INPUT_HASHES.json").write_text("{}", encoding="utf-8")
    (run_dir / "OUTPUT_HASHES.json").write_text("{}", encoding="utf-8")

    domain = jobspec["catalytic_domains"][0]
    post_manifest = {domain: {"file.txt": "a" * 64}}
    (run_dir / "PRE_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2), encoding="utf-8")
    (run_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2), encoding="utf-8")
    (run_dir / "RESTORE_DIFF.json").write_text(json.dumps({domain: {"added": {}, "removed": {}, "changed": {}}}, indent=2), encoding="utf-8")
    (run_dir / "OUTPUTS.json").write_text("[]", encoding="utf-8")
    (run_dir / "LEDGER.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "VALIDATOR_ID.json").write_text(json.dumps({"validator_semver": "0.1.0", "validator_build_id": "adversarial"}, indent=2), encoding="utf-8")
    (run_dir / "RUN_INFO.json").write_text(
        json.dumps({"run_id": "adv-domain-roots", "timestamp": "CATALYTIC-DPT-02_CONFIG", "intent": "x", "catalytic_domains": [domain], "exit_code": 0}, indent=2),
        encoding="utf-8",
    )

    proof_validator = RestorationProofValidator(REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json")
    proof = proof_validator.generate_proof(
        run_id="adv-domain-roots",
        catalytic_domains=[domain],
        pre_state={domain: {"file.txt": "a" * 64}},
        post_state={domain: {"file.txt": "a" * 64}},
        timestamp="CATALYTIC-DPT-02_CONFIG",
    )
    (run_dir / "PROOF.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")

    # DOMAIN_ROOTS.json must be non-empty for the validator to enforce the invariant.
    (run_dir / "DOMAIN_ROOTS.json").write_text(json.dumps({domain: proof["post_state"]["domain_root_hash"]}, indent=2), encoding="utf-8")

    ok, report = CatalyticLedgerValidator(run_dir).validate()
    assert ok is True
    assert report["valid"] is True

    # Tamper 1 byte: change the domain root hash.
    roots_path = run_dir / "DOMAIN_ROOTS.json"
    roots = json.loads(roots_path.read_text(encoding="utf-8"))
    roots[domain] = roots[domain].replace("a", "b", 1)
    roots_path.write_text(json.dumps(roots, indent=2), encoding="utf-8")

    ok2, report2 = CatalyticLedgerValidator(run_dir).validate()
    assert ok2 is False
    assert report2["valid"] is False
    assert any("DOMAIN_ROOTS mismatch" in e for e in report2["errors"])
