from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS"))

from CAPABILITY.PRIMITIVES.restore_proof import RestorationProofValidator
from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier
from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall
from catalytic_validator import CatalyticLedgerValidator


@pytest.fixture
def firewall():
    """Create write firewall for test artifacts under catalytic domains."""
    # Create necessary directories
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp"
    base.mkdir(parents=True, exist_ok=True)

    return WriteFirewall(
        tmp_roots=["CONTRACTS/_runs/_tmp"],
        durable_roots=["CONTRACTS/_runs/durable"],
        project_root=REPO_ROOT,
        exclusions=[],
    )


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


def _safe_write_text(path: Path, content: str, firewall: WriteFirewall) -> None:
    """Write text file through firewall (tmp writes)."""
    rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    firewall.safe_write(rel_path, content, kind="tmp")


def _safe_write_bytes(path: Path, content: bytes, firewall: WriteFirewall) -> None:
    """Write bytes file through firewall (tmp writes)."""
    rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    firewall.safe_write(rel_path, content, kind="tmp")


def _make_valid_spectrum05_run(run_dir: Path, firewall: WriteFirewall) -> tuple[BundleVerifier, Path]:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cryptography is required for SPECTRUM-05 strict verifier tests") from e

    firewall.safe_mkdir(str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"), kind="tmp")
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Output file referenced from OUTPUT_HASHES.json
    out_dir = run_dir / "out"
    firewall.safe_mkdir(str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/"), kind="tmp")
    output_path = out_dir / "result.txt"
    output_bytes = b"OK\n"
    _safe_write_bytes(output_path, output_bytes, firewall)
    output_rel = str(output_path.relative_to(REPO_ROOT)).replace("\\", "/")

    task_spec = {"task": "adversarial", "outputs": {"durable_paths": [output_rel]}}
    task_spec_bytes = json.dumps(task_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _safe_write_bytes(run_dir / "TASK_SPEC.json", task_spec_bytes, firewall)

    status_obj = {"status": "success", "cmp01": "pass"}
    _safe_write_text(run_dir / "STATUS.json", json.dumps(status_obj, sort_keys=True, separators=(",", ":")), firewall)

    output_hashes_obj = {
        "validator_semver": "1.0.0",
        "validator_build_id": "adversarial",
        "hashes": {output_rel: f"sha256:{_sha256_hex(output_bytes)}"},
    }
    _safe_write_text(
        run_dir / "OUTPUT_HASHES.json",
        json.dumps(output_hashes_obj, sort_keys=True, separators=(",", ":")),
        firewall,
    )

    proof_validator = RestorationProofValidator(REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json")
    proof = proof_validator.generate_proof(
        run_id=run_dir.name,
        catalytic_domains=["CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain"],
        pre_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        post_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        timestamp="CATALYTIC-DPT-02_CONFIG",
    )
    _safe_write_text(run_dir / "PROOF.json", json.dumps(proof, sort_keys=True, separators=(",", ":")), firewall)

    # Identity/signature (strict mode)
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_hex = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    validator_id = _sha256_hex(bytes.fromhex(public_key_hex))

    _safe_write_text(
        run_dir / "VALIDATOR_IDENTITY.json",
        json.dumps({"algorithm": "ed25519", "public_key": public_key_hex, "validator_id": validator_id}, sort_keys=True, separators=(",", ":")),
        firewall,
    )

    bundle_root = verifier._compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes)  # type: ignore[attr-defined]
    signed_payload = {"bundle_root": bundle_root, "decision": "ACCEPT", "validator_id": validator_id}
    _safe_write_text(
        run_dir / "SIGNED_PAYLOAD.json",
        json.dumps(signed_payload, sort_keys=True, separators=(",", ":")),
        firewall,
    )

    canonical_payload = verifier._canonicalize_json(signed_payload)  # type: ignore[attr-defined]
    signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload
    signature_hex = private_key.sign(signature_message).hex()
    _safe_write_text(
        run_dir / "SIGNATURE.json",
        json.dumps({"payload_type": "BUNDLE", "signature": signature_hex, "validator_id": validator_id}, sort_keys=True, separators=(",", ":")),
        firewall,
    )

    return verifier, output_path


def test_proof_hash_tamper_fails_closed_deterministically(firewall) -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "proof_tamper"
    _rm(base)
    run_dir = base / "run"
    verifier, _ = _make_valid_spectrum05_run(run_dir, firewall)

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


def test_output_hash_manifest_tamper_fails_closed(firewall) -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "output_hash_tamper"
    _rm(base)
    run_dir = base / "run"
    verifier, _ = _make_valid_spectrum05_run(run_dir, firewall)

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


def test_domain_roots_tamper_rejected_by_cmp01_validator(firewall) -> None:
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "domain_roots_tamper"
    _rm(base)
    run_dir = base / "run"
    firewall.safe_mkdir(str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"), kind="tmp")

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
    _safe_write_text(run_dir / "JOBSPEC.json", json.dumps(jobspec, indent=2), firewall)
    _safe_write_text(
        run_dir / "STATUS.json",
        json.dumps({"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True}, indent=2),
        firewall,
    )
    _safe_write_text(run_dir / "INPUT_HASHES.json", "{}", firewall)
    _safe_write_text(run_dir / "OUTPUT_HASHES.json", "{}", firewall)

    domain = jobspec["catalytic_domains"][0]
    post_manifest = {domain: {"file.txt": "a" * 64}}
    _safe_write_text(run_dir / "PRE_MANIFEST.json", json.dumps(post_manifest, indent=2), firewall)
    _safe_write_text(run_dir / "POST_MANIFEST.json", json.dumps(post_manifest, indent=2), firewall)
    _safe_write_text(run_dir / "RESTORE_DIFF.json", json.dumps({domain: {"added": {}, "removed": {}, "changed": {}}}, indent=2), firewall)
    _safe_write_text(run_dir / "OUTPUTS.json", "[]", firewall)
    _safe_write_text(run_dir / "LEDGER.jsonl", "{}\n", firewall)
    _safe_write_text(run_dir / "VALIDATOR_ID.json", json.dumps({"validator_semver": "0.1.0", "validator_build_id": "adversarial"}, indent=2), firewall)
    _safe_write_text(
        run_dir / "RUN_INFO.json",
        json.dumps({"run_id": "adv-domain-roots", "timestamp": "CATALYTIC-DPT-02_CONFIG", "intent": "x", "catalytic_domains": [domain], "exit_code": 0}, indent=2),
        firewall,
    )

    proof_validator = RestorationProofValidator(REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json")
    proof = proof_validator.generate_proof(
        run_id="adv-domain-roots",
        catalytic_domains=[domain],
        pre_state={domain: {"file.txt": "a" * 64}},
        post_state={domain: {"file.txt": "a" * 64}},
        timestamp="CATALYTIC-DPT-02_CONFIG",
    )
    _safe_write_text(run_dir / "PROOF.json", json.dumps(proof, indent=2), firewall)

    # DOMAIN_ROOTS.json must be non-empty for the validator to enforce the invariant.
    _safe_write_text(run_dir / "DOMAIN_ROOTS.json", json.dumps({domain: proof["post_state"]["domain_root_hash"]}, indent=2), firewall)

    ok, report = CatalyticLedgerValidator(run_dir).validate()
    assert ok is True
    assert report["valid"] is True

    # Tamper 1 byte: change the domain root hash.
    roots_path = run_dir / "DOMAIN_ROOTS.json"
    roots = json.loads(roots_path.read_text(encoding="utf-8"))
    roots[domain] = roots[domain].replace("a", "b", 1)
    _safe_write_text(roots_path, json.dumps(roots, indent=2), firewall)

    ok2, report2 = CatalyticLedgerValidator(run_dir).validate()
    assert ok2 is False
    assert report2["valid"] is False
    assert any("DOMAIN_ROOTS mismatch" in e for e in report2["errors"])
