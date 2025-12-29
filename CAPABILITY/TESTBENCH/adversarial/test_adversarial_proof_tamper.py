from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from CAPABILITY.PRIMITIVES.restore_proof import RestorationProofValidator
from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier
from CAPABILITY.TOOLS.catalytic.catalytic_validator import CatalyticLedgerValidator


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
    # Update the paths to reflect the new directory structure
    run_dir = REPO_ROOT / "TESTBENCH" / "adversarial" / "run"
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cryptography is required for SPECTRUM-05 strict verifier tests") from e

    run_dir.mkdir(parents=True, exist_ok=True)
    verifier = BundleVerifier(project_root=REPO_ROOT)

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

    proof_validator = RestorationProofValidator(REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json")
    proof = proof_validator.generate_proof(
        run_id=run_dir.name,
        catalytic_domains=["CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain"],
        pre_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        post_state={"CONTRACTS/_runs/_tmp/adversarial/proof_tamper/domain": {}},
        timestamp="CATALYTIC-DPT-02_CONFIG",
    )
    (run_dir / "PROOF.json").write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")), encoding="utf-8")

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


def test_proof_hash_tamper_fails_closed_deterministic() -> None:
    # Add your deterministic check here
    pass

def test_proof_hash_tamper_fails_non_deterministic() -> None:
    # Add your non-deterministic check here
    pass

if __name__ == "__main__":
    pytest.main()
