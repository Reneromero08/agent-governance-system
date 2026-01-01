#!/usr/bin/env python3
"""
Verifier interoperability tests (Phase 4).

Goal: prove a second, code-independent verifier produces byte-identical results
vs the primary verifier for the same inputs.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier
from CAPABILITY.PRIMITIVES.verify_bundle_alt import AltBundleVerifier

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization

    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    CRYPTO_AVAILABLE = False


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _make_valid_bundle(tmp_root: Path, run_id: str = "run_ok") -> Path:
    run_dir = tmp_root / "LAW" / "CONTRACTS" / "_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_rel = "NAVIGATION/CORTEX/_generated/interop_output.txt"
    output_abs = tmp_root / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = b"interop"
    output_abs.write_bytes(output_bytes)

    task_spec = {"task_id": run_id, "outputs": {"durable_paths": [output_rel]}}
    task_spec_bytes = json.dumps(task_spec, indent=2).encode("utf-8")
    (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

    status = {"status": "success", "cmp01": "pass", "run_id": run_id}
    _write_json(run_dir / "STATUS.json", status)

    output_hashes = {
        "validator_semver": "1.0.0",
        "validator_build_id": "interop",
        "hashes": {output_rel: "sha256:" + hashlib.sha256(output_bytes).hexdigest()},
    }
    _write_json(run_dir / "OUTPUT_HASHES.json", output_hashes)

    proof = {"restoration_result": {"verified": True, "condition": "RESTORED_IDENTICAL"}}
    _write_json(run_dir / "PROOF.json", proof)

    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography not available (Ed25519 required for strict interop)")

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    public_key_hex = public_key_bytes.hex()
    validator_id = hashlib.sha256(public_key_bytes).hexdigest()

    identity = {"algorithm": "ed25519", "public_key": public_key_hex, "validator_id": validator_id}
    _write_json(run_dir / "VALIDATOR_IDENTITY.json", identity)

    task_spec_hash = hashlib.sha256(task_spec_bytes).hexdigest()
    bundle_preimage = {"output_hashes": output_hashes["hashes"], "status": status, "task_spec_hash": task_spec_hash}
    bundle_root = hashlib.sha256(_canonical(bundle_preimage)).hexdigest()

    signed_payload = {"bundle_root": bundle_root, "decision": "ACCEPT", "validator_id": validator_id}
    _write_json(run_dir / "SIGNED_PAYLOAD.json", signed_payload)

    signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + _canonical(signed_payload)
    signature_hex = private_key.sign(signature_message).hex()
    signature_obj = {"payload_type": "BUNDLE", "signature": signature_hex, "validator_id": validator_id}
    _write_json(run_dir / "SIGNATURE.json", signature_obj)

    return run_dir


def _result_bytes(result: dict) -> bytes:
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_interop_valid_bundle_byte_identical():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = _make_valid_bundle(root, "interop-ok")

        primary = BundleVerifier(project_root=root)
        alt = AltBundleVerifier(project_root=root)

        r1 = primary.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
        r2 = alt.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)

        assert r1 == r2
        assert _result_bytes(r1) == _result_bytes(r2)


def test_interop_tampered_bundle_reject_same_code_and_bytes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = _make_valid_bundle(root, "interop-tamper")

        # Tamper output bytes after hashes/signatures already written.
        output_abs = root / "NAVIGATION/CORTEX/_generated/interop_output.txt"
        output_abs.write_bytes(b"INTEROP-TAMPERED")

        primary = BundleVerifier(project_root=root)
        alt = AltBundleVerifier(project_root=root)

        r1 = primary.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
        r2 = alt.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)

        assert r1["ok"] is False and r2["ok"] is False
        assert r1["code"] == r2["code"]
        assert r1 == r2
        assert _result_bytes(r1) == _result_bytes(r2)


def test_interop_chain_verification_byte_identical():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_a = _make_valid_bundle(root, "chain-a")
        run_b = _make_valid_bundle(root, "chain-b")

        primary = BundleVerifier(project_root=root)
        alt = AltBundleVerifier(project_root=root)

        r1 = primary.verify_chain_spectrum05([run_a, run_b], strict=True, check_proof=True)
        r2 = alt.verify_chain_spectrum05([run_a, run_b], strict=True, check_proof=True)

        assert r1 == r2
        assert _result_bytes(r1) == _result_bytes(r2)


def test_interop_determinism_rerun_same_inputs_same_bytes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = _make_valid_bundle(root, "interop-determinism")

        primary = BundleVerifier(project_root=root)
        alt = AltBundleVerifier(project_root=root)

        r1a = primary.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
        r1b = primary.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
        r2a = alt.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)
        r2b = alt.verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)

        assert _result_bytes(r1a) == _result_bytes(r1b)
        assert _result_bytes(r2a) == _result_bytes(r2b)
        assert _result_bytes(r1a) == _result_bytes(r2a)
