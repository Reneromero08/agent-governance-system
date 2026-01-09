from __future__ import annotations

import hashlib
import shutil
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[3]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.cas_store import CatalyticStore, normalize_relpath
from CAPABILITY.PRIMITIVES.ledger import Ledger
from CAPABILITY.PRIMITIVES.merkle import build_manifest_root
from CAPABILITY.PRIMITIVES.restore_proof import RestorationProofValidator, canonical_json_bytes, compute_domain_manifest
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


def _make_test_writer(test_root: Path) -> GuardedWriter:
    rel_root = str(test_root.relative_to(REPO_ROOT)).replace("\\", "/")
    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=[],  # Empty - we want durable writes only
        durable_roots=[rel_root],
        exclusions=[],
    )
    writer.open_commit_gate()
    return writer


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_jobspec(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "phase": 1,
        "task_type": "primitive_implementation",
        "intent": "proof wiring test",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": [],
        "determinism": "deterministic",
    }


def _ledger_record(*, run_id: str, timestamp: str, pre_manifest: dict, post_manifest: dict, domain_roots: dict) -> dict:
    # Keep minimal but schema-valid. RESTORE_DIFF derived deterministically.
    def restore_diff(pre_m: dict, post_m: dict) -> dict:
        out = {}
        for domain in sorted(set(pre_m.keys()) | set(post_m.keys())):
            pre_files = pre_m.get(domain, {})
            post_files = post_m.get(domain, {})
            added = {p: post_files[p] for p in sorted(set(post_files) - set(pre_files))}
            removed = {p: pre_files[p] for p in sorted(set(pre_files) - set(post_files))}
            changed = {p: post_files[p] for p in sorted(set(pre_files) & set(post_files)) if pre_files[p] != post_files[p]}
            out[domain] = {"added": added, "removed": removed, "changed": changed}
        return out

    return {
        "JOBSPEC": _minimal_jobspec(run_id),
        "RUN_INFO": {
            "run_id": run_id,
            "timestamp": timestamp,
            "intent": "test",
            "catalytic_domains": list(domain_roots.keys()),
            "exit_code": 0,
            "restoration_verified": True,
        },
        "PRE_MANIFEST": pre_manifest,
        "POST_MANIFEST": post_manifest,
        "RESTORE_DIFF": restore_diff(pre_manifest, post_manifest),
        "OUTPUTS": [],
        "STATUS": {"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True},
        "VALIDATOR_ID": {"validator_semver": "0.1.0", "validator_build_id": "test"},
    }


def test_rerun_determinism_proof_and_domain_roots_bytes_identical() -> None:
    run_id = "run-determinism"
    timestamp = "CATALYTIC-DPT-02_CONFIG"

    # Create test_root inside project
    test_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "pytest_tmp" / f"proof_wiring_{uuid.uuid4().hex[:8]}"
    test_root.mkdir(parents=True, exist_ok=True)

    try:
        # Create a test writer for the temp directory
        test_writer = _make_test_writer(test_root)

        # Create deterministic domain content.
        domain = test_root / "domain"
        domain.mkdir(parents=True)
        (domain / "a.txt").write_bytes(b"alpha")
        (domain / "b" / "c.txt").parent.mkdir(parents=True)
        (domain / "b" / "c.txt").write_bytes(b"charlie")

        def generate_once(out_dir: Path) -> tuple[bytes, bytes]:
            cas = CatalyticStore(out_dir / "CAS", writer=test_writer)
            pre_manifest = {"domain": compute_domain_manifest(domain, cas=cas)}
            post_manifest = {"domain": compute_domain_manifest(domain, cas=cas)}

            post_root = build_manifest_root(post_manifest["domain"])
            domain_roots = {"domain": post_root}

            ledger_path = out_dir / "LEDGER.jsonl"
            ledger = Ledger(ledger_path)
            ledger.append(_ledger_record(run_id=run_id, timestamp=timestamp, pre_manifest=pre_manifest, post_manifest=post_manifest, domain_roots=domain_roots))

            jobspec_bytes = canonical_json_bytes(_minimal_jobspec(run_id))
            jobspec_hash = _sha256_hex(jobspec_bytes)
            ledger_hash = _sha256_hex(ledger_path.read_bytes())

            proof_schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"
            validator = RestorationProofValidator(proof_schema_path)
            proof = validator.generate_proof(
                run_id=run_id,
                catalytic_domains=["domain"],
                pre_state=pre_manifest,
                post_state=post_manifest,
                timestamp=timestamp,
                referenced_artifacts={
                    "ledger_hash": ledger_hash,
                    "jobspec_hash": jobspec_hash,
                    "validator_id": {"validator_semver": "0.1.0", "validator_build_id": "test"},
                },
            )

            proof_bytes = canonical_json_bytes(proof)
            roots_bytes = canonical_json_bytes(domain_roots)
            return proof_bytes, roots_bytes

        proof1, roots1 = generate_once(test_root / "out1")
        proof2, roots2 = generate_once(test_root / "out2")
        assert proof1 == proof2
        assert roots1 == roots2
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_tamper_detection_hash_mismatch() -> None:
    # Create test_root inside project
    test_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "pytest_tmp" / f"proof_wiring_{uuid.uuid4().hex[:8]}"
    test_root.mkdir(parents=True, exist_ok=True)

    try:
        test_writer = _make_test_writer(test_root)

        domain = test_root / "domain"
        domain.mkdir(parents=True)
        target = domain / "file.txt"
        target.write_bytes(b"hello")

        cas = CatalyticStore(test_root / "CAS", writer=test_writer)
        pre_manifest = {"domain": compute_domain_manifest(domain, cas=cas)}
        # Tamper 1 byte
        target.write_bytes(b"jello")
        post_manifest = {"domain": compute_domain_manifest(domain, cas=cas)}

        proof_schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"
        validator = RestorationProofValidator(proof_schema_path)
        proof = validator.generate_proof(
            run_id="run_tamper",
            catalytic_domains=["domain"],
            pre_state=pre_manifest,
            post_state=post_manifest,
            timestamp="CATALYTIC-DPT-02_CONFIG",
        )

        assert proof["restoration_result"]["verified"] is False
        assert proof["restoration_result"]["condition"] == "RESTORATION_FAILED_HASH_MISMATCH"
        assert proof["restoration_result"]["mismatches"][0]["path"] == "file.txt"
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_path_normalization_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        normalize_relpath("../escape")
