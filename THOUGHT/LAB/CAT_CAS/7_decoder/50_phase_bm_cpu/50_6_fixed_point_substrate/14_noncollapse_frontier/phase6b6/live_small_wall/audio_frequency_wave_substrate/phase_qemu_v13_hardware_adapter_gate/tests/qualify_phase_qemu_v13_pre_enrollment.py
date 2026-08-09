#!/usr/bin/env python3
"""Fail-closed qualifier for V13 hardware-free pre-enrollment preparation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "pre_enrollment_campaign.py"
REFERENCE = ROOT / "tests" / "phase_qemu_v13_pre_enrollment_separate_reference.py"
CONTRACT = ROOT / "PHASE_QEMU_V13_PRE_ENROLLMENT_CAMPAIGN_CONTRACT.md"
FINDINGS = ROOT / "PHASE_QEMU_V13_PRE_ENROLLMENT_CAMPAIGN_FINDINGS.md"
PRODUCTION_SEAL = ROOT / "evidence" / "PHASE_QEMU_V13_PRE_ENROLLMENT_CAMPAIGN.json"
REFERENCE_SEAL = ROOT / "evidence" / "PHASE_QEMU_V13_PRE_ENROLLMENT_CAMPAIGN_SEPARATE_REFERENCE.json"

EXPECTED_HASHES = {
    PRODUCTION: "1e96207c88cdb660f570956a9ea5143239520bad7a7297094f8aba0c3c959072",
    REFERENCE: "0f721520996598bd290113af65277d4d6dd39e2c06839c185a03db80c053698c",
    CONTRACT: "96054a957222ed2e8e10f7acc544edd905f6ef5910fe0b27973d928fe5a13473",
    FINDINGS: "f73fc9e44c4c4cfc31cced0b4b4dedc91ff502e6578d9fea443ee08225fe3357",
}

AUTHORITY_GATE = "EXPLICIT_USER_AUTHORIZATION_REQUIRED_BEFORE_DEVICE_ENROLLMENT_CONNECTION_OR_CAPTURE"
CLASSIFICATION = "PRE_ENROLLMENT_SOFTWARE_CONFORMANCE_PREPARATION_OUTSIDE_PHYSICAL_EVIDENCE"
PASS_LINE = (
    "PASS_STRICT_SCOPE V13_PRE_ENROLLMENT_CAMPAIGN "
    "SOFTWARE=ENCODED_CBOR_COSE_AND_SEPARATE_REFERENCE "
    "PHYSICAL=NOT_EXECUTED PROMOTION=FALSE M257=INTACT"
)

PRODUCTION_CHECKS = {
    "deterministic_cbor_roundtrip",
    "nonshortest_integer_rejected",
    "indefinite_item_rejected",
    "unsorted_map_rejected",
    "duplicate_map_key_rejected",
    "trailing_bytes_rejected",
    "enrollment_signature_verified",
    "offline_enrollment_cannot_authorize_production",
    "fixture_domain_excluded",
    "preregistration_signature_verified",
    "preregistration_locked_before_data",
    "post_lock_plan_mutation_changes_hash",
    "two_signed_manifests_verified",
    "analysis_consumes_decoded_verified_raw_bytes",
    "bad_manifest_signature_rejected",
    "wrong_signature_domain_rejected",
    "wrong_challenge_rejected",
    "missing_payload_block_rejected",
    "altered_payload_bytes_rejected",
    "sample_count_mismatch_rejected",
    "replay_rejected",
    "role_keys_are_distinct",
    "manifest_inventory_complete",
    "primary_synthetic_endpoints_pass_preregistered_rules",
    "positive_offset_control_fails_preregistered_rules",
    "report_signed_by_distinct_adjudicator",
    "no_authenticated_physical_sample",
    "no_campaign_statistical_certificate",
    "no_physical_or_architecture_promotion",
    "m257_intact",
}

REFERENCE_CHECKS = {
    "four_distinct_fixture_keys",
    "fixture_enrollment_not_authorized",
    "plan_locked_no_interim_looks",
    "two_manifest_signatures_created",
    "analysis_uses_decoded_manifest_bound_bytes",
    "altered_raw_block_rejected",
    "sample_count_mismatch_rejected",
    "invalid_manifest_signature_rejected",
    "manifest_inventory_complete",
    "program_a_passes",
    "program_b_passes",
    "positive_offset_control_fails",
    "no_physical_evidence",
    "no_campaign_certificate",
    "no_architecture_promotion",
    "m257_intact",
}


def fail(message: str) -> None:
    raise SystemExit("FAIL_STRICT_SCOPE: " + message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_audit() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or path.is_symlink():
            fail(f"missing or symlinked dependency: {path}")
        actual = sha(path)
        if actual != expected:
            fail(f"dependency hash drift: {path.name} {actual}")

    production_text = PRODUCTION.read_text(encoding="utf-8")
    production_tree = ast.parse(production_text)
    reference_text = REFERENCE.read_text(encoding="utf-8")
    reference_tree = ast.parse(reference_text)
    allowed_roots = {
        "__future__", "hashlib", "json", "math", "struct", "pathlib", "typing",
        "cryptography", "scipy",
    }
    for label, tree in (("production", production_tree), ("reference", reference_tree)):
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        unexpected = imports - allowed_roots
        if unexpected:
            fail(f"{label} unexpected imports: {sorted(unexpected)}")
    source_dataflow_anchors = {
        "decoded_samples = verify_manifest(": production_text,
        "endpoint_a = analyze_endpoint(decoded_sample_sets[0]": production_text,
        "endpoint_b = analyze_endpoint(decoded_sample_sets[1]": production_text,
        "decoded_samples.append(decode_bound_samples(manifest, [block]))": reference_text,
        "endpoint_a = analyze(decoded_samples[0]": reference_text,
        "endpoint_b = analyze(decoded_samples[1]": reference_text,
    }
    for anchor, text in source_dataflow_anchors.items():
        if anchor not in text:
            fail(f"verified-byte dataflow anchor missing: {anchor}")

    combined_docs = CONTRACT.read_text(encoding="utf-8") + FINDINGS.read_text(encoding="utf-8")
    required = {
        CLASSIFICATION,
        AUTHORITY_GATE,
        "no EAT is issued or parsed here",
        "no TLS or TPM operation occurred",
        "no physical evidence exists",
        "This is not M272",
        "M257 remains intact",
        "QEMU is neither changed nor executed",
    }
    for anchor in required:
        if anchor not in combined_docs:
            fail(f"document scope anchor missing: {anchor}")


def execute(path: Path) -> tuple[bytes, dict[str, Any]]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-B", str(path)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        fail(f"{path.name} exit {process.returncode}: {process.stderr.decode(errors='replace')}")
    if process.stderr:
        fail(f"{path.name} wrote stderr")
    try:
        record = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        fail(f"{path.name} non-JSON stdout: {exc}")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if canonical != process.stdout:
        fail(f"{path.name} stdout not canonical")
    return process.stdout, record


def semantic_audit(production: dict[str, Any], reference: dict[str, Any]) -> None:
    if production.get("schema") != "phase_qemu.v13.pre_enrollment_campaign.v1":
        fail("production schema")
    if production.get("status") != "PASS_HARDWARE_FREE_PRE_ENROLLMENT_SOFTWARE_CONFORMANCE":
        fail("production status")
    if production.get("classification") != CLASSIFICATION:
        fail("production classification")
    if production.get("authority_gate") != AUTHORITY_GATE or reference.get("authority_gate") != AUTHORITY_GATE:
        fail("authority gate parity")
    if production.get("source_sha256") != EXPECTED_HASHES[PRODUCTION]:
        fail("embedded production source hash")
    if reference.get("source_sha256") != EXPECTED_HASHES[REFERENCE]:
        fail("embedded reference source hash")
    if set(production.get("checks", {})) != PRODUCTION_CHECKS or not all(production["checks"].values()):
        fail("production check set")
    if set(reference.get("checks", {})) != REFERENCE_CHECKS or not all(reference["checks"].values()):
        fail("reference check set")
    if production.get("encoded_objects") != reference.get("encoded_objects"):
        fail("encoded-object separate-reference parity")
    if production.get("analysis") != reference.get("analysis"):
        fail("statistical separate-reference parity")

    analysis = production["analysis"]
    if not analysis["program_a"]["endpoint_pass"] or not analysis["program_b"]["endpoint_pass"]:
        fail("primary synthetic control")
    if analysis["positive_offset_control"]["endpoint_pass"]:
        fail("positive-offset negative control")
    if analysis["family_size"] != 4 or analysis["family_alpha"] != 0.05:
        fail("preregistered family")
    if analysis["physical_interpretation_allowed"] is not False:
        fail("physical interpretation leakage")

    architecture = production["architecture_scope"]
    exact_architecture = {
        "existing_common_phase_qemu_v13_backend_targeted": True,
        "qemu_device_modified_by_this_preflight": False,
        "qemu_device_executed_by_this_preflight": False,
        "live_backend_connected": False,
        "device_enrolled": False,
        "physical_capture_executed": False,
        "authenticated_physical_sample_published": False,
        "campaign_statistical_certificate_published": False,
        "physical_restoration_or_reuse_claimed": False,
        "eligible_for_m272_promotion": False,
        "terminal": False,
    }
    if architecture != exact_architecture:
        fail("production architecture scope")
    if any(reference["nonclaims"].values()):
        fail("reference nonclaim promoted")
    if production["fixture_identity"]["production_key_or_certificate_present"] is not False:
        fail("production credential leakage")
    resources = production["resource_accounting"]
    if resources["raw_synthetic_payload_bytes"] != 96 or resources["signature_create_operations"] != 5:
        fail("production resource counts")
    if resources["signature_verify_operations_primary"] != 5 or resources["rematerialized_physical_samples"] != 0:
        fail("production verification/resources")
    if reference["resource_scope"]["production_runtime_or_peak_resource_parity_claimed"] is not False:
        fail("reference resource overclaim")
    if production["strongest_honest_comparators"]["resource_advantage_established"] is not False:
        fail("resource advantage overclaim")


def seal(path: Path, expected: bytes, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    if not path.is_file() or path.is_symlink():
        fail(f"seal missing: {path.name}")
    if path.read_bytes() != expected:
        fail(f"seal mismatch: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-seals", action="store_true")
    args = parser.parse_args()
    static_audit()
    production_first, production_record = execute(PRODUCTION)
    production_second, production_second_record = execute(PRODUCTION)
    reference_first, reference_record = execute(REFERENCE)
    reference_second, reference_second_record = execute(REFERENCE)
    if production_first != production_second or production_record != production_second_record:
        fail("production nondeterminism")
    if reference_first != reference_second or reference_record != reference_second_record:
        fail("reference nondeterminism")
    semantic_audit(production_record, reference_record)
    seal(PRODUCTION_SEAL, production_first, args.write_seals)
    seal(REFERENCE_SEAL, reference_first, args.write_seals)
    print(PASS_LINE)


if __name__ == "__main__":
    main()
