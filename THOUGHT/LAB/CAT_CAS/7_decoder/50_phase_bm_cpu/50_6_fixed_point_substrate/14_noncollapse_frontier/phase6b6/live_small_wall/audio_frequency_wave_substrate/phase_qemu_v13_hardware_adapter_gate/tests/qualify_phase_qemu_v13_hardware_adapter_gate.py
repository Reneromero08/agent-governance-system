#!/usr/bin/env python3
"""Strict M271 compiled Phase-QEMU V13 hardware-gate qualifier.

All executable inputs and disk-backed scratch are explicit.  The only optional
repository writes are the two canonical evidence seals under ``evidence/``.
Wall-time observations are never part of the deterministic qtest seal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence


PASS_LINE = (
    "PASS_STRICT_SCOPE M271_PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE "
    "SCIENCE=SEPARATE_REFERENCE_PARITY RESTORATION=NO_RESTORATION_CLAIM "
    "SCOPE=COMPILED_QEMU_HARDWARE_ABSENCE_FIXED_SELECTOR_GATE "
    "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
)

EXPECTED_HASHES = {
    "device_c": "88889411164e9e49c4029585b90703b306a5ac12d9acf6c1a5479790ddc89eeb",
    "installer": "edd6c7ace7debbaa89bf5ee6085fb7fd9dd9c9500e3bf26fb7d41ae3a85a8599",
    "runner": "ddb2cb169691b829130ee03a446fc14df7189aa4f82eb68629164ca43418c0ad",
    "v12_runner": "5bdc38c369f89f8afe61a574dbafba24e012cb64c18912e5e282ce5a0d151766",
    "reference": "6812712066f29c95de641f880607ffd2fa3be0392ed0626038436dfd86950a87",
    "contract": "9ecaf9c12732cb213969bbc6501dc558112fa4cd44df6174ce026853c59e7962",
    "findings": "60c531768a76b903e16b062ce82cfcacf36482a789dedf3e210a822daf4c85c7",
    "build_receipt": "a31c771972a13992e552d36a28864a210c9a2604bafa421913b28a17ee8205cc",
    "v0_c": "b79ec06f870b611142f5df5c97db2f8e34027458da5acc933f90b694b2055764",
    "v1_c": "8b991d8961a6e108d1a4aa7498172564b017c6e622cb8192c6fa15c33638e362",
    "v11_c": "84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771",
    "v12_c": "5fe4f99e9bf2ff78774e75149c532293adb03c01c035fcac3d05e2fe74b6153f",
    "qemu_binary": "6d70c461f860c83be4436000f7babaa17ff014d9de0fdf8b60a27008d6891050",
    "qtest_payload": "8449c53dec5c28b0aeeef4290c5a94591c6b3b714946799fabb8703e2d9df6ff",
    "reference_stdout": "4dec177f55ba8bf5bfe7ccf814aaa672bccbd739f976a3629c72c69519490d83",
    "reference_payload": "f2bfbd7b0a4b055fcde1fba59ac3472157102bac8729ae5f23f192def4d4e2f9",
}

AUTHORITY = {
    "claim": (
        "COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_"
        "GATE_REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_"
        "AND_TEST_FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_"
        "CAMPAIGN_STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_"
        "COMPLETED_DISPATCHED_ATTEMPT_TERMINAL_ACK_THEN_SPENT"
    ),
    "ceiling": (
        "HARDWARE_ABSENT_PROTOCOL_CONFORMANCE_ONLY_NO_AUTHENTICATED_LIVE_DEVICE_"
        "SESSION_NO_PHYSICAL_SAMPLE_NO_CAMPAIGN_STATISTICAL_CERTIFICATE_NO_"
        "CUSTODY_RETURN_RESTORATION_REUSE_ADVANTAGE_OR_M257_ESCAPE"
    ),
    "restoration_classification": "NO_RESTORATION_CLAIM",
    "scope": (
        "COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_"
        "ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY"
    ),
    "disposition": (
        "V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_"
        "CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_"
        "FIXTURE_DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_"
        "OUTPUT_DIRECT_EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_"
        "REMAINS_INTACT"
    ),
    "successor": (
        "USER_AUTHORIZED_PINNED_DEVICE_ENROLLMENT_FOLLOWED_BY_A_PREREGISTERED_"
        "BLINDED_DUAL_RAIL_DISPERSIVE_CAPTURE_CAMPAIGN_WITH_DEVICE_SIGNED_RAW_"
        "MANIFESTS_INDEPENDENT_MEASUREMENT_AND_FAMILYWISE_STATISTICAL_VALIDATION_"
        "BEHIND_THE_COMMON_PHASE_QEMU_BACKEND"
    ),
}

M257 = (
    "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_"
    "COUNTED_AS_A_PHASE_RESOURCE"
)
QEMU_VERSION = "10.2.4"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_VECTOR_NAMES = [
    "accepted_metadata_fixture_domain",
    "unenrolled",
    "unattested",
    "stale",
    "replayed",
    "downgraded",
    "type_mismatch",
]
REFERENCE_CASES = [
    "absent", "unenrolled", "unattested", "stale_nonce", "replay",
    "downgraded_protocol", "downgraded_firmware",
    "fixture_credential_in_production", "bad_signature", "bad_hash",
    "bad_sequence", "bad_calibration", "bad_resource_schema", "timeout",
    "reset", "migration", "disconnect",
]


class Failure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def die(message: str) -> NoReturn:
    raise Failure(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(path: Path, expected: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular {label}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch: {actual}")


def filesystem_type(path: Path) -> str:
    resolved = path.resolve()
    best_parts = -1
    best_type = "UNKNOWN"
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if "-" not in fields:
            continue
        separator = fields.index("-")
        mount = Path(fields[4].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        if len(mount.parts) > best_parts:
            best_parts = len(mount.parts)
            best_type = fields[separator + 1]
    return best_type


def package_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    lane = package.parent
    evidence = package / "evidence"
    return {
        "package": package,
        "device_c": package / "qemu/phase-qemu-v13.c",
        "installer": package / "apply_to_qemu.py",
        "runner": package / "tests/run_phase_qemu_v13_hardware_gate_qtest.py",
        "v12_runner": lane / (
            "phase_qemu_v12_authenticated_adapter_stub/tests/"
            "run_phase_qemu_v12_authenticated_adapter_qtest.py"
        ),
        "reference": package / "tests/phase_qemu_v13_hardware_gate_separate_reference.py",
        "contract": package / "PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE_CONTRACT.md",
        "findings": package / "PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE_FINDINGS.md",
        "build_receipt": evidence / "PHASE_QEMU_V13_BUILD_RECEIPT.json",
        "v12_c": lane / "phase_qemu_v12_authenticated_adapter_stub/qemu/phase-qemu-v12.c",
        "qtest_seal": evidence / "PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE_QTEST.json",
        "reference_seal": evidence / (
            "PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE_SEPARATE_REFERENCE.json"
        ),
    }


def run_command(
    command: list[str], cwd: Path, timeout: int, *, reduced: bool = False,
) -> bytes:
    actual = command
    if reduced:
        actual = ["ionice", "-c2", "-n7", "nice", "-n10", *command]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        actual, cwd=cwd, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False,
    )
    require(result.returncode == 0, f"command failed {actual}: {result.stderr!r}")
    require(result.stderr == b"", f"command stderr not empty {actual}: {result.stderr!r}")
    return result.stdout


def read_json(value: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Failure(f"{label} is not one JSON object: {error}") from error
    require(isinstance(document, dict), f"{label} root is not object")
    return document


def extract_registers(text: str, enum_name: str) -> list[tuple[str, int]]:
    match = re.search(
        rf"enum\s+{re.escape(enum_name)}\s*\{{(?P<body>.*?)\}};", text, re.S
    )
    require(match is not None, f"missing register enum {enum_name}")
    pairs = [
        (name, int(offset, 16))
        for name, offset in re.findall(
            r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)", match.group("body")
        )
    ]
    require(len(pairs) == len({name for name, _ in pairs}), "duplicate register name")
    return pairs


def validate_package(paths: dict[str, Path]) -> dict[str, Any]:
    for name in (
        "device_c", "installer", "runner", "v12_runner", "reference",
        "contract", "findings", "build_receipt", "v12_c",
    ):
        exact_file(paths[name], EXPECTED_HASHES[name], name)

    for script in (paths["installer"], paths["runner"], paths["reference"], Path(__file__)):
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    reference_tree = ast.parse(paths["reference"].read_text(encoding="utf-8"))
    allowed = {"__future__", "hashlib", "json", "pathlib", "typing"}
    imported: set[str] = set()
    for node in ast.walk(reference_tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    require(imported <= allowed, f"reference imports non-independent module: {imported}")

    for document in (paths["contract"], paths["findings"]):
        text = document.read_text(encoding="utf-8")
        for name, token in AUTHORITY.items():
            expected = text.count(token)
            require(expected == 1 or (name == "restoration_classification" and expected >= 1),
                    f"authority token count in {document.name}: {name}={expected}")
    require("M257 remains intact" in paths["contract"].read_text(encoding="utf-8"),
            "M257 contract disposition missing")
    require(M257 in paths["findings"].read_text(encoding="utf-8"),
            "M257 findings guardrail missing")
    findings = paths["findings"].read_text(encoding="utf-8")
    for phrase in (
        "V2 through V10", "compiled V13 common PCI/backend boundary",
        "seven-choice selector, not an evidence parser", "No volatile artifact hash",
        "symbolic state `SHAM`, no ACK; no compiled live-disconnect input exists",
    ):
        require(phrase in findings, f"architecture/scope anchor missing: {phrase}")
    contract = paths["contract"].read_text(encoding="utf-8")
    require("symbolic disconnect enters\n`SHAM` without ACK" in contract,
            "contract disconnect lifecycle scope missing")

    device = paths["device_c"].read_text(encoding="utf-8")
    v12 = paths["v12_c"].read_text(encoding="utf-8")
    old = extract_registers(v12, "PhaseV12Register")
    new = extract_registers(device, "PhaseV13Register")
    require(len(old) == 78 and new[:78] == old, "V12 register prefix changed")
    require(new[78][1] == 0x280, "V13 extension does not begin at 0x280")
    require("ideal_execute" not in device and "ideal_verify" not in device,
            "ideal success fallback present in V13")
    for forbidden in ("RETURN_EXACT_FORMAL", "RETURN_APPROX_MODEL", "RETURN_STATISTICAL_ONLY"):
        require(f"s->return_class = {forbidden}" not in device,
                f"forbidden return class assigned: {forbidden}")
    for anchor in (
        "device_class->unrealize = phase_qemu_v13_unrealize",
        ".pre_save = phase_qemu_v13_pre_save",
        ".post_load = phase_qemu_v13_post_load", "ERR_HARDWARE_ABSENT",
        "PHASE_V13_BACKEND_OFFLINE_FIXTURE", "RETURN_FAILED",
    ):
        require(anchor in device, f"C lifecycle anchor missing: {anchor}")

    receipt = json.loads(paths["build_receipt"].read_text(encoding="utf-8"))
    require(receipt["schema"] ==
            "PHASE_QEMU_V13_HARDWARE_ADAPTER_GATE_COMPILED_BUILD_RECEIPT_V1",
            "build receipt schema mismatch")
    require(receipt["qemu"]["binary_sha256"] == EXPECTED_HASHES["qemu_binary"],
            "receipt binary mismatch")
    runtime = receipt["runtime_qualification"]
    require(runtime["qtest_deterministic_payload_sha256"] ==
            EXPECTED_HASHES["qtest_payload"] and runtime["qtest_matching_replay_count"] == 2,
            "receipt qtest mismatch")
    require(runtime["separate_reference_stdout_sha256"] ==
            EXPECTED_HASHES["reference_stdout"] and
            runtime["separate_reference_payload_sha256"] ==
            EXPECTED_HASHES["reference_payload"] and
            runtime["separate_reference_matching_replay_count"] == 2,
            "receipt reference mismatch")
    architecture = receipt["architecture"]
    for name in (
        "qemu_device_implemented", "common_guest_visible_contract_compiled",
        "common_guest_visible_contract_exercised", "reintegration_gate_passed",
    ):
        require(architecture[name] is True, f"receipt architecture false: {name}")
    for name in (
        "standalone_twin_qualifies", "fixed_selectors_are_parsers_or_cryptographic_verifiers",
        "external_hardware_connected", "authenticated_physical_sample_published",
        "campaign_statistical_certificate_published", "physical_carrier_custody_observed",
        "physical_restoration_or_reuse_established", "resource_or_query_advantage_established",
        "m257_escape_established",
    ):
        require(architecture[name] is False, f"receipt nonclaim widened: {name}")
    return receipt


def validate_qemu_source(paths: dict[str, Path], source: Path) -> None:
    source = source.resolve()
    require(source.is_dir() and not source.is_symlink(), "QEMU source invalid")
    require((source / "VERSION").read_text(encoding="utf-8").strip() == QEMU_VERSION,
            "QEMU version mismatch")
    installed = {
        "v0_c": source / "hw/misc/phase-qemu-v0.c",
        "v1_c": source / "hw/misc/phase-qemu-v1.c",
        "v11_c": source / "hw/misc/phase-qemu-v11.c",
        "v12_c": source / "hw/misc/phase-qemu-v12.c",
        "device_c": source / "hw/misc/phase-qemu-v13.c",
    }
    for name, path in installed.items():
        exact_file(path, EXPECTED_HASHES[name], f"installed {name}")
    require(installed["device_c"].read_bytes() == paths["device_c"].read_bytes(),
            "installed/package V13 differ")
    kconfig = (source / "hw/misc/Kconfig").read_text(encoding="utf-8")
    meson = (source / "hw/misc/meson.build").read_text(encoding="utf-8")
    for suffix in ("0", "1", "11", "12", "13"):
        require(kconfig.count(f"config PHASE_QEMU_V{suffix}\n") == 1,
                f"Kconfig V{suffix} count")
        require(meson.count(f"'CONFIG_PHASE_QEMU_V{suffix}'") == 1,
                f"Meson V{suffix} count")


def validate_binary(binary: Path) -> Path:
    binary = binary.resolve()
    exact_file(binary, EXPECTED_HASHES["qemu_binary"], "QEMU binary")
    output = subprocess.run(
        [str(binary), "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    require(output.returncode == 0 and output.stderr == b"", "QEMU version command failed")
    require(output.stdout.splitlines()[0] == b"QEMU emulator version 10.2.4",
            "QEMU binary version mismatch")
    return binary


def prepare_runs(scratch: Path) -> dict[str, Path]:
    scratch = scratch.resolve()
    require(scratch.is_dir() and not scratch.is_symlink(), "scratch root invalid")
    require(filesystem_type(scratch) not in {"tmpfs", "ramfs"}, "RAM-backed scratch forbidden")
    root = scratch / "m271-v13-qualifier-runs"
    require(not root.exists() and not root.is_symlink(), "qualifier run root already exists")
    root.mkdir(mode=0o700)
    result: dict[str, Path] = {}
    for name in ("installer", "qtest-1", "qtest-2", "reference-1", "reference-2"):
        path = root / name
        path.mkdir(mode=0o700)
        result[name] = path
    return result


def run_installer_check(paths: dict[str, Path], source: Path, runs: dict[str, Path]) -> None:
    output = run_command(
        [sys.executable, "-B", str(paths["installer"]), "--check", str(source.resolve())],
        runs["installer"], 120,
    ).decode("utf-8")
    required = {
        "mode": "check", "qemu_version": QEMU_VERSION,
        "integration_state": "installed", "target_present": "1", "target_current": "1",
        "kconfig_entries": "1", "meson_entries": "1",
        "qemu_device_source_integrated": "1", "qemu_device_implemented": "0",
        "common_guest_visible_contract_source_integrated": "1",
        "common_guest_visible_contract_compiled": "0", "reintegration_gate_passed": "0",
        "compiled_qtest_qualification_required": "1", "live_hardware_available": "0",
        "physical_output_published": "0", "campaign_statistical_certificate_published": "0",
    }
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    for key, value in required.items():
        require(fields.get(key) == value, f"installer check mismatch: {key}")


def audit_qtest(evidence: dict[str, Any]) -> None:
    expected_keys = {
        "authority", "build_identity", "checks", "classification",
        "clean_process_teardown", "compiled_device_identity",
        "direct_equal_access_comparator", "guest_abi_controls", "migration",
        "offline_standard_vectors", "production_backend", "realization_controls",
        "request_controls", "reset_after_activity", "resource_accounting", "runtime",
        "schema", "unsupported_surfaces",
    }
    require(set(evidence) == expected_keys, "qtest top-level key mismatch")
    require(evidence["schema"] == "M271_PHASE_QEMU_V13_HARDWARE_GATE_QTEST_EVIDENCE_V1",
            "qtest schema mismatch")
    require(evidence["authority"] == {
        "claim": AUTHORITY["claim"], "scope": AUTHORITY["scope"],
        "disposition": AUTHORITY["disposition"],
    }, "qtest authority mismatch")
    checks = evidence["checks"]
    require(len(checks) == 12 and all(value is True for value in checks.values()),
            "qtest check set failed")
    classification = evidence["classification"]
    require(classification["restoration_classification"] == "NO_RESTORATION_CLAIM"
            and classification["physical_evidence_class"] == "NONE"
            and classification["return_classes_observed"] == ["NONE", "RETURN_FAILED"],
            "qtest classification mismatch")
    for name in (
        "M257_escape", "approx_model_return_observed", "architecture_promotion",
        "exact_formal_return_observed", "fixed_selectors_are_cryptographic_verifiers",
        "fixed_selectors_are_evidence_parsers", "mechanism_promotion",
        "offline_fixture_is_production_trust", "production_hardware_connected",
        "standalone_python_twin_qualifies", "statistical_only_return_observed",
    ):
        require(classification[name] is False, f"qtest classification widened: {name}")
    require(classification["physical_sample_count"] == 0
            and classification["campaign_statistical_certificate_count"] == 0,
            "qtest physical artifact count nonzero")
    production = evidence["production_backend"]
    require(production == {
        "backend": "HARDWARE", "backend_id": 0x0D80, "dispatches": 0,
        "lease_error": 38, "physical_hardware_connected": False,
        "pre_dispatch_rejection": True, "receipt_created": False,
        "response_created": False,
    }, "production absence preflight mismatch")

    vectors = evidence["offline_standard_vectors"]
    require(len(vectors) == 7 and [item["vector_id"] for item in vectors] == list(range(7))
            and [item["vector_name"] for item in vectors] == EXPECTED_VECTOR_NAMES,
            "fixed vector set mismatch")
    for item in vectors:
        require(item["trust_domain"] == "TEST_FIXTURE"
                and item["typed_origin"] == "OFFLINE_STANDARD_VECTOR"
                and item["measurement_class"] == "PROTOCOL_CONFORMANCE_ONLY"
                and item["return_class"] == "RETURN_FAILED"
                and item["terminal_state"] == "SPENT"
                and item["ack_required"] is True
                and item["physical_output"] is False
                and item["physical_sample"] is False
                and item["campaign_statistical_certificate"] is False
                and item["begin_reuse_authorized"] is False
                and item["receipt_independently_recomputed"] is True
                and item["resource_digest_independently_recomputed"] is True
                and item["direct_protocol_comparator_matches"] is True,
                f"fixed selector mismatch: {item['vector_id']}")
    require(evidence["request_controls"] == {
        "generation": {"control": "generation", "dispatches": 0, "error": 4,
                       "pre_dispatch_rejection": True, "receipt_created": False},
        "lease_expiry": {"control": "lease_expiry", "dispatches": 0, "error": 46,
                         "pre_dispatch_rejection": True, "receipt_created": False},
        "owner_tag": {"control": "owner_tag", "dispatches": 0, "error": 3,
                      "pre_dispatch_rejection": True, "receipt_created": False},
    }, "request-control mismatch")
    migration = evidence["migration"]
    require(migration["transport"] == "REAL_QMP_UNIX_LIVE_MIGRATION"
            and migration["first_hop_status"] == "completed"
            and migration["second_hop_status"] == "completed"
            and migration["source_pre_save_sanitized_to_sham"] is True
            and migration["destination_post_load_sanitized_to_sham"] is True
            and migration["second_hop_preserved_sham"] is True
            and migration["post_migration_reset_preserved_sham"] is True
            and migration["begin_reuse_authorized"] is False
            and migration["lease_reconstructed"] is False
            and migration["receipt_reconstructed"] is False,
            "migration SHAM mismatch")
    reset = evidence["reset_after_activity"]
    require(reset["reset_latched_sham"] is True
            and reset["second_reset_preserved_sham"] is True
            and reset["begin_reuse_authorized"] is False,
            "reset SHAM mismatch")
    teardown = evidence["clean_process_teardown"]
    require(teardown["process_exit_code"] == 0
            and teardown["qmp_quit_is_clean_process_teardown"] is True
            and teardown["static_compiled_source_audit_required_for_sanitizer_callback"] is True,
            "teardown scope mismatch")
    resources = evidence["resource_accounting"]
    require(resources["resource_schema"] == 0x00030001
            and resources["physical_sample_count"] == 0
            and resources["campaign_statistical_certificate_count"] == 0
            and resources["physical_resource_manifest"] == "NOT_AVAILABLE"
            and resources["total_resource_comparison"] == "UNDETERMINED"
            and resources["resource_advantage_claim"] is False
            and len(resources["unknown_physical_registers"]) == 12,
            "qtest resource scope mismatch")
    comparator = evidence["direct_equal_access_comparator"]
    require(comparator["all_expected_reasons_and_typed_classes_matched"] is True
            and comparator["same_symbolic_vector_ids"] == list(range(7))
            and comparator["evidence_parsing_exercised"] is False
            and comparator["cryptographic_verification_claim"] is False
            and comparator["physical_output_used"] is False
            and comparator["unique_phase_resource"] is False
            and comparator["M257_escape"] is False,
            "qtest comparator/M257 mismatch")


def run_qtest_twice(
    paths: dict[str, Path], binary: Path, runs: dict[str, Path],
) -> tuple[bytes, dict[str, Any]]:
    payloads: list[bytes] = []
    evidence_docs: list[dict[str, Any]] = []
    for index in (1, 2):
        runtime = runs[f"qtest-{index}"] / "runtime"
        runtime.mkdir(mode=0o700)
        stdout = run_command(
            [sys.executable, "-B", str(paths["runner"]), "--qemu-binary", str(binary),
             "--scratch-dir", str(runtime)],
            runs[f"qtest-{index}"], 300, reduced=True,
        )
        outer = read_json(stdout, f"qtest run {index}")
        require(set(outer) == {"deterministic_evidence", "wall_times_ns"},
                "qtest outer envelope mismatch")
        require(isinstance(outer["wall_times_ns"], dict) and outer["wall_times_ns"],
                "qtest wall-time segregation missing")
        evidence = outer["deterministic_evidence"]
        require(isinstance(evidence, dict), "qtest deterministic evidence missing")
        payloads.append(canonical_bytes(evidence) + b"\n")
        evidence_docs.append(evidence)
    require(payloads[0] == payloads[1], "qtest deterministic payload differs")
    require(sha256_bytes(payloads[0]) == EXPECTED_HASHES["qtest_payload"],
            "qtest payload hash mismatch")
    audit_qtest(evidence_docs[0])
    return payloads[0], evidence_docs[0]


def audit_reference(stdout: bytes) -> None:
    document = read_json(stdout, "separate reference")
    require(canonical_bytes(document) + b"\n" == stdout, "reference JSON not canonical")
    require(set(document) == {
        "schema", "source_sha256", "payload_sha256", "payload",
        "two_self_runs_byte_identical",
    }, "reference outer key mismatch")
    require(document["schema"] == "M271_V13_HARDWARE_ADAPTER_GATE_SEPARATE_REFERENCE_V1"
            and document["source_sha256"] == EXPECTED_HASHES["reference"]
            and document["payload_sha256"] == EXPECTED_HASHES["reference_payload"]
            and document["two_self_runs_byte_identical"] is True,
            "reference identity mismatch")
    payload = document["payload"]
    require(sha256_bytes(canonical_bytes(payload)) == EXPECTED_HASHES["reference_payload"],
            "reference payload self-hash mismatch")
    require(payload["authority"] == AUTHORITY, "reference authority mismatch")
    require(payload["all_checks_pass"] is True and payload["check_count"] == 22
            and len(payload["checks"]) == 22
            and len({item["id"] for item in payload["checks"]}) == 22
            and all(item["pass"] is True for item in payload["checks"]),
            "reference checks mismatch")
    classification = payload["classification"]
    require(classification["reference_layer"] ==
            "INDEPENDENT_SYMBOLIC_FIXED_OFFLINE_SELECTOR_REFERENCE"
            and classification["physical_evidence_class"] == "NONE"
            and classification["restoration_classification"] == "NO_RESTORATION_CLAIM",
            "reference classification mismatch")
    for name in (
        "architecture_promotion_claim_from_reference", "custody_claim",
        "hardware_execution_performed", "live_device_session_present",
        "physical_output_present", "physical_return_claim",
        "production_authentication_established", "qemu_execution_performed",
        "reuse_claim", "same_carrier_claim",
    ):
        require(classification[name] is False, f"reference nonclaim widened: {name}")
    table = payload["production_failure_truth_table"]
    require([item["case"] for item in table] == REFERENCE_CASES,
            "reference truth-table order mismatch")
    for item in table:
        receipt = item["symbolic_failure_receipt"]
        require(item["direct_protocol_comparator_match"] is True
                and receipt["physical_output_present"] is False
                and receipt["authenticated_physical_sample_present"] is False
                and receipt["campaign_statistical_certificate_present"] is False
                and receipt["production_trust_established"] is False
                and receipt["begin_reuse_authorized"] is False,
                f"reference case widened: {item['case']}")
        if item["case"] == "timeout":
            require(receipt["dispatched"] is True and receipt["ack_required"] is True
                    and receipt["terminal_state"] == "SPENT"
                    and receipt["destructive_sham"] is False,
                    "reference timeout lifecycle mismatch")
        elif item["case"] in {"reset", "migration", "disconnect"}:
            require(receipt["dispatched"] is True and receipt["ack_required"] is False
                    and receipt["terminal_state"] == "SHAM"
                    and receipt["destructive_sham"] is True,
                    f"reference destructive lifecycle mismatch: {item['case']}")
        else:
            require(receipt["dispatched"] is False and receipt["terminal_state"] == "SPENT",
                    f"reference predispatched lifecycle mismatch: {item['case']}")
    layers = payload["evidence_layers"]
    require(layers["artifacts_are_distinct"] is True
            and layers["neither_artifact_present"] is True
            and layers["per_transaction_artifact"]["class"] ==
            "AUTHENTICATED_PHYSICAL_SAMPLE"
            and layers["per_transaction_artifact"]["present_in_m271"] is False
            and layers["independent_campaign_artifact"]["class"] == "STATISTICAL_ONLY"
            and layers["independent_campaign_artifact"]["present_in_m271"] is False,
            "reference evidence layers mismatch")
    comparator = payload["direct_protocol_comparator"]
    require(comparator["case_count"] == 17
            and comparator["all_truth_table_outcomes_identical"] is True
            and comparator["production_hardware_access"] is False
            and comparator["physical_output_compared"] is False
            and comparator["unique_protocol_advantage"] is False
            and comparator["advantage_claim"] is False,
            "reference comparator mismatch")
    resources = payload["resource_accounting"]
    require(resources["physical_resource_total"] == "UNKNOWN"
            and resources["unknown_value_encoding"] == "NULL_NEVER_NUMERIC_ZERO"
            and resources["resource_advantage_claim"] is False,
            "reference resource scope mismatch")
    for entry in resources["unknown_physical_resources"].values():
        require(entry["status"] == "UNKNOWN" and entry["value"] is None,
                "reference unknown physical resource encoded as known")
    require(payload["m257"] == {
        "escape_established": False, "guardrail": M257, "status": "INTACT",
    }, "reference M257 mismatch")
    nonclaims = payload["nonclaims"]
    require(nonclaims == sorted(set(nonclaims))
            and "NO_AUTHENTICATED_PHYSICAL_SAMPLE_CLAIM" in nonclaims
            and "NO_CAMPAIGN_STATISTICAL_CERTIFICATE_CLAIM" in nonclaims
            and "NO_HARDWARE_EXECUTION_CLAIM" in nonclaims
            and "NO_RESTORATION_CLAIM" in nonclaims
            and "NO_M257_ESCAPE_CLAIM" in nonclaims,
            "reference nonclaim set mismatch")


def run_reference_twice(paths: dict[str, Path], runs: dict[str, Path]) -> bytes:
    outputs = [
        run_command(
            [sys.executable, "-B", str(paths["reference"])],
            runs[f"reference-{index}"], 120,
        ) for index in (1, 2)
    ]
    require(outputs[0] == outputs[1], "reference stdout differs")
    require(sha256_bytes(outputs[0]) == EXPECTED_HASHES["reference_stdout"],
            "reference stdout hash mismatch")
    audit_reference(outputs[0])
    return outputs[0]


def handle_seals(
    paths: dict[str, Path], qtest: bytes, reference: bytes,
    *, write_seals: bool, preseal: bool,
) -> None:
    qseal = paths["qtest_seal"]
    rseal = paths["reference_seal"]
    require(qseal.parent.is_dir() and not qseal.parent.is_symlink(),
            "evidence directory invalid")
    if preseal:
        return
    if write_seals:
        require(not qseal.exists() and not qseal.is_symlink()
                and not rseal.exists() and not rseal.is_symlink(),
                "--write-seals requires both seal paths absent")
        with qseal.open("xb") as handle:
            handle.write(qtest)
        with rseal.open("xb") as handle:
            handle.write(reference)
        return
    missing = [str(path) for path in (qseal, rseal) if not path.is_file()]
    require(not missing, f"missing required seals: {missing}")
    require(qseal.read_bytes() == qtest, "qtest seal byte mismatch")
    require(rseal.read_bytes() == reference, "reference seal byte mismatch")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict M271 V13 qualifier")
    parser.add_argument("--qemu-binary", required=True, type=Path)
    parser.add_argument("--qemu-source", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-seals", action="store_true")
    mode.add_argument("--preseal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        paths = package_paths()
        validate_package(paths)
        validate_qemu_source(paths, args.qemu_source)
        binary = validate_binary(args.qemu_binary)
        runs = prepare_runs(args.scratch_dir)
        run_installer_check(paths, args.qemu_source, runs)
        qtest_payload, _ = run_qtest_twice(paths, binary, runs)
        reference_stdout = run_reference_twice(paths, runs)
        handle_seals(
            paths, qtest_payload, reference_stdout,
            write_seals=args.write_seals, preseal=args.preseal,
        )
    except (Failure, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"FAIL_CLOSED {error}", file=sys.stderr)
        return 1
    print(PASS_LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
