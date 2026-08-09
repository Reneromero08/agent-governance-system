#!/usr/bin/env python3
"""Strict Phase-QEMU V11 package, reference, and qtest qualifier.

Every executable input and scratch root is explicit.  Runtime artifacts are
created only below the caller's disk-backed scratch directory and are retained
for inspection.  Seal writes, when explicitly requested, are limited to the
two frozen evidence paths declared below.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PASS_LINE = (
    "PASS_STRICT_SCOPE M269_PHASE_QEMU_V11_DUAL_RAIL_DEVICE "
    "SCIENCE=SEPARATE_REFERENCE_PARITY "
    "RESTORATION=EXACT_ALGEBRAIC_RESTORATION "
    "ARCHITECTURE=COMPILED_COMMON_DEVICE_BACKEND "
    "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
)

EXPECTED_HASHES = {
    "device_c": "84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771",
    "installer": "0d7aadbf5088cf124544a53c16ae895a3b759255cf01510e9983b7cf85dfe92e",
    "runner": "fa70a83c24cb9bd9ffc78e6cf07ed3c21dc99d342c6302ac1482dd5df48b47dd",
    "reference": "b9791716c6d973c0e119e3666e960efed0f84e16676c74b923e198b4930c1d3a",
    "contract": "2f1eb232a09435176a3cc57ab3d1182226227c700323a777593f24e7a78ee6a3",
    "findings": "405891749a2248397c2b505572ef5550fde309241e57515c2d113be64932fa2b",
    "build_receipt": "7e862a5f4dacbd69573ce3361b27035542b14661b7f45fcd3255cff21b74cab4",
    "gitignore": "6ea1433bfa035fd7b416ac64666d6d6bd3c13a81630d0d89917425b509122d58",
    "v0_c": "b79ec06f870b611142f5df5c97db2f8e34027458da5acc933f90b694b2055764",
    "v1_c": "8b991d8961a6e108d1a4aa7498172564b017c6e622cb8192c6fa15c33638e362",
    "qemu_binary": "02543d7c7df868c19c4fcc040c9f0717d99b4f9f2786c677f4621debe9e7ea40",
    "qtest_payload": "2286987541ebf283c1ed6733b750430fd09f8c978d32dbd9e75a6caf4f9ae49e",
    "reference_stdout": "020dc7235bb43d785553f587be0b43453fb310dc252f79cb17901c4765219b11",
    "reference_claim_payload": "263b791d7898195f94aad9e6f5c7033c94b6a0fc249a2172705333a5502a8d16",
}

QEMU_VERSION = "10.2.4"
QEMU_VERSION_LINE = "QEMU emulator version 10.2.4"
DESCRIPTOR_WORDS = [
    0x50313144,
    0x00010008,
    0x00000002,
    0x00020102,
    0x00020011,
    0x00030021,
    0x00000003,
    0x00010001,
]
EXPECTED_PAIRS = [list(pair) for pair in itertools.product(range(3), repeat=2)]
FAULT_ERRORS = {"1": 23, "2": 23, "3": 23, "4": 8, "5": 24, "6": 11, "7": 30}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

PRODUCTION_AUTHORITY = {
    "claim": (
        "COMMON_PHASE_QEMU_V11_SWAPPABLE_BACKEND_DEVICE_EXECUTES_TWO_LATE_"
        "BOUND_DUAL_RAIL_NUMBER_EIGENSPACE_DISPERSIVE_KICKBACK_QUERIES_ON_"
        "ONE_HIDDEN_CARRIER_REFERENCE_STATE_AND_RELEASES_ATOMIC_RECEIPTS_"
        "ONLY_AFTER_EXACT_COMPLETE_RETURN_WHILE_OPEN_AND_EXTERNAL_BACKENDS_"
        "FAIL_CLOSED"
    ),
    "ceiling": (
        "COMPILED_QEMU_10_2_4_PCI_DEVICE_WITH_EXACT_QOMEGA_IDEAL_BACKEND_"
        "TEST_ONLY_PRIVATE_PROVIDER_AND_ANALYTIC_OPEN_EXTERNAL_STUBS_NO_"
        "PHYSICAL_ORACLE_COHERENT_PORT_CUSTODY_QUERY_SEPARATION_OR_ADVANTAGE"
    ),
    "restoration": "EXACT_ALGEBRAIC_RESTORATION",
    "scope": (
        "EXACT_HIDDEN_96_DIMENSION_QOMEGA_DUAL_RAIL_CARRIER_REFERENCE_AND_"
        "TWO_CLIENT_REFERENCE_RETURN_ON_ONE_RESIDENT_QEMU_ALLOCATION_FOR_"
        "IDEAL_BACKEND_ONLY_WITH_NO_PHYSICAL_SAME_MODE_CUSTODY"
    ),
    "disposition": (
        "COMMON_DEVICE_REINTEGRATION_ESTABLISHES_BACKEND_NEUTRAL_LIFECYCLE_"
        "AND_EXACT_IDEAL_MACHINE_LAW_BUT_TEST_ONLY_PRIVATE_BINDING_AND_"
        "EQUAL_ACCESS_DIRECT_PHASE_SHADOW_PRECLUDE_PHYSICAL_OR_RESOURCE_"
        "PROMOTION"
    ),
    "successor": (
        "AUTHENTICATED_EXTERNAL_DUAL_RAIL_DISPERSIVE_ORACLE_ADAPTER_WITH_"
        "COHERENT_CLIENT_PORT_REFERENCE_PRESERVATION_LATE_BOUND_PRIVATE_"
        "CONTROL_AND_TOTAL_RESOURCE_CERTIFICATION"
    ),
}

REFERENCE_AUTHORITY = {
    "claim": (
        "FINITE_D3_DUAL_RAIL_NUMBER_EIGENSPACE_CHARACTER_KICKBACK_RETURNS_"
        "THE_LOGICAL_CARRIER_REFERENCE_EXACTLY_AND_FACTORS_TWO_DISTINCT_"
        "FRESH_CLIENT_CHOI_BOUNDARIES_FOR_ALL_NINE_RESIDUE_PAIRS_WHILE_"
        "OPEN_NOISY_CONTROLS_FAIL_COMPLETE_RETURN_AND_EQUAL_ACCESS_OR_"
        "DIRECT_SECRET_PHASE_COMPARATORS_ESTABLISH_NO_UNIQUE_PHASE_QEMU_"
        "ADVANTAGE"
    ),
    "ceiling": (
        "DETERMINISTIC_EXACT_Q_OMEGA_SPARSE_DENSITY_REFERENCE_ORACLE_WITH_"
        "ANALYTIC_RATIONAL_CHANNEL_CONTROLS_AND_EXPECTED_COMMON_BACKEND_ABI_"
        "DESCRIPTORS_NO_QEMU_EXECUTION_GUEST_CONTRACT_EXERCISE_PHYSICAL_"
        "CARRIER_CUSTODY_OR_TOTAL_RESOURCE_ADVANTAGE"
    ),
    "restoration": "NO_RESTORATION_CLAIM",
    "scope": (
        "INDEPENDENT_ALGEBRAIC_REFERENCE_COMPLETE_DENSITY_EQUALITIES_ONLY_"
        "WITHOUT_EXECUTED_DEVICE_BACKING_ATOMIC_HOLD_OR_PHYSICAL_RESTORATION"
    ),
    "disposition": (
        "EXACT_DUAL_RAIL_KICKBACK_AND_REFERENCE_COMPLETE_LOGICAL_RETURN_ARE_"
        "VALID_IN_THE_STIPULATED_REFERENCE_ORACLE_BUT_DIRECT_SECRET_PHASE_"
        "COMPILATION_EQUAL_COHERENT_ACCESS_SECRET_PROGRAM_STORAGE_AND_"
        "ORACLE_TOTAL_COSTS_PREVENT_ANY_UNIQUE_RESOURCE_ADVANTAGE_OR_M257_"
        "ESCAPE"
    ),
    "successor": (
        "EXECUTED_COMMON_PHASE_QEMU_DEVICE_BACKEND_QUALIFICATION_WITH_"
        "IDENTICAL_GUEST_DESCRIPTOR_RETURN_CLASS_AND_RESOURCE_COUNTER_PARITY"
    ),
}

M257_GUARDRAIL = (
    "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_"
    "COUNTED_AS_A_PHASE_RESOURCE"
)


class QualificationFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationFailure(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def read_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationFailure(f"{label} is not one canonical JSON object: {error}") from error
    require(isinstance(parsed, dict), f"{label} root is not an object")
    return parsed


def filesystem_type(path: Path) -> str:
    resolved = path.resolve()
    best_mount = Path("/")
    best_type = "UNKNOWN"
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if "-" not in fields:
            continue
        separator = fields.index("-")
        if separator + 1 >= len(fields):
            continue
        mount_point = Path(fields[4].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(mount_point.parts) >= len(best_mount.parts):
            best_mount = mount_point
            best_type = fields[separator + 1]
    return best_type


def exact_file(path: Path, expected_hash: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected_hash, f"{label} hash mismatch: {actual}")


def unique_source_slice(text: str, start_marker: str, end_marker: str,
                        label: str) -> str:
    require(text.count(start_marker) == 1,
            f"{label} start marker count mismatch")
    start = text.index(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    require(end >= 0, f"{label} end marker missing")
    return text[start:end]


def package_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    lane = package.parent
    evidence = package / "evidence"
    return {
        "package": package,
        "device_c": package / "qemu/hw/misc/phase-qemu-v11.c",
        "installer": package / "apply_to_qemu.py",
        "runner": package / "tests/run_phase_qemu_v11_qtest.py",
        "reference": package / "tests/phase_qemu_v11_separate_reference.py",
        "contract": package / "PHASE_QEMU_V11_DUAL_RAIL_ORACLE_CONTRACT.md",
        "findings": package / "PHASE_QEMU_V11_DUAL_RAIL_ORACLE_FINDINGS.md",
        "build_receipt": evidence / "PHASE_QEMU_V11_BUILD_RECEIPT.json",
        "gitignore": package / ".gitignore",
        "v0_c": lane / "phase_qemu_v0/qemu/hw/misc/phase-qemu-v0.c",
        "v1_c": lane / "phase_qemu_v1/qemu/hw/misc/phase-qemu-v1.c",
        "qtest_seal": evidence / "PHASE_QEMU_V11_DUAL_RAIL_ORACLE_QTEST.json",
        "reference_seal": evidence / "PHASE_QEMU_V11_DUAL_RAIL_ORACLE_SEPARATE_REFERENCE.json",
    }


def validate_self_ast(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "validate_package",
        "unique_source_slice",
        "validate_qemu_source",
        "run_installer_check",
        "run_qtest_twice",
        "audit_qtest",
        "run_reference_twice",
        "audit_reference",
        "handle_seals",
        "main",
    }
    require(required <= functions, f"qualifier AST missing hooks: {sorted(required - functions)}")


def validate_package(paths: dict[str, Path]) -> dict[str, Any]:
    for label in (
        "device_c", "installer", "runner", "reference", "contract",
        "findings", "build_receipt", "gitignore", "v0_c", "v1_c",
    ):
        exact_file(paths[label], EXPECTED_HASHES[label], label)
    validate_self_ast(Path(__file__).resolve())
    contract = paths["contract"].read_text(encoding="utf-8")
    findings = paths["findings"].read_text(encoding="utf-8")
    for label, authority in PRODUCTION_AUTHORITY.items():
        require(contract.count(authority) == 1, f"contract {label} authority mismatch")
        require(findings.count(authority) == 1, f"findings {label} authority mismatch")
    require(contract.count(M257_GUARDRAIL) == 1, "contract M257 guardrail mismatch")
    require(findings.count(M257_GUARDRAIL) == 1, "findings M257 guardrail mismatch")
    device_source = paths["device_c"].read_text(encoding="utf-8")
    for fragment, label in (
        ("bool (*lease_preflight)(PhaseQemuV11State *s);", "backend lease-preflight hook"),
        (".lease_preflight = external_lease_preflight,", "external lease-preflight binding"),
        ("!s->ops->lease_preflight(s)", "core lease-preflight dispatch"),
        ("sealed_resource_control_words", "sealed resource snapshot"),
        ("if (s->resource_sealed)", "post-seal mutation guard"),
    ):
        require(fragment in device_source, f"device source missing {label}")
    verify_implementations = (
        "ideal_verify_return", "open_verify_return", "external_verify_return",
    )
    require(device_source.count(
        "uint32_t (*verify_return)(PhaseQemuV11State *s);"
    ) == 1, "backend verify-return hook declaration mismatch")
    for implementation in verify_implementations:
        require(device_source.count(
            f"static uint32_t {implementation}(PhaseQemuV11State *s)"
        ) == 1, f"backend verify-return implementation mismatch: {implementation}")
        require(device_source.count(
            f".verify_return = {implementation},"
        ) == 1, f"backend verify-return binding mismatch: {implementation}")
    core_execute = unique_source_slice(
        device_source,
        "static void execute_atomic(PhaseQemuV11State *s)",
        "static void phase_command(PhaseQemuV11State *s, uint32_t command)",
        "execute_atomic core",
    )
    execute_callback = "result = s->ops->execute(s);"
    verifying_transition = "s->lifecycle = LIFE_VERIFYING_RETURN;"
    verify_callback = "result = s->ops->verify_return(s);"
    require(device_source.count(verifying_transition) == 1,
            "LIFE_VERIFYING_RETURN must have exactly one assignment")
    require(device_source.count(execute_callback) == 1,
            "backend execute callback count mismatch")
    require(device_source.count(verify_callback) == 1,
            "backend verify-return callback count mismatch")
    execute_offset = core_execute.find(execute_callback)
    transition_offset = core_execute.find(verifying_transition)
    verify_offset = core_execute.find(verify_callback)
    require(0 <= execute_offset < transition_offset < verify_offset,
            "core order must be execute callback, LIFE_VERIFYING_RETURN, verify callback")
    require(verifying_transition not in device_source[:device_source.index(
        "static void execute_atomic(PhaseQemuV11State *s)"
    )], "a backend assigns LIFE_VERIFYING_RETURN")
    require(verifying_transition not in device_source[
        device_source.index("static void phase_command(PhaseQemuV11State *s, uint32_t command)"):
    ], "non-core code assigns LIFE_VERIFYING_RETURN")

    installer_source = paths["installer"].read_text(encoding="utf-8")
    installer_cross_counts = (
        "kconfig_count = kconfig.count(KCONFIG_BLOCK.strip())",
        "kconfig_token_count = kconfig.count(KCONFIG_TOKEN)",
        'kconfig_symbol_count = kconfig.count("PHASE_QEMU_V11")',
        "meson_count = meson.count(MESON_LINE.strip())",
        'meson_symbol_count = meson.count("CONFIG_PHASE_QEMU_V11")',
        "meson_filename_count = meson.count(DEVICE_FILENAME)",
        "kconfig_count == kconfig_token_count == kconfig_symbol_count",
        "meson_count == meson_symbol_count == meson_filename_count",
    )
    require(installer_source.count(
        "def validate_integration_texts(kconfig: str, meson: str) -> tuple[int, int]:"
    ) == 1, "installer validate_integration_texts definition mismatch")
    for fragment in installer_cross_counts:
        require(installer_source.count(fragment) == 1,
                f"installer cross-count assertion mismatch: {fragment}")
    require(installer_source.count(
        "prospective_counts = validate_integration_texts("
    ) == 2, "installer must validate prospective counts in two paths")
    require(installer_source.count(
        "if prospective_counts != (1, 1):"
    ) == 2, "installer prospective-count rejection count mismatch")
    integrate_source = unique_source_slice(
        installer_source,
        "def integrate(source: Path, package_device: Path) -> IntegrationState:",
        "def main() -> int:",
        "installer integrate",
    )
    check_source = unique_source_slice(
        installer_source,
        "    if args.check:",
        "    else:",
        "installer --check branch",
    )
    for section, label in ((integrate_source, "integrate"), (check_source, "--check")):
        require(section.count(
            "prospective_counts = validate_integration_texts("
        ) == 1, f"installer {label} lacks prospective cross-count validation")
        require(section.count(
            "if prospective_counts != (1, 1):"
        ) == 1, f"installer {label} lacks prospective-count rejection")
    runner_source = paths["runner"].read_text(encoding="utf-8")
    require("def resource_seal_immutability_case" in runner_source,
            "runner missing sealed-resource mutation control")
    require("second_hop_preserved_sham" in runner_source,
            "runner missing second-hop SHAM control")
    receipt = json.loads(paths["build_receipt"].read_text(encoding="utf-8"))
    require(isinstance(receipt, dict), "build receipt root is not an object")
    return receipt


def validate_qemu_source(qemu_source: Path) -> dict[str, Path]:
    qemu_source = qemu_source.resolve(strict=True)
    require(qemu_source.is_dir(), "--qemu-source must be a directory")
    version = qemu_source / "VERSION"
    require(version.is_file(), "QEMU VERSION file is missing")
    require(version.read_text(encoding="utf-8").strip() == QEMU_VERSION,
            "QEMU source version mismatch")
    installed = {
        "v0": qemu_source / "hw/misc/phase-qemu-v0.c",
        "v1": qemu_source / "hw/misc/phase-qemu-v1.c",
        "v11": qemu_source / "hw/misc/phase-qemu-v11.c",
    }
    exact_file(installed["v0"], EXPECTED_HASHES["v0_c"], "installed V0 C")
    exact_file(installed["v1"], EXPECTED_HASHES["v1_c"], "installed V1 C")
    exact_file(installed["v11"], EXPECTED_HASHES["device_c"], "installed V11 C")
    kconfig = (qemu_source / "hw/misc/Kconfig").read_text(encoding="utf-8")
    meson = (qemu_source / "hw/misc/meson.build").read_text(encoding="utf-8")
    kconfig_lines = [line.strip() for line in kconfig.splitlines()]
    meson_lines = [line.strip() for line in meson.splitlines()]
    for version_name in ("V0", "V1", "V11"):
        require(kconfig_lines.count(f"config PHASE_QEMU_{version_name}") == 1,
                f"QEMU Kconfig PHASE_QEMU_{version_name} count mismatch")
        require(sum(
            f"'CONFIG_PHASE_QEMU_{version_name}'" in line
            for line in meson_lines
        ) == 1,
                f"QEMU Meson PHASE_QEMU_{version_name} count mismatch")
        require(sum(
            f"'phase-qemu-{version_name.lower()}.c'" in line
            for line in meson_lines
        ) == 1,
                f"QEMU Meson phase-qemu-{version_name.lower()}.c count mismatch")
    return installed


def validate_binary(qemu_binary: Path) -> Path:
    qemu_binary = qemu_binary.resolve(strict=True)
    require(qemu_binary.is_file(), "--qemu-binary must be a regular file")
    require(os.access(qemu_binary, os.X_OK), "--qemu-binary must be executable")
    exact_file(qemu_binary, EXPECTED_HASHES["qemu_binary"], "QEMU binary")
    completed = subprocess.run(
        [str(qemu_binary), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    require(completed.returncode == 0, "QEMU --version failed")
    require(completed.stderr == b"", "QEMU --version emitted stderr")
    lines = completed.stdout.decode("utf-8", "strict").splitlines()
    require(bool(lines) and lines[0] == QEMU_VERSION_LINE,
            "QEMU binary version line mismatch")
    return qemu_binary


def validate_receipt(receipt: dict[str, Any]) -> None:
    qemu = receipt.get("qemu")
    build = receipt.get("build_environment")
    runtime = receipt.get("runtime_qualification")
    coexist = receipt.get("coexisting_phase_qemu_devices")
    architecture = receipt.get("architecture")
    require(isinstance(qemu, dict) and isinstance(build, dict), "receipt build sections missing")
    require(isinstance(runtime, dict), "receipt runtime section missing")
    require(isinstance(coexist, dict) and isinstance(architecture, dict), "receipt scope missing")
    require(qemu.get("version") == QEMU_VERSION, "receipt QEMU version mismatch")
    require(qemu.get("binary_sha256") == EXPECTED_HASHES["qemu_binary"],
            "receipt binary hash mismatch")
    require(qemu.get("binary_version_line") == QEMU_VERSION_LINE,
            "receipt binary version line mismatch")
    require(qemu.get("system_install_modified") is False, "receipt claims system install mutation")
    require(build.get("managed_disk_backed_scratch") is True
            and build.get("ram_backed_scratch_used") is False
            and build.get("rebuild_result") == "INCREMENTAL_COMPILE_AND_LINK_PASS",
            "receipt build-scope accounting mismatch")
    require(runtime.get("runner_sha256") == EXPECTED_HASHES["runner"],
            "receipt runner hash mismatch")
    require(runtime.get("qtest_deterministic_payload_sha256") == EXPECTED_HASHES["qtest_payload"],
            "receipt qtest payload mismatch")
    require(runtime.get("separate_reference_source_sha256") == EXPECTED_HASHES["reference"],
            "receipt reference source mismatch")
    require(runtime.get("separate_reference_stdout_sha256") == EXPECTED_HASHES["reference_stdout"],
            "receipt reference stdout mismatch")
    require(runtime.get("separate_reference_claim_payload_sha256") == EXPECTED_HASHES["reference_claim_payload"],
            "receipt reference claim payload mismatch")
    expected_runtime = {
        "headless_qtest_qmp_only": True,
        "all_nine_residue_pairs": True,
        "same_allocation_generations": 9,
        "prepare_count": 1,
        "client_supply_count": 18,
        "reuse_count": 8,
        "real_qmp_unix_migration_sham": True,
        "second_hop_migration_preserves_sham": True,
        "sham_latched_across_qmp_system_reset_after_pci_reenumeration": True,
        "sealed_resource_snapshot_rejects_post_commit_mutation": True,
        "external_unavailability_dispatched_by_backend_lease_preflight": True,
        "qemu_stdout_stderr_empty": True,
    }
    for key, value in expected_runtime.items():
        require(runtime.get(key) == value, f"receipt runtime mismatch: {key}")
    for name, expected_hash in (("v0", EXPECTED_HASHES["v0_c"]), ("v1", EXPECTED_HASHES["v1_c"])):
        item = coexist.get(name)
        require(isinstance(item, dict), f"receipt {name} section missing")
        require(item.get("source_sha256") == expected_hash, f"receipt {name} hash mismatch")
        require(item.get("kconfig_entry_count") == 1 and item.get("meson_entry_count") == 1,
                f"receipt {name} integration count mismatch")
    v11 = coexist.get("v11")
    require(isinstance(v11, dict), "receipt V11 section missing")
    require(v11.get("package_source_sha256") == EXPECTED_HASHES["device_c"],
            "receipt V11 package hash mismatch")
    require(v11.get("installed_source_sha256") == EXPECTED_HASHES["device_c"],
            "receipt V11 installed hash mismatch")
    require(v11.get("installer_sha256") == EXPECTED_HASHES["installer"],
            "receipt installer hash mismatch")
    require(v11.get("kconfig_entry_count") == 1 and v11.get("meson_entry_count") == 1,
            "receipt V11 integration count mismatch")
    expected_architecture = {
        "qemu_device_source_integrated": True,
        "qemu_device_implemented": True,
        "common_guest_visible_contract_source_integrated": True,
        "common_guest_visible_contract_compiled": True,
        "common_guest_visible_contract_exercised": True,
        "reintegration_gate_passed": True,
        "standalone_twin_qualifies": False,
        "physical_oracle_adapter_implemented": False,
        "physical_oracle_or_carrier_custody_observed": False,
        "resource_or_query_advantage_established": False,
        "m257_escape_established": False,
    }
    for key, value in expected_architecture.items():
        require(architecture.get(key) is value, f"receipt architecture mismatch: {key}")
    require(receipt.get("status") == "PASS_COMPILED_COMMON_PHASE_QEMU_V11_REINTEGRATION_BUILD_AND_QTEST",
            "receipt status mismatch")


def create_run_tree(scratch_dir: Path) -> dict[str, Path]:
    scratch_dir = scratch_dir.resolve(strict=True)
    require(scratch_dir.is_dir(), "--scratch-dir must be a directory")
    require(os.access(scratch_dir, os.W_OK | os.X_OK), "--scratch-dir is not writable")
    fs_type = filesystem_type(scratch_dir)
    require(fs_type not in {"tmpfs", "ramfs"}, "RAM-backed scratch is forbidden")
    run_root = scratch_dir / f"phase-qemu-v11-qualify-{os.getpid()}-{time.monotonic_ns()}"
    require(not run_root.exists(), "fresh qualifier run root already exists")
    run_root.mkdir(mode=0o700)
    children: dict[str, Path] = {"root": run_root}
    for name in (
        "installer-check", "qtest-1", "qtest-2", "reference-1", "reference-2",
    ):
        child = run_root / name
        child.mkdir(mode=0o700)
        children[name] = child
    return children


def run_python(script: Path, arguments: Sequence[str], cwd: Path, timeout: int) -> bytes:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    require(completed.returncode == 0,
            f"{script.name} failed with exit code {completed.returncode}")
    require(completed.stderr == b"", f"{script.name} emitted stderr")
    return completed.stdout


def run_installer_check(paths: dict[str, Path], qemu_source: Path,
                        runs: dict[str, Path]) -> None:
    stdout = run_python(
        paths["installer"],
        ["--check", str(qemu_source)],
        runs["installer-check"],
        300,
    )
    lines = stdout.decode("utf-8", "strict").splitlines()
    fields: dict[str, str] = {}
    for line in lines:
        require(line.count("=") == 1, f"installer check line malformed: {line!r}")
        key, value = line.split("=", 1)
        require(key not in fields, f"installer check duplicate field: {key}")
        fields[key] = value
    expected = {
        "mode": "check",
        "qemu_version": QEMU_VERSION,
        "integration_state": "installed",
        "package_device_sha256": EXPECTED_HASHES["device_c"],
        "installer_sha256": EXPECTED_HASHES["installer"],
        "target_present": "1",
        "target_current": "1",
        "kconfig_entries": "1",
        "meson_entries": "1",
        "qemu_device_source_integrated": "1",
        "qemu_device_implemented": "0",
        "common_guest_visible_contract_source_integrated": "1",
        "common_guest_visible_contract_compiled": "0",
        "reintegration_gate_passed": "0",
        "compiled_device_qtest_qualification_required": "1",
        "standalone_twin_qualifies": "0",
        "installed_device_sha256": EXPECTED_HASHES["device_c"],
    }
    require(fields == expected, "installer --check output mismatch")


def run_qtest_twice(paths: dict[str, Path], qemu_binary: Path, runs: dict[str, Path]) -> bytes:
    canonical_runs: list[bytes] = []
    for index in (1, 2):
        stdout = run_python(
            paths["runner"],
            ["--qemu-binary", str(qemu_binary), "--scratch-dir", str(runs[f"qtest-{index}"])],
            runs[f"qtest-{index}"],
            900,
        )
        outer = read_json_bytes(stdout, f"qtest run {index}")
        require(outer.get("status") == "PASS_PHASE_QEMU_V11_QTEST_QMP_EVIDENCE",
                f"qtest run {index} status mismatch")
        require(outer.get("source_sha256") == EXPECTED_HASHES["runner"],
                f"qtest run {index} source hash mismatch")
        evidence = outer.get("deterministic_evidence")
        require(isinstance(evidence, dict), f"qtest run {index} deterministic evidence missing")
        payload = canonical_bytes(evidence)
        require(outer.get("deterministic_payload_sha256") == EXPECTED_HASHES["qtest_payload"],
                f"qtest run {index} declared payload hash mismatch")
        require(sha256_bytes(payload) == EXPECTED_HASHES["qtest_payload"],
                f"qtest run {index} computed payload hash mismatch")
        canonical_runs.append(payload)
    require(canonical_runs[0] == canonical_runs[1], "qtest deterministic payloads differ")
    return canonical_runs[0]


def require_empty_stream_record(record: object, label: str) -> None:
    require(isinstance(record, dict), f"{label} stream record missing")
    require(record.get("stdout_bytes") == 0 and record.get("stderr_bytes") == 0,
            f"{label} process streams were not empty")
    require(record.get("stdout_sha256") == EMPTY_SHA256,
            f"{label} stdout hash was not empty")
    require(record.get("stderr_sha256") == EMPTY_SHA256,
            f"{label} stderr hash was not empty")


def audit_qtest(payload: bytes) -> dict[str, Any]:
    deterministic = read_json_bytes(payload, "canonical qtest payload")
    require(deterministic.get("qemu_binary_sha256") == EXPECTED_HASHES["qemu_binary"],
            "qtest binary hash mismatch")
    evidence = deterministic.get("evidence")
    require(isinstance(evidence, dict), "qtest evidence missing")
    contract = evidence.get("contract")
    cases = evidence.get("cases")
    require(isinstance(contract, dict) and isinstance(cases, dict), "qtest contract/cases missing")
    require(contract.get("pci_identity") == [0x1234, 0x11FB, 1], "PCI identity mismatch")
    require(contract.get("magic") == 0x50483131 and contract.get("abi") == 0x00010000,
            "magic/ABI mismatch")
    require(contract.get("bar0_size") == 0x1000, "BAR size mismatch")
    require(contract.get("descriptor_words") == DESCRIPTOR_WORDS, "descriptor mismatch")
    require(contract.get("backend_ids") == [0x0B01, 0x0B02, 0x0B80], "backend IDs mismatch")
    require(contract.get("fault_modes") == list(range(1, 8)), "fault modes mismatch")
    require(contract.get("resource_schema") == 0x00010001, "resource schema mismatch")
    require(contract.get("allocated_private_secret_storage_bits") == 168,
            "allocated secret storage mismatch")
    require(contract.get("logical_secret_entropy_bits") == 4,
            "logical secret entropy mismatch")
    require(contract.get("resource_unknown_u64") == 0xFFFFFFFFFFFFFFFF,
            "resource unknown sentinel mismatch")
    require(contract.get("resource_registers") == {
        "allocated_secret_storage_bits": 0x0F0,
        "compiler_ops": 0x198,
        "controller_ops": 0x1A0,
        "construction_ops": 0x1A8,
        "logical_secret_entropy_bits": 0x1B0,
        "carrier_photon_number": 0x1B8,
    }, "resource register map mismatch")
    require(contract.get("private_property_names") == ["test-private-a", "test-private-b"],
            "private QOM property contract mismatch")
    require(contract.get("observer_property_names") == [
        "test-observe-phase-a", "test-observe-phase-b",
        "test-observe-prepare-count", "test-observe-client-supply-count",
        "test-observe-allocation-lo", "test-observe-allocation-hi",
        "test-observe-same-backing", "test-observe-kr-return",
        "test-observe-factorized", "test-observe-port-clear",
        "test-observe-env-factored", "test-observe-return-class",
    ], "observer QOM property contract mismatch")
    assumptions = evidence.get("runner_assumptions")
    require(isinstance(assumptions, dict), "runner assumptions missing")
    require(assumptions.get("source_isolation_required_every_generation") is True
            and assumptions.get("reused_lease_lifecycle") == 2
            and assumptions.get("response_outputs_held_until_ack") is True
            and assumptions.get("service_mode_private_and_observer_qom_rejected") is True
            and assumptions.get("migration_sham_is_reset_irreversible") is True,
            "runner lifecycle/architecture assumptions mismatch")

    ideal_wrapper = cases.get("ideal_all_nine_and_reuse")
    require(isinstance(ideal_wrapper, dict), "ideal case missing")
    ideal = ideal_wrapper.get("evidence")
    require(isinstance(ideal, dict), "ideal case evidence missing")
    transactions = ideal.get("transactions")
    require(isinstance(transactions, list) and len(transactions) == 9,
            "ideal transaction count mismatch")
    require(ideal.get("ordered_residue_pairs") == EXPECTED_PAIRS,
            "ordered residue pairs mismatch")
    require(ideal.get("all_nine_pairs_exercised") is True,
            "all-nine execution flag false")
    require(ideal.get("descriptor_byte_identical_for_all_pairs") is True,
            "descriptor varied across pairs")
    require(ideal.get("same_allocation_for_all_pairs") is True,
            "allocation varied across pairs")
    allocations = {tuple(item.get("allocation_id", [])) for item in transactions}
    fingerprints = {item.get("descriptor_fingerprint") for item in transactions}
    require(len(allocations) == 1 and len(fingerprints) == 1,
            "allocation/fingerprint uniqueness mismatch")
    for index, transaction in enumerate(transactions, 1):
        pair = EXPECTED_PAIRS[index - 1]
        require(transaction.get("generation") == index, f"generation {index} mismatch")
        require(transaction.get("residue_pair") == pair, f"generation {index} pair mismatch")
        require(transaction.get("lease_result_lifecycle") == (1 if index == 1 else 2),
                f"generation {index} reused LEASE lifecycle mismatch")
        require(transaction.get("source_isolation_issued_this_generation") is True,
                f"generation {index} omitted source isolation")
        require(transaction.get("boundary_locked_until_atomic_commit") is True,
                f"generation {index} boundary lock mismatch")
        require(transaction.get("outputs_held_before_ack") is True,
                f"generation {index} did not hold outputs")
        require(transaction.get("outputs_held_after_ack") is False,
                f"generation {index} held outputs after ACK")
        require(transaction.get("response_ready_after_ack") is False,
                f"generation {index} response remained ready")
        observers = transaction.get("observers")
        resources = transaction.get("resources")
        require(isinstance(observers, dict) and isinstance(resources, dict),
                f"generation {index} observations/resources missing")
        require(observers.get("test-observe-phase-a") == pair[0]
                and observers.get("test-observe-phase-b") == pair[1],
                f"generation {index} phase observation mismatch")
        for observer in (
            "test-observe-same-backing", "test-observe-kr-return",
            "test-observe-factorized", "test-observe-port-clear",
            "test-observe-env-factored",
        ):
            require(observers.get(observer) == 1,
                    f"generation {index} observer false: {observer}")
        require(observers.get("test-observe-prepare-count") == 1,
                f"generation {index} prepare count mismatch")
        require(observers.get("test-observe-client-supply-count") == 2 * index,
                f"generation {index} fresh-client supply mismatch")
        require(resources.get("state_cells") == 9216 and resources.get("scratch_cells") == 9216,
                f"generation {index} state/scratch mismatch")
        require(resources.get("query_applications") == 2 * index
                and resources.get("logical_queries") == 2 * index,
                f"generation {index} query count mismatch")
        expected_controls = 29 + 30 * (index - 1)
        require(resources.get("control_words") == expected_controls
                and resources.get("controller_ops") == expected_controls,
                f"generation {index} controller accounting mismatch")
        require(resources.get("allocated_secret_storage_bits") == 168
                and resources.get("logical_secret_entropy_bits") == 4,
                f"generation {index} secret accounting mismatch")
        require(resources.get("compiler_ops") == 0xFFFFFFFFFFFFFFFF
                and resources.get("construction_ops") == 0xFFFFFFFFFFFFFFFF,
                f"generation {index} unknown resource sentinel mismatch")
        require(resources.get("carrier_photon_number") == 1
                and resources.get("precision_bits") == 64
                and resources.get("resource_schema") == 0x00010001,
                f"generation {index} fixed resource mismatch")
    require(ideal.get("final_prepare_count") == 1, "final prepare count mismatch")
    require(ideal.get("final_client_supply_count") == 18, "final client supply mismatch")
    require(ideal.get("final_reuse_count") == 8, "final reuse count mismatch")
    reuse = ideal.get("two_generation_reuse")
    require(isinstance(reuse, dict) and reuse.get("allocation_id_equal") is True,
            "two-generation allocation reuse mismatch")

    sealed = cases["sealed_resource_snapshot"]["evidence"]
    require(sealed.get("sealed_control_words") == 29,
            "sealed resource control count mismatch")
    sealed_digest = sealed.get("sealed_resource_digest")
    require(isinstance(sealed_digest, list) and len(sealed_digest) == 2
            and sealed_digest != [0, 0],
            "sealed resource digest missing")
    require(sealed.get("width_correct_post_seal_write_error") == 14
            and sealed.get("argument_state_unchanged") is True
            and sealed.get("resource_snapshot_unchanged") is True
            and sealed.get("boundary_unchanged") is True,
            "post-seal resource immutability control mismatch")

    open_zero = cases["open_zero_parity"]["evidence"]
    require(open_zero.get("zero_open_model_exact_parity") is True,
            "zero-noise open parity failed")
    open_nonzero = cases["open_nonzero_fail_closed"]["evidence"]
    require(open_nonzero.get("return_class") == 2
            and open_nonzero.get("exact_return_status") is False
            and open_nonzero.get("reuse_rejected") is True,
            "nonzero open classification mismatch")
    require(open_nonzero.get("outputs_held_before_ack") is True
            and open_nonzero.get("outputs_held_after_ack") is False,
            "nonzero open hold/ACK mismatch")
    require(open_nonzero.get("loss_q63", 0) != 0 and open_nonzero.get("dephasing_q63", 0) != 0,
            "nonzero open resources missing")
    external = cases["external_unavailable"]["evidence"]
    require(external.get("lease_error") == 15 and external.get("lifecycle") == 0
            and external.get("response_ready") is False,
            "external adapter failure mismatch")

    faults = cases.get("fault_modes")
    require(isinstance(faults, dict) and sorted(faults) == sorted(FAULT_ERRORS),
            "fault evidence set mismatch")
    for mode, expected_error in FAULT_ERRORS.items():
        fault = faults[mode]["evidence"]
        require(fault.get("fault_mode") == int(mode)
                and fault.get("error") == expected_error
                and fault.get("return_class") == 3
                and fault.get("lifecycle") == 12
                and fault.get("boundary_locked") is True,
                f"fault mode {mode} mismatch")

    order = cases["wrong_order_and_generation"]["evidence"]
    require(order.get("execute_before_lease_error") == 3
            and order.get("prepare_before_lease_error") == 3
            and order.get("wrong_generation_lease_error") == 4
            and order.get("lifecycle_unchanged") is True,
            "order/generation controls mismatch")
    mmio = cases["invalid_mmio_widths"]["evidence"]
    require(mmio.get("all_invalid_widths_fail_closed") is True
            and mmio.get("lifecycle_unchanged") is True,
            "invalid MMIO fail-closed flag mismatch")
    require(mmio.get("invalid_byte_read") == {"returned": 0xFF, "latched_error": 2}
            and mmio.get("invalid_word_read") == {"returned": 0xFFFF, "latched_error": 2}
            and mmio.get("invalid_unaligned_read") == {"returned": 0xFFFFFFFFFFFFFFFF, "latched_error": 2},
            "invalid MMIO read controls mismatch")
    for key in (
        "invalid_byte_write_latched_error", "invalid_word_write_latched_error",
        "invalid_unaligned_write_latched_error",
    ):
        require(mmio.get(key) == 2, f"invalid MMIO write mismatch: {key}")
    service = cases["service_mode_private_and_observer_rejection"]["evidence"]
    require(service.get("test_provider_enabled") is False
            and service.get("private_provider_capability_exposed") is False
            and service.get("observer_capability_exposed") is False,
            "service-mode capability rejection mismatch")
    require(service.get("private_property") == "test-private-a"
            and service.get("observer_property") == "test-observe-phase-a"
            and service.get("private_qom_error_class") == "GenericError"
            and service.get("observer_qom_error_class") == "GenericError"
            and service.get("arm_private_error") == 15,
            "service-mode QOM rejection mismatch")
    require(cases["missing_private_binding"]["evidence"].get("boundary_locked") is True,
            "missing-private control leaked")
    require(cases["duplicate_private_binding"]["evidence"].get("transaction_remained_exact") is True,
            "duplicate-private control corrupted transaction")
    smuggle = cases["private_smuggle_descriptor"]["evidence"]
    require(smuggle.get("seal_error") == 21 and smuggle.get("descriptor_rejected") is True,
            "private-smuggle control mismatch")
    require(cases["snapshot_command_rejection"]["evidence"].get("snapshot_command_error") == 12,
            "snapshot rejection mismatch")

    migration = cases["real_migration_sham"]
    require(migration.get("transport") == "REAL_QMP_UNIX_LIVE_MIGRATION"
            and migration.get("source_query_migrate_status") == "completed",
            "real migration transport mismatch")
    require(migration.get("qmp_system_reset_issued_after_migration") is True
            and migration.get("pci_bar_reenumerated_after_system_reset") is True
            and migration.get("sham_latched_across_system_reset") is True,
            "migration reset/reenumeration SHAM proof mismatch")
    require(migration.get("second_hop_query_migrate_status") == "completed"
            and migration.get("second_hop_preserved_sham") is True
            and migration.get("second_hop_boundary_locked") is True,
            "second-hop migration SHAM proof mismatch")
    require(migration.get("destination_lifecycle") == 13
            and migration.get("post_reset_lifecycle") == 13
            and migration.get("destination_sham") is True
            and migration.get("post_reset_sham") is True
            and migration.get("post_reset_snapshot_lineage") is True,
            "migration SHAM lineage mismatch")
    require(migration.get("post_reset_pci") == migration.get("destination_pci"),
            "post-reset BAR re-enumeration identity mismatch")
    require_empty_stream_record(migration.get("captured_process_streams", {}).get("source"),
                                "migration source")
    require_empty_stream_record(migration.get("captured_process_streams", {}).get("destination"),
                                "migration destination")
    require_empty_stream_record(
        migration.get("captured_process_streams", {}).get("second_destination"),
        "migration second destination",
    )
    for case_name, wrapper in cases.items():
        if case_name == "fault_modes" or case_name == "real_migration_sham":
            continue
        require_empty_stream_record(wrapper.get("captured_process_streams"), case_name)
    for mode, wrapper in faults.items():
        require_empty_stream_record(wrapper.get("captured_process_streams"), f"fault {mode}")
    require(evidence.get("all_qemu_stdout_stderr_empty") is True,
            "global QEMU stream assertion false")
    require(evidence.get("headless_qtest_qmp_only") is True,
            "headless qtest/QMP assertion false")
    require(evidence.get("physical_oracle_or_carrier_custody_claim") is False
            and evidence.get("query_separation_or_advantage_claim") is False,
            "qtest evidence widened a nonclaim")
    require(evidence.get("status") == "PASS_PHASE_QEMU_V11_QTEST_QMP_EVIDENCE",
            "qtest evidence status mismatch")
    return deterministic


def run_reference_twice(paths: dict[str, Path], runs: dict[str, Path]) -> bytes:
    outputs = [
        run_python(paths["reference"], [], runs[f"reference-{index}"], 300)
        for index in (1, 2)
    ]
    require(outputs[0] == outputs[1], "independent-reference stdout differs across runs")
    require(sha256_bytes(outputs[0]) == EXPECTED_HASHES["reference_stdout"],
            "independent-reference stdout hash mismatch")
    return outputs[0]


def audit_reference(stdout: bytes, qtest: dict[str, Any]) -> dict[str, Any]:
    reference = read_json_bytes(stdout, "independent reference")
    require(canonical_bytes(reference) + b"\n" == stdout,
            "independent-reference stdout is not canonical JSON plus newline")
    require(reference.get("source_sha256") == EXPECTED_HASHES["reference"],
            "reference self-hash mismatch")
    require(reference.get("claim_payload_sha256") == EXPECTED_HASHES["reference_claim_payload"],
            "reference claim payload hash mismatch")
    mapping = {
        "claim": "claim",
        "ceiling": "ceiling",
        "restoration": "restoration_classification",
        "scope": "restoration_scope",
        "disposition": "resource_disposition",
        "successor": "next_mechanism",
    }
    for label, key in mapping.items():
        require(reference.get(key) == REFERENCE_AUTHORITY[label],
                f"reference {label} authority mismatch")
    claim_payload = reference.get("claim_payload")
    require(isinstance(claim_payload, dict), "reference claim payload missing")
    for label, key in mapping.items():
        require(claim_payload.get(key) == REFERENCE_AUTHORITY[label],
                f"reference claim payload {label} mismatch")

    checks = reference.get("checks")
    require(isinstance(checks, dict) and len(checks) == 27,
            "reference check count is not 27")
    require(all(value is True for value in checks.values()),
            "one or more reference checks failed")
    require(reference.get("status") == "PASS_INDEPENDENT_V11_DUAL_RAIL_EXACT_NOISY_REFERENCE"
            and reference.get("reference_self_assertion") == "PASS_INDEPENDENT_V11_DUAL_RAIL_EXACT_NOISY_REFERENCE",
            "reference status mismatch")

    exact = reference.get("exact_model")
    require(isinstance(exact, dict), "reference exact model missing")
    require(exact.get("dimensions") == [2, 2, 2, 2, 3, 2]
            and exact.get("total_dimension") == 96
            and exact.get("prepared_component_count") == 8
            and exact.get("prepared_density_denominator") == 8
            and exact.get("prepared_density_nonzero_entries") == 64
            and exact.get("arithmetic_field") == "Q(omega)"
            and exact.get("floating_point_decisions") == 0,
            "reference exact-model dimensions/arithmetic mismatch")
    pairs = reference.get("all_nine_residue_pairs")
    require(isinstance(pairs, list) and len(pairs) == 9, "reference pair count mismatch")
    for index, pair in enumerate(pairs):
        expected = EXPECTED_PAIRS[index]
        require([pair.get("residue_a"), pair.get("residue_b")] == expected,
                f"reference pair {index} order mismatch")
        for key in (
            "client_a_choi_exact", "client_b_choi_exact",
            "carrier_reference_joint_return_exact", "complete_factorization_exact",
            "environment_trivial", "fresh_client_a_distinct_from_fresh_client_b",
            "all_phase_ports_exact", "same_prepared_carrier_value",
            "density_hermitian", "qualifies_exact_logical_return",
            "qualifies_reuse_boundary",
        ):
            require(pair.get(key) is True, f"reference pair {index} failed {key}")
        require(pair.get("coherent_query_count") == 2
                and pair.get("query_order") == ["A", "B"]
                and pair.get("return_class") == "EXACT_COMPLETE_RETURN",
                f"reference pair {index} query/return mismatch")
        require(pair.get("phase_ports") == pair.get("expected_phase_ports"),
                f"reference pair {index} phase ports mismatch")

    controls = reference.get("analytic_controls")
    require(isinstance(controls, dict) and set(controls) == {
        "vacuum_carrier_null", "vacuum_dual_rail_non_eigenstate",
        "differential_rail_phase", "balanced_loss_to_vacuum",
        "dephasing_environment_tag",
    }, "reference analytic control set mismatch")
    for name, control in controls.items():
        require(control.get("qualifies_exact_logical_return") is False
                and control.get("qualifies_reuse_boundary") is False,
                f"reference control qualified unexpectedly: {name}")

    direct = reference.get("direct_secret_controlled_phase_comparator")
    equal = reference.get("equal_coherent_access_comparator")
    wave = reference.get("coherent_wave_comparator")
    require(direct.get("all_nine_client_boundaries_match") is True
            and direct.get("unique_phase_resource_advantage") is False,
            "direct comparator mismatch")
    require(equal.get("identical_query_count_and_order") is True
            and equal.get("phase_route_query_count") == 2
            and equal.get("comparator_query_count") == 2
            and equal.get("unique_query_advantage") is False,
            "equal-access comparator mismatch")
    require(wave.get("all_nine_exact_boundaries_reproduced") is True
            and wave.get("physical_total_resource_comparison_established") is False,
            "coherent-wave comparator mismatch")
    theorem = reference.get("nielsen_chuang_program_theorem")
    require(theorem.get("general_program_dimension_law") == "d^(N-1)"
            and theorem.get("minimum_exact_secret_program_dimension_for_residue_pairs") == 9
            and theorem.get("secret_storage_states_required") == 9
            and theorem.get("program_states_materialized") is False
            and theorem.get("program_overlap_executed_or_measured") is False,
            "Nielsen-Chuang program theorem scope mismatch")
    caveats = reference.get("m241_m242_forrelation_m257_caveats")
    require(caveats.get("m257_guardrail") == M257_GUARDRAIL
            and caveats.get("m257_escape") is False
            and caveats.get("total_resource_advantage") is False
            and caveats.get("forrelation_is_prospective_only") is True,
            "M241/M242/Forrelation/M257 caveats mismatch")

    descriptor = reference.get("expected_guest_descriptor")
    backend = reference.get("expected_backend_contract")
    require(descriptor.get("words_unsigned") == DESCRIPTOR_WORDS
            and descriptor.get("mmio_neutral") is True
            and descriptor.get("query_count") == 2
            and descriptor.get("query_order") == ["A", "B"]
            and descriptor.get("guest_descriptor_executed") is False,
            "reference guest descriptor mismatch")
    require(backend.get("common_device_id") == "COMMON_PHASE_QEMU_DEVICE"
            and backend.get("backend_id") == "D3_DUAL_RAIL_BELL_CHOI_EXACT_NOISY_BACKEND_V11"
            and backend.get("abi_id") == "COMMON_PHASE_QEMU_GUEST_DESCRIPTOR_ABI_V1"
            and backend.get("register_widths") == {
                "control_bits": 32,
                "status_bits": 32,
                "descriptor_address_bits": 64,
                "result_address_bits": 64,
                "resource_counter_bits": 64,
                "residue_field_bits": 2,
            }
            and backend.get("backend_executed_by_reference") is False
            and backend.get("registers_accessed_by_reference") is False,
            "reference backend contract mismatch")
    require(qtest["evidence"]["contract"]["descriptor_words"] == descriptor["words_unsigned"],
            "qtest/reference descriptor parity mismatch")

    architecture = reference.get("architecture_scope")
    require(architecture.get("phase_qemu_layer_classification")
            == "COMMON_PHASE_QEMU_DEVICE_BACKEND_REFERENCE_ORACLE"
            and architecture.get("reference_oracle_only") is True
            and architecture.get("qemu_device_implemented_by_reference") is False
            and architecture.get("qemu_device_executed_by_reference") is False
            and architecture.get("common_guest_visible_device_contract_exercised") is False
            and architecture.get("eligible_for_architecture_promotion") is False
            and architecture.get("reference_can_promote_architecture") is False,
            "reference architecture discipline mismatch")
    nonclaims = reference.get("nonclaims")
    require(isinstance(nonclaims, dict) and all(value is False for value in nonclaims.values()),
            "reference nonclaim widened")
    resource_scope = reference.get("resource_scope_accounting")
    require(resource_scope.get("reference_total_hilbert_dimension_enumerated") == 96
            and resource_scope.get("reference_residue_pairs_enumerated") == 9
            and resource_scope.get("integrated_backend_resource_parity_claimed") is False
            and resource_scope.get("physical_oracle_generation_cost_measured") is False,
            "reference resource scope mismatch")
    return reference


def handle_seals(paths: dict[str, Path], qtest_payload: bytes, reference_stdout: bytes,
                 *, write_seals: bool, preseal: bool) -> None:
    qtest_seal = paths["qtest_seal"]
    reference_seal = paths["reference_seal"]
    require(qtest_seal.parent.is_dir(), "evidence directory is missing")
    if preseal:
        return
    if write_seals:
        require(not qtest_seal.exists() and not reference_seal.exists(),
                "--write-seals requires both seal paths to be absent")
        with qtest_seal.open("xb") as handle:
            handle.write(qtest_payload)
        with reference_seal.open("xb") as handle:
            handle.write(reference_stdout)
        return
    missing = [str(path) for path in (qtest_seal, reference_seal) if not path.is_file()]
    require(not missing, f"missing required seals: {missing}")
    require(qtest_seal.read_bytes() == qtest_payload, "qtest seal byte mismatch")
    require(reference_seal.read_bytes() == reference_stdout,
            "separate-reference seal byte mismatch")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Phase-QEMU V11 qualifier")
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
        receipt = validate_package(paths)
        validate_receipt(receipt)
        validate_qemu_source(args.qemu_source)
        qemu_binary = validate_binary(args.qemu_binary)
        runs = create_run_tree(args.scratch_dir)
        run_installer_check(paths, args.qemu_source.resolve(), runs)
        qtest_payload = run_qtest_twice(paths, qemu_binary, runs)
        qtest = audit_qtest(qtest_payload)
        reference_stdout = run_reference_twice(paths, runs)
        audit_reference(reference_stdout, qtest)
        handle_seals(
            paths,
            qtest_payload,
            reference_stdout,
            write_seals=args.write_seals,
            preseal=args.preseal,
        )
        print(PASS_LINE)
        return 0
    except Exception as error:
        print(f"FAIL_CLOSED M269_PHASE_QEMU_V11_DUAL_RAIL_DEVICE: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
