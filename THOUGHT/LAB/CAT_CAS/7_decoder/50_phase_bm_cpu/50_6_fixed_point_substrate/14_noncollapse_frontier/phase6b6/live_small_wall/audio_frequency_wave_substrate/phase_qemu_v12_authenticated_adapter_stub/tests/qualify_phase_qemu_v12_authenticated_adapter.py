#!/usr/bin/env python3
"""Strict M270 Phase-QEMU V12 package, reference, and qtest qualifier.

Every executable input and scratch root is explicit. Runtime artifacts are
created only below the caller's disk-backed scratch directory and retained for
inspection. Seal writes, when explicitly requested, are limited to the two
frozen evidence paths declared below. The qtest seal excludes wall-time data.
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
import time
from pathlib import Path
from typing import Any, Sequence


PASS_LINE = (
    "PASS_STRICT_SCOPE M270_PHASE_QEMU_V12_AUTHENTICATED_ADAPTER "
    "SCIENCE=SEPARATE_REFERENCE_PARITY "
    "RESTORATION=NO_RESTORATION_CLAIM "
    "SCOPE=COMPILED_QEMU_EXTERNAL_ADAPTER_TEST_STUB "
    "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
)

EXPECTED_HASHES = {
    "device_c": "5fe4f99e9bf2ff78774e75149c532293adb03c01c035fcac3d05e2fe74b6153f",
    "installer": "f8cd2b819e4c3ae72d40b3edd6dce5eb996dc17cb4dc78e4c62cc7ff116e0ea9",
    "runner": "5bdc38c369f89f8afe61a574dbafba24e012cb64c18912e5e282ce5a0d151766",
    "base_runner": "fa70a83c24cb9bd9ffc78e6cf07ed3c21dc99d342c6302ac1482dd5df48b47dd",
    "reference": "f97e53b52502eb357df4a8bbc0411655c64a6f4a5f4a2705378088d6133ea080",
    "contract": "18a88c8edbfdc68f7a7e50fcb28df475f218618a3042eb3df9bca7e7212294bc",
    "findings": "328efad7d74288659b4508540b7af3cf0a4343d22e86bf3ed43cd30d4c82b62d",
    "build_receipt": "cfd61e0c68e97e953803dfad7f550d4053e7c404da3c22f754b80b6752907bae",
    "v0_c": "b79ec06f870b611142f5df5c97db2f8e34027458da5acc933f90b694b2055764",
    "v1_c": "8b991d8961a6e108d1a4aa7498172564b017c6e622cb8192c6fa15c33638e362",
    "v11_c": "84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771",
    "qemu_binary": "917f9f76047b8b255545dd67f7de12a1c1448a55397db0e811ff86858a55dffe",
    "qtest_payload": "f16afc21f7d4a054d7aba67715507566bb28a234d11a2594dfe8c4211bc099bd",
    "reference_stdout": "f738f46ce13bc18b717c6f56767f6c94fd40dad763dfb0031e612db5682638fd",
    "reference_payload": "3cf0771d4826485fcfc350a42b023f49fc6138a612b94ac2bfd856e881fd7aba",
}

QEMU_VERSION = "10.2.4"
QEMU_VERSION_LINE = "QEMU emulator version 10.2.4"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_PAIRS = [[a, b] for a in range(3) for b in range(3)]

PRODUCTION_AUTHORITY = {
    "claim": (
        "COMPILED_QEMU_10_2_4_PHASE_QEMU_V12_COMMON_GUEST_VISIBLE_PCI_DEVICE_"
        "BACKEND_EXERCISES_A_HARDWARE_DISCONNECTED_TEST_ONLY_DETERMINISTIC_21_"
        "BIT_INTEGRITY_TWO_SLOT_ASYNCHRONOUS_EXTERNAL_ADAPTER_FOR_ALL_NINE_"
        "INTERNAL_ZOMEGA_PAIRS_AND_COMMITS_ONLY_APPROX_MODEL_OUTPUTS_HELD_UNTIL_"
        "ACK_THEN_SPENT_WITH_NO_BEGIN_REUSE_WHILE_PENDING_RESET_REAL_QMP_"
        "MIGRATION_AND_SERVICE_DEFAULT_FAIL_CLOSED"
    ),
    "ceiling": (
        "NO_HARDWARE_PHYSICAL_CARRIER_COHERENT_PORT_CRYPTOGRAPHIC_SECURITY_"
        "STATISTICAL_PHYSICAL_EVIDENCE_CUSTODY_PHYSICAL_RETURN_RESTORATION_REUSE_"
        "OR_ADVANTAGE_ALL_EXTERNAL_PHYSICAL_QUANTITIES_UNKNOWN_AND_FINITE_"
        "PHYSICAL_EVIDENCE_MAY_NEVER_BE_EXACT_FORMAL_OR_AUTHORIZE_BEGIN_REUSE"
    ),
    "restoration": "NO_RESTORATION_CLAIM",
    "scope": (
        "COMPILED_QEMU_10_2_4_PHASE_QEMU_V12_PCI_ABI_HARDWARE_DISCONNECTED_TEST_"
        "ONLY_TWO_SLOT_ASYNC_ADAPTER_PROTOCOL_AND_INTERNAL_96_DIMENSION_ZOMEGA_"
        "SOFTWARE_MODEL_ONLY"
    ),
    "disposition": (
        "V12_ESTABLISHES_COMMON_COMPILED_ASYNC_ADAPTER_LIFECYCLE_INTEGRITY_ORDER_"
        "EXPIRY_CANCEL_TIMEOUT_ACK_SPENT_AND_SHAM_CONTROLS_BUT_NOT_PHYSICAL_OR_"
        "CRYPTOGRAPHIC_PROMOTION_DIRECT_EQUAL_ACCESS_PHASE_COMPILER_CONTROLS_AND_"
        "M257_REMAINS_INTACT"
    ),
    "successor": (
        "REAL_AUTHENTICATED_HARDWARE_CONNECTED_DUAL_RAIL_DISPERSIVE_ADAPTER_"
        "BEHIND_THE_COMMON_PHASE_QEMU_BACKEND_WITH_BOUNDED_PHYSICAL_RECEIPTS_AND_"
        "INDEPENDENT_STATISTICAL_VALIDATION_WITHOUT_EXACT_FORMAL_PHYSICAL_RETURN_"
        "OR_REUSE"
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
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


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
        raise QualificationFailure(f"{label} is not one JSON object: {error}") from error
    require(isinstance(parsed, dict), f"{label} root is not an object")
    return parsed


def exact_file(path: Path, expected_hash: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected_hash, f"{label} hash mismatch: {actual}")


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


def unique_source_slice(
    text: str, start_marker: str, end_marker: str, label: str
) -> str:
    require(text.count(start_marker) == 1, f"{label} start marker count mismatch")
    start = text.index(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    require(end >= 0, f"{label} end marker missing")
    return text[start:end]


def extract_register_map(text: str, enum_name: str) -> dict[str, int]:
    match = re.search(
        rf"enum\s+{re.escape(enum_name)}\s*\{{(?P<body>.*?)\}};", text, re.S
    )
    require(match is not None, f"register enum {enum_name} missing")
    pairs = re.findall(
        r"\b(REG_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+)",
        match.group("body"),
    )
    result = {name: int(encoded, 16) for name, encoded in pairs}
    require(len(result) == len(pairs), f"duplicate names in {enum_name}")
    return result


def package_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    lane = package.parent
    evidence = package / "evidence"
    return {
        "package": package,
        "device_c": package / "qemu/phase-qemu-v12.c",
        "installer": package / "apply_to_qemu.py",
        "runner": package / "tests/run_phase_qemu_v12_authenticated_adapter_qtest.py",
        "base_runner": lane / "phase_qemu_v11/tests/run_phase_qemu_v11_qtest.py",
        "reference": package / "tests/phase_qemu_v12_adapter_separate_reference.py",
        "contract": package / "PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_CONTRACT.md",
        "findings": package / "PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_FINDINGS.md",
        "build_receipt": evidence / "PHASE_QEMU_V12_BUILD_RECEIPT.json",
        "v11_c": lane / "phase_qemu_v11/qemu/hw/misc/phase-qemu-v11.c",
        "qtest_seal": evidence / "PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_QTEST.json",
        "reference_seal": (
            evidence / "PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_SEPARATE_REFERENCE.json"
        ),
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
        "validate_device_source",
        "validate_installer_source",
        "validate_qemu_source",
        "validate_receipt",
        "run_installer_check",
        "run_qtest_twice",
        "audit_qtest",
        "run_reference_twice",
        "audit_reference",
        "handle_seals",
        "main",
    }
    require(required <= functions, f"qualifier AST missing hooks: {sorted(required - functions)}")


def validate_device_source(source: str) -> None:
    required_once = {
        '#define TYPE_PHASE_QEMU_V12 "phase-qemu-v12"': "QOM identity",
        "#define PHASE_V12_DEVICE_ID 0x11fc": "PCI identity",
        "#define PHASE_V12_MAGIC 0x50483132u": "PH12 magic",
        "#define PHASE_V12_ABI 0x00020000u": "V12 ABI",
        "#define PHASE_V12_STANDALONE_TWIN_QUALIFIES 0": "promotion gate",
        "bool (*lease_preflight)(PhaseQemuV12State *s);": "lease preflight hook",
        "uint32_t (*execute)(PhaseQemuV12State *s);": "execute hook",
        "uint32_t (*verify_return)(PhaseQemuV12State *s);": "verify hook",
        "void (*cancel)(PhaseQemuV12State *s);": "cancel hook",
        "void (*sanitize)(PhaseQemuV12State *s);": "sanitize hook",
        ".lease_preflight = external_lease_preflight,": "external preflight binding",
        ".execute = external_execute,": "external execute binding",
        ".poll = external_poll,": "external poll binding",
        ".verify_return = external_verify_return,": "external verify binding",
        ".cancel = external_cancel,": "external cancel binding",
        ".sanitize = external_sanitize,": "external sanitizer binding",
        "bool adapter_authenticated_lineage;": "authenticated-lineage latch",
    }
    for fragment, label in required_once.items():
        require(source.count(fragment) == 1, f"device {label} count mismatch")

    lease = unique_source_slice(
        source,
        "    case CMD_LEASE:",
        "    case CMD_PREPARE:",
        "LEASE core",
    )
    require(
        "!s->ops->lease_preflight(s)" in lease
        and lease.index("!s->ops->lease_preflight(s)") < lease.index("s->leased = true;"),
        "external availability is not dispatched before LEASE mutation",
    )

    execute = unique_source_slice(
        source,
        "static void execute_atomic(PhaseQemuV12State *s)",
        "static void phase_command(PhaseQemuV12State *s, uint32_t command)",
        "execute core",
    )
    finish = unique_source_slice(
        source,
        "static void finish_backend_execution(PhaseQemuV12State *s, uint32_t result)",
        "static void execute_atomic(PhaseQemuV12State *s)",
        "finish core",
    )
    require(
        execute.count("result = s->ops->execute(s);") == 1
        and execute.count("finish_backend_execution(s, result);") == 1,
        "execute callback/core completion split mismatch",
    )
    transition = "s->lifecycle = LIFE_VERIFYING_RETURN;"
    verifier = "result = s->ops->verify_return(s);"
    require(
        finish.count(transition) == 1
        and finish.count(verifier) == 1
        and finish.index(transition) < finish.index(verifier),
        "VERIFYING_RETURN does not precede the distinct verify callback",
    )
    require(
        "s->return_verified = result == RETURN_EXACT_FORMAL;" in finish
        and "if (result == RETURN_EXACT_FORMAL)" in finish
        and "s->reuse_qualified = true;" in finish,
        "only EXACT_FORMAL return may qualify reuse",
    )
    external_verify = unique_source_slice(
        source,
        "static uint32_t external_verify_return(PhaseQemuV12State *s)",
        "static void external_cancel(PhaseQemuV12State *s)",
        "external verifier",
    )
    require(
        "s->same_backing = false;" in external_verify
        and "return RETURN_APPROX_MODEL;" in external_verify,
        "external model does not forcibly deny exact return/reuse",
    )

    setter = unique_source_slice(
        source,
        "static void adapter_envelope_set(Object *object, Visitor *visitor,",
        "enum PhaseV12ObserverId {",
        "adapter envelope setter",
    )
    sealed_guard = "if (s->resource_sealed) {"
    mutations = (
        "s->resource_control_words++;",
        "s->adapter_auth_rejected++;",
        "s->adapter_envelope[property_slot] = value;",
        "s->adapter_authenticated_lineage = true;",
    )
    require(setter.count(sealed_guard) == 1, "QOM sealed-resource guard mismatch")
    guard_offset = setter.index(sealed_guard)
    require(
        all(guard_offset < setter.index(fragment) for fragment in mutations),
        "QOM adapter state mutates before the sealed-resource guard",
    )

    spend = unique_source_slice(
        source,
        "static void spend_generation(PhaseQemuV12State *s, uint32_t error)",
        "static bool seal_response(PhaseQemuV12State *s)",
        "failed receipt path",
    )
    require(
        "s->return_class = RETURN_FAILED;" in spend
        and "s->reuse_qualified = false;" in spend
        and "if (!seal_response(s))" in spend,
        "timeout/cancel failure path does not seal a nonreusable FAILED receipt",
    )
    command = unique_source_slice(
        source,
        "static void phase_command(PhaseQemuV12State *s, uint32_t command)",
        "static bool register_is_u32(hwaddr address)",
        "command core",
    )
    ack = unique_source_slice(command, "    case CMD_ACK_RESPONSE:", "    case CMD_BEGIN_REUSE:", "ACK")
    reuse = unique_source_slice(command, "    case CMD_BEGIN_REUSE:", "    case CMD_SNAPSHOT:", "reuse")
    require(
        "s->lifecycle = LIFE_SPENT;" in ack
        and "s->adapter_authenticated_lineage = false;" in ack,
        "ACK does not spend and clear authenticated lineage",
    )
    for fragment in (
        "!s->response_acked",
        "!s->restored",
        "!s->reuse_qualified",
        "!s->same_backing",
        "s->return_class != RETURN_EXACT_FORMAL",
    ):
        require(fragment in reuse, f"BEGIN_REUSE check missing: {fragment}")

    pre_save = unique_source_slice(
        source,
        "static int phase_qemu_v12_pre_save(void *opaque)",
        "static int phase_qemu_v12_post_load(void *opaque, int version_id)",
        "pre-save",
    )
    post_load = unique_source_slice(
        source,
        "static int phase_qemu_v12_post_load(void *opaque, int version_id)",
        "static const VMStateDescription vmstate_phase_qemu_v12",
        "post-load",
    )
    vmstate = unique_source_slice(
        source,
        "static const VMStateDescription vmstate_phase_qemu_v12",
        "static void phase_qemu_v12_realize(PCIDevice *pci_device, Error **errp)",
        "VMState",
    )
    require(
        "s->ops->cancel(s);" in pre_save
        and "enter_migration_sham(s);" in pre_save
        and "enter_migration_sham(s);" in post_load
        and ".pre_save = phase_qemu_v12_pre_save," in vmstate
        and ".post_load = phase_qemu_v12_post_load," in vmstate,
        "pre-save/post-load SHAM lineage hooks mismatch",
    )
    for forbidden in (
        "adapter_envelope", "private_residue", "density", "scratch",
        "descriptor", "boundary", "arm_nonce",
    ):
        require(forbidden not in vmstate, f"VMState leaked hidden field: {forbidden}")

    unrealize = unique_source_slice(
        source,
        "static void phase_qemu_v12_unrealize(DeviceState *device)",
        "static void phase_qemu_v12_instance_init(Object *object)",
        "unrealize",
    )
    cancel_offset = unrealize.index("s->ops->cancel(s);")
    sanitize_offset = unrealize.index("s->ops->sanitize(s);")
    clear_offset = unrealize.index("clear_private(s);")
    require(
        cancel_offset < sanitize_offset < clear_offset
        and "clear_descriptor(s);" in unrealize
        and "clear_boundary(s);" in unrealize
        and "release_tags(s);" in unrealize
        and "s->adapter_authenticated_lineage = false;" in unrealize,
        "unrealize does not cancel then sanitize and clear private lineage",
    )


def validate_installer_source(source: str) -> None:
    required = {
        'DEVICE_ID_PATTERN = re.compile(r"\\b0x11fc\\b", re.IGNORECASE)': "PCI collision scanner",
        'TYPE_PATTERN = re.compile(r\'"phase-qemu-v12"\')': "QOM collision scanner",
        "validate_canonical_update_target(target)": "canonical update target validation",
        "validate_v11_prefix(v11, package_device)": "frozen V11 prefix validation",
        "scan_pci_collisions(source, source / DEVICE_RELATIVE)": "tree collision scan",
        "if target.is_symlink() or (target.exists() and not target.is_file()):": "symlink/nonregular rejection",
        'target.name + ".failed-install-recovery"': "recoverable first-install path",
        "failed_target_recovery.exists() or": "recovery collision check",
        "failed_target_recovery.is_symlink()": "dangling recovery collision check",
        "target.replace(failed_target_recovery)": "recoverable partial-target move",
        "if sha256(target) not in ALLOWED_PRIOR_V12_SHA256:": "unknown overwrite rejection",
        "if frozen_before != frozen_snapshot(source):": "V0/V1/V11 preservation",
    }
    for fragment, label in required.items():
        require(fragment in source, f"installer missing {label}")
    require(
        source.count("validate_integration_texts(new_kconfig, new_meson)") == 1,
        "installer prospective integration cross-check mismatch",
    )
    require(
        'if kconfig.count(V11_KCONFIG_BLOCK.strip()) != 1:' in source
        and 'if meson.count(V11_MESON_LINE.strip()) != 1:' in source,
        "installer does not pin canonical V11 integration",
    )


def validate_package(paths: dict[str, Path]) -> dict[str, Any]:
    for label in (
        "device_c", "installer", "runner", "base_runner", "reference",
        "contract", "findings", "build_receipt", "v11_c",
    ):
        exact_file(paths[label], EXPECTED_HASHES[label], label)
    validate_self_ast(Path(__file__).resolve())

    for document_name in ("contract", "findings"):
        document = paths[document_name].read_text(encoding="utf-8")
        authority = document.split("## Architectural discipline", 1)[0]
        if document_name == "findings":
            authority = document.split("## Evidence identity and status", 1)[0]
        for label, value in PRODUCTION_AUTHORITY.items():
            fenced = f"```text\n{value}\n```"
            require(authority.count(fenced) == 1,
                    f"{document_name} {label} authority mismatch")
        guardrail = f"```text\n{M257_GUARDRAIL}\n```"
        require(authority.count(guardrail) == 1,
                f"{document_name} M257 guardrail mismatch")
        for prohibited in ("physical promotion is established", "m257 escape is established"):
            require(prohibited not in document.lower(),
                    f"{document_name} widened a prohibited claim")

    device_source = paths["device_c"].read_text(encoding="utf-8")
    validate_device_source(device_source)
    v11_map = extract_register_map(
        paths["v11_c"].read_text(encoding="utf-8"), "PhaseV11Register"
    )
    v12_map = extract_register_map(device_source, "PhaseV12Register")
    changed = sorted(name for name, offset in v11_map.items() if v12_map.get(name) != offset)
    require(not changed, f"V12 changed frozen V11 register prefix: {changed}")

    installer_source = paths["installer"].read_text(encoding="utf-8")
    validate_installer_source(installer_source)
    runner_source = paths["runner"].read_text(encoding="utf-8")
    for fragment, label in (
        ("def recompute_resource_digest", "independent resource-digest recomputation"),
        ("def direct_phase_compiler", "direct equal-access compiler"),
        ("source_pre_save_canceled_once_and_latched_sham", "source pre-save SHAM check"),
        ("second_hop_preserved_sham", "second-hop SHAM check"),
        ("unrealize_callback_semantics_not_observable_from_qmp_quit", "teardown scope caveat"),
        ("static_source_audit_required_for_sanitizer_claim", "unrealize static-audit caveat"),
        ("failed_attempt_resource_sealed", "sealed FAILED receipt check"),
        ("failure_receipt_acked_to_spent", "FAILED receipt ACK check"),
    ):
        require(fragment in runner_source, f"runner missing {label}")

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
        "v0_c": qemu_source / "hw/misc/phase-qemu-v0.c",
        "v1_c": qemu_source / "hw/misc/phase-qemu-v1.c",
        "v11_c": qemu_source / "hw/misc/phase-qemu-v11.c",
        "device_c": qemu_source / "hw/misc/phase-qemu-v12.c",
    }
    for label, path in installed.items():
        exact_file(path, EXPECTED_HASHES[label], f"installed {label}")
    v11_map = extract_register_map(
        installed["v11_c"].read_text(encoding="utf-8"), "PhaseV11Register"
    )
    v12_map = extract_register_map(
        installed["device_c"].read_text(encoding="utf-8"), "PhaseV12Register"
    )
    require(
        all(v12_map.get(name) == offset for name, offset in v11_map.items()),
        "installed V12 does not preserve the full V11 register prefix",
    )
    kconfig = (qemu_source / "hw/misc/Kconfig").read_text(encoding="utf-8")
    meson = (qemu_source / "hw/misc/meson.build").read_text(encoding="utf-8")
    kconfig_lines = [line.strip() for line in kconfig.splitlines()]
    meson_lines = [line.strip() for line in meson.splitlines()]
    for version_name in ("V0", "V1", "V11", "V12"):
        filename = version_name.lower()
        require(kconfig_lines.count(f"config PHASE_QEMU_{version_name}") == 1,
                f"QEMU Kconfig PHASE_QEMU_{version_name} count mismatch")
        require(sum(f"'CONFIG_PHASE_QEMU_{version_name}'" in line for line in meson_lines) == 1,
                f"QEMU Meson PHASE_QEMU_{version_name} count mismatch")
        require(sum(f"'phase-qemu-{filename}.c'" in line for line in meson_lines) == 1,
                f"QEMU Meson phase-qemu-{filename}.c count mismatch")
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
    require(receipt.get("schema") ==
            "PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_COMPILED_BUILD_RECEIPT_V1",
            "receipt schema mismatch")
    qemu = receipt.get("qemu")
    build = receipt.get("build_environment")
    coexist = receipt.get("coexisting_phase_qemu_devices")
    runtime = receipt.get("runtime_qualification")
    architecture = receipt.get("architecture")
    for value, label in (
        (qemu, "qemu"), (build, "build"), (coexist, "coexistence"),
        (runtime, "runtime"), (architecture, "architecture"),
    ):
        require(isinstance(value, dict), f"receipt {label} section missing")
    require(qemu == {
        "version": QEMU_VERSION,
        "official_source_archive_sha256":
            "821b545b92f165e57dddccac5077d76d4d436a226595b8813ad59306bbfd0746",
        "configured_targets": ["x86_64-softmmu"],
        "configure_options": [
            "--target-list=x86_64-softmmu", "--disable-docs",
            "--disable-werror", "--enable-debug",
        ],
        "binary_sha256": EXPECTED_HASHES["qemu_binary"],
        "binary_version_line": QEMU_VERSION_LINE,
        "system_install_modified": False,
    }, "receipt QEMU identity mismatch")
    require(build == {
        "managed_disk_backed_scratch": True,
        "post_crash_runtime_evidence_used_fresh_managed_disk_scratch": True,
        "ram_backed_scratch_used": False,
        "rebuild_result": "INCREMENTAL_COMPILE_AND_LINK_PASS",
    }, "receipt build environment mismatch")

    expected_devices = {
        "v0": {
            "source_sha256": EXPECTED_HASHES["v0_c"],
            "kconfig_entry_count": 1, "meson_entry_count": 1,
        },
        "v1": {
            "source_sha256": EXPECTED_HASHES["v1_c"],
            "kconfig_entry_count": 1, "meson_entry_count": 1,
        },
        "v11": {
            "source_sha256": EXPECTED_HASHES["v11_c"],
            "qtest_transport_runner_sha256": EXPECTED_HASHES["base_runner"],
            "kconfig_entry_count": 1, "meson_entry_count": 1,
        },
        "v12": {
            "package_source_sha256": EXPECTED_HASHES["device_c"],
            "installed_source_sha256": EXPECTED_HASHES["device_c"],
            "installer_sha256": EXPECTED_HASHES["installer"],
            "kconfig_entry_count": 1, "meson_entry_count": 1,
        },
    }
    require(coexist == expected_devices, "receipt device coexistence mismatch")

    expected_runtime = {
        "runner_sha256": EXPECTED_HASHES["runner"],
        "qtest_deterministic_payload_sha256": EXPECTED_HASHES["qtest_payload"],
        "qtest_matching_replay_count": 2,
        "qtest_aggregate_check_count": 11,
        "separate_reference_source_sha256": EXPECTED_HASHES["reference"],
        "separate_reference_stdout_sha256": EXPECTED_HASHES["reference_stdout"],
        "separate_reference_payload_sha256": EXPECTED_HASHES["reference_payload"],
        "separate_reference_matching_replay_count": 2,
        "separate_reference_check_count": 29,
        "headless_qtest_qmp_only": True,
        "all_nine_residue_pairs": True,
        "all_external_results_are_approx_model": True,
        "authentication_negative_control_count": 7,
        "timeout_and_cancel_commit_sealed_failed_receipts": True,
        "timeout_cancel_reset_and_migration_reject_begin_reuse": True,
        "real_qmp_pre_save_and_two_hop_migration_sham": True,
        "pending_process_teardown_exits_cleanly": True,
        "unrealize_cancel_and_sanitize_is_pinned_source_static_audit": True,
        "qemu_stdout_stderr_empty": True,
    }
    require(runtime == expected_runtime, "receipt runtime qualification mismatch")
    expected_architecture = {
        "qemu_device_source_integrated": True,
        "qemu_device_implemented": True,
        "common_guest_visible_contract_source_integrated": True,
        "common_guest_visible_contract_compiled": True,
        "common_guest_visible_contract_exercised": True,
        "reintegration_gate_passed": True,
        "standalone_twin_qualifies": False,
        "external_adapter_is_hardware_disconnected_test_stub": True,
        "physical_adapter_or_carrier_custody_observed": False,
        "cryptographic_authentication_established": False,
        "physical_restoration_or_reuse_established": False,
        "resource_or_query_advantage_established": False,
        "m257_escape_established": False,
    }
    require(architecture == expected_architecture, "receipt architecture mismatch")
    require(receipt.get("status") ==
            "PASS_COMPILED_COMMON_PHASE_QEMU_V12_TEST_ADAPTER_BUILD_AND_QTEST_WITHOUT_PHYSICAL_PROMOTION",
            "receipt status mismatch")


def create_run_tree(scratch_dir: Path) -> dict[str, Path]:
    scratch_dir = scratch_dir.resolve(strict=True)
    require(scratch_dir.is_dir(), "--scratch-dir must be a directory")
    require(os.access(scratch_dir, os.W_OK | os.X_OK), "--scratch-dir is not writable")
    require(filesystem_type(scratch_dir) not in {"tmpfs", "ramfs"},
            "RAM-backed scratch is forbidden")
    run_root = scratch_dir / f"phase-qemu-v12-qualify-{os.getpid()}-{time.monotonic_ns()}"
    require(not run_root.exists(), "fresh qualifier run root already exists")
    run_root.mkdir(mode=0o700)
    children: dict[str, Path] = {"root": run_root}
    for name in ("installer-check", "qtest-1", "qtest-2", "reference-1", "reference-2"):
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


def run_installer_check(
    paths: dict[str, Path], qemu_source: Path, runs: dict[str, Path]
) -> None:
    stdout = run_python(
        paths["installer"], ["--check", str(qemu_source)],
        runs["installer-check"], 300,
    )
    fields: dict[str, str] = {}
    for line in stdout.decode("utf-8", "strict").splitlines():
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
        "frozen_v11_sha256": EXPECTED_HASHES["v11_c"],
        "target_present": "1",
        "target_current": "1",
        "kconfig_entries": "1",
        "meson_entries": "1",
        "qemu_device_source_integrated": "1",
        "qemu_device_implemented": "0",
        "common_guest_visible_contract_source_integrated": "1",
        "common_guest_visible_contract_compiled": "0",
        "reintegration_gate_passed": "0",
        "compiled_qtest_qualification_required": "1",
        "standalone_twin_qualifies": "0",
        "installed_device_sha256": EXPECTED_HASHES["device_c"],
    }
    require(fields == expected, "installer --check output mismatch")


def run_qtest_twice(
    paths: dict[str, Path], qemu_binary: Path, runs: dict[str, Path]
) -> bytes:
    canonical_runs: list[bytes] = []
    for index in (1, 2):
        stdout = run_python(
            paths["runner"],
            ["--qemu-binary", str(qemu_binary), "--scratch-dir", str(runs[f"qtest-{index}"])],
            runs[f"qtest-{index}"],
            1200,
        )
        outer = read_json_bytes(stdout, f"qtest run {index}")
        require(set(outer) == {"deterministic_evidence", "wall_times_ns"},
                f"qtest run {index} outer schema mismatch")
        require(isinstance(outer.get("wall_times_ns"), dict),
                f"qtest run {index} wall-time metadata missing")
        evidence = outer.get("deterministic_evidence")
        require(isinstance(evidence, dict),
                f"qtest run {index} deterministic evidence missing")
        payload = canonical_bytes(evidence) + b"\n"
        require(sha256_bytes(payload) == EXPECTED_HASHES["qtest_payload"],
                f"qtest run {index} deterministic payload hash mismatch")
        canonical_runs.append(payload)
    require(canonical_runs[0] == canonical_runs[1],
            "qtest deterministic payloads differ across runs")
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
    evidence = read_json_bytes(payload, "canonical qtest evidence")
    require(canonical_bytes(evidence) + b"\n" == payload,
            "qtest payload is not canonical JSON plus newline")
    require(evidence.get("schema") ==
            "M270_PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_QTEST_EVIDENCE_V1",
            "qtest schema mismatch")
    classification = evidence.get("classification")
    require(classification == {
        "phase_qemu_layer_classification":
            "COMPILED_PHASE_QEMU_V12_COMMON_DEVICE_HARDWARE_DISCONNECTED_EXTERNAL_ADAPTER_TEST_STUB",
        "common_guest_visible_device_contract_exercised": True,
        "external_hardware_connected": False,
        "authentication_is_test_only_deterministic_integrity_tag": True,
        "cryptographic_security_claim": False,
        "physical_evidence_class": "NONE",
        "model_return_class": "APPROX_MODEL",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "begin_reuse_supported": False,
        "architecture_promotion": False,
        "mechanism_promotion": False,
    }, "qtest classification widened scope")
    require(evidence.get("build_identity") == {
        "qemu_binary_sha256": EXPECTED_HASHES["qemu_binary"],
        "runner_sha256": EXPECTED_HASHES["runner"],
        "base_runner_sha256": EXPECTED_HASHES["base_runner"],
    }, "qtest build identity mismatch")

    pairs = evidence.get("ideal_algebra_pairs")
    require(isinstance(pairs, list) and len(pairs) == 9, "qtest pair count mismatch")
    omega = ([1, 0], [0, 1], [-1, -1])
    for index, pair in enumerate(pairs):
        residues = EXPECTED_PAIRS[index]
        require(pair.get("fixture") == f"ideal_pair_{residues[0]}_{residues[1]}"
                and pair.get("residues") == residues,
                f"qtest pair {index} order mismatch")
        require(pair.get("authenticated_envelopes") == 2
                and pair.get("dispatches") == 2
                and pair.get("completions") == 2
                and pair.get("virtual_ticks") == 2,
                f"qtest pair {index} adapter counts mismatch")
        require(pair.get("return_class") == "APPROX_MODEL"
                and pair.get("physical_return_claim") is False
                and pair.get("begin_reuse_authorized") is False,
                f"qtest pair {index} return/reuse scope mismatch")
        for key in (
            "outputs_held_until_ack",
            "sealed_resource_snapshot_immutable_before_ack",
            "post_commit_qom_envelope_rejected_without_live_counter_mutation",
            "resource_digest_independently_recomputed",
            "direct_compiler_matches_observed_model_phases",
        ):
            require(pair.get(key) is True, f"qtest pair {index} failed {key}")
        comparator = pair.get("direct_equal_access_compiler")
        require(comparator == {
            "client_a_diagonal": [[1, 0], omega[residues[0]]],
            "client_b_diagonal": [[1, 0], omega[residues[1]]],
            "residue_accesses": 2,
        }, f"qtest pair {index} direct compiler mismatch")

    controls = evidence.get("authentication_controls")
    expected_errors = {
        "bad_tag": 32, "wrong_slot": 36, "stale_generation": 4,
        "slot_b_before_a": 34, "expired": 35, "direct_bypass": 32, "replay": 33,
    }
    require(isinstance(controls, dict) and set(controls) == set(expected_errors),
            "authentication control set mismatch")
    for name, error in expected_errors.items():
        control = controls[name]
        require(control.get("error") == error
                and control.get("dispatch_delta") == 0
                and control.get("qmp_error_class") == "GenericError",
                f"authentication control mismatch: {name}")

    asynchronous = evidence.get("asynchronous_controls")
    require(isinstance(asynchronous, dict)
            and set(asynchronous) == {"timeout", "cancel", "pending_reset"},
            "asynchronous control set mismatch")
    for name in ("timeout", "cancel"):
        control = asynchronous[name]
        require(control.get("cancel_count") == 1
                and control.get("failed_attempt_resource_sealed") is True
                and control.get("failure_receipt_acked_to_spent") is True
                and control.get("late_completion_rejected") is True
                and control.get("begin_reuse_authorized") is False,
                f"{name} FAILED receipt lifecycle mismatch")
    reset = asynchronous["pending_reset"]
    require(reset.get("cancel_count") == 1
            and reset.get("second_reset_preserved_sham") is True
            and reset.get("boundary_locked") is True
            and reset.get("begin_reuse_authorized") is False,
            "pending reset SHAM mismatch")

    migration = evidence.get("migration_reset")
    require(migration.get("transport") == "REAL_QMP_UNIX_LIVE_MIGRATION"
            and migration.get("source_query_migrate_status") == "completed"
            and migration.get("source_pre_save_canceled_once_and_latched_sham") is True
            and migration.get("second_hop_query_migrate_status") == "completed"
            and migration.get("destination_rejected_late_completion") is True
            and migration.get("second_hop_preserved_sham") is True
            and migration.get("system_reset_preserved_sham") is True
            and migration.get("boundary_locked") is True
            and migration.get("begin_reuse_authorized") is False,
            "real QMP migration/second-hop SHAM mismatch")
    streams = migration.get("captured_process_streams")
    require(isinstance(streams, dict), "migration streams missing")
    for name in ("source", "destination", "second_destination"):
        require_empty_stream_record(streams.get(name), f"migration {name}")

    unrealize = evidence.get("unrealize")
    require(unrealize.get("fixture") ==
            "pending_process_teardown_with_unrealize_sanitizer"
            and unrealize.get("adapter_was_pending_before_qmp_quit") is True
            and unrealize.get("qmp_quit_is_process_teardown_not_hot_unplug") is True
            and unrealize.get("unrealize_callback_semantics_not_observable_from_qmp_quit") is True
            and unrealize.get("static_source_audit_required_for_sanitizer_claim") is True
            and unrealize.get("process_exit_code") == 0
            and unrealize.get("process_teardown_streams_empty") is True,
            "qtest teardown scope mismatch")
    require_empty_stream_record(unrealize.get("captured_process_streams"),
                                "pending process teardown")

    abi = evidence.get("guest_abi_fault_controls")
    require(abi.get("fixture") == "common_prefix_width_alignment_and_adapter_read_only"
            and abi.get("adapter_read_only_write_error") == 2
            and abi.get("adapter_ledger_value_unchanged") is True
            and abi.get("lifecycle_unchanged") is True,
            "guest ABI fault control mismatch")
    invalid = abi.get("invalid_width_alignment")
    require(isinstance(invalid, dict)
            and invalid.get("all_invalid_widths_fail_closed") is True
            and invalid.get("lifecycle_unchanged") is True,
            "common-prefix width/alignment controls failed")
    service = evidence.get("service_mode")
    require(service == {
        "fixture": "external_service_mode_unavailable",
        "lease_error": 15,
        "lifecycle_unchanged": True,
        "physical_hardware_connected": False,
    }, "service-default failure mismatch")

    resources = evidence.get("resource_accounting")
    require(resources.get("resource_schema") == 0x00020001
            and resources.get("unknown_physical_values_use_uint64_max_or_null") is True
            and resources.get("external_carrier_photon_number_is_unknown") is True
            and resources.get("device_object_plus_largest_named_local_peak_is_only_a_floor") is True
            and resources.get("whole_qemu_process_allocator_and_rss_peak") == "NOT_ESTABLISHED"
            and resources.get("model_action_is_not_physical_action") is True
            and resources.get("total_resource_comparison") == "UNDETERMINED"
            and resources.get("resource_advantage_claim") is False,
            "qtest resource scope mismatch")
    unknown = resources.get("unknown_physical_resources")
    require(isinstance(unknown, dict) and len(unknown) == 8,
            "qtest unknown physical resource map mismatch")
    for name, entry in unknown.items():
        require(entry.get("status") == "UNKNOWN"
                and entry.get("value") is None
                and isinstance(entry.get("reason"), str)
                and bool(entry["reason"]),
                f"qtest physical resource encoded as known: {name}")
    comparator = evidence.get("equal_access_comparator")
    require(comparator == {
        "direct_phase_compiler_has_same_two_residue_accesses": True,
        "all_nine_direct_compilers_executed": True,
        "ideal_algebra_output_equal": True,
        "unique_query_advantage": False,
        "M257_escape": False,
    }, "qtest equal-access/M257 scope mismatch")
    checks = evidence.get("checks")
    require(isinstance(checks, dict) and len(checks) == 11
            and all(value is True for value in checks.values()),
            "qtest aggregate checks mismatch")
    return evidence


def run_reference_twice(paths: dict[str, Path], runs: dict[str, Path]) -> bytes:
    outputs = [
        run_python(paths["reference"], [], runs[f"reference-{index}"], 300)
        for index in (1, 2)
    ]
    require(outputs[0] == outputs[1], "separate-reference stdout differs across runs")
    require(sha256_bytes(outputs[0]) == EXPECTED_HASHES["reference_stdout"],
            "separate-reference stdout hash mismatch")
    return outputs[0]


def audit_reference(stdout: bytes, qtest: dict[str, Any]) -> dict[str, Any]:
    document = read_json_bytes(stdout, "separate reference")
    require(canonical_bytes(document) + b"\n" == stdout,
            "separate-reference stdout is not canonical JSON plus newline")
    require(document.get("schema") ==
            "M270_V12_AUTHENTICATED_ADAPTER_SEPARATE_REFERENCE_V1",
            "reference schema mismatch")
    require(document.get("source_sha256") == EXPECTED_HASHES["reference"],
            "reference self-hash mismatch")
    require(document.get("payload_sha256") == EXPECTED_HASHES["reference_payload"],
            "reference declared payload hash mismatch")
    payload = document.get("payload")
    require(isinstance(payload, dict), "reference payload missing")
    require(sha256_bytes(canonical_bytes(payload)) == EXPECTED_HASHES["reference_payload"],
            "reference computed payload hash mismatch")
    require(payload.get("authority") == {
        "scope": "NINE_EXACT_INTERNAL_SYMBOLIC_RESIDUE_PAIRS_ONLY",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "physical_evidence_class": "NONE",
        "future_physical_receipt_ceiling": "APPROX_MODEL_OR_STATISTICAL_ONLY",
    }, "reference authority mismatch")
    classification = payload.get("classification")
    require(classification.get("phase_qemu_layer_classification") ==
            "INDEPENDENT_SYMBOLIC_REFERENCE_OUTSIDE_QEMU_EXECUTION"
            and classification.get("stipulated_target_architecture") ==
            "V12_COMMON_DEVICE_EXTERNAL_ADAPTER_HARDWARE_DISCONNECTED"
            and classification.get("target_architecture_independently_verified_by_reference") is False
            and classification.get("common_guest_visible_device_contract_exercised_by_reference") is False
            and classification.get("qemu_execution_performed_by_reference") is False
            and classification.get("external_hardware_connected") is False
            and classification.get("physical_evidence_class") == "NONE"
            and classification.get("allowed_future_physical_receipt_classes") ==
            ["APPROX_MODEL", "STATISTICAL_ONLY"]
            and classification.get("restoration_classification") == "NO_RESTORATION_CLAIM"
            and classification.get("physical_restoration_claim") is False
            and classification.get("same_carrier_claim") is False
            and classification.get("carrier_custody_claim") is False,
            "reference classification widened scope")

    model = payload.get("cyclotomic_model")
    require(model.get("dimension") == 3
            and model.get("floating_point_decision_count") == 0
            and model.get("formal_joint_dimension") == 96
            and model.get("prepared_support_component_count") == 8
            and model.get("prepared_density_denominator") == 8,
            "reference exact model mismatch")
    pairs = payload.get("ideal_pairs")
    require(isinstance(pairs, list) and len(pairs) == 9,
            "reference ideal-pair count mismatch")
    for index, case in enumerate(pairs):
        residues = EXPECTED_PAIRS[index]
        require(case.get("fixture") == f"ideal_pair_{residues[0]}_{residues[1]}"
                and case.get("residues") == {"a": residues[0], "b": residues[1]}
                and case.get("formal_fresh_client_count") == 2
                and case.get("formal_query_slot_order") == ["A", "B"]
                and case.get("formal_query_slot_count") == 2
                and case.get("begin_reuse_count") == 0
                and case.get("internal_symbolic_complete_factorization") is True
                and case.get("internal_symbolic_carrier_reference_factor_unchanged") is True
                and case.get("direct_equal_access_compiler_parity") is True
                and case.get("physical_return_inference") is False
                and case.get("same_carrier_inference") is False,
                f"reference ideal pair mismatch: {index}")
    require(
        [case["fixture"] for case in pairs]
        == [item["fixture"] for item in qtest["ideal_algebra_pairs"]],
        "qtest/reference pair order mismatch",
    )

    theorem = payload.get("finite_evidence_theorem")
    require(theorem.get("name") == "FINITE_EVIDENCE_DOES_NOT_ENTAIL_EXACT_PHYSICAL_RETURN"
            and theorem.get("argument_classification") ==
            "SYMBOLIC_COUNTERMODEL_ARGUMENT_NOT_NUMERICAL_CONVERGENCE_TEST"
            and theorem.get("premises_machine_derived") is False
            and theorem.get("every_finite_n_is_nonexact") is True
            and theorem.get("countermodels_converge_to_ideal_without_becoming_ideal") is True
            and theorem.get("threshold_used_by_reference") is None,
            "reference finite-evidence theorem mismatch")
    tag = payload.get("integrity_tag")
    require(tag.get("width_bits") == 21
            and tag.get("deterministic") is True
            and tag.get("scope") == "OUTSIDE_INDEPENDENT_SCIENTIFIC_PARITY"
            and tag.get("computed_or_verified_by_reference") is False
            and tag.get("included_in_independent_scientific_parity") is False
            and tag.get("cryptographic_authentication_claim") is False
            and tag.get("collision_resistance_claim") is False
            and tag.get("unforgeability_claim") is False,
            "reference integrity-tag scope mismatch")
    comparator = payload.get("equal_access_comparator")
    require(comparator.get("formal_secret_residue_accesses_per_pair") == 2
            and comparator.get("adapter_formal_secret_residue_accesses_per_pair") == 2
            and comparator.get("all_nine_symbolic_outputs_identical") is True
            and comparator.get("unique_query_advantage") is False
            and comparator.get("total_resource_advantage") == "UNDETERMINED"
            and comparator.get("resource_advantage_claim") is False,
            "reference equal-access comparator mismatch")

    resources = payload.get("resource_accounting")
    require(resources.get("known_formal_counts") == {
        "residue_pairs": 9,
        "fresh_clients_per_pair": 2,
        "formal_query_slots_per_pair": 2,
        "formal_query_slots_total": 18,
        "begin_reuse": 0,
    } and resources.get("unknown_value_encoding") == "NULL_NEVER_NUMERIC_ZERO"
      and resources.get("unknown_physical_total_precludes_advantage_claim") is True
      and resources.get("production_resident_or_transient_backing_measured") is False,
      "reference resource accounting mismatch")
    unknown = resources.get("unknown_physical_resources")
    require(isinstance(unknown, dict) and len(unknown) == 8,
            "reference unknown-resource map mismatch")
    for name, entry in unknown.items():
        require(entry.get("status") == "UNKNOWN"
                and entry.get("value") is None
                and bool(entry.get("unit"))
                and bool(entry.get("reason")),
                f"reference physical resource encoded as known: {name}")

    promotion = payload.get("promotion_gate")
    require(promotion == {
        "mechanism_twins_can_kill": True,
        "mechanism_twins_can_nominate": True,
        "nomination_is_architecture_promotion": False,
        "eligible_for_architecture_promotion": False,
        "architecture_promotion": False,
        "required_for_promotion": [
            "INDEPENDENTLY_QUALIFIED_COMMON_COMPILED_BACKEND",
            "REAL_EXTERNAL_ADAPTER_STATISTICAL_VALIDATION",
        ],
        "requirements_satisfied_by_this_reference": [],
    }, "reference promotion gate mismatch")
    require(payload.get("m257") == {"status": "INTACT", "escape_established": False},
            "reference M257 status mismatch")
    nonclaims = payload.get("nonclaims")
    require(isinstance(nonclaims, list)
            and nonclaims == sorted(set(nonclaims))
            and "NO_ARCHITECTURE_PROMOTION_CLAIM" in nonclaims
            and "NO_CRYPTOGRAPHIC_AUTHENTICATION_CLAIM" in nonclaims
            and "NO_PHYSICAL_RETURN_CLAIM" in nonclaims
            and "NO_RESTORATION_CLAIM" in nonclaims
            and "NO_M257_ESCAPE_CLAIM" in nonclaims,
            "reference nonclaim set mismatch")
    checks = payload.get("checks")
    require(isinstance(checks, list) and len(checks) == 29
            and payload.get("check_count") == 29
            and payload.get("all_checks_pass") is True
            and len({item.get("id") for item in checks}) == 29
            and all(item.get("pass") is True for item in checks),
            "reference checks mismatch")
    return document


def handle_seals(
    paths: dict[str, Path], qtest_payload: bytes, reference_stdout: bytes,
    *, write_seals: bool, preseal: bool,
) -> None:
    qtest_seal = paths["qtest_seal"]
    reference_seal = paths["reference_seal"]
    require(qtest_seal.parent.is_dir(), "evidence directory is missing")
    if preseal:
        return
    if write_seals:
        require(not qtest_seal.exists() and not qtest_seal.is_symlink()
                and not reference_seal.exists() and not reference_seal.is_symlink(),
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
    parser = argparse.ArgumentParser(
        description="Strict Phase-QEMU V12 authenticated-adapter qualifier"
    )
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
        print(
            "FAIL_CLOSED M270_PHASE_QEMU_V12_AUTHENTICATED_ADAPTER: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
