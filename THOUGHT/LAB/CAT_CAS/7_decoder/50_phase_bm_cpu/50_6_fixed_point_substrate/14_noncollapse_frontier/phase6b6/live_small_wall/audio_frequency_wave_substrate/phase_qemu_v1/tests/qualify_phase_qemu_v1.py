#!/usr/bin/env python3
"""Reexecute and strictly qualify the bounded Phase-QEMU V1 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


RAM_FILESYSTEMS = {"tmpfs", "ramfs", "hugetlbfs"}
FORBIDDEN_SCRATCH_ROOTS = (Path("/tmp"), Path("/dev/shm"), Path("/run/shm"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def unescape_mount_field(value: str) -> str:
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"),
                             ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def filesystem_type(path: Path) -> str:
    candidates: list[tuple[int, str]] = []
    mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount = Path(unescape_mount_field(left_fields[4])).resolve()
        if is_below(path, mount):
            candidates.append((len(mount.parts), right_fields[0]))
    if not candidates:
        raise RuntimeError(f"cannot identify filesystem for {path}")
    return max(candidates)[1]


def require_managed_disk_scratch(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    if "/Codex/Scratch/turns/" not in text:
        raise RuntimeError("scratch must be a codex-scratch managed payload")
    if any(is_below(resolved, root) for root in FORBIDDEN_SCRATCH_ROOTS):
        raise RuntimeError("/tmp and RAM-backed scratch roots are forbidden")
    resolved.mkdir(parents=True, exist_ok=True)
    fs_type = filesystem_type(resolved)
    if fs_type in RAM_FILESYSTEMS:
        raise RuntimeError(f"RAM-backed scratch is forbidden: {fs_type}")
    return resolved


def run(command: list[str], cwd: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract_c_function(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"device source lacks {signature}")
    brace = source.find("{", start)
    if brace < 0:
        raise AssertionError(f"device source lacks body for {signature}")
    depth = 0
    for offset in range(brace, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:offset + 1]
    raise AssertionError(f"unterminated body for {signature}")


def require_ordered(text: str, tokens: list[str], scope: str) -> None:
    cursor = 0
    for token in tokens:
        location = text.find(token, cursor)
        if location < 0:
            raise AssertionError(f"{scope} lacks ordered token: {token}")
        cursor = location + len(token)


def all_control_values_pass(controls: dict[str, Any]) -> bool:
    for value in controls.values():
        if isinstance(value, bool):
            if not value:
                return False
        elif isinstance(value, dict) and "passes" in value:
            if value["passes"] is not True:
                return False
        else:
            return False
    return True


def qualify_source(package: Path, result: dict[str, Any]) -> None:
    device_path = package / "qemu" / "hw" / "misc" / "phase-qemu-v1.c"
    device = device_path.read_text(encoding="utf-8")
    contract = (package / "PHASE_QEMU_V1_CONTRACT.md").read_text(encoding="utf-8")
    contract_flat = " ".join(contract.split())
    runner = (package / "tests" / "run_phase_qemu_v1_qtest.py").read_text(
        encoding="utf-8"
    )
    installer = (package / "apply_to_qemu.py").read_text(encoding="utf-8")
    execute = extract_c_function(device, "static void execute_atomic")
    inverse = extract_c_function(device, "static bool execute_inverse_with_fault")
    pointer = extract_c_function(device, "static bool apply_pointer_gate")
    initial_state = extract_c_function(device, "static bool state_is_initial")
    mmio_read = extract_c_function(device, "static uint64_t phase_mmio_read")
    mmio_write = extract_c_function(device, "static void phase_mmio_write")
    command = extract_c_function(device, "static void phase_command")
    post_load = extract_c_function(device, "static int phase_qemu_v1_post_load")

    require_ordered(
        execute,
        [
            "apply_gate(s, s->descriptor[step])",
            "apply_pointer_gate(s)",
            "even_zero = coefficient_array_zero",
            "odd_zero = coefficient_array_zero",
            "if (even_zero == odd_zero)",
            "saved_boundary = even_zero ? 1 : 0",
            "apply_pointer_gate(s)",
            "execute_inverse_with_fault(s)",
            "state_is_initial(s)",
            "s->boundary_value = saved_boundary",
            "s->restoration_generation++",
            "s->restored = true",
            "s->response_ready = true",
        ],
        "accepted atomic source order",
    )
    require(
        "descriptor_index = s->descriptor_length - 1 - step" in inverse
        and "adjoint_gate(s->descriptor[descriptor_index])" in inverse,
        "inverse is not visibly derived from the sealed public descriptor",
    )
    require(
        not any(
            token in device
            for token in (
                "inverse_descriptor", "inverse_history", "expected_boundary",
                "expected_output", "answer_table", "lookup_table",
                "g_new", "g_malloc", "malloc(", "calloc(", "realloc(",
            )
        ),
        "device contains expected-answer, inverse-history, or dynamic-state storage",
    )
    require(
        "occupation(basis, s->boundary_mode) & 1" in pointer
        and pointer.count("s->coefficient[") >= 4,
        "pointer is not implemented as a computational-basis parity latch",
    )
    require(
        "s->coefficient[" not in mmio_read
        and "s->coefficient[" not in mmio_write,
        "carrier or pointer amplitudes are exposed through MMIO",
    )
    require(
        "REG_DESCRIPTOR_WORD" in mmio_write and "inverse" not in mmio_write.lower(),
        "MMIO accepts an inverse or lacks public descriptor staging",
    )
    require(
        "s->response_ready && !s->snapshot_lineage" in mmio_read
        and "PHASE_V1_LOCKED_BOUNDARY" in mmio_read,
        "boundary MMIO is not locked until the accepted response state",
    )
    require(
        command.count("if (s->snapshot_lineage)") >= 2
        and "case CMD_SNAPSHOT:" in command
        and "ERR_SNAPSHOT_REJECTED" in command,
        "snapshot lineage or snapshot-command denial is missing",
    )
    require_ordered(
        post_load,
        [
            "s->snapshot_lineage = true",
            "s->response_ready = false",
            "s->lifecycle = LIFE_SHAM",
            "release_tags(s)",
        ],
        "migration sham post-load order",
    )
    require(
        "VMSTATE_PCI_DEVICE(parent_obj" in device
        and "VMSTATE_INT64_ARRAY(coefficient" in device
        and "REG_RESOURCE_PEAK_BITS = 0x068" in device
        and "REG_REQUEST_OWNER = 0x070" in device
        and "REG_REQUEST_PROGRAM = 0x074" in device
        and "REG_REQUEST_GENERATION = 0x078" in device
        and "REG_DESCRIPTOR_INDEX = 0x07c" in device
        and "REG_DESCRIPTOR_WORD = 0x080" in device
        and "REG_DESCRIPTOR_LENGTH = 0x084" in device
        and "REG_BOUNDARY_MODE = 0x088" in device
        and "REG_DESCRIPTOR_FINGERPRINT = 0x090" in device
        and "REG_FAULT_MODE = 0x098" in device
        and "REG_RESET_SHAMS" not in device,
        "VMState PCI/hidden-state coverage or the fixed nonoverlapping ABI is absent",
    )
    require_ordered(
        initial_state,
        [
            "s->denominator_power != 0",
            "s->coefficient[coefficient_index(0, 2)] != 1",
            "index != coefficient_index(0, 2)",
            "s->coefficient[index] != 0",
            "s->pointer_clear && !s->coupler_active",
        ],
        "exact accepted initial-state predicate",
    )
    begin_reuse = command[
        command.find("case CMD_BEGIN_REUSE:"):command.find("case CMD_SNAPSHOT:")
    ]
    require(
        begin_reuse.startswith("case CMD_BEGIN_REUSE:")
        and "s->generation++" in begin_reuse
        and "s->lifecycle = LIFE_REUSABLE" in begin_reuse
        and "memset" not in begin_reuse
        and "s->coefficient" not in begin_reuse
        and "CMD_PREPARE" not in begin_reuse,
        "generation-two reuse reparses, reloads, or prepares carrier state",
    )
    execute_transaction = runner[
        runner.find("def execute_transaction("):runner.find("def run_success_suite(")
    ]
    require(
        execute_transaction.startswith("def execute_transaction(")
        and "if reuse:" in execute_transaction
        and "issue(device, CMD_BEGIN_REUSE)" in execute_transaction
        and "issue(device, CMD_LEASE)" in execute_transaction
        and "else:\n        initial_prepare(device, transaction)" in execute_transaction
        and "CMD_PREPARE" not in execute_transaction,
        "qtest reuse path issues a second PREPARE or bypasses BEGIN_REUSE",
    )
    require(
        'DEFINE_PROP_UINT32("test-fault-mode"' in device
        and "fault_mode, FAULT_NONE" in device
        and result["success"]["abi"]["fault_mode"] == 0,
        "test fault injection is absent or enabled on the accepted path",
    )
    require(
        installer.count("config PHASE_QEMU_V1") == 1
        and installer.count("CONFIG_PHASE_QEMU_V1") == 1
        and installer.count("files('phase-qemu-v1.c')") == 1,
        "installer does not pin exactly one V1 Kconfig and meson declaration",
    )
    require(
        not any(token in device.lower() for token in ("hmac", "signature", "nonce"))
        and result["claim_limits"]["authenticated_custody"] is False
        and "nominal guest command tags" in contract_flat,
        "direct-process equality tags were promoted to authenticated custody",
    )
    for statement in (
        "ordinary deterministic software",
        "remains inside the M257 forward-shadow domain",
        "not evidence of physical bosons, phonons, QND measurement, restoration,",
        "computational advantage or Small Wall crossing",
    ):
        require(statement in contract_flat,
                f"contract lacks strict ceiling: {statement}")
    for statement in (
        "V1 does not execute a migration test",
        "This source-local guard is not a controller-visible no-smuggle result",
        "does not inherit V0's positive migration evidence",
    ):
        require(statement in contract_flat,
                f"migration scope was promoted beyond source audit: {statement}")
    require(
        not list((package / "evidence").glob("*MIGRATION*")),
        "V1 migration evidence exists but this qualifier only classifies source",
    )


def qualify_semantics(result: dict[str, Any], oracle: dict[str, Any]) -> None:
    require(result["schema"] == "phase-qemu-v1-qtest-result-v1",
            "unexpected qtest schema")
    require(oracle["schema"] == "phase-qemu-v1-separate-reference-v1",
            "unexpected separate-reference schema")
    success = result["success"]
    primary = success["primary"]
    reuse = success["descriptor_distinct_reuse"]
    plus = success["valid_plus_one_selector"]
    held_out = success["held_out_public_descriptor"]
    oracle_primary = oracle["primary"]
    oracle_reuse = oracle["descriptor_distinct_reuse"]
    oracle_held_out = oracle["held_out_public_descriptor"]

    comparisons = (
        (primary, oracle_primary["pointer_transaction"],
         oracle_primary["boundary"]),
        (reuse, oracle_reuse["pointer_transaction"],
         oracle_reuse["boundary"]),
        (plus, oracle_primary["valid_plus_one_pointer_transaction"],
         oracle_primary["valid_plus_one_boundary"]),
        (held_out, oracle_held_out["pointer_transaction"], None),
    )
    for production, pointer_transaction, boundary in comparisons:
        require(
            production["boundary_parity_bit"]
            == pointer_transaction["boundary_parity_bit"],
            "production/reference parity-bit mismatch",
        )
        require(
            production["boundary_pointer_z"]
            == pointer_transaction["boundary_pointer_z"],
            "production/reference pointer-Z mismatch",
        )
        if boundary is not None:
            require(production["boundary_pointer_z"] == boundary["pointer_z"],
                    "production/reference boundary mismatch")
        require(
            pointer_transaction["factorized"]
            and pointer_transaction["pointer_cleared_before_inverse"]
            and pointer_transaction["no_retention_roundtrip_restores"]
            and pointer_transaction["boundary_retained_through_inverse"]
            and pointer_transaction["retained_copy_carrier_fidelity"] == "1",
            "independent pointer latch/unlatch/restoration law failed",
        )
        require(
            production["canonical"] and production["restored"]
            and production["response_ready"] and production["pointer_clear"]
            and production["status"] & (1 << 3)
            and production["status"] & (1 << 7)
            and production["status"] & (1 << 10)
            and not production["spent"] and not production["snapshot_lineage"],
            "accepted production transaction lacks source-isolated restored state",
        )

    require(
        primary["generation"] == primary["restoration_generation"] == 1
        and reuse["generation"] == reuse["restoration_generation"] == 2
        and success["same_qemu_process_primary_and_reuse"]
        and success["snapshot_command_rejected_without_machine_state_change"]
        and success["qtest_reset_commands_issued"] == 0
        and success["same_device_primary_reuse_phase_command_counts"]
        == {
            "lease": 2,
            "prepare": 1,
            "isolate_source": 1,
            "seal_descriptor": 2,
            "execute_atomic": 2,
            "begin_reuse": 1,
            "snapshot_reject_control": 1,
        },
        "generation-two actual-backing reuse or snapshot denial failed",
    )
    require(
        "reset_shams" not in success
        and success["abi"]["fault_mode"] == 0
        and success["abi"]["capabilities"] & (1 << 10) == 0,
        "accepted path used reset recreation or active fault injection",
    )
    require(primary["descriptor_fingerprint"] != reuse["descriptor_fingerprint"],
            "reuse did not consume a descriptor-distinct program")
    require(all(value is True for value in result["controls"].values()),
            "one MMIO/custody control failed")
    require(all_control_values_pass(oracle["controls"]),
            "one separate-reference control failed")

    sham = result["kerr_disabled_sham"]
    sham_reference = oracle_primary["kerr_disabled_pointer_transaction"]
    sham_reuse = sham["same_carrier_generation2_reuse_after_result_free_unwind"]
    require(
        sham["pointer_entanglement_rejected"] and sham["boundary_locked"]
        and sham["carrier_restored_after_result_free_unwind"]
        and sham["canonical_after_result_free_unwind"] and not sham["spent"]
        and not sham_reference["factorized"]
        and sham_reference["pointer_cleared_before_inverse"]
        and sham_reference["no_retention_roundtrip_restores"]
        and not sham_reference["boundary_retained_through_inverse"],
        "Kerr-disabled entangled-pointer sham law failed",
    )
    require(
        sham_reuse == {
            "boundary_parity_bit": 1,
            "generation": 2,
            "restoration_generation": 2,
            "canonical": True,
            "restored": True,
            "pointer_clear": True,
            "response_ready": True,
            "snapshot_lineage": False,
            "resources": {
                "coefficient_cells": 20,
                "scratch_cells": 20,
                "forward_gates": 10,
                "inverse_gates": 10,
                "pointer_gates": 4,
                "postcanonical_resident_peak_coefficient_signed_bits": 2,
                "virtual_cycles": 24,
            },
        },
        "result-free unwind did not support same-carrier generation-two reuse",
    )
    for fault in result["inverse_fault_controls"].values():
        require(
            fault["restoration_failure_detected"] and fault["boundary_locked"]
            and fault["spent"] and not fault["restored"]
            and not fault["canonical"] and fault["pointer_clear"]
            and not fault["response_ready"]
            and fault["restoration_generation"] == 0,
            "inverse fault released a response or acquired restoration",
        )

    primary_resources = primary["resources"]
    reuse_resources = reuse["resources"]
    require(
        primary_resources == {
            "coefficient_cells": 20,
            "scratch_cells": 20,
            "forward_gates": 5,
            "inverse_gates": 5,
            "pointer_gates": 2,
            "postcanonical_resident_peak_coefficient_signed_bits": 2,
            "virtual_cycles": 12,
        }
        and reuse_resources == {
            "coefficient_cells": 20,
            "scratch_cells": 20,
            "forward_gates": 10,
            "inverse_gates": 10,
            "pointer_gates": 4,
            "postcanonical_resident_peak_coefficient_signed_bits": 2,
            "virtual_cycles": 24,
        },
        "accepted resource receipts differ from the bounded device law",
    )
    resource = result["resource_law"]
    require(
        resource["carrier_pointer_coefficient_cells"] == 20
        and resource["gate_scratch_coefficient_cells"] == 20
        and resource["coefficient_cell_counts_are_allocated_backing_counts"]
        and resource["denominator_power_scalar_cells"] == 1
        and resource["retained_private_boundary_bit_cells_during_inverse"] == 1
        and resource["public_descriptor_allocated_uint32_cells"] == 16
        and resource["public_descriptor_gate_cells_primary"] == 5
        and resource["postcanonical_resident_peak_coefficient_width_instrumented"]
        and not resource["transient_and_whole_process_live_payload_instrumented"]
        and not resource[
            "canonicalization_factorization_norm_and_verification_work_instrumented"
        ]
        and resource["retained_dynamic_inverse_history_cells"] == 0
        and resource["accepted_inverse_compiled_from_public_forward_descriptor"]
        and resource["migration_state_is_trusted_backend_sham_not_controller_boundary"]
        and not resource["whole_process_memory_and_qemu_runtime_state_accounted"]
        and not resource["mmio_traffic_count_instrumented"]
        and not resource["physical_energy_latency_noise_and_control_precision_modeled"],
        "resource receipt overstates allocated cells or resident coefficient width",
    )
    require(resource["resource_measurement_verification_level"] == "PACKAGE_SELF_REVIEW",
            "bounded resource counters were promoted beyond package self-review")
    require(result["qemu_stdout_stderr_no_smuggle"] is True,
            "QEMU stdout/stderr no-smuggle control failed")
    forbidden_result_keys = {
        "amplitudes", "carrier_amplitudes", "pointer_amplitudes",
        "coefficient_vector", "occupation_vector", "hidden_state",
        "inverse_descriptor", "expected_output",
    }

    def walk_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(walk_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(walk_keys(item) for item in value))
        return set()

    require(not (walk_keys(result) & forbidden_result_keys),
            "production receipt smuggles hidden carrier or answer state")
    require(result["restoration_classification"] == "EXACT_ALGEBRAIC_RESTORATION",
            "accepted restoration classification changed")
    require(
        result["claim_ceiling"]
        == "DETERMINISTIC_EXACT_IDEAL_QEMU_4_MODE_2_BOSON_BACKEND_ONLY"
        and oracle["verification_classification"]
        == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        and oracle["verification_level"] == "SEPARATE_REFERENCE_PARITY"
        and oracle["restoration_classification"] == "EXACT_ALGEBRAIC_RESTORATION"
        and not any(result["claim_limits"].values())
        and not any(oracle["claim_limits"].values()),
        "physical, advantage, M257, Small Wall, or unbounded ceiling was promoted",
    )


def qualify_receipt(
    package: Path,
    qemu: Path,
    sealed_result: Path,
    sealed_reference: Path,
    result: dict[str, Any],
) -> None:
    receipt_path = package / "evidence" / "PHASE_QEMU_V1_BUILD_RECEIPT.json"
    receipt = read_json(receipt_path)
    require(receipt["schema"] == "PHASE_QEMU_V1_BUILD_RECEIPT_V1",
            "unexpected build-receipt schema")
    require(receipt["qemu"]["version"] == "10.2.4",
            "build receipt is not pinned to QEMU 10.2.4")
    require(
        receipt["qemu"]["source_archive_sha256"]
        == "821b545b92f165e57dddccac5077d76d4d436a226595b8813ad59306bbfd0746",
        "QEMU source archive hash differs from the pinned 10.2.4 source",
    )
    binary_hash = sha256(qemu)
    require(binary_hash == result["qemu_binary_sha256"],
            "QEMU binary differs from regenerated/sealed evidence")
    require(binary_hash == receipt["qemu_binary_sha256"],
            "QEMU binary differs from the build receipt")
    required_dependencies = {
        "apply_to_qemu.py",
        "qemu/hw/misc/phase-qemu-v1.c",
        "tests/phase_qemu_v1_separate_reference.py",
        "tests/qualify_phase_qemu_v1.py",
        "tests/run_phase_qemu_v1_qtest.py",
    }
    dependencies = receipt["source_dependencies"]
    require(required_dependencies <= set(dependencies),
            "build receipt omits a required V1 source dependency")
    for relative, expected in dependencies.items():
        source = package / relative
        require(source.is_file(), f"build dependency is missing: {relative}")
        require(sha256(source) == expected,
                f"build dependency differs from receipt: {relative}")
    device = package / "qemu" / "hw" / "misc" / "phase-qemu-v1.c"
    require(sha256(device) == receipt["phase_qemu_device_source_sha256"],
            "device source differs from the build receipt")
    require(
        sha256(device) == receipt["installed_device_source_sha256"],
        "installed QEMU device source differs from the package source",
    )
    require(
        receipt["installed_kconfig_phase_qemu_v1_entry_count"] == 1
        and receipt["installed_meson_phase_qemu_v1_entry_count"] == 1,
        "installed QEMU tree does not contain exactly one V1 build entry",
    )
    require(sha256(sealed_result) == receipt["sealed_qtest_result_sha256"],
            "qtest evidence hash differs from the build receipt")
    require(
        sha256(sealed_reference) == receipt["sealed_separate_reference_sha256"],
        "separate-reference hash differs from the build receipt",
    )
    for key in (
        "system_install_performed", "system_qemu_modified",
        "additional_ags_worktree_or_clone_created", "ram_backed_scratch_used",
        "physical_hardware_accessed",
    ):
        require(receipt[key] is False, f"forbidden build side effect recorded: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent.parent
    qemu = args.qemu.resolve()
    require(qemu.is_file() and os.access(qemu, os.X_OK),
            "--qemu must name an executable QEMU binary")
    scratch = require_managed_disk_scratch(args.scratch)
    run_root = scratch / f"phase-qemu-v1-qualifier-{uuid.uuid4().hex}"
    run_root.mkdir(parents=False, exist_ok=False)
    regenerated_qtest = run_root / "PHASE_QEMU_V1_QTEST_RESULT.json"
    regenerated_reference = run_root / "PHASE_QEMU_V1_SEPARATE_REFERENCE.json"

    run(
        [
            sys.executable, "-B", "tests/run_phase_qemu_v1_qtest.py",
            "--qemu", str(qemu), "--scratch", str(run_root / "qtest"),
            "--output", str(regenerated_qtest),
        ],
        package,
    )
    run(
        [
            sys.executable, "-B", "tests/phase_qemu_v1_separate_reference.py",
            "--output", str(regenerated_reference),
        ],
        package,
    )

    sealed_qtest = package / "evidence" / "PHASE_QEMU_V1_QTEST_RESULT.json"
    sealed_reference = (
        package / "evidence" / "PHASE_QEMU_V1_SEPARATE_REFERENCE.json"
    )
    require(regenerated_qtest.read_bytes() == sealed_qtest.read_bytes(),
            "regenerated qtest evidence differs byte-for-byte from the seal")
    require(regenerated_reference.read_bytes() == sealed_reference.read_bytes(),
            "regenerated separate reference differs byte-for-byte from the seal")
    result = read_json(sealed_qtest)
    oracle = read_json(sealed_reference)
    qualify_semantics(result, oracle)
    qualify_source(package, result)
    qualify_receipt(package, qemu, sealed_qtest, sealed_reference, result)
    print(
        "PASS_STRICT_SCOPE PHASE_QEMU_V1_EXACT_BOUNDED_BACKEND "
        "MIGRATION_LINEAGE=SOURCE_AUDITED_PACKAGE_LOCAL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
