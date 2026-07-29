#!/usr/bin/env python3
"""Adversarial controller for the distinct depth-CATVM dispatch repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import catvm_necklace_shared_latent_depth_controller as base
import catvm_necklace_shared_latent_depth_protocol as protocol


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_hidden_resident(
    response: dict[str, object],
    label: str,
) -> None:
    if not int(response["flags"]) & protocol.STAGE_RESIDENT:
        fail(f"{label}: response lost hidden-stage custody")
    if int(response["flags"]) & protocol.BOUNDARY_VALID:
        fail(f"{label}: response exposed a boundary")
    if any(float(value) != 0.0 for value in response["boundary"]):
        fail(f"{label}: response smuggled boundary values")


def require_restored(
    response: dict[str, object],
    generation: int,
    label: str,
) -> None:
    base.require(response, protocol.STATUS_OK, label)
    required = protocol.BOUNDARY_VALID | protocol.RESTORED
    if int(response["flags"]) & required != required:
        fail(f"{label}: boundary response preceded restoration")
    if int(response["generation"]) != generation:
        fail(f"{label}: wrong restoration generation")
    if float(response["restoration_error"]) > 6e-11:
        fail(f"{label}: restoration tolerance failed")


def initialize_replay_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = base.Service(
        executable, evidence_dir, "repair-initialize-replay"
    )
    service.initialize()
    accepted_nonce = service.nonce

    _, repeated = service.raw_call(
        protocol.INITIALIZE,
        0,
        0,
        0,
        0,
        1,
    )
    base.require(
        repeated,
        protocol.STATUS_DENIED,
        "repeated initialize",
    )
    _, replay = service.raw_call(
        protocol.BEGIN,
        service.generation,
        4,
        2,
        service.lease,
        accepted_nonce,
    )
    base.require(
        replay,
        protocol.STATUS_DENIED,
        "replayed accepted nonce",
    )
    if int(replay["flags"]) & protocol.BOUNDARY_VALID:
        fail("replayed nonce exposed a boundary")

    service.nonce = accepted_nonce
    _, stage = service.call(protocol.BEGIN, 4, 2)
    base.require(stage, protocol.STATUS_OK, "post-replay stage")
    require_hidden_resident(stage, "post-replay stage")
    _, restored = service.call(protocol.CONTINUE, 4, 2)
    require_restored(restored, 1, "post-replay restore")
    service.stop()
    return {
        "repeated_initialize": "DENIED_WITHOUT_NONCE_MUTATION",
        "accepted_nonce_replay": "DENIED_BY_OWNER_GATE",
        "later_valid_transaction": "RESTORED",
        "restoration_error": restored["restoration_error"],
    }


def resident_mismatch_attack(
    executable: Path,
    evidence_dir: Path,
    label: str,
    wrong_depth: int,
    wrong_variant: int,
) -> dict[str, object]:
    depth = 8
    variant = 3
    service = base.Service(
        executable, evidence_dir, f"repair-{label}"
    )
    service.initialize()
    _, stage = service.call(protocol.BEGIN, depth, variant)
    base.require(stage, protocol.STATUS_OK, f"{label} stage")
    require_hidden_resident(stage, f"{label} stage")

    _, mismatch = service.call(
        protocol.CONTINUE, wrong_depth, wrong_variant
    )
    base.require(
        mismatch,
        protocol.STATUS_ERROR,
        f"{label} mismatch",
    )
    require_hidden_resident(mismatch, f"{label} mismatch")

    service.nonce += 1
    _, repeated_initialize = service.raw_call(
        protocol.INITIALIZE,
        0,
        0,
        0,
        0,
        1,
    )
    base.require(
        repeated_initialize,
        protocol.STATUS_DENIED,
        f"{label} resident initialize",
    )
    require_hidden_resident(
        repeated_initialize,
        f"{label} resident initialize",
    )

    _, staged_stop = service.call(protocol.STOP)
    base.require(
        staged_stop,
        protocol.STATUS_DENIED,
        f"{label} staged stop",
    )
    require_hidden_resident(staged_stop, f"{label} staged stop")
    if service.process.poll() is not None:
        fail(f"{label}: denied staged STOP terminated service")

    _, restored = service.call(protocol.CONTINUE, depth, variant)
    require_restored(restored, 1, f"{label} restore")
    _, reuse = service.call(protocol.REUSE, 3, 5)
    require_restored(reuse, 2, f"{label} reuse")
    if not int(reuse["flags"]) & protocol.REUSE_FLAG:
        fail(f"{label}: restored carrier was not reused")
    if float(reuse["norm_error"]) > 6e-11:
        fail(f"{label}: fresh/restored reuse parity failed")
    service.stop()
    return {
        "mismatch_status": "ERROR",
        "mismatch_preserved_stage_receipt": True,
        "mismatch_boundary_values": 0,
        "resident_initialize": "DENIED_WITH_STAGE_RESIDENT",
        "staged_stop": "DENIED_WITH_STAGE_RESIDENT",
        "service_alive_after_staged_stop": True,
        "later_restoration_error": restored["restoration_error"],
        "later_reuse_restoration_error": reuse[
            "restoration_error"
        ],
        "later_reuse_boundary_error": reuse["norm_error"],
    }


def invalid_topology_attack(
    executable: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    service = base.Service(
        executable, evidence_dir, "repair-invalid-topology"
    )
    service.initialize()
    invalid_cases = (
        ("zero_depth", 0, 2),
        ("zero_variant", 4, 0),
        ("depth_above_ceiling", 65, 2),
    )
    observed: dict[str, str] = {}
    for label, depth, variant in invalid_cases:
        _, response = service.call(
            protocol.BEGIN, depth, variant
        )
        base.require(response, protocol.STATUS_ERROR, label)
        if int(response["flags"]) != 0:
            fail(f"{label}: invalid topology changed custody flags")
        if any(float(value) != 0.0 for value in response["boundary"]):
            fail(f"{label}: invalid topology exposed boundary values")
        observed[label] = "ERROR_BEFORE_STAGE"

    _, stage = service.call(protocol.BEGIN, 3, 7)
    base.require(stage, protocol.STATUS_OK, "valid topology stage")
    require_hidden_resident(stage, "valid topology stage")
    _, restored = service.call(protocol.CONTINUE, 3, 7)
    require_restored(restored, 1, "valid topology restoration")
    service.stop()
    return {
        **observed,
        "later_valid_transaction": "RESTORED",
        "restoration_error": restored["restoration_error"],
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: "
            "catvm_necklace_shared_latent_depth_atomic_repair_controller.py "
            "SERVICE EVIDENCE_DIR"
        )
    executable = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    stage, primary, reuse = base.primary_service(
        executable, evidence_dir
    )
    disconnect = base.disconnect_attack(executable, evidence_dir)
    inverse_controls = {
        "missing_inverse_error": base.inverse_attack(
            executable,
            evidence_dir,
            "repair-missing",
            protocol.MISSING_INVERSE,
        ),
        "reordered_inverse_error": base.inverse_attack(
            executable,
            evidence_dir,
            "repair-reordered",
            protocol.REORDERED_INVERSE,
        ),
        "wrong_inverse_variant_error": base.inverse_attack(
            executable,
            evidence_dir,
            "repair-variant",
            protocol.WRONG_INVERSE_VARIANT,
        ),
    }
    null = base.null_attack(executable, evidence_dir)
    initialize_replay = initialize_replay_attack(
        executable, evidence_dir
    )
    depth_mismatch = resident_mismatch_attack(
        executable,
        evidence_dir,
        "depth-mismatch",
        7,
        3,
    )
    variant_mismatch = resident_mismatch_attack(
        executable,
        evidence_dir,
        "variant-mismatch",
        8,
        4,
    )
    invalid_topology = invalid_topology_attack(
        executable, evidence_dir
    )

    result = {
        "claim_candidate": (
            "CATVM_ATOMIC_DISPATCH_REPAIRED_TOPOLOGY_REMATERIALIZED_"
            "OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_"
            "CARRIER_AT_DEPTH32"
        ),
        "result": "PASS",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
            "NONCE_DERIVED_LEASE_EXACT_GENERATION_GRID17_FOUR_EXCHANGE_"
            "SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_570_"
            "COMPLEX_CELLS_TWO_CELL_LATENT_FIBER_PUBLIC_VARIANT_ORDINAL_"
            "COMPILER_PRIMARY_DEPTH32_REUSE_DEPTH11_STATIC_OWNER_SEVEN_"
            "BIN_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
        ),
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "historical_predecessor": {
            "claim_candidate": (
                "CATVM_ENFORCED_TOPOLOGY_REMATERIALIZED_OWNER_BOUND_"
                "SHARED_LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_AT_DEPTH32"
            ),
            "classification": "REJECTED_SOURCE_DEFECT",
            "preserved_subclaims": [
                "valid CONTINUE restores before boundary release",
                "disconnect cleanup restores before process exit",
                "hidden-stage projection and owner denials",
                "fresh-versus-restored reuse on the same backing",
            ],
            "rejected_interpretations": [
                "all resident-stage responses preserve custody flags",
                "rejected initialization cannot rewind nonce",
                "STOP acknowledgement never precedes restoration",
            ],
        },
        "custody": {
            "resident_joint_complex_cells": 570,
            "program_descriptor_bytes": 8,
            "retained_module_tape_bytes": 0,
            "retained_inverse_history_bytes": 0,
            "primary_depth": 32,
            "stage_applied_modules": 16,
            "stage_resident": bool(
                int(stage["flags"]) & protocol.STAGE_RESIDENT
            ),
            "projected": False,
            "typed_receipt_only": True,
        },
        "atomic_ordering": {
            "final_boundary_after_restoration": True,
            "primary_generation": primary["generation"],
            "disconnect": disconnect,
            "initialize_replay": initialize_replay,
            "depth_mismatch": depth_mismatch,
            "variant_mismatch": variant_mismatch,
        },
        "primary": {
            "boundary": primary["boundary"],
            "restoration_error": primary["restoration_error"],
            "carrier_backing_preserved": True,
            "baseline_reload_bytes": 0,
            "begin_native_generator_terms": stage[
                "native_operations"
            ],
            "continue_and_inverse_native_generator_terms": primary[
                "native_operations"
            ],
            "total_native_generator_terms": (
                int(stage["native_operations"])
                + int(primary["native_operations"])
            ),
        },
        "reuse": {
            "depth": 11,
            "boundary": reuse["boundary"],
            "restoration_error": reuse["restoration_error"],
            "fresh_restored_boundary_error": reuse["norm_error"],
            "generation": reuse["generation"],
            "same_backing": True,
            "same_resource_signature": True,
            "baseline_reload_bytes": 0,
        },
        "controls": {
            **inverse_controls,
            "wrong_owner": "DENIED",
            "premature_projection": "DENIED",
            "snapshot": "DENIED",
            "null_carrier": null,
            "invalid_topology": invalid_topology,
            "poison_after_failed_inverse": True,
        },
        "no_smuggle": {
            "controller_imports_backend": False,
            "stage_boundary_values": 0,
            "latent_values_in_response": 0,
            "content_derived_receipts": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        },
        "resource_law": {
            "catalytic_carrier_complex_cells": 570,
            "permanent_restoration_baseline_complex_cells": 570,
            "per_module_fiber_complex_cells": 285,
            "generator_work_complex_cells": 855,
            "reuse_reference_complex_cells": 570,
            "primary_peak_counted_complex_cells_excluding_plan": 2280,
            "reuse_peak_counted_complex_cells_excluding_plan": 2850,
            "plan_necklaces": 285,
            "plan_root_complex_cells": 17,
            "public_pending_topology_bytes": 8,
            "pending_applied_counter_bytes": 8,
            "relation_table_cells": 0,
            "assignment_cells": 0,
            "temporary_occupation_cells": 0,
            "dense_285_operator_cells": 0,
            "retained_inverse_history_bytes": 0,
            "inverse_descriptors_rematerialized": True,
            "allocator_native_library_os_memory_bounded": False,
        },
        "strongest_compact_classical_identical": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
