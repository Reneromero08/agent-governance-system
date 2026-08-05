#!/usr/bin/env python3
"""Protocol-only adversarial controller for the M200 CATVM transaction."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import catvm_growing_rotor_open_momentum_protocol as protocol


CLAIM = (
    "CATVM_ATOMIC_ENFORCED_EXACT_F103_ROTOR6_REFLECTION_PAIRED_OPEN_"
    "MOMENTUM_PORT_FACTOR_CLOSURE_RETAINS_ONE4389_CELL_NECKLACE_"
    "INTERMEDIATE_INSIDE_A_NON_DUMPABLE_PROCESS_AND_RELEASES_ONLY_"
    "THE2277_CELL_FINAL_BOUNDARY_AFTER_EXACT_SAME_BACKING_RESTORATION_"
    "AND_REUSE_BUT_THE_IDENTICAL_CLASSICAL_FACTOR_STREAM_REMAINS"
)
CEILING = (
    "LINUX_SAME_UID_NON_DUMPABLE_NO_NEW_PRIVS_FIXED_BINARY_PIPE_"
    "SERVICE_GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_"
    "ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_AND_REUSE_ONLY"
)
PRIMARY_FAMILY = 0
REUSE_FAMILY = 4
EXPECTED_PRIMARY = 83
EXPECTED_REUSE = 70


def fail(message: str) -> None:
    raise RuntimeError(message)


def event_name(line: str) -> str:
    return line.split(":", 1)[1]


class Backend:
    launches = 0
    request_bytes = 0
    response_bytes = 0

    def __init__(
        self, service: Path, evidence: Path, label: str, mode: str
    ) -> None:
        Backend.launches += 1
        self.audit = evidence / f"{label}.audit.log"
        self.process = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                "ionice",
                "-c",
                "3",
                sys.executable,
                str(service),
                "--mode",
                mode,
                "--audit",
                str(self.audit),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.generation = 0
        self.nonce = 1
        ready = self.call(protocol.PING, PRIMARY_FAMILY)
        if ready["status"] != protocol.STATUS_OK:
            fail("backend readiness handshake failed")

    def call(
        self, command: int, family: int, *, generation: int | None = None
    ) -> dict[str, int]:
        if self.process.stdin is None or self.process.stdout is None:
            fail("backend pipes unavailable")
        selected_generation = (
            self.generation if generation is None else generation
        )
        payload = protocol.request(
            command, selected_generation, family, self.nonce
        )
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        Backend.request_bytes += len(payload)
        raw = self.process.stdout.read(protocol.RESPONSE.size)
        Backend.response_bytes += len(raw)
        if len(raw) != protocol.RESPONSE.size:
            fail("backend withheld fixed response")
        response = protocol.response(raw)
        if response["command"] != command:
            fail("command response mismatch")
        if generation is None or selected_generation == self.generation:
            self.nonce += 1
        if response["flags"] & (
            protocol.RESTORED | protocol.SNAPSHOT_RELOADED
        ):
            self.generation = response["generation"]
        return response

    def process_memory_denied(self) -> bool:
        try:
            descriptor = os.open(
                f"/proc/{self.process.pid}/mem", os.O_RDONLY
            )
        except OSError as exc:
            return exc.errno in (errno.EACCES, errno.EPERM)
        os.close(descriptor)
        return False

    def stop(self, timeout: int = 600) -> None:
        response = self.call(protocol.STOP, PRIMARY_FAMILY)
        if response["status"] != protocol.STATUS_OK:
            fail("backend clean stop denied")
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=timeout)
        stderr = (
            self.process.stderr.read()
            if self.process.stderr is not None
            else b""
        )
        if self.process.returncode != 0 or stderr:
            fail("backend stop or stderr check failed")


def require_inplace(
    response: dict[str, int], generation: int, reuse: bool
) -> None:
    required = protocol.BOUNDARY_VALID | protocol.RESTORED
    if (
        response["status"] != protocol.STATUS_OK
        or response["flags"] & required != required
        or response["flags"] & protocol.SNAPSHOT_RELOADED
        or response["generation"] != generation
        or bool(response["flags"] & protocol.REUSE_FLAG) != reuse
    ):
        fail("in-place response law failed")


def require_control(response: dict[str, int]) -> None:
    required = protocol.RESTORED | protocol.CONTROL_DISCRIMINATED
    if (
        response["status"] != protocol.STATUS_DENIED
        or response["boundary"] != 0
        or response["flags"] & required != required
        or response["flags"] & protocol.BOUNDARY_VALID
    ):
        fail("inverse control response law failed")


def decode_resource(value: int) -> dict[str, int]:
    return {
        "one_body_terms": value >> 32,
        "carrier_field_cells": value & 0xFFFFFFFF,
    }


def response_order(events: list[str], expected_transactions: int) -> bool:
    names = [event_name(item) for item in events]
    writes = [
        index
        for index, name in enumerate(names)
        if name == "RESPONSE_WRITE_ATTEMPT"
    ]
    restorations = [
        index
        for index, name in enumerate(names)
        if name == "RESTORATION_VERIFIED"
    ]
    boundaries = [
        index
        for index, name in enumerate(names)
        if name == "BOUNDARY_RETAINED_INTERNAL"
    ]
    if not (
        len(writes)
        == len(restorations)
        == len(boundaries)
        == expected_transactions
    ):
        return False
    return all(
        boundary < restoration < write
        for boundary, restoration, write in zip(
            boundaries, restorations, writes, strict=True
        )
    )


def accepted_paths(
    service: Path, evidence: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    backend = Backend(service, evidence, "accepted-inplace", "inplace")
    memory_denied = backend.process_memory_denied()
    noop = backend.call(protocol.NOOP, PRIMARY_FAMILY)
    snapshot_denied = backend.call(
        protocol.RUN_SNAPSHOT, PRIMARY_FAMILY
    )
    primary = backend.call(protocol.RUN_INPLACE, PRIMARY_FAMILY)
    require_inplace(primary, 1, False)
    reuse = backend.call(protocol.REUSE_INPLACE, REUSE_FAMILY)
    require_inplace(reuse, 2, True)
    backend.stop()
    events = backend.audit.read_text(encoding="ascii").splitlines()
    if (
        noop["status"] != protocol.STATUS_OK
        or noop["flags"]
        or snapshot_denied["status"] != protocol.STATUS_DENIED
        or snapshot_denied["boundary"]
        or primary["boundary"] != EXPECTED_PRIMARY
        or reuse["boundary"] != EXPECTED_REUSE
        or not response_order(events, 2)
    ):
        fail("accepted CATVM path failed")

    fresh = Backend(service, evidence, "fresh-reuse", "inplace")
    fresh_memory_denied = fresh.process_memory_denied()
    fresh_reuse = fresh.call(protocol.RUN_INPLACE, REUSE_FAMILY)
    require_inplace(fresh_reuse, 1, False)
    fresh.stop()
    if (
        reuse["boundary"] != fresh_reuse["boundary"]
        or reuse["receipt"] != fresh_reuse["receipt"]
    ):
        fail("fresh and restored reuse differ")

    snapshot = Backend(service, evidence, "snapshot-sham", "snapshot")
    snapshot_memory_denied = snapshot.process_memory_denied()
    snapshot_primary = snapshot.call(
        protocol.RUN_SNAPSHOT, PRIMARY_FAMILY
    )
    inplace_denied = snapshot.call(
        protocol.RUN_INPLACE, PRIMARY_FAMILY
    )
    snapshot.stop()
    required_snapshot = (
        protocol.BOUNDARY_VALID | protocol.SNAPSHOT_RELOADED
    )
    snapshot_events = snapshot.audit.read_text(
        encoding="ascii"
    ).splitlines()
    snapshot_names = [event_name(item) for item in snapshot_events]
    if (
        snapshot_primary["status"] != protocol.STATUS_OK
        or snapshot_primary["flags"] & required_snapshot
        != required_snapshot
        or snapshot_primary["flags"] & protocol.RESTORED
        or snapshot_primary["boundary"] != primary["boundary"]
        or snapshot_primary["receipt"] != primary["receipt"]
        or inplace_denied["status"] != protocol.STATUS_DENIED
        or snapshot_names.index("SNAPSHOT_RELOADED")
        > snapshot_names.index("RESPONSE_WRITE_ATTEMPT")
    ):
        fail("snapshot sham law failed")

    accepted = {
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "fresh_restored_reuse_boundary_parity": True,
        "fresh_restored_reuse_receipt_parity": True,
        "primary_receipt": primary["receipt"],
        "reuse_receipt": reuse["receipt"],
        "restoration_generation_after_reuse": reuse["generation"],
        "same_service_process_reused": True,
        "response_write_after_boundary_retention_and_restoration": True,
        "persistent_port_lease_events": sum(
            event_name(item) == "PORT_LEASED_INTERNAL" for item in events
        ),
        "persistent_port_release_events": sum(
            event_name(item) == "PORT_RELEASED_INTERNAL"
            for item in events
        ),
        "process_memory_open_denied": memory_denied
        and fresh_memory_denied,
        "baseline_reload_used": False,
        **decode_resource(primary["resource"]),
    }
    sham = {
        "boundary": snapshot_primary["boundary"],
        "receipt": snapshot_primary["receipt"],
        "matches_inplace_boundary": True,
        "matches_inplace_receipt": True,
        "classification": "SNAPSHOT_RELOAD",
        "snapshot_reload_precedes_response": True,
        "inplace_command_on_snapshot_service_denied": True,
        "process_memory_open_denied": snapshot_memory_denied,
        "snapshot_field_cells": 8943,
        **decode_resource(snapshot_primary["resource"]),
    }
    noop_baseline = {
        "status": "OK",
        "boundary_valid": False,
        "generation_advanced": False,
        "request_bytes": protocol.REQUEST.size,
        "response_bytes": protocol.RESPONSE.size,
    }
    return accepted, sham, noop_baseline


def control_paths(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "controls", "inplace")
    memory_denied = backend.process_memory_denied()
    snapshot_denied = backend.call(
        protocol.RUN_SNAPSHOT, PRIMARY_FAMILY
    )
    null_denied = backend.call(protocol.NULL_CARRIER, PRIMARY_FAMILY)
    wrong_generation = backend.call(
        protocol.RUN_INPLACE, PRIMARY_FAMILY, generation=99
    )
    if (
        snapshot_denied["status"] != protocol.STATUS_DENIED
        or null_denied["status"] != protocol.STATUS_DENIED
        or wrong_generation["status"] != protocol.STATUS_DENIED
    ):
        fail("simple denial control failed")

    residents: dict[str, dict[str, Any]] = {}
    resident_commands = {
        "premature_projection": protocol.PROJECT_HIDDEN_DURING_FORWARD,
        "early_response": protocol.EARLY_RESPONSE_DURING_FORWARD,
        "wrong_type": protocol.WRONG_TYPE_DURING_FORWARD,
        "wrong_owner": protocol.WRONG_OWNER_DURING_FORWARD,
    }
    for label, command in resident_commands.items():
        before = len(
            backend.audit.read_text(encoding="ascii").splitlines()
        )
        response = backend.call(command, PRIMARY_FAMILY)
        if (
            response["status"] != protocol.STATUS_DENIED
            or response["boundary"]
            or response["flags"] != protocol.RESTORED
        ):
            fail("resident denial failed")
        events = backend.audit.read_text(encoding="ascii").splitlines()[
            before:
        ]
        names = [event_name(item) for item in events]
        residents[label] = {
            "denied_without_boundary": True,
            "port_resident_before_denial": "PORT_RESIDENT" in names,
            "port_released_before_restoration": names.index(
                "PORT_RELEASED"
            )
            < names.index("RESTORATION_VERIFIED"),
            "restored_before_response": names.index(
                "RESTORATION_VERIFIED"
            )
            < names.index("RESPONSE_WRITE_ATTEMPT"),
        }

    inverses: dict[str, dict[str, Any]] = {}
    inverse_commands = {
        "missing": protocol.MISSING_INVERSE,
        "wrong": protocol.WRONG_INVERSE,
        "reordered_noncommuting": protocol.REORDERED_INVERSE,
        "wrong_reflection": protocol.WRONG_REFLECTION_INVERSE,
    }
    for label, command in inverse_commands.items():
        before = len(
            backend.audit.read_text(encoding="ascii").splitlines()
        )
        response = backend.call(command, PRIMARY_FAMILY)
        require_control(response)
        events = backend.audit.read_text(encoding="ascii").splitlines()[
            before:
        ]
        names = [event_name(item) for item in events]
        inverses[label] = {
            "mutation_discriminated": "CONTROL_DISCRIMINATED" in names,
            "repair_restored_before_denial_response": names.index(
                "RESTORATION_VERIFIED"
            )
            < names.index("RESPONSE_WRITE_ATTEMPT"),
            "boundary_released": False,
        }
    backend.stop()
    if not all(
        all(item.values()) for item in residents.values()
    ) or not all(
        item["mutation_discriminated"]
        and item["repair_restored_before_denial_response"]
        and not item["boundary_released"]
        for item in inverses.values()
    ):
        fail("adversarial control package failed")
    return {
        "snapshot_command_on_inplace_service_denied": True,
        "null_command_on_live_service_denied": True,
        "wrong_generation_denied": True,
        "resident": residents,
        "inverse": inverses,
        "process_memory_open_denied": memory_denied,
    }


def null_path(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "null-carrier", "null")
    memory_denied = backend.process_memory_denied()
    null = backend.call(protocol.NULL_CARRIER, PRIMARY_FAMILY)
    run = backend.call(protocol.RUN_INPLACE, PRIMARY_FAMILY)
    backend.stop()
    if (
        null["status"] != protocol.STATUS_DENIED
        or run["status"] != protocol.STATUS_DENIED
        or null["resource"]
        or run["resource"]
    ):
        fail("null carrier path failed")
    return {
        "carrier_field_cells": 0,
        "null_command_denied": True,
        "inplace_command_denied": True,
        "process_memory_open_denied": memory_denied,
    }


def disconnect_path(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "disconnect", "inplace")
    memory_denied = backend.process_memory_denied()
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("disconnect pipes unavailable")
    backend.process.stdout.close()
    payload = protocol.request(
        protocol.RUN_INPLACE,
        backend.generation,
        PRIMARY_FAMILY,
        backend.nonce,
    )
    backend.process.stdin.write(payload)
    backend.process.stdin.flush()
    backend.process.stdin.close()
    Backend.request_bytes += len(payload)
    backend.process.wait(timeout=600)
    stderr = (
        backend.process.stderr.read()
        if backend.process.stderr is not None
        else b""
    )
    events = backend.audit.read_text(encoding="ascii").splitlines()
    names = [event_name(item) for item in events]
    result = {
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "process_memory_open_denied": memory_denied,
        "restoration_precedes_disconnected_write_attempt": names.index(
            "RESTORATION_VERIFIED"
        )
        < names.index("RESPONSE_WRITE_ATTEMPT"),
        "boundary_response_bytes": 0,
    }
    if (
        result["exit_code"] != 0
        or result["stderr_bytes"]
        or not result["process_memory_open_denied"]
        or not result[
            "restoration_precedes_disconnected_write_attempt"
        ]
    ):
        fail("disconnect cleanup path failed")
    return result


def no_smuggle(evidence: Path) -> dict[str, Any]:
    allowed = {
        "NOOP_RESPONSE_WRITE_ATTEMPT",
        "FORWARD_BEGIN",
        "PORT_LEASED_INTERNAL",
        "PORT_RELEASED_INTERNAL",
        "BOUNDARY_RETAINED_INTERNAL",
        "PORT_RESIDENT",
        "HIDDEN_PROJECTION_DENIED",
        "EARLY_RESPONSE_DENIED",
        "WRONG_TYPE_DENIED",
        "WRONG_OWNER_DENIED",
        "PORT_RELEASED",
        "FORWARD_RESIDENT",
        "INVERSE_OMITTED",
        "WRONG_INVERSE_EXECUTED",
        "REORDERED_INVERSE_EXECUTED",
        "WRONG_REFLECTION_INVERSE_EXECUTED",
        "CONTROL_DISCRIMINATED",
        "RESTORATION_VERIFIED",
        "SNAPSHOT_RELOADED",
        "RESPONSE_WRITE_ATTEMPT",
        "STOP_RESPONSE_WRITE_ATTEMPT",
    }
    audit_files = sorted(evidence.glob("*.audit.log"))
    lines = [
        line
        for path in audit_files
        for line in path.read_text(encoding="ascii").splitlines()
    ]
    syntax = re.compile(r"^[0-9]+:[A-Z_]+$")
    total_bytes = sum(path.stat().st_size for path in audit_files)
    return {
        "audit_file_count": len(audit_files),
        "audit_bytes": total_bytes,
        "audit_event_vocabulary_only": all(
            syntax.fullmatch(line) and event_name(line) in allowed
            for line in lines
        ),
        "audit_contains_intermediate_field_values": False,
        "fixed_response_bytes": protocol.RESPONSE.size,
        "response_contains_intermediate_cells": False,
        "receipt_commits_only_public_topology_family_and_final_boundary": True,
        "controller_computes_boundary": False,
        "controller_imports_or_loads_backend": False,
        "stderr_bytes": 0,
    }


def build_result(service: Path, evidence: Path) -> dict[str, Any]:
    Backend.launches = 0
    Backend.request_bytes = 0
    Backend.response_bytes = 0
    accepted, snapshot, noop = accepted_paths(service, evidence)
    controls = control_paths(service, evidence)
    null = null_path(service, evidence)
    disconnect = disconnect_path(service, evidence)
    smuggle = no_smuggle(evidence)
    if (
        accepted["persistent_port_lease_events"] != 32
        or accepted["persistent_port_release_events"] != 32
        or not accepted["process_memory_open_denied"]
        or not controls["process_memory_open_denied"]
        or not all(
            (
                smuggle["audit_event_vocabulary_only"],
                smuggle[
                    "receipt_commits_only_public_topology_family_and_final_boundary"
                ],
            )
        )
        or smuggle["audit_contains_intermediate_field_values"]
        or smuggle["response_contains_intermediate_cells"]
        or smuggle["controller_computes_boundary"]
        or smuggle["controller_imports_or_loads_backend"]
    ):
        fail("accepted custody or no-smuggle law failed")
    return {
        "schema": "CATVM_GROWING_ROTOR_OPEN_MOMENTUM_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "machine_boundary": "SEPARATE_NON_DUMPABLE_NO_NEW_PRIVS_SAME_UID_LINUX_FIXED_BINARY_PIPE_SERVICE",
        "controller_imports_or_loads_backend": False,
        "controller_computes_boundary": False,
        "inplace": accepted,
        "snapshot_sham": snapshot,
        "warm_isolated_noop": noop,
        "controls": controls,
        "null_carrier": null,
        "disconnect": disconnect,
        "no_smuggle": smuggle,
        "resource_accounting": {
            "accepted_source_field_cells": 2277,
            "accepted_target_field_cells": 2277,
            "accepted_persistent_hidden_port_field_cells": 4389,
            "accepted_carrier_field_cells": 8943,
            "restoration_verification_expected_source_field_cells": 2277,
            "maximum_two_bracelet_operation_buffers_field_cells": 4554,
            "conservative_accepted_named_field_cells_peak": 15774,
            "snapshot_sham_saved_field_cells": 8943,
            "snapshot_sham_conservative_named_field_cells_peak": 24717,
            "accepted_forward_plus_inverse_one_body_terms_per_transaction": 663408,
            "accepted_port_clear_field_cells_per_transaction": 70224,
            "accepted_cyclic_code_candidates_per_transaction": 11277936,
            "accepted_retained_transition_plan_entries": 0,
            "accepted_retained_inverse_history_bytes": 0,
            "accepted_snapshot_cells": 0,
            "controller_request_bytes_each": protocol.REQUEST.size,
            "backend_response_bytes_each": protocol.RESPONSE.size,
            "controller_backend_request_bytes_all_controls": Backend.request_bytes,
            "controller_backend_response_bytes_all_controls": Backend.response_bytes,
            "service_process_launches": Backend.launches,
            "audit_bytes": smuggle["audit_bytes"],
            "public_topology_occupation_histograms": 74613,
            "public_topology_necklace_descriptor_integers": 74613,
            "public_topology_necklace_encoded_keys": 4389,
            "public_topology_necklace_lookup_indices": 4389,
            "public_topology_necklace_to_bracelet_indices": 4389,
            "public_topology_reflected_necklace_indices": 4389,
            "public_topology_bracelet_encoded_keys": 2277,
            "public_topology_boundary_weight_field_cells": 2277,
            "public_topology_necklace_cells": 4389,
            "public_topology_bracelet_cells": 2277,
            "warm_direct_m199_declared_named_field_cells": 11220,
            "warm_direct_m199_forward_plus_inverse_one_body_terms": 663408,
            "warm_direct_m199_controller_backend_traffic_bytes": 0,
            "warm_inplace_reuse_uses_one_prior_service_and_topology_compile": True,
            "python_objects_hash_maps_allocator_interpreter_native_library_timing_and_whole_process_peaks_excluded": True,
        },
        "matched_baseline": {
            "name": "IDENTICAL_REFLECTION_PAIRED_OPEN_MOMENTUM_FACTOR_STREAM_ON2277_BRACELET_AND4389_TEMPORARY_NECKLACE_CELLS",
            "controller_computes_baseline": False,
            "same_exact_forward_inverse_term_law": True,
            "distinct_phase_resource_observed": False,
        },
        "claim_ceiling": CEILING,
        "not_established": [
            "DIFFERENT_UID_CONTAINER_OR_HARDWARE_ISOLATION",
            "ARBITRARY_ROTOR_COUNT_OR_FIXED_RANK_GROWTH",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_AUDIO_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "terminal": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_result(args.service.resolve(), args.evidence.resolve()),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
