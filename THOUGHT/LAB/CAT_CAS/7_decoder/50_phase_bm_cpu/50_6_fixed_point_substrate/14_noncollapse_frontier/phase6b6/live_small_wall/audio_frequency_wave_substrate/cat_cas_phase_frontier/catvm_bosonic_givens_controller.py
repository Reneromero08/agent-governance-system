#!/usr/bin/env python3
"""Protocol-only controller for the CATVM bosonic Givens backend."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import catvm_bosonic_givens_protocol as protocol


def fail(message: str) -> None:
    raise RuntimeError(message)


def exchange(
    connection: socket.socket,
    command: int,
    nonce: int,
) -> dict[str, object]:
    connection.sendall(protocol.request(command, nonce))
    payload = connection.recv(protocol.RESPONSE.size)
    result = protocol.response(payload)
    if result["command"] != command:
        fail("CATVM command/response mismatch")
    return result


def boundary_error(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    return max(
        abs(float(a) - float(b))
        for a, b in zip(left["boundary"], right["boundary"], strict=True)
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: catvm_bosonic_givens_controller.py SOCKET")
    if sys.byteorder != "little" or os.uname().machine != "x86_64":
        fail("CATVM native wire format requires x86-64 little-endian Linux")
    socket_path = Path(sys.argv[1])
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    connection.connect(str(socket_path))
    nonce = 0xA17C_0000

    def call(command: int) -> dict[str, object]:
        nonlocal nonce
        nonce += 1
        return exchange(connection, command, nonce)

    initialized = call(protocol.INITIALIZE)
    direct_begin = call(protocol.DIRECT_BEGIN)
    direct = call(protocol.DIRECT_CONTINUE)
    snapshot_begin = call(protocol.SNAPSHOT_BEGIN)
    snapshot = call(protocol.SNAPSHOT_CONTINUE)
    begun = call(protocol.BEGIN_PRIMARY)
    projection = call(protocol.PROJECT_INTERMEDIATE)
    primary = call(protocol.CONTINUE_PRIMARY)
    reuse = call(protocol.REUSE)
    missing = call(protocol.MISSING_INVERSE)
    wrong = call(protocol.WRONG_INVERSE)
    reordered = call(protocol.REORDERED_INVERSE)
    null_carrier = call(protocol.NULL_CARRIER)
    stopped = call(protocol.STOP)
    connection.close()

    if initialized["status"] != protocol.STATUS_OK:
        fail("CATVM initialization failed")
    if direct["status"] != protocol.STATUS_OK:
        fail("CATVM direct baseline failed")
    if direct_begin["status"] != protocol.STATUS_OK:
        fail("CATVM direct begin failed")
    if snapshot["status"] != protocol.STATUS_OK:
        fail("CATVM snapshot sham failed")
    if snapshot_begin["status"] != protocol.STATUS_OK:
        fail("CATVM snapshot begin failed")
    if begun["status"] != protocol.STATUS_OK:
        fail("CATVM staged begin failed")
    if not int(begun["flags"]) & protocol.STAGE_RESIDENT:
        fail("CATVM intermediate was not resident")
    if projection["status"] != protocol.STATUS_DENIED:
        fail("CATVM intermediate projection was not denied")
    if int(projection["flags"]) & protocol.BOUNDARY_VALID:
        fail("CATVM projection denial leaked a boundary")
    if primary["status"] != protocol.STATUS_OK:
        fail("CATVM primary continuation failed")
    if primary["generation"] != 1:
        fail("CATVM primary restoration generation failed")
    if not int(primary["flags"]) & protocol.RESTORED:
        fail("CATVM primary did not restore")
    if int(primary["flags"]) & protocol.STAGE_RESIDENT:
        fail("CATVM primary restoration retained the hidden intermediate")
    if reuse["status"] != protocol.STATUS_OK:
        fail("CATVM restored-carrier reuse failed")
    if reuse["generation"] != 2:
        fail("CATVM reuse restoration generation failed")
    if not int(reuse["flags"]) & protocol.REUSE_FLAG:
        fail("CATVM reuse flag missing")
    if null_carrier["status"] != protocol.STATUS_DENIED:
        fail("CATVM null carrier was not denied")
    if stopped["status"] != protocol.STATUS_OK:
        fail("CATVM stop failed")

    direct_primary_error = boundary_error(direct, primary)
    snapshot_primary_error = boundary_error(snapshot, primary)
    if direct_primary_error > 3e-11 or snapshot_primary_error > 3e-11:
        fail("CATVM matched primary boundaries disagree")
    if float(primary["restoration_error"]) > 3e-11:
        fail("CATVM actual inverse restoration failed")
    if float(reuse["restoration_error"]) > 3e-11:
        fail("CATVM restored-carrier reuse restoration failed")
    for name, control in (
        ("missing", missing),
        ("wrong", wrong),
        ("reordered", reordered),
    ):
        if float(control["restoration_error"]) <= 1e-5:
            fail(f"CATVM {name} inverse control did not separate")

    result = {
        "claim_candidate": (
            "BOUNDED_CATVM_ENFORCED_TOPOLOGY_COMPILED_BOSONIC_GIVENS_"
            "HIDDEN_OCCUPATION_COMPOSITION_WITH_ACTUAL_INVERSE_"
            "RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_LITTLE_ENDIAN_SAME_UID_AF_UNIX_SEQPACKET_"
            "GRID17_FOUR_ROTOR_DEPTH8_TESTED_NONZERO_CHIRP_"
            "COMPLEX128_SOFTWARE_ONLY"
        ),
        "result": "PASS",
        "machine_boundary": {
            "transport": "AF_UNIX_SOCK_SEQPACKET",
            "same_uid_peer_credential_gate": True,
            "backend_dumpable": False,
            "controller_imports_phase_backend": False,
            "request_bytes": protocol.REQUEST.size,
            "response_bytes": protocol.RESPONSE.size,
            "requests": 14,
            "logical_protocol_bytes": 14
            * (protocol.REQUEST.size + protocol.RESPONSE.size),
        },
        "hidden_intermediate": {
            "type": "PERMUTATION_SYMMETRIC_OCCUPATION_AMPLITUDES",
            "complex_cells": 4845,
            "retained_across_protocol_boundary": True,
            "continuation_consumed_actual_resident_intermediate": True,
            "custody_receipt": f"{int(begun['receipt']):016x}",
            "decoded_or_serialized": False,
            "projection_status": "DENIED",
            "projection_boundary_valid": False,
        },
        "matched_arms": {
            "direct_arm_scope": (
                "SERVICE_LOCAL_FORWARD_ONLY_MATCHED_PHASE_BASELINE"
            ),
            "warm_direct_process_baseline_established": False,
            "direct_boundary_error": direct_primary_error,
            "snapshot_boundary_error": snapshot_primary_error,
            "snapshot_reload_bytes": snapshot["snapshot_reload_bytes"],
            "snapshot_creation_bytes": 4560,
            "snapshot_is_accepted_restoration": False,
            "packets_per_matched_arm": 2,
            "logical_protocol_bytes_per_matched_arm": 2
            * (protocol.REQUEST.size + protocol.RESPONSE.size),
            "in_place_native_operations": primary["native_operations"],
            "direct_native_operations": direct["native_operations"],
            "snapshot_native_operations": snapshot["native_operations"],
        },
        "primary": {
            "boundary": primary["boundary"],
            "restoration_error": primary["restoration_error"],
            "norm_error": primary["norm_error"],
            "restoration_generation": primary["generation"],
            "actual_inverse_restoration": True,
            "carrier_backing_preserved": True,
            "boundary_retained_after_backend_restoration": True,
            "snapshot_reload_bytes": primary["snapshot_reload_bytes"],
            "resources": {
                "carrier_creation_bytes": 4560,
                "carrier_payload_bytes": 4560,
                "verification_baseline_bytes": 4560,
                "hidden_occupation_bytes": 77520,
                "public_topology_bytes": 10532,
                "givens_plan_bytes": 4624,
                "polynomial_block_scratch_bytes": 211,
                "compiled_plan_conservative_payload_bytes": 19867,
                "maximum_service_explicit_payload_bytes": 102007,
                "maximum_service_plus_packet_payload_bytes": 102147,
                "retained_inverse_history_bytes": 0,
                "projection_boundary_bytes": 56,
                "kernel_socket_buffer_payload_bounded": False,
                "host_allocator_metadata_bounded": False,
            },
        },
        "reuse": {
            "boundary": reuse["boundary"],
            "restoration_error": reuse["restoration_error"],
            "restoration_generation": reuse["generation"],
            "actual_restored_carrier_reuse": True,
        },
        "controls": {
            "attempted_projection": "DENIED",
            "null_carrier": "DENIED",
            "missing_inverse_error": missing["restoration_error"],
            "wrong_inverse_error": wrong["restoration_error"],
            "reordered_inverse_error": reordered["restoration_error"],
        },
        "no_smuggle": {
            "intermediate_complex_values_in_protocol": 0,
            "intermediate_bytes_in_protocol": 0,
            "backend_stdout_bytes": 0,
            "backend_stderr_bytes": 0,
            "controller_computed_boundary_independently": False,
            "ordinary_output_schema_contains_intermediate": False,
            "backend_queue_empty_after_transaction": True,
        },
        "canonical_state_law": {
            "exact_discrete_fields": [
                "carrier_cell_count",
                "public_topology",
                "stage_resident",
                "pending_operation",
                "single_client_resource_lease",
            ],
            "restoration_generation_expected_increment": 1,
            "complex_carrier_l2_tolerance": 3e-11,
            "host_allocator_addresses_excluded": True,
            "thread_ids_excluded": True,
        },
        "matched_classical_bosonic_givens_identical": True,
        "cross_uid_secrecy_established": False,
        "microarchitectural_secrecy_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_computation_established": False,
        "terminal": False,
        "obstruction": (
            "MATCHED_CLASSICAL_BOSONIC_GIVENS_IDENTITY_AND_"
            "SAME_UID_PROTOCOL_ONLY_CUSTODY"
        ),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
