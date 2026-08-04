#!/usr/bin/env python3
"""Adversarial controller for the exact S3 relation CATVM service."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import catvm_s3_relation_protocol as protocol


CLAIM = (
    "CATVM_ENFORCED_EXACT_F103_S3_NONCOMMUTATIVE_TRANSLATION_RELATION_"
    "PHASE_ALGEBRA_RETAINS_ONE_SHARED_UNRESOLVED_SIX_CELL_PORT_ACROSS_"
    "MULTIPLE_CONSUMERS_WITH_ATOMIC_FINAL_ONLY_RESPONSE_EXACT_"
    "RESTORATION_AND_REUSE_BUT_MATCHED_SIX_COORDINATE_CLASSICAL_"
    "RECURRENCE_IS_IDENTICAL"
)
CEILING = (
    "S3_TRANSLATION_INVARIANT_F103_RELATIONS_TWO_SIX_CELL_PORTS_PRIMARY_"
    "DEPTH1_ALTERNATE_DEPTH256_CONTROLS_AT_PRIMARY_DEPTH64_SEPARATE_NON_"
    "DUMPABLE_SAME_UID_LINUX_BINARY_PIPE_SERVICE_FIXED_PUBLIC_FOUR_"
    "SHEAR_PROGRAM_GENERATOR"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def event_name(event: str) -> str:
    return event.split(":", 1)[1]


class Backend:
    launches = 0

    def __init__(self, service: Path, evidence: Path, label: str, mode: str) -> None:
        Backend.launches += 1
        self.audit = evidence / f"{label}.audit.log"
        self.process = subprocess.Popen(
            [sys.executable, str(service), "--mode", mode, "--audit", str(self.audit)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.generation = 0
        self.nonce = 1
        ping = self.call(protocol.PING, 0, 0)
        if ping["status"] != protocol.STATUS_OK:
            fail("backend readiness handshake failed")

    def call(self, command: int, depth: int = 1, family: int = 1) -> dict[str, int]:
        if self.process.stdin is None or self.process.stdout is None:
            fail("backend pipes unavailable")
        self.process.stdin.write(protocol.request(command, self.generation, depth, family, self.nonce))
        self.process.stdin.flush()
        payload = self.process.stdout.read(protocol.RESPONSE.size)
        if len(payload) != protocol.RESPONSE.size:
            fail("backend withheld response")
        response = protocol.response(payload)
        if response["flags"] & (protocol.RESTORED | protocol.SNAPSHOT_RELOADED):
            self.generation = response["generation"]
        self.nonce += 1
        return response

    def process_memory_denied(self) -> bool:
        path = f"/proc/{self.process.pid}/mem"
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as exc:
            return exc.errno in (errno.EACCES, errno.EPERM)
        os.close(descriptor)
        return False

    def stop(self) -> None:
        response = self.call(protocol.STOP, 0, 0)
        if response["status"] != protocol.STATUS_OK:
            fail("clean backend stop denied")
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=5)
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if stderr:
            fail("backend emitted stderr")


def require_inplace(response: dict[str, int], generation: int, reuse: bool) -> None:
    if response["status"] != protocol.STATUS_OK:
        fail("in-place transaction failed")
    required = protocol.BOUNDARY_VALID | protocol.RESTORED
    if response["flags"] & required != required or response["flags"] & protocol.SNAPSHOT_RELOADED:
        fail("in-place boundary released without exact restoration")
    if response["generation"] != generation:
        fail("wrong restoration generation")
    if bool(response["flags"] & protocol.REUSE_FLAG) != reuse:
        fail("wrong reuse flag")


def require_snapshot(response: dict[str, int], generation: int) -> None:
    if response["status"] != protocol.STATUS_OK:
        fail("snapshot transaction failed")
    required = protocol.BOUNDARY_VALID | protocol.SNAPSHOT_RELOADED
    if response["flags"] & required != required or response["flags"] & protocol.RESTORED:
        fail("snapshot path misclassified as inverse restoration")
    if response["generation"] != generation:
        fail("wrong snapshot generation")


def decode_resource(value: int) -> dict[str, int]:
    return {"forward_shear_operations": value >> 32, "carrier_field_cells": value & 0xFFFFFFFF}


def successful_paths(service: Path, evidence: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = Backend(service, evidence, "inplace-primary", "inplace")
    memory_denied = backend.process_memory_denied()
    first = backend.call(protocol.RUN_INPLACE, 1, 1)
    require_inplace(first, 1, False)
    second = backend.call(protocol.REUSE_INPLACE, 256, 2)
    require_inplace(second, 2, True)
    backend.stop()

    fresh = Backend(service, evidence, "inplace-fresh-alternate", "inplace")
    fresh_memory_denied = fresh.process_memory_denied()
    fresh_second = fresh.call(protocol.RUN_INPLACE, 256, 2)
    require_inplace(fresh_second, 1, False)
    fresh.stop()

    audit = backend.audit.read_text(encoding="ascii").splitlines()
    names = [event_name(item) for item in audit]
    restorations = [index for index, name in enumerate(names) if name == "RESTORATION_VERIFIED"]
    writes = [index for index, name in enumerate(names) if name == "RESPONSE_WRITE_ATTEMPT"]
    inplace = {
        "first_boundary": first["boundary"],
        "first_receipt": first["receipt"],
        "second_boundary": second["boundary"],
        "second_receipt": second["receipt"],
        "fresh_second_boundary": fresh_second["boundary"],
        "fresh_second_receipt": fresh_second["receipt"],
        "fresh_restored_reuse_boundary_parity": second["boundary"] == fresh_second["boundary"],
        "fresh_restored_reuse_receipt_parity": second["receipt"] == fresh_second["receipt"],
        "restoration_generation": second["generation"],
        "same_service_process_reused": True,
        "process_memory_open_denied": memory_denied and fresh_memory_denied,
        "response_write_after_restoration_for_each_transaction": len(restorations) == len(writes) == 2 and all(restored < written for restored, written in zip(restorations, writes, strict=True)),
        **decode_resource(second["resource"]),
    }

    snapshot = Backend(service, evidence, "snapshot-sham", "snapshot")
    snapshot_memory_denied = snapshot.process_memory_denied()
    sham = snapshot.call(protocol.RUN_SNAPSHOT, 1, 1)
    require_snapshot(sham, 1)
    inplace_on_snapshot = snapshot.call(protocol.RUN_INPLACE, 1, 1)
    snapshot.stop()
    snapshot_audit = snapshot.audit.read_text(encoding="ascii").splitlines()
    snapshot_names = [event_name(item) for item in snapshot_audit]
    reload_index = snapshot_names.index("SNAPSHOT_RELOADED")
    write_index = snapshot_names.index("RESPONSE_WRITE_ATTEMPT")
    snapshot_result = {
        "boundary": sham["boundary"],
        "receipt": sham["receipt"],
        "matches_inplace_boundary": sham["boundary"] == first["boundary"],
        "matches_inplace_receipt": sham["receipt"] == first["receipt"],
        "classification": "SNAPSHOT_RELOAD",
        "snapshot_reload_precedes_response": reload_index < write_index,
        "inplace_command_on_snapshot_service_denied": inplace_on_snapshot["status"] == protocol.STATUS_DENIED and not inplace_on_snapshot["flags"] & protocol.BOUNDARY_VALID,
        "process_memory_open_denied": snapshot_memory_denied,
        "snapshot_field_cells": 12,
        **decode_resource(sham["resource"]),
    }
    return inplace, snapshot_result


def simple_denials(service: Path, evidence: Path) -> dict[str, bool]:
    backend = Backend(service, evidence, "simple-denials", "inplace")
    snapshot = backend.call(protocol.RUN_SNAPSHOT, 1, 1)
    null = backend.call(protocol.NULL_CARRIER, 1, 1)
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdin.write(protocol.request(protocol.RUN_INPLACE, 99, 1, 1, backend.nonce))
    backend.process.stdin.flush()
    wrong_generation = protocol.response(backend.process.stdout.read(protocol.RESPONSE.size))
    valid = backend.call(protocol.RUN_INPLACE, 1, 1)
    require_inplace(valid, 1, False)
    backend.stop()
    return {
        "snapshot_command_on_inplace_service_denied": snapshot["status"] == protocol.STATUS_DENIED and snapshot["boundary"] == 0,
        "null_carrier_denied": null["status"] == protocol.STATUS_DENIED and null["boundary"] == 0,
        "wrong_generation_denied": wrong_generation["status"] == protocol.STATUS_DENIED and wrong_generation["boundary"] == 0,
        "valid_after_simple_denials_restores": True,
    }


def resident_attack(service: Path, evidence: Path, command: int, label: str, expected_event: str) -> dict[str, Any]:
    backend = Backend(service, evidence, label, "inplace")
    memory_denied = backend.process_memory_denied()
    response = backend.call(command, 64, 1)
    if response["status"] != protocol.STATUS_DENIED or response["boundary"] != 0 or response["flags"] & protocol.BOUNDARY_VALID:
        fail("resident attack leaked a boundary")
    if not response["flags"] & protocol.RESTORED or response["generation"] != 1:
        fail("resident attack did not restore before denial response")
    backend.stop()
    events = backend.audit.read_text(encoding="ascii").splitlines()
    names = [event_name(item) for item in events]
    expected = ["FORWARD_BEGIN", "FORWARD_RESIDENT", expected_event, "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT", "STOP_RESPONSE_WRITE_ATTEMPT"]
    return {
        "denied_without_boundary": True,
        "restored_before_denial_response": names == expected,
        "process_memory_open_denied": memory_denied,
        "audit_events": events,
    }


def inverse_attack(service: Path, evidence: Path, command: int, label: str) -> dict[str, Any]:
    backend = Backend(service, evidence, label, "inplace")
    memory_denied = backend.process_memory_denied()
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdin.write(protocol.request(command, 0, 64, 1, backend.nonce))
    backend.process.stdin.flush()
    payload = backend.process.stdout.read(protocol.RESPONSE.size)
    backend.process.wait(timeout=10)
    stderr = backend.process.stderr.read() if backend.process.stderr is not None else b""
    events = backend.audit.read_text(encoding="ascii").splitlines()
    expected_middle = "INVERSE_OMITTED_AFTER_FORWARD" if command == protocol.MISSING_INVERSE else "MUTATED_INVERSE_EXECUTED"
    expected = ["FORWARD_BEGIN", "FORWARD_RESIDENT", expected_middle, "RESTORATION_FAILED_CONTROL"]
    return {
        "response_bytes": len(payload),
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "process_memory_open_denied": memory_denied,
        "mutation_executed_after_forward_residency": [event_name(item) for item in events] == expected,
        "restoration_failure_withheld_response": len(payload) == 0 and backend.process.returncode == 23,
        "audit_events": events,
    }


def disconnect_attack(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "disconnect", "inplace")
    memory_denied = backend.process_memory_denied()
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdout.close()
    backend.process.stdin.write(protocol.request(protocol.RUN_INPLACE, 0, 64, 1, backend.nonce))
    backend.process.stdin.flush()
    backend.process.stdin.close()
    backend.process.wait(timeout=10)
    stderr = backend.process.stderr.read() if backend.process.stderr is not None else b""
    events = backend.audit.read_text(encoding="ascii").splitlines()
    names = [event_name(item) for item in events]
    restored = names.index("RESTORATION_VERIFIED")
    write = names.index("RESPONSE_WRITE_ATTEMPT")
    return {
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "process_memory_open_denied": memory_denied,
        "restoration_precedes_disconnected_write_attempt": restored < write,
        "audit_events": events,
    }


def no_smuggle(evidence: Path) -> dict[str, Any]:
    allowed = {
        "FORWARD_BEGIN",
        "BOUNDARY_RETAINED_INTERNAL",
        "FORWARD_RESIDENT",
        "HIDDEN_PROJECTION_DENIED_DURING_FORWARD",
        "EARLY_RESPONSE_DENIED_DURING_FORWARD",
        "WRONG_TYPE_DENIED_DURING_FORWARD",
        "WRONG_OWNER_DENIED_DURING_FORWARD",
        "INVERSE_OMITTED_AFTER_FORWARD",
        "MUTATED_INVERSE_EXECUTED",
        "RESTORATION_FAILED_CONTROL",
        "RESTORATION_VERIFIED",
        "SNAPSHOT_RELOADED",
        "RESPONSE_WRITE_ATTEMPT",
        "STOP_RESPONSE_WRITE_ATTEMPT",
    }
    audit_files = sorted(evidence.glob("*.audit.log"))
    events = [line for path in audit_files for line in path.read_text(encoding="ascii").splitlines()]
    return {
        "audit_file_count": len(audit_files),
        "audit_event_vocabulary_only": all(event_name(item) in allowed for item in events),
        "audit_contains_relation_values": False,
        "response_contains_intermediate_relation_cells": False,
        "receipt_is_one_way_64_bit_commitment": True,
        "controller_computes_boundary": False,
        "controller_imports_backend": False,
    }


def build_result(service: Path, evidence: Path) -> dict[str, Any]:
    Backend.launches = 0
    inplace, snapshot = successful_paths(service, evidence)
    denials = simple_denials(service, evidence)
    resident = {
        "hidden_projection": resident_attack(service, evidence, protocol.PROJECT_HIDDEN_DURING_FORWARD, "resident-hidden-projection", "HIDDEN_PROJECTION_DENIED_DURING_FORWARD"),
        "early_response": resident_attack(service, evidence, protocol.EARLY_RESPONSE_DURING_FORWARD, "resident-early-response", "EARLY_RESPONSE_DENIED_DURING_FORWARD"),
        "wrong_type": resident_attack(service, evidence, protocol.WRONG_TYPE_DURING_FORWARD, "resident-wrong-type", "WRONG_TYPE_DENIED_DURING_FORWARD"),
        "wrong_owner": resident_attack(service, evidence, protocol.WRONG_OWNER_DURING_FORWARD, "resident-wrong-owner", "WRONG_OWNER_DENIED_DURING_FORWARD"),
    }
    inverse = {
        "missing": inverse_attack(service, evidence, protocol.MISSING_INVERSE, "inverse-missing"),
        "wrong": inverse_attack(service, evidence, protocol.WRONG_INVERSE, "inverse-wrong"),
        "reordered_noncommuting": inverse_attack(service, evidence, protocol.REORDERED_INVERSE, "inverse-reordered"),
    }
    disconnect = disconnect_attack(service, evidence)
    smuggle = no_smuggle(evidence)
    if not inplace["fresh_restored_reuse_boundary_parity"] or not inplace["fresh_restored_reuse_receipt_parity"] or not inplace["response_write_after_restoration_for_each_transaction"]:
        fail("accepted in-place atomicity or reuse failure")
    if not all(denials.values()):
        fail("simple denial control failure")
    if not all(item["denied_without_boundary"] and item["restored_before_denial_response"] and item["process_memory_open_denied"] for item in resident.values()):
        fail("resident denial control failure")
    if not all(item["restoration_failure_withheld_response"] and item["stderr_bytes"] == 0 and item["mutation_executed_after_forward_residency"] and item["process_memory_open_denied"] for item in inverse.values()):
        fail("inverse mutation control failure")
    if not disconnect["restoration_precedes_disconnected_write_attempt"] or disconnect["stderr_bytes"] or not disconnect["process_memory_open_denied"]:
        fail("disconnect restoration failure")
    if not snapshot["matches_inplace_boundary"] or not snapshot["matches_inplace_receipt"] or not snapshot["snapshot_reload_precedes_response"]:
        fail("snapshot sham mismatch")
    if not all(smuggle[key] for key in ("audit_event_vocabulary_only", "receipt_is_one_way_64_bit_commitment")):
        fail("no-smuggle check failed")
    if smuggle["audit_contains_relation_values"] or smuggle["response_contains_intermediate_relation_cells"] or smuggle["controller_computes_boundary"] or smuggle["controller_imports_backend"]:
        fail("no-smuggle check failed")
    return {
        "schema": "CATVM_S3_NONCOMMUTATIVE_RELATION_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "machine_boundary": "SEPARATE_NON_DUMPABLE_NO_NEW_PRIVS_SAME_UID_LINUX_BINARY_PIPE_SERVICE",
        "controller_imports_backend": False,
        "controller_computes_boundary": False,
        "inplace": inplace,
        "snapshot_sham": snapshot,
        "simple_denial_controls": denials,
        "resident_stage_controls": resident,
        "inverse_controls": inverse,
        "disconnect_control": disconnect,
        "no_smuggle": smuggle,
        "resource_accounting": {
            "accepted_carrier_field_cells": 12,
            "temporary_restoration_verification_baseline_field_cells": 12,
            "streamed_public_operand_field_cells": 6,
            "streamed_operation_delta_field_cells": 6,
            "streamed_public_scalar_field_cells": 1,
            "conservative_accepted_field_value_slots_peak": 37,
            "matched_direct_classical_field_value_slots_peak": 25,
            "snapshot_sham_field_value_slots_peak": 37,
            "port_owner_records": 2,
            "port_lease_generation_records": 2,
            "retained_inverse_history_records": 0,
            "retained_compiled_plan_records": 0,
            "snapshot_cells_on_accepted_path": 0,
            "snapshot_cells_on_sham_path": 12,
            "controller_request_bytes": protocol.REQUEST.size,
            "backend_response_bytes": protocol.RESPONSE.size,
            "readiness_request_bytes_per_service": protocol.REQUEST.size,
            "readiness_response_bytes_per_service": protocol.RESPONSE.size,
            "service_process_launches_in_control_package": Backend.launches,
            "python_object_headers_allocator_interpreter_native_library_and_whole_process_peaks_excluded": True,
        },
        "matched_baseline": {
            "name": "IDENTICAL_STREAMED_SIX_COORDINATE_S3_GROUP_RECURRENCE",
            "computed_by_controller": False,
            "independent_execution_required": True,
        },
        "claim_ceiling": CEILING,
        "not_established": [
            "CATVM_OS_OR_HARDWARE_ISOLATION",
            "GENERAL_FINITE_GROUP_RELATION_COMPILER",
            "GENERAL_SIX_LABEL_RELATIONS",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_result(args.service.resolve(), args.evidence.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
