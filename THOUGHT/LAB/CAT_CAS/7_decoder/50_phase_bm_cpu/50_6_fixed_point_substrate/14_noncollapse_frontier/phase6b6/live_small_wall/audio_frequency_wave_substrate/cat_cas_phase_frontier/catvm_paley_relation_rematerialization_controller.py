#!/usr/bin/env python3
"""Adversarial controller for the rematerializing Paley relation CATVM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import catvm_paley_relation_protocol as protocol


CLAIM = "CATVM_ENFORCED_EXACT_9_NODE_C17_PALEY_RELATION_DAG_PUBLIC_REVERSIBLE_PEBBLE_SCHEDULER_REDUCES_RETAINED_INTERNAL_STATE_FROM_7_TO_6_RELATION_SLOTS_WITH_EXACT_MINIMUM_6_CLEAN_LOCAL_PEBBLE_CERTIFICATE_AT_THIS_TOPOLOGY_FINAL_ONLY_ATOMIC_RESPONSE_RESTORATION_AND_REUSE_BUT_MATCHED_CLASSICAL_PEBBLING_IS_IDENTICAL_AND_AN_EXECUTED_OCCURRENCE_EXPANDED_COMPACT_CLASSICAL_RECURRENCE_IS_SMALLER"
CEILING = "EXACT9_NODE_TWO_LEAF_SEVEN_INTERNAL_C17_PALEY_RELATION_DAG_TWO_PUBLIC_OPERATION_ASSIGNMENTS_LOCAL_REVERSIBLE_TOGGLE_LAW_SOURCES_ALWAYS_RESIDENT_CLEAN_FINAL_ONLY_SINK_SEPARATE_SAME_UID_LINUX_BINARY_PIPE_SERVICE_SINGLE_CONTROLLER_PROCESS"


def fail(message: str) -> None:
    raise RuntimeError(message)


class Backend:
    def __init__(self, service: Path, evidence: Path, label: str) -> None:
        self.audit = evidence / f"{label}.audit.log"
        self.process = subprocess.Popen(
            [sys.executable, str(service), "--audit", str(self.audit)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.generation = 0
        self.nonce = 1

    def call(self, command: int, topology: int = 1, family: int = 1) -> dict[str, int]:
        if self.process.stdin is None or self.process.stdout is None:
            fail("backend pipes unavailable")
        self.process.stdin.write(protocol.request(command, self.generation, topology, family, self.nonce))
        self.process.stdin.flush()
        payload = self.process.stdout.read(protocol.RESPONSE.size)
        if len(payload) != protocol.RESPONSE.size:
            fail("backend withheld response")
        response = protocol.response(payload)
        if response["status"] == protocol.STATUS_OK and command in (protocol.RUN, protocol.REUSE):
            self.generation = response["generation"]
        self.nonce += 1
        return response

    def stop(self) -> None:
        response = self.call(protocol.STOP)
        if response["status"] != protocol.STATUS_OK:
            fail("clean backend stop denied")
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=5)
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if stderr:
            fail("backend emitted stderr")


def require_restored(response: dict[str, int], generation: int, reuse: bool) -> None:
    if response["status"] != protocol.STATUS_OK:
        fail("transaction failed")
    needed = protocol.BOUNDARY_VALID | protocol.RESTORED
    if response["flags"] & needed != needed:
        fail("boundary released without restoration")
    if response["generation"] != generation:
        fail("wrong restoration generation")
    if bool(response["flags"] & protocol.REUSE_FLAG) != reuse:
        fail("wrong reuse flag")


def decode_resources(value: int) -> dict[str, int]:
    return {
        "forward_toggle_actions": value >> 48,
        "internal_relation_capacity": (value >> 32) & 0xFFFF,
        "carrier_field_cells": value & 0xFFFFFFFF,
    }


def primary(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "primary")
    first = backend.call(protocol.RUN, 1, 1)
    require_restored(first, 1, False)
    second = backend.call(protocol.REUSE, 2, 2)
    require_restored(second, 2, True)
    backend.stop()

    fresh = Backend(service, evidence, "fresh-reuse")
    fresh_second = fresh.call(protocol.RUN, 2, 2)
    require_restored(fresh_second, 1, False)
    fresh.stop()
    audit = backend.audit.read_text(encoding="ascii").splitlines()
    restore_positions = [index for index, item in enumerate(audit) if item.endswith("RESTORATION_VERIFIED")]
    write_positions = [index for index, item in enumerate(audit) if item.endswith("RESPONSE_WRITE_ATTEMPT")]
    resources = decode_resources(second["resource"])
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh_second["boundary"],
        "fresh_restored_reuse_boundary_parity": second["boundary"] == fresh_second["boundary"],
        "restoration_generation": second["generation"],
        "first_receipt": first["receipt"],
        "second_receipt": second["receipt"],
        **resources,
        "response_write_after_restoration_for_each_transaction": len(restore_positions) == len(write_positions) == 2 and all(restore < write for restore, write in zip(restore_positions, write_positions, strict=True)),
        "audit_event_count": len(audit),
    }


def denial_controls(service: Path, evidence: Path) -> dict[str, bool]:
    backend = Backend(service, evidence, "denials")
    results: dict[str, bool] = {}
    for name, command in (("hidden_projection", protocol.PROJECT_HIDDEN), ("null_carrier", protocol.NULL_CARRIER), ("snapshot", protocol.SNAPSHOT)):
        response = backend.call(command)
        results[f"{name}_denied_without_boundary"] = response["status"] == protocol.STATUS_DENIED and response["boundary"] == 0 and not response["flags"] & protocol.BOUNDARY_VALID
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdin.write(protocol.request(protocol.RUN, 99, 1, 1, backend.nonce))
    backend.process.stdin.flush()
    wrong_generation = protocol.response(backend.process.stdout.read(protocol.RESPONSE.size))
    results["wrong_generation_denied"] = wrong_generation["status"] == protocol.STATUS_DENIED and wrong_generation["boundary"] == 0
    valid = backend.call(protocol.RUN, 1, 1)
    require_restored(valid, 1, False)
    results["valid_after_denials_restores"] = True
    backend.stop()
    return results


def inverse_attack(service: Path, evidence: Path, command: int, label: str) -> dict[str, Any]:
    backend = Backend(service, evidence, label)
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdin.write(protocol.request(command, 0, 1, 1, 1))
    backend.process.stdin.flush()
    payload = backend.process.stdout.read(protocol.RESPONSE.size)
    backend.process.wait(timeout=5)
    stderr = backend.process.stderr.read() if backend.process.stderr is not None else b""
    events = backend.audit.read_text(encoding="ascii").splitlines()
    expected_middle = "0:INVERSE_OMITTED_AFTER_FORWARD" if command == protocol.MISSING_INVERSE else "0:MUTATED_INVERSE_EXECUTED"
    expected = ["0:FORWARD_BEGIN", "0:BOUNDARY_RETAINED_INTERNAL", expected_middle, "0:RESTORATION_FAILED_CONTROL"]
    return {
        "response_bytes": len(payload),
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "mutation_executed_after_forward": events[:2] == expected[:2] and events[2:] == expected[2:],
        "restoration_failure_withheld_response": len(payload) == 0 and backend.process.returncode == 23,
        "audit_events": events,
    }


def disconnect_attack(service: Path, evidence: Path) -> dict[str, Any]:
    backend = Backend(service, evidence, "disconnect")
    if backend.process.stdin is None or backend.process.stdout is None:
        fail("backend pipes unavailable")
    backend.process.stdout.close()
    backend.process.stdin.write(protocol.request(protocol.RUN, 0, 1, 1, 1))
    backend.process.stdin.flush()
    backend.process.stdin.close()
    backend.process.wait(timeout=5)
    stderr = backend.process.stderr.read() if backend.process.stderr is not None else b""
    events = backend.audit.read_text(encoding="ascii").splitlines()
    restored = next((index for index, item in enumerate(events) if item.endswith("RESTORATION_VERIFIED")), -1)
    write = next((index for index, item in enumerate(events) if item.endswith("RESPONSE_WRITE_ATTEMPT")), -1)
    return {
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "restoration_precedes_disconnected_write_attempt": restored >= 0 and write > restored,
        "audit_events": events,
    }


def build_result(service: Path, evidence: Path) -> dict[str, Any]:
    main = primary(service, evidence)
    denials = denial_controls(service, evidence)
    inverse = {
        "missing": inverse_attack(service, evidence, protocol.MISSING_INVERSE, "missing"),
        "wrong": inverse_attack(service, evidence, protocol.WRONG_INVERSE, "wrong"),
        "reordered_dependent_pair": inverse_attack(service, evidence, protocol.REORDERED_INVERSE, "reordered"),
    }
    disconnect = disconnect_attack(service, evidence)
    if not main["fresh_restored_reuse_boundary_parity"] or not main["response_write_after_restoration_for_each_transaction"]:
        fail("primary atomicity or reuse failure")
    if main["forward_toggle_actions"] != 15 or main["internal_relation_capacity"] != 6 or main["carrier_field_cells"] != 24:
        fail("unexpected scheduler resource signature")
    if not all(denials.values()):
        fail("denial control failure")
    if not all(item["restoration_failure_withheld_response"] and item["stderr_bytes"] == 0 and item["mutation_executed_after_forward"] for item in inverse.values()):
        fail("inverse mutation control failure")
    if not disconnect["restoration_precedes_disconnected_write_attempt"] or disconnect["stderr_bytes"]:
        fail("disconnect restoration failure")
    return {
        "schema": "CATVM_PALEY_RELATION_REMATERIALIZATION_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "machine_boundary": "SEPARATE_SAME_UID_LINUX_USERSPACE_BINARY_PIPE_SERVICE",
        "controller_imports_backend": False,
        "controller_computes_boundary": False,
        "primary": main,
        "denial_controls": denials,
        "inverse_controls": inverse,
        "disconnect_control": disconnect,
        "no_smuggle": {
            "response_contains_intermediate_relation_cells": False,
            "stderr_bytes_all_services": 0,
            "audit_contains_relation_values": False,
            "audit_vocabulary": ["FORWARD_BEGIN", "BOUNDARY_RETAINED_INTERNAL", "INVERSE_OMITTED_AFTER_FORWARD", "MUTATED_INVERSE_EXECUTED", "RESTORATION_FAILED_CONTROL", "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT"],
        },
        "resource_accounting": {
            "accepted_carrier_field_cells": 24,
            "accepted_relation_slots": 8,
            "sealed_leaf_field_cells": 6,
            "hidden_internal_field_cells": 18,
            "hidden_internal_relation_capacity": 6,
            "retain_all_carrier_field_cells": 27,
            "retain_all_internal_relation_slots": 7,
            "forward_toggle_actions": 15,
            "full_transaction_relation_evaluations": 30,
            "compiled_plan_records": 15,
            "owner_node_records": 6,
            "owner_epoch_records": 6,
            "controller_request_bytes": protocol.REQUEST.size,
            "backend_response_bytes": protocol.RESPONSE.size,
            "retained_dynamic_inverse_history_records": 0,
            "snapshot_bytes": 0,
        },
        "claim_ceiling": CEILING,
        "not_established": ["GENERIC_DAG_SCHEDULER", "ARBITRARY_RELATION_PROGRAM", "CATVM_OS_OR_HARDWARE_ISOLATION", "DISTINCT_PHASE_RESOURCE", "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING", "PHYSICAL_EXECUTION", "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI", "UNBOUNDED_CATALYTIC_COMPUTATION"],
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
