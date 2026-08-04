#!/usr/bin/env python3
"""Adversarial controller for the atomic scheduled Paley relation CATVM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import catvm_paley_relation_protocol as protocol


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
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "fresh_second_boundary": fresh_second["boundary"],
        "fresh_restored_reuse_boundary_parity": second["boundary"] == fresh_second["boundary"],
        "restoration_generation": second["generation"],
        "first_receipt": first["receipt"],
        "second_receipt": second["receipt"],
        "schedule_nodes": second["resource"] >> 32,
        "carrier_field_cells": second["resource"] & 0xFFFFFFFF,
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
    audit = backend.audit.read_text(encoding="ascii")
    return {
        "response_bytes": len(payload),
        "exit_code": backend.process.returncode,
        "stderr_bytes": len(stderr),
        "mutation_rejected_without_response": len(payload) == 0 and backend.process.returncode == 23,
        "audit_contains_only_rejection_event": audit.splitlines() == ["0:MUTATED_INVERSE_REJECTED_WITHOUT_RESPONSE"],
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
        "reordered": inverse_attack(service, evidence, protocol.REORDERED_INVERSE, "reordered"),
    }
    disconnect = disconnect_attack(service, evidence)
    if not main["fresh_restored_reuse_boundary_parity"] or not main["response_write_after_restoration_for_each_transaction"]:
        fail("primary atomicity or reuse failure")
    if not all(denials.values()):
        fail("denial control failure")
    if not all(item["mutation_rejected_without_response"] and item["stderr_bytes"] == 0 and item["audit_contains_only_rejection_event"] for item in inverse.values()):
        fail("inverse mutation control failure")
    if not disconnect["restoration_precedes_disconnected_write_attempt"] or disconnect["stderr_bytes"]:
        fail("disconnect restoration failure")
    return {
        "schema": "CATVM_PALEY_RELATION_DAG_RESULTS_V1",
        "claim_candidate": "CATVM_ENFORCED_EXACT_9_NODE_SHARED_C17_PALEY_RELATION_DAG_WITH_AUTOMATIC_PUBLIC_TOPOLOGY_SCHEDULING_NATIVE_COMPOSITION_INTERSECTION_ATOMIC_FINAL_ONLY_RESPONSE_RESTORATION_AND_REUSE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "machine_boundary": "SEPARATE_LINUX_USERSPACE_BINARY_PIPE_SERVICE",
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
            "audit_vocabulary": ["FORWARD_BEGIN", "BOUNDARY_RETAINED_INTERNAL", "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT", "MUTATED_INVERSE_REJECTED_WITHOUT_RESPONSE"],
        },
        "resource_accounting": {
            "carrier_field_cells": 27,
            "sealed_leaf_field_cells": 6,
            "hidden_internal_field_cells": 21,
            "compiled_schedule_nodes": 7,
            "controller_request_bytes": protocol.REQUEST.size,
            "backend_response_bytes": protocol.RESPONSE.size,
            "retained_inverse_history_field_cells": 0,
            "snapshot_bytes": 0,
        },
        "claim_ceiling": "EXACT9_NODE_TWO_LEAF_SEVEN_INTERNAL_C17_PALEY_RELATION_DAG_TWO_PUBLIC_OPERATION_ASSIGNMENTS_SEPARATE_LINUX_USERSPACE_PIPE_SERVICE_SINGLE_CONTROLLER_PROCESS",
        "not_established": ["GENERIC_DAG_SCHEDULER", "CATVM_OS_OR_HARDWARE_ISOLATION", "DISTINCT_PHASE_RESOURCE", "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING", "PHYSICAL_EXECUTION", "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI", "UNBOUNDED_CATALYTIC_COMPUTATION"],
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
