#!/usr/bin/env python3
"""Atomic process boundary for the Rotor-6 open-momentum phase carrier.

The service owns the M199 topology, the 2,277-cell bracelet source and target,
and one persistent 4,389-cell typed necklace port.  Only fixed binary protocol
records leave the process.  Accepted in-place responses are constructed after
the additive inverse has restored and verified the same carrier backing.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import resource
import sys
from pathlib import Path

import catvm_growing_rotor_open_momentum_protocol as protocol
import growing_rotor_open_momentum_factor_closure as backend


PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PRIMARY_FAMILY = 0
REUSE_FAMILY = 4
WRONG_FAMILY = 1
OWNER_TAG = 0x4D4F


def fail(message: str) -> None:
    raise RuntimeError(message)


def configure_process_boundary() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        fail("PR_SET_DUMPABLE failed")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail("PR_SET_NO_NEW_PRIVS failed")


class Service:
    def __init__(self, mode: str, audit_path: Path) -> None:
        if mode not in ("inplace", "snapshot", "null"):
            fail("invalid CATVM mode")
        self.mode = mode
        self.audit_path = audit_path
        self.generation = 0
        self.nonce = 1
        self.stage = "IDLE"
        self.active_family: int | None = None
        self.pending_operations = 0
        self.active_momentum: int | None = None
        self.active_owner: int | None = None
        if mode == "null":
            self.topology = None
            self.source: list[int] = []
            self.target: list[int] = []
            self.expected_source: tuple[int, ...] = ()
            self.port = None
            self.topology_digest = ""
            self.snapshot = None
            return
        self.topology = backend.compile_topology()
        source, _ = backend.source_and_signature_order(self.topology, 0)
        self.source = source
        self.target = [0] * len(source)
        self.expected_source = tuple(source)
        self.port = backend.OpenMomentumPort(
            [0] * len(self.topology.necklaces)
        )
        self.topology_digest = backend.topology_commitment(self.topology)
        self.snapshot = (
            (
                tuple(self.source),
                tuple(self.target),
                tuple(self.port.values),
            )
            if mode == "snapshot"
            else None
        )

    def audit(self, event: str) -> None:
        with self.audit_path.open("a", encoding="ascii") as handle:
            handle.write(f"{self.generation}:{event}\n")

    def carrier_cells(self) -> int:
        if self.port is None:
            return 0
        return len(self.source) + len(self.target) + len(self.port.values)

    def backing_identity(self) -> tuple[int, int, int]:
        if self.port is None:
            return (0, 0, 0)
        return id(self.source), id(self.target), id(self.port.values)

    def validate_idle(self) -> None:
        if self.topology is None or self.port is None:
            fail("null carrier")
        if (
            self.stage != "IDLE"
            or self.active_family is not None
            or self.pending_operations
            or self.active_momentum is not None
            or self.active_owner is not None
            or self.port.live
            or self.port.momentum is not None
            or self.port.owner_generation is not None
            or any(self.port.values)
            or any(self.target)
        ):
            fail("carrier is not idle")

    def verify_restoration(
        self, backing: tuple[int, int, int]
    ) -> None:
        self.validate_idle()
        if tuple(self.source) != self.expected_source:
            fail("source carrier restoration failed")
        if self.backing_identity() != backing:
            fail("carrier backing identity changed")

    def lease_owner(self, phase_code: int, momentum: int) -> int:
        return (
            (OWNER_TAG << 32)
            | ((self.generation + 1) << 16)
            | (phase_code << 8)
            | momentum
        )

    def apply_scattering(
        self,
        state: list[int],
        step: int,
        tag: int,
        phase_code: int,
        *,
        wrong_reflection: bool = False,
    ) -> tuple[list[int], backend.Work]:
        if self.topology is None or self.port is None:
            fail("null scattering carrier")
        output = [0] * len(state)
        port_backing = id(self.port.values)
        work = backend.Work(scatterings=1)
        for momentum in range(1, 9):
            owner = self.lease_owner(phase_code, momentum)
            backend.fill_open_port(
                state,
                self.topology,
                self.port,
                momentum,
                owner,
                work,
            )
            self.active_momentum = momentum
            self.active_owner = owner
            self.audit("PORT_LEASED_INTERNAL")
            backend.close_reflection_pair(
                state,
                output,
                self.topology,
                self.port,
                momentum,
                owner,
                step,
                tag,
                work,
                wrong_reflection=wrong_reflection,
            )
            self.active_momentum = None
            self.active_owner = None
            self.audit("PORT_RELEASED_INTERNAL")
        if (
            self.port.live
            or any(self.port.values)
            or id(self.port.values) != port_backing
        ):
            fail("persistent open port did not close")
        return output, work

    def run_word(
        self,
        family: int,
        phase_code: int,
        *,
        reordered: bool = False,
        wrong_reflection: bool = False,
    ) -> tuple[list[int], backend.Work]:
        if self.topology is None:
            fail("null word carrier")
        operations = backend.public_law.public_program(1, family)
        current = self.source.copy()
        total = backend.Work()
        for step, tag in operations:
            if reordered:
                current, scattering = self.apply_scattering(
                    current,
                    step,
                    tag,
                    phase_code,
                    wrong_reflection=wrong_reflection,
                )
                current, diagonal = backend.apply_diagonal(
                    current, self.topology, step, tag
                )
            else:
                current, diagonal = backend.apply_diagonal(
                    current, self.topology, step, tag
                )
                current, scattering = self.apply_scattering(
                    current,
                    step,
                    tag,
                    phase_code,
                    wrong_reflection=wrong_reflection,
                )
            total.add(diagonal)
            total.add(scattering)
        return current, total

    def add_target(self, delta: list[int], sign: int) -> None:
        self.target[:] = [
            (left + sign * right) % backend.PRIME
            for left, right in zip(self.target, delta, strict=True)
        ]

    def receipt(self, family: int, boundary: int) -> int:
        digest = hashlib.sha256()
        digest.update(self.topology_digest.encode("ascii"))
        digest.update(family.to_bytes(1, "little"))
        digest.update(boundary.to_bytes(2, "little"))
        digest.update(b"FINAL_BOUNDARY_AFTER_RESTORATION")
        return int.from_bytes(digest.digest()[:8], "little")

    def response(
        self,
        command: int,
        status: int,
        boundary: int = 0,
        flags: int = 0,
        receipt: int = 0,
        work_terms: int = 0,
    ) -> bytes:
        resource = (work_terms << 32) | self.carrier_cells()
        return protocol.RESPONSE.pack(
            protocol.MAGIC,
            status,
            command,
            self.generation,
            boundary,
            flags,
            receipt,
            resource,
        )

    def run_inplace(self, family: int, reuse: bool) -> bytes:
        self.validate_idle()
        backing = self.backing_identity()
        self.stage = "FORWARD"
        self.active_family = family
        self.pending_operations = 1
        self.audit("FORWARD_BEGIN")
        forward, forward_work = self.run_word(family, 1)
        self.pending_operations = 0
        self.stage = "FORWARD_COMPLETE"
        self.add_target(forward, 1)
        boundary = backend.boundary(self.target, self.topology)
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        self.stage = "INVERSE"
        self.pending_operations = 1
        inverse, inverse_work = self.run_word(family, 2)
        self.add_target(inverse, -1)
        self.pending_operations = 0
        self.stage = "IDLE"
        self.active_family = None
        self.verify_restoration(backing)
        self.generation += 1
        self.audit("RESTORATION_VERIFIED")
        flags = protocol.BOUNDARY_VALID | protocol.RESTORED
        if reuse:
            flags |= protocol.REUSE_FLAG
        response = self.response(
            protocol.REUSE_INPLACE if reuse else protocol.RUN_INPLACE,
            protocol.STATUS_OK,
            boundary,
            flags,
            self.receipt(family, boundary),
            forward_work.first_pass_one_body_terms
            + forward_work.closure_one_body_terms
            + inverse_work.first_pass_one_body_terms
            + inverse_work.closure_one_body_terms,
        )
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def run_snapshot(self, family: int) -> bytes:
        if self.snapshot is None or self.port is None:
            fail("snapshot backing unavailable")
        self.validate_idle()
        backing = self.backing_identity()
        self.stage = "FORWARD"
        self.active_family = family
        self.pending_operations = 1
        self.audit("FORWARD_BEGIN")
        forward, forward_work = self.run_word(family, 1)
        self.add_target(forward, 1)
        self.pending_operations = 0
        self.stage = "FORWARD_COMPLETE"
        boundary = backend.boundary(self.target, self.topology)
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        source, target, port = self.snapshot
        self.source[:] = source
        self.target[:] = target
        self.port.values[:] = port
        self.port.momentum = None
        self.port.owner_generation = None
        self.port.live = False
        self.stage = "IDLE"
        self.active_family = None
        self.verify_restoration(backing)
        self.generation += 1
        self.audit("SNAPSHOT_RELOADED")
        response = self.response(
            protocol.RUN_SNAPSHOT,
            protocol.STATUS_OK,
            boundary,
            protocol.BOUNDARY_VALID | protocol.SNAPSHOT_RELOADED,
            self.receipt(family, boundary),
            forward_work.first_pass_one_body_terms
            + forward_work.closure_one_body_terms,
        )
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def resident_denial(self, command: int, family: int) -> bytes:
        self.validate_idle()
        if self.topology is None or self.port is None:
            fail("null resident carrier")
        backing = self.backing_identity()
        step, tag = backend.public_law.public_program(1, family)[0]
        diagonal, _ = backend.apply_diagonal(
            self.source, self.topology, step, tag
        )
        owner = self.lease_owner(3, 1)
        work = backend.Work()
        self.stage = "FORWARD"
        self.active_family = family
        self.pending_operations = 1
        self.audit("FORWARD_BEGIN")
        backend.fill_open_port(
            diagonal, self.topology, self.port, 1, owner, work
        )
        self.active_momentum = 1
        self.active_owner = owner
        self.audit("PORT_RESIDENT")
        denied = False
        try:
            if command == protocol.PROJECT_HIDDEN_DURING_FORWARD:
                backend.boundary(self.port.values, self.topology)
            elif command == protocol.EARLY_RESPONSE_DURING_FORWARD:
                fail("response release before restoration denied")
            elif command == protocol.WRONG_TYPE_DURING_FORWARD:
                self.port.require(2, owner)
            elif command == protocol.WRONG_OWNER_DURING_FORWARD:
                self.port.require(1, owner + 1)
            else:
                fail("unknown resident attack")
        except (RuntimeError, ValueError):
            denied = True
        if not denied:
            fail("resident attack unexpectedly accepted")
        event = {
            protocol.PROJECT_HIDDEN_DURING_FORWARD: "HIDDEN_PROJECTION_DENIED",
            protocol.EARLY_RESPONSE_DURING_FORWARD: "EARLY_RESPONSE_DENIED",
            protocol.WRONG_TYPE_DURING_FORWARD: "WRONG_TYPE_DENIED",
            protocol.WRONG_OWNER_DURING_FORWARD: "WRONG_OWNER_DENIED",
        }[command]
        self.audit(event)
        self.port.release(1, owner)
        self.active_momentum = None
        self.active_owner = None
        self.audit("PORT_RELEASED")
        self.pending_operations = 0
        self.stage = "IDLE"
        self.active_family = None
        self.verify_restoration(backing)
        self.generation += 1
        self.audit("RESTORATION_VERIFIED")
        response = self.response(
            command,
            protocol.STATUS_DENIED,
            flags=protocol.RESTORED,
            work_terms=work.first_pass_one_body_terms,
        )
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def inverse_control(self, command: int, family: int) -> bytes:
        self.validate_idle()
        backing = self.backing_identity()
        self.stage = "FORWARD"
        self.active_family = family
        self.audit("FORWARD_BEGIN")
        correct, correct_work = self.run_word(family, 4)
        self.add_target(correct, 1)
        self.stage = "FORWARD_COMPLETE"
        self.audit("FORWARD_RESIDENT")
        mutation: list[int] | None = None
        mutation_work = backend.Work()
        if command == protocol.MISSING_INVERSE:
            self.audit("INVERSE_OMITTED")
        elif command == protocol.WRONG_INVERSE:
            mutation, mutation_work = self.run_word(WRONG_FAMILY, 5)
            self.add_target(mutation, -1)
            self.audit("WRONG_INVERSE_EXECUTED")
        elif command == protocol.REORDERED_INVERSE:
            mutation, mutation_work = self.run_word(
                family, 5, reordered=True
            )
            self.add_target(mutation, -1)
            self.audit("REORDERED_INVERSE_EXECUTED")
        elif command == protocol.WRONG_REFLECTION_INVERSE:
            mutation, mutation_work = self.run_word(
                family, 5, wrong_reflection=True
            )
            self.add_target(mutation, -1)
            self.audit("WRONG_REFLECTION_INVERSE_EXECUTED")
        else:
            fail("unknown inverse control")
        if not any(self.target):
            fail("mutated inverse did not discriminate")
        self.audit("CONTROL_DISCRIMINATED")
        self.stage = "CONTROL_REPAIR"
        if mutation is not None:
            self.add_target(mutation, 1)
        self.add_target(correct, -1)
        self.stage = "IDLE"
        self.active_family = None
        self.verify_restoration(backing)
        self.generation += 1
        self.audit("RESTORATION_VERIFIED")
        work_terms = (
            correct_work.first_pass_one_body_terms
            + correct_work.closure_one_body_terms
            + mutation_work.first_pass_one_body_terms
            + mutation_work.closure_one_body_terms
        )
        response = self.response(
            command,
            protocol.STATUS_DENIED,
            flags=protocol.RESTORED | protocol.CONTROL_DISCRIMINATED,
            work_terms=work_terms,
        )
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def denied(self, command: int) -> bytes:
        return self.response(command, protocol.STATUS_DENIED)

    def clean(self) -> bool:
        if self.mode == "null":
            return True
        try:
            self.validate_idle()
            return tuple(self.source) == self.expected_source
        except RuntimeError:
            return False


def read_exact(size: int) -> bytes:
    payload = sys.stdin.buffer.read(size)
    if not payload:
        return b""
    if len(payload) != size:
        fail("truncated request")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("inplace", "snapshot", "null"), required=True
    )
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    args.audit.write_text("", encoding="ascii")
    configure_process_boundary()
    service = Service(args.mode, args.audit)
    while True:
        payload = read_exact(protocol.REQUEST.size)
        if not payload:
            return 0
        magic, command, generation, family, nonce = protocol.REQUEST.unpack(
            payload
        )
        valid = (
            magic == protocol.MAGIC
            and generation == service.generation
            and nonce == service.nonce
        )
        valid_family = family in (PRIMARY_FAMILY, REUSE_FAMILY)
        if not valid:
            response = service.denied(command)
        elif command == protocol.PING:
            response = service.response(command, protocol.STATUS_OK)
            service.nonce += 1
        elif command == protocol.NOOP:
            response = service.response(command, protocol.STATUS_OK)
            service.audit("NOOP_RESPONSE_WRITE_ATTEMPT")
            service.nonce += 1
        elif command in (protocol.RUN_INPLACE, protocol.REUSE_INPLACE):
            expected_family = (
                REUSE_FAMILY
                if command == protocol.REUSE_INPLACE
                else family
            )
            allowed = (
                service.mode == "inplace"
                and valid_family
                and family == expected_family
            )
            response = (
                service.run_inplace(
                    family, command == protocol.REUSE_INPLACE
                )
                if allowed
                else service.denied(command)
            )
            service.nonce += 1
        elif command == protocol.RUN_SNAPSHOT:
            response = (
                service.run_snapshot(family)
                if service.mode == "snapshot" and valid_family
                else service.denied(command)
            )
            service.nonce += 1
        elif command in (
            protocol.PROJECT_HIDDEN_DURING_FORWARD,
            protocol.EARLY_RESPONSE_DURING_FORWARD,
            protocol.WRONG_TYPE_DURING_FORWARD,
            protocol.WRONG_OWNER_DURING_FORWARD,
        ):
            response = (
                service.resident_denial(command, family)
                if service.mode == "inplace" and valid_family
                else service.denied(command)
            )
            service.nonce += 1
        elif command in (
            protocol.MISSING_INVERSE,
            protocol.WRONG_INVERSE,
            protocol.REORDERED_INVERSE,
            protocol.WRONG_REFLECTION_INVERSE,
        ):
            response = (
                service.inverse_control(command, family)
                if service.mode == "inplace" and valid_family
                else service.denied(command)
            )
            service.nonce += 1
        elif command == protocol.NULL_CARRIER:
            response = service.denied(command)
            service.nonce += 1
        elif command == protocol.STOP:
            response = (
                service.response(
                    command,
                    protocol.STATUS_OK,
                    flags=(
                        protocol.RESTORED
                        if service.mode == "inplace"
                        else (
                            protocol.SNAPSHOT_RELOADED
                            if service.mode == "snapshot"
                            else 0
                        )
                    ),
                )
                if service.clean()
                else service.denied(command)
            )
            service.audit("STOP_RESPONSE_WRITE_ATTEMPT")
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
            return 0
        else:
            response = service.denied(command)
            service.nonce += 1
        try:
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            os._exit(0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        os._exit(30)
