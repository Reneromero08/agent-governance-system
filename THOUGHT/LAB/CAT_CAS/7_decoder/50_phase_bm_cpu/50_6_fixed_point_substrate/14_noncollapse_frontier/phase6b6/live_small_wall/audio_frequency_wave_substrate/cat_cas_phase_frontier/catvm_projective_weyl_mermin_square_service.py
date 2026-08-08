#!/usr/bin/env python3
"""M250 exact projective-Weyl Mermin-square CATVM backend.

The accepted machine composes typed two-qubit Pauli signatures through their
native projective 2-cocycle.  Nine observable ports are each consumed by two
commuting contexts.  Only the final central phase leaves the service, after
the actual four-cell ``Q(i)`` carrier and every custody register have been
restored in reverse order.  This is an exact software contextuality
calibration, not physical contextuality or a computational-advantage claim.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import socket
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


PORT_TYPE = "CATVM_TYPED_TWO_QUBIT_PROJECTIVE_WEYL_PORT_V1"
OUTPUT_TYPE = "PROJECTIVE_WEYL_CENTRAL_PHASE_BOUNDARY_V1"
OWNER = 250004
CONTROLLER_ID = 250001
VARIANTS = ("BASE", "H_CONJUGATED_REORDERED")


@dataclass(frozen=True)
class GI:
    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __add__(self, other: "GI") -> "GI":
        return GI(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "GI") -> "GI":
        return GI(self.real - other.real, self.imag - other.imag)

    def __neg__(self) -> "GI":
        return GI(-self.real, -self.imag)

    def __mul__(self, other: "GI") -> "GI":
        return GI(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def conjugate(self) -> "GI":
        return GI(self.real, -self.imag)

    def phase(self, exponent: int) -> "GI":
        exponent %= 4
        if exponent == 0:
            return self
        if exponent == 1:
            return GI(-self.imag, self.real)
        if exponent == 2:
            return -self
        return GI(self.imag, -self.real)


ZERO = GI()
ONE = GI(Fraction(1))
INITIAL_VECTOR = (
    GI(Fraction(1, 2)),
    GI(Fraction(0), Fraction(1, 2)),
    GI(Fraction(-1, 2)),
    GI(Fraction(0), Fraction(-1, 2)),
)


def fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def gi_json(value: GI) -> dict[str, list[int]]:
    return {"real": fraction_json(value.real), "imag": fraction_json(value.imag)}


@dataclass(frozen=True)
class Sig:
    q: int
    x: tuple[int, int]
    z: tuple[int, int]

    def __post_init__(self) -> None:
        if self.q not in range(4) or any(bit not in (0, 1) for bit in self.x + self.z):
            raise RuntimeError("invalid M250 Weyl signature")

    def compose(self, other: "Sig") -> "Sig":
        cocycle = 2 * sum(left * right for left, right in zip(self.z, other.x))
        return Sig(
            (self.q + other.q + cocycle) % 4,
            tuple(left ^ right for left, right in zip(self.x, other.x)),
            tuple(left ^ right for left, right in zip(self.z, other.z)),
        )

    def compose_without_cocycle(self, other: "Sig") -> "Sig":
        return Sig(
            (self.q + other.q) % 4,
            tuple(left ^ right for left, right in zip(self.x, other.x)),
            tuple(left ^ right for left, right in zip(self.z, other.z)),
        )

    def inverse(self) -> "Sig":
        self_cocycle = 2 * sum(left * right for left, right in zip(self.z, self.x))
        return Sig((-self.q - self_cocycle) % 4, self.x, self.z)

    def commutes(self, other: "Sig") -> bool:
        symplectic = sum(
            self.z[index] * other.x[index] + self.x[index] * other.z[index]
            for index in range(2)
        )
        return symplectic % 2 == 0

    def central(self) -> bool:
        return self.x == (0, 0) and self.z == (0, 0)


IDENTITY = Sig(0, (0, 0), (0, 0))


BASE_OBSERVABLES = {
    "XI": Sig(0, (1, 0), (0, 0)),
    "IX": Sig(0, (0, 1), (0, 0)),
    "XX": Sig(0, (1, 1), (0, 0)),
    "IY": Sig(1, (0, 1), (0, 1)),
    "YI": Sig(1, (1, 0), (1, 0)),
    "YY": Sig(2, (1, 1), (1, 1)),
    "XY": Sig(1, (1, 1), (0, 1)),
    "YX": Sig(1, (1, 1), (1, 0)),
    "ZZ": Sig(0, (0, 0), (1, 1)),
}

BASE_CONTEXTS = (
    ("R0", ("XI", "IX", "XX")),
    ("R1", ("IY", "YI", "YY")),
    ("R2", ("XY", "YX", "ZZ")),
    ("C0", ("XI", "IY", "XY")),
    ("C1", ("IX", "YI", "YX")),
    ("C2", ("XX", "YY", "ZZ")),
)


def h_conjugate(signature: Sig) -> Sig:
    phase = 2 * sum(left * right for left, right in zip(signature.x, signature.z))
    return Sig((signature.q + phase) % 4, signature.z, signature.x)


@dataclass(frozen=True)
class Action:
    context: str
    observable: str
    signature: Sig


@dataclass(frozen=True)
class Program:
    variant: str
    actions: tuple[Action, ...]
    contexts: tuple[str, ...]
    port_consumers: tuple[tuple[str, tuple[str, str]], ...]


def compile_program(variant: str) -> Program:
    if variant not in VARIANTS:
        raise RuntimeError("M250 variant outside declared suite")
    observables = dict(BASE_OBSERVABLES)
    contexts = list(BASE_CONTEXTS)
    if variant == "H_CONJUGATED_REORDERED":
        observables = {name: h_conjugate(signature) for name, signature in observables.items()}
        contexts = [(name, tuple(reversed(ports))) for name, ports in reversed(contexts)]
    consumers: dict[str, list[str]] = {name: [] for name in observables}
    actions: list[Action] = []
    for context, ports in contexts:
        if len(ports) != 3 or len(set(ports)) != 3:
            raise RuntimeError("invalid M250 public context arity")
        signatures = [observables[name] for name in ports]
        if not all(
            signatures[left].commutes(signatures[right])
            for left in range(3) for right in range(left + 1, 3)
        ):
            raise RuntimeError("M250 context is not commuting")
        for name in ports:
            consumers[name].append(context)
            actions.append(Action(context, name, observables[name]))
    if len(actions) != 18 or set(consumers) != set(BASE_OBSERVABLES):
        raise RuntimeError("invalid M250 public topology")
    if any(len(set(value)) != 2 for value in consumers.values()):
        raise RuntimeError("M250 observable port must have exactly two consumers")
    return Program(
        variant,
        tuple(actions),
        tuple(name for name, _ in contexts),
        tuple(sorted((name, tuple(sorted(value))) for name, value in consumers.items())),
    )


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[str]:
    return (str(descriptor["variant"]),)


def descriptor_digest(descriptor: tuple[str]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def validate_descriptor(descriptor: dict[str, Any]) -> tuple[str]:
    forbidden = {
        "answer", "expected_phase", "context_products", "carrier_vector",
        "observable_values", "assignments", "dense_matrices",
    }
    if set(descriptor) != {"variant"} or forbidden.intersection(descriptor):
        raise RuntimeError("answer-bearing or malformed M250 descriptor")
    canonical = canonical_descriptor(descriptor)
    compile_program(canonical[0])
    return canonical


@dataclass
class Work:
    forward_pauli_actions: int = 0
    inverse_pauli_actions: int = 0
    forward_vector_cell_reads: int = 0
    forward_vector_cell_writes: int = 0
    inverse_vector_cell_reads: int = 0
    inverse_vector_cell_writes: int = 0
    forward_signature_compositions: int = 0
    inverse_signature_compositions: int = 0
    forward_port_consumptions: int = 0
    inverse_port_releases: int = 0
    final_overlap_field_multiplications: int = 0
    final_overlap_accumulations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def apply_signature(
    vector: list[GI], scratch: list[GI], signature: Sig, inverse: bool,
    work: Work | None = None,
) -> None:
    if len(vector) != 4 or len(scratch) != 4 or any(value != ZERO for value in scratch):
        raise RuntimeError("M250 dirty or mistyped vector scratch")
    for basis, amplitude in enumerate(vector):
        bits = ((basis >> 1) & 1, basis & 1)
        target = basis ^ (signature.x[0] << 1) ^ signature.x[1]
        sign = 2 * (sum(left * right for left, right in zip(signature.z, bits)) % 2)
        if scratch[target] != ZERO:
            raise RuntimeError("M250 Pauli permutation collision")
        scratch[target] = amplitude.phase(signature.q + sign)
    vector[:] = scratch
    scratch[:] = [ZERO, ZERO, ZERO, ZERO]
    if work is not None:
        if inverse:
            work.inverse_pauli_actions += 1
            work.inverse_vector_cell_reads += 4
            work.inverse_vector_cell_writes += 8
        else:
            work.forward_pauli_actions += 1
            work.forward_vector_cell_reads += 4
            work.forward_vector_cell_writes += 8


def overlap_with_initial(vector: list[GI], work: Work | None = None) -> GI:
    result = ZERO
    for initial, current in zip(INITIAL_VECTOR, vector):
        result = result + initial.conjugate() * current
        if work is not None:
            work.final_overlap_field_multiplications += 1
            work.final_overlap_accumulations += 1
    return result


def context_closures(program: Program, cocycle: bool = True) -> dict[str, Sig]:
    result = {name: IDENTITY for name in program.contexts}
    for action in program.actions:
        if cocycle:
            result[action.context] = result[action.context].compose(action.signature)
        else:
            result[action.context] = result[action.context].compose_without_cocycle(action.signature)
    return result


def total_context_phase(closures: dict[str, Sig], order: tuple[str, ...]) -> Sig:
    result = IDENTITY
    for name in order:
        result = result.compose(closures[name])
    return result


def vector_after(actions: list[Sig]) -> list[GI]:
    vector = list(INITIAL_VECTOR)
    scratch = [ZERO, ZERO, ZERO, ZERO]
    for signature in actions:
        apply_signature(vector, scratch, signature, False)
    return vector


def mechanism_controls() -> dict[str, bool]:
    programs = [compile_program(variant) for variant in VARIANTS]
    for program in programs:
        closures = context_closures(program)
        if not all(signature.central() for signature in closures.values()):
            raise RuntimeError("M250 noncentral context closure")
        total = total_context_phase(closures, program.contexts)
        if total != Sig(2, (0, 0), (0, 0)):
            raise RuntimeError("M250 contextual central phase mismatch")
        vector = vector_after([action.signature for action in program.actions])
        if overlap_with_initial(vector) != GI(Fraction(-1)):
            raise RuntimeError("M250 carrier boundary mismatch")
        restored = vector_after(
            [action.signature for action in program.actions]
            + [action.signature.inverse() for action in reversed(program.actions)]
        )
        if restored != list(INITIAL_VECTOR):
            raise RuntimeError("M250 exact inverse mismatch")

    base = programs[0]
    no_cocycle = total_context_phase(context_closures(base, False), base.contexts)
    wrong_observables = {
        name: Sig((signature.q - (1 if "Y" in name else 0)) % 4, signature.x, signature.z)
        for name, signature in BASE_OBSERVABLES.items()
    }
    wrong_actions = [wrong_observables[action.observable] for action in base.actions]
    wrong_y_boundary = overlap_with_initial(vector_after(wrong_actions))

    missing = vector_after([action.signature for action in base.actions])
    missing_scratch = [ZERO, ZERO, ZERO, ZERO]
    for action in reversed(base.actions[1:]):
        apply_signature(missing, missing_scratch, action.signature.inverse(), True)

    wrong = vector_after([action.signature for action in base.actions])
    wrong_scratch = [ZERO, ZERO, ZERO, ZERO]
    apply_signature(wrong, wrong_scratch, base.actions[-2].signature.inverse(), True)
    for action in reversed(base.actions[:-1]):
        apply_signature(wrong, wrong_scratch, action.signature.inverse(), True)

    pair_index = next(
        index for index in range(len(base.actions) - 1)
        if not base.actions[index].signature.commutes(base.actions[index + 1].signature)
    )
    reordered = vector_after([action.signature for action in base.actions])
    reordered_scratch = [ZERO, ZERO, ZERO, ZERO]
    for action in reversed(base.actions[pair_index + 2:]):
        apply_signature(reordered, reordered_scratch, action.signature.inverse(), True)
    apply_signature(reordered, reordered_scratch, base.actions[pair_index].signature.inverse(), True)
    apply_signature(reordered, reordered_scratch, base.actions[pair_index + 1].signature.inverse(), True)
    for action in reversed(base.actions[:pair_index]):
        apply_signature(reordered, reordered_scratch, action.signature.inverse(), True)

    dirty_scratch_rejected = False
    try:
        apply_signature(
            list(INITIAL_VECTOR), [ONE, ZERO, ZERO, ZERO],
            base.actions[0].signature, False,
        )
    except RuntimeError:
        dirty_scratch_rejected = True

    premature_projection_rejected = False
    duplicate_consumer_rejected = False
    control_carrier = Carrier("m250-mechanism-control")
    control_request = {
        "owner": OWNER, "generation": 1,
        "program_id": descriptor_digest(("BASE",)),
        "transaction_id": "M250_MECHANISM_CONTROL",
    }
    control_carrier.lease(("BASE",), control_request)
    control_work = Work()
    try:
        control_carrier.project_boundary(base, control_work)
    except RuntimeError:
        premature_projection_rejected = True
    control_carrier.forward_action(base, 0, control_work)
    try:
        control_carrier.forward_action(base, 0, control_work)
    except RuntimeError:
        duplicate_consumer_rejected = True
    control_carrier.inverse_action(base, 0, control_work)
    control_carrier.release()

    return {
        "all_nine_observables_have_two_distinct_consumers": all(
            len(consumers) == 2 and consumers[0] != consumers[1]
            for _, consumers in base.port_consumers
        ),
        "all_six_contexts_are_pairwise_commuting": True,
        "all_six_contexts_close_centrally": True,
        "five_contexts_close_plus_one_one_closes_minus_one": sorted(
            signature.q for signature in context_closures(base).values()
        ) == [0, 0, 0, 0, 0, 2],
        "projective_total_context_phase_is_minus_one": total_context_phase(
            context_closures(base), base.contexts
        ).q == 2,
        "deleting_two_cocycle_collapses_total_context_phase_to_plus_one": no_cocycle.q == 0,
        "wrong_y_phase_changes_contextual_boundary": wrong_y_boundary != GI(Fraction(-1)),
        "base_and_h_conjugated_programs_have_exact_minus_one_boundary": True,
        "base_and_h_conjugated_programs_restore_exactly": True,
        "missing_inverse_fails_restoration": missing != list(INITIAL_VECTOR),
        "wrong_inverse_completed_path_fails_restoration": wrong != list(INITIAL_VECTOR),
        "prospectively_noncommuting_reordered_inverse_fails_restoration": reordered != list(INITIAL_VECTOR),
        "dirty_vector_scratch_rejected_before_mutation": dirty_scratch_rejected,
        "premature_final_projection_rejected": premature_projection_rejected,
        "duplicate_observable_consumer_or_cursor_rejected": duplicate_consumer_rejected,
        "noncontextual_parity_contradiction_requires_no_assignment_enumeration": True,
        "dense_four_by_four_operator_not_materialized_on_accepted_path": True,
    }


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.vector = list(INITIAL_VECTOR)
        self.scratch = [ZERO, ZERO, ZERO, ZERO]
        self.context_accumulators = {name: IDENTITY for name, _ in BASE_CONTEXTS}
        self.port_consumers_used = {name: set() for name in BASE_OBSERVABLES}
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.descriptor: tuple[str] | None = None
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.vector == list(INITIAL_VECTOR)
            and self.scratch == [ZERO, ZERO, ZERO, ZERO]
            and all(value == IDENTITY for value in self.context_accumulators.values())
            and all(not value for value in self.port_consumers_used.values())
            and self.owner == 0 and self.generation == 0
            and self.program_id == "" and self.transaction_id == ""
            and self.descriptor is None and self.cursor == 0
            and not self.projected and not self.leased
        )

    def lease(self, descriptor: tuple[str], request: dict[str, Any]) -> None:
        if not self.canonical():
            raise RuntimeError("M250 lease requires canonical carrier")
        if request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M250 stale or skipped generation")
        self.owner = int(request["owner"])
        self.generation = int(request["generation"])
        self.program_id = str(request["program_id"])
        self.transaction_id = str(request["transaction_id"])
        self.descriptor = descriptor
        self.leased = True

    def require(self, request: dict[str, Any], descriptor: tuple[str]) -> None:
        if (
            not self.leased or self.owner != request["owner"]
            or self.generation != request["generation"]
            or self.program_id != request["program_id"]
            or self.transaction_id != request["transaction_id"]
            or self.descriptor != descriptor
        ):
            raise RuntimeError("M250 custody mismatch")

    def forward_action(self, program: Program, index: int, work: Work) -> None:
        if index != self.cursor or self.projected or any(value != ZERO for value in self.scratch):
            raise RuntimeError("M250 forward cursor or scratch violation")
        action = program.actions[index]
        allowed = dict(program.port_consumers)[action.observable]
        used = self.port_consumers_used[action.observable]
        if action.context not in allowed or action.context in used:
            raise RuntimeError("M250 duplicate or wrong observable consumer")
        apply_signature(self.vector, self.scratch, action.signature, False, work)
        self.context_accumulators[action.context] = self.context_accumulators[
            action.context
        ].compose(action.signature)
        used.add(action.context)
        self.cursor += 1
        work.forward_signature_compositions += 1
        work.forward_port_consumptions += 1

    def project_boundary(self, program: Program, work: Work) -> tuple[int, GI]:
        if (
            self.cursor != len(program.actions) or self.projected
            or any(value != ZERO for value in self.scratch)
            or any(len(value) != 2 for value in self.port_consumers_used.values())
            or not all(value.central() for value in self.context_accumulators.values())
        ):
            raise RuntimeError("M250 premature or malformed projection")
        total = IDENTITY
        for context in program.contexts:
            total = total.compose(self.context_accumulators[context])
        if not total.central():
            raise RuntimeError("M250 noncentral final boundary")
        overlap = overlap_with_initial(self.vector, work)
        if overlap != ONE.phase(total.q):
            raise RuntimeError("M250 operator and carrier boundary disagree")
        self.projected = True
        return total.q, overlap

    def inverse_action(self, program: Program, index: int, work: Work) -> None:
        if index != self.cursor - 1 or any(value != ZERO for value in self.scratch):
            raise RuntimeError("M250 inverse cursor or scratch violation")
        action = program.actions[index]
        used = self.port_consumers_used[action.observable]
        if action.context not in used:
            raise RuntimeError("M250 inverse port ownership mismatch")
        self.context_accumulators[action.context] = self.context_accumulators[
            action.context
        ].compose(action.signature.inverse())
        used.remove(action.context)
        apply_signature(self.vector, self.scratch, action.signature.inverse(), True, work)
        self.cursor -= 1
        if self.cursor == 0:
            self.projected = False
        work.inverse_signature_compositions += 1
        work.inverse_port_releases += 1

    def release(self) -> None:
        if not self.canonical_except_lease():
            raise RuntimeError("M250 release before exact restoration")
        restored_generation = self.generation
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.descriptor = None
        self.leased = False
        self.last_restored_generation = restored_generation

    def canonical_except_lease(self) -> bool:
        return (
            self.vector == list(INITIAL_VECTOR)
            and self.scratch == [ZERO, ZERO, ZERO, ZERO]
            and all(value == IDENTITY for value in self.context_accumulators.values())
            and all(not value for value in self.port_consumers_used.values())
            and self.cursor == 0 and not self.projected and self.leased
            and self.descriptor is not None
        )


def run_transaction(
    carrier: Carrier, descriptor: tuple[str], request: dict[str, Any]
) -> dict[str, Any]:
    program = compile_program(descriptor[0])
    carrier.lease(descriptor, request)
    carrier.require(request, descriptor)
    work = Work()
    backing_ids = (
        id(carrier.vector), id(carrier.scratch), id(carrier.context_accumulators),
        id(carrier.port_consumers_used),
    )
    boundary: tuple[int, GI] | None = None
    failure: Exception | None = None
    try:
        for index in range(len(program.actions)):
            carrier.forward_action(program, index, work)
            if request.get("inject_failure_after_partial") and index == 6:
                raise RuntimeError("injected M250 partial-forward failure")
        boundary = carrier.project_boundary(program, work)
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M250 post-projection failure")
        delay = int(request.get("test_delay_before_inverse_ms", 0))
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        while carrier.cursor:
            carrier.inverse_action(program, carrier.cursor - 1, work)
        same_backings = backing_ids == (
            id(carrier.vector), id(carrier.scratch), id(carrier.context_accumulators),
            id(carrier.port_consumers_used),
        )
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M250 final boundary absent")
    return {
        "variant": program.variant,
        "generation": carrier.last_restored_generation,
        "central_phase_exponent_mod4": boundary[0],
        "central_phase": gi_json(boundary[1]),
        "observable_port_count": 9,
        "context_count": 6,
        "hidden_carrier_field_cells": 4,
        "hidden_scratch_field_cells": 4,
        "hidden_context_signature_cells": 6,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_carrier_and_custody_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, Carrier] = {}

    def carrier_for(self, carrier_id: str) -> Carrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = Carrier(carrier_id)
        return self.carriers[carrier_id]

    def validate_request(self, request: dict[str, Any]) -> tuple[str]:
        descriptor_value = request.get("descriptor")
        if not isinstance(descriptor_value, dict):
            raise RuntimeError("missing M250 descriptor")
        descriptor = validate_descriptor(descriptor_value)
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("controller_id") != CONTROLLER_ID
            or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(descriptor)
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id") or not request.get("carrier_id")
        ):
            raise RuntimeError("invalid M250 public request")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("invalid M250 delay")
        return descriptor

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            carrier = self.carriers.get(str(request.get("carrier_id", "")))
            if carrier is None:
                return {"status": "REJECTED"}
            return {
                "status": "OK", "canonical": carrier.canonical(),
                "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            try:
                self.validate_request(request)
                controls = mechanism_controls()
            except Exception:
                return {"status": "REJECTED"}
            return {"status": "OK", "controls": controls}
        if command == "RUN":
            try:
                descriptor = self.validate_request(request)
                response = run_transaction(
                    self.carrier_for(str(request["carrier_id"])), descriptor, request
                )
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_OBSERVABLE", "PROJECT_CONTEXT", "PROJECT_VECTOR",
            "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR", "DENSE_MATRIX",
            "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
            "NULL_CARRIER", "DUMP", "DEBUG", "CONTEXT_SIGNATURES",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m250-"):
        raise RuntimeError("M250 requires a declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m250-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M250 could not disable core dumps")
    startup = json.loads(sys.stdin.readline())
    sys.stdin.close()
    if startup != {"service": "M250_PROJECTIVE_WEYL_MERMIN_MODE"}:
        raise RuntimeError("invalid M250 startup mode")
    service = Service()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address(sys.argv[1]))
    listener.listen(8)
    running = True
    while running:
        connection, _ = listener.accept()
        try:
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                payload += chunk
            if not payload:
                continue
            response = service.handle(json.loads(payload))
            running = not response.get("shutdown", False)
            connection.sendall(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            connection.close()
    listener.close()


if __name__ == "__main__":
    main()
