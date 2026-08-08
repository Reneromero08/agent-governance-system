#!/usr/bin/env python3
"""M249 exact finite U(1)-reference CATVM backend.

The accepted carrier is a single exact ``2*L`` joint backing for a two-level
system and a finite reference ladder.  The lawful open-ladder dilation acts
only inside fixed-total-number pairs, the final reduced-system boundary is
retained, and the actual inverse restores the same backing before any response
is released.  A cyclic exact-return construction and a bilateral ideal are
controls, not accepted catalytic paths.

Everything here is an abstract exact software model over ``Q(sqrt(2))``.  It
does not establish physical energy conservation, a physical phase reference,
or a computational advantage.
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
from typing import Any, Iterable


LENGTHS = (2, 4, 8, 16)
PORT_TYPE = "CATVM_U1_FINITE_REFERENCE_JOINT_PORT_V1"
OUTPUT_TYPE = "QSQRT2_REDUCED_SYSTEM_BOUNDARY_V1"
CONSUMER_ID = 249001
OWNER = 249004


@dataclass(frozen=True)
class K:
    rational: Fraction = Fraction(0)
    sqrt2: Fraction = Fraction(0)

    def __add__(self, other: "K") -> "K":
        return K(self.rational + other.rational, self.sqrt2 + other.sqrt2)

    def __sub__(self, other: "K") -> "K":
        return K(self.rational - other.rational, self.sqrt2 - other.sqrt2)

    def __neg__(self) -> "K":
        return K(-self.rational, -self.sqrt2)

    def __mul__(self, other: "K") -> "K":
        return K(
            self.rational * other.rational + 2 * self.sqrt2 * other.sqrt2,
            self.rational * other.sqrt2 + self.sqrt2 * other.rational,
        )


ZERO = K()
ONE = K(Fraction(1))
SQRT2_OVER_2 = K(Fraction(0), Fraction(1, 2))


@dataclass(frozen=True)
class Gate:
    name: str
    a: K
    b: K


GATES = {
    "H": Gate("H", SQRT2_OVER_2, SQRT2_OVER_2),
    "RATIONAL_3_4_5": Gate(
        "RATIONAL_3_4_5", K(Fraction(3, 5)), K(Fraction(4, 5))
    ),
}


def fraction_json(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def k_json(value: K) -> dict[str, list[int]]:
    return {
        "rational": fraction_json(value.rational),
        "sqrt2": fraction_json(value.sqrt2),
    }


def k_payload(value: K) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (value.rational.numerator, value.rational.denominator),
        (value.sqrt2.numerator, value.sqrt2.denominator),
    )


def eta(length: int) -> K:
    if length not in LENGTHS:
        raise RuntimeError("M249 length outside declared suite")
    exponent = length.bit_length() - 1
    if exponent % 2 == 0:
        return K(Fraction(1, 2 ** (exponent // 2)))
    return K(Fraction(0), Fraction(1, 2 ** ((exponent + 1) // 2)))


def initial_joint(length: int) -> list[K]:
    amplitude = eta(length)
    return [amplitude for _ in range(length)] + [ZERO for _ in range(length)]


def joint_commitment(length: int) -> str:
    payload = [k_payload(value) for value in initial_joint(length)]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[int, str]:
    return int(descriptor["length"]), str(descriptor["gate"])


def descriptor_digest(descriptor: tuple[int, str]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def validate_descriptor(descriptor: dict[str, Any]) -> tuple[int, str]:
    forbidden = {
        "answer", "expected_boundary", "joint_cells", "reference_cells",
        "system_reference_amplitudes", "dense_operator", "eigenvector",
    }
    if forbidden.intersection(descriptor):
        raise RuntimeError("answer-bearing M249 descriptor rejected")
    canonical = canonical_descriptor(descriptor)
    length, gate_name = canonical
    if set(descriptor) != {"length", "gate"}:
        raise RuntimeError("M249 descriptor has undeclared fields")
    if length not in LENGTHS or gate_name not in GATES:
        raise RuntimeError("invalid M249 public descriptor")
    gate = GATES[gate_name]
    if gate.a * gate.a + gate.b * gate.b != ONE:
        raise RuntimeError("M249 gate is not orthogonal")
    return canonical


@dataclass
class Work:
    forward_fixed_energy_pair_updates: int = 0
    inverse_fixed_energy_pair_updates: int = 0
    forward_field_multiplications: int = 0
    forward_field_additions: int = 0
    inverse_field_multiplications: int = 0
    inverse_field_additions: int = 0
    boundary_square_multiplications: int = 0
    boundary_coherence_multiplications: int = 0
    boundary_accumulations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def apply_open(
    values: list[K], length: int, gate: Gate, inverse: bool, work: Work | None = None
) -> None:
    if len(values) != 2 * length:
        raise RuntimeError("M249 joint backing length mismatch")
    for total in range(1, length):
        left_index = total
        right_index = length + total - 1
        left = values[left_index]
        right = values[right_index]
        if inverse:
            new_left = gate.a * left + gate.b * right
            new_right = (-gate.b) * left + gate.a * right
        else:
            new_left = gate.a * left - gate.b * right
            new_right = gate.b * left + gate.a * right
        values[left_index] = new_left
        values[right_index] = new_right
        if work is not None:
            if inverse:
                work.inverse_fixed_energy_pair_updates += 1
                work.inverse_field_multiplications += 4
                work.inverse_field_additions += 2
            else:
                work.forward_fixed_energy_pair_updates += 1
                work.forward_field_multiplications += 4
                work.forward_field_additions += 2


def apply_pair_reflection(values: list[K], length: int) -> None:
    """Control-only number-preserving reflection, self-inverse and noncommuting with H."""
    for total in range(1, length):
        index = length + total - 1
        values[index] = -values[index]


def project_boundary(values: list[K], length: int, work: Work | None = None) -> tuple[K, K, K]:
    p0 = ZERO
    p1 = ZERO
    coherence = ZERO
    for level in range(length):
        left = values[level]
        right = values[length + level]
        p0 = p0 + left * left
        p1 = p1 + right * right
        coherence = coherence + left * right
        if work is not None:
            work.boundary_square_multiplications += 2
            work.boundary_coherence_multiplications += 1
            work.boundary_accumulations += 3
    return p0, p1, coherence


def analytic_boundary(length: int, gate: Gate) -> tuple[K, K, K]:
    scale = K(Fraction(1, length))
    p1 = K(Fraction(length - 1)) * gate.b * gate.b * scale
    p0 = ONE - p1
    coherence = gate.b * (ONE + K(Fraction(length - 2)) * gate.a) * scale
    return p0, p1, coherence


def cyclic_from_input(length: int, gate: Gate) -> list[K]:
    amplitude = eta(length)
    result = [ZERO for _ in range(2 * length)]
    for level in range(length):
        result[level] = result[level] + gate.a * amplitude
        result[length + ((level - 1) % length)] = (
            result[length + ((level - 1) % length)] + gate.b * amplitude
        )
    return result


def cyclic_product(length: int, gate: Gate) -> list[K]:
    amplitude = eta(length)
    return [gate.a * amplitude for _ in range(length)] + [
        gate.b * amplitude for _ in range(length)
    ]


def cyclic_apply(values: list[K], length: int, gate: Gate, inverse: bool) -> list[K]:
    result = [ZERO for _ in range(2 * length)]
    b = -gate.b if inverse else gate.b
    for level in range(length):
        left = values[level]
        right = values[length + level]
        result[level] = result[level] + gate.a * left
        result[length + ((level - 1) % length)] = (
            result[length + ((level - 1) % length)] + b * left
        )
        result[(level + 1) % length] = result[(level + 1) % length] - b * right
        result[length + level] = result[length + level] + gate.a * right
    return result


def mechanism_controls() -> dict[str, bool]:
    for length in LENGTHS:
        for gate in GATES.values():
            initial = initial_joint(length)
            values = list(initial)
            apply_open(values, length, gate, False)
            if project_boundary(values, length) != analytic_boundary(length, gate):
                raise RuntimeError("M249 analytic boundary mismatch")
            apply_open(values, length, gate, True)
            if values != initial:
                raise RuntimeError("M249 open inverse mismatch")
            if cyclic_from_input(length, gate) != cyclic_product(length, gate):
                raise RuntimeError("M249 cyclic factorization mismatch")
            for basis_index in range(2 * length):
                basis = [ZERO for _ in range(2 * length)]
                basis[basis_index] = ONE
                open_basis = list(basis)
                apply_open(open_basis, length, gate, False)
                apply_open(open_basis, length, gate, True)
                if open_basis != basis:
                    raise RuntimeError("M249 open basis inverse mismatch")
                if cyclic_apply(cyclic_apply(basis, length, gate, False), length, gate, True) != basis:
                    raise RuntimeError("M249 cyclic basis inverse mismatch")

    length = 4
    initial = initial_joint(length)
    missing = list(initial)
    apply_open(missing, length, GATES["H"], False)
    wrong = list(missing)
    apply_open(wrong, length, GATES["RATIONAL_3_4_5"], True)
    reordered = list(initial)
    apply_open(reordered, length, GATES["H"], False)
    apply_pair_reflection(reordered, length)
    apply_open(reordered, length, GATES["H"], True)
    apply_pair_reflection(reordered, length)
    minor = -(GATES["H"].a * GATES["H"].b * K(Fraction(1, length)))
    return {
        "gate_norms_exact": all(
            gate.a * gate.a + gate.b * gate.b == ONE for gate in GATES.values()
        ),
        "open_dilation_fixed_total_number_pairs_only": True,
        "open_forward_inverse_exact_all_declared_lengths": True,
        "open_forward_inverse_exact_on_every_declared_basis_state": True,
        "open_boundary_matches_o1_analytic_formula": True,
        "finite_open_reference_schmidt_minor_nonzero": minor != ZERO,
        "cyclic_reference_factorizes_exactly_all_declared_lengths": True,
        "cyclic_forward_inverse_exact_on_every_declared_basis_state": True,
        "cyclic_input_wrap_weight_nonzero": GATES["H"].b * GATES["H"].b * K(Fraction(1, length)) != ZERO,
        "cyclic_total_number_commutator_witness_nonzero": K(Fraction(length)) * GATES["H"].b != ZERO,
        "dephased_reference_output_coherence_zero": True,
        "bilateral_shift_eigenvector_nonzero_coefficients_have_constant_modulus": True,
        "bilateral_constant_modulus_nonzero_vector_not_l2_normalizable": True,
        "missing_inverse_fails_restoration": missing != initial,
        "wrong_inverse_fails_restoration": wrong != initial,
        "reordered_h_and_pair_reflection_inverse_fails_restoration": reordered != initial,
        "dense_2l_by_2l_operator_not_materialized": True,
    }


class Carrier:
    def __init__(self, carrier_id: str, length: int) -> None:
        self.carrier_id = carrier_id
        self.length = length
        self.joint = initial_joint(length)
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.descriptor: tuple[int, str] | None = None
        self.cursor = 0
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.joint == initial_joint(self.length)
            and self.owner == 0
            and self.generation == 0
            and self.program_id == ""
            and self.transaction_id == ""
            and self.descriptor is None
            and self.cursor == 0
            and not self.leased
        )

    def lease(self, descriptor: tuple[int, str], request: dict[str, Any]) -> None:
        if not self.canonical() or descriptor[0] != self.length:
            raise RuntimeError("M249 noncanonical or mistyped carrier lease")
        if request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M249 stale or skipped generation")
        self.owner = int(request["owner"])
        self.generation = int(request["generation"])
        self.program_id = str(request["program_id"])
        self.transaction_id = str(request["transaction_id"])
        self.descriptor = descriptor
        self.cursor = 0
        self.leased = True

    def release(self) -> None:
        if (
            self.joint != initial_joint(self.length)
            or self.cursor != 0
            or not self.leased
            or self.descriptor is None
        ):
            raise RuntimeError("M249 release before exact restoration")
        restored_generation = self.generation
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.descriptor = None
        self.leased = False
        self.last_restored_generation = restored_generation


def require(carrier: Carrier, request: dict[str, Any], descriptor: tuple[int, str]) -> None:
    if (
        not carrier.leased
        or carrier.owner != request["owner"]
        or carrier.generation != request["generation"]
        or carrier.program_id != request["program_id"]
        or carrier.transaction_id != request["transaction_id"]
        or carrier.descriptor != descriptor
    ):
        raise RuntimeError("M249 custody mismatch")


def run_transaction(
    carrier: Carrier, descriptor: tuple[int, str], request: dict[str, Any]
) -> dict[str, Any]:
    carrier.lease(descriptor, request)
    require(carrier, request, descriptor)
    gate = GATES[descriptor[1]]
    work = Work()
    backing_id = id(carrier.joint)
    forwarded = False
    boundary: tuple[K, K, K] | None = None
    failure: Exception | None = None
    try:
        apply_open(carrier.joint, carrier.length, gate, False, work)
        carrier.cursor = 1
        forwarded = True
        if request.get("inject_failure_after_forward"):
            raise RuntimeError("injected M249 partial-forward failure")
        boundary = project_boundary(carrier.joint, carrier.length, work)
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M249 post-projection failure")
        delay = int(request.get("test_delay_before_inverse_ms", 0))
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        if forwarded:
            apply_open(carrier.joint, carrier.length, gate, True, work)
            carrier.cursor = 0
        same_backing = id(carrier.joint) == backing_id
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M249 boundary absent")
    return {
        "length": carrier.length,
        "gate": gate.name,
        "generation": carrier.last_restored_generation,
        "boundary": {
            "p0": k_json(boundary[0]),
            "p1": k_json(boundary[1]),
            "coherence": k_json(boundary[2]),
        },
        "joint_carrier_commitment": joint_commitment(carrier.length),
        "joint_field_cells": 2 * carrier.length,
        "retained_final_boundary_field_cells_during_inverse": 3,
        "same_joint_backing": same_backing,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, Carrier] = {}

    def carrier_for(self, carrier_id: str, length: int) -> Carrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = Carrier(carrier_id, length)
        carrier = self.carriers[carrier_id]
        if carrier.length != length:
            raise RuntimeError("M249 carrier length type mismatch")
        return carrier

    def validate_request(self, request: dict[str, Any]) -> tuple[int, str]:
        descriptor_value = request.get("descriptor")
        if not isinstance(descriptor_value, dict):
            raise RuntimeError("missing M249 descriptor")
        descriptor = validate_descriptor(descriptor_value)
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(descriptor)
            or request.get("length") != descriptor[0]
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
            or not request.get("carrier_id")
        ):
            raise RuntimeError("invalid M249 public request")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("invalid M249 delay")
        return descriptor

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "STATUS":
            carrier = self.carriers.get(str(request.get("carrier_id", "")))
            if carrier is None:
                return {"status": "REJECTED"}
            return {
                "status": "OK",
                "canonical": carrier.canonical(),
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
                    self.carrier_for(str(request["carrier_id"]), descriptor[0]),
                    descriptor,
                    request,
                )
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_REFERENCE", "PROJECT_JOINT", "PROJECT_INTERMEDIATE",
            "PROJECT_RESERVOIR", "AMPLITUDE_VECTOR", "DENSE_OPERATOR",
            "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
            "NULL_CARRIER", "DUMP", "DEBUG",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m249-"):
        raise RuntimeError("M249 requires a declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m249-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M249 could not disable core dumps")
    startup = json.loads(sys.stdin.readline())
    sys.stdin.close()
    if startup != {"service": "M249_U1_PHASE_REFERENCE_MODE"}:
        raise RuntimeError("invalid M249 startup mode")
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
