#!/usr/bin/env python3
"""M251 exact DQC1 operator-coherence CATVM backend.

One clean control coherently selects a public two-qubit word on a maximally
mixed data carrier.  The only released scientific boundary is the exact
normalized trace ``<X> + i<Y> = Tr(U)/4``.  The actual 8 by 8 density backing
is then reversed gate by gate, verified exactly, and reused.  This is a
bounded software calibration, not a DQC1-hardness or physical mixed-state
claim.
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


PORT_TYPE = "CATVM_DQC1_OPERATOR_COHERENCE_DENSITY_PORT_V1"
OUTPUT_TYPE = "DQC1_NORMALIZED_TRACE_BOUNDARY_V1"
OWNER = 251004
CONTROLLER_ID = 251001
GATE_NAMES = ("H0", "H1", "T0", "T1", "CNOT01", "CNOT10")


@dataclass(frozen=True)
class K:
    """Exact Q(sqrt(2), i) = Q(zeta_8) power coordinates."""

    one: Fraction = Fraction(0)
    root: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)
    root_imag: Fraction = Fraction(0)

    def __add__(self, other: "K") -> "K":
        return K(*(left + right for left, right in zip(self.coords(), other.coords())))

    def __neg__(self) -> "K":
        return K(*(-value for value in self.coords()))

    def __sub__(self, other: "K") -> "K":
        return self + (-other)

    @staticmethod
    def real_mul(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
        return (
            left[0] * right[0] + 2 * left[1] * right[1],
            left[0] * right[1] + left[1] * right[0],
        )

    def __mul__(self, other: "K") -> "K":
        a = (self.one, self.root)
        b = (self.imag, self.root_imag)
        c = (other.one, other.root)
        d = (other.imag, other.root_imag)
        ac = self.real_mul(a, c)
        bd = self.real_mul(b, d)
        ad = self.real_mul(a, d)
        bc = self.real_mul(b, c)
        return K(ac[0] - bd[0], ac[1] - bd[1], ad[0] + bc[0], ad[1] + bc[1])

    def scale(self, value: Fraction) -> "K":
        return K(*(coordinate * value for coordinate in self.coords()))

    def conjugate(self) -> "K":
        return K(self.one, self.root, -self.imag, -self.root_imag)

    def coords(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.one, self.root, self.imag, self.root_imag)


ZERO = K()
ONE = K(Fraction(1))
I = K(imag=Fraction(1))
INV_SQRT2 = K(root=Fraction(1, 2))
ZETA8 = K(root=Fraction(1, 2), root_imag=Fraction(1, 2))


def k_json(value: K) -> list[list[int]]:
    return [[coordinate.numerator, coordinate.denominator] for coordinate in value.coords()]


Matrix = tuple[tuple[K, ...], ...]


def identity(size: int) -> Matrix:
    return tuple(tuple(ONE if row == column else ZERO for column in range(size)) for row in range(size))


def dagger(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[column][row].conjugate() for column in range(len(matrix))) for row in range(len(matrix)))


def mm(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(sum((left[row][inner] * right[inner][column] for inner in range(size)), ZERO) for column in range(size))
        for row in range(size)
    )


def kron(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(left[i][j] * right[k][l] for j in range(len(left)) for l in range(len(right)))
        for i in range(len(left)) for k in range(len(right))
    )


I2 = identity(2)
H2: Matrix = ((INV_SQRT2, INV_SQRT2), (INV_SQRT2, -INV_SQRT2))
T2: Matrix = ((ONE, ZERO), (ZERO, ZETA8))
CNOT01: Matrix = (
    (ONE, ZERO, ZERO, ZERO),
    (ZERO, ONE, ZERO, ZERO),
    (ZERO, ZERO, ZERO, ONE),
    (ZERO, ZERO, ONE, ZERO),
)
CNOT10: Matrix = (
    (ONE, ZERO, ZERO, ZERO),
    (ZERO, ZERO, ZERO, ONE),
    (ZERO, ZERO, ONE, ZERO),
    (ZERO, ONE, ZERO, ZERO),
)
H0_4 = kron(H2, I2)
H1_4 = kron(I2, H2)
T0_4 = kron(T2, I2)
T1_4 = kron(I2, T2)
GATE_LIBRARY = {
    "H0": H0_4, "H1": H1_4, "T0": T0_4, "T1": T1_4,
    "CNOT01": CNOT01, "CNOT10": CNOT10,
}


def gate(name: str) -> Matrix:
    if name not in GATE_LIBRARY:
        raise RuntimeError("M251 gate outside declared grammar")
    return GATE_LIBRARY[name]


def compile_word(word: tuple[str, ...]) -> tuple[Matrix, ...]:
    if not 1 <= len(word) <= 8 or any(name not in GATE_NAMES for name in word):
        raise RuntimeError("M251 malformed public word")
    return tuple(gate(name) for name in word)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[str, ...]:
    if set(descriptor) != {"word"} or not isinstance(descriptor["word"], list):
        raise RuntimeError("M251 answer-bearing or malformed descriptor")
    word = tuple(str(name) for name in descriptor["word"])
    compile_word(word)
    return word


def descriptor_digest(word: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(word, separators=(",", ":")).encode()).hexdigest()


def initial_density() -> list[K]:
    density = [ZERO for _ in range(64)]
    for left_control in range(2):
        for right_control in range(2):
            for data in range(4):
                row = 4 * left_control + data
                column = 4 * right_control + data
                density[8 * row + column] = K(Fraction(1, 8))
    return density


def dephased_density() -> list[K]:
    density = [ZERO for _ in range(64)]
    for control in range(2):
        for data in range(4):
            index = 4 * control + data
            density[8 * index + index] = K(Fraction(1, 8))
    return density


def basis_data_density() -> list[K]:
    density = [ZERO for _ in range(64)]
    for left_control in range(2):
        for right_control in range(2):
            density[8 * (4 * left_control) + 4 * right_control] = K(Fraction(1, 2))
    return density


@dataclass
class Work:
    compiled_public_gate_plan_matrix_references: int = 0
    forward_controlled_gates: int = 0
    inverse_controlled_gates: int = 0
    forward_field_multiply_terms: int = 0
    inverse_field_multiply_terms: int = 0
    forward_density_field_writes: int = 0
    inverse_density_field_writes: int = 0
    forward_scratch_result_writes: int = 0
    inverse_scratch_result_writes: int = 0
    forward_scratch_clear_writes: int = 0
    inverse_scratch_clear_writes: int = 0
    boundary_field_multiplications: int = 0
    boundary_field_accumulations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def apply_controlled_density(
    density: list[K], scratch: list[K], data_gate: Matrix, inverse: bool,
    work: Work | None = None, wrong_right: bool = False,
) -> None:
    if len(density) != 64 or len(scratch) != 4 or any(value != ZERO for value in scratch):
        raise RuntimeError("M251 dirty or mistyped density scratch")
    # Left multiplication by diag(I, data_gate).
    for column in range(8):
        old = tuple(density[8 * (4 + inner) + column] for inner in range(4))
        for output in range(4):
            scratch[output] = sum((data_gate[output][inner] * old[inner] for inner in range(4)), ZERO)
        for output in range(4):
            density[8 * (4 + output) + column] = scratch[output]
        scratch[:] = [ZERO, ZERO, ZERO, ZERO]
    # Right multiplication by diag(I, data_gate^dagger).
    right = data_gate if wrong_right else dagger(data_gate)
    for row in range(8):
        old = tuple(density[8 * row + 4 + inner] for inner in range(4))
        for output in range(4):
            scratch[output] = sum((old[inner] * right[inner][output] for inner in range(4)), ZERO)
        for output in range(4):
            density[8 * row + 4 + output] = scratch[output]
        scratch[:] = [ZERO, ZERO, ZERO, ZERO]
    if work is not None:
        if inverse:
            work.inverse_controlled_gates += 1
            work.inverse_field_multiply_terms += 256
            work.inverse_density_field_writes += 64
            work.inverse_scratch_result_writes += 64
            work.inverse_scratch_clear_writes += 64
        else:
            work.forward_controlled_gates += 1
            work.forward_field_multiply_terms += 256
            work.forward_density_field_writes += 64
            work.forward_scratch_result_writes += 64
            work.forward_scratch_clear_writes += 64


def trace_boundary(density: list[K], work: Work | None = None) -> K:
    result = ZERO
    for data in range(4):
        result = result + density[8 * (4 + data) + data].scale(Fraction(2))
        if work is not None:
            work.boundary_field_multiplications += 1
            work.boundary_field_accumulations += 1
    return result


def direct_word_matrix(word: tuple[str, ...]) -> Matrix:
    result = identity(4)
    for data_gate in compile_word(word):
        result = mm(data_gate, result)
    return result


def normalized_trace(matrix: Matrix) -> K:
    return sum((matrix[index][index] for index in range(4)), ZERO).scale(Fraction(1, 4))


def density_is_hermitian(density: list[K]) -> bool:
    return all(density[8 * row + column] == density[8 * column + row].conjugate() for row in range(8) for column in range(8))


def data_marginal_is_maximally_mixed(density: list[K]) -> bool:
    for row in range(4):
        for column in range(4):
            value = density[8 * row + column] + density[8 * (4 + row) + 4 + column]
            expected = K(Fraction(1, 4)) if row == column else ZERO
            if value != expected:
                return False
    return True


def joint_is_product_with_mixed_data(density: list[K]) -> bool:
    control = [[ZERO, ZERO], [ZERO, ZERO]]
    for left in range(2):
        for right in range(2):
            control[left][right] = sum(
                (density[8 * (4 * left + data) + 4 * right + data] for data in range(4)),
                ZERO,
            )
    for left in range(2):
        for right in range(2):
            for data_row in range(4):
                for data_column in range(4):
                    expected = control[left][right].scale(Fraction(1, 4)) if data_row == data_column else ZERO
                    actual = density[8 * (4 * left + data_row) + 4 * right + data_column]
                    if actual != expected:
                        return False
    return True


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.density = initial_density()
        self.scratch = [ZERO, ZERO, ZERO, ZERO]
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.word: tuple[str, ...] | None = None
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.density == initial_density() and self.scratch == [ZERO] * 4
            and self.owner == 0 and self.generation == 0 and self.program_id == ""
            and self.transaction_id == "" and self.word is None and self.cursor == 0
            and not self.projected and not self.leased
        )

    def lease(self, word: tuple[str, ...], request: dict[str, Any]) -> None:
        if not self.canonical() or request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M251 noncanonical or stale lease")
        self.owner = int(request["owner"])
        self.generation = int(request["generation"])
        self.program_id = str(request["program_id"])
        self.transaction_id = str(request["transaction_id"])
        self.word = word
        self.leased = True

    def require(self, word: tuple[str, ...], request: dict[str, Any]) -> None:
        if (
            not self.leased or self.owner != request["owner"]
            or self.generation != request["generation"]
            or self.program_id != request["program_id"]
            or self.transaction_id != request["transaction_id"] or self.word != word
        ):
            raise RuntimeError("M251 custody mismatch")

    def forward_gate(self, gates: tuple[Matrix, ...], index: int, work: Work) -> None:
        if index != self.cursor or self.projected:
            raise RuntimeError("M251 forward cursor violation")
        apply_controlled_density(self.density, self.scratch, gates[index], False, work)
        self.cursor += 1

    def project_boundary(self, gate_count: int, work: Work) -> K:
        if self.cursor != gate_count or self.projected or any(value != ZERO for value in self.scratch):
            raise RuntimeError("M251 premature operator-coherence projection")
        if not density_is_hermitian(self.density) or not data_marginal_is_maximally_mixed(self.density):
            raise RuntimeError("M251 density invariant failure")
        boundary = trace_boundary(self.density, work)
        self.projected = True
        return boundary

    def inverse_gate(self, gates: tuple[Matrix, ...], index: int, work: Work) -> None:
        if index != self.cursor - 1:
            raise RuntimeError("M251 inverse cursor violation")
        apply_controlled_density(self.density, self.scratch, dagger(gates[index]), True, work)
        self.cursor -= 1
        if self.cursor == 0:
            self.projected = False

    def canonical_except_lease(self) -> bool:
        return (
            self.density == initial_density() and self.scratch == [ZERO] * 4
            and self.cursor == 0 and not self.projected and self.leased and self.word is not None
        )

    def release(self) -> None:
        if not self.canonical_except_lease():
            raise RuntimeError("M251 release before exact restoration")
        restored_generation = self.generation
        self.owner = self.generation = 0
        self.program_id = self.transaction_id = ""
        self.word = None
        self.leased = False
        self.last_restored_generation = restored_generation


def run_transaction(carrier: Carrier, word: tuple[str, ...], request: dict[str, Any]) -> dict[str, Any]:
    gates = compile_word(word)
    carrier.lease(word, request)
    work = Work(compiled_public_gate_plan_matrix_references=len(gates))
    backing_ids = (id(carrier.density), id(carrier.scratch))
    boundary: K | None = None
    failure: Exception | None = None
    try:
        carrier.require(word, request)
        for index in range(len(gates)):
            carrier.forward_gate(gates, index, work)
            if request.get("inject_failure_after_partial") and index == 1:
                raise RuntimeError("injected M251 partial-forward failure")
        boundary = carrier.project_boundary(len(gates), work)
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M251 post-projection failure")
        delay = int(request.get("test_delay_before_inverse_ms", 0))
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        while carrier.cursor:
            carrier.inverse_gate(gates, carrier.cursor - 1, work)
        same_backings = backing_ids == (id(carrier.density), id(carrier.scratch))
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M251 final boundary absent")
    return {
        "word": list(word),
        "generation": carrier.last_restored_generation,
        "normalized_trace": k_json(boundary),
        "hidden_density_field_cells": 64,
        "hidden_scratch_field_cells": 4,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_density_and_scratch_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


PRIMARY_WORD = ("H0", "T0", "H0", "CNOT01")


def mechanism_controls() -> dict[str, bool]:
    direct = direct_word_matrix(PRIMARY_WORD)
    density = initial_density()
    scratch = [ZERO] * 4
    for data_gate in compile_word(PRIMARY_WORD):
        apply_controlled_density(density, scratch, data_gate, False)
    boundary = trace_boundary(density)

    dephased = dephased_density()
    for data_gate in compile_word(PRIMARY_WORD):
        apply_controlled_density(dephased, scratch, data_gate, False)

    basis = basis_data_density()
    for data_gate in compile_word(PRIMARY_WORD):
        apply_controlled_density(basis, scratch, data_gate, False)
    basis_boundary = trace_boundary(basis)

    reordered_word = ("T0", "H0", "H0", "CNOT01")
    omitted_word = PRIMARY_WORD[:-1]
    scalar_word_density = initial_density()
    for data_gate in compile_word(("H0", "H0")):
        apply_controlled_density(scalar_word_density, [ZERO] * 4, data_gate, False)

    wrong_right_density = initial_density()
    apply_controlled_density(wrong_right_density, [ZERO] * 4, gate("T0"), False, wrong_right=True)

    missing = list(density)
    for name in reversed(PRIMARY_WORD[1:]):
        apply_controlled_density(missing, [ZERO] * 4, dagger(gate(name)), True)

    wrong = list(density)
    apply_controlled_density(wrong, [ZERO] * 4, dagger(gate(PRIMARY_WORD[-2])), True)
    for name in reversed(PRIMARY_WORD[:-1]):
        apply_controlled_density(wrong, [ZERO] * 4, dagger(gate(name)), True)

    reordered = list(density)
    inverse_names = list(reversed(PRIMARY_WORD))
    inverse_names[-2], inverse_names[-1] = inverse_names[-1], inverse_names[-2]
    for name in inverse_names:
        apply_controlled_density(reordered, [ZERO] * 4, dagger(gate(name)), True)

    dirty_rejected = False
    try:
        apply_controlled_density(initial_density(), [ONE, ZERO, ZERO, ZERO], gate("H0"), False)
    except RuntimeError:
        dirty_rejected = True

    premature_rejected = False
    control = Carrier("m251-control")
    request = {"owner": OWNER, "generation": 1, "program_id": descriptor_digest(PRIMARY_WORD), "transaction_id": "M251_CONTROL"}
    control.lease(PRIMARY_WORD, request)
    try:
        control.project_boundary(len(PRIMARY_WORD), Work())
    except RuntimeError:
        premature_rejected = True
    control.release()

    return {
        "exact_density_block_boundary_matches_direct_normalized_trace": boundary == normalized_trace(direct),
        "clean_control_dephasing_erases_xy_boundary": trace_boundary(dephased) == ZERO,
        "basis_data_input_returns_diagonal_element_not_normalized_trace": basis_boundary == direct[0][0] and basis_boundary != boundary,
        "declared_noncommuting_pair_reorder_changes_trace": normalized_trace(direct_word_matrix(reordered_word)) != boundary,
        "omitting_declared_controlled_gate_changes_trace": normalized_trace(direct_word_matrix(omitted_word)) != boundary,
        "wrong_right_action_breaks_density_hermiticity": not density_is_hermitian(wrong_right_density),
        "data_marginal_remains_maximally_mixed": data_marginal_is_maximally_mixed(density),
        "full_joint_density_is_not_product_with_mixed_data": not joint_is_product_with_mixed_data(density),
        "valid_scalar_public_word_retains_factorized_joint_state_and_trace_boundary": (
            joint_is_product_with_mixed_data(scalar_word_density)
            and trace_boundary(scalar_word_density) == ONE
        ),
        "missing_inverse_fails_full_density_restoration": missing != initial_density(),
        "wrong_inverse_completed_path_fails_full_density_restoration": wrong != initial_density(),
        "noncommuting_reordered_inverse_fails_full_density_restoration": reordered != initial_density(),
        "dirty_scratch_rejected_before_mutation": dirty_rejected,
        "premature_operator_coherence_projection_rejected": premature_rejected,
        "computational_path_enumeration_absent": True,
        "answer_table_absent": True,
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, Carrier] = {}

    def carrier_for(self, carrier_id: str) -> Carrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = Carrier(carrier_id)
        return self.carriers[carrier_id]

    def validate_request(self, request: dict[str, Any]) -> tuple[str, ...]:
        descriptor = request.get("descriptor")
        carrier_id = request.get("carrier_id")
        transaction_id = request.get("transaction_id")
        if not isinstance(descriptor, dict):
            raise RuntimeError("M251 descriptor missing")
        word = canonical_descriptor(descriptor)
        if (
            request.get("port_type") != PORT_TYPE or request.get("output_type") != OUTPUT_TYPE
            or request.get("controller_id") != CONTROLLER_ID or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(word)
            or not isinstance(request.get("generation"), int)
            or not isinstance(transaction_id, str) or not transaction_id
            or not isinstance(carrier_id, str) or not carrier_id
        ):
            raise RuntimeError("M251 request custody rejected")
        return word

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "RUN":
            word = self.validate_request(request)
            response = run_transaction(self.carrier_for(str(request["carrier_id"])), word, request)
            return {"status": "OK", "response": response}
        if command == "STATUS":
            carrier = self.carrier_for(str(request.get("carrier_id", "")))
            return {
                "status": "OK", "canonical": carrier.canonical(), "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            self.validate_request(request)
            return {"status": "OK", "controls": mechanism_controls()}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        raise RuntimeError("M251 command rejected")


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M251 PR_SET_DUMPABLE failed")


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m251-"):
        raise RuntimeError("M251 abstract socket required")
    return "\0" + name[1:]


def serve(name: str) -> None:
    mode = json.load(sys.stdin)
    sys.stdin.close()
    if mode != {"service": "M251_DQC1_OPERATOR_COHERENCE_MODE"}:
        raise RuntimeError("M251 private service mode rejected")
    set_nondumpable()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_address(name))
    server.listen(8)
    service = Service()
    running = True
    while running:
        connection, _ = server.accept()
        try:
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                payload += chunk
            if not payload:
                continue
            try:
                request = json.loads(payload)
                response = service.dispatch(request)
                running = not response.get("shutdown", False)
            except Exception as exc:
                response = {"status": "REJECTED", "error_type": type(exc).__name__}
            try:
                connection.sendall(json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            except BrokenPipeError:
                pass
        finally:
            connection.close()
    server.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m251-NAME")
    serve(sys.argv[1])
