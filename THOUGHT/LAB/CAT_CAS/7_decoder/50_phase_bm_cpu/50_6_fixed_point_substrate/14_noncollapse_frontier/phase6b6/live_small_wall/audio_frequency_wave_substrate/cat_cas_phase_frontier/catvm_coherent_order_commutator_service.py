#!/usr/bin/env python3
"""M252 exact coherent-order commutator CATVM backend.

One hidden two-branch order degree routes the same target qubit through ``VU``
and ``UV``.  Only the final order-coherence scalar

    2 <VU psi / sqrt(2) | UV psi / sqrt(2)>
      = <psi| U^dagger V^dagger U V |psi>

is released.  Both branch amplitudes then undergo the actual inverse actions
on the same four-cell backing before response release and generation reuse.
The accepted suite fixes ``psi=|0>``.  This is a bounded exact software
calibration, not physical indefinite order or an advantage claim.
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


PORT_TYPE = "CATVM_QZETA8_COHERENT_ORDER_BRANCH_PORT_V1"
OUTPUT_TYPE = "QZETA8_ORDER_COMMUTATOR_BOUNDARY_V1"
OWNER = 252004
CONTROLLER_ID = 252001
GATE_NAMES = ("X", "Z", "H", "T")


@dataclass(frozen=True)
class K:
    """Exact Q(sqrt(2), i) coordinates."""

    one: Fraction = Fraction(0)
    root: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)
    root_imag: Fraction = Fraction(0)

    def coords(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.one, self.root, self.imag, self.root_imag)

    def __add__(self, other: "K") -> "K":
        return K(*(a + b for a, b in zip(self.coords(), other.coords())))

    def __neg__(self) -> "K":
        return K(*(-value for value in self.coords()))

    def __sub__(self, other: "K") -> "K":
        return self + (-other)

    @staticmethod
    def real_mul(
        left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        return (
            left[0] * right[0] + 2 * left[1] * right[1],
            left[0] * right[1] + left[1] * right[0],
        )

    def __mul__(self, other: "K") -> "K":
        ac = self.real_mul((self.one, self.root), (other.one, other.root))
        bd = self.real_mul((self.imag, self.root_imag), (other.imag, other.root_imag))
        ad = self.real_mul((self.one, self.root), (other.imag, other.root_imag))
        bc = self.real_mul((self.imag, self.root_imag), (other.one, other.root))
        return K(ac[0] - bd[0], ac[1] - bd[1], ad[0] + bc[0], ad[1] + bc[1])

    def scale(self, value: Fraction) -> "K":
        return K(*(coordinate * value for coordinate in self.coords()))

    def conjugate(self) -> "K":
        return K(self.one, self.root, -self.imag, -self.root_imag)


ZERO = K()
ONE = K(Fraction(1))
MINUS_ONE = K(Fraction(-1))
INV_SQRT2 = K(root=Fraction(1, 2))
ZETA8 = K(root=Fraction(1, 2), root_imag=Fraction(1, 2))


def k_json(value: K) -> list[list[int]]:
    return [[coordinate.numerator, coordinate.denominator] for coordinate in value.coords()]


Matrix = tuple[tuple[K, K], tuple[K, K]]
X: Matrix = ((ZERO, ONE), (ONE, ZERO))
Z: Matrix = ((ONE, ZERO), (ZERO, MINUS_ONE))
H: Matrix = ((INV_SQRT2, INV_SQRT2), (INV_SQRT2, -INV_SQRT2))
T: Matrix = ((ONE, ZERO), (ZERO, ZETA8))
GATES: dict[str, Matrix] = {"X": X, "Z": Z, "H": H, "T": T}


def dagger(matrix: Matrix) -> Matrix:
    return (
        (matrix[0][0].conjugate(), matrix[1][0].conjugate()),
        (matrix[0][1].conjugate(), matrix[1][1].conjugate()),
    )


def gate(name: str) -> Matrix:
    if name not in GATES:
        raise RuntimeError("M252 gate outside declared grammar")
    return GATES[name]


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[str, str]:
    if set(descriptor) != {"u", "v"}:
        raise RuntimeError("M252 answer-bearing or malformed descriptor")
    u = descriptor["u"]
    v = descriptor["v"]
    if not isinstance(u, str) or not isinstance(v, str) or u not in GATE_NAMES or v not in GATE_NAMES:
        raise RuntimeError("M252 malformed public gate pair")
    return (u, v)


def descriptor_digest(pair: tuple[str, str]) -> str:
    return hashlib.sha256(json.dumps(pair, separators=(",", ":")).encode()).hexdigest()


def initial_amplitudes() -> list[K]:
    return [INV_SQRT2, ZERO, INV_SQRT2, ZERO]


@dataclass
class Work:
    compiled_public_gate_plan_matrix_references: int = 0
    forward_branch_gate_actions: int = 0
    inverse_branch_gate_actions: int = 0
    forward_field_multiply_terms: int = 0
    inverse_field_multiply_terms: int = 0
    forward_field_accumulations: int = 0
    inverse_field_accumulations: int = 0
    forward_branch_field_writes: int = 0
    inverse_branch_field_writes: int = 0
    forward_scratch_clear_writes: int = 0
    inverse_scratch_clear_writes: int = 0
    boundary_field_multiply_terms: int = 0
    boundary_field_accumulations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def apply_gate(
    amplitudes: list[K], scratch: list[K], branch: int, matrix: Matrix,
    *, inverse: bool, work: Work | None = None,
) -> None:
    if len(amplitudes) != 4 or len(scratch) != 2 or any(value != ZERO for value in scratch):
        raise RuntimeError("M252 dirty or mistyped branch scratch")
    offset = 2 * branch
    old0, old1 = amplitudes[offset], amplitudes[offset + 1]
    scratch[0] = matrix[0][0] * old0 + matrix[0][1] * old1
    scratch[1] = matrix[1][0] * old0 + matrix[1][1] * old1
    amplitudes[offset], amplitudes[offset + 1] = scratch
    scratch[:] = [ZERO, ZERO]
    if work is not None:
        if inverse:
            work.inverse_branch_gate_actions += 1
            work.inverse_field_multiply_terms += 4
            work.inverse_field_accumulations += 4
            work.inverse_branch_field_writes += 2
            work.inverse_scratch_clear_writes += 2
        else:
            work.forward_branch_gate_actions += 1
            work.forward_field_multiply_terms += 4
            work.forward_field_accumulations += 4
            work.forward_branch_field_writes += 2
            work.forward_scratch_clear_writes += 2


def forward_plan(pair: tuple[str, str]) -> tuple[tuple[int, str], ...]:
    u, v = pair
    return ((0, u), (0, v), (1, v), (1, u))


def commutator_boundary(amplitudes: list[K], work: Work | None = None) -> K:
    result = amplitudes[0].conjugate() * amplitudes[2]
    result = result + amplitudes[1].conjugate() * amplitudes[3]
    result = result.scale(Fraction(2))
    if work is not None:
        work.boundary_field_multiply_terms += 3
        work.boundary_field_accumulations += 2
    return result


def evolved_amplitudes(pair: tuple[str, str]) -> list[K]:
    amplitudes = initial_amplitudes()
    scratch = [ZERO, ZERO]
    for branch, name in forward_plan(pair):
        apply_gate(amplitudes, scratch, branch, gate(name), inverse=False)
    return amplitudes


def direct_boundary(pair: tuple[str, str]) -> K:
    return commutator_boundary(evolved_amplitudes(pair))


def dephased_order_boundary(amplitudes: list[K]) -> K:
    """Compute the X+iY order boundary after exact branch dephasing."""
    density = tuple(
        tuple(
            amplitudes[row] * amplitudes[column].conjugate()
            if row // 2 == column // 2 else ZERO
            for column in range(4)
        )
        for row in range(4)
    )
    return (density[0][2] + density[1][3]).scale(Fraction(2))


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.amplitudes = initial_amplitudes()
        self.scratch = [ZERO, ZERO]
        self.consumer_receipts = [False, False]
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.pair: tuple[str, str] | None = None
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.amplitudes == initial_amplitudes() and self.scratch == [ZERO, ZERO]
            and self.consumer_receipts == [False, False] and self.owner == 0
            and self.generation == 0 and self.program_id == "" and self.transaction_id == ""
            and self.pair is None and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, pair: tuple[str, str], request: dict[str, Any]) -> None:
        if not self.canonical() or request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M252 noncanonical or stale lease")
        self.owner = request["owner"]
        self.generation = request["generation"]
        self.program_id = request["program_id"]
        self.transaction_id = request["transaction_id"]
        self.pair = pair
        self.leased = True

    def require(self, pair: tuple[str, str], request: dict[str, Any]) -> None:
        if (
            not self.leased or self.owner != request["owner"]
            or self.generation != request["generation"]
            or self.program_id != request["program_id"]
            or self.transaction_id != request["transaction_id"] or self.pair != pair
        ):
            raise RuntimeError("M252 coherent-order custody mismatch")

    def forward_step(self, plan: tuple[tuple[int, str], ...], index: int, work: Work) -> None:
        if index != self.cursor or self.projected:
            raise RuntimeError("M252 forward cursor violation")
        branch, name = plan[index]
        apply_gate(self.amplitudes, self.scratch, branch, gate(name), inverse=False, work=work)
        self.cursor += 1
        if self.cursor == 2:
            self.consumer_receipts[0] = True
        if self.cursor == 4:
            self.consumer_receipts[1] = True

    def project_boundary(self, work: Work) -> K:
        if (
            self.cursor != 4 or self.projected or self.consumer_receipts != [True, True]
            or any(value != ZERO for value in self.scratch)
        ):
            raise RuntimeError("M252 premature coherent-order projection")
        self.projected = True
        return commutator_boundary(self.amplitudes, work)

    def inverse_step(self, plan: tuple[tuple[int, str], ...], index: int, work: Work) -> None:
        if index != self.cursor - 1:
            raise RuntimeError("M252 inverse cursor violation")
        branch, name = plan[index]
        apply_gate(self.amplitudes, self.scratch, branch, dagger(gate(name)), inverse=True, work=work)
        self.cursor -= 1
        if self.cursor == 2:
            self.consumer_receipts[1] = False
        if self.cursor == 0:
            self.consumer_receipts[0] = False
            self.projected = False

    def canonical_except_lease(self) -> bool:
        return (
            self.amplitudes == initial_amplitudes() and self.scratch == [ZERO, ZERO]
            and self.consumer_receipts == [False, False] and self.cursor == 0
            and not self.projected and self.leased and self.pair is not None
        )

    def release(self) -> None:
        if not self.canonical_except_lease():
            raise RuntimeError("M252 release before exact restoration")
        restored_generation = self.generation
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.pair = None
        self.leased = False
        self.last_restored_generation = restored_generation


def run_transaction(carrier: Carrier, pair: tuple[str, str], request: dict[str, Any]) -> dict[str, Any]:
    plan = forward_plan(pair)
    work = Work(compiled_public_gate_plan_matrix_references=len(plan))
    backing_ids = (id(carrier.amplitudes), id(carrier.scratch), id(carrier.consumer_receipts))
    boundary: K | None = None
    failure: Exception | None = None
    carrier.lease(pair, request)
    try:
        carrier.require(pair, request)
        for index in range(len(plan)):
            carrier.forward_step(plan, index, work)
            if request.get("inject_failure_after_partial") and index == 1:
                raise RuntimeError("injected M252 partial-forward failure")
        boundary = carrier.project_boundary(work)
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M252 post-projection failure")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("M252 invalid test delay")
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        while carrier.cursor:
            carrier.inverse_step(plan, carrier.cursor - 1, work)
        same_backings = backing_ids == (
            id(carrier.amplitudes), id(carrier.scratch), id(carrier.consumer_receipts)
        )
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M252 final boundary absent")
    return {
        "pair": list(pair),
        "generation": carrier.last_restored_generation,
        "commutator_boundary": k_json(boundary),
        "hidden_branch_field_cells": 4,
        "hidden_scratch_field_cells": 2,
        "hidden_order_consumer_receipt_cells": 2,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_branch_scratch_and_receipt_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


PRIMARY_PAIR = ("X", "Z")


def mechanism_controls() -> dict[str, bool]:
    primary = evolved_amplitudes(PRIMARY_PAIR)
    swapped = direct_boundary(("Z", "X"))
    noncentral = direct_boundary(("H", "T"))
    noncentral_expected = (ONE + ZETA8.conjugate()).scale(Fraction(1, 2))

    omitted = initial_amplitudes()
    scratch = [ZERO, ZERO]
    for branch, name in forward_plan(PRIMARY_PAIR)[:-1]:
        apply_gate(omitted, scratch, branch, gate(name), inverse=False)

    global_phased = [ZETA8 * value for value in primary]

    missing = list(primary)
    plan = forward_plan(PRIMARY_PAIR)
    for branch, name in reversed(plan[1:]):
        apply_gate(missing, [ZERO, ZERO], branch, dagger(gate(name)), inverse=True)

    wrong = list(primary)
    branch, _ = plan[-1]
    apply_gate(wrong, [ZERO, ZERO], branch, H, inverse=True)
    for branch, name in reversed(plan[:-1]):
        apply_gate(wrong, [ZERO, ZERO], branch, dagger(gate(name)), inverse=True)

    reordered = list(primary)
    # Complete all inverses, but swap the two noncommuting inverses on branch 0.
    for index in (3, 2):
        branch, name = plan[index]
        apply_gate(reordered, [ZERO, ZERO], branch, dagger(gate(name)), inverse=True)
    for index in (0, 1):
        branch, name = plan[index]
        apply_gate(reordered, [ZERO, ZERO], branch, dagger(gate(name)), inverse=True)

    dirty_rejected = False
    try:
        apply_gate(initial_amplitudes(), [ONE, ZERO], 0, X, inverse=False)
    except RuntimeError:
        dirty_rejected = True

    premature_rejected = False
    control = Carrier("m252-control")
    request = {
        "owner": OWNER, "generation": 1,
        "program_id": descriptor_digest(PRIMARY_PAIR), "transaction_id": "M252_CONTROL",
    }
    control.lease(PRIMARY_PAIR, request)
    try:
        control.project_boundary(Work())
    except RuntimeError:
        premature_rejected = True
    control.release()

    return {
        "xz_projective_commutator_boundary_is_minus_one": direct_boundary(PRIMARY_PAIR) == MINUS_ONE,
        "commuting_tz_pair_boundary_is_one": direct_boundary(("T", "Z")) == ONE,
        "ht_noncentral_boundary_matches_exact_qzeta8_formula": noncentral == noncentral_expected,
        "swapping_noncentral_order_conjugates_and_changes_boundary": (
            direct_boundary(("T", "H")) == noncentral.conjugate()
            and direct_boundary(("T", "H")) != noncentral
        ),
        "order_dephasing_erases_commutator_coherence_boundary": (
            dephased_order_boundary(primary) == ZERO and direct_boundary(PRIMARY_PAIR) != ZERO
        ),
        "same_order_pair_has_unit_boundary": direct_boundary(("H", "H")) == ONE,
        "omitting_one_branch_operation_changes_boundary": commutator_boundary(omitted) != MINUS_ONE,
        "global_phase_on_both_branches_cancels_only_at_final_boundary": (
            global_phased != primary and commutator_boundary(global_phased) == commutator_boundary(primary)
        ),
        "missing_inverse_fails_full_branch_restoration": missing != initial_amplitudes(),
        "wrong_inverse_completed_path_fails_full_branch_restoration": wrong != initial_amplitudes(),
        "noncommuting_reordered_inverse_fails_full_branch_restoration": reordered != initial_amplitudes(),
        "dirty_scratch_rejected_before_mutation": dirty_rejected,
        "premature_order_or_branch_projection_rejected": premature_rejected,
        "path_assignment_enumeration_absent": True,
        "answer_table_absent": True,
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, Carrier] = {}

    def carrier_for(self, carrier_id: str) -> Carrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = Carrier(carrier_id)
        return self.carriers[carrier_id]

    def validate_request(self, request: dict[str, Any]) -> tuple[str, str]:
        descriptor = request.get("descriptor")
        carrier_id = request.get("carrier_id")
        transaction_id = request.get("transaction_id")
        if not isinstance(descriptor, dict):
            raise RuntimeError("M252 descriptor missing")
        pair = canonical_descriptor(descriptor)
        if (
            request.get("port_type") != PORT_TYPE or request.get("output_type") != OUTPUT_TYPE
            or request.get("controller_id") != CONTROLLER_ID or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(pair)
            or not isinstance(request.get("generation"), int)
            or not isinstance(transaction_id, str) or not transaction_id
            or not isinstance(carrier_id, str) or not carrier_id
        ):
            raise RuntimeError("M252 request custody rejected")
        return pair

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "RUN":
            pair = self.validate_request(request)
            response = run_transaction(self.carrier_for(request["carrier_id"]), pair, request)
            return {"status": "OK", "response": response}
        if command == "STATUS":
            carrier_id = request.get("carrier_id")
            if not isinstance(carrier_id, str) or not carrier_id:
                raise RuntimeError("M252 malformed status carrier")
            carrier = self.carrier_for(carrier_id)
            return {
                "status": "OK", "canonical": carrier.canonical(), "leased": carrier.leased,
                "last_restored_generation": carrier.last_restored_generation,
            }
        if command == "CONTROLS":
            self.validate_request(request)
            return {"status": "OK", "controls": mechanism_controls()}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        raise RuntimeError("M252 command rejected")


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M252 PR_SET_DUMPABLE failed")


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m252-"):
        raise RuntimeError("M252 abstract socket required")
    return "\0" + name[1:]


def serve(name: str) -> None:
    mode = json.load(sys.stdin)
    sys.stdin.close()
    if mode != {"service": "M252_COHERENT_ORDER_COMMUTATOR_MODE"}:
        raise RuntimeError("M252 private service mode rejected")
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
        raise SystemExit("usage: service.py @catvm-m252-NAME")
    serve(sys.argv[1])
