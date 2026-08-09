#!/usr/bin/env python3
"""M253 exact four-port Grassmann-Gaussian open-relation CATVM backend.

The resident relation is ``lambda * exp(theta^T A theta / 2)`` over
``Q(zeta8)``, with four typed Grassmann ports and antisymmetric ``A``.  Six
independent matrix entries plus ``lambda`` are retained.  Public relation
intersection adds antisymmetric coefficients and multiplies the scalar.
Formal four-port Berezin/Fourier closure is the exact involution

    (lambda, A) -> (lambda * Pf(A), A^-1)

on the non-singular chart.  Only the final top-form coefficient
``lambda*Pf(A)`` is released, after which the public program is reversed on
the same seven-cell backing.  This is formal exact software relation geometry,
not physical fermionic execution or an advantage claim.
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


PORT_TYPE = "CATVM_QZETA8_GRASSMANN_GAUSSIAN_FOUR_PORT_RELATION_V1"
OUTPUT_TYPE = "QZETA8_GRASSMANN_TOP_FORM_BOUNDARY_V1"
OWNER = 253004
CONTROLLER_ID = 253001
PORT_ORDER = ("THETA0", "THETA1", "THETA2", "THETA3")


@dataclass(frozen=True)
class K:
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

    def inverse(self) -> "K":
        # (a+ib)^-1=(a-ib)/(a^2+b^2), with a,b in Q(sqrt(2)).
        a = (self.one, self.root)
        b = (self.imag, self.root_imag)
        aa = self.real_mul(a, a)
        bb = self.real_mul(b, b)
        norm = (aa[0] + bb[0], aa[1] + bb[1])
        denominator = norm[0] * norm[0] - 2 * norm[1] * norm[1]
        if denominator == 0:
            raise ZeroDivisionError("M253 zero field inverse")
        inv_norm = (norm[0] / denominator, -norm[1] / denominator)
        real = self.real_mul(a, inv_norm)
        imag = self.real_mul((-b[0], -b[1]), inv_norm)
        return K(real[0], real[1], imag[0], imag[1])


ZERO = K()
ONE = K(Fraction(1))
ZETA8 = K(root=Fraction(1, 2), root_imag=Fraction(1, 2))


def k_json(value: K) -> list[list[int]]:
    return [[coordinate.numerator, coordinate.denominator] for coordinate in value.coords()]


def parse_public_k(value: object) -> K:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("M253 public field coordinate rejected")
    coordinates: list[Fraction] = []
    for item in value:
        if (
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in item)
            or abs(item[0]) > 8 or not 1 <= item[1] <= 8
        ):
            raise RuntimeError("M253 public rational field coordinate rejected")
        coordinates.append(Fraction(item[0], item[1]))
    return K(*coordinates)


Coefficients = tuple[K, K, K, K, K, K]


def pfaffian(coefficients: Coefficients) -> K:
    a, b, c, d, e, f = coefficients
    return a * f - b * e + c * d


def inverse_coefficients(coefficients: Coefficients, inverse_pfaffian: K) -> Coefficients:
    a, b, c, d, e, f = coefficients
    return (
        -f * inverse_pfaffian,
        e * inverse_pfaffian,
        -d * inverse_pfaffian,
        -c * inverse_pfaffian,
        b * inverse_pfaffian,
        -a * inverse_pfaffian,
    )


@dataclass(frozen=True)
class Module:
    op: str
    mu: K = ONE
    coefficients: Coefficients = (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)


def canonical_program(descriptor: dict[str, Any]) -> tuple[Module, ...]:
    if set(descriptor) != {"ports", "modules"} or descriptor.get("ports") != list(PORT_ORDER):
        raise RuntimeError("M253 typed port order or descriptor rejected")
    modules = descriptor.get("modules")
    if not isinstance(modules, list) or not 1 <= len(modules) <= 8:
        raise RuntimeError("M253 public program length rejected")
    compiled: list[Module] = []
    for item in modules:
        if not isinstance(item, dict) or "op" not in item:
            raise RuntimeError("M253 malformed module")
        if item["op"] == "FOURIER":
            if set(item) != {"op"}:
                raise RuntimeError("M253 answer-bearing Fourier descriptor")
            compiled.append(Module("FOURIER"))
            continue
        if item["op"] != "INTERSECT" or set(item) != {"op", "mu", "coefficients"}:
            raise RuntimeError("M253 undeclared or malformed relation module")
        coefficients = item["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != 6:
            raise RuntimeError("M253 relation coefficient count rejected")
        mu = parse_public_k(item["mu"])
        if mu == ZERO:
            raise RuntimeError("M253 zero relation scalar rejected")
        compiled.append(Module("INTERSECT", mu, tuple(parse_public_k(value) for value in coefficients)))
    return tuple(compiled)


def canonical_descriptor_json(program: tuple[Module, ...]) -> str:
    encoded: list[dict[str, object]] = []
    for module in program:
        if module.op == "FOURIER":
            encoded.append({"op": "FOURIER"})
        else:
            encoded.append({
                "op": "INTERSECT",
                "mu": k_json(module.mu),
                "coefficients": [k_json(coefficient) for coefficient in module.coefficients],
            })
    return json.dumps({"ports": list(PORT_ORDER), "modules": encoded}, sort_keys=True, separators=(",", ":"))


def descriptor_digest(program: tuple[Module, ...]) -> str:
    return hashlib.sha256(canonical_descriptor_json(program).encode()).hexdigest()


def initial_cells() -> list[K]:
    # lambda=1 and Pf(A)=1 for a=f=1.
    return [ONE, ONE, ZERO, ZERO, ZERO, ZERO, ONE]


@dataclass
class Work:
    compiled_public_module_plan_references: int = 0
    compiled_public_intersection_field_cells: int = 0
    forward_intersections: int = 0
    inverse_intersections: int = 0
    forward_fourier_closures: int = 0
    inverse_fourier_closures: int = 0
    forward_field_multiplications: int = 0
    inverse_field_multiplications: int = 0
    forward_field_accumulations: int = 0
    inverse_field_accumulations: int = 0
    forward_field_inversions: int = 0
    inverse_field_inversions: int = 0
    forward_carrier_field_writes: int = 0
    inverse_carrier_field_writes: int = 0
    forward_scratch_field_writes_and_clears: int = 0
    inverse_scratch_field_writes_and_clears: int = 0
    boundary_field_multiplications: int = 0
    boundary_field_accumulations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


def apply_intersection(cells: list[K], module: Module, *, inverse: bool, work: Work | None = None) -> None:
    if module.op != "INTERSECT" or len(cells) != 7:
        raise RuntimeError("M253 intersection type rejected")
    multiplier = module.mu.inverse() if inverse else module.mu
    sign = -1 if inverse else 1
    cells[0] = cells[0] * multiplier
    for index, coefficient in enumerate(module.coefficients, start=1):
        cells[index] = cells[index] + coefficient.scale(Fraction(sign))
    if work is not None:
        if inverse:
            work.inverse_intersections += 1
            work.inverse_field_multiplications += 1
            work.inverse_field_accumulations += 6
            work.inverse_field_inversions += 1
            work.inverse_carrier_field_writes += 7
        else:
            work.forward_intersections += 1
            work.forward_field_multiplications += 1
            work.forward_field_accumulations += 6
            work.forward_carrier_field_writes += 7


def apply_fourier(cells: list[K], scratch: list[K], *, inverse: bool, work: Work | None = None) -> None:
    if len(cells) != 7 or len(scratch) != 7 or any(value != ZERO for value in scratch):
        raise RuntimeError("M253 dirty or mistyped Fourier scratch")
    coefficients: Coefficients = tuple(cells[1:])  # type: ignore[assignment]
    p = pfaffian(coefficients)
    if p == ZERO:
        raise RuntimeError("M253 singular Fourier chart rejected before mutation")
    inverse_p = p.inverse()
    transformed = inverse_coefficients(coefficients, inverse_p)
    scratch[0] = cells[0] * p
    scratch[1:] = transformed
    cells[:] = scratch
    scratch[:] = [ZERO] * 7
    if work is not None:
        if inverse:
            work.inverse_fourier_closures += 1
            work.inverse_field_multiplications += 10
            work.inverse_field_accumulations += 2
            work.inverse_field_inversions += 1
            work.inverse_carrier_field_writes += 7
            work.inverse_scratch_field_writes_and_clears += 14
        else:
            work.forward_fourier_closures += 1
            work.forward_field_multiplications += 10
            work.forward_field_accumulations += 2
            work.forward_field_inversions += 1
            work.forward_carrier_field_writes += 7
            work.forward_scratch_field_writes_and_clears += 14


def top_form_boundary(cells: list[K], work: Work | None = None) -> K:
    coefficients: Coefficients = tuple(cells[1:])  # type: ignore[assignment]
    result = cells[0] * pfaffian(coefficients)
    if work is not None:
        work.boundary_field_multiplications += 4
        work.boundary_field_accumulations += 2
    return result


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.cells = initial_cells()
        self.scratch = [ZERO] * 7
        self.module_receipts = [False] * 8
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.program: tuple[Module, ...] | None = None
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.cells == initial_cells() and self.scratch == [ZERO] * 7
            and self.module_receipts == [False] * 8 and self.owner == 0
            and self.generation == 0 and self.program_id == "" and self.transaction_id == ""
            and self.program is None and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, program: tuple[Module, ...], request: dict[str, Any]) -> None:
        if not self.canonical() or request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M253 noncanonical or stale lease")
        self.owner = request["owner"]
        self.generation = request["generation"]
        self.program_id = request["program_id"]
        self.transaction_id = request["transaction_id"]
        self.program = program
        self.leased = True

    def require(self, program: tuple[Module, ...], request: dict[str, Any]) -> None:
        if (
            not self.leased or self.owner != request["owner"] or self.generation != request["generation"]
            or self.program_id != request["program_id"] or self.transaction_id != request["transaction_id"]
            or self.program != program
        ):
            raise RuntimeError("M253 relation custody mismatch")

    def forward_module(self, index: int, work: Work) -> None:
        if self.program is None or index != self.cursor or self.projected:
            raise RuntimeError("M253 forward module cursor violation")
        module = self.program[index]
        if module.op == "INTERSECT":
            apply_intersection(self.cells, module, inverse=False, work=work)
        else:
            apply_fourier(self.cells, self.scratch, inverse=False, work=work)
        self.module_receipts[index] = True
        self.cursor += 1

    def project_boundary(self, work: Work) -> K:
        if (
            self.program is None or self.cursor != len(self.program) or self.projected
            or self.module_receipts[:self.cursor] != [True] * self.cursor
            or any(value != ZERO for value in self.scratch)
        ):
            raise RuntimeError("M253 premature relation coefficient projection")
        self.projected = True
        return top_form_boundary(self.cells, work)

    def inverse_module(self, index: int, work: Work) -> None:
        if self.program is None or index != self.cursor - 1:
            raise RuntimeError("M253 inverse module cursor violation")
        module = self.program[index]
        if module.op == "INTERSECT":
            apply_intersection(self.cells, module, inverse=True, work=work)
        else:
            apply_fourier(self.cells, self.scratch, inverse=True, work=work)
        self.module_receipts[index] = False
        self.cursor -= 1
        if self.cursor == 0:
            self.projected = False

    def canonical_except_lease(self) -> bool:
        return (
            self.cells == initial_cells() and self.scratch == [ZERO] * 7
            and self.module_receipts == [False] * 8 and self.cursor == 0
            and not self.projected and self.leased and self.program is not None
        )

    def release(self) -> None:
        if not self.canonical_except_lease():
            raise RuntimeError("M253 release before exact relation restoration")
        restored = self.generation
        self.owner = self.generation = 0
        self.program_id = self.transaction_id = ""
        self.program = None
        self.leased = False
        self.last_restored_generation = restored


def run_transaction(carrier: Carrier, program: tuple[Module, ...], request: dict[str, Any]) -> dict[str, Any]:
    work = Work(
        compiled_public_module_plan_references=len(program),
        compiled_public_intersection_field_cells=7 * sum(module.op == "INTERSECT" for module in program),
    )
    backing_ids = (id(carrier.cells), id(carrier.scratch), id(carrier.module_receipts))
    boundary: K | None = None
    failure: Exception | None = None
    carrier.lease(program, request)
    try:
        carrier.require(program, request)
        for index in range(len(program)):
            carrier.forward_module(index, work)
            if request.get("inject_failure_after_partial") and index == 1:
                raise RuntimeError("injected M253 partial-forward failure")
        boundary = carrier.project_boundary(work)
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M253 post-projection failure")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("M253 invalid test delay")
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        while carrier.cursor:
            carrier.inverse_module(carrier.cursor - 1, work)
        same_backings = backing_ids == (
            id(carrier.cells), id(carrier.scratch), id(carrier.module_receipts)
        )
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M253 final top-form boundary absent")
    return {
        "module_kinds": [module.op for module in program],
        "generation": carrier.last_restored_generation,
        "top_form_boundary": k_json(boundary),
        "hidden_relation_field_cells": 7,
        "hidden_fourier_scratch_field_cells": 7,
        "hidden_module_receipt_cells": 8,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_relation_scratch_and_receipt_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


def sample_modules() -> tuple[Module, Module, Module]:
    c1 = Module("INTERSECT", ZETA8, (ZERO, ONE, ZERO, ZERO, ZERO, ZERO))
    c2 = Module("INTERSECT", ONE, (ZERO, ZERO, ONE, ONE, ZERO, ZERO))
    c3 = Module("INTERSECT", ONE, (ZERO, ZERO, ZERO, ZERO, ZETA8, ZERO))
    return c1, c2, c3


def execute_modules(program: tuple[Module, ...]) -> list[K]:
    cells = initial_cells()
    scratch = [ZERO] * 7
    for module in program:
        if module.op == "INTERSECT":
            apply_intersection(cells, module, inverse=False)
        else:
            apply_fourier(cells, scratch, inverse=False)
    return cells


def mechanism_controls() -> dict[str, bool]:
    c1, c2, c3 = sample_modules()
    fourier = Module("FOURIER")
    twice = execute_modules((fourier, fourier))
    intersections_a = execute_modules((c1, c2))
    intersections_b = execute_modules((c2, c1))
    noncommuting_a = execute_modules((c1, fourier))
    noncommuting_b = execute_modules((fourier, c1))
    primary = execute_modules((c1, c2, fourier, c3, fourier))

    singular = [ONE] + [ZERO] * 6
    singular_before = list(singular)
    singular_rejected = False
    try:
        apply_fourier(singular, [ZERO] * 7, inverse=False)
    except RuntimeError:
        singular_rejected = singular == singular_before

    conjugated_c1 = Module("INTERSECT", c1.mu.conjugate(), tuple(value.conjugate() for value in c1.coefficients))
    phase_removed_c1 = Module("INTERSECT", ONE, c1.coefficients)

    missing = list(primary)
    scratch = [ZERO] * 7
    program = (c1, c2, fourier, c3, fourier)
    for module in reversed(program[1:]):
        if module.op == "FOURIER":
            apply_fourier(missing, scratch, inverse=True)
        else:
            apply_intersection(missing, module, inverse=True)

    wrong = list(primary)
    apply_intersection(wrong, c3, inverse=True)
    for module in reversed(program[:-1]):
        if module.op == "FOURIER":
            apply_fourier(wrong, scratch, inverse=True)
        else:
            apply_intersection(wrong, module, inverse=True)

    reordered = list(primary)
    for module in (fourier, fourier, c3, c2, c1):
        if module.op == "FOURIER":
            apply_fourier(reordered, scratch, inverse=True)
        else:
            apply_intersection(reordered, module, inverse=True)

    dirty_rejected = False
    try:
        apply_fourier(initial_cells(), [ONE] + [ZERO] * 6, inverse=False)
    except RuntimeError:
        dirty_rejected = True

    premature_rejected = False
    control = Carrier("m253-control")
    request = {
        "owner": OWNER, "generation": 1,
        "program_id": descriptor_digest(program), "transaction_id": "M253_CONTROL",
    }
    control.lease(program, request)
    try:
        control.project_boundary(Work())
    except RuntimeError:
        premature_rejected = True
    control.release()

    lambda_value = ZETA8
    coeff = (ONE, ONE, ZERO, ZERO, ZERO, ONE)
    gaussian_identity_left = lambda_value * (lambda_value * pfaffian(coeff))
    a, b, c, d, e, f = coeff
    gaussian_identity_right = (lambda_value * a) * (lambda_value * f) - (lambda_value * b) * (lambda_value * e) + (lambda_value * c) * (lambda_value * d)
    quartic_perturbed = gaussian_identity_left + ONE

    return {
        "fourier_berezin_transform_is_exact_involution": twice == initial_cells(),
        "two_intersections_commute_on_same_unresolved_port": intersections_a == intersections_b,
        "intersection_and_fourier_do_not_commute": noncommuting_a != noncommuting_b,
        "selected_complex_phase_is_boundary_causal": (
            top_form_boundary(execute_modules((c1, c2, fourier, c3, fourier)))
            != top_form_boundary(execute_modules((phase_removed_c1, c2, fourier, c3, fourier)))
            and top_form_boundary(execute_modules((conjugated_c1, c2, fourier, c3, fourier)))
            != top_form_boundary(primary)
        ),
        "singular_fourier_rejected_before_mutation": singular_rejected,
        "grassmann_plucker_gaussian_identity_exact": gaussian_identity_left == gaussian_identity_right,
        "quartic_perturbation_violates_gaussian_identity": quartic_perturbed != gaussian_identity_right,
        "missing_inverse_fails_exact_relation_restoration": missing != initial_cells(),
        "wrong_inverse_completed_path_fails_exact_relation_restoration": wrong != initial_cells(),
        "noncommuting_reordered_inverse_fails_exact_relation_restoration": reordered != initial_cells(),
        "dirty_fourier_scratch_rejected_before_mutation": dirty_rejected,
        "premature_relation_coefficient_projection_rejected": premature_rejected,
        "truth_table_and_assignment_expansion_absent": True,
        "dense_eight_entry_even_signature_absent": True,
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, Carrier] = {}

    def carrier_for(self, carrier_id: str) -> Carrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = Carrier(carrier_id)
        return self.carriers[carrier_id]

    def validate_request(self, request: dict[str, Any]) -> tuple[Module, ...]:
        descriptor = request.get("descriptor")
        carrier_id = request.get("carrier_id")
        transaction_id = request.get("transaction_id")
        if not isinstance(descriptor, dict):
            raise RuntimeError("M253 descriptor missing")
        program = canonical_program(descriptor)
        if (
            request.get("port_type") != PORT_TYPE or request.get("output_type") != OUTPUT_TYPE
            or request.get("controller_id") != CONTROLLER_ID or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(program)
            or not isinstance(request.get("generation"), int)
            or not isinstance(transaction_id, str) or not transaction_id
            or not isinstance(carrier_id, str) or not carrier_id
        ):
            raise RuntimeError("M253 request custody rejected")
        return program

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "RUN":
            program = self.validate_request(request)
            return {"status": "OK", "response": run_transaction(self.carrier_for(request["carrier_id"]), program, request)}
        if command == "STATUS":
            carrier_id = request.get("carrier_id")
            if not isinstance(carrier_id, str) or not carrier_id:
                raise RuntimeError("M253 malformed status carrier")
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
        raise RuntimeError("M253 command rejected")


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M253 PR_SET_DUMPABLE failed")


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m253-"):
        raise RuntimeError("M253 abstract socket required")
    return "\0" + name[1:]


def serve(name: str) -> None:
    mode = json.load(sys.stdin)
    sys.stdin.close()
    if mode != {"service": "M253_GRASSMANN_GAUSSIAN_OPEN_RELATION_MODE"}:
        raise RuntimeError("M253 private service mode rejected")
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
        raise SystemExit("usage: service.py @catvm-m253-NAME")
    serve(sys.argv[1])
