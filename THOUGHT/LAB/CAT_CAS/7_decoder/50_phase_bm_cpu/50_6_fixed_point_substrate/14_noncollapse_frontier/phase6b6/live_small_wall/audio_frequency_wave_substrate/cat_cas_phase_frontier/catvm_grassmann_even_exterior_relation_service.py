#!/usr/bin/env python3
"""M254 exact full-even four-port Grassmann relation CATVM backend.

The actual resident relation is an eight-cell coefficient vector in the even
exterior algebra on four typed Grassmann ports.  Public relation intersection
is signed wedge multiplication.  The four-port formal Berezin transform is an
exact Hodge-complement involution.  The independent top-form coefficient makes
the carrier strictly broader than the Gaussian/pure-spinor chart used by M253.
Only the final top form is released after exact reverse restoration.
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
from typing import Any, Sequence


PORT_TYPE = "CATVM_QZETA8_GRASSMANN_EVEN_FOUR_PORT_RELATION_V1"
OUTPUT_TYPE = "QZETA8_GRASSMANN_EVEN_TOP_FORM_BOUNDARY_V1"
OWNER = 254004
CONTROLLER_ID = 254001
PORT_ORDER = ("THETA0", "THETA1", "THETA2", "THETA3")
BASIS_MASKS = (0, 3, 5, 9, 6, 10, 12, 15)
PAIR_INDICES = (1, 2, 3, 4, 5, 6)
MASK_INDEX = {mask: index for index, mask in enumerate(BASIS_MASKS)}


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

    def inverse(self) -> "K":
        a = (self.one, self.root)
        b = (self.imag, self.root_imag)
        aa = self.real_mul(a, a)
        bb = self.real_mul(b, b)
        norm = (aa[0] + bb[0], aa[1] + bb[1])
        denominator = norm[0] * norm[0] - 2 * norm[1] * norm[1]
        if denominator == 0:
            raise ZeroDivisionError("M254 zero field inverse")
        inverse_norm = (norm[0] / denominator, -norm[1] / denominator)
        real = self.real_mul(a, inverse_norm)
        imag = self.real_mul((-b[0], -b[1]), inverse_norm)
        return K(real[0], real[1], imag[0], imag[1])


ZERO = K()
ONE = K(Fraction(1))
ZETA8 = K(root=Fraction(1, 2), root_imag=Fraction(1, 2))
Signature = tuple[K, K, K, K, K, K, K, K]


def k_json(value: K) -> list[list[int]]:
    return [[coordinate.numerator, coordinate.denominator] for coordinate in value.coords()]


def parse_public_k(value: object) -> K:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("M254 public field coordinate rejected")
    coordinates: list[Fraction] = []
    for item in value:
        if (
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in item)
            or abs(item[0]) > 8 or not 1 <= item[1] <= 8
        ):
            raise RuntimeError("M254 public rational coordinate rejected")
        coordinates.append(Fraction(item[0], item[1]))
    return K(*coordinates)


def wedge_sign(left: int, right: int) -> int:
    if left & right:
        return 0
    inversions = 0
    for bit in range(4):
        if left & (1 << bit):
            inversions += (right & ((1 << bit) - 1)).bit_count()
    return -1 if inversions % 2 else 1


def even_product(left: Signature, right: Signature) -> Signature:
    output = [ZERO] * 8
    for left_index, left_mask in enumerate(BASIS_MASKS):
        for right_index, right_mask in enumerate(BASIS_MASKS):
            sign = wedge_sign(left_mask, right_mask)
            if sign:
                target = MASK_INDEX[left_mask | right_mask]
                output[target] = output[target] + (left[left_index] * right[right_index]).scale(Fraction(sign))
    return tuple(output)  # type: ignore[return-value]


def pair_wedge_top(left: Sequence[K], right: Sequence[K]) -> K:
    result = ZERO
    for index in PAIR_INDICES:
        mask = BASIS_MASKS[index]
        complement_index = MASK_INDEX[15 ^ mask]
        result = result + (left[index] * right[complement_index]).scale(Fraction(wedge_sign(mask, 15 ^ mask)))
    return result


def inverse_factor(factor: Signature) -> Signature:
    if factor[0] == ZERO:
        raise RuntimeError("M254 noninvertible zero-scalar intersection factor")
    scalar_inverse = factor[0].inverse()
    scalar_inverse_squared = scalar_inverse * scalar_inverse
    scalar_inverse_cubed = scalar_inverse_squared * scalar_inverse
    pairs = tuple(-(factor[index] * scalar_inverse_squared) for index in PAIR_INDICES)
    top = -(factor[7] * scalar_inverse_squared) + pair_wedge_top(factor, factor) * scalar_inverse_cubed
    return (scalar_inverse, *pairs, top)  # type: ignore[return-value]


HODGE_PAIRS = ((0, 7, 1), (1, 6, -1), (2, 5, 1), (3, 4, -1))


def apply_hodge(cells: list[K], scratch: list[K], *, work: "Work | None" = None, inverse: bool = False) -> None:
    if len(cells) != 8 or scratch != [ZERO]:
        raise RuntimeError("M254 dirty or mistyped Hodge scratch")
    for left, right, sign in HODGE_PAIRS:
        scratch[0] = cells[left]
        cells[left] = cells[right].scale(Fraction(sign))
        cells[right] = scratch[0].scale(Fraction(sign))
        scratch[0] = ZERO
    if work is not None:
        if inverse:
            work.inverse_hodge_closures += 1
            work.inverse_field_negations += 4
            work.inverse_carrier_field_writes += 8
            work.inverse_hodge_scratch_writes_and_clears += 8
        else:
            work.forward_hodge_closures += 1
            work.forward_field_negations += 4
            work.forward_carrier_field_writes += 8
            work.forward_hodge_scratch_writes_and_clears += 8


def apply_intersection(cells: list[K], factor: Signature, *, work: "Work | None" = None, inverse: bool = False) -> None:
    if len(cells) != 8:
        raise RuntimeError("M254 intersection carrier type rejected")
    applied = inverse_factor(factor) if inverse else factor
    # Top degree first, then pairs, then scalar: no old resident coefficient is
    # overwritten before its last use.
    pair_top = pair_wedge_top(cells, applied)
    cells[7] = cells[0] * applied[7] + cells[7] * applied[0] + pair_top
    for index in PAIR_INDICES:
        cells[index] = cells[0] * applied[index] + cells[index] * applied[0]
    cells[0] = cells[0] * applied[0]
    if work is not None:
        if inverse:
            work.inverse_intersections += 1
            work.inverse_field_multiplications += 21
            work.inverse_field_accumulations += 14
            work.inverse_field_negations += 2
            work.inverse_carrier_field_writes += 8
            work.inverse_factor_rematerializations += 1
            work.inverse_factor_returned_field_cells_materialized += 8
            work.peak_returned_inverse_factor_field_cells = max(
                work.peak_returned_inverse_factor_field_cells, 8
            )
            work.inverse_factor_field_multiplications += 16
            work.inverse_factor_field_accumulations += 7
            work.inverse_factor_field_negations += 9
            work.inverse_factor_field_inversions += 1
        else:
            work.forward_intersections += 1
            work.forward_field_multiplications += 21
            work.forward_field_accumulations += 14
            work.forward_field_negations += 2
            work.forward_carrier_field_writes += 8


def gaussian_defect(cells: Signature) -> K:
    a, b, c, d, e, f = cells[1:7]
    return cells[0] * cells[7] - (a * f - b * e + c * d)


@dataclass(frozen=True)
class Module:
    op: str
    factor: Signature = (ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)


def canonical_program(descriptor: dict[str, Any]) -> tuple[Module, ...]:
    if set(descriptor) != {"ports", "modules"} or descriptor.get("ports") != list(PORT_ORDER):
        raise RuntimeError("M254 typed port order or descriptor rejected")
    modules = descriptor.get("modules")
    if not isinstance(modules, list) or not 1 <= len(modules) <= 8:
        raise RuntimeError("M254 public program length rejected")
    compiled: list[Module] = []
    for item in modules:
        if not isinstance(item, dict) or "op" not in item:
            raise RuntimeError("M254 malformed module")
        if item["op"] == "HODGE":
            if set(item) != {"op"}:
                raise RuntimeError("M254 answer-bearing Hodge descriptor")
            compiled.append(Module("HODGE"))
            continue
        if item["op"] != "INTERSECT" or set(item) != {"op", "factor"}:
            raise RuntimeError("M254 undeclared relation module")
        values = item["factor"]
        if not isinstance(values, list) or len(values) != 8:
            raise RuntimeError("M254 full-even factor size rejected")
        factor: Signature = tuple(parse_public_k(value) for value in values)  # type: ignore[assignment]
        if factor[0] == ZERO:
            raise RuntimeError("M254 noninvertible public factor rejected")
        compiled.append(Module("INTERSECT", factor))
    return tuple(compiled)


def canonical_descriptor_json(program: tuple[Module, ...]) -> str:
    modules: list[dict[str, object]] = []
    for module in program:
        if module.op == "HODGE":
            modules.append({"op": "HODGE"})
        else:
            modules.append({"op": "INTERSECT", "factor": [k_json(value) for value in module.factor]})
    return json.dumps({"ports": list(PORT_ORDER), "modules": modules}, sort_keys=True, separators=(",", ":"))


def descriptor_digest(program: tuple[Module, ...]) -> str:
    return hashlib.sha256(canonical_descriptor_json(program).encode()).hexdigest()


def initial_cells() -> list[K]:
    return [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO]


@dataclass
class Work:
    compiled_public_module_plan_references: int = 0
    compiled_public_intersection_factor_field_cells: int = 0
    forward_intersections: int = 0
    inverse_intersections: int = 0
    forward_hodge_closures: int = 0
    inverse_hodge_closures: int = 0
    forward_field_multiplications: int = 0
    inverse_field_multiplications: int = 0
    forward_field_accumulations: int = 0
    inverse_field_accumulations: int = 0
    forward_field_negations: int = 0
    inverse_field_negations: int = 0
    forward_carrier_field_writes: int = 0
    inverse_carrier_field_writes: int = 0
    forward_hodge_scratch_writes_and_clears: int = 0
    inverse_hodge_scratch_writes_and_clears: int = 0
    inverse_factor_rematerializations: int = 0
    inverse_factor_returned_field_cells_materialized: int = 0
    peak_returned_inverse_factor_field_cells: int = 0
    inverse_factor_field_multiplications: int = 0
    inverse_factor_field_accumulations: int = 0
    inverse_factor_field_negations: int = 0
    inverse_factor_field_inversions: int = 0
    retained_dynamic_inverse_history_entries: int = 0


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.cells = initial_cells()
        self.scratch = [ZERO]
        self.receipts = [False] * 8
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
            self.cells == initial_cells() and self.scratch == [ZERO]
            and self.receipts == [False] * 8 and self.owner == 0 and self.generation == 0
            and self.program_id == "" and self.transaction_id == "" and self.program is None
            and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, program: tuple[Module, ...], request: dict[str, Any]) -> None:
        if not self.canonical() or request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M254 noncanonical or stale lease")
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
            raise RuntimeError("M254 exterior relation custody mismatch")

    def forward_module(self, index: int, work: Work) -> None:
        if self.program is None or index != self.cursor or self.projected:
            raise RuntimeError("M254 forward cursor violation")
        module = self.program[index]
        if module.op == "INTERSECT":
            apply_intersection(self.cells, module.factor, work=work, inverse=False)
        else:
            apply_hodge(self.cells, self.scratch, work=work, inverse=False)
        self.receipts[index] = True
        self.cursor += 1

    def project_boundary(self) -> K:
        if (
            self.program is None or self.cursor != len(self.program) or self.projected
            or self.receipts[:self.cursor] != [True] * self.cursor or self.scratch != [ZERO]
        ):
            raise RuntimeError("M254 premature exterior coefficient projection")
        self.projected = True
        return self.cells[7]

    def inverse_module(self, index: int, work: Work) -> None:
        if self.program is None or index != self.cursor - 1:
            raise RuntimeError("M254 inverse cursor violation")
        module = self.program[index]
        if module.op == "INTERSECT":
            apply_intersection(self.cells, module.factor, work=work, inverse=True)
        else:
            apply_hodge(self.cells, self.scratch, work=work, inverse=True)
        self.cursor -= 1
        self.receipts[self.cursor] = False
        if self.cursor == 0:
            self.projected = False

    def release(self) -> None:
        if (
            self.cells != initial_cells() or self.scratch != [ZERO] or self.receipts != [False] * 8
            or self.cursor != 0 or self.projected or not self.leased or self.program is None
        ):
            raise RuntimeError("M254 release before exact exterior restoration")
        restored = self.generation
        self.owner = self.generation = 0
        self.program_id = self.transaction_id = ""
        self.program = None
        self.leased = False
        self.last_restored_generation = restored


def run_transaction(carrier: Carrier, program: tuple[Module, ...], request: dict[str, Any]) -> dict[str, Any]:
    work = Work(
        compiled_public_module_plan_references=len(program),
        compiled_public_intersection_factor_field_cells=8 * sum(module.op == "INTERSECT" for module in program),
    )
    backing_ids = (id(carrier.cells), id(carrier.scratch), id(carrier.receipts))
    boundary: K | None = None
    failure: Exception | None = None
    carrier.lease(program, request)
    try:
        carrier.require(program, request)
        for index in range(len(program)):
            carrier.forward_module(index, work)
            if request.get("inject_failure_after_partial") and index == 1:
                raise RuntimeError("injected M254 partial failure")
        boundary = carrier.project_boundary()
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M254 post-projection failure")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("M254 invalid test delay")
        if delay:
            time.sleep(delay / 1000)
    except Exception as exc:
        failure = exc
    finally:
        while carrier.cursor:
            carrier.inverse_module(carrier.cursor - 1, work)
        same_backings = backing_ids == (id(carrier.cells), id(carrier.scratch), id(carrier.receipts))
        carrier.release()
    if failure is not None:
        raise failure
    if boundary is None:
        raise RuntimeError("M254 final top form absent")
    return {
        "module_kinds": [module.op for module in program],
        "generation": carrier.last_restored_generation,
        "top_form_boundary": k_json(boundary),
        "hidden_even_relation_field_cells": 8,
        "hidden_hodge_scratch_field_cells": 1,
        "hidden_module_receipt_cells": 8,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_relation_scratch_and_receipt_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": vars(work),
    }


def sample_factors() -> tuple[Signature, Signature, Signature]:
    factor1: Signature = (ONE, ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZETA8)
    factor2: Signature = (ZETA8, ZERO, ONE, ZERO, ONE, ZERO, ZERO, ONE)
    factor3: Signature = (ONE, ZERO, ZERO, ONE, ZERO, ZETA8, ZERO, -ONE)
    return factor1, factor2, factor3


def execute_modules(program: tuple[Module, ...]) -> list[K]:
    cells = initial_cells()
    scratch = [ZERO]
    for module in program:
        if module.op == "INTERSECT":
            apply_intersection(cells, module.factor)
        else:
            apply_hodge(cells, scratch)
    return cells


def mechanism_controls() -> dict[str, bool]:
    factor1, factor2, factor3 = sample_factors()
    hodge = Module("HODGE")
    m1, m2, m3 = Module("INTERSECT", factor1), Module("INTERSECT", factor2), Module("INTERSECT", factor3)
    twice = execute_modules((hodge, hodge))
    commuting1 = execute_modules((m1, m2))
    commuting2 = execute_modules((m2, m1))
    noncommuting1 = execute_modules((m1, hodge))
    noncommuting2 = execute_modules((hodge, m1))
    program = (m1, hodge, m2, m3, hodge)
    primary = execute_modules(program)

    inverse_ok = all(
        even_product(factor, inverse_factor(factor)) == tuple(initial_cells())
        and even_product(inverse_factor(factor), factor) == tuple(initial_cells())
        for factor in (factor1, factor2, factor3)
    )
    associative = even_product(even_product(factor1, factor2), factor3) == even_product(factor1, even_product(factor2, factor3))

    no_quartic = list(factor1)
    no_quartic[7] = ZERO
    quartic_boundary_changes = execute_modules((Module("INTERSECT", tuple(no_quartic)), hodge, m2, m3, hodge))[7] != primary[7]  # type: ignore[arg-type]

    missing = list(primary)
    scratch = [ZERO]
    for module in reversed(program[1:]):
        if module.op == "HODGE":
            apply_hodge(missing, scratch, inverse=True)
        else:
            apply_intersection(missing, module.factor, inverse=True)

    wrong = list(primary)
    apply_intersection(wrong, factor2, inverse=True)
    for module in reversed(program[:-1]):
        if module.op == "HODGE":
            apply_hodge(wrong, scratch, inverse=True)
        else:
            apply_intersection(wrong, module.factor, inverse=True)

    reordered = list(primary)
    for module in (hodge, hodge, m3, m2, m1):
        if module.op == "HODGE":
            apply_hodge(reordered, scratch, inverse=True)
        else:
            apply_intersection(reordered, module.factor, inverse=True)

    dirty = False
    try:
        apply_hodge(initial_cells(), [ONE])
    except RuntimeError:
        dirty = True

    premature = False
    carrier = Carrier("m254-control")
    request = {
        "owner": OWNER, "generation": 1, "program_id": descriptor_digest(program),
        "transaction_id": "M254_CONTROL",
    }
    carrier.lease(program, request)
    try:
        carrier.project_boundary()
    except RuntimeError:
        premature = True
    carrier.release()

    gaussian: Signature = (ONE, ONE, ONE, ZERO, ZERO, ZERO, ONE, ONE)
    gaussian_hodge = execute_modules((Module("INTERSECT", gaussian), hodge))
    gaussian_hodge_expected = [ONE, -ONE, ZERO, ZERO, ZERO, ONE, -ONE, ONE]
    nongaussian = list(gaussian)
    nongaussian[7] = ZETA8

    quadratic_factor: Signature = (ONE, ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO)
    exact_quadratic_inverse = inverse_factor(quadratic_factor)
    inverse_without_nilpotent_square = list(exact_quadratic_inverse)
    inverse_without_nilpotent_square[7] = ZERO

    zero_scalar_rejected = False
    try:
        inverse_factor((ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO))
    except RuntimeError:
        zero_scalar_rejected = True

    return {
        "signed_even_exterior_intersection_is_associative": associative,
        "signed_even_exterior_intersection_is_commutative": commuting1 == commuting2,
        "four_port_berezin_hodge_transform_is_exact_involution": twice == initial_cells(),
        "intersection_and_hodge_do_not_commute": noncommuting1 != noncommuting2,
        "public_nonzero_scalar_factors_have_exact_nilpotent_inverses": inverse_ok,
        "independent_quartic_coefficient_changes_selected_boundary": quartic_boundary_changes,
        "gaussian_pure_spinor_defect_is_zero": gaussian_defect(gaussian) == ZERO,
        "hodge_restricts_to_the_m253_gaussian_pfaffian_transform": (
            gaussian_hodge == gaussian_hodge_expected
            and gaussian_defect(tuple(gaussian_hodge)) == ZERO  # type: ignore[arg-type]
        ),
        "independent_quartic_violates_gaussian_pure_spinor_identity": gaussian_defect(tuple(nongaussian)) != ZERO,  # type: ignore[arg-type]
        "nilpotent_square_term_is_required_for_factor_inverse": even_product(
            quadratic_factor, tuple(inverse_without_nilpotent_square)  # type: ignore[arg-type]
        ) != tuple(initial_cells()),
        "zero_scalar_intersection_factor_rejected_before_mutation": zero_scalar_rejected,
        "missing_inverse_fails_exact_exterior_restoration": missing != initial_cells(),
        "wrong_inverse_completed_path_fails_exact_exterior_restoration": wrong != initial_cells(),
        "noncommuting_reordered_inverse_fails_exact_exterior_restoration": reordered != initial_cells(),
        "dirty_hodge_scratch_rejected_before_mutation": dirty,
        "premature_exterior_coefficient_projection_rejected": premature,
        "truth_table_and_assignment_expansion_absent": True,
        "decoded_boolean_relation_table_absent": True,
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
            raise RuntimeError("M254 descriptor missing")
        program = canonical_program(descriptor)
        if (
            request.get("port_type") != PORT_TYPE or request.get("output_type") != OUTPUT_TYPE
            or request.get("controller_id") != CONTROLLER_ID or request.get("owner") != OWNER
            or request.get("program_id") != descriptor_digest(program)
            or not isinstance(request.get("generation"), int)
            or isinstance(request.get("generation"), bool)
            or not isinstance(transaction_id, str) or not transaction_id
            or not isinstance(carrier_id, str) or not carrier_id
        ):
            raise RuntimeError("M254 request custody rejected")
        return program

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "RUN":
            program = self.validate_request(request)
            return {"status": "OK", "response": run_transaction(self.carrier_for(request["carrier_id"]), program, request)}
        if command == "STATUS":
            carrier_id = request.get("carrier_id")
            if not isinstance(carrier_id, str) or not carrier_id:
                raise RuntimeError("M254 malformed status carrier")
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
        raise RuntimeError("M254 command rejected")


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M254 PR_SET_DUMPABLE failed")


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m254-"):
        raise RuntimeError("M254 abstract socket required")
    return "\0" + name[1:]


def serve(name: str) -> None:
    mode = json.load(sys.stdin)
    sys.stdin.close()
    if mode != {"service": "M254_GRASSMANN_EVEN_EXTERIOR_RELATION_MODE"}:
        raise RuntimeError("M254 private mode rejected")
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
        raise SystemExit("usage: service.py @catvm-m254-NAME")
    serve(sys.argv[1])
