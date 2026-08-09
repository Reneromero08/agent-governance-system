#!/usr/bin/env python3
"""M256 CATVM backend for an exact formal Schur all-pass waveform phase.

The hidden carrier is the actual rational function f(z)=N(z)/D(z), with N
and D retained as fixed Q(zeta8) coefficient backings.  A public Schur section
acts by f -> (a+z f)/(1+a z f).  Only winding and one exact final evaluation
leave the service, after exact inverse restoration and release.
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


MILESTONE = 256
PORT_TYPE = "CATVM_QZETA8_FORMAL_SCHUR_ALLPASS_WAVEFORM_PHASE_V1"
OUTPUT_TYPE = "QZETA8_ALLPASS_WINDING_AND_POINT_EVALUATION_V1"
OWNER = 256004
CONTROLLER = 256001
BOUNDARY_CONSUMER = 256002
CAPACITY = 4


@dataclass(frozen=True)
class K:
    one: Fraction = Fraction(0)
    root: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)
    root_imag: Fraction = Fraction(0)

    def coords(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return self.one, self.root, self.imag, self.root_imag

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
        ac = self.real_mul((self.one, self.root), (other.one, other.root))
        bd = self.real_mul((self.imag, self.root_imag), (other.imag, other.root_imag))
        ad = self.real_mul((self.one, self.root), (other.imag, other.root_imag))
        bc = self.real_mul((self.imag, self.root_imag), (other.one, other.root))
        return K(ac[0] - bd[0], ac[1] - bd[1], ad[0] + bc[0], ad[1] + bc[1])

    def scale(self, scalar: Fraction) -> "K":
        return K(*(value * scalar for value in self.coords()))

    def inverse(self) -> "K":
        aa = self.real_mul((self.one, self.root), (self.one, self.root))
        bb = self.real_mul((self.imag, self.root_imag), (self.imag, self.root_imag))
        norm = (aa[0] + bb[0], aa[1] + bb[1])
        denominator = norm[0] * norm[0] - 2 * norm[1] * norm[1]
        if denominator == 0:
            raise RuntimeError("M256 zero Q(zeta8) inverse")
        inverse_norm = (norm[0] / denominator, -norm[1] / denominator)
        real = self.real_mul((self.one, self.root), inverse_norm)
        imag = self.real_mul((-self.imag, -self.root_imag), inverse_norm)
        return K(real[0], real[1], imag[0], imag[1])


ZERO = K()
ONE = K(one=Fraction(1))
ZETA8 = K(root=Fraction(1, 2), root_imag=Fraction(1, 2))


def k_json(value: K) -> list[list[int]]:
    return [[coordinate.numerator, coordinate.denominator] for coordinate in value.coords()]


def canonical_payload(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class Program:
    sections: tuple[Fraction, Fraction, Fraction]
    evaluation: str
    output_type: str
    program_id: str

    def custody_descriptor(self) -> tuple[tuple[Fraction, Fraction, Fraction], str, str, str]:
        return self.sections, self.evaluation, self.output_type, self.program_id


def canonical_program(descriptor: dict[str, Any]) -> Program:
    if set(descriptor) != {"sections", "evaluation", "output_type"}:
        raise RuntimeError("M256 malformed public descriptor")
    if descriptor["evaluation"] != "ZETA8" or descriptor["output_type"] != OUTPUT_TYPE:
        raise RuntimeError("M256 evaluation or output type rejected")
    raw_sections = descriptor["sections"]
    if not isinstance(raw_sections, list) or len(raw_sections) != 3:
        raise RuntimeError("M256 exactly three Schur sections required")
    sections: list[Fraction] = []
    for item in raw_sections:
        if (
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in item)
        ):
            raise RuntimeError("M256 Schur coefficient rejected")
        numerator, denominator = item
        if not 1 <= denominator <= 16 or abs(numerator) >= denominator:
            raise RuntimeError("M256 non-lossless or singular Schur coefficient")
        sections.append(Fraction(numerator, denominator))
    program_id = hashlib.sha256(canonical_payload(descriptor)).hexdigest()
    return Program((sections[0], sections[1], sections[2]), "ZETA8", OUTPUT_TYPE, program_id)


@dataclass
class Work:
    forward_sections: int = 0
    inverse_sections: int = 0
    forward_field_scalar_multiplications: int = 0
    forward_field_additions: int = 0
    inverse_field_scalar_multiplications: int = 0
    inverse_field_subtractions: int = 0
    inverse_public_rational_multiplications: int = 0
    inverse_public_rational_subtractions: int = 0
    inverse_public_rational_divisions: int = 0
    forward_scratch_writes: int = 0
    inverse_scratch_writes: int = 0
    carrier_coefficient_writes: int = 0
    scratch_clears: int = 0
    inverse_divisibility_checks: int = 0
    evaluation_field_multiplications: int = 0
    evaluation_field_additions: int = 0
    evaluation_field_inversions: int = 0


class Carrier:
    def __init__(self, carrier_id: str) -> None:
        self.carrier_id = carrier_id
        self.numerator = [ZETA8, ZERO, ZERO, ZERO]
        self.denominator = [ONE, ZERO, ZERO, ZERO]
        self.numerator_scratch = [ZERO] * CAPACITY
        self.denominator_scratch = [ZERO] * CAPACITY
        self.receipts = [Fraction(0)] * 3
        self.cursor = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.owner = 0
        self.controller = 0
        self.program_id = ""
        self.output_type = ""
        self.transaction_id = ""
        self.consumer = 0
        self.leased_descriptor: tuple[
            tuple[Fraction, Fraction, Fraction], str, str, str
        ] | None = None
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.numerator == [ZETA8, ZERO, ZERO, ZERO]
            and self.denominator == [ONE, ZERO, ZERO, ZERO]
            and self.numerator_scratch == [ZERO] * CAPACITY
            and self.denominator_scratch == [ZERO] * CAPACITY
            and self.receipts == [Fraction(0)] * 3
            and self.cursor == 0
            and self.generation == 0
            and self.owner == 0
            and self.controller == 0
            and self.program_id == ""
            and self.output_type == ""
            and self.transaction_id == ""
            and self.consumer == 0
            and self.leased_descriptor is None
            and not self.leased
        )

    def lease(self, program: Program, expected_generation: int, transaction_id: str) -> int:
        if not self.canonical() or expected_generation != self.last_restored_generation + 1:
            raise RuntimeError("M256 dirty carrier or generation mismatch")
        self.generation = expected_generation
        self.owner = OWNER
        self.controller = CONTROLLER
        self.program_id = program.program_id
        self.output_type = program.output_type
        self.transaction_id = transaction_id
        self.consumer = BOUNDARY_CONSUMER
        self.leased_descriptor = program.custody_descriptor()
        self.leased = True
        return expected_generation

    def require(self, program: Program, generation: int, transaction_id: str) -> None:
        if (
            not self.leased or self.owner != OWNER or self.controller != CONTROLLER
            or self.program_id != program.program_id or self.output_type != program.output_type
            or self.generation != generation
            or self.transaction_id != transaction_id or self.consumer != BOUNDARY_CONSUMER
            or self.leased_descriptor != program.custody_descriptor()
        ):
            raise RuntimeError("M256 typed custody mismatch")

    def forward(self, program: Program, generation: int, transaction_id: str, work: Work) -> None:
        self.require(program, generation, transaction_id)
        if (
            self.cursor >= len(program.sections)
            or any(value != ZERO for value in self.numerator_scratch)
            or any(value != ZERO for value in self.denominator_scratch)
        ):
            raise RuntimeError("M256 forward cursor or scratch rejected")
        coefficient = program.sections[self.cursor]
        for index in range(CAPACITY):
            shifted_numerator = self.numerator[index - 1] if index else ZERO
            self.numerator_scratch[index] = self.denominator[index].scale(coefficient) + shifted_numerator
            self.denominator_scratch[index] = self.denominator[index] + shifted_numerator.scale(coefficient)
            work.forward_field_scalar_multiplications += 2
            work.forward_field_additions += 2
            work.forward_scratch_writes += 2
        self.numerator[:] = self.numerator_scratch
        self.denominator[:] = self.denominator_scratch
        work.carrier_coefficient_writes += 2 * CAPACITY
        self.numerator_scratch[:] = [ZERO] * CAPACITY
        self.denominator_scratch[:] = [ZERO] * CAPACITY
        work.scratch_clears += 2 * CAPACITY
        self.receipts[self.cursor] = coefficient
        self.cursor += 1
        work.forward_sections += 1

    def inverse(
        self, program: Program, generation: int, transaction_id: str, work: Work,
        *, override: Fraction | None = None,
    ) -> None:
        self.require(program, generation, transaction_id)
        if (
            self.cursor <= 0
            or any(value != ZERO for value in self.numerator_scratch)
            or any(value != ZERO for value in self.denominator_scratch)
        ):
            raise RuntimeError("M256 inverse cursor or scratch rejected")
        index = self.cursor - 1
        coefficient = program.sections[index] if override is None else override
        if self.receipts[index] != coefficient:
            raise RuntimeError("M256 wrong inverse section")
        divisor = Fraction(1) - coefficient * coefficient
        work.inverse_public_rational_multiplications += 1
        work.inverse_public_rational_subtractions += 1
        if divisor == 0:
            raise RuntimeError("M256 singular inverse")
        inverse_divisor = Fraction(1) / divisor
        work.inverse_public_rational_divisions += 1
        work.inverse_divisibility_checks += 1
        if self.numerator[0] - self.denominator[0].scale(coefficient) != ZERO:
            raise RuntimeError("M256 inverse numerator is not divisible by z")
        for target in range(CAPACITY):
            source = target + 1
            self.numerator_scratch[target] = (
                (self.numerator[source] - self.denominator[source].scale(coefficient)).scale(inverse_divisor)
                if source < CAPACITY else ZERO
            )
            self.denominator_scratch[target] = (
                self.denominator[target] - self.numerator[target].scale(coefficient)
            ).scale(inverse_divisor)
            work.inverse_field_scalar_multiplications += 2 + (2 if source < CAPACITY else 0)
            work.inverse_field_subtractions += 1 + (1 if source < CAPACITY else 0)
            work.inverse_scratch_writes += 2
        self.numerator[:] = self.numerator_scratch
        self.denominator[:] = self.denominator_scratch
        work.carrier_coefficient_writes += 2 * CAPACITY
        self.numerator_scratch[:] = [ZERO] * CAPACITY
        self.denominator_scratch[:] = [ZERO] * CAPACITY
        work.scratch_clears += 2 * CAPACITY
        self.receipts[index] = Fraction(0)
        self.cursor -= 1
        work.inverse_sections += 1

    def release(self, program: Program, generation: int, transaction_id: str) -> None:
        self.require(program, generation, transaction_id)
        if (
            self.cursor != 0 or self.numerator != [ZETA8, ZERO, ZERO, ZERO]
            or self.denominator != [ONE, ZERO, ZERO, ZERO]
            or any(value != ZERO for value in self.numerator_scratch)
            or any(value != ZERO for value in self.denominator_scratch)
            or any(self.receipts)
        ):
            raise RuntimeError("M256 exact restoration predicate failed")
        self.last_restored_generation = generation
        self.generation = 0
        self.owner = 0
        self.controller = 0
        self.program_id = ""
        self.output_type = ""
        self.transaction_id = ""
        self.consumer = 0
        self.leased_descriptor = None
        self.leased = False


def evaluate_polynomial(coefficients: Sequence[K], point: K, work: Work | None = None) -> K:
    result = ZERO
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
        if work is not None:
            work.evaluation_field_multiplications += 1
            work.evaluation_field_additions += 1
    return result


def polynomial_degree(coefficients: Sequence[K]) -> int:
    return next(
        (index for index in range(len(coefficients) - 1, -1, -1) if coefficients[index] != ZERO),
        -1,
    )


def project_boundary(
    carrier: Carrier, program: Program, generation: int, transaction_id: str, work: Work,
) -> tuple[int, K]:
    carrier.require(program, generation, transaction_id)
    if (
        carrier.cursor != 3
        or any(value != ZERO for value in carrier.numerator_scratch)
        or any(value != ZERO for value in carrier.denominator_scratch)
    ):
        raise RuntimeError("M256 premature waveform projection")
    numerator = evaluate_polynomial(carrier.numerator, ZETA8, work)
    denominator = evaluate_polynomial(carrier.denominator, ZETA8, work)
    if denominator == ZERO:
        raise RuntimeError("M256 zero final evaluation denominator")
    work.evaluation_field_inversions += 1
    return 3, numerator * denominator.inverse()


def run_transaction(carrier: Carrier, program: Program, request: dict[str, Any]) -> dict[str, Any]:
    backing_ids = tuple(map(id, (
        carrier.numerator, carrier.denominator, carrier.numerator_scratch,
        carrier.denominator_scratch, carrier.receipts,
    )))
    work = Work()
    transaction_id = request["transaction_id"]
    generation = carrier.lease(program, request["expected_generation"], transaction_id)
    response: tuple[int, K] | None = None
    final_degrees: tuple[int, int] | None = None
    failure: Exception | None = None
    try:
        for index in range(len(program.sections)):
            carrier.forward(program, generation, transaction_id, work)
            if request.get("inject_failure_after_partial") and index == 1:
                raise RuntimeError("injected M256 partial-forward failure")
        response = project_boundary(carrier, program, generation, transaction_id, work)
        final_degrees = (
            polynomial_degree(carrier.numerator), polynomial_degree(carrier.denominator)
        )
        retained = response, final_degrees
        if request.get("inject_failure_after_projection"):
            raise RuntimeError("injected M256 post-projection failure")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if delay:
            time.sleep(delay / 1000)
    except Exception as error:
        failure = error
    finally:
        while carrier.cursor:
            carrier.inverse(program, generation, transaction_id, work)
        carrier.release(program, generation, transaction_id)
    if failure is not None:
        raise failure
    if (response, final_degrees) != retained:
        raise RuntimeError("M256 boundary did not survive inverse")
    if response is None or final_degrees is None:
        raise RuntimeError("M256 response missing")
    same_backings = backing_ids == tuple(map(id, (
        carrier.numerator, carrier.denominator, carrier.numerator_scratch,
        carrier.denominator_scratch, carrier.receipts,
    )))
    return {
        "status": "OK",
        "output_type": OUTPUT_TYPE,
        "generation": generation,
        "winding": response[0],
        "evaluation": k_json(response[1]),
        "same_waveform_scratch_and_receipt_backings": same_backings,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "response_released_after_restoration": True,
        "full_descriptor_bound_to_lease": True,
        "resource_shape": {
            "resident_waveform_field_cells": 2 * CAPACITY,
            "scratch_waveform_field_cells": 2 * CAPACITY,
            "receipt_rational_cells": 3,
            "retained_final_boundary_field_cells_during_inverse": 1,
            "retained_final_winding_integer_cells_during_inverse": 1,
            "retained_final_degree_integer_cells_during_inverse": 2,
            "dynamic_inverse_history_field_cells": 0,
            "allocated_polynomial_capacity_each": CAPACITY,
            "actual_final_numerator_degree": final_degrees[0],
            "actual_final_denominator_degree": final_degrees[1],
        },
        "work": work.__dict__,
    }


def rejected(action: Any) -> bool:
    try:
        action()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def self_tests() -> dict[str, bool]:
    descriptor = {
        "sections": [[1, 2], [-1, 3], [1, 4]],
        "evaluation": "ZETA8",
        "output_type": OUTPUT_TYPE,
    }
    program = canonical_program(descriptor)

    def forwarded(name: str) -> tuple[Carrier, int, str]:
        carrier = Carrier(name)
        transaction_id = f"M256_{name.upper()}"
        generation = carrier.lease(program, 1, transaction_id)
        for _ in program.sections:
            carrier.forward(program, generation, transaction_id, Work())
        return carrier, generation, transaction_id

    missing_carrier = Carrier("missing")
    missing_transaction = "M256_MISSING"
    generation = missing_carrier.lease(program, 1, missing_transaction)
    for _ in program.sections:
        missing_carrier.forward(program, generation, missing_transaction, Work())
    missing = rejected(lambda: missing_carrier.release(program, generation, missing_transaction))

    wrong_carrier, wrong_generation, wrong_transaction = forwarded("wrong")
    wrong = rejected(lambda: wrong_carrier.inverse(
        program, wrong_generation, wrong_transaction, Work(), override=Fraction(1, 5)
    ))

    reorder_carrier, reorder_generation, reorder_transaction = forwarded("reorder")
    # Receipt custody rejects the first prospectively reordered inverse before mutation.
    reordered = rejected(lambda: reorder_carrier.inverse(
        program, reorder_generation, reorder_transaction, Work(), override=program.sections[1]
    ))

    premature_carrier = Carrier("premature")
    premature_transaction = "M256_PREMATURE"
    premature_generation = premature_carrier.lease(program, 1, premature_transaction)
    premature_carrier.forward(program, premature_generation, premature_transaction, Work())
    premature = rejected(lambda: project_boundary(
        premature_carrier, program, premature_generation, premature_transaction, Work()
    ))

    dirty_divisibility, divisibility_generation, divisibility_transaction = forwarded("divisibility")
    dirty_divisibility.numerator[0] = dirty_divisibility.numerator[0] + ONE
    divisibility = rejected(lambda: dirty_divisibility.inverse(
        program, divisibility_generation, divisibility_transaction, Work()
    ))

    zero_denominator, zero_generation, zero_transaction = forwarded("zero-denominator")
    zero_denominator.denominator[:] = [ZERO] * CAPACITY
    zero_evaluation = rejected(lambda: project_boundary(
        zero_denominator, program, zero_generation, zero_transaction, Work()
    ))

    mutated = Program(
        (program.sections[0], program.sections[1], Fraction(1, 5)),
        program.evaluation, program.output_type, program.program_id,
    )
    mutation_carrier = Carrier("same-id-mutation")
    mutation_transaction = "M256_SAME_ID_MUTATION"
    mutation_generation = mutation_carrier.lease(program, 1, mutation_transaction)
    same_id_mutation = rejected(lambda: mutation_carrier.require(
        mutated, mutation_generation, mutation_transaction
    ))

    custody_carrier = Carrier("custody")
    custody_transaction = "M256_CUSTODY"
    custody_generation = custody_carrier.lease(program, 1, custody_transaction)
    custody_carrier.consumer += 1
    wrong_consumer = rejected(lambda: custody_carrier.require(
        program, custody_generation, custody_transaction
    ))

    dirty_scratch = Carrier("scratch")
    dirty_transaction = "M256_SCRATCH"
    dirty_generation = dirty_scratch.lease(program, 1, dirty_transaction)
    dirty_scratch.numerator_scratch[0] = ONE
    scratch_rejected = rejected(lambda: dirty_scratch.forward(
        program, dirty_generation, dirty_transaction, Work()
    ))

    order_ab = Carrier("order-ab")
    transaction_ab = "M256_ORDER_AB"
    generation_ab = order_ab.lease(program, 1, transaction_ab)
    order_ab.forward(program, generation_ab, transaction_ab, Work())
    order_ab.forward(program, generation_ab, transaction_ab, Work())
    swapped_program = canonical_program({
        "sections": [[-1, 3], [1, 2], [1, 4]],
        "evaluation": "ZETA8", "output_type": OUTPUT_TYPE,
    })
    order_ba = Carrier("order-ba")
    transaction_ba = "M256_ORDER_BA"
    generation_ba = order_ba.lease(swapped_program, 1, transaction_ba)
    order_ba.forward(swapped_program, generation_ba, transaction_ba, Work())
    order_ba.forward(swapped_program, generation_ba, transaction_ba, Work())
    noncommuting = (
        order_ab.numerator != order_ba.numerator
        or order_ab.denominator != order_ba.denominator
    )

    malformed = {"sections": [[1, 1], [0, 1], [0, 1]], "evaluation": "ZETA8", "output_type": OUTPUT_TYPE}
    return {
        "missing_inverse_release_rejected": missing,
        "wrong_inverse_rejected_before_mutation": wrong,
        "reordered_inverse_rejected_by_receipt_custody": reordered,
        "premature_projection_rejected": premature,
        "failed_inverse_divisibility_rejected": divisibility,
        "zero_evaluation_denominator_rejected": zero_evaluation,
        "same_id_changed_descriptor_rejected": same_id_mutation,
        "wrong_consumer_rejected": wrong_consumer,
        "dirty_scratch_rejected_before_mutation": scratch_rejected,
        "declared_section_pair_is_prospectively_noncommuting": noncommuting,
        "singular_section_rejected": rejected(lambda: canonical_program(malformed)),
        "null_carrier_rejected": rejected(lambda: Carrier("null").forward(
            program, 1, "M256_NULL", Work()
        )),
    }


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise RuntimeError(f"M256 PR_SET_DUMPABLE failed errno={ctypes.get_errno()}")


def recv_json(connection: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
        if len(data) > 1_000_000:
            raise RuntimeError("M256 request too large")
    value = json.loads(bytes(data).split(b"\n", 1)[0])
    if not isinstance(value, dict):
        raise RuntimeError("M256 request object required")
    return value


def send_json(connection: socket.socket, value: dict[str, Any]) -> None:
    connection.sendall(canonical_payload(value) + b"\n")


RUN_KEYS = {
    "command", "carrier_id", "expected_generation", "descriptor", "port_type",
    "output_type", "controller_id", "owner", "consumer_id", "transaction_id",
    "program_id",
}
TEST_KEYS = {
    "inject_failure_after_partial", "inject_failure_after_projection",
    "test_delay_before_inverse_ms",
}


def validate_run_request(request: dict[str, Any]) -> Program:
    if not RUN_KEYS <= set(request) or set(request) - RUN_KEYS - TEST_KEYS:
        raise RuntimeError("M256 request shape rejected")
    carrier_id = request["carrier_id"]
    transaction_id = request["transaction_id"]
    expected_generation = request["expected_generation"]
    if (
        request["command"] != "RUN"
        or not isinstance(carrier_id, str) or not carrier_id
        or not isinstance(transaction_id, str) or not transaction_id
        or not isinstance(expected_generation, int) or isinstance(expected_generation, bool)
        or expected_generation <= 0
        or request["port_type"] != PORT_TYPE
        or request["output_type"] != OUTPUT_TYPE
        or request["controller_id"] != CONTROLLER
        or request["owner"] != OWNER
        or request["consumer_id"] != BOUNDARY_CONSUMER
    ):
        raise RuntimeError("M256 request custody rejected")
    for key in ("inject_failure_after_partial", "inject_failure_after_projection"):
        if key in request and not isinstance(request[key], bool):
            raise RuntimeError("M256 fault flag rejected")
    delay = request.get("test_delay_before_inverse_ms", 0)
    if not isinstance(delay, int) or isinstance(delay, bool) or not 0 <= delay <= 1000:
        raise RuntimeError("M256 test delay rejected")
    program = canonical_program(request["descriptor"])
    if request["program_id"] != program.program_id:
        raise RuntimeError("M256 program digest mismatch")
    return program


def serve(socket_name: str) -> None:
    if not socket_name.startswith("@catvm-m256-"):
        raise RuntimeError("M256 requires a declared abstract Unix socket")
    mode_line = sys.stdin.readline()
    sys.stdin.close()
    if json.loads(mode_line) != {"service": "M256_SCHUR_ALLPASS_WAVEFORM_MODE"}:
        raise RuntimeError("M256 service mode rejected")
    set_nondumpable()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = "\0" + socket_name[1:]
    server.bind(address)
    server.listen(8)
    carriers: dict[str, Carrier] = {}
    while True:
        connection, _ = server.accept()
        should_stop = False
        with connection:
            try:
                request = recv_json(connection)
                command = request.get("command")
                if command == "STOP":
                    response = {"status": "STOPPED"}
                    should_stop = True
                elif command == "SELF_TEST":
                    tests = self_tests()
                    if not all(tests.values()):
                        raise RuntimeError("M256 backend self-test failed")
                    response = {"status": "OK", "controls": tests}
                elif command in {
                    "SNAPSHOT_RUN", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
                    "PROJECT_COEFFICIENTS", "PROJECT_NUMERATOR", "PROJECT_DENOMINATOR",
                    "PROJECT_SEED", "PROJECT_INTERMEDIATE", "DEBUG_DUMP", "DUMP",
                    "NULL_CARRIER",
                }:
                    raise RuntimeError("M256 forbidden command")
                elif command == "STATUS":
                    carrier_id = request.get("carrier_id")
                    if not isinstance(carrier_id, str) or not carrier_id:
                        raise RuntimeError("M256 carrier id rejected")
                    carrier = carriers.get(carrier_id)
                    response = {
                        "status": "OK",
                        "exists": carrier is not None,
                        "canonical": carrier.canonical() if carrier else True,
                        "leased": carrier.leased if carrier else False,
                        "last_restored_generation": carrier.last_restored_generation if carrier else 0,
                    }
                else:
                    program = validate_run_request(request)
                    carrier_id = request["carrier_id"]
                    carrier = carriers.setdefault(carrier_id, Carrier(carrier_id))
                    response = run_transaction(carrier, program, request)
            except Exception as error:
                response = {"status": "REJECTED", "error_type": type(error).__name__}
            try:
                send_json(connection, response)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        if should_stop:
            break
    server.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: catvm_schur_allpass_waveform_service.py SOCKET")
    serve(sys.argv[1])
