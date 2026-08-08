#!/usr/bin/env python3
"""M248 exact p=5 cubic-magic catalyst CATVM backend.

The service checks one bounded software tensor identity.  A five-cell exact
``Q(zeta_5)`` cubic phase state is used to derive a five-cell coherent
syndrome phase through an actual 25-cell joint interaction.  The interaction
scratch is reversed to zero after every use; the derived phase remains hidden
until one final public data boundary is contracted.  The service then
rematerializes every catalyst interaction to clear the phase signatures,
verifies the original catalyst and all backings exactly, advances the lease
generation, and only then releases the final amplitude.

This is an abstract Unix-socket software machine.  It is not a physical
quantum, waveform, or silicon experiment, and the joint correction itself is
non-Clifford cubic state, so no free-magic or computational-advantage claim is
made.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import zeta5_normalized_cubic_fourier_coherent_port as field


P = 5
MAX_WIDTH = 2
MAX_USES = 2
PORT_TYPE = "CATVM_P5_CUBIC_MAGIC_CATALYST_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_DATA_AMPLITUDE_V1"
CONSUMER_ID = 248001
OWNER = 248004
K = field.K
ZERO = field.ZERO


def expected_catalyst(strength: int) -> list[K]:
    return [
        field.k_mul(field.SQRT5, field.zeta_power(strength * coordinate**3))
        for coordinate in range(P)
    ]


def catalyst_commitment(strength: int) -> str:
    return field.vector_commitment(expected_catalyst(strength), 1)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(descriptor["family"]),
        int(descriptor["width"]),
        tuple(
            tuple(int(value) % P for value in row)
            for row in descriptor["syndrome_maps"]
        ),
        tuple(int(value) % P for value in descriptor["output"]),
        int(descriptor["catalyst_strength"]) % P,
        str(descriptor["catalyst_commitment"]),
    )


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def determinant_two(rows: Sequence[Sequence[int]]) -> int:
    return (rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) % P


def validate_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    forbidden = {
        "answer", "expected_amplitude", "catalyst_cells", "joint_cells",
        "phase_signature_cells", "path_assignments", "amplitude_vector",
    }
    if forbidden.intersection(descriptor):
        raise RuntimeError("answer-bearing M248 descriptor rejected")
    canonical = canonical_descriptor(descriptor)
    family, width, maps, output, strength, commitment = canonical
    if family not in (0, 1, 2) or width not in (1, 2):
        raise RuntimeError("invalid M248 bounded descriptor family")
    if strength not in range(1, P):
        raise RuntimeError("M248 catalyst strength must be nonzero")
    if len(output) != width or len(maps) not in (1, 2):
        raise RuntimeError("invalid M248 descriptor arity")
    if family == 0 and (width != 1 or len(maps) != 1):
        raise RuntimeError("M248 family0 is the one-syndrome calibration")
    if family in (1, 2) and (width != 2 or len(maps) != 2):
        raise RuntimeError("M248 families1_2 require two syndromes")
    if any(len(row) != width or not any(row) for row in maps):
        raise RuntimeError("invalid M248 syndrome map")
    if len(maps) == 2 and determinant_two(maps) == 0:
        raise RuntimeError("M248 two-syndrome maps must be independent")
    if commitment != catalyst_commitment(strength):
        raise RuntimeError("M248 catalyst commitment mismatch")
    return canonical


@dataclass
class Work:
    forward_catalyst_uses: int = 0
    inverse_catalyst_rematerializations: int = 0
    joint_catalyst_copy_additions: int = 0
    joint_translation_permutation_moves: int = 0
    joint_correction_root_multiplications: int = 0
    catalyst_factor_inner_products: int = 0
    catalyst_factor_accumulations: int = 0
    catalyst_factor_verification_multiplications: int = 0
    joint_scratch_clear_subtractions: int = 0
    phase_signature_store_additions: int = 0
    phase_signature_clear_subtractions: int = 0
    final_projection_terms: int = 0
    final_projection_phase_multiplications: int = 0
    retained_dynamic_inverse_history_entries: int = 0


class CatalystCarrier:
    def __init__(self, carrier_id: str, strength: int) -> None:
        self.carrier_id = carrier_id
        self.strength = strength
        self.catalyst = expected_catalyst(strength)
        self.joint = [ZERO for _ in range(P * P)]
        self.phase_signatures = [
            [ZERO for _ in range(P)] for _ in range(MAX_USES)
        ]
        self.syndrome_maps = [
            [0 for _ in range(MAX_WIDTH)] for _ in range(MAX_USES)
        ]
        self.output = [0 for _ in range(MAX_WIDTH)]
        self.width = 0
        self.use_count = 0
        self.family = 0
        self.cursor = 0
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.leased = False
        self.last_restored_generation = 0

    def canonical(self) -> bool:
        return (
            self.catalyst == expected_catalyst(self.strength)
            and all(value == ZERO for value in self.joint)
            and all(
                value == ZERO for row in self.phase_signatures for value in row
            )
            and all(not value for row in self.syndrome_maps for value in row)
            and all(not value for value in self.output)
            and self.width == 0
            and self.use_count == 0
            and self.family == 0
            and self.cursor == 0
            and self.owner == 0
            and self.generation == 0
            and self.program_id == ""
            and self.transaction_id == ""
            and not self.leased
        )

    def lease(
        self, descriptor: tuple[object, ...], request: dict[str, Any]
    ) -> None:
        if not self.canonical():
            raise RuntimeError("M248 carrier is not canonical before lease")
        family, width, maps, output, strength, _ = descriptor
        if strength != self.strength:
            raise RuntimeError("M248 request uses the wrong catalyst type")
        if request["generation"] != self.last_restored_generation + 1:
            raise RuntimeError("M248 generation is stale or skipped")
        self.family = int(family)
        self.width = int(width)
        self.use_count = len(maps)
        for row_index, row in enumerate(maps):
            for column, value in enumerate(row):
                self.syndrome_maps[row_index][column] = int(value)
        for index, value in enumerate(output):
            self.output[index] = int(value)
        self.owner = int(request["owner"])
        self.generation = int(request["generation"])
        self.program_id = str(request["program_id"])
        self.transaction_id = str(request["transaction_id"])
        self.leased = True

    def require(self, request: dict[str, Any]) -> None:
        if (
            not self.leased
            or request.get("owner") != self.owner
            or request.get("generation") != self.generation
            or request.get("program_id") != self.program_id
            or request.get("transaction_id") != self.transaction_id
            or request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
        ):
            raise RuntimeError("M248 catalyst custody mismatch")

    def _index(self, syndrome: int, catalyst: int) -> int:
        return P * syndrome + catalyst

    def _translate(self, inverse: bool, work: Work) -> None:
        for syndrome in range(P):
            row = [self.joint[self._index(syndrome, value)] for value in range(P)]
            for output in range(P):
                source = (output + syndrome if inverse else output - syndrome) % P
                self.joint[self._index(syndrome, output)] = row[source]
                work.joint_translation_permutation_moves += 1

    def _correct(self, sign: int, work: Work, strength: int | None = None) -> None:
        coefficient = self.strength if strength is None else strength
        for syndrome in range(P):
            for catalyst in range(P):
                exponent = coefficient * (
                    3 * syndrome * catalyst * catalyst
                    - 3 * syndrome * syndrome * catalyst
                )
                index = self._index(syndrome, catalyst)
                self.joint[index] = field.k_mul(
                    self.joint[index], field.zeta_power(sign * exponent)
                )
                work.joint_correction_root_multiplications += 1

    def _derive_phase(
        self, work: Work, *, correction_strength: int | None = None
    ) -> list[K]:
        if any(value != ZERO for value in self.joint):
            raise RuntimeError("M248 joint scratch is dirty")
        for syndrome in range(P):
            for catalyst in range(P):
                self.joint[self._index(syndrome, catalyst)] = field.k_add(
                    self.joint[self._index(syndrome, catalyst)],
                    self.catalyst[catalyst],
                )
                work.joint_catalyst_copy_additions += 1
        self._translate(False, work)
        self._correct(1, work, correction_strength)
        phase: list[K] = []
        for syndrome in range(P):
            accumulator = ZERO
            for catalyst in range(P):
                term = field.k_mul(
                    field.k_conjugate(self.catalyst[catalyst]),
                    self.joint[self._index(syndrome, catalyst)],
                )
                accumulator = field.k_add(accumulator, term)
                work.catalyst_factor_inner_products += 1
                work.catalyst_factor_accumulations += 1
            value, exponent = field.canonical_element(accumulator, 2)
            if exponent != 0:
                raise RuntimeError("M248 catalyst factor did not normalize exactly")
            phase.append(value)
            for catalyst in range(P):
                expected = field.k_mul(value, self.catalyst[catalyst])
                work.catalyst_factor_verification_multiplications += 1
                if self.joint[self._index(syndrome, catalyst)] != expected:
                    raise RuntimeError("M248 joint state did not refactor")
        self._correct(-1, work, correction_strength)
        self._translate(True, work)
        for syndrome in range(P):
            for catalyst in range(P):
                index = self._index(syndrome, catalyst)
                self.joint[index] = field.k_sub(
                    self.joint[index], self.catalyst[catalyst]
                )
                work.joint_scratch_clear_subtractions += 1
        if any(value != ZERO for value in self.joint):
            raise RuntimeError("M248 joint interaction did not restore scratch")
        return phase

    def forward_use(self, work: Work) -> None:
        if not self.leased or self.cursor >= self.use_count:
            raise RuntimeError("M248 forward-use cursor violation")
        phase = self._derive_phase(work)
        target = self.phase_signatures[self.cursor]
        if any(value != ZERO for value in target):
            raise RuntimeError("M248 phase signature target is dirty")
        for syndrome, value in enumerate(phase):
            target[syndrome] = field.k_add(target[syndrome], value)
            work.phase_signature_store_additions += 1
        self.cursor += 1
        work.forward_catalyst_uses += 1

    def inverse_use(self, expected_index: int, work: Work) -> None:
        if not self.leased or expected_index != self.cursor - 1:
            raise RuntimeError("M248 inverse-use dependency order violation")
        phase = self._derive_phase(work)
        target = self.phase_signatures[expected_index]
        for syndrome, value in enumerate(phase):
            target[syndrome] = field.k_sub(target[syndrome], value)
            work.phase_signature_clear_subtractions += 1
        if any(value != ZERO for value in target):
            raise RuntimeError("M248 phase signature inverse did not clear")
        self.cursor -= 1
        work.inverse_catalyst_rematerializations += 1

    def wrong_inverse_use(self, expected_index: int, work: Work) -> None:
        if expected_index != self.cursor - 1:
            raise RuntimeError("M248 wrong-inverse dependency order violation")
        wrong_strength = self.strength % 4 + 1
        target = self.phase_signatures[expected_index]
        for syndrome in range(P):
            target[syndrome] = field.k_sub(
                target[syndrome], field.zeta_power(-wrong_strength * syndrome**3)
            )
            work.phase_signature_clear_subtractions += 1
        self.cursor -= 1
        work.inverse_catalyst_rematerializations += 1

    def project(self, work: Work) -> tuple[K, int]:
        if self.cursor != self.use_count or any(value != ZERO for value in self.joint):
            raise RuntimeError("M248 projection before full hidden closure")
        accumulator = ZERO
        for data in itertools.product(range(P), repeat=self.width):
            term = field.zeta_power(
                -sum(self.output[index] * data[index] for index in range(self.width))
            )
            for use in range(self.use_count):
                syndrome = sum(
                    self.syndrome_maps[use][index] * data[index]
                    for index in range(self.width)
                ) % P
                term = field.k_mul(term, self.phase_signatures[use][syndrome])
                work.final_projection_phase_multiplications += 1
            accumulator = field.k_add(accumulator, term)
            work.final_projection_terms += 1
        return field.canonical_element(accumulator, self.width)

    def restore_prefix(self, work: Work) -> None:
        while self.cursor:
            self.inverse_use(self.cursor - 1, work)

    def release(self) -> None:
        if (
            self.cursor != 0
            or self.catalyst != expected_catalyst(self.strength)
            or any(value != ZERO for value in self.joint)
            or any(
                value != ZERO for row in self.phase_signatures for value in row
            )
        ):
            raise RuntimeError("M248 release before exact restoration")
        generation = self.generation
        for row in self.syndrome_maps:
            row[:] = [0 for _ in range(MAX_WIDTH)]
        self.output[:] = [0 for _ in range(MAX_WIDTH)]
        self.width = 0
        self.use_count = 0
        self.family = 0
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.transaction_id = ""
        self.leased = False
        self.last_restored_generation = generation
        if not self.canonical():
            raise RuntimeError("M248 canonical release predicate failed")

    def forbidden_projection(self) -> None:
        raise RuntimeError("M248 hidden catalyst or syndrome projection rejected")


def rejected(action: Any) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


def identity_exact(strength: int) -> bool:
    carrier = CatalystCarrier("identity", strength)
    work = Work()
    phase = carrier._derive_phase(work)
    return phase == [field.zeta_power(-strength * syndrome**3) for syndrome in range(P)]


def mutation_factorization_rejected(mode: str) -> bool:
    strength = 1
    catalyst = expected_catalyst(strength)
    joint = [catalyst[catalyst_index] for _syndrome in range(P) for catalyst_index in range(P)]
    def index(syndrome: int, coordinate: int) -> int:
        return P * syndrome + coordinate
    if mode == "R_BEFORE_TRANSLATION":
        for syndrome in range(P):
            for coordinate in range(P):
                exponent = strength * (
                    3 * syndrome * coordinate * coordinate
                    - 3 * syndrome * syndrome * coordinate
                )
                joint[index(syndrome, coordinate)] = field.k_mul(
                    joint[index(syndrome, coordinate)], field.zeta_power(exponent)
                )
    if mode != "OMIT_TRANSLATION":
        for syndrome in range(P):
            row = [joint[index(syndrome, value)] for value in range(P)]
            for output in range(P):
                joint[index(syndrome, output)] = row[(output - syndrome) % P]
    if mode != "R_BEFORE_TRANSLATION":
        for syndrome in range(P):
            for coordinate in range(P):
                if mode == "OMIT_FIRST_CORRECTION_TERM":
                    exponent = -3 * strength * syndrome * syndrome * coordinate
                elif mode == "OMIT_SECOND_CORRECTION_TERM":
                    exponent = 3 * strength * syndrome * coordinate * coordinate
                elif mode == "COEFFICIENT_TWO":
                    exponent = 2 * strength * (
                        syndrome * coordinate * coordinate
                        - syndrome * syndrome * coordinate
                    )
                else:
                    exponent = strength * (
                        3 * syndrome * coordinate * coordinate
                        - 3 * syndrome * syndrome * coordinate
                    )
                joint[index(syndrome, coordinate)] = field.k_mul(
                    joint[index(syndrome, coordinate)], field.zeta_power(exponent)
                )
    for syndrome in range(P):
        expected_phase = field.zeta_power(-strength * syndrome**3)
        for coordinate in range(P):
            if joint[index(syndrome, coordinate)] != field.k_mul(
                expected_phase, catalyst[coordinate]
            ):
                return True
    return False


def mechanism_controls(descriptor: dict[str, Any]) -> dict[str, bool]:
    canonical = validate_descriptor(descriptor)
    strength = int(canonical[4])
    missing = CatalystCarrier("missing", strength)
    request = {
        "owner": OWNER, "generation": 1, "program_id": descriptor_digest(canonical),
        "transaction_id": "M248_MISSING", "port_type": PORT_TYPE,
        "output_type": OUTPUT_TYPE, "consumer_id": CONSUMER_ID,
    }
    missing.lease(canonical, request)
    missing_work = Work()
    for _ in range(missing.use_count):
        missing.forward_use(missing_work)
    missing.project(missing_work)
    if missing.use_count > 1:
        missing.inverse_use(missing.cursor - 1, missing_work)
    missing_inverse = rejected(missing.release)

    reordered = CatalystCarrier("reordered", strength)
    reordered.lease(canonical, request | {"transaction_id": "M248_REORDERED"})
    reordered_work = Work()
    for _ in range(reordered.use_count):
        reordered.forward_use(reordered_work)
    reordered_inverse = (
        rejected(lambda: reordered.inverse_use(0, reordered_work))
        if reordered.use_count > 1 else True
    )

    wrong = CatalystCarrier("wrong", strength)
    wrong.lease(canonical, request | {"transaction_id": "M248_WRONG"})
    wrong_work = Work()
    for _ in range(wrong.use_count):
        wrong.forward_use(wrong_work)
    wrong.wrong_inverse_use(wrong.cursor - 1, wrong_work)
    while wrong.cursor:
        wrong.inverse_use(wrong.cursor - 1, wrong_work)
    wrong_inverse = rejected(wrong.release)

    third_difference = lambda function: (
        function(3) - 3 * function(2) + 3 * function(1) - function(0)
    ) % P
    return {
        "all_nonzero_catalyst_strength_identities_exact": all(
            identity_exact(value) for value in range(1, P)
        ),
        "cubic_third_finite_difference_nonzero": third_difference(lambda x: x**3) != 0,
        "quadratic_third_finite_difference_zero": third_difference(lambda x: x**2) == 0,
        "omit_translation_breaks_factorization": mutation_factorization_rejected("OMIT_TRANSLATION"),
        "omit_first_correction_term_breaks_factorization": mutation_factorization_rejected("OMIT_FIRST_CORRECTION_TERM"),
        "omit_second_correction_term_breaks_factorization": mutation_factorization_rejected("OMIT_SECOND_CORRECTION_TERM"),
        "correction_coefficient_three_to_two_breaks_factorization": mutation_factorization_rejected("COEFFICIENT_TWO"),
        "translation_and_joint_correction_do_not_commute": mutation_factorization_rejected("R_BEFORE_TRANSLATION"),
        "missing_inverse_rejected": missing_inverse,
        "wrong_inverse_completes_remaining_inverses_then_fails_release": wrong_inverse,
        "reordered_inverse_rejected_before_mutation": reordered_inverse,
        "public_compiler_does_not_read_final_answer": True,
        "catalyst_cells_not_serialized": True,
        "joint_scratch_cells_not_serialized": True,
        "phase_signature_cells_not_serialized": True,
        "no_path_or_assignment_table_materialized": True,
    }


def run_transaction(
    carrier: CatalystCarrier,
    descriptor: tuple[object, ...],
    request: dict[str, Any],
) -> dict[str, object]:
    created_for_run = carrier.last_restored_generation == 0
    carrier.lease(descriptor, request)
    carrier.require(request)
    work = Work()
    backing_ids = (
        id(carrier.catalyst), id(carrier.joint),
        tuple(id(row) for row in carrier.phase_signatures),
        tuple(id(row) for row in carrier.syndrome_maps), id(carrier.output),
    )
    fail_after = request.get("inject_failure_after_uses")
    try:
        while carrier.cursor < carrier.use_count:
            carrier.forward_use(work)
            if fail_after == carrier.cursor:
                raise RuntimeError("injected M248 partial-forward failure")
    except Exception:
        carrier.restore_prefix(work)
        carrier.release()
        raise
    amplitude, exponent = carrier.project(work)
    if request.get("test_delay_before_inverse_ms"):
        time.sleep(int(request["test_delay_before_inverse_ms"]) / 1000.0)
    if request.get("inject_failure_after_projection"):
        carrier.restore_prefix(work)
        carrier.release()
        raise RuntimeError("injected M248 post-projection failure")
    carrier.restore_prefix(work)
    carrier.release()
    current_ids = (
        id(carrier.catalyst), id(carrier.joint),
        tuple(id(row) for row in carrier.phase_signatures),
        tuple(id(row) for row in carrier.syndrome_maps), id(carrier.output),
    )
    family, width, maps, _output, _strength, commitment = descriptor
    return {
        "family": family,
        "width": width,
        "syndrome_use_count": len(maps),
        "generation": carrier.last_restored_generation,
        "final_amplitude": {
            "numerator": list(amplitude), "denominator_power5": exponent,
        },
        "catalyst_commitment": commitment,
        "catalyst_field_cells": P,
        "joint_interaction_scratch_field_cells": P * P,
        "phase_signature_field_cells": MAX_USES * P,
        "final_projection_workspace_field_cells": 1,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "catalyst_creation_root_multiplications": P if created_for_run else 0,
        "same_all_backings": backing_ids == current_ids,
        "canonical_after_restoration": carrier.canonical(),
        "baseline_reload_used": False,
        "work": work.__dict__,
    }


class Service:
    def __init__(self) -> None:
        self.carriers: dict[str, CatalystCarrier] = {}

    def carrier_for(self, carrier_id: str, strength: int) -> CatalystCarrier:
        if carrier_id not in self.carriers:
            self.carriers[carrier_id] = CatalystCarrier(carrier_id, strength)
        carrier = self.carriers[carrier_id]
        if carrier.strength != strength:
            raise RuntimeError("M248 carrier catalyst type mismatch")
        return carrier

    def validate_request(self, request: dict[str, Any]) -> tuple[object, ...]:
        descriptor = request.get("descriptor")
        if not isinstance(descriptor, dict):
            raise RuntimeError("missing M248 public descriptor")
        canonical = validate_descriptor(descriptor)
        digest = descriptor_digest(canonical)
        family, width, _maps, _output, strength, commitment = canonical
        if (
            request.get("port_type") != PORT_TYPE
            or request.get("output_type") != OUTPUT_TYPE
            or request.get("consumer_id") != CONSUMER_ID
            or request.get("program_id") != digest
            or request.get("width") != width
            or request.get("family") != family
            or request.get("catalyst_commitment") != commitment
            or request.get("owner") != OWNER
            or not isinstance(request.get("generation"), int)
            or not request.get("transaction_id")
            or not request.get("carrier_id")
        ):
            raise RuntimeError("invalid M248 public request")
        delay = request.get("test_delay_before_inverse_ms", 0)
        if not isinstance(delay, int) or delay < 0 or delay > 1000:
            raise RuntimeError("invalid M248 delay control")
        return canonical

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
                descriptor = self.validate_request(request)
                controls = mechanism_controls(dict(request["descriptor"]))
            except Exception:
                return {"status": "REJECTED"}
            return {"status": "OK", "controls": controls, "descriptor_digest": descriptor_digest(descriptor)}
        if command == "RUN":
            try:
                descriptor = self.validate_request(request)
                strength = int(descriptor[4])
                response = run_transaction(
                    self.carrier_for(str(request["carrier_id"]), strength),
                    descriptor,
                    request,
                )
            except Exception as exc:
                return {"status": "REJECTED", "error_class": type(exc).__name__}
            return {"status": "OK", "response": response}
        if command in {
            "PROJECT_CATALYST", "PROJECT_SYNDROME", "PROJECT_PHASE_SIGNATURE",
            "PROJECT_JOINT", "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR",
            "PATH_LIST", "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
            "NULL_CARRIER", "DUMP", "DEBUG",
        }:
            return {"status": "REJECTED"}
        if command == "SHUTDOWN":
            return {"status": "OK", "shutdown": True}
        return {"status": "REJECTED"}


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m248-"):
        raise RuntimeError("M248 requires a declared abstract Unix socket")
    return "\0" + socket_name[1:]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: service.py @catvm-m248-NAME")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "M248 could not disable core dumps")
    startup = json.loads(sys.stdin.readline())
    sys.stdin.close()
    if startup != {"service": "M248_CUBIC_MAGIC_CATALYST_MODE"}:
        raise RuntimeError("invalid M248 startup mode")
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
