#!/usr/bin/env python3
"""M242 backend: atomic tensor-factored p=5 hidden phase oracle.

The linear phase oracle preserves an exact rank-one tensor product.  The
backend therefore retains five exact Q(zeta5) amplitudes per wire rather than
materializing the 5**n global amplitude vector.  Responses are constructed
only after exact inverse restoration of the actual factor backings.
"""

from __future__ import annotations

import ctypes
import json
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import zeta5_normalized_cubic_fourier_coherent_port as m237


P = 5
DIMENSIONS = (1, 2, 4, 8, 16, 32)
PORT_TYPE = "CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_V1"
OUTPUT_TYPE = "F5_SECRET_VECTOR_FINAL_BOUNDARY_V1"
CONSUMER_ID = 242001
K = m237.K
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5


def canonicalize(values: list[K], exponent: int) -> int:
    while exponent and all(coefficient % P == 0 for value in values for coefficient in value):
        for index, value in enumerate(values):
            values[index] = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


@dataclass
class Work:
    forward_coherent_oracle_queries: int = 0
    inverse_coherent_oracle_queries: int = 0
    hidden_secret_residue_accesses: int = 0
    oracle_factor_cell_visits: int = 0
    factor_fourier_character_terms: int = 0
    boundary_factor_cell_visits: int = 0
    restoration_verification_factor_cell_visits: int = 0
    common_factor_cancellations: int = 0
    retained_dynamic_inverse_history_entries: int = 0
    dense_global_amplitude_cells_materialized: int = 0
    exception_rollback_factor_cells: int = 0
    peak_common_denominator_power5_per_factor: int = 0
    retained_final_boundary_residue_cells_during_inverse: int = 0


class FactorCarrier:
    def __init__(self, dimension: int, carrier_id: str) -> None:
        self.dimension = dimension
        self.carrier_id = carrier_id
        self.values = [ZERO] * (P * dimension)
        for wire in range(dimension):
            self.values[P * wire] = ONE
        self.scratch = [ZERO] * (P * dimension)
        self.exponent = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.public_descriptor: tuple[object, ...] = ()
        self.generation = 0
        self.last_restored_generation = 0
        self.leased = False
        self.pending_oracle_factors = 0

    @property
    def factor_cells(self) -> int:
        return P * self.dimension

    def canonical(self) -> bool:
        return (
            all(
                self.values[P * wire] == ONE
                and all(self.values[P * wire + value] == ZERO for value in range(1, P))
                for wire in range(self.dimension)
            )
            and all(value == ZERO for value in self.scratch)
            and self.exponent == 0
            and self.stage == "CANONICAL"
            and self.pending_oracle_factors == 0
            and self.owner == 0
            and self.transaction_id == ""
            and self.oracle_id == ""
            and self.program_id == ""
            and self.public_descriptor == ()
            and self.generation == 0
            and not self.leased
        )

    def lease(
        self,
        oracle_id: str,
        program_id: str,
        public_descriptor: tuple[object, ...],
        owner: int,
        generation: int,
        transaction_id: str,
    ) -> None:
        if owner <= 0 or not transaction_id or self.leased or not self.canonical():
            raise RuntimeError("invalid M242 CATVM lease")
        if generation != self.last_restored_generation + 1:
            raise RuntimeError("nonmonotone M242 generation")
        self.leased = True
        self.owner = owner
        self.generation = generation
        self.oracle_id = oracle_id
        self.program_id = program_id
        self.public_descriptor = public_descriptor
        self.transaction_id = transaction_id
        self.stage = "LEASED"

    def fourier_all(self, direction: int, work: Work) -> None:
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("dirty M242 factor scratch")
        for wire in range(self.dimension):
            offset = P * wire
            for output in range(P):
                total = ZERO
                for source in range(P):
                    total = m237.k_add(
                        total,
                        m237.k_mul(
                            m237.zeta_power(direction * source * output),
                            self.values[offset + source],
                        ),
                    )
                    work.factor_fourier_character_terms += 1
                self.scratch[offset + output] = m237.k_mul(SQRT5, total)
        self.values[:] = self.scratch
        self.scratch[:] = [ZERO] * self.factor_cells
        self.exponent += 1
        work.peak_common_denominator_power5_per_factor = max(
            work.peak_common_denominator_power5_per_factor,
            self.exponent,
        )
        before = self.exponent
        self.exponent = canonicalize(self.values, self.exponent)
        work.common_factor_cancellations += before - self.exponent

    def _apply_oracle_factor(self, wire: int, residue: int, direction: int, work: Work) -> None:
        offset = P * wire
        for value in range(P):
            self.scratch[offset + value] = m237.k_mul(
                m237.zeta_power(direction * residue * value),
                self.values[offset + value],
            )
            work.oracle_factor_cell_visits += 1
        self.values[offset:offset + P] = self.scratch[offset:offset + P]
        self.scratch[offset:offset + P] = [ZERO] * P

    def oracle(
        self,
        secret: Sequence[int],
        direction: int,
        work: Work,
        inject_failure_after_factors: int = 0,
    ) -> None:
        if len(secret) != self.dimension:
            raise RuntimeError("M242 oracle dimension mismatch")
        if inject_failure_after_factors < 0 or inject_failure_after_factors > self.dimension:
            raise RuntimeError("invalid M242 partial failure cursor")
        self.pending_oracle_factors = 0
        try:
            for wire, residue in enumerate(secret):
                work.hidden_secret_residue_accesses += 1
                self._apply_oracle_factor(wire, residue, direction, work)
                self.pending_oracle_factors += 1
                if inject_failure_after_factors == self.pending_oracle_factors:
                    raise RuntimeError("injected M242 partial oracle failure")
        except Exception:
            self.scratch[:] = [ZERO] * self.factor_cells
            while self.pending_oracle_factors:
                wire = self.pending_oracle_factors - 1
                self._apply_oracle_factor(wire, secret[wire], -direction, work)
                work.exception_rollback_factor_cells += P
                self.pending_oracle_factors -= 1
            raise
        self.pending_oracle_factors = 0
        if direction == 1:
            work.forward_coherent_oracle_queries += 1
        else:
            work.inverse_coherent_oracle_queries += 1

    def project_secret(self, work: Work) -> tuple[int, ...]:
        if self.stage != "FINAL_BOUNDARY_RESIDENT":
            raise RuntimeError("premature M242 projection")
        if self.exponent != 0:
            raise RuntimeError("M242 final factors are not canonical basis factors")
        inferred: list[int] = []
        for wire in range(self.dimension):
            offset = P * wire
            support: list[int] = []
            for value in range(P):
                work.boundary_factor_cell_visits += 1
                if self.values[offset + value] != ZERO:
                    support.append(value)
            if len(support) != 1 or self.values[offset + support[0]] != ONE:
                raise RuntimeError("M242 boundary factor is not an exact basis state")
            inferred.append(support[0])
        return tuple(inferred)

    def release(self, work: Work) -> None:
        if self.stage != "RESTORATION_VERIFIED":
            raise RuntimeError("M242 response release before restoration")
        for wire in range(self.dimension):
            offset = P * wire
            for value in range(P):
                work.restoration_verification_factor_cell_visits += 1
                expected = ONE if value == 0 else ZERO
                if self.values[offset + value] != expected:
                    raise RuntimeError("M242 factor carrier not exactly restored")
        if (
            any(value != ZERO for value in self.scratch)
            or self.exponent != 0
            or self.pending_oracle_factors != 0
        ):
            raise RuntimeError("M242 scratch or exponent not restored")
        self.last_restored_generation = self.generation
        self.leased = False
        self.owner = 0
        self.generation = 0
        self.oracle_id = ""
        self.program_id = ""
        self.public_descriptor = ()
        self.transaction_id = ""
        self.stage = "CANONICAL"


def execute_atomic(
    carrier: FactorCarrier,
    oracle_id: str,
    program_id: str,
    public_descriptor: tuple[object, ...],
    secret: Sequence[int],
    owner: int,
    generation: int,
    transaction_id: str,
    delay_before_inverse_ms: int = 0,
    inject_failure_after_projection: bool = False,
    inject_failure_after_factors: int = 0,
) -> dict[str, object]:
    if delay_before_inverse_ms < 0:
        raise RuntimeError("negative M242 delay rejected before lease")
    carrier.lease(
        oracle_id,
        program_id,
        public_descriptor,
        owner,
        generation,
        transaction_id,
    )
    values_backing = id(carrier.values)
    scratch_backing = id(carrier.scratch)
    work = Work()
    applied_count = 0

    def forward(step: int) -> None:
        if step == 0:
            carrier.stage = "FORWARD_PRODUCT_SUPERPOSITION"
            carrier.fourier_all(1, work)
        elif step == 1:
            carrier.stage = "ORACLE_PRODUCT_PHASE_RESIDENT"
            carrier.oracle(secret, 1, work, inject_failure_after_factors)
        elif step == 2:
            carrier.stage = "PRODUCT_DECODING"
            carrier.fourier_all(-1, work)
        else:
            raise RuntimeError("invalid M242 forward cursor")

    def inverse(step: int) -> None:
        if step == 0:
            carrier.stage = "INVERSE_PRODUCT_SUPERPOSITION"
            carrier.fourier_all(-1, work)
        elif step == 1:
            carrier.stage = "INVERSE_PRODUCT_ORACLE"
            carrier.oracle(secret, -1, work)
        elif step == 2:
            carrier.stage = "INVERSE_PRODUCT_DECODING"
            carrier.fourier_all(1, work)
        else:
            raise RuntimeError("invalid M242 inverse cursor")

    try:
        for step in range(3):
            forward(step)
            applied_count += 1
        carrier.stage = "FINAL_BOUNDARY_RESIDENT"
        inferred = carrier.project_secret(work)
        work.retained_final_boundary_residue_cells_during_inverse = len(inferred)
        if delay_before_inverse_ms:
            time.sleep(delay_before_inverse_ms / 1000)
        if inject_failure_after_projection:
            raise RuntimeError("injected M242 post-projection failure")
        while applied_count:
            inverse(applied_count - 1)
            applied_count -= 1
        carrier.stage = "RESTORATION_VERIFIED"
        carrier.release(work)
    except Exception:
        if carrier.leased:
            while applied_count:
                inverse(applied_count - 1)
                applied_count -= 1
            carrier.stage = "RESTORATION_VERIFIED"
            carrier.release(work)
        raise

    return {
        "oracle_id": oracle_id,
        "carrier_id": carrier.carrier_id,
        "dimension": carrier.dimension,
        "inferred_secret": list(inferred),
        "abstract_forward_coherent_phase_queries": 1,
        "actual_inverse_oracle_queries": 1,
        "factorization_rank": 1,
        "global_basis_cells_not_materialized": P**carrier.dimension,
        "response_released_after_restoration": True,
        "canonical_post_inverse_state_exact": carrier.canonical(),
        "same_values_and_scratch_backings": (
            id(carrier.values) == values_backing and id(carrier.scratch) == scratch_backing
        ),
        "restoration_generation": carrier.last_restored_generation,
        "baseline_reload_used": False,
        "carrier_field_cells": carrier.factor_cells,
        "scratch_field_cells": carrier.factor_cells,
        "carrier_integer_coordinate_cells": 4 * carrier.factor_cells,
        "scratch_integer_coordinate_cells": 4 * carrier.factor_cells,
        "secret_residue_cells": carrier.dimension,
        "work": asdict(work),
    }


def control_suite(dimension: int, secret: Sequence[int]) -> dict[str, bool]:
    def restored_after(mode: str) -> bool:
        carrier = FactorCarrier(dimension, "control")
        work = Work()
        carrier.fourier_all(1, work)
        carrier.oracle(secret, 1, work)
        carrier.fourier_all(-1, work)
        if mode == "WRONG":
            carrier.fourier_all(1, work)
            wrong = list(secret)
            wrong[0] = (wrong[0] + 1) % P
            carrier.oracle(wrong, -1, work)
            carrier.fourier_all(-1, work)
        elif mode == "MISSING":
            carrier.fourier_all(1, work)
            carrier.oracle(secret, -1, work)
        elif mode == "REORDERED":
            carrier.oracle(secret, -1, work)
            carrier.fourier_all(1, work)
            carrier.fourier_all(-1, work)
        return carrier.canonical()

    premature = FactorCarrier(dimension, "premature")
    premature.stage = "ORACLE_PRODUCT_PHASE_RESIDENT"
    try:
        premature.project_secret(Work())
        premature_rejected = False
    except RuntimeError:
        premature_rejected = True

    negative = FactorCarrier(dimension, "negative-delay")
    try:
        execute_atomic(
            negative,
            "negative-delay",
            "negative-delay",
            (PORT_TYPE, dimension, "negative-delay", OUTPUT_TYPE, CONSUMER_ID),
            secret,
            242999,
            1,
            "M242_NEGATIVE_DELAY",
            delay_before_inverse_ms=-1,
        )
        negative_delay_rejected = False
    except RuntimeError:
        negative_delay_rejected = negative.canonical() and negative.last_restored_generation == 0

    def accepted(control_secret: Sequence[int], carrier_id: str) -> tuple[int, ...]:
        carrier = FactorCarrier(dimension, carrier_id)
        response = execute_atomic(
            carrier,
            carrier_id,
            carrier_id,
            (PORT_TYPE, dimension, carrier_id, OUTPUT_TYPE, CONSUMER_ID),
            control_secret,
            242998,
            1,
            f"M242_{carrier_id}",
        )
        if not carrier.canonical() or carrier.last_restored_generation != 1:
            raise RuntimeError("M242 algebra control failed restoration")
        return tuple(response["inferred_secret"])  # type: ignore[arg-type]

    zero = (0,) * dimension
    repeated = (2,) * dimension
    mixed = tuple((index * index + index) % P for index in range(dimension))
    perturbed = list(mixed)
    perturb_index = dimension // 2
    perturbed[perturb_index] = (perturbed[perturb_index] + 1) % P
    mixed_boundary = accepted(mixed, "mixed")
    perturbed_boundary = accepted(tuple(perturbed), "perturbed")
    changed = [
        index
        for index, (left, right) in enumerate(zip(mixed_boundary, perturbed_boundary))
        if left != right
    ]
    return {
        "wrong_inverse_fails_exact_restoration": not restored_after("WRONG"),
        "missing_inverse_fails_exact_restoration": not restored_after("MISSING"),
        "reordered_inverse_fails_exact_restoration": not restored_after("REORDERED"),
        "projection_during_hidden_factor_phase_residency_rejected": premature_rejected,
        "negative_delay_rejected_before_lease": negative_delay_rejected,
        "all_zero_secret_exact_and_restored": accepted(zero, "zero") == zero,
        "repeated_secret_exact_and_restored": accepted(repeated, "repeated") == repeated,
        "mixed_secret_exact_and_restored": mixed_boundary == mixed,
        "one_coordinate_perturbation_changes_only_that_boundary_coordinate": (
            changed == [perturb_index]
        ),
    }


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError("unable to disable M242 backend dumpability")


def send(connection: socket.socket, payload: dict[str, object]) -> None:
    connection.sendall(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def main(socket_name: str) -> None:
    set_nondumpable()
    config_line = sys.stdin.buffer.readline()
    if not config_line:
        raise RuntimeError("missing private M242 oracle configuration")
    config = json.loads(config_line)
    sys.stdin.close()
    oracles = config["oracles"]
    carriers: dict[str, FactorCarrier] = {}
    for oracle_id, descriptor in oracles.items():
        dimension = int(descriptor["dimension"])
        secret = tuple(int(value) % P for value in descriptor["secret"])
        if dimension not in DIMENSIONS or len(secret) != dimension:
            raise RuntimeError("private oracle configuration outside M242 scope")
        descriptor["secret"] = secret
        delay = int(descriptor.get("delay_before_inverse_ms", 0))
        if delay < 0:
            raise RuntimeError("negative private delay outside M242 scope")
        descriptor["delay_before_inverse_ms"] = delay
        descriptor["inject_failure_after_projection"] = bool(
            descriptor.get("inject_failure_after_projection", False)
        )
        partial_failure = int(descriptor.get("inject_failure_after_factors", 0))
        if partial_failure < 0 or partial_failure > dimension:
            raise RuntimeError("partial failure cursor outside M242 scope")
        descriptor["inject_failure_after_factors"] = partial_failure
        descriptor["program_id"] = str(descriptor.get("program_id", oracle_id))
        carrier_id = str(descriptor["carrier_id"])
        if carrier_id in carriers and carriers[carrier_id].dimension != dimension:
            raise RuntimeError("M242 carrier reused across incompatible dimensions")
        carriers.setdefault(carrier_id, FactorCarrier(dimension, carrier_id))
    if not socket_name.startswith("@catvm-m242-"):
        raise RuntimeError("M242 requires an abstract Unix socket")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\0" + socket_name[1:])
    listener.listen(8)
    running = True
    while running:
        connection, _ = listener.accept()
        with connection:
            request_line = b""
            while not request_line.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                request_line += chunk
            if not request_line:
                continue
            try:
                request = json.loads(request_line)
                command = request.get("command")
                if command == "RUN":
                    oracle_id = str(request["oracle_id"])
                    descriptor = oracles[oracle_id]
                    if (
                        str(request.get("port_type")) != PORT_TYPE
                        or int(request.get("dimension", -1)) != int(descriptor["dimension"])
                        or str(request.get("program_id")) != descriptor["program_id"]
                        or str(request.get("output_type")) != OUTPUT_TYPE
                        or int(request.get("consumer_id", -1)) != CONSUMER_ID
                    ):
                        raise RuntimeError("M242 public descriptor mismatch")
                    carrier = carriers[str(descriptor["carrier_id"])]
                    response = execute_atomic(
                        carrier,
                        oracle_id,
                        descriptor["program_id"],
                        (
                            PORT_TYPE,
                            int(descriptor["dimension"]),
                            descriptor["program_id"],
                            OUTPUT_TYPE,
                            CONSUMER_ID,
                        ),
                        descriptor["secret"],
                        int(request["owner"]),
                        int(request["generation"]),
                        str(request["transaction_id"]),
                        descriptor["delay_before_inverse_ms"],
                        descriptor["inject_failure_after_projection"],
                        descriptor["inject_failure_after_factors"],
                    )
                    send(connection, {"status": "OK", "response": response})
                elif command == "STATUS":
                    descriptor = oracles[str(request["oracle_id"])]
                    carrier = carriers[str(descriptor["carrier_id"])]
                    send(connection, {
                        "status": "OK",
                        "canonical": carrier.canonical(),
                        "leased": carrier.leased,
                        "last_restored_generation": carrier.last_restored_generation,
                    })
                elif command == "CONTROLS":
                    descriptor = oracles[str(request["oracle_id"])]
                    send(connection, {
                        "status": "OK",
                        "controls": control_suite(int(descriptor["dimension"]), descriptor["secret"]),
                    })
                elif command in {
                    "PROJECT_INTERMEDIATE", "PROJECT_FACTORS", "DENSE_GLOBAL_VECTOR",
                    "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER",
                }:
                    send(connection, {"status": "REJECTED", "reason": "COMMAND_OUTSIDE_M242_ATOMIC_SERVICE"})
                elif command == "STOP":
                    send(connection, {
                        "status": "OK",
                        "all_carriers_canonical": all(carrier.canonical() for carrier in carriers.values()),
                    })
                    running = False
                else:
                    send(connection, {"status": "REJECTED", "reason": "UNKNOWN_COMMAND"})
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                try:
                    send(connection, {"status": "REJECTED", "reason": type(error).__name__})
                except BrokenPipeError:
                    pass
            except BrokenPipeError:
                pass
    listener.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: catvm_p5_tensor_factored_hidden_linear_phase_oracle_service.py SOCKET")
    main(sys.argv[1])
