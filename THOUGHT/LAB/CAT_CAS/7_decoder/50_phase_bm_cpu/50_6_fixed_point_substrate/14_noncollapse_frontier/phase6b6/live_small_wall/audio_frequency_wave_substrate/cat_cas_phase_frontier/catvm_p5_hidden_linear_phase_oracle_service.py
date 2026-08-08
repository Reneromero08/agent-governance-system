#!/usr/bin/env python3
"""M241 backend: atomic exact p=5 hidden linear phase-oracle service.

The service receives its oracle configuration on stdin before accepting Unix
socket requests.  RUN responses are constructed only after exact inverse
restoration of the actual carrier.  No command projects the resident phase
vector or oracle value before that release point.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import zeta5_normalized_cubic_fourier_coherent_port as m237


P = 5
PORT_TYPE = "CATVM_P5_HIDDEN_LINEAR_PHASE_ORACLE_V1"
K = m237.K
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5


def canonicalize(values: list[K], exponent: int) -> int:
    while exponent and all(c % P == 0 for value in values for c in value):
        for index, value in enumerate(values):
            values[index] = tuple(c // P for c in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def flat_index(state: Sequence[int]) -> int:
    result = 0
    for value in state:
        result = P * result + value
    return result


def coordinates(index: int, dimension: int) -> tuple[int, ...]:
    result = [0] * dimension
    for position in range(dimension - 1, -1, -1):
        result[position] = index % P
        index //= P
    return tuple(result)


def vector_commitment(values: Sequence[K], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(v) for v in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def secret_commitment(secret: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps({"p": P, "secret": list(secret)}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Work:
    forward_coherent_oracle_queries: int = 0
    inverse_coherent_oracle_queries: int = 0
    oracle_phase_cell_visits: int = 0
    fourier_character_terms: int = 0
    boundary_cell_visits: int = 0
    restoration_verification_cell_visits: int = 0
    common_factor_cancellations: int = 0
    retained_dynamic_inverse_history_entries: int = 0


class Carrier:
    def __init__(self, dimension: int, carrier_id: str) -> None:
        self.dimension = dimension
        self.cell_count = P**dimension
        self.carrier_id = carrier_id
        self.values = [ZERO] * self.cell_count
        self.values[0] = ONE
        self.scratch = [ZERO] * self.cell_count
        self.exponent = 0
        self.stage = "CANONICAL"
        self.owner = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.generation = 0
        self.last_restored_generation = 0
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.values[0] == ONE
            and all(value == ZERO for value in self.values[1:])
            and all(value == ZERO for value in self.scratch)
            and self.exponent == 0
            and self.stage == "CANONICAL"
        )

    def lease(self, oracle_id: str, owner: int, generation: int, transaction_id: str) -> None:
        if owner <= 0 or not transaction_id or self.leased or not self.canonical():
            raise RuntimeError("invalid CATVM oracle lease")
        if generation != self.last_restored_generation + 1:
            raise RuntimeError("nonmonotone CATVM oracle generation")
        self.leased = True
        self.owner = owner
        self.generation = generation
        self.oracle_id = oracle_id
        self.transaction_id = transaction_id
        self.stage = "LEASED"

    def require(self, oracle_id: str, owner: int, generation: int, transaction_id: str) -> None:
        if (
            not self.leased or oracle_id != self.oracle_id or owner != self.owner
            or generation != self.generation or transaction_id != self.transaction_id
        ):
            raise RuntimeError("CATVM oracle custody mismatch")

    def fourier(self, wire: int, direction: int, work: Work) -> None:
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("dirty CATVM oracle scratch")
        for destination in range(self.cell_count):
            output_state = list(coordinates(destination, self.dimension))
            output = output_state[wire]
            total = ZERO
            for source in range(P):
                source_state = output_state.copy()
                source_state[wire] = source
                total = m237.k_add(
                    total,
                    m237.k_mul(
                        m237.zeta_power(direction * source * output),
                        self.values[flat_index(source_state)],
                    ),
                )
                work.fourier_character_terms += 1
            self.scratch[destination] = m237.k_mul(SQRT5, total)
        self.values[:] = self.scratch
        self.scratch[:] = [ZERO] * self.cell_count
        self.exponent += 1
        before = self.exponent
        self.exponent = canonicalize(self.values, self.exponent)
        work.common_factor_cancellations += before - self.exponent

    def oracle(self, secret: Sequence[int], direction: int, work: Work) -> None:
        if len(secret) != self.dimension:
            raise RuntimeError("oracle dimension mismatch")
        for location in range(self.cell_count):
            state = coordinates(location, self.dimension)
            phase = sum(secret[index] * state[index] for index in range(self.dimension))
            self.values[location] = m237.k_mul(
                m237.zeta_power(direction * phase), self.values[location]
            )
            work.oracle_phase_cell_visits += 1
        if direction == 1:
            work.forward_coherent_oracle_queries += 1
        else:
            work.inverse_coherent_oracle_queries += 1

    def project_secret(self, work: Work) -> tuple[int, ...]:
        if self.stage != "FINAL_BOUNDARY_RESIDENT":
            raise RuntimeError("premature CATVM oracle projection")
        support: list[int] = []
        for location, value in enumerate(self.values):
            work.boundary_cell_visits += 1
            if value != ZERO:
                support.append(location)
        if len(support) != 1 or self.exponent != 0 or self.values[support[0]] != ONE:
            raise RuntimeError("oracle boundary is not an exact basis state")
        return coordinates(support[0], self.dimension)

    def release(self, work: Work) -> None:
        if self.stage != "RESTORATION_VERIFIED":
            raise RuntimeError("CATVM oracle response release before restoration")
        for location, value in enumerate(self.values):
            expected = ONE if location == 0 else ZERO
            work.restoration_verification_cell_visits += 1
            if value != expected:
                raise RuntimeError("CATVM oracle carrier not exactly restored")
        if any(value != ZERO for value in self.scratch) or self.exponent != 0:
            raise RuntimeError("CATVM oracle scratch or exponent not restored")
        self.last_restored_generation = self.generation
        self.leased = False
        self.owner = 0
        self.generation = 0
        self.oracle_id = ""
        self.transaction_id = ""
        self.stage = "CANONICAL"


def execute_atomic(
    carrier: Carrier,
    oracle_id: str,
    secret: Sequence[int],
    owner: int,
    generation: int,
    transaction_id: str,
    delay_before_inverse_ms: int = 0,
    inject_failure_after_projection: bool = False,
) -> dict[str, object]:
    if delay_before_inverse_ms < 0:
        raise RuntimeError("negative CATVM delay rejected before lease")
    carrier.lease(oracle_id, owner, generation, transaction_id)
    values_backing = id(carrier.values)
    scratch_backing = id(carrier.scratch)
    work = Work()
    applied_count = 0

    def apply_forward_step(step: int) -> None:
        if step < carrier.dimension:
            carrier.stage = "FORWARD_SUPERPOSITION"
            carrier.fourier(step, 1, work)
        elif step == carrier.dimension:
            carrier.stage = "ORACLE_PHASE_RESIDENT"
            carrier.oracle(secret, 1, work)
        else:
            carrier.stage = "DECODING"
            carrier.fourier(step - carrier.dimension - 1, -1, work)

    def apply_inverse_step(step: int) -> None:
        if step < carrier.dimension:
            carrier.stage = "INVERSE_SUPERPOSITION"
            carrier.fourier(step, -1, work)
        elif step == carrier.dimension:
            carrier.stage = "INVERSE_ORACLE"
            carrier.oracle(secret, -1, work)
        else:
            carrier.stage = "INVERSE_DECODING"
            carrier.fourier(step - carrier.dimension - 1, 1, work)

    try:
        for step in range(2 * carrier.dimension + 1):
            apply_forward_step(step)
            applied_count += 1
        carrier.stage = "FINAL_BOUNDARY_RESIDENT"
        inferred = carrier.project_secret(work)
        boundary_commitment = secret_commitment(inferred)
        state_commitment = vector_commitment(carrier.values, carrier.exponent)
        if delay_before_inverse_ms:
            time.sleep(delay_before_inverse_ms / 1000)
        if inject_failure_after_projection:
            raise RuntimeError("injected post-projection CATVM failure")
        while applied_count:
            apply_inverse_step(applied_count - 1)
            applied_count -= 1
        carrier.stage = "RESTORATION_VERIFIED"
        carrier.release(work)
    except Exception:
        if carrier.leased:
            while applied_count:
                apply_inverse_step(applied_count - 1)
                applied_count -= 1
            carrier.stage = "RESTORATION_VERIFIED"
            carrier.release(work)
        raise
    return {
        "oracle_id": oracle_id,
        "carrier_id": carrier.carrier_id,
        "dimension": carrier.dimension,
        "inferred_secret": list(inferred),
        "boundary_commitment": boundary_commitment,
        "final_basis_state_commitment": state_commitment,
        "abstract_forward_coherent_phase_queries": 1,
        "actual_inverse_oracle_queries": 1,
        "response_released_after_restoration": True,
        "canonical_post_inverse_state_exact": carrier.canonical(),
        "same_values_and_scratch_backings": (
            id(carrier.values) == values_backing and id(carrier.scratch) == scratch_backing
        ),
        "restoration_generation": carrier.last_restored_generation,
        "baseline_reload_used": False,
        "carrier_field_cells": carrier.cell_count,
        "scratch_field_cells": carrier.cell_count,
        "secret_residue_cells": carrier.dimension,
        "work": asdict(work),
    }


def control_suite(dimension: int, secret: Sequence[int]) -> dict[str, bool]:
    def restored_after(actions: str) -> bool:
        carrier = Carrier(dimension, "control")
        work = Work()
        for wire in range(dimension):
            carrier.fourier(wire, 1, work)
        carrier.oracle(secret, 1, work)
        for wire in range(dimension):
            carrier.fourier(wire, -1, work)
        if actions == "WRONG":
            for wire in reversed(range(dimension)):
                carrier.fourier(wire, 1, work)
            wrong = list(secret); wrong[0] = (wrong[0] + 1) % P
            carrier.oracle(wrong, -1, work)
            for wire in reversed(range(dimension)):
                carrier.fourier(wire, -1, work)
        elif actions == "MISSING":
            for wire in reversed(range(dimension)):
                carrier.fourier(wire, 1, work)
            carrier.oracle(secret, -1, work)
        elif actions == "REORDERED":
            carrier.oracle(secret, -1, work)
            for wire in reversed(range(dimension)):
                carrier.fourier(wire, 1, work)
            for wire in reversed(range(dimension)):
                carrier.fourier(wire, -1, work)
        return carrier.values[0] == ONE and all(value == ZERO for value in carrier.values[1:]) and carrier.exponent == 0

    premature = Carrier(dimension, "premature")
    premature.stage = "ORACLE_PHASE_RESIDENT"
    try:
        premature.project_secret(Work())
        premature_rejected = False
    except RuntimeError:
        premature_rejected = True
    return {
        "wrong_inverse_fails_exact_restoration": not restored_after("WRONG"),
        "missing_inverse_fails_exact_restoration": not restored_after("MISSING"),
        "reordered_inverse_fails_exact_restoration": not restored_after("REORDERED"),
        "projection_during_hidden_phase_residency_rejected": premature_rejected,
    }


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError("unable to disable backend dumpability")


def send(connection: socket.socket, payload: dict[str, object]) -> None:
    connection.sendall(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def main(socket_name: str) -> None:
    set_nondumpable()
    config_line = sys.stdin.buffer.readline()
    if not config_line:
        raise RuntimeError("missing private oracle configuration")
    config = json.loads(config_line)
    sys.stdin.close()
    oracles = config["oracles"]
    carriers: dict[str, Carrier] = {}
    for oracle_id, descriptor in oracles.items():
        dimension = int(descriptor["dimension"])
        secret = tuple(int(value) % P for value in descriptor["secret"])
        if dimension not in (1, 2, 3, 4) or len(secret) != dimension:
            raise RuntimeError("private oracle configuration outside M241 scope")
        descriptor["secret"] = secret
        delay = int(descriptor.get("delay_before_inverse_ms", 0))
        if delay < 0:
            raise RuntimeError("negative private delay outside M241 scope")
        descriptor["delay_before_inverse_ms"] = delay
        descriptor["inject_failure_after_projection"] = bool(
            descriptor.get("inject_failure_after_projection", False)
        )
        carrier_id = str(descriptor["carrier_id"])
        if carrier_id in carriers and carriers[carrier_id].dimension != dimension:
            raise RuntimeError("carrier reused across incompatible dimensions")
        carriers.setdefault(carrier_id, Carrier(dimension, carrier_id))
    if not socket_name.startswith("@catvm-m241-"):
        raise RuntimeError("M241 requires an abstract contained Unix socket")
    socket_address = "\0" + socket_name[1:]
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address)
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
                    carrier = carriers[str(descriptor["carrier_id"])]
                    response = execute_atomic(
                        carrier,
                        oracle_id,
                        descriptor["secret"],
                        int(request["owner"]),
                        int(request["generation"]),
                        str(request["transaction_id"]),
                        descriptor["delay_before_inverse_ms"],
                        descriptor["inject_failure_after_projection"],
                    )
                    send(connection, {"status": "OK", "response": response})
                elif command == "STATUS":
                    oracle_id = str(request["oracle_id"])
                    descriptor = oracles[oracle_id]
                    carrier = carriers[str(descriptor["carrier_id"])]
                    send(connection, {
                        "status": "OK",
                        "canonical": carrier.canonical(),
                        "leased": carrier.leased,
                        "last_restored_generation": carrier.last_restored_generation,
                    })
                elif command == "CONTROLS":
                    oracle_id = str(request["oracle_id"])
                    descriptor = oracles[oracle_id]
                    send(connection, {
                        "status": "OK",
                        "controls": control_suite(int(descriptor["dimension"]), descriptor["secret"]),
                    })
                elif command in {"PROJECT_INTERMEDIATE", "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER"}:
                    send(connection, {"status": "REJECTED", "reason": "COMMAND_OUTSIDE_IN_PLACE_ATOMIC_SERVICE"})
                elif command == "STOP":
                    send(connection, {"status": "OK", "all_carriers_canonical": all(c.canonical() for c in carriers.values())})
                    running = False
                else:
                    send(connection, {"status": "REJECTED", "reason": "UNKNOWN_COMMAND"})
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                try:
                    send(connection, {"status": "REJECTED", "reason": type(error).__name__})
                except BrokenPipeError:
                    pass
            except BrokenPipeError:
                # RUN has already restored and released before send() is attempted.
                pass
    listener.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: catvm_p5_hidden_linear_phase_oracle_service.py SOCKET_PATH")
    main(sys.argv[1])
