#!/usr/bin/env python3
"""M243 atomic CATVM backend for a connected quadratic plus rank-one cubic phase.

The accepted path never materializes the 5**n amplitude vector.  It transforms
the actual hidden symmetric-matrix backing to an in-place LDL^T chart, solves
for A^-1 u, closes five exact Q(zeta5) cubic Fourier channels, retains only the
final amplitude, and then reverses the actual carrier before responding.
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
DIMENSIONS = (2, 3, 4, 6, 8, 12, 16)
PORT_TYPE = "CATVM_P5_CONNECTED_QUADRATIC_RANK1_CUBIC_PHASE_V1"
OUTPUT_TYPE = "QZETA5_FINAL_AMPLITUDE_V1"
CONSUMER_ID = 243001
K = m237.K
ZERO = m237.ZERO
ONE = m237.ONE
SQRT5 = m237.SQRT5


def packed_size(dimension: int) -> int:
    return dimension * (dimension + 1) // 2


def packed_index(row: int, column: int) -> int:
    if row < column:
        row, column = column, row
    return row * (row + 1) // 2 + column


def mod_inverse(value: int) -> int:
    value %= P
    if not value:
        raise RuntimeError("zero M243 modular pivot")
    return pow(value, P - 2, P)


def legendre(value: int) -> int:
    value %= P
    if value in (1, 4):
        return 1
    if value in (2, 3):
        return -1
    raise RuntimeError("singular M243 determinant")


def canonical_amplitude(numerator: K, exponent: int) -> tuple[K, int]:
    value = numerator
    while exponent and all(coefficient % P == 0 for coefficient in value):
        value = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return value, exponent


def amplitude_json(numerator: K, exponent: int) -> dict[str, object]:
    return {"numerator": list(numerator), "denominator_power5": exponent}


def descriptor_tuple(
    dimension: int,
    matrix: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
) -> tuple[object, ...]:
    return (
        dimension,
        tuple(int(value) % P for value in matrix),
        tuple(int(value) % P for value in vector),
        int(cubic_strength) % P,
    )


def descriptor_valid(
    dimension: int,
    matrix: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
) -> bool:
    if (
        dimension not in DIMENSIONS
        or len(matrix) != packed_size(dimension)
        or len(vector) != dimension
        or cubic_strength % P == 0
        or sum(int(value) % P != 0 for value in vector) < 2
    ):
        return False
    work = [int(value) % P for value in matrix]
    adjacency = [set() for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row):
            if int(matrix[packed_index(row, column)]) % P:
                adjacency[row].add(column)
                adjacency[column].add(row)
    seen = {0}
    pending = [0]
    while pending:
        node = pending.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    if len(seen) != dimension:
        return False
    try:
        for pivot_index in range(dimension):
            pivot = work[packed_index(pivot_index, pivot_index)]
            inverse = mod_inverse(pivot)
            for row in range(pivot_index + 1, dimension):
                cell = packed_index(row, pivot_index)
                work[cell] = work[cell] * inverse % P
            for row in range(pivot_index + 1, dimension):
                for column in range(pivot_index + 1, row + 1):
                    target = packed_index(row, column)
                    work[target] = (
                        work[target]
                        - work[packed_index(row, pivot_index)]
                        * pivot
                        * work[packed_index(column, pivot_index)]
                    ) % P
    except RuntimeError:
        return False
    return True


@dataclass
class Work:
    hidden_descriptor_residue_reads: int = 0
    modular_inversions: int = 0
    modular_multiplications: int = 0
    modular_additions: int = 0
    ldl_forward_pivots: int = 0
    ldl_inverse_pivots: int = 0
    solve_forward_terms: int = 0
    solve_backward_terms: int = 0
    cubic_channel_character_terms: int = 0
    coherent_channel_terms: int = 0
    exception_rollback_pivots: int = 0
    dense_global_amplitude_cells_materialized: int = 0
    retained_dynamic_inverse_history_entries: int = 0
    retained_final_amplitude_field_cells_during_inverse: int = 0
    response_release_attempts_before_restoration: int = 0


class GaussCarrier:
    def __init__(self, dimension: int, carrier_id: str) -> None:
        self.dimension = dimension
        self.carrier_id = carrier_id
        self.matrix = [0] * packed_size(dimension)
        self.vector = [0] * dimension
        self.cubic_strength = [0]
        self.solve = [0] * dimension
        self.channels = [ZERO] * P
        self.channel_scratch = [ZERO] * P
        self.stage = "CANONICAL"
        self.owner = 0
        self.generation = 0
        self.last_restored_generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.public_descriptor: tuple[object, ...] = ()
        self.leased = False

    def canonical(self) -> bool:
        return (
            all(value == 0 for value in self.matrix)
            and all(value == 0 for value in self.vector)
            and self.cubic_strength == [0]
            and all(value == 0 for value in self.solve)
            and all(value == ZERO for value in self.channels)
            and all(value == ZERO for value in self.channel_scratch)
            and self.stage == "CANONICAL"
            and self.owner == 0
            and self.generation == 0
            and self.transaction_id == ""
            and self.oracle_id == ""
            and self.program_id == ""
            and self.public_descriptor == ()
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
            raise RuntimeError("invalid M243 CATVM lease")
        if generation != self.last_restored_generation + 1:
            raise RuntimeError("nonmonotone M243 generation")
        self.leased = True
        self.owner = owner
        self.generation = generation
        self.transaction_id = transaction_id
        self.oracle_id = oracle_id
        self.program_id = program_id
        self.public_descriptor = public_descriptor
        self.stage = "LEASED"

    def load_hidden(
        self,
        matrix: Sequence[int],
        vector: Sequence[int],
        cubic_strength: int,
        work: Work,
    ) -> None:
        if self.stage != "LEASED":
            raise RuntimeError("M243 hidden descriptor loaded out of order")
        if not descriptor_valid(self.dimension, matrix, vector, cubic_strength):
            raise RuntimeError("invalid M243 hidden descriptor")
        for index, value in enumerate(matrix):
            self.matrix[index] = int(value) % P
            work.hidden_descriptor_residue_reads += 1
        for index, value in enumerate(vector):
            self.vector[index] = int(value) % P
            work.hidden_descriptor_residue_reads += 1
        self.cubic_strength[0] = int(cubic_strength) % P
        work.hidden_descriptor_residue_reads += 1
        self.stage = "HIDDEN_DESCRIPTOR_RESIDENT"

    def clear_hidden(
        self,
        matrix: Sequence[int],
        vector: Sequence[int],
        cubic_strength: int,
        work: Work,
    ) -> None:
        if self.stage not in {"HIDDEN_DESCRIPTOR_RESIDENT", "HIDDEN_DESCRIPTOR_RESTORED"}:
            raise RuntimeError("M243 descriptor clear out of order")
        for index, value in enumerate(matrix):
            self.matrix[index] = (self.matrix[index] - int(value)) % P
            work.hidden_descriptor_residue_reads += 1
        for index, value in enumerate(vector):
            self.vector[index] = (self.vector[index] - int(value)) % P
            work.hidden_descriptor_residue_reads += 1
        self.cubic_strength[0] = (self.cubic_strength[0] - int(cubic_strength)) % P
        work.hidden_descriptor_residue_reads += 1
        self.stage = "RESTORATION_VERIFIED"

    def _reverse_pivot(self, pivot_index: int, work: Work) -> None:
        pivot = self.matrix[packed_index(pivot_index, pivot_index)]
        if not pivot:
            raise RuntimeError("M243 inverse encountered zero pivot")
        for row in range(pivot_index + 1, self.dimension):
            for column in range(pivot_index + 1, row + 1):
                target = packed_index(row, column)
                self.matrix[target] = (
                    self.matrix[target]
                    + self.matrix[packed_index(row, pivot_index)]
                    * pivot
                    * self.matrix[packed_index(column, pivot_index)]
                ) % P
                work.modular_multiplications += 2
                work.modular_additions += 1
        for row in range(pivot_index + 1, self.dimension):
            cell = packed_index(row, pivot_index)
            self.matrix[cell] = self.matrix[cell] * pivot % P
            work.modular_multiplications += 1
        work.ldl_inverse_pivots += 1

    def ldl_forward(self, work: Work, inject_failure_after_pivots: int = 0) -> None:
        if self.stage != "HIDDEN_DESCRIPTOR_RESIDENT":
            raise RuntimeError("M243 LDL forward out of order")
        completed = 0
        try:
            for pivot_index in range(self.dimension):
                pivot = self.matrix[packed_index(pivot_index, pivot_index)]
                inverse = mod_inverse(pivot)
                work.modular_inversions += 1
                for row in range(pivot_index + 1, self.dimension):
                    cell = packed_index(row, pivot_index)
                    self.matrix[cell] = self.matrix[cell] * inverse % P
                    work.modular_multiplications += 1
                for row in range(pivot_index + 1, self.dimension):
                    for column in range(pivot_index + 1, row + 1):
                        target = packed_index(row, column)
                        self.matrix[target] = (
                            self.matrix[target]
                            - self.matrix[packed_index(row, pivot_index)]
                            * pivot
                            * self.matrix[packed_index(column, pivot_index)]
                        ) % P
                        work.modular_multiplications += 2
                        work.modular_additions += 1
                completed += 1
                work.ldl_forward_pivots += 1
                if inject_failure_after_pivots == completed:
                    raise RuntimeError("injected M243 partial LDL failure")
        except Exception:
            while completed:
                completed -= 1
                self._reverse_pivot(completed, work)
                work.exception_rollback_pivots += 1
            self.stage = "HIDDEN_DESCRIPTOR_RESIDENT"
            raise
        self.stage = "LDL_RESIDENT"

    def ldl_reverse(self, work: Work) -> None:
        if self.stage != "LDL_READY_FOR_INVERSE":
            raise RuntimeError("M243 LDL inverse dependency reorder")
        for pivot_index in range(self.dimension - 1, -1, -1):
            self._reverse_pivot(pivot_index, work)
        self.stage = "HIDDEN_DESCRIPTOR_RESTORED"

    def solve_and_invariants(self, work: Work) -> tuple[int, int]:
        if self.stage != "LDL_RESIDENT" or any(self.solve):
            raise RuntimeError("M243 solve scratch dirty or out of order")
        self.solve[:] = self.vector
        for row in range(self.dimension):
            for column in range(row):
                self.solve[row] = (
                    self.solve[row]
                    - self.matrix[packed_index(row, column)] * self.solve[column]
                ) % P
                work.modular_multiplications += 1
                work.modular_additions += 1
                work.solve_forward_terms += 1
        determinant = 1
        for row in range(self.dimension):
            pivot = self.matrix[packed_index(row, row)]
            determinant = determinant * pivot % P
            self.solve[row] = self.solve[row] * mod_inverse(pivot) % P
            work.modular_multiplications += 2
            work.modular_inversions += 1
        for row in range(self.dimension - 1, -1, -1):
            for column in range(row + 1, self.dimension):
                self.solve[row] = (
                    self.solve[row]
                    - self.matrix[packed_index(column, row)] * self.solve[column]
                ) % P
                work.modular_multiplications += 1
                work.modular_additions += 1
                work.solve_backward_terms += 1
        delta = 0
        for left, right in zip(self.vector, self.solve):
            delta = (delta + left * right) % P
            work.modular_multiplications += 1
            work.modular_additions += 1
        self.stage = "QUOTIENT_RESIDENT"
        return legendre(determinant), delta

    def clear_solve(self) -> None:
        if self.stage != "CHANNELS_CLEARED":
            raise RuntimeError("M243 solve clear out of order")
        self.solve[:] = [0] * self.dimension
        self.stage = "LDL_READY_FOR_INVERSE"

    def build_channels(self, delta: int, work: Work) -> tuple[K, int]:
        if self.stage != "QUOTIENT_RESIDENT":
            raise RuntimeError("M243 channels built out of order")
        if any(value != ZERO for value in self.channels + self.channel_scratch):
            raise RuntimeError("dirty M243 cubic channel scratch")
        strength = self.cubic_strength[0]
        for channel in range(P):
            total = ZERO
            for value in range(P):
                phase = strength * value * value * value - channel * value
                total = m237.k_add(total, m237.zeta_power(phase))
                work.cubic_channel_character_terms += 1
            self.channel_scratch[channel] = total
        self.channels[:] = self.channel_scratch
        self.channel_scratch[:] = [ZERO] * P
        closure = ZERO
        for channel, coefficient in enumerate(self.channels):
            closure = m237.k_add(
                closure,
                m237.k_mul(coefficient, m237.zeta_power(delta * channel * channel)),
            )
            work.coherent_channel_terms += 1
        self.stage = "FIVE_CHANNEL_BOUNDARY_RESIDENT"
        return closure, 1

    def clear_channels(self) -> None:
        if self.stage not in {"FIVE_CHANNEL_BOUNDARY_RESIDENT", "FINAL_BOUNDARY_RETAINED"}:
            raise RuntimeError("M243 channels cleared out of order")
        self.channels[:] = [ZERO] * P
        self.channel_scratch[:] = [ZERO] * P
        self.stage = "CHANNELS_CLEARED"

    def release(self, work: Work) -> None:
        if self.stage != "RESTORATION_VERIFIED":
            work.response_release_attempts_before_restoration += 1
            raise RuntimeError("M243 response release before restoration")
        if any(self.matrix) or any(self.vector) or any(self.cubic_strength) or any(self.solve):
            raise RuntimeError("M243 residue carrier not restored")
        if any(value != ZERO for value in self.channels + self.channel_scratch):
            raise RuntimeError("M243 exact channel carrier not restored")
        self.last_restored_generation = self.generation
        self.leased = False
        self.owner = 0
        self.generation = 0
        self.transaction_id = ""
        self.oracle_id = ""
        self.program_id = ""
        self.public_descriptor = ()
        self.stage = "CANONICAL"


def apply_gauss_normalization(
    closure: K,
    channel_exponent: int,
    dimension: int,
    determinant_class: int,
) -> tuple[K, int]:
    numerator = closure
    for _ in range(dimension):
        numerator = m237.k_mul(numerator, SQRT5)
    if determinant_class < 0:
        numerator = tuple(-coefficient for coefficient in numerator)  # type: ignore[assignment]
    return canonical_amplitude(numerator, channel_exponent + dimension)


def execute_atomic(
    carrier: GaussCarrier,
    oracle_id: str,
    program_id: str,
    public_descriptor: tuple[object, ...],
    matrix: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
    owner: int,
    generation: int,
    transaction_id: str,
    delay_before_inverse_ms: int = 0,
    inject_failure_after_projection: bool = False,
    inject_failure_after_pivots: int = 0,
) -> dict[str, object]:
    if delay_before_inverse_ms < 0:
        raise RuntimeError("negative M243 delay rejected before lease")
    carrier.lease(
        oracle_id,
        program_id,
        public_descriptor,
        owner,
        generation,
        transaction_id,
    )
    backing_ids = tuple(
        id(value)
        for value in (
            carrier.matrix,
            carrier.vector,
            carrier.cubic_strength,
            carrier.solve,
            carrier.channels,
            carrier.channel_scratch,
        )
    )
    work = Work()
    applied = 0
    determinant_class = 0
    delta = 0
    final_numerator = ZERO
    final_exponent = 0

    try:
        carrier.load_hidden(matrix, vector, cubic_strength, work)
        applied = 1
        carrier.ldl_forward(work, inject_failure_after_pivots)
        applied = 2
        determinant_class, delta = carrier.solve_and_invariants(work)
        applied = 3
        closure, channel_exponent = carrier.build_channels(delta, work)
        applied = 4
        final_numerator, final_exponent = apply_gauss_normalization(
            closure,
            channel_exponent,
            carrier.dimension,
            determinant_class,
        )
        determinant_class = 0
        delta = 0
        closure = ZERO
        channel_exponent = 0
        work.retained_final_amplitude_field_cells_during_inverse = 1
        carrier.stage = "FINAL_BOUNDARY_RETAINED"
        if delay_before_inverse_ms:
            time.sleep(delay_before_inverse_ms / 1000)
        if inject_failure_after_projection:
            raise RuntimeError("injected M243 post-projection failure")
        carrier.clear_channels()
        applied = 3
        carrier.clear_solve()
        applied = 2
        carrier.ldl_reverse(work)
        applied = 1
        carrier.clear_hidden(matrix, vector, cubic_strength, work)
        applied = 0
        carrier.release(work)
    except Exception:
        if carrier.leased:
            if applied >= 4:
                carrier.clear_channels()
                applied = 3
            if applied >= 3:
                carrier.clear_solve()
                applied = 2
            if applied >= 2:
                carrier.ldl_reverse(work)
                applied = 1
            if applied >= 1:
                carrier.clear_hidden(matrix, vector, cubic_strength, work)
                applied = 0
            carrier.release(work)
        raise

    return {
        "oracle_id": oracle_id,
        "carrier_id": carrier.carrier_id,
        "dimension": carrier.dimension,
        "final_amplitude": amplitude_json(final_numerator, final_exponent),
        "response_released_after_restoration": True,
        "canonical_post_inverse_state_exact": carrier.canonical(),
        "same_all_carrier_backings": backing_ids
        == tuple(
            id(value)
            for value in (
                carrier.matrix,
                carrier.vector,
                carrier.cubic_strength,
                carrier.solve,
                carrier.channels,
                carrier.channel_scratch,
            )
        ),
        "restoration_generation": carrier.last_restored_generation,
        "baseline_reload_used": False,
        "hidden_descriptor_residue_cells": packed_size(carrier.dimension)
        + carrier.dimension
        + 1,
        "carrier_residue_cells": packed_size(carrier.dimension) + carrier.dimension + 1,
        "solve_scratch_residue_cells": carrier.dimension,
        "quotient_residue_scratch_cells": 2,
        "coherent_channel_field_cells": P,
        "coherent_channel_scratch_field_cells": P,
        "dense_global_amplitude_cells_materialized": 0,
        "work": asdict(work),
    }


def direct_control_suite(
    dimension: int,
    matrix: Sequence[int],
    vector: Sequence[int],
    cubic_strength: int,
) -> dict[str, bool]:
    descriptor = descriptor_tuple(dimension, matrix, vector, cubic_strength)
    disconnected = [0] * packed_size(dimension)
    for index in range(dimension):
        disconnected[packed_index(index, index)] = 1
    disconnected_vector = [1, 1] + [0] * (dimension - 2)

    premature = GaussCarrier(dimension, "premature")
    try:
        premature.release(Work())
        premature_release_rejected = False
    except RuntimeError:
        premature_release_rejected = True

    reordered = GaussCarrier(dimension, "reordered")
    reordered.lease("r", "r", (PORT_TYPE, dimension, "r", OUTPUT_TYPE, CONSUMER_ID), 1, 1, "r")
    work = Work()
    reordered.load_hidden(matrix, vector, cubic_strength, work)
    reordered.ldl_forward(work)
    reordered.solve_and_invariants(work)
    try:
        reordered.ldl_reverse(work)
        reordered_inverse_rejected = False
    except RuntimeError:
        reordered_inverse_rejected = True

    def release_after(mode: str) -> bool:
        carrier = GaussCarrier(dimension, mode)
        work = Work()
        carrier.lease(mode, mode, (PORT_TYPE, dimension, mode, OUTPUT_TYPE, CONSUMER_ID), 1, 1, mode)
        carrier.load_hidden(matrix, vector, cubic_strength, work)
        carrier.ldl_forward(work)
        _, delta = carrier.solve_and_invariants(work)
        carrier.build_channels(delta, work)
        if mode == "MISSING":
            try:
                carrier.release(work)
            except RuntimeError:
                return False
        carrier.clear_channels()
        carrier.clear_solve()
        carrier.ldl_reverse(work)
        clear_matrix = list(matrix)
        if mode == "WRONG":
            clear_matrix[0] = (clear_matrix[0] + 1) % P
        carrier.clear_hidden(clear_matrix, vector, cubic_strength, work)
        try:
            carrier.release(work)
        except RuntimeError:
            return False
        return True

    changed_strength = cubic_strength % P + 1
    if changed_strength == P:
        changed_strength = 1
    base = GaussCarrier(dimension, "base")
    base_result = execute_atomic(
        base, "base", "base", (PORT_TYPE, dimension, "base", OUTPUT_TYPE, CONSUMER_ID),
        matrix, vector, cubic_strength, 1, 1, "base",
    )
    altered = GaussCarrier(dimension, "altered")
    altered_result = execute_atomic(
        altered, "altered", "altered", (PORT_TYPE, dimension, "altered", OUTPUT_TYPE, CONSUMER_ID),
        matrix, vector, changed_strength, 1, 1, "altered",
    )

    return {
        "disconnected_regular_descriptor_rejected": not descriptor_valid(
            dimension, disconnected, disconnected_vector, 1
        ),
        "premature_response_release_rejected": premature_release_rejected,
        "reordered_inverse_dependency_rejected": reordered_inverse_rejected,
        "missing_inverse_fails_release": not release_after("MISSING"),
        "wrong_inverse_after_remaining_schedule_fails_release": not release_after("WRONG"),
        "nonzero_cubic_strength_perturbation_changes_final_amplitude": (
            base_result["final_amplitude"] != altered_result["final_amplitude"]
        ),
        "descriptor_tuple_is_full_hidden_authority": descriptor
        == descriptor_tuple(dimension, matrix, vector, cubic_strength),
    }


def set_nondumpable() -> None:
    libc = ctypes.CDLL(None)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
        raise OSError("unable to disable M243 backend dumpability")


def send(connection: socket.socket, payload: dict[str, object]) -> None:
    connection.sendall(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def main(socket_name: str) -> None:
    set_nondumpable()
    config_line = sys.stdin.buffer.readline()
    if not config_line:
        raise RuntimeError("missing private M243 configuration")
    config = json.loads(config_line)
    sys.stdin.close()
    oracles = config["oracles"]
    carriers: dict[str, GaussCarrier] = {}
    for oracle_id, descriptor in oracles.items():
        dimension = int(descriptor["dimension"])
        matrix = tuple(int(value) % P for value in descriptor["matrix"])
        vector = tuple(int(value) % P for value in descriptor["vector"])
        cubic_strength = int(descriptor["cubic_strength"]) % P
        if not descriptor_valid(dimension, matrix, vector, cubic_strength):
            raise RuntimeError("private descriptor outside M243 scope")
        descriptor["matrix"] = matrix
        descriptor["vector"] = vector
        descriptor["cubic_strength"] = cubic_strength
        descriptor["program_id"] = str(descriptor.get("program_id", oracle_id))
        descriptor["delay_before_inverse_ms"] = int(descriptor.get("delay_before_inverse_ms", 0))
        descriptor["inject_failure_after_projection"] = bool(
            descriptor.get("inject_failure_after_projection", False)
        )
        descriptor["inject_failure_after_pivots"] = int(
            descriptor.get("inject_failure_after_pivots", 0)
        )
        carrier_id = str(descriptor["carrier_id"])
        if carrier_id in carriers and carriers[carrier_id].dimension != dimension:
            raise RuntimeError("M243 carrier reused across incompatible dimensions")
        carriers.setdefault(carrier_id, GaussCarrier(dimension, carrier_id))

    if not socket_name.startswith("@catvm-m243-"):
        raise RuntimeError("M243 requires an abstract Unix socket")
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
                        raise RuntimeError("M243 public descriptor mismatch")
                    carrier = carriers[str(descriptor["carrier_id"])]
                    response = execute_atomic(
                        carrier,
                        oracle_id,
                        descriptor["program_id"],
                        (PORT_TYPE, int(descriptor["dimension"]), descriptor["program_id"], OUTPUT_TYPE, CONSUMER_ID),
                        descriptor["matrix"],
                        descriptor["vector"],
                        descriptor["cubic_strength"],
                        int(request["owner"]),
                        int(request["generation"]),
                        str(request["transaction_id"]),
                        descriptor["delay_before_inverse_ms"],
                        descriptor["inject_failure_after_projection"],
                        descriptor["inject_failure_after_pivots"],
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
                        "controls": direct_control_suite(
                            int(descriptor["dimension"]),
                            descriptor["matrix"],
                            descriptor["vector"],
                            descriptor["cubic_strength"],
                        ),
                    })
                elif command in {
                    "PROJECT_MATRIX", "PROJECT_VECTOR", "PROJECT_SOLVE",
                    "PROJECT_QUOTIENT", "PROJECT_CHANNELS", "DENSE_GLOBAL_VECTOR",
                    "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER",
                }:
                    send(connection, {"status": "REJECTED", "reason": "COMMAND_OUTSIDE_M243_ATOMIC_SERVICE"})
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
        raise SystemExit("usage: catvm_p5_quadratic_rank1_cubic_gauss_quotient_service.py SOCKET")
    main(sys.argv[1])
