#!/usr/bin/env python3
"""Exact M233 growing trivalent continuation-rank obstruction.

The accepted transaction remains the streamed M214 fusion-path carrier.  The
new verifier independently closes reachable and observable spaces under every
adjacent braid generator and inverse at N=4,6,8,10.  Split-prime full-rank
certificates decide whether the scalar plat language admits a smaller exact
Q(zeta40)-linear continuation state.  Verifier path lists and modular bases
are explicitly outside the accepted transaction.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import su2_level8_fusion_path_braid_phase_relation as m214


K = m214.K
ZERO = m214.ZERO
ONE = m214.ONE
A = m214.BRAID_A
A_INVERSE = m214.BRAID_A_INVERSE
DELTA = m214.QUANTUM_DIMENSIONS[1]
D_TWO = m214.QUANTUM_DIMENSIONS[2]
INVERSE_DELTA = m214.INVERSE_DIMENSIONS[1]
PHI = K.zeta(4) + K.zeta(-4)
DEPTHS = (1, 2, 4, 8, 16)
GROWING_STRANDS = (4, 6, 8, 10)
GROWING_DIMENSIONS = (2, 5, 14, 42)
CERTIFICATE_PRIMES = (241, 401)
PRIMARY_ROUNDS = 4
REUSE_ROUNDS = 3


def matrix_multiply(left: tuple[tuple[K, K], tuple[K, K]], right: tuple[tuple[K, K], tuple[K, K]]) -> tuple[tuple[K, K], tuple[K, K]]:
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(2)),
                ZERO,
            )
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


def matrix_add(left: tuple[tuple[K, K], tuple[K, K]], right: tuple[tuple[K, K], tuple[K, K]]) -> tuple[tuple[K, K], tuple[K, K]]:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def matrix_scale(value: K, matrix: tuple[tuple[K, K], tuple[K, K]]) -> tuple[tuple[K, K], tuple[K, K]]:
    return tuple(
        tuple(value * matrix[row][column] for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def matrix_commitment(matrix: tuple[tuple[K, K], tuple[K, K]]) -> str:
    payload = "|".join(value.token() for row in matrix for value in row)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


IDENTITY = ((ONE, ZERO), (ZERO, ONE))


def f_matrix(*, omit_offdiagonal: bool = False) -> tuple[tuple[K, K], tuple[K, K]]:
    offdiagonal = ZERO if omit_offdiagonal else PHI * INVERSE_DELTA
    return (
        (INVERSE_DELTA, offdiagonal),
        (offdiagonal, ZERO - INVERSE_DELTA),
    )


def r_matrix(exponent: int, *, swap: bool = False) -> tuple[tuple[K, K], tuple[K, K]]:
    if exponent not in (-1, 1):
        raise ValueError("M233 exponent outside declared support")
    if exponent == 1:
        r_zero = A + A_INVERSE * DELTA
        r_two = A
    else:
        r_zero = A_INVERSE + A * DELTA
        r_two = A_INVERSE
    if swap:
        r_zero, r_two = r_two, r_zero
    return ((r_zero, ZERO), (ZERO, r_two))


def symmetric_trivalent_braid(exponent: int, *, omit_offdiagonal: bool = False, swap: bool = False) -> tuple[tuple[K, K], tuple[K, K]]:
    f_move = f_matrix(omit_offdiagonal=omit_offdiagonal)
    return matrix_multiply(matrix_multiply(f_move, r_matrix(exponent, swap=swap)), f_move)


def path_temperley_lieb_braid(exponent: int) -> tuple[tuple[K, K], tuple[K, K]]:
    e_path = (
        (INVERSE_DELTA, INVERSE_DELTA * D_TWO),
        (INVERSE_DELTA, INVERSE_DELTA * D_TWO),
    )
    alpha, beta = (A, A_INVERSE) if exponent == 1 else (A_INVERSE, A)
    return matrix_add(matrix_scale(alpha, IDENTITY), matrix_scale(beta, e_path))


def gauge_to_symmetric(matrix: tuple[tuple[K, K], tuple[K, K]], *, gauge_phi: K = PHI) -> tuple[tuple[K, K], tuple[K, K]]:
    gauge = ((ONE, ZERO), (ZERO, gauge_phi))
    inverse = ((ONE, ZERO), (ZERO, gauge_phi.inverse()))
    return matrix_multiply(matrix_multiply(gauge, matrix), inverse)


def serial_case(exponent: int) -> dict[str, Any]:
    block = symmetric_trivalent_braid(exponent)
    trace = block[0][0] + block[1][1]
    determinant = block[0][0] * block[1][1] - block[0][1] * block[1][0]
    squared = matrix_multiply(block, block)
    cayley_hamilton = matrix_add(
        matrix_add(squared, matrix_scale(ZERO - trace, block)),
        matrix_scale(determinant, IDENTITY),
    )
    current = IDENTITY
    commitments = []
    for depth in range(1, max(DEPTHS) + 1):
        current = matrix_multiply(current, block)
        if depth in DEPTHS:
            commitments.append(
                {"depth": depth, "matrix_commitment": matrix_commitment(current)}
            )
    return {
        "exponent": exponent,
        "symmetric_block_commitment": matrix_commitment(block),
        "path_block_commitment": matrix_commitment(path_temperley_lieb_braid(exponent)),
        "gauge_transformed_path_commitment": matrix_commitment(
            gauge_to_symmetric(path_temperley_lieb_braid(exponent))
        ),
        "exact_gauge_equivalence": block
        == gauge_to_symmetric(path_temperley_lieb_braid(exponent)),
        "cayley_hamilton_degree2_exact": cayley_hamilton
        == ((ZERO, ZERO), (ZERO, ZERO)),
        "distinct_braid_eigenvalues": r_matrix(exponent)[0][0]
        != r_matrix(exponent)[1][1],
        "serial_powers": commitments,
        "serial_resident_channel_cells": 2,
        "serial_compact_classical_state_cells": 2,
    }


@dataclass
class RankWork:
    retained_path_records: int = 0
    retained_path_label_cells: int = 0
    modular_vector_applications: int = 0
    modular_vector_cell_updates: int = 0
    basis_insertions: int = 0
    elimination_cell_updates: int = 0
    peak_basis_field_cells: int = 0
    peak_hankel_field_cells: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def prime_factors(value: int) -> tuple[int, ...]:
    factors = []
    residual = value
    candidate = 2
    while candidate * candidate <= residual:
        if residual % candidate == 0:
            factors.append(candidate)
            while residual % candidate == 0:
                residual //= candidate
        candidate += 1
    if residual > 1:
        factors.append(residual)
    return tuple(factors)


def primitive_root(prime: int) -> int:
    if prime < 2 or any(prime % divisor == 0 for divisor in range(2, math.isqrt(prime) + 1)):
        raise ValueError("M233 certificate modulus must be prime")
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    raise RuntimeError("M233 primitive-root search failed")


def root_of_order40(prime: int) -> int:
    if (prime - 1) % 40:
        raise ValueError("M233 certificate prime does not split zeta40")
    root = pow(primitive_root(prime), (prime - 1) // 40, prime)
    if pow(root, 40, prime) != 1 or any(
        pow(root, 40 // factor, prime) == 1 for factor in (2, 5)
    ):
        raise RuntimeError("M233 certificate root lacks exact order40")
    return root


def modular_dimensions(prime: int, root: int) -> tuple[int, ...]:
    delta = (pow(root, 2, prime) + pow(root, -2, prime)) % prime
    dimensions = [1, delta]
    for _ in range(2, m214.LABELS + 1):
        dimensions.append((delta * dimensions[-1] - dimensions[-2]) % prime)
    if dimensions[m214.LABELS] != 0 or any(value == 0 for value in dimensions[:m214.LABELS]):
        raise RuntimeError("M233 modular Jones-Wenzl/dimension law failed")
    return tuple(dimensions[:m214.LABELS])


def modular_action(
    vector: list[int],
    paths: list[tuple[int, ...]],
    path_index: dict[tuple[int, ...], int],
    generator: int,
    exponent: int,
    prime: int,
    root: int,
    dimensions: tuple[int, ...],
    work: RankWork,
    *,
    transpose: bool = False,
) -> list[int]:
    alpha = pow(root, 11 if exponent == 1 else -11, prime)
    beta = pow(root, -11 if exponent == 1 else 11, prime)
    output = [0] * len(vector)
    for index, path in enumerate(paths):
        left, middle, right = path[generator - 1 : generator + 2]
        if left != right:
            output[index] = alpha * vector[index] % prime
            continue
        alternatives = tuple(
            label for label in (left - 1, left + 1) if 0 <= label <= m214.LEVEL
        )
        if len(alternatives) == 2 and middle == alternatives[1]:
            continue
        inverse_dimension = pow(dimensions[left], -1, prime)
        if len(alternatives) == 1:
            factor = (
                alpha + beta * dimensions[middle] * inverse_dimension
            ) % prime
            output[index] = factor * vector[index] % prime
            continue
        peer_path = path[:generator] + (alternatives[1],) + path[generator + 1 :]
        peer = path_index[peer_path]
        m00 = (alpha + beta * dimensions[alternatives[0]] * inverse_dimension) % prime
        m01 = beta * dimensions[alternatives[1]] * inverse_dimension % prime
        m10 = beta * dimensions[alternatives[0]] * inverse_dimension % prime
        m11 = (alpha + beta * dimensions[alternatives[1]] * inverse_dimension) % prime
        first, second = vector[index], vector[peer]
        if transpose:
            output[index] = (m00 * first + m10 * second) % prime
            output[peer] = (m01 * first + m11 * second) % prime
        else:
            output[index] = (m00 * first + m01 * second) % prime
            output[peer] = (m10 * first + m11 * second) % prime
    work.modular_vector_applications += 1
    work.modular_vector_cell_updates += len(vector)
    return output


def insert_basis(
    basis: list[list[int]], pivots: list[int], candidate: list[int], prime: int, work: RankWork
) -> list[int] | None:
    vector = [value % prime for value in candidate]
    for row, pivot in zip(basis, pivots, strict=True):
        if vector[pivot]:
            factor = vector[pivot]
            vector = [
                (value - factor * basis_value) % prime
                for value, basis_value in zip(vector, row, strict=True)
            ]
            work.elimination_cell_updates += len(vector)
    pivot = next((index for index, value in enumerate(vector) if value), None)
    if pivot is None:
        return None
    inverse = pow(vector[pivot], -1, prime)
    vector = [value * inverse % prime for value in vector]
    for index, row in enumerate(basis):
        if row[pivot]:
            factor = row[pivot]
            basis[index] = [
                (value - factor * new_value) % prime
                for value, new_value in zip(row, vector, strict=True)
            ]
            work.elimination_cell_updates += len(vector)
    position = next(
        (index for index, old_pivot in enumerate(pivots) if old_pivot > pivot),
        len(pivots),
    )
    pivots.insert(position, pivot)
    basis.insert(position, vector)
    work.basis_insertions += 1
    work.peak_basis_field_cells = max(
        work.peak_basis_field_cells, len(basis) * len(vector)
    )
    return vector


def invariant_closure(
    source: list[int],
    paths: list[tuple[int, ...]],
    generators: tuple[tuple[int, int], ...],
    prime: int,
    root: int,
    dimensions: tuple[int, ...],
    work: RankWork,
    *,
    transpose: bool,
) -> list[list[int]]:
    path_index = {path: index for index, path in enumerate(paths)}
    basis: list[list[int]] = []
    pivots: list[int] = []
    queue = [insert_basis(basis, pivots, source, prime, work)]
    while queue:
        vector = queue.pop(0)
        if vector is None:
            continue
        for generator, exponent in generators:
            image = modular_action(
                vector,
                paths,
                path_index,
                generator,
                exponent,
                prime,
                root,
                dimensions,
                work,
                transpose=transpose,
            )
            inserted = insert_basis(basis, pivots, image, prime, work)
            if inserted is not None:
                queue.append(inserted)
    return basis


def matrix_rank(matrix: list[list[int]], prime: int, work: RankWork) -> int:
    basis: list[list[int]] = []
    pivots: list[int] = []
    for row in matrix:
        insert_basis(basis, pivots, row, prime, work)
    return len(basis)


def vectors_commitment(vectors: list[list[int]]) -> str:
    token = "|".join(",".join(map(str, vector)) for vector in vectors)
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def continuation_case(strands: int, prime: int) -> dict[str, Any]:
    topology = m214.FusionPathTopology.compile(strands)
    paths = [topology.unrank(index) for index in range(topology.dimension)]
    work = RankWork(
        retained_path_records=len(paths),
        retained_path_label_cells=sum(len(path) for path in paths),
    )
    root = root_of_order40(prime)
    dimensions = modular_dimensions(prime, root)
    generators = tuple(
        (generator, exponent)
        for generator in range(1, strands)
        for exponent in (-1, 1)
    )
    source = [0] * topology.dimension
    source[topology.rank(m214.vacuum_path(strands))] = 1
    reachable = invariant_closure(
        source, paths, generators, prime, root, dimensions, work, transpose=False
    )
    observable = invariant_closure(
        source, paths, generators, prime, root, dimensions, work, transpose=True
    )
    hankel = [
        [
            sum(left[index] * right[index] for index in range(topology.dimension))
            % prime
            for right in reachable
        ]
        for left in observable
    ]
    work.peak_hankel_field_cells = len(observable) * len(reachable)
    hankel_rank = matrix_rank(hankel, prime, work)
    cut = strands // 2
    same_label_pair = next((
        (left, right)
        for left in range(topology.dimension)
        for right in range(left + 1, topology.dimension)
        if paths[left][cut] == paths[right][cut]
    ), None)
    distinguishing_observable = (
        next(
            index
            for index, row in enumerate(observable)
            if row[same_label_pair[0]] != row[same_label_pair[1]]
        )
        if same_label_pair is not None
        else None
    )
    return {
        "strands": strands,
        "fusion_path_dimension": topology.dimension,
        "prime": prime,
        "root_of_order40": root,
        "reachable_rank": len(reachable),
        "observable_rank": len(observable),
        "continuation_hankel_rank": hankel_rank,
        "all_ranks_full": len(reachable) == len(observable) == hankel_rank == topology.dimension,
        "same_midcut_label_overmerge_pair": list(same_label_pair) if same_label_pair is not None else None,
        "same_midcut_label": paths[same_label_pair[0]][cut] if same_label_pair is not None else None,
        "distinguishing_observable_basis_index": distinguishing_observable,
        "reachable_basis_commitment": vectors_commitment(reachable),
        "observable_basis_commitment": vectors_commitment(observable),
        "hankel_commitment": vectors_commitment(hankel),
        "verification_work": work.as_dict(),
    }


def semantic_transaction_case(strands: int) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = m214.FusionPathTopology.compile(strands)
    source = m214.source_state(topology)
    carrier = m214.Carrier(m214.OpenFusionPathPort(topology, source.copy()))
    primary_program = m214.BraidProgram(strands, PRIMARY_ROUNDS, 0)
    reuse_program = m214.BraidProgram(strands, REUSE_ROUNDS, 1)
    primary, primary_work = m214.transaction(carrier, source, primary_program)
    reuse, reuse_work = m214.transaction(carrier, source, reuse_program)
    fresh = m214.Carrier(m214.OpenFusionPathPort(topology, source.copy()))
    fresh_reuse, _ = m214.transaction(fresh, source, reuse_program)
    semantic = {
        "strands": strands,
        "fusion_path_dimension": topology.dimension,
        "primary_boundary_commitment": primary["boundary_commitment"],
        "primary_forward_state_commitment": primary["forward_state_commitment"],
        "primary_restoration_error_field_cells": primary["restoration_error_field_cells"],
        "primary_same_coefficient_backing": primary["same_coefficient_backing"],
        "primary_canonical_post_restoration_state_exact": primary["canonical_post_restoration_state_exact"],
        "primary_missing_inverse_error_nonzero": primary["missing_inverse_error_cells_and_cursor"] > 0,
        "reuse_boundary_commitment": reuse["boundary_commitment"],
        "reuse_forward_state_commitment": reuse["forward_state_commitment"],
        "reuse_restoration_error_field_cells": reuse["restoration_error_field_cells"],
        "reuse_same_coefficient_backing": reuse["same_coefficient_backing"],
        "reuse_canonical_post_restoration_state_exact": reuse["canonical_post_restoration_state_exact"],
        "fresh_reuse_boundary_commitment": fresh_reuse["boundary_commitment"],
        "fresh_reuse_state_commitment": fresh_reuse["forward_state_commitment"],
        "fresh_same_coefficient_backing": fresh_reuse["same_coefficient_backing"],
        "fresh_canonical_post_restoration_state_exact": fresh_reuse["canonical_post_restoration_state_exact"],
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"],
        "fresh_restored_reuse_state_agreement": reuse["forward_state_commitment"] == fresh_reuse["forward_state_commitment"],
        "restoration_generation_after_reuse": carrier.restoration_generation,
        "fresh_restoration_generation": fresh.restoration_generation,
        "baseline_reload_used": False,
    }
    resource = {
        "strands": strands,
        "accepted_carrier_field_cells": topology.dimension,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "accepted_carrier_plus_retained_boundary_field_cells": topology.dimension + 1,
        "primary_retained_boundary_payload_bits": primary["boundary_payload_bits"],
        "reuse_retained_boundary_payload_bits": reuse["boundary_payload_bits"],
        "public_topology_integer_cells": topology.integer_cells,
        "public_topology_payload_bits": topology.integer_payload,
        "primary_work": primary_work.as_dict(),
        "reuse_work": reuse_work.as_dict(),
        "retained_inverse_history_entries": 0,
    }
    return semantic, resource


def exact_boundary_with_mutation(strands: int, *, flip_first_exponent: bool) -> K:
    topology = m214.FusionPathTopology.compile(strands)
    coefficients = m214.source_state(topology)
    program = m214.BraidProgram(strands, PRIMARY_ROUNDS, 0)
    work = m214.Work()
    for index in range(program.steps):
        operation = program.operation(index)
        if index == 0 and flip_first_exponent:
            operation = m214.BraidOperation(operation.generator, -operation.exponent)
        m214.apply_braid(coefficients, topology, operation, work)
    return coefficients[topology.rank(m214.vacuum_path(strands))]


def modular_operator_controls(strands: int, prime: int) -> dict[str, bool]:
    topology = m214.FusionPathTopology.compile(strands)
    paths = [topology.unrank(index) for index in range(topology.dimension)]
    path_index = {path: index for index, path in enumerate(paths)}
    root = root_of_order40(prime)
    dimensions = modular_dimensions(prime, root)
    work = RankWork()

    def apply(vector: list[int], word: tuple[tuple[int, int], ...]) -> list[int]:
        for generator, exponent in word:
            vector = modular_action(
                vector, paths, path_index, generator, exponent, prime, root, dimensions, work
            )
        return vector

    basis = [[int(index == column) for index in range(topology.dimension)] for column in range(topology.dimension)]
    inverse_ok = all(
        apply(vector.copy(), ((generator, 1), (generator, -1))) == vector
        for generator in range(1, strands)
        for vector in basis
    )
    yang_baxter_ok = all(
        apply(vector.copy(), ((generator, 1), (generator + 1, 1), (generator, 1)))
        == apply(vector.copy(), ((generator + 1, 1), (generator, 1), (generator + 1, 1)))
        for generator in range(1, strands - 1)
        for vector in basis
    )
    distant_ok = all(
        apply(vector.copy(), ((left, 1), (right, 1)))
        == apply(vector.copy(), ((right, 1), (left, 1)))
        for left in range(1, strands)
        for right in range(left + 2, strands)
        for vector in basis
    )
    return {
        "all_generators_have_exact_modular_inverse": inverse_ok,
        "all_adjacent_generators_satisfy_yang_baxter": yang_baxter_ok,
        "all_distant_generators_commute": distant_ok,
    }


def custody_controls_growing() -> dict[str, bool]:
    strands = 10
    topology = m214.FusionPathTopology.compile(strands)
    source = m214.source_state(topology)
    program = m214.BraidProgram(strands, PRIMARY_ROUNDS, 0)
    work = m214.Work()
    premature = null_carrier = wrong_program = False
    port = m214.OpenFusionPathPort(topology, source.copy())
    port.lease(233001, 1, program, work)
    port.forward(233001, program, 0, work)
    try:
        port.project_final(233001, program, work)
    except PermissionError:
        premature = True
    try:
        m214.OpenFusionPathPort(topology, []).lease(233002, 1, program, m214.Work())
    except ValueError:
        null_carrier = True
    try:
        port.require(233001, m214.BraidProgram(strands, PRIMARY_ROUNDS + 1, 0), work)
    except ValueError:
        wrong_program = True
    return {
        "premature_projection_rejected": premature,
        "null_carrier_rejected": null_carrier,
        "wrong_public_program_rejected": wrong_program,
        "public_topology_compiler_reads_no_boundary_answer": True,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: su2_level8_trivalent_extension_dichotomy.py REFERENCE_JSON M232_RESULT"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    m232_path = Path(sys.argv[2]).resolve()
    for path in (reference_path, m232_path):
        if str(path).startswith(("/dev/shm/", "/run/shm/")):
            raise ValueError("RAM-backed M233 input forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_trivalent_continuation_rank_reference.v1":
        raise RuntimeError("M233 reference schema mismatch")
    m232 = json.loads(m232_path.read_text())
    if m232.get("result") != "PASS_BOUNDED_EXACT_TRIVALENT_SHARED_CHANNEL_F_MOVE_BRAID_CONTRACTION_WITH_IDENTICAL_SMALLER_CLASSICAL_FACTOR_GRAPH":
        raise RuntimeError("M233 predecessor M232 mismatch")

    transaction_pairs = [semantic_transaction_case(strands) for strands in GROWING_STRANDS]
    transactions = [pair[0] for pair in transaction_pairs]
    resources = [pair[1] for pair in transaction_pairs]
    certificates = [
        continuation_case(strands, prime)
        for strands in GROWING_STRANDS
        for prime in CERTIFICATE_PRIMES
    ]
    rank_law = [
        {
            "strands": strands,
            "fusion_path_dimension": dimension,
            "reachable_rank_both_primes": [
                case["reachable_rank"] for case in certificates if case["strands"] == strands
            ],
            "observable_rank_both_primes": [
                case["observable_rank"] for case in certificates if case["strands"] == strands
            ],
            "continuation_hankel_rank_both_primes": [
                case["continuation_hankel_rank"] for case in certificates if case["strands"] == strands
            ],
            "full_minimal_qzeta40_linear_continuation_dimension_certified": all(
                case["all_ranks_full"] for case in certificates if case["strands"] == strands
            ),
        }
        for strands, dimension in zip(GROWING_STRANDS, GROWING_DIMENSIONS, strict=True)
    ]
    local_cases = [serial_case(exponent) for exponent in (-1, 1)]
    current_controls = {
        "n4_m232_gauge_equivalence_both_signs": all(case["exact_gauge_equivalence"] for case in local_cases),
        "n4_local_braid_inverse_exact": matrix_multiply(
            symmetric_trivalent_braid(1), symmetric_trivalent_braid(-1)
        ) == IDENTITY,
        "all_declared_continuation_ranks_full_at_both_production_primes": all(
            case["all_ranks_full"] for case in certificates
        ),
        "n8_and_n10_exceed_nine_charge_labels": all(
            dimension > m214.LABELS for dimension in GROWING_DIMENSIONS[2:]
        ),
        "same_charge_label_overmerge_distinguished_at_n8_and_n10": all(
            case["distinguishing_observable_basis_index"] is not None
            for case in certificates if case["strands"] >= 8
        ),
        "flipping_first_public_braid_exponent_changes_every_selected_boundary": all(
            exact_boundary_with_mutation(strands, flip_first_exponent=False)
            != exact_boundary_with_mutation(strands, flip_first_exponent=True)
            for strands in GROWING_STRANDS
        ),
        **modular_operator_controls(10, CERTIFICATE_PRIMES[0]),
        **custody_controls_growing(),
    }
    if transactions != reference.get("transactions"):
        raise RuntimeError("M233 independent exact transaction parity failed")
    if [case["fusion_path_dimension"] for case in reference.get("rank_cases", [])] != [
        dimension for dimension in GROWING_DIMENSIONS for _prime in (641, 881)
    ]:
        raise RuntimeError("M233 independent rank dimensions mismatch")
    reference_rank_law = reference.get("rank_law")
    if not isinstance(reference_rank_law, list) or [
        case["continuation_hankel_rank_both_primes"] for case in reference_rank_law
    ] != [[dimension, dimension] for dimension in GROWING_DIMENSIONS]:
        raise RuntimeError("M233 independent continuation-rank parity failed")
    if not all(current_controls.values()):
        raise RuntimeError("M233 production control failed")
    if not all(reference.get("controls", {}).values()):
        raise RuntimeError("M233 independent reference control failed")

    result = {
        "schema": "cat_cas.su2_level8_trivalent_continuation_rank.v1",
        "result": "PASS_EXACT_GROWING_SU2_LEVEL8_TRIVALENT_CONTINUATION_RANKS_REJECT_FIXED_TWO_AND_NINE_STATE_QUOTIENTS_THROUGH_N10",
        "claim": "EXACT_DUAL_SPLIT_PRIME_REACHABLE_OBSERVABLE_AND_HANKEL_CERTIFICATES_SHOW_THE_FIXED_VACUUM_PLAT_BOUNDARY_LANGUAGE_FOR_ALL_ADJACENT_SU2_LEVEL8_FUNDAMENTAL_BRAID_GENERATORS_REQUIRES_FULL_QZETA40_LINEAR_CONTINUATION_DIMENSIONS2_5_14_42_AT_STRANDS4_6_8_10_SO_THE_M232_FIXED_TWO_CHANNEL_EXTENSION_AND_ANY_FIXED_NINE_CHARGE_LINEAR_QUOTIENT_FAIL_WITHIN_THE_DECLARED_CASES_BY_N8_WHILE_NO_ALL_N_UNBOUNDED_RANK_THEOREM_IS_CLAIMED_WITH_FINAL_ONLY_BOUNDARY_EXACT_SAME_BACKING_RESTORATION_REUSE_AND_THE_IDENTICAL_SPARSE_CLASSICAL_FUSION_PATH_RECURRENCE",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "QZETA40_SU2_LEVEL8_FUNDAMENTAL_VACUUM_FUSION_PATH_SECTORS_N4_6_8_10_ALL_ADJACENT_GENERATORS_AND_INVERSES_REACHABLE_OBSERVABLE_CONTINUATION_RANKS2_5_14_42_AT_PRODUCTION_SPLIT_PRIMES241_401_AND_INDEPENDENT_PRIMES641_881_SELECTED_FOUR_SWEEP_PRIMARY_THREE_SWEEP_REUSE_DIRECT_PROCESS_ONLY",
        "rank_law": rank_law,
        "rank_certificates": certificates,
        "transactions": transactions,
        "controls": current_controls,
        "mechanism_law": {
            "phase_primitive_state": "QZETA40_COEFFICIENTS_ON_PUBLIC_SU2_LEVEL8_VACUUM_FUSION_PATHS",
            "multiple_overlapping_shared_channels": True,
            "all_adjacent_noncommuting_braid_consumers": True,
            "accepted_generator_action_is_streamed_local_one_or_two_cell_blocks": True,
            "accepted_dense_operator_materialized": False,
            "accepted_fusion_relation_table_materialized": False,
            "accepted_assignment_expansion_materialized": False,
            "global_fusion_path_coefficient_basis_materialized": True,
            "final_vacuum_plat_scalar_only": True,
            "forward_state_only_one_way_committed": True,
            "direct_process_logical_custody_only": True,
            "rank_verifier_materializes_paths_bases_and_hankel_matrix": True,
        },
        "m232_local_bridge": {
            "cases": local_cases,
            "n4_is_exactly_the_two_channel_sector": True,
            "m232_block_alone_implies_global_rank_growth": False,
            "new_m233_continuation_certificates_are_direct_not_inherited": True,
        },
        "resource_law": {
            "accepted_transactions": resources,
            "accepted_path_dimensions": list(GROWING_DIMENSIONS),
            "accepted_retained_inverse_history_entries": 0,
            "rank_verifier_paths_modular_bases_and_hankel_are_verifier_only": True,
            "rank_verifier_not_accepted_runtime_output": True,
            "verifier_peak_basis_and_hankel_cells_are_component_local_not_combined": True,
            "whole_transaction_live_field_and_payload_accounting_complete": False,
            "maximum_verifier_hankel_cells": max(
                case["verification_work"]["peak_hankel_field_cells"] for case in certificates
            ),
            "maximum_verifier_basis_cells": max(
                case["verification_work"]["peak_basis_field_cells"] for case in certificates
            ),
            "strongest_implemented_matched_classical": "IDENTICAL_EXACT_QZETA40_SPARSE_FUSION_PATH_TRANSFER_WITH_PUBLIC_A9_RANK_UNRANK_AND_FULL_MINIMAL_LINEAR_CONTINUATION_DIMENSION_FOR_THE_DECLARED_PLAT_LANGUAGE",
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "excluded_not_zero": "RANK_VERIFIER_PATH_LISTS_MODULAR_BASES_HANKEL_MATRICES_PYTHON_OBJECT_HEADERS_ALLOCATOR_INTERPRETER_HASH_BYTE_TRAFFIC_SERIALIZATION_WALL_TIME_RSS_AND_TREE_DECOMPOSITION_COMPILER_COST",
        },
        "obstruction": {
            "fixed_two_channel_extension_rejected_through_n10": True,
            "fixed_nine_charge_label_linear_quotient_rejected_by_n8": True,
            "any_fixed_qzeta40_linear_rank_at_most41_rejected_through_n10": True,
            "uniform_fixed_rank_linear_continuation_quotient_for_unbounded_family_rejected": False,
            "all_n_continuation_rank_theorem_established": False,
            "nonlinear_or_non_qzeta40_quotient_excluded": False,
            "route_disposition": "RETIRE_THE_FIXED_TWO_AND_NINE_STATE_RECOUPLING_COMPRESSION_PROPOSALS_AND_CHANGE_PHASE_LAW",
        },
        "matched_classical": {
            "same_sparse_path_state_cells": True,
            "same_local_block_work": True,
            "smaller_qzeta40_linear_continuation_quotient_exists_for_declared_language": False,
            "treewidth_optimized_scalar_word_baseline_implemented": False,
            "computational_advantage": False,
            "distinct_phase_resource": False,
        },
        "separate_reference": {
            "certificate_primes": reference.get("certificate_primes"),
            "imports_m233_production": reference.get("imports_m233_production"),
            "imports_m232_production": reference.get("imports_m232_production"),
            "imports_m214_production": reference.get("imports_m214_production"),
            "uses_independent_cyclotomic_polynomial_oracle": reference.get("uses_independent_cyclotomic_polynomial_oracle"),
            "exact_transaction_parity": True,
            "continuation_rank_parity_at_distinct_primes": True,
        },
        "claim_limits": {
            "finite_declared_strand_counts_only": True,
            "materialized_fusion_path_basis_expansion": True,
            "fixed_qzeta40_linear_quotient_rank_at_most41_through_n10": False,
            "unbounded_continuation_rank_growth_proved": False,
            "nonlinear_quotient_excluded": False,
            "general_tensor_network_closure": False,
            "machine_enforced_catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m214_production_sha256": sha256_file(here / "su2_level8_fusion_path_braid_phase_relation.py"),
            "m232_result_sha256": sha256_file(m232_path),
            "m233_production_sha256": sha256_file(Path(__file__).resolve()),
            "m233_reference_code_sha256": sha256_file(here / "su2_level8_trivalent_extension_dichotomy_separate_reference.py"),
            "m233_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
