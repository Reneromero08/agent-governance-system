#!/usr/bin/env python3
"""Exact scalar-Krylov diagnostic for the M197 Rotor-6 boundary.

This package asks whether repeated application of the exact public M197
``step=0, tag=0`` word has a compact final-boundary recurrence over F103.  It
uses the M196 retained sparse plan only as explicitly reported diagnostic
state.  That plan is not reintroduced into the accepted zero-plan M197 phase
path.  The strongest matched classical method is the scalar recurrence found
here, including its full initialization cost and public coefficients.

This is a bounded no-go diagnostic.  It neither restores a carrier nor claims
CATVM custody, a distinct phase resource, computational advantage, or a Small
Wall crossing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse

import growing_rotor_dihedral_third_order_signature_closure as retained_plan
import growing_rotor_dihedral_third_order_signature_streamed_closure as streamed


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
STEP = 0
PROGRAM_TAG = 0
SOURCE_FAMILY = 0
TRAINING_TERMS = 2 * 2277
HOLDOUT_TERMS = 64
EXPECTED_M197_BOUNDARY = 83
EXPECTED_M197_FORWARD_COMMITMENT = (
    "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
)


def integer_commitment(values: list[int] | tuple[int, ...] | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)},".encode())
    return digest.hexdigest()


def topology_commitment(
    signatures: tuple[tuple[int, ...], ...],
    representatives: tuple[tuple[int, ...], ...],
    boundary_weights: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    for signature, representative, weight in zip(
        signatures, representatives, boundary_weights, strict=True
    ):
        digest.update(
            (
                ",".join(map(str, signature))
                + "|"
                + ",".join(map(str, representative))
                + f"|{weight};"
            ).encode()
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactOperator:
    matrix: sparse.csr_matrix
    diagonal: np.ndarray
    source_plan_entries: int
    csr_commitment: str


def compile_exact_operator(
    topology: retained_plan.RefinedTopology,
) -> ExactOperator:
    expected_entries = topology.prior_materialized_plan_nonzeros
    rows = np.empty(expected_entries, dtype=np.int32)
    columns = np.empty(expected_entries, dtype=np.int32)
    coefficients = np.empty(expected_entries, dtype=np.int64)
    cursor = 0
    for shift, basis in enumerate(topology.shift_rows, 1):
        public_weight = retained_plan.predecessor.public_scattering_integer(
            shift, STEP, PROGRAM_TAG
        )
        for target, row in enumerate(basis):
            for source, coefficient in row:
                rows[cursor] = target
                columns[cursor] = source
                coefficients[cursor] = coefficient * public_weight
                cursor += 1
    if cursor != expected_entries:
        raise RuntimeError("retained diagnostic plan entry count changed")
    dimension = len(topology.signatures)
    matrix = sparse.coo_matrix(
        (coefficients, (rows, columns)),
        shape=(dimension, dimension),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    diagonal = np.asarray(
        [
            pow(
                ROOT,
                retained_plan.predecessor.phase_exponent(
                    signature[: retained_plan.predecessor.PAIR_CHANNELS],
                    STEP,
                    PROGRAM_TAG,
                ),
                PRIME,
            )
            for signature in topology.signatures
        ],
        dtype=np.int64,
    )
    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(integer_commitment(values).encode())
    return ExactOperator(
        matrix=matrix,
        diagonal=diagonal,
        source_plan_entries=cursor,
        csr_commitment=digest.hexdigest(),
    )


def apply_operator(state: np.ndarray, operator: ExactOperator) -> np.ndarray:
    if state.ndim != 1 or state.shape[0] != operator.matrix.shape[0]:
        raise ValueError("null or malformed Krylov diagnostic state")
    phased = state * operator.diagonal % PRIME
    return np.asarray(operator.matrix.dot(phased) % PRIME, dtype=np.int64)


@dataclass(frozen=True)
class BMResult:
    degree: int
    coefficients: tuple[int, ...]
    peak_connection_field_cells: int


def berlekamp_massey(sequence: list[int]) -> BMResult:
    connection = [1]
    backup = [1]
    degree = 0
    delay = 1
    backup_discrepancy = 1
    peak_cells = 2
    for position, sample in enumerate(sequence):
        discrepancy = sample
        for index in range(1, degree + 1):
            discrepancy = (
                discrepancy + connection[index] * sequence[position - index]
            ) % PRIME
        if discrepancy == 0:
            delay += 1
            continue
        previous = connection.copy()
        scale = discrepancy * pow(backup_discrepancy, PRIME - 2, PRIME) % PRIME
        required = len(backup) + delay
        if len(connection) < required:
            connection.extend([0] * (required - len(connection)))
        for index, value in enumerate(backup):
            connection[index + delay] = (
                connection[index + delay] - scale * value
            ) % PRIME
        if 2 * degree <= position:
            degree = position + 1 - degree
            backup = previous
            backup_discrepancy = discrepancy
            delay = 1
        else:
            delay += 1
        peak_cells = max(peak_cells, len(connection) + len(backup))
    return BMResult(degree, tuple(connection), peak_cells)


def recurrence_violations(
    sequence: list[int],
    offset: int,
    training_terms: int,
    result: BMResult,
) -> tuple[int, int]:
    shifted = sequence[offset:]
    training = 0
    holdout = 0
    for position in range(result.degree, len(shifted)):
        discrepancy = shifted[position]
        for index in range(1, result.degree + 1):
            discrepancy += result.coefficients[index] * shifted[position - index]
        if discrepancy % PRIME:
            if position < training_terms:
                training += 1
            else:
                holdout += 1
    return training, holdout


def scalar_sequence(
    source: np.ndarray,
    probe: np.ndarray,
    operator: ExactOperator,
    terms: int,
) -> tuple[list[int], np.ndarray]:
    state = source.copy()
    sequence: list[int] = []
    first_forward = np.empty(0, dtype=np.int64)
    for index in range(terms):
        sequence.append(int(np.dot(probe, state) % PRIME))
        state = apply_operator(state, operator)
        if index == 0:
            first_forward = state.copy()
    return sequence, first_forward


def main() -> None:
    topology = retained_plan.compile_topology(ROTORS, PRIME, ROOT)
    if (
        len(topology.signatures) != 2277
        or topology.bracelet_cells != 2277
        or topology.necklace_cells != 4389
        or topology.occupation_histograms != 74613
    ):
        raise RuntimeError("declared Rotor-6 refined topology changed")
    operator = compile_exact_operator(topology)
    source = np.asarray(
        streamed.source_state(topology.signatures, PRIME, SOURCE_FAMILY),
        dtype=np.int64,
    )
    probe = np.asarray(topology.boundary_weights, dtype=np.int64)
    sequence, first_forward = scalar_sequence(
        source,
        probe,
        operator,
        TRAINING_TERMS + HOLDOUT_TERMS + 1,
    )
    first_boundary = sequence[1]
    first_commitment = streamed.commitment(first_forward.tolist())
    if (
        first_boundary != EXPECTED_M197_BOUNDARY
        or first_commitment != EXPECTED_M197_FORWARD_COMMITMENT
    ):
        raise RuntimeError("diagnostic operator differs from M197 first word")

    full = berlekamp_massey(sequence[:TRAINING_TERMS])
    shifted = berlekamp_massey(sequence[1 : 1 + TRAINING_TERMS])
    full_training, full_holdout = recurrence_violations(
        sequence, 0, TRAINING_TERMS, full
    )
    shifted_training, shifted_holdout = recurrence_violations(
        sequence, 1, TRAINING_TERMS, shifted
    )
    if (
        full.degree != 2261
        or shifted.degree != 2260
        or full_training
        or full_holdout
        or shifted_training
        or shifted_holdout
        or shifted.coefficients[-1] == 0
    ):
        raise RuntimeError("Rotor-6 scalar recurrence certificate changed")

    scattered_first = np.asarray(operator.matrix.dot(source) % PRIME, dtype=np.int64)
    reordered_first = scattered_first * operator.diagonal % PRIME
    reordered_boundary = int(np.dot(probe, reordered_first) % PRIME)
    inverse_root = pow(ROOT, PRIME - 2, PRIME)
    wrong_diagonal = np.asarray(
        [
            pow(
                inverse_root,
                retained_plan.predecessor.phase_exponent(
                    signature[: retained_plan.predecessor.PAIR_CHANNELS],
                    STEP,
                    PROGRAM_TAG,
                ),
                PRIME,
            )
            for signature in topology.signatures
        ],
        dtype=np.int64,
    )
    wrong_root_boundary = int(
        np.dot(
            probe,
            np.asarray(operator.matrix.dot(source * wrong_diagonal % PRIME) % PRIME),
        )
        % PRIME
    )
    undersampled = berlekamp_massey(sequence[:512])
    undersampled_training, undersampled_holdout = recurrence_violations(
        sequence, 0, 512, undersampled
    )
    null_rejected = False
    try:
        apply_operator(np.empty(0, dtype=np.int64), operator)
    except ValueError:
        null_rejected = True
    if (
        reordered_boundary == first_boundary
        or wrong_root_boundary == first_boundary
        or undersampled_holdout == 0
        or not null_rejected
    ):
        raise RuntimeError("Krylov diagnostic control failed")

    dimension = len(topology.signatures)
    shifted_nonzero = sum(value != 0 for value in shifted.coefficients)
    result = {
        "claim_candidate": "EXACT_F103_ROTOR6_TWO_TRIANGLE_REFINED_FINAL_BOUNDARY_SEQUENCE_HAS_DEGREE2261_FROM_K0_AND2260_AFTER_ONE_PUBLIC_WORD_ON_THE2277_CELL_PHASE_QUOTIENT_SO_THE_STRONGEST_SHIFTED_SCALAR_RECURRENCE_SAVES_ONLY17_DYNAMIC_CELLS_REQUIRES2244_NONZERO_COEFFICIENTS_AND_DOES_NOT_RESTORE_OR_REPLACE_THE_PHASE_CARRIER",
        "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_TWO_TRIANGLE_REFINED_SIGNATURE_ROTOR6_F103_ROOT72_REPEATED_STEP0_TAG0_SOURCE_FAMILY0_PUBLIC_BOUNDARY_DIRECT_PROCESS_DIAGNOSTIC_ONLY",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "result": "PASS_DIAGNOSTIC_NO_COMPACT_RECURRENCE",
        "declared_program": {
            "prime": PRIME,
            "seventeenth_root": ROOT,
            "rotors": ROTORS,
            "step": STEP,
            "program_tag": PROGRAM_TAG,
            "source_family": SOURCE_FAMILY,
            "operator_order": "SCATTERING_AFTER_DIAGONAL",
        },
        "topology": {
            "occupation_histograms": topology.occupation_histograms,
            "necklace_cells": topology.necklace_cells,
            "bracelet_and_refined_signature_cells": dimension,
            "signature_integer_cells": sum(map(len, topology.signatures)),
            "representative_integer_cells": sum(map(len, topology.representatives)),
            "boundary_weight_field_cells": len(topology.boundary_weights),
            "topology_commitment": topology_commitment(
                topology.signatures,
                topology.representatives,
                topology.boundary_weights,
            ),
            "public_topology_compile_reads_final_answers": False,
        },
        "exact_boundary_recurrence": {
            "carrier_dimension": dimension,
            "training_terms": TRAINING_TERMS,
            "holdout_terms_per_shifted_test": HOLDOUT_TERMS,
            "k0_sequence_degree": full.degree,
            "k0_dimension_deficit": dimension - full.degree,
            "k0_connection_slots": len(full.coefficients),
            "k0_connection_nonzero_coefficients": sum(
                value != 0 for value in full.coefficients
            ),
            "k0_last_coefficient": full.coefficients[-1],
            "k1_shifted_sequence_degree": shifted.degree,
            "k1_shifted_dynamic_cell_saving": dimension - shifted.degree,
            "k1_shifted_connection_slots": len(shifted.coefficients),
            "k1_shifted_connection_nonzero_coefficients": shifted_nonzero,
            "k1_shifted_last_coefficient": shifted.coefficients[-1],
            "k0_training_violations": full_training,
            "k0_holdout_violations": full_holdout,
            "k1_training_violations": shifted_training,
            "k1_holdout_violations": shifted_holdout,
            "sequence_commitment": integer_commitment(sequence),
            "k0_connection_commitment": integer_commitment(full.coefficients),
            "k1_connection_commitment": integer_commitment(shifted.coefficients),
            "sequence_prefix": sequence[:8],
            "first_public_word_boundary": first_boundary,
            "first_public_word_state_commitment": first_commitment,
            "one_full_state_word_required_before_shifted_recurrence": True,
            "recurrence_does_not_reconstruct_internal_phase_state": True,
            "fixed_rank_or_depth_independent_recurrence_established": False,
        },
        "controls": {
            "reordered_first_boundary": reordered_boundary,
            "reordered_differs": reordered_boundary != first_boundary,
            "inverse_root_first_boundary": wrong_root_boundary,
            "inverse_root_differs": wrong_root_boundary != first_boundary,
            "undersampled_term_count": 512,
            "undersampled_apparent_degree": undersampled.degree,
            "undersampled_training_violations": undersampled_training,
            "undersampled_out_of_sample_violations": undersampled_holdout,
            "null_state_rejected": null_rejected,
        },
        "resource_law": {
            "accepted_m197_retained_shift_plans": 0,
            "accepted_m197_retained_plan_nonzeros": 0,
            "diagnostic_source_shift_plan_entries": operator.source_plan_entries,
            "diagnostic_aggregated_csr_nonzeros": int(operator.matrix.nnz),
            "diagnostic_aggregated_csr_integer_cells": (
                2 * int(operator.matrix.nnz) + len(operator.matrix.indptr)
            ),
            "diagnostic_csr_commitment": operator.csr_commitment,
            "diagnostic_diagonal_field_cells": len(operator.diagonal),
            "diagnostic_probe_field_cells": len(probe),
            "diagnostic_state_and_next_field_cells": 2 * dimension,
            "diagnostic_retained_sequence_field_cells": len(sequence),
            "diagnostic_peak_bm_connection_field_cells": max(
                full.peak_connection_field_cells,
                shifted.peak_connection_field_cells,
            ),
            "shifted_recurrence_initialization_operator_applications": shifted.degree,
            "shifted_recurrence_dynamic_field_cells": shifted.degree,
            "shifted_recurrence_public_coefficient_slots": len(
                shifted.coefficients
            ),
            "shifted_recurrence_public_nonzero_coefficients": shifted_nonzero,
            "m197_rematerialized_moves_per_scattering": (
                topology.streamed_mode_pair_shift_terms_per_scattering
            ),
            "m197_triangle_monomial_evaluations_per_scattering": 24767280,
            "diagnostic_csr_is_verification_only": True,
            "diagnostic_python_containers_allocator_scipy_native_bytes_timing_and_whole_process_peaks_excluded": True,
        },
        "predecessor_restoration_evidence": {
            "milestone": "M197",
            "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "primary_boundary": EXPECTED_M197_BOUNDARY,
            "primary_forward_commitment": EXPECTED_M197_FORWARD_COMMITMENT,
            "same_backing_restoration_and_reuse_previously_established": True,
            "this_diagnostic_does_not_reexecute_or_extend_that_claim": True,
        },
        "matched_classical_recurrence": "ONE_M197_FULL_STATE_INITIAL_WORD_THEN_THE_IDENTICAL_EXACT2260_SCALAR_COMPANION_RECURRENCE_WITH2244_NONZERO_PUBLIC_COEFFICIENTS",
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catvm_custody": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
