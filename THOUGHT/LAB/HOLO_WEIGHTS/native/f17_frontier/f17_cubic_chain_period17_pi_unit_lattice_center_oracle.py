#!/usr/bin/env python3
"""Separate parity oracle for the bounded unit-lattice-center diagnostic.

This oracle imports only sealed independent cyclotomic recurrence kernels.  It
does not import the production lattice-center module.  It independently
constructs the fixed-precision embedding table, proposes a seven-dimensional
center through normal equations rather than the production least-squares
call, and permits a carrier update only after an exact integer trace-energy
comparison.  Exact boundaries, payload/width tuples, proposal commitments,
inverse rematerialization, and restored reuse are compared with production.

The fixed-precision proposal remains non-certified.  Agreement does not turn
it into an exact closest-vector method or establish a resource advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import numpy as np

import f17_cubic_chain_period17_pi_unit_embedding_balance_oracle as prior
import f17_cubic_chain_period17_unit_height_reduction_oracle as ring


PRIME = 17
DIMENSION = 16
UNIT_RANK = 7
EXPECTED_PERIODS = (1, 64)
PRECISION_BITS = 65536
EMBEDDINGS = tuple(range(1, 9))
GENERATOR_INDICES = (2, 3, 4, 5, 6, 7, 8)
COSINE_CELLS = 9
LOG_MATRIX_CELLS = 8 * UNIT_RANK
SOLVE_SCRATCH_CELLS = 8 + 8 + UNIT_RANK

RingElement = tuple[int, ...]
RingVector = list[RingElement]


with mp.workprec(PRECISION_BITS):
    COSINES = tuple(
        mp.cos(mp.mpf(2) * mp.pi * index / PRIME)
        for index in range(COSINE_CELLS)
    )


LOG_MATRIX = np.array(
    [
        [
            math.log(
                (
                    math.sin(math.pi * generator * embedding / PRIME)
                    / math.sin(math.pi * embedding / PRIME)
                )
                ** 2
            )
            for generator in GENERATOR_INDICES
        ]
        for embedding in EMBEDDINGS
    ],
    dtype=np.float64,
)
NORMAL_MATRIX = LOG_MATRIX.T @ LOG_MATRIX


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass
class LatticeStats(prior.OracleStats):
    lattice_center_proposals: int = 0
    lattice_center_proposals_accepted: int = 0
    lattice_center_proposals_rejected: int = 0
    fixed_precision_nonpositive_embedding_rejections: int = 0
    embedding_dot_multiply_accumulations: int = 0
    embedding_log_evaluations: int = 0
    embedding_linear_solves: int = 0
    proposal_commitment_updates: int = 0
    proposal_transcript_commitment: bytes = bytes(32)
    maximum_norm_element_payload_bits: int = 0
    maximum_proposal_scale_payload_bits: int = 0
    maximum_proposal_norm_factor_payload_bits: int = 0
    maximum_trial_norm_element_payload_bits: int = 0
    maximum_accepted_proposal_vector_payload_bits: int = 0
    maximum_lattice_move_abs: int = 0
    minimum_embedding_log2_floor: int = 0
    maximum_embedding_log2_ceiling: int = 0
    maximum_linear_solve_residual: float = 0.0


def cosine_index(exponent: int) -> int:
    index = exponent % PRIME
    return index if index <= PRIME // 2 else PRIME - index


def propose_center(
    norm: RingElement,
    stats: LatticeStats,
) -> list[int] | None:
    stats.lattice_center_proposals += 1
    logs: list[float] = []
    with mp.workprec(PRECISION_BITS):
        for embedding in EMBEDDINGS:
            value = mp.mpf(0)
            for exponent, coefficient in enumerate(norm):
                value += (
                    mp.mpf(coefficient)
                    * COSINES[cosine_index(embedding * exponent)]
                )
                stats.embedding_dot_multiply_accumulations += 1
            if value <= 0:
                stats.fixed_precision_nonpositive_embedding_rejections += 1
                return None
            logged = mp.log(value)
            stats.embedding_log_evaluations += 1
            log2_value = logged / mp.log(2)
            stats.minimum_embedding_log2_floor = min(
                stats.minimum_embedding_log2_floor,
                math.floor(float(log2_value)),
            )
            stats.maximum_embedding_log2_ceiling = max(
                stats.maximum_embedding_log2_ceiling,
                math.ceil(float(log2_value)),
            )
            logs.append(float(logged))
    log_vector = np.array(logs, dtype=np.float64)
    target = -log_vector + np.mean(log_vector)
    solution = np.linalg.solve(
        NORMAL_MATRIX,
        LOG_MATRIX.T @ target,
    )
    stats.embedding_linear_solves += 1
    residual = float(
        np.max(np.abs(LOG_MATRIX @ solution - target))
    )
    stats.maximum_linear_solve_residual = max(
        stats.maximum_linear_solve_residual,
        residual,
    )
    move = [int(value) for value in np.rint(solution)]
    encoded = ",".join(str(value) for value in move).encode("ascii")
    stats.proposal_transcript_commitment = hashlib.sha256(
        stats.proposal_transcript_commitment + b"|" + encoded
    ).digest()
    stats.proposal_commitment_updates += 1
    stats.maximum_lattice_move_abs = max(
        stats.maximum_lattice_move_abs,
        *(abs(value) for value in move),
    )
    return move


def balance(
    vector: RingVector,
    base_ledger: tuple[int, ...] | list[int],
    stats: LatticeStats,
) -> tuple[RingVector, tuple[int, ...]]:
    if len(base_ledger) != UNIT_RANK:
        fail("oracle unit ledger width changed")
    if all(element == ring.ring_zero() for element in vector):
        return (
            [ring.ring_zero() for _ in vector],
            tuple(0 for _ in range(UNIT_RANK)),
        )
    stats.balance_calls += 1
    current = list(vector)
    norm = prior.norm_element(current, stats)
    stats.maximum_norm_element_payload_bits = max(
        stats.maximum_norm_element_payload_bits,
        prior.element_payload_bits(norm),
    )
    current_energy = prior.observe_energy(
        prior.field_trace(norm),
        stats,
    )
    move = propose_center(norm, stats)
    stats.balance_candidate_evaluations += 1
    if move is None or not any(move):
        stats.lattice_center_proposals_rejected += 1
        return current, tuple(base_ledger)
    scale = prior.ledger_scale(move, stats)
    stats.maximum_proposal_scale_payload_bits = max(
        stats.maximum_proposal_scale_payload_bits,
        prior.element_payload_bits(scale),
    )
    norm_factor = ring.ring_multiply(
        scale,
        prior.conjugate(scale),
    )
    stats.maximum_proposal_norm_factor_payload_bits = max(
        stats.maximum_proposal_norm_factor_payload_bits,
        prior.element_payload_bits(norm_factor),
    )
    trial_norm = ring.ring_multiply(norm_factor, norm)
    stats.candidate_norm_ring_multiplications += 2
    stats.maximum_trial_norm_element_payload_bits = max(
        stats.maximum_trial_norm_element_payload_bits,
        prior.element_payload_bits(trial_norm),
    )
    trial_energy = prior.observe_energy(
        prior.field_trace(trial_norm),
        stats,
    )
    if trial_energy >= current_energy:
        stats.lattice_center_proposals_rejected += 1
        return current, tuple(base_ledger)
    accepted = prior.multiply_vector(scale, current)
    stats.unit_vector_ring_multiplications += len(accepted)
    stats.maximum_accepted_proposal_vector_payload_bits = max(
        stats.maximum_accepted_proposal_vector_payload_bits,
        prior.vector_payload_bits(accepted),
    )
    stats.balance_selected_steps += 1
    stats.lattice_center_proposals_accepted += 1
    return (
        accepted,
        tuple(
            base - delta
            for base, delta in zip(
                base_ledger,
                move,
                strict=True,
            )
        ),
    )


prior.OracleStats = LatticeStats
prior.balance = balance


def extended_metrics(
    messages: list[prior.BalancedVector],
    coefficients: list[prior.BalancedElement],
    stats: LatticeStats,
) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = prior.record_metrics(
        messages,
        coefficients,
        stats,
    )
    metrics.update(
        {
            "lattice_center_proposals": stats.lattice_center_proposals,
            "lattice_center_proposals_accepted": (
                stats.lattice_center_proposals_accepted
            ),
            "lattice_center_proposals_rejected": (
                stats.lattice_center_proposals_rejected
            ),
            "fixed_precision_nonpositive_embedding_rejections": (
                stats.fixed_precision_nonpositive_embedding_rejections
            ),
            "embedding_dot_multiply_accumulations": (
                stats.embedding_dot_multiply_accumulations
            ),
            "embedding_log_evaluations": stats.embedding_log_evaluations,
            "embedding_linear_solves": stats.embedding_linear_solves,
            "proposal_commitment_updates": (
                stats.proposal_commitment_updates
            ),
            "proposal_transcript_sha256": (
                stats.proposal_transcript_commitment.hex()
            ),
            "maximum_norm_element_payload_bits": (
                stats.maximum_norm_element_payload_bits
            ),
            "maximum_proposal_scale_payload_bits": (
                stats.maximum_proposal_scale_payload_bits
            ),
            "maximum_proposal_norm_factor_payload_bits": (
                stats.maximum_proposal_norm_factor_payload_bits
            ),
            "maximum_trial_norm_element_payload_bits": (
                stats.maximum_trial_norm_element_payload_bits
            ),
            "maximum_accepted_proposal_vector_payload_bits": (
                stats.maximum_accepted_proposal_vector_payload_bits
            ),
            "maximum_lattice_move_abs": stats.maximum_lattice_move_abs,
            "minimum_embedding_log2_floor": (
                stats.minimum_embedding_log2_floor
            ),
            "maximum_embedding_log2_ceiling": (
                stats.maximum_embedding_log2_ceiling
            ),
            "maximum_linear_solve_residual": (
                stats.maximum_linear_solve_residual
            ),
        }
    )
    return metrics


def exact_metric_parity(
    oracle: dict[str, int | float | str],
    production: dict[str, Any],
) -> bool:
    return all(
        production[key] == value
        for key, value in oracle.items()
        if key != "maximum_linear_solve_residual"
    )


def residual_parity(
    oracle: dict[str, int | float | str],
    production: dict[str, Any],
) -> bool:
    return math.isclose(
        float(oracle["maximum_linear_solve_residual"]),
        float(production["maximum_linear_solve_residual"]),
        rel_tol=0.0,
        abs_tol=5e-10,
    )


def named_component_sum(
    case: dict[str, Any],
    metrics: dict[str, int | float | str],
) -> int:
    exact_temporaries = sum(
        int(metrics[key])
        for key in (
            "maximum_norm_element_payload_bits",
            "maximum_proposal_scale_payload_bits",
            "maximum_proposal_norm_factor_payload_bits",
            "maximum_trial_norm_element_payload_bits",
            "maximum_accepted_proposal_vector_payload_bits",
            "maximum_unit_materialization_payload_bits",
        )
    )
    return (
        int(metrics["maximum_declared_live_state_payload_bits"])
        + COSINE_CELLS * PRECISION_BITS
        + LOG_MATRIX_CELLS * 64
        + SOLVE_SCRATCH_CELLS * 64
        + 2 * PRECISION_BITS
        + 256
        + exact_temporaries
    )


def case_check(
    case: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    stats = LatticeStats()
    messages, coefficients = prior.build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        case["periods"],
        stats,
    )
    boundary = prior.project(messages[-1], stats)
    metrics = extended_metrics(messages, coefficients, stats)
    inverse_stats = LatticeStats()
    inverse_messages, inverse_coefficients = prior.build_state(
        context["operator"],
        context["seed"],
        context["characteristic"],
        case["periods"],
        inverse_stats,
    )
    inverse_metrics = extended_metrics(
        inverse_messages,
        inverse_coefficients,
        inverse_stats,
    )
    production = case["balanced_stats"]
    production_inverse = case["inverse_rematerialization_stats"]
    component_sum = named_component_sum(case, metrics)
    return {
        "periods": case["periods"],
        "family": case["family"],
        "boundary_sha256_equal": (
            hashlib.sha256(ring.encoded(boundary)).hexdigest()
            == case["boundary_sha256"]
        ),
        "exact_integer_precision_and_resource_tuple_equal": (
            exact_metric_parity(metrics, production)
        ),
        "inverse_exact_integer_precision_and_resource_tuple_equal": (
            exact_metric_parity(inverse_metrics, production_inverse)
        ),
        "solver_residual_within_declared_tolerance": (
            residual_parity(metrics, production)
            and residual_parity(inverse_metrics, production_inverse)
        ),
        "proposal_transcript_equal": (
            metrics["proposal_transcript_sha256"]
            == production["proposal_transcript_sha256"]
            and inverse_metrics["proposal_transcript_sha256"]
            == production_inverse["proposal_transcript_sha256"]
        ),
        "named_component_maxima_sum_equal": (
            component_sum == case["named_component_maxima_sum_bits"]
        ),
        "resident_payload_relation_equal": (
            (
                int(metrics["maximum_resident_payload_bits"])
                < case["raw_recurrence_payload_bits"]
            )
            == case["balanced_beats_raw_recurrence_payload"]
        ),
        "declared_live_payload_relation_equal": (
            (
                int(metrics["maximum_declared_live_state_payload_bits"])
                < case["raw_recurrence_payload_bits"]
            )
            == case[
                "balanced_declared_live_beats_raw_recurrence_payload"
            ]
        ),
        "named_component_raw_relation_equal": (
            (
                component_sum < case["raw_recurrence_payload_bits"]
            )
            == case[
                "named_component_maxima_sum_beats_raw_recurrence_payload"
            ]
        ),
        "exact_resource_tuple": metrics,
        "exact_inverse_resource_tuple": inverse_metrics,
    }


def all_boolean_checks(values: dict[str, Any]) -> bool:
    return all(
        value
        for key, value in values.items()
        if key not in {
            "periods",
            "family",
            "exact_resource_tuple",
            "exact_inverse_resource_tuple",
        }
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "pi_unit_lattice_center_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)
    if tuple(production["tested_periods"]) != EXPECTED_PERIODS:
        fail("oracle tested periods changed")

    contexts: dict[str, dict[str, Any]] = {}
    family_checks: dict[str, dict[str, bool]] = {}
    for family in ("primary", "reuse"):
        checks, context = prior.family_context(
            family,
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context
    cases = [
        case_check(case, contexts[case["family"].lower()])
        for case in production["cases"]
    ]
    restoration = prior.restoration_check(contexts, production)
    mutations = prior.mutation_check(contexts["primary"])
    scope = {
        "production_result_pass": production["result"] == "PASS",
        "fixed_precision_declared": (
            production["embedding_proposal_precision_bits"]
            == PRECISION_BITS
        ),
        "proposal_not_claimed_exact": (
            not production["embedding_proposal_is_exact"]
        ),
        "proposal_not_claimed_certified_cvp": (
            not production[
                "embedding_proposal_is_certified_closest_vector"
            ]
        ),
        "exact_trace_acceptance_declared": (
            production[
                "exact_trace_acceptance_controls_carrier_update"
            ]
        ),
        "all_resident_payloads_beat_raw": (
            production["all_cases_beat_raw_recurrence_payload"]
        ),
        "all_declared_live_payloads_beat_raw": (
            production[
                "all_declared_live_cases_beat_raw_recurrence_payload"
            ]
        ),
        "identical_classical_method_retained": (
            production["matched_classical"][
                "identical_fixed_precision_proposal_recurrence_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "total_resource_advantage_not_claimed": (
            "COMPUTATIONAL_ADVANTAGE"
            in production["not_established"]
            and not all(
                case[
                    "named_component_maxima_sum_beats_raw_recurrence_payload"
                ]
                for case in production["cases"]
            )
        ),
        "closest_vector_solution_not_claimed": (
            "EXACT_CLOSEST_VECTOR_SOLUTION"
            in production["not_established"]
        ),
        "distinct_phase_resource_not_claimed": (
            "DISTINCT_PHASE_RESOURCE"
            in production["not_established"]
        ),
    }
    passed = (
        all(all(checks.values()) for checks in family_checks.values())
        and all(all_boolean_checks(case) for case in cases)
        and all(restoration.values())
        and all(mutations.values())
        and all(scope.values())
    )
    result = {
        "result": "PASS" if passed else "FAIL",
        "experiment": (
            "SEPARATE_FIXED_PRECISION_LOG_EMBEDDING_CENTER_"
            "NORMAL_EQUATION_PROPOSAL_AND_EXACT_INTEGER_ACCEPTANCE_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_coefficient_method": (
            "SEQUENTIAL_MULTIPLICATION_BY_X_MOD_Q"
        ),
        "production_coefficient_method": (
            "BINARY_POLYNOMIAL_POWERING_MOD_Q"
        ),
        "oracle_center_solver": (
            "FLOAT64_NORMAL_EQUATION_SOLVE"
        ),
        "production_center_solver": (
            "FLOAT64_LEAST_SQUARES"
        ),
        "independent_exact_integer_precision_tuple_reexecution": True,
        "family_checks": family_checks,
        "case_checks": cases,
        "restoration_checks": restoration,
        "mutation_checks": mutations,
        "production_scope_checks": scope,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_"
            "FAMILIES_PERIODS1_AND64_65536_BIT_LOG_EMBEDDING_CENTER_"
            "PROPOSAL_EXACT_INTEGER_TRACE_ACCEPTANCE_EXACT_BOUNDARY_"
            "PAYLOAD_WIDTH_PROPOSAL_COMMITMENT_INVERSE_AND_REUSE_PARITY_"
            "COMPONENT_LEVEL_ACCOUNTING_SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
