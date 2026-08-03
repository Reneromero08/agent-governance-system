#!/usr/bin/env python3
"""Separate full-power and two-by-eight oracle for the M117 trace shears.

The oracle does not import the production M117 module.  It constructs the
public seeds and shear descriptors independently, executes the semantic law in
the canonical degree-sixteen cyclotomic power basis, and separately executes
the strongest compact two-by-eight integer recurrence through the M116 oracle
algebra.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_quadratic_extension_resident_carrier_oracle as ref


PRIME = 17
DIMENSION = 16

RingElement = tuple[int, ...]
SPair = ref.SPair
SElement = ref.SElement

ZERO: RingElement = (0,) * DIMENSION
ONE: RingElement = (1,) + (0,) * (DIMENSION - 1)

PLAN = (
    (0, 1, 2, 5),
    (1, 2, 0, 11),
    (2, 0, 1, 3),
)
ALTERNATE_PLAN = (
    (0, 2, 3, 4),
    (2, 3, 1, 7),
    (3, 1, 0, 9),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def ring_add(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def ring_subtract(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def ring_scale(value: RingElement, scalar: int) -> RingElement:
    return tuple(scalar * coefficient for coefficient in value)


def ring_monomial(exponent: int) -> RingElement:
    exponent %= PRIME
    if exponent < DIMENSION:
        return tuple(1 if index == exponent else 0 for index in range(DIMENSION))
    return (-1,) * DIMENSION


def ring_multiply(left: RingElement, right: RingElement) -> RingElement:
    work = [0 for _ in range(2 * DIMENSION - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            work[left_index + right_index] += left_value * right_value
    for degree in range(len(work) - 1, DIMENSION - 1, -1):
        coefficient = work[degree]
        if coefficient == 0:
            continue
        offset = degree - DIMENSION
        for reduced_degree in range(offset, offset + DIMENSION):
            work[reduced_degree] -= coefficient
        work[degree] = 0
    return tuple(work[:DIMENSION])


def ring_conjugate(value: RingElement) -> RingElement:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        result = ring_add(result, ring_scale(ring_monomial(-exponent), coefficient))
    return result


def full_trace(value: RingElement) -> int:
    return DIMENSION * value[0] - sum(value[1:])


def hermitian_relative_full(left: RingElement, right: RingElement) -> RingElement:
    product = ring_multiply(left, ring_conjugate(right))
    return ring_add(product, ring_conjugate(product))


def public_coefficients(family: str) -> tuple[int, int, int]:
    if family == "PRIMARY":
        return 1, 2, 4
    if family == "REUSE":
        return 8, 16, 11
    fail("unknown public family")


def unary_phase(coefficients: tuple[int, int, int], value: int) -> int:
    cubic, quadratic, linear = coefficients
    return (cubic * value**3 + quadratic * value**2 + linear * value) % PRIME


def seed_full(
    family: str,
    *,
    global_phase_offset: int = 0,
    single_site_perturbation: tuple[int, int] | None = None,
) -> tuple[RingElement, ...]:
    coefficients = public_coefficients(family)
    return tuple(
        ring_monomial(
            unary_phase(coefficients, value)
            + global_phase_offset
            + (
                single_site_perturbation[1]
                if single_site_perturbation is not None
                and single_site_perturbation[0] == value
                else 0
            )
        )
        for value in range(PRIME)
    )


@dataclass
class PairStats:
    real_subfield_ring_multiplications: int = 0
    real_subfield_coefficient_multiplications: int = 0
    relative_trace_calls: int = 0
    injection_calls: int = 0
    root_action_steps: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def s_scale(value: SElement, scalar: int) -> SElement:
    return tuple(scalar * coefficient for coefficient in value)  # type: ignore[return-value]


def hermitian_relative_pair(left: SPair, right: SPair, stats: PairStats) -> SElement:
    a_value, b_value = left
    c_value, d_value = right
    p_value = ref.s_multiply(a_value, c_value, stats)
    q_value = ref.s_multiply(b_value, d_value, stats)
    r_value = ref.s_multiply(
        ref.s_add(a_value, b_value),
        ref.s_add(c_value, d_value),
        stats,
    )
    cross = ref.s_subtract(ref.s_subtract(r_value, p_value), q_value)
    result = ref.s_add(
        ref.s_add(s_scale(p_value, 2), s_scale(q_value, 2)),
        ref.s_s1_multiply(cross),
    )
    stats.relative_trace_calls += 1
    return result


def root_action(value: SPair, exponent: int, stats: PairStats) -> SPair:
    output = value
    forward_steps = exponent % PRIME
    inverse_steps = (PRIME - forward_steps) % PRIME
    if forward_steps <= inverse_steps:
        for _ in range(forward_steps):
            a_value, b_value = output
            output = (
                ref.s_negate(b_value),
                ref.s_add(a_value, ref.s_s1_multiply(b_value)),
            )
        steps = forward_steps
    else:
        for _ in range(inverse_steps):
            a_value, b_value = output
            output = (
                ref.s_add(ref.s_s1_multiply(a_value), b_value),
                ref.s_negate(a_value),
            )
        steps = inverse_steps
    stats.root_action_steps += steps
    return output


def inject_pair(source: SPair, trace_value: SElement, exponent: int, stats: PairStats) -> SPair:
    scaled = (
        ref.s_multiply(source[0], trace_value, stats),
        ref.s_multiply(source[1], trace_value, stats),
    )
    stats.injection_calls += 1
    return root_action(scaled, exponent, stats)


def pair_trace(value: SElement) -> int:
    return 8 * value[0] - sum(value[1:])


def seed_pair(full_seed: tuple[RingElement, ...]) -> tuple[SPair, ...]:
    return tuple(ref.full_to_s_pair(value) for value in full_seed)


def execute_dual(
    full_seed: tuple[RingElement, ...],
    plan: tuple[tuple[int, int, int, int], ...] = PLAN,
) -> dict[str, Any]:
    full_vector = list(full_seed)
    pair_vector = list(seed_pair(full_seed))
    pair_stats = PairStats()
    step_parity: list[bool] = []
    for left, right, target, exponent in plan:
        full_h = hermitian_relative_full(full_vector[left], full_vector[right])
        full_injected = ring_multiply(
            ring_monomial(exponent),
            ring_multiply(full_vector[left], full_h),
        )
        full_vector[target] = ring_add(full_vector[target], full_injected)

        pair_h = hermitian_relative_pair(pair_vector[left], pair_vector[right], pair_stats)
        pair_injected = inject_pair(pair_vector[left], pair_h, exponent, pair_stats)
        pair_vector[target] = ref.pair_add(pair_vector[target], pair_injected)
        step_parity.append(
            all(
                ref.s_pair_to_full(pair_value) == full_value
                for pair_value, full_value in zip(pair_vector, full_vector, strict=True)
            )
            and ref.s_pair_to_full((pair_h, ref.S_ZERO)) == full_h
        )

    full_h = hermitian_relative_full(full_vector[1], full_vector[2])
    full_h_trace = full_trace(full_h)
    if full_h_trace % 2:
        fail("full relative-Hermitian trace was not even")
    full_boundary = full_h_trace // 2
    pair_h = hermitian_relative_pair(pair_vector[1], pair_vector[2], pair_stats)
    pair_boundary = pair_trace(pair_h)
    boundary_parity = full_boundary == pair_boundary

    for left, right, target, exponent in reversed(plan):
        full_h = hermitian_relative_full(full_vector[left], full_vector[right])
        full_injected = ring_multiply(
            ring_monomial(exponent),
            ring_multiply(full_vector[left], full_h),
        )
        full_vector[target] = ring_subtract(full_vector[target], full_injected)

        pair_h = hermitian_relative_pair(pair_vector[left], pair_vector[right], pair_stats)
        pair_injected = inject_pair(pair_vector[left], pair_h, exponent, pair_stats)
        pair_vector[target] = ref.pair_subtract(pair_vector[target], pair_injected)
        step_parity.append(
            all(
                ref.s_pair_to_full(pair_value) == full_value
                for pair_value, full_value in zip(pair_vector, full_vector, strict=True)
            )
        )
    return {
        "boundary": full_boundary,
        "full_pair_boundary_equal": boundary_parity,
        "all_forward_inverse_step_states_equal": all(step_parity),
        "full_restored_exactly": tuple(full_vector) == full_seed,
        "pair_restored_exactly": tuple(pair_vector) == seed_pair(full_seed),
        "pair_stats": pair_stats.as_json(),
    }


def same_order_inverse_fails(full_seed: tuple[RingElement, ...]) -> bool:
    vector = list(forward_full(full_seed, PLAN))
    for left, right, target, exponent in PLAN:
        h_value = hermitian_relative_full(vector[left], vector[right])
        injected = ring_multiply(
            ring_monomial(exponent), ring_multiply(vector[left], h_value)
        )
        vector[target] = ring_subtract(vector[target], injected)
    return tuple(vector) != full_seed


def forward_full(
    full_seed: tuple[RingElement, ...],
    plan: tuple[tuple[int, int, int, int], ...],
) -> tuple[RingElement, ...]:
    vector = list(full_seed)
    for left, right, target, exponent in plan:
        h_value = hermitian_relative_full(vector[left], vector[right])
        injected = ring_multiply(
            ring_monomial(exponent), ring_multiply(vector[left], h_value)
        )
        vector[target] = ring_add(vector[target], injected)
    return tuple(vector)


def wrong_arithmetic_inverse_fails(full_seed: tuple[RingElement, ...]) -> bool:
    vector = list(forward_full(full_seed, PLAN))
    for ordinal in reversed(range(len(PLAN))):
        left, right, target, exponent = PLAN[ordinal]
        if ordinal == 2:
            exponent = 4
        h_value = hermitian_relative_full(vector[left], vector[right])
        injected = ring_multiply(
            ring_monomial(exponent), ring_multiply(vector[left], h_value)
        )
        vector[target] = ring_subtract(vector[target], injected)
    return tuple(vector) != full_seed


def no_shear_boundary(full_seed: tuple[RingElement, ...]) -> int:
    value = full_trace(hermitian_relative_full(full_seed[1], full_seed[2]))
    if value % 2:
        fail("no-shear full trace was not even")
    return value // 2


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_three_shear_relative_hermitian_trace_feedback_oracle.py")
    cases = []
    for family, expected, disabled in (
        ("PRIMARY", 197, 16),
        ("REUSE", 112, -1),
    ):
        seed = seed_full(family)
        execution = execute_dual(seed)
        rotated = [
            execute_dual(seed_full(family, global_phase_offset=offset))["boundary"]
            for offset in (1, 8, 16)
        ]
        perturbed = execute_dual(
            seed_full(family, single_site_perturbation=(2, 1))
        )
        alternate = execute_dual(seed, ALTERNATE_PLAN)
        cases.append(
            {
                "family": family,
                "boundary": execution["boundary"],
                "expected_boundary": expected,
                "coupling_disabled_boundary": no_shear_boundary(seed),
                "expected_coupling_disabled_boundary": disabled,
                "declared_cross_cell_erasure_boundary": 0,
                "executed_physical_dephasing_model": False,
                "global_phase_rotated_boundaries": rotated,
                "single_site_perturbed_boundary": perturbed["boundary"],
                "alternate_descriptor_boundary": alternate["boundary"],
                "same_order_inverse_fails": same_order_inverse_fails(seed),
                "wrong_arithmetic_inverse_fails": (
                    wrong_arithmetic_inverse_fails(seed)
                ),
                "all_three_shear_pairs_noncommute": all(
                    forward_full(seed, (PLAN[left], PLAN[right]))
                    != forward_full(seed, (PLAN[right], PLAN[left]))
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ),
                "execution": execution,
                "perturbed_execution": perturbed,
                "alternate_execution": alternate,
            }
        )
    result = {
        "result": "PASS",
        "experiment": "THREE_CELL_TRACE_FEEDBACK_TRIANGULAR_COUPLING_COLLAPSE_TEST_ORACLE",
        "production_module_imported": False,
        "semantic_oracle": "INDEPENDENT_CANONICAL_DEGREE16_CYCLOTOMIC_POWER_BASIS",
        "compact_reference": "M116_ORACLE_TWO_BY_EIGHT_INTEGER_RECURRENCE",
        "public_plan": [list(shear) for shear in PLAN],
        "cases": cases,
        "observed_resource_law": {
            "logical_carrier_cells": PRIME,
            "logical_integer_coordinates": PRIME * DIMENSION,
            "accepted_pair_relative_trace_calls": 7,
            "accepted_pair_relative_trace_real_multiplications": 21,
            "accepted_pair_injection_calls": 6,
            "accepted_pair_injection_real_multiplications": 12,
            "accepted_pair_root_action_steps": 28,
            "full_power_basis_is_semantic_oracle_not_strongest_baseline": True,
            "strongest_compact_reference_is_identical_to_production_recurrence": True,
            "metrics_are_selected_operation_counts_not_whole_process_peak": True,
        },
        "claim_ceiling": (
            "INDEPENDENT_BOUNDARY_AND_STATE_PARITY_FOR_TWO_PUBLIC_FAMILIES_"
            "GLOBAL_ROTATIONS_ONE_PHASE_PERTURBATION_AND_ONE_ALTERNATE_PLAN"
        ),
        "not_established": [
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATVM_CUSTODY",
            "PHYSICAL_WAVEFORM_EXECUTION",
        ],
    }
    hard_gate = {
        "expected_boundaries": all(
            case["boundary"] == case["expected_boundary"]
            and case["coupling_disabled_boundary"]
            == case["expected_coupling_disabled_boundary"]
            for case in cases
        ),
        "semantic_parity": all(
            case["execution"]["full_pair_boundary_equal"]
            and case["execution"]["all_forward_inverse_step_states_equal"]
            and case["perturbed_execution"]["full_pair_boundary_equal"]
            and case["alternate_execution"]["full_pair_boundary_equal"]
            for case in cases
        ),
        "restoration": all(
            case[key]["full_restored_exactly"]
            and case[key]["pair_restored_exactly"]
            for case in cases
            for key in ("execution", "perturbed_execution", "alternate_execution")
        ),
        "global_phase_invariance": all(
            all(boundary == case["boundary"] for boundary in case["global_phase_rotated_boundaries"])
            for case in cases
        ),
        "single_site_perturbation": all(
            case["single_site_perturbed_boundary"] != case["boundary"]
            for case in cases
        ),
        "inverse_controls": all(
            case["same_order_inverse_fails"]
            and case["wrong_arithmetic_inverse_fails"]
            for case in cases
        ),
        "pairwise_commutators": all(
            case["all_three_shear_pairs_noncommute"] for case in cases
        ),
        "resource_counts": all(
            case["execution"]["pair_stats"]
            == {
                "real_subfield_ring_multiplications": 33,
                "real_subfield_coefficient_multiplications": 2112,
                "relative_trace_calls": 7,
                "injection_calls": 6,
                "root_action_steps": 28,
            }
            for case in cases
        ),
    }
    if not all(hard_gate.values()):
        fail("three-shear oracle gate failed: " + json.dumps(hard_gate, sort_keys=True))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
