#!/usr/bin/env python3
"""Separate exact oracle for the period-17 pi-height certificate.

The oracle imports only the previously sealed independent reference kernel,
not either production phase module.  It recompiles both public descriptors
and operators, checks the supplied exact annihilators, independently derives
pi-valuations and the tropical induction inequalities, and uses Brent cycle
detection instead of the production Floyd implementation.

The result concerns exact lossless discrete boundary representation.  It does
not establish a distinct phase resource, a machine-memory lower bound, or a
computational advantage over the identical compact classical recurrence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

import f17_cubic_chain_period17_unit_height_reduction_oracle as reference


PRIME = 17
DIMENSION = 16
INITIAL_PERIODS = 17
CYCLE_STEP_CAP = 10_000_000

RingElement = tuple[int, ...]
State = tuple[int, tuple[int, ...]]


def fail(message: str) -> None:
    raise RuntimeError(message)


def height_lower_bound(periods: int) -> int:
    if periods < 1:
        fail("oracle height bound requires a positive period")
    return (272 * periods + 18) // 3


PI = reference.ring_subtract(
    reference.ring_one(),
    reference.ring_monomial(1),
)


def divide_pi_exact(element: RingElement) -> RingElement:
    """Divide by 1-zeta using the coefficient-prefix identity."""
    total = sum(element)
    if total % PRIME:
        fail("oracle element is not divisible by pi")
    last = total // PRIME
    prefix = 0
    quotient = []
    for index, coefficient in enumerate(element):
        prefix += coefficient
        quotient.append(prefix - (index + 1) * last)
    result = tuple(quotient)
    if reference.ring_multiply(PI, result) != element:
        fail("oracle pi-division identity failed")
    return result


def pi_valuation(element: RingElement) -> int | None:
    if element == reference.ring_zero():
        return None
    current = element
    valuation = 0
    while sum(current) % PRIME == 0:
        current = divide_pi_exact(current)
        valuation += 1
    return valuation


def residue_after_division(
    element: RingElement,
    exponent: int,
) -> int:
    current = element
    for _ in range(exponent):
        current = divide_pi_exact(current)
    return sum(current) % PRIME


def exact_direct_boundaries(
    operator: reference.RingMatrix,
    seed: reference.RingVector,
) -> list[RingElement]:
    current = seed
    boundaries = []
    for _ in range(INITIAL_PERIODS):
        current = reference.matrix_vector_multiply(operator, current)
        boundaries.append(reference.project(current))
    return boundaries


def induction_records(
    characteristic: list[RingElement],
) -> list[dict[str, Any]]:
    if len(characteristic) != PRIME + 1:
        fail("oracle characteristic width changed")
    if characteristic[0] != reference.ring_one():
        fail("oracle characteristic is not monic")
    if characteristic[-1] != reference.ring_zero():
        fail("oracle characteristic constant is not zero")
    records = []
    for lag in range(1, DIMENSION + 1):
        coefficient = characteristic[lag]
        valuation = pi_valuation(coefficient)
        requirements = [
            height_lower_bound(300 + phase)
            - height_lower_bound(300 + phase - lag)
            for phase in (0, 1, 2)
        ]
        if valuation is not None and valuation < max(requirements):
            fail("oracle coefficient fails tropical induction")
        residues = [
            (
                residue_after_division(coefficient, requirement)
                if valuation == requirement
                else 0
            )
            for requirement in requirements
        ]
        records.append(
            {
                "lag": lag,
                "valuation": valuation,
                "required_by_target_mod3": {
                    str(phase): requirement
                    for phase, requirement in zip(
                        (0, 1, 2),
                        requirements,
                        strict=True,
                    )
                },
                "normalized_residue_by_target_mod3": {
                    str(phase): residue
                    for phase, residue in zip(
                        (0, 1, 2),
                        residues,
                        strict=True,
                    )
                },
            }
        )
    return records


def transition(
    state: State,
    rows: dict[int, tuple[int, ...]],
) -> State:
    target_mod3, values = state
    if len(values) != DIMENSION:
        fail("oracle recurrence state width changed")
    coefficients = rows[target_mod3]
    output = -sum(
        coefficient * values[-lag]
        for lag, coefficient in enumerate(coefficients, start=1)
    )
    return (
        (target_mod3 + 1) % 3,
        (*values[1:], output % PRIME),
    )


def brent_cycle(
    initial_residues: list[int],
    rows: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    if len(initial_residues) != DIMENSION:
        fail("oracle cycle seed width changed")
    start: State = (
        INITIAL_PERIODS % 3,
        tuple(initial_residues),
    )

    power = 1
    cycle_length = 1
    tortoise = start
    hare = transition(start, rows)
    meeting_steps = 1
    while tortoise != hare:
        if meeting_steps >= CYCLE_STEP_CAP:
            fail("oracle Brent meeting exceeded cap")
        if power == cycle_length:
            tortoise = hare
            power *= 2
            cycle_length = 0
        hare = transition(hare, rows)
        cycle_length += 1
        meeting_steps += 1

    prefix = 0
    tortoise = start
    hare = start
    for _ in range(cycle_length):
        hare = transition(hare, rows)
    while tortoise != hare:
        if prefix >= CYCLE_STEP_CAP:
            fail("oracle Brent prefix exceeded cap")
        tortoise = transition(tortoise, rows)
        hare = transition(hare, rows)
        prefix += 1

    cursor = tortoise
    cycle_hash = hashlib.sha256()
    nonzero = 0
    for _ in range(cycle_length):
        next_state = transition(cursor, rows)
        output = next_state[1][-1]
        cycle_hash.update(bytes([output]))
        nonzero += int(output != 0)
        cursor = next_state
    if cursor != tortoise:
        fail("oracle Brent cycle did not close")
    return {
        "algorithm": "BRENT_CONSTANT_STATE_CYCLE_DETECTION",
        "step_cap": CYCLE_STEP_CAP,
        "meeting_steps": meeting_steps,
        "prefix_length": prefix,
        "cycle_length": cycle_length,
        "cycle_nonzero_outputs": nonzero,
        "cycle_zero_outputs": cycle_length - nonzero,
        "cycle_output_sha256": cycle_hash.hexdigest(),
        "cycle_has_nonzero_output": nonzero > 0,
        "retained_brent_bound_state_recurrence_residue_cells": (
            3 * DIMENSION
        ),
        "retained_brent_bound_state_phase_scalar_cells": 3,
        "retained_cycle_dictionary_entries": 0,
    }


def cycle_comparable(cycle: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cycle[key]
        for key in (
            "step_cap",
            "prefix_length",
            "cycle_length",
            "cycle_nonzero_outputs",
            "cycle_zero_outputs",
            "cycle_output_sha256",
            "cycle_has_nonzero_output",
            "retained_cycle_dictionary_entries",
        )
    }


def check_family(
    production_family: dict[str, Any],
    certificate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = production_family["family"].lower()
    descriptor = reference.compile_descriptor(
        PRIME + 1,
        family.upper(),
    )
    operator = reference.compile_operator(descriptor)
    characteristic = [
        tuple(element)
        for element in certificate["characteristic"]
    ]
    seed = reference.seed_vector(descriptor)

    boundaries = exact_direct_boundaries(operator, seed)
    valuations = [pi_valuation(boundary) for boundary in boundaries]
    if any(value is None for value in valuations):
        fail("oracle initial exact boundary vanished")
    finite_valuations = [int(value) for value in valuations]
    lower_bounds = [
        height_lower_bound(period)
        for period in range(1, INITIAL_PERIODS + 1)
    ]
    residues = [
        residue_after_division(boundary, bound)
        for boundary, bound in zip(
            boundaries,
            lower_bounds,
            strict=True,
        )
    ]
    records = induction_records(characteristic)
    rows = {
        phase: tuple(
            record["normalized_residue_by_target_mod3"][str(phase)]
            for record in records
        )
        for phase in (0, 1, 2)
    }
    generated_period17 = transition(
        (INITIAL_PERIODS % 3, tuple(residues[:DIMENSION])),
        rows,
    )[1][-1]
    cycle = brent_cycle(residues[:DIMENSION], rows)

    checks = {
        "descriptor_sha256_equal": (
            hashlib.sha256(reference.encoded(descriptor)).hexdigest()
            == certificate["public_program_sha256"]
        ),
        "operator_sha256_equal": (
            hashlib.sha256(reference.encoded(operator)).hexdigest()
            == certificate["operator_sha256"]
        ),
        "characteristic_sha256_equal": (
            hashlib.sha256(
                reference.encoded(characteristic)
            ).hexdigest()
            == certificate["characteristic_sha256"]
        ),
        "whole_operator_annihilator_identity_exact": (
            reference.check_annihilator(operator, characteristic)
        ),
        "production_characteristic_identity_asserted": (
            production_family["characteristic_identity_exact"]
            and certificate["characteristic_identity_exact"]
        ),
        "initial_boundary_valuations_equal": (
            finite_valuations
            == production_family["initial_boundary_valuations"]
        ),
        "initial_height_lower_bounds_equal": (
            lower_bounds
            == production_family["initial_height_lower_bounds"]
        ),
        "initial_normalized_residues_equal": (
            residues
            == production_family["initial_normalized_residues_mod17"]
        ),
        "period17_normalized_recurrence_equal": (
            generated_period17 == residues[DIMENSION]
        ),
        "coefficient_records_equal": (
            records == production_family["coefficient_records"]
        ),
        "cycle_certificate_equal": (
            cycle_comparable(cycle)
            == cycle_comparable(production_family["cycle"])
        ),
        "direct_ring_multiply_accumulations_equal": (
            INITIAL_PERIODS * PRIME * PRIME
            == production_family[
                "direct_ring_multiply_accumulations"
            ]
        ),
    }
    context = {
        "descriptor": descriptor,
        "operator": operator,
        "characteristic": characteristic,
        "seed": seed,
        "rows": rows,
        "residues": residues,
        "cycle": cycle,
    }
    return checks, context


def mutation_checks(
    contexts: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    characteristic = contexts["primary"]["characteristic"]
    lag = 3
    weakened = divide_pi_exact(characteristic[lag])
    required = max(
        height_lower_bound(300 + phase)
        - height_lower_bound(300 + phase - lag)
        for phase in (0, 1, 2)
    )
    weakened_rejected = (
        pi_valuation(weakened) is not None
        and int(pi_valuation(weakened)) < required
    )

    rows = dict(contexts["primary"]["rows"])
    row_zero = list(rows[0])
    row_zero[0] = (row_zero[0] + 1) % PRIME
    rows[0] = tuple(row_zero)
    perturbed_cycle = brent_cycle(
        contexts["primary"]["residues"][:DIMENSION],
        rows,
    )
    cycle_changed = (
        cycle_comparable(perturbed_cycle)
        != cycle_comparable(contexts["primary"]["cycle"])
    )
    return {
        "one_pi_weakened_induction_coefficient_rejected": (
            weakened_rejected
        ),
        "normalized_recurrence_coefficient_perturbation_detected": (
            cycle_changed
        ),
    }


def restoration_checks(
    contexts: dict[str, dict[str, Any]],
    production: dict[str, Any],
) -> dict[str, Any]:
    shared = reference.create_carrier()
    original_backing = reference.carrier_backing_identity(shared)
    primary_boundary, primary = reference.execute_on_carrier(
        shared,
        contexts["primary"]["operator"],
        contexts["primary"]["seed"],
        contexts["primary"]["characteristic"],
        DIMENSION,
    )
    reuse_boundary, reuse = reference.execute_on_carrier(
        shared,
        contexts["reuse"]["operator"],
        contexts["reuse"]["seed"],
        contexts["reuse"]["characteristic"],
        DIMENSION,
    )
    fresh_boundary, fresh = reference.execute_on_carrier(
        reference.create_carrier(),
        contexts["reuse"]["operator"],
        contexts["reuse"]["seed"],
        contexts["reuse"]["characteristic"],
        DIMENSION,
    )
    expected = production["restoration_reuse_case"]
    return {
        "primary_restored_exactly": primary["restored_exactly"],
        "reuse_restored_exactly": reuse["restored_exactly"],
        "fresh_reuse_restored_exactly": fresh["restored_exactly"],
        "same_original_backing": (
            primary["same_message_and_coefficient_backing"]
            and reuse["same_message_and_coefficient_backing"]
            and reference.carrier_backing_identity(shared)
            == original_backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_boundary == fresh_boundary
        ),
        "generation_equal": (
            shared["generation"] == expected["generation"]
        ),
        "lease_equal": shared["lease"] == expected["lease"],
        "all_messages_zero": reference.carrier_is_zero(shared),
        "restoration_scope_explicit": (
            expected["restoration_scope"]
            == "EXACT_MESSAGE_PAYLOAD_ZERO_AND_SAME_BACKING_WITH_"
            "MONOTONE_GENERATION_AND_LEASE_METADATA"
        ),
        "full_carrier_object_state_equality_not_claimed": (
            not expected["full_carrier_object_state_equal"]
        ),
        "repeated_use_metadata_width_not_claimed_bounded": (
            not expected["repeated_use_metadata_width_bounded"]
        ),
        "primary_boundary_nonzero": (
            primary_boundary != reference.ring_zero()
        ),
        "retained_inverse_history_bytes_equal": (
            expected["retained_inverse_history_bytes"] == 0
        ),
        "baseline_reload_bytes_equal": (
            expected["baseline_reload_bytes"] == 0
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_chain_period17_"
            "height_lower_bound_oracle.py PRODUCTION_RESULT"
        )
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        production = json.load(handle)

    family_records = {
        family["family"].lower(): family
        for family in production["families"]
    }
    if set(family_records) != {"primary", "reuse"}:
        fail("oracle expected exactly the two declared families")

    family_checks = {}
    contexts = {}
    for family in ("primary", "reuse"):
        checks, context = check_family(
            family_records[family],
            production["block_certificates"][family],
        )
        family_checks[family] = checks
        contexts[family] = context

    mutations = mutation_checks(contexts)
    restoration = restoration_checks(contexts, production)
    exact_densities = {}
    for family, context in contexts.items():
        numerator = context["cycle"]["cycle_nonzero_outputs"]
        denominator = context["cycle"]["cycle_length"]
        common = math.gcd(numerator, denominator)
        exact_densities[family] = {
            "numerator": numerator // common,
            "denominator": denominator // common,
        }

    production_gate = {
        "result_pass": production["result"] == "PASS",
        "height_formula_equal": (
            production["height_lower_bound_formula"]
            == "CEIL((272*N+16)/3)"
        ),
        "all_initial_boundaries_satisfy_bound": (
            production["all_initial_boundaries_satisfy_bound"]
        ),
        "all_characteristic_identities_exact": (
            production["all_characteristic_identities_exact"]
        ),
        "all_coefficient_valuations_prove_induction": (
            production["all_coefficient_valuations_prove_induction"]
        ),
        "all_normalized_cycles_have_nonzero_outputs": (
            production["all_normalized_cycles_have_nonzero_outputs"]
        ),
        "exact_cycle_densities_equal": (
            exact_densities
            == production["exact_cycle_nonzero_densities"]
        ),
        "infinite_nonzero_boundaries_claim_bounded_by_cycle": (
            production[
                "infinitely_many_exact_nonzero_boundaries_certified"
            ]
        ),
        "infinite_distinct_valuations_claim_bounded_by_cycle": (
            production[
                "infinitely_many_distinct_pi_valuations_certified"
            ]
        ),
        "fixed_finite_lossless_boundary_alphabet_rejection_present": (
            production[
                "fixed_finite_lossless_discrete_boundary_alphabet_rejected"
            ]
        ),
        "only_logarithmic_bit_lower_bound_claimed": (
            production[
                "asymptotic_lossless_exact_boundary_"
                "encoding_horizon_lower_bound"
            ]
            == "MAX_CODE_WIDTH_THROUGH_N_IS_OMEGA_LOG_N_WITHOUT_"
            "FREE_PERIOD_INDEX"
        ),
        "boundary_encoding_scope_explicit": (
            production["lower_bound_scope"]
            == "WORST_CASE_INJECTIVE_EXACT_BOUNDARY_OR_VALUATION_"
            "ENCODING_THROUGH_PERIOD_N_DECODER_GETS_NO_FREE_N"
        ),
        "compact_exponent_upper_bound_present": (
            production["compact_exponent_ledger_upper_bound"]
            == "O_LOG_N_BITS_FOR_THE_PI_EXPONENT_ALONE"
        ),
        "exact_regauging_and_content_factoring_scoped": (
            production["exact_regauging_preserves_pi_valuation"]
            and production[
                "pi_content_factoring_transfers_"
                "valuation_to_exponent_ledger"
            ]
            and not production[
                "cyclotomic_units_can_remove_pi_valuation"
            ]
        ),
        "matched_identical_classical_recurrence_retained": (
            production["matched_classical"][
                "identical_characteristic_recurrence_available"
            ]
            and production["matched_classical"][
                "identical_pi_valuation_certificate_available"
            ]
            and not production["matched_classical"][
                "comparison_establishes_advantage"
            ]
        ),
        "no_linear_bit_lower_bound_claimed": (
            "LINEAR_BIT_LOWER_BOUND"
            in production["not_established"]
        ),
        "no_generic_machine_memory_lower_bound_claimed": (
            "GENERIC_MACHINE_STATE_MEMORY_LOWER_BOUND"
            in production["not_established"]
            and "ONLINE_SPACE_LOWER_BOUND"
            in production["not_established"]
            and "LOWER_BOUND_WITH_PERIOD_INDEX_OR_EXTERNAL_COUNTER_FREE"
            in production["not_established"]
        ),
        "no_distinct_phase_resource_claimed": (
            "DISTINCT_PHASE_RESOURCE"
            in production["not_established"]
        ),
    }
    all_family_checks = all(
        all(checks.values())
        for checks in family_checks.values()
    )
    all_mutations_detected = all(mutations.values())
    all_restoration_checks = all(restoration.values())
    all_production_gate_checks = all(production_gate.values())
    result = {
        "result": (
            "PASS"
            if (
                all_family_checks
                and all_mutations_detected
                and all_restoration_checks
                and all_production_gate_checks
            )
            else "FAIL"
        ),
        "experiment": (
            "SEPARATE_EXACT_F17_PERIOD17_BOUNDARY_"
            "HEIGHT_LOWER_BOUND_ORACLE"
        ),
        "oracle_imports_production_module": False,
        "oracle_reference_kernel": (
            "F17_PERIOD17_UNIT_HEIGHT_REDUCTION_ORACLE"
        ),
        "oracle_cycle_algorithm_distinct": True,
        "family_checks": family_checks,
        "mutation_checks": mutations,
        "restoration_checks": restoration,
        "production_gate_checks": production_gate,
        "exact_cycle_nonzero_densities": exact_densities,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_"
            "CUBIC_PATH_FAMILIES_Q_ZETA17_PI_ADIC_BOUNDARY_"
            "HEIGHT_LOWER_BOUND_IDENTICAL_COMPACT_CLASSICAL_"
            "RECURRENCE_EXACT_SUBTRACTIVE_RESTORATION_"
            "SOFTWARE_ONLY"
        ),
        "not_established": production["not_established"],
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
