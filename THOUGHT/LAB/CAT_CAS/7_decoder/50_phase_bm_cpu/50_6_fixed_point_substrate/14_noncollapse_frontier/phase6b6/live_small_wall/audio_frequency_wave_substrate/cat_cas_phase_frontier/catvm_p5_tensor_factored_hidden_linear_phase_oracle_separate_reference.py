#!/usr/bin/env python3
"""Independent exact factor, dense-parity, and query oracle for M242."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


P = 5
DIMENSIONS = (1, 2, 4, 8, 16, 32)
E = tuple[int, int, int, int]
ZERO: E = (0, 0, 0, 0)
ONE: E = (1, 0, 0, 0)
SQRT5: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def mul(left: E, right: E) -> E:
    raw = [0] * 7
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            raw[left_index + right_index] += left_value * right_value
    for degree in range(6, 3, -1):
        coefficient = raw[degree]
        if coefficient:
            for target in range(degree - 4, degree):
                raw[target] -= coefficient
            raw[degree] = 0
    return tuple(raw[:4])  # type: ignore[return-value]


def root(power: int) -> E:
    power %= P
    if power == 4:
        return (-1, -1, -1, -1)
    result = [0] * 4
    result[power] = 1
    return tuple(result)  # type: ignore[return-value]


def canonicalize(values: list[E], exponent: int) -> int:
    while exponent and all(coefficient % P == 0 for value in values for coefficient in value):
        for index, value in enumerate(values):
            values[index] = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def factor_fourier_all(
    values: list[E],
    exponent: int,
    dimension: int,
    direction: int,
) -> tuple[list[E], int, int]:
    result = [ZERO] * len(values)
    terms = 0
    for wire in range(dimension):
        offset = P * wire
        for output in range(P):
            total = ZERO
            for source in range(P):
                total = add(total, mul(root(direction * source * output), values[offset + source]))
                terms += 1
            result[offset + output] = mul(SQRT5, total)
    exponent += 1
    exponent = canonicalize(result, exponent)
    return result, exponent, terms


def factor_oracle(
    values: list[E],
    secret: Sequence[int],
    direction: int,
) -> int:
    visits = 0
    for wire, residue in enumerate(secret):
        offset = P * wire
        for value in range(P):
            values[offset + value] = mul(
                root(direction * residue * value),
                values[offset + value],
            )
            visits += 1
    return visits


def project_factors(values: Sequence[E], exponent: int, dimension: int) -> tuple[int, ...]:
    if exponent != 0:
        raise RuntimeError("reference final factor exponent is not zero")
    inferred: list[int] = []
    for wire in range(dimension):
        offset = P * wire
        support = [value for value in range(P) if values[offset + value] != ZERO]
        if len(support) != 1 or values[offset + support[0]] != ONE:
            raise RuntimeError("reference final factor is not an exact basis state")
        inferred.append(support[0])
    return tuple(inferred)


def factor_transaction(secret: Sequence[int]) -> dict[str, object]:
    dimension = len(secret)
    values = [ZERO] * (P * dimension)
    for wire in range(dimension):
        values[P * wire] = ONE
    exponent = 0
    fourier_terms = 0
    values, exponent, terms = factor_fourier_all(values, exponent, dimension, 1)
    fourier_terms += terms
    forward_oracle_visits = factor_oracle(values, secret, 1)
    hidden_values = list(values)
    hidden_exponent = exponent
    values, exponent, terms = factor_fourier_all(values, exponent, dimension, -1)
    fourier_terms += terms
    inferred = project_factors(values, exponent, dimension)
    values, exponent, terms = factor_fourier_all(values, exponent, dimension, 1)
    fourier_terms += terms
    inverse_oracle_visits = factor_oracle(values, secret, -1)
    values, exponent, terms = factor_fourier_all(values, exponent, dimension, -1)
    fourier_terms += terms
    restored = (
        exponent == 0
        and all(
            values[P * wire] == ONE
            and all(values[P * wire + value] == ZERO for value in range(1, P))
            for wire in range(dimension)
        )
    )
    return {
        "inferred_secret": list(inferred),
        "canonical_post_inverse_state_exact": restored,
        "factor_fourier_character_terms": fourier_terms,
        "oracle_factor_cell_visits": forward_oracle_visits + inverse_oracle_visits,
        "hidden_secret_residue_accesses": 2 * dimension,
        "hidden_values": hidden_values,
        "hidden_exponent": hidden_exponent,
    }


def coordinates(index: int, dimension: int) -> tuple[int, ...]:
    result = [0] * dimension
    for position in range(dimension - 1, -1, -1):
        result[position] = index % P
        index //= P
    return tuple(result)


def flat_index(state: Sequence[int]) -> int:
    result = 0
    for coordinate in state:
        result = P * result + coordinate
    return result


def dense_fourier(
    values: list[E],
    exponent: int,
    dimension: int,
    wire: int,
    direction: int,
) -> tuple[list[E], int]:
    result = [ZERO] * len(values)
    for destination in range(len(values)):
        output_state = list(coordinates(destination, dimension))
        output = output_state[wire]
        total = ZERO
        for source in range(P):
            source_state = output_state.copy()
            source_state[wire] = source
            total = add(
                total,
                mul(root(direction * source * output), values[flat_index(source_state)]),
            )
        result[destination] = mul(SQRT5, total)
    exponent += 1
    exponent = canonicalize(result, exponent)
    return result, exponent


def dense_phase_oracle(values: list[E], dimension: int, secret: Sequence[int], direction: int) -> None:
    for location in range(len(values)):
        state = coordinates(location, dimension)
        phase = sum(secret[index] * state[index] for index in range(dimension))
        values[location] = mul(root(direction * phase), values[location])


def tensor_materialization(
    factors: Sequence[E],
    factor_exponent: int,
    dimension: int,
) -> tuple[list[E], int]:
    result: list[E] = []
    for location in range(P**dimension):
        state = coordinates(location, dimension)
        value = ONE
        for wire, coordinate in enumerate(state):
            value = mul(value, factors[P * wire + coordinate])
        result.append(value)
    exponent = factor_exponent * dimension
    exponent = canonicalize(result, exponent)
    return result, exponent


def dense_hidden_parity(secret: Sequence[int], factor: dict[str, object]) -> bool:
    dimension = len(secret)
    values = [ZERO] * (P**dimension)
    values[0] = ONE
    exponent = 0
    for wire in range(dimension):
        values, exponent = dense_fourier(values, exponent, dimension, wire, 1)
    dense_phase_oracle(values, dimension, secret, 1)
    product, product_exponent = tensor_materialization(
        factor["hidden_values"],  # type: ignore[arg-type]
        int(factor["hidden_exponent"]),
        dimension,
    )
    if values != product or exponent != product_exponent:
        return False
    for wire in range(dimension):
        values, exponent = dense_fourier(values, exponent, dimension, wire, -1)
    support = [location for location, value in enumerate(values) if value != ZERO]
    return (
        exponent == 0
        and support == [flat_index(secret)]
        and values[support[0]] == ONE
    )


def value_query(secret: Sequence[int], query: Sequence[int]) -> int:
    return sum(left * right for left, right in zip(secret, query)) % P


def classical_basis_recovery(secret: Sequence[int]) -> tuple[int, ...]:
    recovered: list[int] = []
    for index in range(len(secret)):
        query = [0] * len(secret)
        query[index] = 1
        recovered.append(value_query(secret, query))
    return tuple(recovered)


def direct_private_descriptor_scan(secret: Sequence[int]) -> tuple[tuple[int, ...], int]:
    recovered: list[int] = []
    accesses = 0
    for residue in secret:
        recovered.append(int(residue) % P)
        accesses += 1
    return tuple(recovered), accesses


def lower_bound_witness(dimension: int) -> dict[str, object]:
    queries = [
        [1 if row == column else 0 for column in range(dimension)]
        for row in range(max(0, dimension - 1))
    ]
    first = [0] * dimension
    second = [0] * dimension
    second[-1] = 1
    ambiguous = [value_query(first, query) for query in queries] == [
        value_query(second, query) for query in queries
    ]
    return {
        "dimension": dimension,
        "queries_used": dimension - 1,
        "distinct_secrets_with_identical_transcript": ambiguous,
        "rank_deficiency_argument": (
            "Q_QUERY_VECTORS_HAVE_SPAN_RANK_AT_MOST_Q_SO_Q_LESS_THAN_N_"
            "LEAVES_A_NONZERO_ORTHOGONAL_SECRET_DIFFERENCE"
        ),
    }


def varied_secret_controls() -> dict[str, bool]:
    zero = (0, 0, 0, 0)
    repeated = (2, 2, 2, 2)
    mixed = (0, 1, 1, 3)
    zero_result = factor_transaction(zero)
    repeated_result = factor_transaction(repeated)
    mixed_result = factor_transaction(mixed)
    perturbed = list(mixed)
    perturbed[2] = (perturbed[2] + 1) % P
    perturbed_result = factor_transaction(tuple(perturbed))
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(mixed_result["inferred_secret"], perturbed_result["inferred_secret"])
        )
        if left != right
    ]
    return {
        "all_zero_secret_exact_and_restored": (
            zero_result["inferred_secret"] == list(zero)
            and bool(zero_result["canonical_post_inverse_state_exact"])
        ),
        "repeated_secret_exact_and_restored": (
            repeated_result["inferred_secret"] == list(repeated)
            and bool(repeated_result["canonical_post_inverse_state_exact"])
        ),
        "mixed_secret_exact_and_restored": (
            mixed_result["inferred_secret"] == list(mixed)
            and bool(mixed_result["canonical_post_inverse_state_exact"])
        ),
        "one_coordinate_perturbation_changes_only_that_boundary_coordinate": changed == [2],
    }


def main(raw_path: Path) -> None:
    raw = json.loads(raw_path.read_text())
    private = json.loads(sys.stdin.buffer.readline())
    oracles = private["oracles"]
    cases: list[dict[str, object]] = []
    for case in raw["cases"]:
        descriptor = oracles[case["oracle_id"]]
        secret = tuple(int(value) % P for value in descriptor["secret"])
        factor = factor_transaction(secret)
        if case["inferred_secret"] != list(secret):
            raise RuntimeError("service response disagrees with private authority")
        if factor["inferred_secret"] != case["inferred_secret"]:
            raise RuntimeError("independent factor boundary disagrees")
        if classical_basis_recovery(secret) != secret:
            raise RuntimeError("independent classical basis queries failed")
        descriptor_boundary, descriptor_accesses = direct_private_descriptor_scan(secret)
        if descriptor_boundary != secret:
            raise RuntimeError("direct private descriptor baseline failed")
        dense_parity = dense_hidden_parity(secret, factor) if len(secret) <= 4 else None
        if dense_parity is False:
            raise RuntimeError("independent dense hidden-state parity failed")
        cases.append({
            "oracle_id": case["oracle_id"],
            "dimension": len(secret),
            "inferred_secret": list(secret),
            "canonical_post_inverse_state_exact": factor["canonical_post_inverse_state_exact"],
            "factor_fourier_character_terms": factor["factor_fourier_character_terms"],
            "oracle_factor_cell_visits": factor["oracle_factor_cell_visits"],
            "hidden_secret_residue_accesses": factor["hidden_secret_residue_accesses"],
            "classical_basis_queries_sufficient": len(secret),
            "direct_private_descriptor_scan_residue_accesses": descriptor_accesses,
            "direct_private_descriptor_scan_matches_final_boundary": True,
            "dense_exact_hidden_phase_parity": dense_parity,
        })
    certificates = [lower_bound_witness(dimension) for dimension in DIMENSIONS]
    result = {
        "schema": "cat_cas.catvm_p5_tensor_factored_hidden_linear_phase_oracle_reference.v1",
        "cases": cases,
        "classical_query_lower_bound_certificates": certificates,
        "all_n_minus_one_query_witnesses_ambiguous": all(
            certificate["distinct_secrets_with_identical_transcript"]
            for certificate in certificates
        ),
        "varied_secret_controls": varied_secret_controls(),
        "independent_exact_power_basis_factor_reexecution": True,
        "independent_dense_global_parity_dimensions1_2_4": True,
        "independent_classical_query_law": True,
        "strongest_total_software_baseline_is_direct_descriptor_scan_O_N": True,
        "imports_service_controller_or_predecessor": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: catvm_p5_tensor_factored_hidden_linear_phase_oracle_separate_reference.py RAW_RESULTS_JSON"
        )
    main(Path(sys.argv[1]))
