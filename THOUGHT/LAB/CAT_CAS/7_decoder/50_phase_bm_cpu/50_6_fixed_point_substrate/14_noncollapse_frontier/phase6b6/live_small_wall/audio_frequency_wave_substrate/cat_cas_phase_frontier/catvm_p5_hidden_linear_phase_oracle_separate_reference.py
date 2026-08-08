#!/usr/bin/env python3
"""Independent exact oracle and query-law checker for M241."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


P = 5
E = tuple[int, int, int, int]
Z: E = (0, 0, 0, 0)
O: E = (1, 0, 0, 0)
S: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def mul(left: E, right: E) -> E:
    raw = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            raw[i + j] += a * b
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


def normalize(values: list[E], exponent: int) -> int:
    while exponent and all(coefficient % P == 0 for value in values for coefficient in value):
        for index, value in enumerate(values):
            values[index] = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def flat_index(state: Sequence[int]) -> int:
    result = 0
    for coordinate in state:
        result = P * result + coordinate
    return result


def coordinates(index: int, dimension: int) -> tuple[int, ...]:
    result = [0] * dimension
    for position in range(dimension - 1, -1, -1):
        result[position] = index % P
        index //= P
    return tuple(result)


def commitment(values: Sequence[E], exponent: int) -> str:
    payload = {"denominator_power5": exponent, "numerators": [list(value) for value in values]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def secret_commitment(secret: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps({"p": P, "secret": list(secret)}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fourier(values: list[E], exponent: int, dimension: int, wire: int, direction: int) -> tuple[list[E], int, int]:
    result = [Z] * len(values)
    terms = 0
    for destination in range(len(values)):
        output_state = list(coordinates(destination, dimension))
        output = output_state[wire]
        total = Z
        for source in range(P):
            source_state = output_state.copy()
            source_state[wire] = source
            total = add(total, mul(root(direction * source * output), values[flat_index(source_state)]))
            terms += 1
        result[destination] = mul(S, total)
    exponent += 1
    exponent = normalize(result, exponent)
    return result, exponent, terms


def phase_oracle(values: list[E], dimension: int, secret: Sequence[int], direction: int) -> int:
    visits = 0
    for location in range(len(values)):
        state = coordinates(location, dimension)
        phase = sum(secret[index] * state[index] for index in range(dimension))
        values[location] = mul(root(direction * phase), values[location])
        visits += 1
    return visits


def direct_transaction(secret: Sequence[int]) -> dict[str, object]:
    dimension = len(secret)
    values = [Z] * (P**dimension); values[0] = O
    exponent = 0; fourier_terms = 0
    for wire in range(dimension):
        values, exponent, terms = fourier(values, exponent, dimension, wire, 1)
        fourier_terms += terms
    forward_visits = phase_oracle(values, dimension, secret, 1)
    for wire in range(dimension):
        values, exponent, terms = fourier(values, exponent, dimension, wire, -1)
        fourier_terms += terms
    support = [location for location, value in enumerate(values) if value != Z]
    inferred = coordinates(support[0], dimension) if len(support) == 1 else ()
    boundary_commitment = commitment(values, exponent)
    for wire in reversed(range(dimension)):
        values, exponent, terms = fourier(values, exponent, dimension, wire, 1)
        fourier_terms += terms
    inverse_visits = phase_oracle(values, dimension, secret, -1)
    for wire in reversed(range(dimension)):
        values, exponent, terms = fourier(values, exponent, dimension, wire, -1)
        fourier_terms += terms
    restored = exponent == 0 and values[0] == O and all(value == Z for value in values[1:])
    return {
        "inferred_secret": list(inferred),
        "secret_commitment": secret_commitment(inferred),
        "final_basis_state_commitment": boundary_commitment,
        "canonical_post_inverse_state_exact": restored,
        "fourier_character_terms": fourier_terms,
        "forward_oracle_cell_visits": forward_visits,
        "inverse_oracle_cell_visits": inverse_visits,
    }


def value_query(secret: Sequence[int], query: Sequence[int]) -> int:
    return sum(left * right for left, right in zip(secret, query)) % P


def classical_basis_recovery(secret: Sequence[int]) -> tuple[int, ...]:
    dimension = len(secret)
    recovered = []
    for index in range(dimension):
        query = [0] * dimension; query[index] = 1
        recovered.append(value_query(secret, query))
    return tuple(recovered)


def lower_bound_witness(dimension: int) -> dict[str, object]:
    if dimension == 1:
        queries: list[list[int]] = []
    else:
        queries = [[1 if row == column else 0 for column in range(dimension)] for row in range(dimension - 1)]
    first = [0] * dimension
    second = [0] * dimension; second[-1] = 1
    same = [value_query(first, query) for query in queries] == [value_query(second, query) for query in queries]
    return {
        "queries_used": dimension - 1,
        "distinct_secrets_with_identical_transcript": same,
        "rank_deficiency_argument": "Q_QUERY_VECTORS_SPAN_RANK_AT_MOST_Q_SO_Q_LESS_THAN_N_HAS_NONZERO_ORTHOGONAL_SECRET_DIFFERENCE",
    }


def main(raw_path: Path) -> None:
    raw = json.loads(raw_path.read_text())
    private = json.loads(sys.stdin.buffer.readline())
    oracles = private["oracles"]
    cases: list[dict[str, object]] = []
    for case in raw["cases"]:
        descriptor = oracles[case["oracle_id"]]
        secret = tuple(int(value) % P for value in descriptor["secret"])
        direct = direct_transaction(secret)
        if case["inferred_secret"] != list(secret):
            raise RuntimeError("service response disagrees with private oracle authority")
        if direct["inferred_secret"] != case["inferred_secret"]:
            raise RuntimeError("independent coherent-query boundary disagrees")
        if direct["final_basis_state_commitment"] != case["final_basis_state_commitment"]:
            raise RuntimeError("independent final state commitment disagrees")
        if direct["secret_commitment"] != case["boundary_commitment"]:
            raise RuntimeError("independent secret commitment disagrees")
        if classical_basis_recovery(secret) != secret:
            raise RuntimeError("classical basis queries failed")
        cases.append({
            "oracle_id": case["oracle_id"],
            "dimension": len(secret),
            "inferred_secret": list(secret),
            "boundary_commitment": direct["secret_commitment"],
            "final_basis_state_commitment": direct["final_basis_state_commitment"],
            "canonical_post_inverse_state_exact": direct["canonical_post_inverse_state_exact"],
            "classical_basis_queries_sufficient": len(secret),
            "direct_phase_software": direct,
        })
    result = {
        "schema": "cat_cas.catvm_p5_hidden_linear_phase_oracle_reference.v1",
        "cases": cases,
        "classical_query_lower_bound_certificates": [lower_bound_witness(dimension) for dimension in (1, 2, 3, 4)],
        "all_n_minus_one_query_witnesses_ambiguous": all(
            lower_bound_witness(dimension)["distinct_secrets_with_identical_transcript"]
            for dimension in (1, 2, 3, 4)
        ),
        "independent_exact_power_basis_reexecution": True,
        "independent_classical_query_law": True,
        "imports_service_or_controller": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: catvm_p5_hidden_linear_phase_oracle_separate_reference.py RAW_RESULTS_JSON")
    main(Path(sys.argv[1]))
