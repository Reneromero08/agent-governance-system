#!/usr/bin/env python3
"""Independent M245 arithmetic, quotient, rank, and baseline reference."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Iterable


P = 5
E = tuple[int, int, int, int]
ZERO: E = (0, 0, 0, 0)
ONE: E = (1, 0, 0, 0)
ORBITS = tuple((left, right) for left in range(P) for right in range(left, P))
ORBIT_INDEX = {pair: index for index, pair in enumerate(ORBITS)}
POINTS = tuple((left, right) for left in range(P) for right in range(P))


def add(left: E, right: E) -> E:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def scale(value: E, scalar: int) -> E:
    return tuple(scalar * item for item in value)  # type: ignore[return-value]


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


def root(exponent: int) -> E:
    exponent %= 5
    if exponent < 4:
        values = [0] * 4
        values[exponent] = 1
        return tuple(values)  # type: ignore[return-value]
    return (-1, -1, -1, -1)


def canonical(values: list[E], denominator_exponent: int) -> int:
    exponent = denominator_exponent
    while exponent and all(coefficient % 5 == 0 for value in values for coefficient in value):
        for index, value in enumerate(values):
            values[index] = tuple(coefficient // 5 for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def canonical_scalar(value: E, exponent: int) -> tuple[E, int]:
    values = [value]
    exponent = canonical(values, exponent)
    return values[0], exponent


def ordered_orbit(pair: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return (pair,) if pair[0] == pair[1] else (pair, (pair[1], pair[0]))


def phase(parameters: tuple[int, int, int], pair: tuple[int, int]) -> int:
    lam, quadratic, rung = parameters
    left, right = pair
    return (
        lam * (left**3 + right**3)
        + quadratic * (left**2 + right**2)
        + 2 * rung * left * right
    )


def digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    depth: int
    lambdas: tuple[int, ...]
    quadratics: tuple[int, ...]
    rungs: tuple[int, ...]
    couplings: tuple[int, ...]
    output_orbit: tuple[int, int]
    descriptor_digest: str

    def __post_init__(self) -> None:
        descriptor = (
            self.depth, self.lambdas, self.quadratics,
            self.rungs, self.couplings, self.output_orbit,
        )
        if (
            self.depth != 4
            or len(self.lambdas) != 4
            or len(self.quadratics) != 4
            or len(self.rungs) != 4
            or len(self.couplings) != 3
            or not all(self.lambdas)
            or not all(self.rungs)
            or not all(self.couplings)
            or self.output_orbit not in ORBIT_INDEX
            or self.descriptor_digest != digest(descriptor)
        ):
            raise ValueError("invalid independent M245 descriptor")

    @staticmethod
    def from_json(config: dict[str, Any]) -> "Spec":
        descriptor = (
            int(config["depth"]),
            tuple(int(value) % 5 for value in config["lambdas"]),
            tuple(int(value) % 5 for value in config["quadratics"]),
            tuple(int(value) % 5 for value in config["rungs"]),
            tuple(int(value) % 5 for value in config["couplings"]),
            tuple(sorted(int(value) % 5 for value in config["output_orbit"])),
        )
        return Spec(*descriptor, digest(descriptor))

    def parameters(self, index: int) -> tuple[tuple[int, int, int], int]:
        return (
            (self.lambdas[index], self.quadratics[index], self.rungs[index]),
            1 if index == 0 else self.couplings[index - 1],
        )


def orbit_transform(
    values: list[E], exponent: int, parameters: tuple[int, int, int], coupling: int,
    inverse: bool, first_specialized: bool = False,
) -> tuple[list[E], int, int]:
    result = [ZERO] * 15
    terms = 0
    if first_specialized:
        if values != [ONE] + [ZERO] * 14 or inverse:
            raise RuntimeError("invalid independent first specialization")
        for index, pair in enumerate(ORBITS):
            result[index] = root(phase(parameters, pair))
            terms += 1
    elif inverse:
        for input_index, input_pair in enumerate(ORBITS):
            accumulator = ZERO
            for output_index, output_pair in enumerate(ORBITS):
                for y in ordered_orbit(output_pair):
                    character = -phase(parameters, y) - 2 * coupling * (
                        y[0] * input_pair[0] + y[1] * input_pair[1]
                    )
                    accumulator = add(accumulator, mul(root(character), values[output_index]))
                    terms += 1
            result[input_index] = accumulator
    else:
        for output_index, output_pair in enumerate(ORBITS):
            accumulator = ZERO
            pvalue = phase(parameters, output_pair)
            for input_index, input_pair in enumerate(ORBITS):
                for x in ordered_orbit(input_pair):
                    character = pvalue + 2 * coupling * (
                        output_pair[0] * x[0] + output_pair[1] * x[1]
                    )
                    accumulator = add(accumulator, mul(root(character), values[input_index]))
                    terms += 1
            result[output_index] = accumulator
    exponent += 1
    exponent = canonical(result, exponent)
    return result, exponent, terms


def full25_transform(
    values: list[E], exponent: int, parameters: tuple[int, int, int], coupling: int,
) -> tuple[list[E], int, int]:
    first = [ZERO] * 25
    terms = 0
    for y0 in range(5):
        for x1 in range(5):
            accumulator = ZERO
            for x0 in range(5):
                accumulator = add(
                    accumulator,
                    mul(root(2 * coupling * y0 * x0), values[5 * x0 + x1]),
                )
                terms += 1
            first[5 * y0 + x1] = accumulator
    result = [ZERO] * 25
    for y0 in range(5):
        for y1 in range(5):
            accumulator = ZERO
            for x1 in range(5):
                accumulator = add(
                    accumulator,
                    mul(root(2 * coupling * y1 * x1), first[5 * y0 + x1]),
                )
                terms += 1
            result[5 * y0 + y1] = mul(root(phase(parameters, (y0, y1))), accumulator)
    exponent += 1
    exponent = canonical(result, exponent)
    return result, exponent, terms


def factorized_orbit_interior(
    values: list[E], exponent: int, parameters: tuple[int, int, int], coupling: int,
) -> tuple[list[E], int, int]:
    temporary = [ZERO] * 25
    terms = 0
    for y0 in range(5):
        for x1 in range(5):
            accumulator = ZERO
            for x0 in range(5):
                accumulator = add(
                    accumulator,
                    mul(root(2 * coupling * y0 * x0), values[ORBIT_INDEX[tuple(sorted((x0, x1)))]]),
                )
                terms += 1
            temporary[5 * y0 + x1] = accumulator
    # The complete first-axis transform no longer needs the input values, so
    # reuse that 15-cell backing for the outputs while the 25-cell temporary
    # remains live.  The implemented baseline peak is 40 field cells.
    result = values
    for output_index, (y0, y1) in enumerate(ORBITS):
        accumulator = ZERO
        for x1 in range(5):
            accumulator = add(
                accumulator,
                mul(root(2 * coupling * y1 * x1), temporary[5 * y0 + x1]),
            )
            terms += 1
        result[output_index] = mul(root(phase(parameters, (y0, y1))), accumulator)
    exponent += 1
    exponent = canonical(result, exponent)
    return result, exponent, terms


def endpoint_baseline(spec: Spec) -> tuple[E, int, int]:
    values = [ONE] + [ZERO] * 14
    exponent = 0
    parameters, coupling = spec.parameters(0)
    values, exponent, terms = orbit_transform(
        values, exponent, parameters, coupling, False, first_specialized=True
    )
    total_terms = terms
    for index in (1, 2):
        parameters, coupling = spec.parameters(index)
        values, exponent, terms = factorized_orbit_interior(
            values, exponent, parameters, coupling
        )
        total_terms += terms
    parameters, coupling = spec.parameters(3)
    output = spec.output_orbit
    accumulator = ZERO
    for input_index, input_pair in enumerate(ORBITS):
        for x in ordered_orbit(input_pair):
            character = phase(parameters, output) + 2 * coupling * (
                output[0] * x[0] + output[1] * x[1]
            )
            accumulator = add(accumulator, mul(root(character), values[input_index]))
            total_terms += 1
    return canonical_scalar(accumulator, exponent + 1) + (total_terms,)


class ReferenceCarrier:
    def __init__(self) -> None:
        self.cells = [ZERO] * 15
        self.scratch = [ZERO] * 15
        self.last_restored_generation = 0

    def run(self, spec: Spec, generation: int) -> dict[str, object]:
        if self.cells != [ZERO] * 15 or self.scratch != [ZERO] * 15:
            raise RuntimeError("independent carrier not canonical")
        if generation != self.last_restored_generation + 1:
            raise RuntimeError("independent stale generation")
        cell_id, scratch_id = id(self.cells), id(self.scratch)
        self.cells[0] = ONE
        exponent = 0
        forward_terms = 0
        for index in range(4):
            parameters, coupling = spec.parameters(index)
            next_cells, exponent, terms = orbit_transform(
                self.cells, exponent, parameters, coupling, False,
                first_specialized=index == 0,
            )
            self.cells[:] = next_cells
            forward_terms += terms
        retained = self.cells[ORBIT_INDEX[spec.output_orbit]]
        retained_exponent = exponent
        inverse_terms = 0
        for index in range(3, -1, -1):
            parameters, coupling = spec.parameters(index)
            next_cells, exponent, terms = orbit_transform(
                self.cells, exponent, parameters, coupling, True
            )
            self.cells[:] = next_cells
            inverse_terms += terms
        restored = self.cells == [ONE] + [ZERO] * 14 and exponent == 0
        self.cells[0] = ZERO
        if not restored or self.cells != [ZERO] * 15:
            raise RuntimeError("independent exact restoration failed")
        self.last_restored_generation = generation
        return {
            "final_amplitude": {
                "numerator": list(retained),
                "denominator_power5": retained_exponent,
            },
            "generation": generation,
            "same_cell_backing": id(self.cells) == cell_id,
            "same_scratch_backing": id(self.scratch) == scratch_id,
            "canonical_after_restoration": self.cells == [ZERO] * 15,
            "baseline_reload_used": False,
            "forward_character_terms": forward_terms,
            "inverse_character_terms": inverse_terms,
        }


def root5(prime: int) -> int:
    for candidate in range(2, prime):
        if pow(candidate, 5, prime) == 1 and candidate != 1:
            return candidate
    raise RuntimeError("split prime lacks fifth root")


def add_basis(basis: list[tuple[int, list[int]]], vector: list[int], prime: int) -> bool:
    row = vector[:]
    for pivot, existing in basis:
        if row[pivot]:
            coefficient = row[pivot]
            row = [(left - coefficient * right) % prime for left, right in zip(row, existing)]
    try:
        pivot = next(index for index, value in enumerate(row) if value)
    except StopIteration:
        return False
    inverse = pow(row[pivot], -1, prime)
    row = [value * inverse % prime for value in row]
    for index, (old_pivot, existing) in enumerate(basis):
        if existing[pivot]:
            coefficient = existing[pivot]
            basis[index] = (
                old_pivot,
                [(left - coefficient * right) % prime for left, right in zip(existing, row)],
            )
    basis.append((pivot, row))
    basis.sort()
    return True


def rank(rows: Iterable[list[int]], prime: int) -> int:
    basis: list[tuple[int, list[int]]] = []
    for row in rows:
        add_basis(basis, row, prime)
    return len(basis)


def orbit_matrix_mod(
    parameters: tuple[int, int, int], coupling: int, prime: int, zeta: int
) -> list[list[int]]:
    inverse5 = pow(5, -1, prime)
    matrix: list[list[int]] = []
    for output in ORBITS:
        row: list[int] = []
        pvalue = phase(parameters, output)
        for input_pair in ORBITS:
            total = sum(
                pow(
                    zeta,
                    (pvalue + 2 * coupling * (output[0] * x[0] + output[1] * x[1])) % 5,
                    prime,
                )
                for x in ordered_orbit(input_pair)
            )
            row.append(inverse5 * total % prime)
        matrix.append(row)
    return matrix


def matvec(matrix: list[list[int]], vector: list[int], prime: int) -> list[int]:
    return [sum(left * right for left, right in zip(row, vector)) % prime for row in matrix]


def rowmat(vector: list[int], matrix: list[list[int]], prime: int) -> list[int]:
    return [
        sum(vector[index] * matrix[index][column] for index in range(len(vector))) % prime
        for column in range(len(vector))
    ]


def matmul(left: list[list[int]], right: list[list[int]], prime: int) -> list[list[int]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(len(right))) % prime
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


BROKEN_GATES = {
    "A": (((1, 0), (0, 1)), (1, 2), (0, 1)),
    "B": (((1, 1), (0, 1)), (2, 1), (1, 0)),
    "D": (((1, 0), (1, 1)), (3, 1), (2, 4)),
}


def labelled_matrix_mod(
    descriptor: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int], tuple[int, int]],
    prime: int, zeta: int, inverse: bool,
) -> list[list[int]]:
    coupling, lambdas, quadratics = descriptor
    inverse5 = pow(5, -1, prime)
    matrix: list[list[int]] = []
    for output in POINTS:
        row: list[int] = []
        for input_pair in POINTS:
            if inverse:
                x, y = output, input_pair
                character = -(
                    lambdas[0] * y[0] ** 3 + lambdas[1] * y[1] ** 3
                    + quadratics[0] * y[0] ** 2 + quadratics[1] * y[1] ** 2
                    + 2 * (
                        y[0] * (coupling[0][0] * x[0] + coupling[0][1] * x[1])
                        + y[1] * (coupling[1][0] * x[0] + coupling[1][1] * x[1])
                    )
                )
            else:
                x, y = input_pair, output
                character = (
                    lambdas[0] * y[0] ** 3 + lambdas[1] * y[1] ** 3
                    + quadratics[0] * y[0] ** 2 + quadratics[1] * y[1] ** 2
                    + 2 * (
                        y[0] * (coupling[0][0] * x[0] + coupling[0][1] * x[1])
                        + y[1] * (coupling[1][0] * x[0] + coupling[1][1] * x[1])
                    )
                )
            row.append(inverse5 * pow(zeta, character % 5, prime) % prime)
        matrix.append(row)
    return matrix


def closure_rank(matrices: dict[str, list[list[int]]], row_side: bool, prime: int) -> tuple[int, str]:
    initial = [1] + [0] * (len(next(iter(matrices.values()))) - 1)
    queue: deque[tuple[list[int], str]] = deque([(initial, "I")])
    basis: list[tuple[int, list[int]]] = []
    words: list[str] = []
    while queue and len(basis) < len(initial):
        vector, word = queue.popleft()
        if not add_basis(basis, vector, prime):
            continue
        words.append(word)
        for name, matrix in matrices.items():
            next_vector = rowmat(vector, matrix, prime) if row_side else matvec(matrix, vector, prime)
            queue.append((next_vector, word + name))
    return len(basis), hashlib.sha256("|".join(words).encode()).hexdigest()


def rank_certificates() -> dict[str, object]:
    certificates: dict[str, object] = {}
    for prime in (41, 61):
        zeta = root5(prime)
        reach_candidates = []
        for lam in range(1, 5):
            for quadratic in range(5):
                for rung in range(1, 5):
                    reach_candidates.append(
                        [
                            pow(zeta, phase((lam, quadratic, rung), pair) % 5, prime)
                            * pow(5, -1, prime) % prime
                            for pair in ORBITS
                        ]
                    )
        reach_basis: list[tuple[int, list[int]]] = []
        selected: list[int] = []
        for index, vector in enumerate(reach_candidates):
            if add_basis(reach_basis, vector, prime):
                selected.append(index)
            if len(reach_basis) == 15:
                break
        suffix = orbit_matrix_mod((2, 1, 3), 2, prime, zeta)
        reach_matrix = [[reach_candidates[column][row] for column in selected] for row in range(15)]
        hankel = matmul(suffix, reach_matrix, prime)

        labelled: dict[str, list[list[int]]] = {}
        for name, descriptor in BROKEN_GATES.items():
            labelled[name] = labelled_matrix_mod(descriptor, prime, zeta, False)
            labelled[name.lower()] = labelled_matrix_mod(descriptor, prime, zeta, True)
        broken_reach, broken_reach_digest = closure_rank(labelled, False, prime)
        broken_observe, broken_observe_digest = closure_rank(labelled, True, prime)

        first_layer = [
            [pow(zeta, phase((1, 0, 1), (left, right)) % 5, prime) for right in range(5)]
            for left in range(5)
        ]
        certificates[str(prime)] = {
            "symmetric_reachability_rank": len(reach_basis),
            "symmetric_observability_rank": rank(suffix, prime),
            "symmetric_hankel_rank": rank(hankel, prime),
            "selected_prefix_count": len(selected),
            "selected_prefix_indices_digest": hashlib.sha256(
                json.dumps(selected, separators=(",", ":")).encode()
            ).hexdigest(),
            "first_rung_labelled_matrix_rank": rank(first_layer, prime),
            "declared_exchange_broken_three_gate_reachability_rank": broken_reach,
            "declared_exchange_broken_three_gate_observability_rank": broken_observe,
            "declared_exchange_broken_three_gate_reachability_words_digest": broken_reach_digest,
            "declared_exchange_broken_three_gate_observability_words_digest": broken_observe_digest,
        }
    return certificates


def exact_labelled_parity(spec: Spec) -> tuple[bool, dict[str, object]]:
    orbit_values = [ONE] + [ZERO] * 14
    labelled = [ZERO] * 25
    labelled[0] = ONE
    orbit_exponent = labelled_exponent = 0
    stage_parity = True
    labelled_terms = 0
    for index in range(4):
        parameters, coupling = spec.parameters(index)
        orbit_values, orbit_exponent, _ = orbit_transform(
            orbit_values, orbit_exponent, parameters, coupling, False,
            first_specialized=index == 0,
        )
        labelled, labelled_exponent, terms = full25_transform(
            labelled, labelled_exponent, parameters, coupling
        )
        labelled_terms += terms
        for point in POINTS:
            orbit_index = ORBIT_INDEX[tuple(sorted(point))]
            if labelled[5 * point[0] + point[1]] != orbit_values[orbit_index]:
                stage_parity = False
        stage_parity = stage_parity and labelled_exponent == orbit_exponent
    boundary = labelled[5 * spec.output_orbit[0] + spec.output_orbit[1]]
    return stage_parity, {
        "final_amplitude": {"numerator": list(boundary), "denominator_power5": labelled_exponent},
        "verifier_labelled_character_terms": labelled_terms,
        "verifier_labelled_state_field_cells": 25,
    }


def main() -> None:
    config = json.loads(sys.stdin.readline())
    specs = {name: Spec.from_json(value) for name, value in config["oracles"].items()}
    carrier = ReferenceCarrier()
    fresh = ReferenceCarrier()
    cases = []
    primary = carrier.run(specs["primary"], 1)
    primary["run_kind"] = "PRIMARY"
    cases.append(primary)
    reuse = carrier.run(specs["reuse"], 2)
    reuse["run_kind"] = "RESTORED_REUSE"
    cases.append(reuse)
    fresh_case = fresh.run(specs["reuse_fresh"], 1)
    fresh_case["run_kind"] = "FRESH_REUSE_REFERENCE"
    cases.append(fresh_case)

    factorized: dict[str, object] = {}
    labelled: dict[str, object] = {}
    for name in ("primary", "reuse", "reuse_fresh"):
        value, exponent, terms = endpoint_baseline(specs[name])
        factorized[name] = {
            "final_amplitude": {"numerator": list(value), "denominator_power5": exponent},
            "character_terms": terms,
        }
        parity, full = exact_labelled_parity(specs[name])
        labelled[name] = {"all_stages_match_orbit_quotient": parity, **full}

    baseline_primary = factorized["primary"]["final_amplitude"]
    perturbations: dict[str, bool] = {}
    base_json = config["oracles"]["primary"]
    for label, key, index in (
        ("lambda", "lambdas", 0),
        ("rung", "rungs", 1),
        ("coupling", "couplings", 1),
    ):
        mutated = {name: (list(value) if isinstance(value, list) else value) for name, value in base_json.items()}
        mutated[key][index] = int(mutated[key][index]) % 4 + 1
        mutated_spec = Spec.from_json(mutated)
        value, exponent, _ = endpoint_baseline(mutated_spec)
        perturbations[f"{label}_perturbation_changes_boundary"] = (
            {"numerator": list(value), "denominator_power5": exponent} != baseline_primary
        )
    mutated = {name: (list(value) if isinstance(value, list) else value) for name, value in base_json.items()}
    mutated["output_orbit"] = [0, 4]
    value, exponent, _ = endpoint_baseline(Spec.from_json(mutated))
    perturbations["output_selector_changes_boundary"] = (
        {"numerator": list(value), "denominator_power5": exponent} != baseline_primary
    )

    broken_left = root(1**3 + 2 * 2**3 + 2 * 1 * 1 * 2)
    broken_right = root(2**3 + 2 * 1**3 + 2 * 1 * 2 * 1)
    ranks = rank_certificates()
    controls = {
        "primary_service_boundary_matches_independent_reference": (
            cases[0]["final_amplitude"] == factorized["primary"]["final_amplitude"]
        ),
        "reuse_service_boundary_matches_fresh_reference": (
            reuse["final_amplitude"] == fresh_case["final_amplitude"]
        ),
        "all_factorized_baseline_terms_equal440": all(
            item["character_terms"] == 440 for item in factorized.values()
        ),
        "full25_labelled_recurrence_matches15_orbit_quotient_at_every_stage": all(
            item["all_stages_match_orbit_quotient"] for item in labelled.values()
        ),
        "all_rank15_certificates_hold": all(
            item["symmetric_reachability_rank"] == 15
            and item["symmetric_observability_rank"] == 15
            and item["symmetric_hankel_rank"] == 15
            and item["selected_prefix_count"] == 15
            for item in ranks.values()
        ),
        "first_rung_rejects_matrix_bond_rank_below5": all(
            item["first_rung_labelled_matrix_rank"] == 5 for item in ranks.values()
        ),
        "declared_three_gate_exchange_broken_alphabet_has_rank25_reachable_observable_ceiling": all(
            item["declared_exchange_broken_three_gate_reachability_rank"] == 25
            and item["declared_exchange_broken_three_gate_observability_rank"] == 25
            for item in ranks.values()
        ),
        "asymmetric_lambda_breaks_swapped_amplitude_equality": broken_left != broken_right,
        "exact_restoration_and_same_backing": all(
            case["canonical_after_restoration"]
            and case["same_cell_backing"]
            and case["same_scratch_backing"]
            and not case["baseline_reload_used"]
            for case in cases
        ),
        "generation_sequence_primary1_reuse2_fresh1": [
            case["generation"] for case in cases
        ] == [1, 2, 1],
        **perturbations,
    }
    if not all(controls.values()):
        raise RuntimeError(f"independent M245 control failure: {controls}")

    output = {
        "result": "PASS_SEPARATE_REFERENCE_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_QUOTIENT",
        "cases": cases,
        "factorized_endpoint_baseline": factorized,
        "full25_labelled_parity": labelled,
        "rank_certificates": ranks,
        "declared_exchange_broken_three_gate_alphabet": {
            name: {
                "coupling_matrix": [list(row) for row in descriptor[0]],
                "lambdas": list(descriptor[1]),
                "quadratics": list(descriptor[2]),
            }
            for name, descriptor in BROKEN_GATES.items()
        },
        "controls": controls,
        "baseline": {
            "strongest_implemented": (
                "ENDPOINT_SPECIALIZED_SYMMETRIC_TWO_RAIL_FACTORIZED_FIVE_BY_FIVE_"
                "TRANSFORM_WITH15_ORBIT_RESIDENT_AND25_TEMPORARY_CELLS"
            ),
            "character_terms": 440,
            "explicit_depth_path_enumeration_used": False,
            "classical_smaller_in_work_than_catvm_forward": True,
            "total_advantage_established": False,
        },
        "imports_production_service_client_or_m237": False,
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
