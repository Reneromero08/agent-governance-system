#!/usr/bin/env python3
"""Independent M246 occupation arithmetic, transaction, and rank oracle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable


P = 5
DEPTH = 3
RAILS = (2, 3, 4, 6)
ZERO = (0, 0, 0, 0)
ONE = (1, 0, 0, 0)
SQRT5 = (-1, 0, -2, -2)
E = tuple[int, int, int, int]
GATE_ALPHABET = {
    "A": (1, 0, 1, 1),
    "B": (2, 1, 2, 2),
    "C": (3, 2, 1, 3),
}


def add(left: E, right: E) -> E:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


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
        result = [0] * 4
        result[exponent] = 1
        return tuple(result)  # type: ignore[return-value]
    return (-1, -1, -1, -1)


def canonical(values: list[E], exponent: int) -> int:
    while exponent and all(coordinate % 5 == 0 for value in values for coordinate in value):
        for index, value in enumerate(values):
            values[index] = tuple(coordinate // 5 for coordinate in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


@lru_cache(maxsize=None)
def occupations(total: int) -> tuple[tuple[int, int, int, int, int], ...]:
    result: list[tuple[int, int, int, int, int]] = []

    def rec(remaining: int, slot: int, prefix: tuple[int, ...]) -> None:
        if slot == 4:
            result.append(tuple(prefix + (remaining,)))  # type: ignore[arg-type]
            return
        for value in range(remaining + 1):
            rec(remaining - value, slot + 1, prefix + (value,))

    rec(total, 0, ())
    return tuple(result)


def phase(parameters: tuple[int, int, int], occupation: tuple[int, ...]) -> int:
    lam, quadratic, rung = parameters
    s1 = sum(symbol * occupation[symbol] for symbol in range(P))
    s2 = sum(symbol * symbol * occupation[symbol] for symbol in range(P))
    s3 = sum(symbol**3 * occupation[symbol] for symbol in range(P))
    return (lam * s3 + quadratic * s2 + rung * (s1 * s1 - s2)) % P


def descriptor_digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    rails: int
    depth: int
    lambdas: tuple[int, ...]
    quadratics: tuple[int, ...]
    rungs: tuple[int, ...]
    couplings: tuple[int, ...]
    output: tuple[int, ...]
    carrier_id: str
    descriptor_digest: str

    @staticmethod
    def from_json(config: dict[str, Any]) -> "Spec":
        descriptor: tuple[object, ...] = (
            int(config["rails"]), int(config["depth"]),
            tuple(int(value) % P for value in config["lambdas"]),
            tuple(int(value) % P for value in config["quadratics"]),
            tuple(int(value) % P for value in config["rungs"]),
            tuple(int(value) % P for value in config["couplings"]),
            tuple(int(value) for value in config["output_occupation"]),
        )
        n, depth, lambdas, quadratics, rungs, couplings, output = descriptor
        if not (
            n in RAILS and depth == DEPTH
            and len(lambdas) == len(quadratics) == len(rungs) == DEPTH
            and len(couplings) == DEPTH - 1 and len(output) == P
            and sum(output) == n and all(value >= 0 for value in output)
            and all(lambdas) and all(rungs) and all(couplings)
            and tuple(output) in occupations(n)
            and not any(key in config for key in ("rail_lambdas", "labelled_output"))
        ):
            raise ValueError("invalid independent M246 descriptor")
        return Spec(
            n, depth, lambdas, quadratics, rungs, couplings, output,
            str(config["carrier_id"]), descriptor_digest(descriptor),
        )

    def module(self, index: int) -> tuple[tuple[int, int, int], int]:
        return (
            (self.lambdas[index], self.quadratics[index], self.rungs[index]),
            1 if index == 0 else self.couplings[index - 1],
        )


def kernel_row_exact(
    outer: tuple[int, ...], beta: int, wrong_multiplicity: bool = False
) -> tuple[dict[tuple[int, ...], E], int]:
    polynomial: dict[tuple[int, ...], E] = {(0, 0, 0, 0, 0): ONE}
    terms = 0
    for output_symbol, count in enumerate(outer):
        for _ in range(count):
            updated: dict[tuple[int, ...], E] = {}
            for current, coefficient in polynomial.items():
                for input_symbol in range(P):
                    target = list(current)
                    target[input_symbol] += 1
                    key = tuple(target)
                    contribution = mul(
                        coefficient, root(2 * beta * input_symbol * output_symbol)
                    )
                    if wrong_multiplicity:
                        contribution = root(2 * beta * input_symbol * output_symbol)
                    updated[key] = add(updated.get(key, ZERO), contribution)
                    terms += 1
            polynomial = updated
    return polynomial, terms


def normalize(values: list[E], exponent: int, n: int, omit_sqrt: bool = False) -> int:
    if n % 2 and not omit_sqrt:
        for index, value in enumerate(values):
            values[index] = mul(SQRT5, value)
    exponent += (n + 1) // 2
    return canonical(values, exponent)


def transform(
    values: list[E], exponent: int, n: int,
    parameters: tuple[int, int, int], beta: int, inverse: bool,
    wrong_multiplicity: bool = False, omit_sqrt: bool = False,
) -> tuple[list[E], int, int, int]:
    plan = occupations(n)
    result = [ZERO] * len(plan)
    coefficient_terms = 0
    dot_terms = 0
    if inverse:
        for input_index, input_occupation in enumerate(plan):
            row, built = kernel_row_exact(input_occupation, -beta, wrong_multiplicity)
            coefficient_terms += built
            accumulator = ZERO
            for output_index, output in enumerate(plan):
                phased = mul(root(-phase(parameters, output)), values[output_index])
                accumulator = add(accumulator, mul(row[output], phased))
                dot_terms += 1
            result[input_index] = accumulator
    else:
        for output_index, output in enumerate(plan):
            row, built = kernel_row_exact(output, beta, wrong_multiplicity)
            coefficient_terms += built
            accumulator = ZERO
            for input_index, input_occupation in enumerate(plan):
                accumulator = add(accumulator, mul(row[input_occupation], values[input_index]))
                dot_terms += 1
            result[output_index] = mul(root(phase(parameters, output)), accumulator)
    exponent = normalize(result, exponent, n, omit_sqrt)
    return result, exponent, coefficient_terms, dot_terms


def first_layer(spec: Spec) -> tuple[list[E], int]:
    values = [root(phase(spec.module(0)[0], output)) for output in occupations(spec.rails)]
    return values, normalize(values, 0, spec.rails)


class ReferenceCarrier:
    def __init__(self, n: int) -> None:
        self.n = n
        self.width = len(occupations(n))
        self.cells = [ZERO] * self.width
        self.scratch = [ZERO] * self.width
        self.row = [ZERO] * self.width
        self.last_generation = 0
        self.leased = False
        self.descriptor = ""

    def canonical_state(self) -> bool:
        return (
            self.cells == [ZERO] * self.width
            and self.scratch == [ZERO] * self.width
            and self.row == [ZERO] * self.width
            and not self.leased and self.descriptor == ""
        )

    def run(self, spec: Spec, generation: int) -> dict[str, object]:
        if (
            self.leased or not self.canonical_state()
            or generation != self.last_generation + 1 or spec.rails != self.n
        ):
            raise RuntimeError("independent M246 lease rejected")
        self.leased = True
        self.descriptor = spec.descriptor_digest
        cell_id, scratch_id, row_id = id(self.cells), id(self.scratch), id(self.row)
        delta = occupations(self.n).index((self.n, 0, 0, 0, 0))
        self.cells[delta] = ONE
        exponent = 0
        forward_coeff = forward_dot = inverse_coeff = inverse_dot = 0
        for index in range(DEPTH):
            if index == 0:
                self.scratch[:], exponent = first_layer(spec)
            else:
                self.scratch[:], exponent, built, dots = transform(
                    self.cells, exponent, self.n, *spec.module(index), False
                )
                forward_coeff += built
                forward_dot += dots
            self.cells[:], self.scratch[:] = self.scratch, [ZERO] * self.width
        boundary = self.cells[occupations(self.n).index(spec.output)]
        boundary_exponent = exponent
        for index in range(DEPTH - 1, -1, -1):
            self.scratch[:], exponent, built, dots = transform(
                self.cells, exponent, self.n, *spec.module(index), True
            )
            inverse_coeff += built
            inverse_dot += dots
            self.cells[:], self.scratch[:] = self.scratch, [ZERO] * self.width
        expected = [ZERO] * self.width
        expected[delta] = ONE
        if self.cells != expected or exponent != 0 or any(value != ZERO for value in self.row):
            raise RuntimeError("independent M246 restoration failed")
        self.cells[delta] = ZERO
        self.leased = False
        self.descriptor = ""
        self.last_generation = generation
        if not self.canonical_state():
            raise RuntimeError("independent M246 canonical release failed")
        return {
            "rails": self.n,
            "occupation_dimension": self.width,
            "final_amplitude": {
                "numerator": list(boundary),
                "denominator_power5": boundary_exponent,
            },
            "generation": self.last_generation,
            "same_message_backing": id(self.cells) == cell_id,
            "same_output_scratch_backing": id(self.scratch) == scratch_id,
            "same_coefficient_row_backing": id(self.row) == row_id,
            "canonical_after_restoration": self.canonical_state(),
            "baseline_reload_used": False,
            "forward_kernel_coefficient_terms": forward_coeff,
            "inverse_kernel_coefficient_terms": inverse_coeff,
            "forward_orbit_dot_terms": forward_dot,
            "inverse_orbit_dot_terms": inverse_dot,
        }


def endpoint_baseline(spec: Spec) -> dict[str, object]:
    n = spec.rails
    values, exponent = first_layer(spec)
    values, exponent, interior_coeff, interior_dot = transform(
        values, exponent, n, *spec.module(1), False
    )
    output = spec.output
    row, final_coeff = kernel_row_exact(output, spec.module(2)[1])
    accumulator = ZERO
    for input_index, input_occupation in enumerate(occupations(n)):
        accumulator = add(accumulator, mul(row[input_occupation], values[input_index]))
    boundary = mul(root(phase(spec.module(2)[0], output)), accumulator)
    boundary_values = [boundary]
    boundary_exponent = normalize(boundary_values, exponent, n)
    return {
        "final_amplitude": {
            "numerator": list(boundary_values[0]),
            "denominator_power5": boundary_exponent,
        },
        "first_layer_direct_phase_terms": len(occupations(n)),
        "interior_kernel_coefficient_terms": interior_coeff,
        "interior_orbit_dot_terms": interior_dot,
        "final_row_kernel_coefficient_terms": final_coeff,
        "final_row_orbit_dot_terms": len(occupations(n)),
        "resident_message_field_cells": len(occupations(n)),
        "output_scratch_field_cells": len(occupations(n)),
        "coefficient_row_field_cells": len(occupations(n)),
        "inverse_or_restoration_work": 0,
    }


def labelled_parity(spec: Spec) -> bool:
    n = spec.rails
    assignments = tuple(itertools.product(range(P), repeat=n))
    values: dict[tuple[int, ...], E] = {assignment: ZERO for assignment in assignments}
    values[(0,) * n] = ONE
    exponent = 0
    for module_index in range(DEPTH):
        parameters, beta = spec.module(module_index)
        updated: dict[tuple[int, ...], E] = {}
        for output in assignments:
            output_occ = tuple(output.count(symbol) for symbol in range(P))
            accumulator = ZERO
            for input_assignment, amplitude in values.items():
                character = phase(parameters, output_occ) + 2 * beta * sum(
                    output[index] * input_assignment[index] for index in range(n)
                )
                accumulator = add(accumulator, mul(root(character), amplitude))
            updated[output] = accumulator
        vector = list(updated.values())
        exponent = normalize(vector, exponent, n)
        values = {key: vector[index] for index, key in enumerate(updated)}
    carrier = ReferenceCarrier(n)
    reference_case = carrier.run(spec, 1)
    representative = tuple(
        symbol for symbol, count in enumerate(spec.output) for _ in range(count)
    )
    return (
        values[representative] == tuple(reference_case["final_amplitude"]["numerator"])
        and exponent == reference_case["final_amplitude"]["denominator_power5"]
        and all(
            values[assignment] == values[representative]
            for assignment in assignments
            if tuple(assignment.count(symbol) for symbol in range(P)) == spec.output
        )
    )


def primitive_root(prime: int) -> int:
    factors: list[int] = []
    value = prime - 1
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    raise RuntimeError("primitive root missing")


def kernel_row_mod(outer: tuple[int, ...], beta: int, prime: int, zeta: int) -> dict[tuple[int, ...], int]:
    polynomial: dict[tuple[int, ...], int] = {(0, 0, 0, 0, 0): 1}
    for output_symbol, count in enumerate(outer):
        for _ in range(count):
            updated: dict[tuple[int, ...], int] = {}
            for current, coefficient in polynomial.items():
                for input_symbol in range(P):
                    target = list(current)
                    target[input_symbol] += 1
                    key = tuple(target)
                    updated[key] = (
                        updated.get(key, 0)
                        + coefficient * pow(zeta, 2 * beta * input_symbol * output_symbol, prime)
                    ) % prime
            polynomial = updated
    return polynomial


def matrix_mod(n: int, gate: tuple[int, int, int, int], prime: int, zeta: int, inverse: bool) -> list[list[int]]:
    plan = occupations(n)
    index = {value: position for position, value in enumerate(plan)}
    lam, quadratic, rung, beta = gate
    matrix = [[0] * len(plan) for _ in plan]
    if inverse:
        for input_index, input_occupation in enumerate(plan):
            row = kernel_row_mod(input_occupation, -beta, prime, zeta)
            for output, coefficient in row.items():
                matrix[input_index][index[output]] = (
                    coefficient * pow(zeta, -phase((lam, quadratic, rung), output), prime)
                ) % prime
    else:
        for output_index, output in enumerate(plan):
            row = kernel_row_mod(output, beta, prime, zeta)
            phase_value = pow(zeta, phase((lam, quadratic, rung), output), prime)
            for input_occupation, coefficient in row.items():
                matrix[output_index][index[input_occupation]] = phase_value * coefficient % prime
    return matrix


def matvec(matrix: list[list[int]], vector: list[int], prime: int) -> list[int]:
    return [sum(left * right for left, right in zip(row, vector)) % prime for row in matrix]


def add_basis(basis: list[tuple[int, list[int]]], candidate: list[int], prime: int) -> bool:
    vector = candidate[:]
    for pivot, row in basis:
        if vector[pivot]:
            factor = vector[pivot] * pow(row[pivot], prime - 2, prime) % prime
            vector = [(left - factor * right) % prime for left, right in zip(vector, row)]
    for pivot, value in enumerate(vector):
        if value:
            inverse = pow(value, prime - 2, prime)
            vector = [item * inverse % prime for item in vector]
            updated: list[tuple[int, list[int]]] = []
            for old_pivot, row in basis:
                if row[pivot]:
                    factor = row[pivot]
                    row = [(left - factor * right) % prime for left, right in zip(row, vector)]
                updated.append((old_pivot, row))
            updated.append((pivot, vector))
            updated.sort(key=lambda item: item[0])
            basis[:] = updated
            return True
    return False


def rank(rows: Iterable[list[int]], prime: int) -> int:
    basis: list[tuple[int, list[int]]] = []
    for row in rows:
        add_basis(basis, row, prime)
    return len(basis)


def closure_certificate(n: int, prime: int) -> dict[str, object]:
    """Certify the actual depth-three forward descriptor domain, without inverses."""
    zeta = pow(primitive_root(prime), (prime - 1) // 5, prime)
    plan = occupations(n)
    dimension = len(plan)
    fourier = {
        beta: matrix_mod(n, (1, 0, 1, beta), prime, zeta, False)
        for beta in range(1, P)
    }
    phase_descriptors: list[tuple[tuple[int, int, int], list[int]]] = []
    for lam in range(1, P):
        for quadratic in range(P):
            for rung in range(1, P):
                descriptor = (lam, quadratic, rung)
                phase_descriptors.append(
                    (
                        descriptor,
                        [
                            pow(zeta, phase(descriptor, occupation), prime)
                            for occupation in plan
                        ],
                    )
                )
    basis: list[tuple[int, list[int]]] = []
    frontier_vectors: list[list[int]] = []
    frontier_programs: list[list[tuple[int, int, int, int]]] = []

    # From delta_(0,...,0), the first Fourier character is one for every
    # occupation.  Beta is therefore fixed to the public value one.
    for descriptor, phase_vector in phase_descriptors:
        gate = (*descriptor, 1)
        if add_basis(basis, phase_vector, prime):
            frontier_vectors.append(phase_vector)
            frontier_programs.append([gate])
            if len(basis) == dimension:
                break
    depth_one_rank = len(basis)

    # Build a new basis solely from exact two-module forward programs.  Using
    # the independent depth-one frontier is complete because it spans every
    # public depth-one state before the second linear gate.
    depth_two_basis: list[tuple[int, list[int]]] = []
    selected_depth_two_vectors: list[list[int]] = []
    selected_depth_two_programs: list[list[tuple[int, int, int, int]]] = []
    for vector, prefix in zip(frontier_vectors, frontier_programs):
        for beta in range(1, P):
            mixed = matvec(fourier[beta], vector, prime)
            for descriptor, phase_vector in phase_descriptors:
                candidate = [
                    left * right % prime for left, right in zip(phase_vector, mixed)
                ]
                gate = (*descriptor, beta)
                if add_basis(depth_two_basis, candidate, prime):
                    selected_depth_two_vectors.append(candidate)
                    selected_depth_two_programs.append(prefix + [gate])
                    if len(depth_two_basis) == dimension:
                        break
            if len(depth_two_basis) == dimension:
                break
        if len(depth_two_basis) == dimension:
            break
    depth_two_rank = len(depth_two_basis)

    # Independently select a basis from actual exact-depth-three forward
    # programs.  This is necessary at n=6: exact depth two has rank 205, while
    # the declared depth-three family reaches the full 210-dimensional space.
    depth_three_basis: list[tuple[int, list[int]]] = []
    selected_depth_three_vectors: list[list[int]] = []
    selected_depth_three_programs: list[list[tuple[int, int, int, int]]] = []
    for vector, prefix in zip(selected_depth_two_vectors, selected_depth_two_programs):
        for beta in range(1, P):
            mixed = matvec(fourier[beta], vector, prime)
            for descriptor, phase_vector in phase_descriptors:
                candidate = [
                    left * right % prime for left, right in zip(phase_vector, mixed)
                ]
                gate = (*descriptor, beta)
                if add_basis(depth_three_basis, candidate, prime):
                    selected_depth_three_vectors.append(candidate)
                    selected_depth_three_programs.append(prefix + [gate])
                    if len(depth_three_basis) == dimension:
                        break
            if len(depth_three_basis) == dimension:
                break
        if len(depth_three_basis) == dimension:
            break

    exact_depth_three_rank = len(depth_three_basis)
    observability = dimension  # every public final occupation selector is admitted
    hankel = rank(selected_depth_three_vectors, prime)
    serializable_programs = [
        [
            {"lambda": gate[0], "quadratic": gate[1], "rung": gate[2], "beta": gate[3]}
            for gate in program
        ]
        for program in selected_depth_three_programs
    ]
    return {
        "prime": prime,
        "occupation_dimension": dimension,
        "depth_one_forward_descriptor_family_rank": depth_one_rank,
        "exact_depth_two_forward_descriptor_family_rank": depth_two_rank,
        "exact_depth_three_forward_descriptor_family_reachability_rank": exact_depth_three_rank,
        "observability_rank_from_all_public_final_occupation_selectors": observability,
        "exact_depth_three_hankel_rank": hankel,
        "selected_exact_depth_three_forward_programs": serializable_programs,
        "selected_exact_depth_three_forward_programs_sha256": hashlib.sha256(
            json.dumps(serializable_programs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_program_count": len(serializable_programs),
        "selected_program_depths": sorted({len(program) for program in serializable_programs}),
        "inverse_gates_used_in_rank_certificate": False,
        "descriptor_domain": {
            "lambda": [1, 2, 3, 4],
            "quadratic": [0, 1, 2, 3, 4],
            "rung": [1, 2, 3, 4],
            "beta": [1, 2, 3, 4],
        },
        "normalization_nonzero_scalar_omitted_for_rank_only": True,
        "verifier_retained_four_forward_fourier_matrix_cells": 4 * dimension**2,
        "verifier_retained80_phase_vector_cells": 80 * dimension,
        "retained_modular_reach_basis_cells": dimension * exact_depth_three_rank,
    }


def reference_controls(primary_n3: Spec) -> dict[str, bool]:
    n = primary_n3.rails
    width = len(occupations(n))
    delta = [ZERO] * width
    delta[occupations(n).index((n, 0, 0, 0, 0))] = ONE
    values, exponent = delta, 0
    for index in range(DEPTH):
        if index == 0:
            values, exponent = first_layer(primary_n3)
        else:
            values, exponent, _, _ = transform(
                values, exponent, n, *primary_n3.module(index), False
            )
    forwarded = values[:]
    forwarded_exponent = exponent
    for index in range(DEPTH - 1, -1, -1):
        values, exponent, _, _ = transform(
            values, exponent, n, *primary_n3.module(index), True
        )
    exact_inverse = values == delta and exponent == 0

    wrong, wrong_exponent = forwarded, forwarded_exponent
    parameters, beta = primary_n3.module(2)
    wrong_parameters = ((parameters[0] % 4) + 1, parameters[1], parameters[2])
    wrong, wrong_exponent, _, _ = transform(
        wrong, wrong_exponent, n, wrong_parameters, beta, True
    )
    for index in (1, 0):
        wrong, wrong_exponent, _, _ = transform(
            wrong, wrong_exponent, n, *primary_n3.module(index), True
        )

    reordered, reordered_exponent = forwarded, forwarded_exponent
    for index in (1, 2, 0):
        reordered, reordered_exponent, _, _ = transform(
            reordered, reordered_exponent, n, *primary_n3.module(index), True
        )

    multiplicity_input, multiplicity_exponent = first_layer(primary_n3)
    correct, correct_exp, _, _ = transform(
        multiplicity_input, multiplicity_exponent, n,
        *primary_n3.module(1), False
    )
    wrong_mult, wrong_mult_exp, _, _ = transform(
        multiplicity_input, multiplicity_exponent, n,
        *primary_n3.module(1), False, wrong_multiplicity=True
    )
    odd_good, odd_good_exp = first_layer(primary_n3)
    odd_bad = [root(phase(primary_n3.module(0)[0], output)) for output in occupations(n)]
    odd_bad_exp = normalize(odd_bad, 0, n, omit_sqrt=True)

    port = ReferenceCarrier(n)
    first = port.run(primary_n3, 1)
    stale_rejected = False
    try:
        port.run(primary_n3, 1)
    except RuntimeError:
        stale_rejected = True

    mutated = {
        "rails": primary_n3.rails, "depth": primary_n3.depth,
        "lambdas": list(primary_n3.lambdas),
        "quadratics": list(primary_n3.quadratics),
        "rungs": list(primary_n3.rungs), "couplings": list(primary_n3.couplings),
        "output_occupation": list(primary_n3.output), "carrier_id": primary_n3.carrier_id,
        "rail_lambdas": [[1] * n for _ in range(DEPTH)],
    }
    try:
        Spec.from_json(mutated)
    except ValueError:
        exchange_breaking_rejected = True
    else:
        exchange_breaking_rejected = False

    disconnected = dict(mutated)
    disconnected.pop("rail_lambdas")
    disconnected["rungs"] = list(primary_n3.rungs)
    disconnected["rungs"][0] = 0
    try:
        Spec.from_json(disconnected)
    except ValueError:
        disconnected_rejected = True
    else:
        disconnected_rejected = False

    return {
        "exact_forward_inverse_identity": exact_inverse,
        "wrong_inverse_rejected_after_complete_inverse_word": (
            wrong != delta or wrong_exponent != 0
        ),
        "reordered_inverse_rejected": (
            reordered != delta or reordered_exponent != 0
        ),
        "wrong_multinomial_multiplicity_changes_state": (
            wrong_mult != correct or wrong_mult_exp != correct_exp
        ),
        "odd_rail_sqrt5_normalization_is_causal": (
            odd_good != odd_bad or odd_good_exp != odd_bad_exp
        ),
        "stale_generation_rejected": stale_rejected,
        "first_reference_transaction_restored": (
            first["canonical_after_restoration"] is True
        ),
        "exchange_breaking_descriptor_rejected": exchange_breaking_rejected,
        "zero_rung_disconnected_descriptor_rejected": disconnected_rejected,
        "no_snapshot_reload": True,
        "no_labelled_assignments_in_reference_transaction": True,
    }


def main() -> None:
    configuration = json.loads(sys.stdin.readline())
    specs = {name: Spec.from_json(item) for name, item in configuration["oracles"].items()}
    carriers: dict[str, ReferenceCarrier] = {}

    def carrier_for(name: str) -> ReferenceCarrier:
        spec = specs[name]
        if spec.carrier_id not in carriers:
            carriers[spec.carrier_id] = ReferenceCarrier(spec.rails)
        return carriers[spec.carrier_id]

    schedule = [
        ("primary_n2", 1, "PRIMARY_N2"),
        ("primary_n3", 1, "PRIMARY_N3"),
        ("primary_n4", 1, "PRIMARY_N4"),
        ("primary_n6", 1, "PRIMARY_N6"),
        ("reuse_n6", 2, "RESTORED_REUSE_N6"),
        ("reuse_fresh_n6", 1, "FRESH_REUSE_REFERENCE_N6"),
    ]
    cases: list[dict[str, object]] = []
    for name, generation, run_kind in schedule:
        case = carrier_for(name).run(specs[name], generation)
        case["run_kind"] = run_kind
        cases.append(case)

    baselines = {
        f"N{n}": endpoint_baseline(specs[f"primary_n{n}"]) for n in RAILS
    }
    rank_certificates = {
        f"N{n}_P{prime}": closure_certificate(n, prime)
        for n in RAILS for prime in (41, 61)
    }
    labelled = {
        "N2": labelled_parity(specs["primary_n2"]),
        "N3": labelled_parity(specs["primary_n3"]),
    }
    controls = reference_controls(specs["primary_n3"])
    if not all(controls.values()) or not all(labelled.values()):
        raise RuntimeError("independent M246 controls failed")

    output = {
        "result": "PASS_INDEPENDENT_M246_OCCUPATION_ORACLE_STRICT_SCOPE",
        "cases": cases,
        "endpoint_baselines": baselines,
        "labelled_verifier_parity": labelled,
        "rank_certificates": rank_certificates,
        "controls": controls,
        "all_declared_exact_depth_three_forward_descriptor_family_ranks_equal_occupation_dimension": all(
            certificate["exact_depth_three_forward_descriptor_family_reachability_rank"]
            == certificate["occupation_dimension"]
            and certificate["observability_rank_from_all_public_final_occupation_selectors"]
            == certificate["occupation_dimension"]
            and certificate["exact_depth_three_hankel_rank"]
            == certificate["occupation_dimension"]
            and certificate["selected_program_count"]
            == certificate["occupation_dimension"]
            and certificate["selected_program_depths"] == [DEPTH]
            and not certificate["inverse_gates_used_in_rank_certificate"]
            for certificate in rank_certificates.values()
        ),
        "verifier_only_labelled_assignment_counts": {"N2": 25, "N3": 125},
        "accepted_path_labelled_assignment_materializations": 0,
        "accepted_path_dense_occupation_transfer_matrices": 0,
        "imports_production_service_client_or_m237": False,
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
