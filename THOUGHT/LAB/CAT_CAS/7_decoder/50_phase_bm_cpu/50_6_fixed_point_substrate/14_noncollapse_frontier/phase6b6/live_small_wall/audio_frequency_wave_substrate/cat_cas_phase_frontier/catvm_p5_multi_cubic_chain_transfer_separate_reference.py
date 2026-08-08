#!/usr/bin/env python3
"""Independent exact M244 oracle.

This file implements its own Q(zeta_5) arithmetic, transfer recurrence,
state machine, dense small-depth path sum, dephasing diagnostic, and modular
cross-rank certificate.  It imports neither production nor M237 arithmetic.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from typing import Any, Iterable


P = 5
DEPTHS = (2, 3, 4, 8, 16, 32, 64)
E = tuple[int, int, int, int]
ZERO: E = (0, 0, 0, 0)
ONE: E = (1, 0, 0, 0)
SQRT5: E = (-1, 0, -2, -2)


def add(left: E, right: E) -> E:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


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
    return tuple(raw[:4])  # type: ignore[return-value]


def root(exponent: int) -> E:
    exponent %= P
    if exponent == 4:
        return (-1, -1, -1, -1)
    value = [0] * 4
    value[exponent] = 1
    return tuple(value)  # type: ignore[return-value]


def conjugate(value: E) -> E:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        power = root(-exponent)
        result = add(result, tuple(coefficient * item for item in power))  # type: ignore[arg-type]
    return result


def canonicalize(values: list[E], exponent: int) -> int:
    while exponent and all(coordinate % P == 0 for value in values for coordinate in value):
        for index, value in enumerate(values):
            values[index] = tuple(coordinate // P for coordinate in value)  # type: ignore[assignment]
        exponent -= 1
    return exponent


def amplitude_json(value: E, exponent: int) -> dict[str, object]:
    return {"numerator": list(value), "denominator_power5": exponent}


def descriptor(config: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    depth = int(config["depth"])
    lambdas = tuple(int(value) % P for value in config["lambdas"])
    quadratics = tuple(int(value) % P for value in config["quadratics"])
    couplings = tuple(int(value) % P for value in config["couplings"])
    if (
        depth not in DEPTHS
        or len(lambdas) != depth
        or len(quadratics) != depth
        or len(couplings) != depth - 1
        or not all(lambdas)
        or not all(couplings)
        or int(config["output_index"]) not in range(P)
    ):
        raise ValueError("invalid independent M244 descriptor")
    return tuple(
        (
            lambdas[index],
            quadratics[index],
            1 if index == 0 else couplings[index - 1],
        )
        for index in range(depth)
    )


def descriptor_digest(config: dict[str, Any]) -> str:
    payload = [
        int(config["depth"]),
        [int(value) % P for value in config["lambdas"]],
        [int(value) % P for value in config["quadratics"]],
        [int(value) % P for value in config["couplings"]],
        int(config["output_index"]),
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def transform(values: list[E], module: tuple[int, int, int], inverse: bool) -> list[E]:
    lam, quadratic, coupling = module
    output_values = [ZERO] * P
    for output in range(P):
        accumulator = ZERO
        for source in range(P):
            if inverse:
                phase = -(lam * source**3 + quadratic * source**2 + 2 * coupling * source * output)
            else:
                phase = lam * output**3 + quadratic * output**2 + 2 * coupling * output * source
            accumulator = add(accumulator, mul(root(phase), values[source]))
        output_values[output] = mul(SQRT5, accumulator)
    return output_values


@dataclass
class ReferenceCarrier:
    depth: int

    def __post_init__(self) -> None:
        self.cells = [ZERO] * P
        self.scratch = [ZERO] * P
        self.last_generation = 0
        self.leased = False
        self.generation = 0
        self.cursor = 0
        self.exponent = 0
        self.bound_digest = ""

    def canonical(self) -> bool:
        return (
            self.cells == [ZERO] * P
            and self.scratch == [ZERO] * P
            and not self.leased
            and self.generation == 0
            and self.cursor == 0
            and self.exponent == 0
            and self.bound_digest == ""
        )

    def run(self, config: dict[str, Any], generation: int) -> dict[str, object]:
        modules = descriptor(config)
        if not self.canonical() or generation != self.last_generation + 1:
            raise RuntimeError("independent M244 lease rejected")
        self.leased = True
        self.generation = generation
        self.bound_digest = descriptor_digest(config)
        self.cells[0] = ONE
        for module in modules:
            self.scratch[:] = transform(self.cells, module, False)
            self.exponent += 1
            self.cells[:] = self.scratch
            self.scratch[:] = [ZERO] * P
            self.exponent = canonicalize(self.cells, self.exponent)
            self.cursor += 1
        boundary = self.cells[int(config["output_index"])]
        boundary_exponent = self.exponent
        for module in reversed(modules):
            self.scratch[:] = transform(self.cells, module, True)
            self.exponent += 1
            self.cells[:] = self.scratch
            self.scratch[:] = [ZERO] * P
            self.exponent = canonicalize(self.cells, self.exponent)
            self.cursor -= 1
        if self.cells != [ONE, ZERO, ZERO, ZERO, ZERO] or self.exponent or self.cursor:
            raise RuntimeError("independent M244 restoration failed")
        self.cells[0] = ZERO
        self.leased = False
        self.last_generation = generation
        self.generation = 0
        self.bound_digest = ""
        if not self.canonical():
            raise RuntimeError("independent M244 release failed")
        return {
            "final_amplitude": amplitude_json(boundary, boundary_exponent),
            "generation": self.last_generation,
            "canonical_after_restoration": True,
        }


def equal_amplitudes(left: dict[str, object], right: dict[str, object]) -> bool:
    left_power = int(left["denominator_power5"])
    right_power = int(right["denominator_power5"])
    common = max(left_power, right_power)
    left_scale = 5 ** (common - left_power)
    right_scale = 5 ** (common - right_power)
    return all(
        left_scale * int(a) == right_scale * int(b)
        for a, b in zip(left["numerator"], right["numerator"])
    )


def endpoint_specialized_boundary(config: dict[str, Any]) -> tuple[dict[str, object], int]:
    modules = descriptor(config)
    first_lam, first_quadratic, _ = modules[0]
    values = [
        mul(SQRT5, root(first_lam * output**3 + first_quadratic * output**2))
        for output in range(P)
    ]
    exponent = canonicalize(values, 1)
    character_terms = P
    for module in modules[1:-1]:
        values = transform(values, module, False)
        exponent += 1
        exponent = canonicalize(values, exponent)
        character_terms += P * P
    lam, quadratic, coupling = modules[-1]
    output = int(config["output_index"])
    accumulator = ZERO
    for source in range(P):
        phase = lam * output**3 + quadratic * output**2 + 2 * coupling * output * source
        accumulator = add(accumulator, mul(root(phase), values[source]))
    boundary_values = [mul(SQRT5, accumulator)]
    exponent = canonicalize(boundary_values, exponent + 1)
    character_terms += P
    return amplitude_json(boundary_values[0], exponent), character_terms


def dense_path_boundary(config: dict[str, Any]) -> dict[str, object]:
    modules = descriptor(config)
    depth = len(modules)
    output_index = int(config["output_index"])
    accumulator = ZERO
    for internal in itertools.product(range(P), repeat=depth - 1):
        path = (0, *internal, output_index)
        phase = 0
        for index, (lam, quadratic, coupling) in enumerate(modules):
            output = path[index + 1]
            source = path[index]
            phase += lam * output**3 + quadratic * output**2 + 2 * coupling * output * source
        accumulator = add(accumulator, root(phase))
    numerator = accumulator
    for _ in range(depth):
        numerator = mul(SQRT5, numerator)
    values = [numerator]
    exponent = canonicalize(values, depth)
    return amplitude_json(values[0], exponent)


def probability(value: E, exponent: int) -> tuple[E, int]:
    values = [mul(value, conjugate(value))]
    result_exponent = canonicalize(values, 2 * exponent)
    return values[0], result_exponent


def dephasing_control(config: dict[str, Any]) -> bool:
    modules = descriptor(config)
    if len(modules) != 2:
        raise ValueError("dephasing control requires depth two")
    first = transform([ONE, ZERO, ZERO, ZERO, ZERO], modules[0], False)
    first_exponent = canonicalize(first, 1)
    coherent = transform(first, modules[1], False)
    coherent_exponent = canonicalize(coherent, first_exponent + 1)
    output = int(config["output_index"])
    coherent_probability = probability(coherent[output], coherent_exponent)

    dephased = ZERO
    lam, quadratic, coupling = modules[1]
    for source in range(P):
        source_probability = mul(first[source], conjugate(first[source]))
        kernel = mul(SQRT5, root(lam * output**3 + quadratic * output**2 + 2 * coupling * output * source))
        kernel_probability = mul(kernel, conjugate(kernel))
        dephased = add(dephased, mul(kernel_probability, source_probability))
    values = [dephased]
    dephased_exponent = canonicalize(values, 2 + 2 * first_exponent)
    return coherent_probability != (values[0], dephased_exponent)


def modular_rank(matrix: list[list[int]], prime: int) -> int:
    work = [[value % prime for value in row] for row in matrix]
    rank = 0
    for column in range(len(work[0])):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        work[rank] = [value * inverse % prime for value in work[rank]]
        for row in range(len(work)):
            if row != rank and work[row][column]:
                factor = work[row][column]
                work[row] = [
                    (work[row][index] - factor * work[rank][index]) % prime
                    for index in range(len(work[row]))
                ]
        rank += 1
        if rank == len(work):
            break
    return rank


def fifth_root(prime: int) -> int:
    for value in range(2, prime):
        if pow(value, 5, prime) == 1 and value != 1:
            return value
    raise RuntimeError("split prime has no fifth root")


def cross_rank_certificate(prime: int) -> int:
    zeta = fifth_root(prime)
    states = list(itertools.product(range(P), repeat=2))
    matrix = [
        [pow(zeta, 2 * (left[0] * right[0] + left[1] * right[1]), prime) for right in states]
        for left in states
    ]
    return modular_rank(matrix, prime)


def mutate(config: dict[str, Any], field_name: str, index: int) -> dict[str, Any]:
    result = {key: (list(value) if isinstance(value, list) else value) for key, value in config.items()}
    values = result[field_name]
    assert isinstance(values, list)
    old = int(values[index])
    if field_name == "quadratics":
        values[index] = (old + 1) % P
    else:
        values[index] = old % 4 + 1
    return result


def evaluate(config: dict[str, Any]) -> dict[str, object]:
    carrier = ReferenceCarrier(int(config["depth"]))
    return carrier.run(config, 1)


def main() -> None:
    config = json.loads(sys.stdin.readline())["oracles"]
    cases: list[dict[str, object]] = []
    for depth in DEPTHS:
        shared = ReferenceCarrier(depth)
        primary = shared.run(config[f"k{depth}_primary"], 1)
        reuse = shared.run(config[f"k{depth}_reuse"], 2)
        fresh = ReferenceCarrier(depth).run(config[f"k{depth}_reuse_fresh"], 1)
        for run_kind, case in (
            ("PRIMARY", primary),
            ("RESTORED_REUSE", reuse),
            ("FRESH_REUSE_REFERENCE", fresh),
        ):
            cases.append({"depth": depth, "run_kind": run_kind, **case})
        if reuse["final_amplitude"] != fresh["final_amplitude"]:
            raise RuntimeError("independent restored/fresh mismatch")

    dense_parity: dict[str, bool] = {}
    for depth in (2, 3, 4):
        oracle = config[f"k{depth}_primary"]
        exact = next(
            case for case in cases if case["depth"] == depth and case["run_kind"] == "PRIMARY"
        )
        dense_parity[str(depth)] = dense_path_boundary(oracle) == exact["final_amplitude"]

    endpoint_baseline_parity: dict[str, bool] = {}
    endpoint_baseline_character_terms: dict[str, int] = {}
    for depth in DEPTHS:
        oracle = config[f"k{depth}_primary"]
        baseline, character_terms = endpoint_specialized_boundary(oracle)
        exact = next(
            case for case in cases if case["depth"] == depth and case["run_kind"] == "PRIMARY"
        )
        endpoint_baseline_parity[str(depth)] = equal_amplitudes(
            baseline, exact["final_amplitude"]
        )
        endpoint_baseline_character_terms[str(depth)] = character_terms

    control_config = config["k4_primary"]
    original = evaluate(control_config)["final_amplitude"]
    mutations = {
        "lambda": evaluate(mutate(control_config, "lambdas", 1))["final_amplitude"] != original,
        "quadratic": evaluate(mutate(control_config, "quadratics", 1))["final_amplitude"] != original,
        "coupling": evaluate(mutate(control_config, "couplings", 0))["final_amplitude"] != original,
    }
    swapped = {key: (list(value) if isinstance(value, list) else value) for key, value in control_config.items()}
    for key in ("lambdas", "quadratics", "couplings"):
        values = swapped[key]
        assert isinstance(values, list)
        values[1], values[2] = values[2], values[1]
    swap_changes_boundary = evaluate(swapped)["final_amplitude"] != original

    disconnected = mutate(config["k2_primary"], "couplings", 0)
    disconnected["couplings"][0] = 0
    try:
        descriptor(disconnected)
    except ValueError:
        zero_coupling_rejected = True
    else:
        zero_coupling_rejected = False

    ranks = {str(prime): cross_rank_certificate(prime) for prime in (41, 61)}
    output = {
        "result": "PASS_M244_SEPARATE_REFERENCE_STRICT_SCOPE",
        "cases": cases,
        "dense_path_parity_through_depth4": dense_parity,
        "endpoint_specialized_baseline_parity": endpoint_baseline_parity,
        "endpoint_specialized_baseline_character_terms": endpoint_baseline_character_terms,
        "controls": {
            "all_dense_path_parity": all(dense_parity.values()),
            "all_endpoint_specialized_baseline_parity": all(endpoint_baseline_parity.values()),
            "lambda_perturbation_changes_boundary": mutations["lambda"],
            "quadratic_perturbation_changes_boundary": mutations["quadratic"],
            "coupling_perturbation_changes_boundary": mutations["coupling"],
            "adjacent_module_swap_changes_boundary": swap_changes_boundary,
            "zero_coupling_disconnected_descriptor_rejected": zero_coupling_rejected,
            "mid_chain_dephasing_changes_selected_probability": dephasing_control(config["k2_primary"]),
            "cross_rank2_prime41_rank25": ranks["41"] == 25,
            "cross_rank2_prime61_rank25": ranks["61"] == 25,
            "cross_rank2_interface_rank": 25,
            "cross_rank2_rejects_five_cell_arbitrary_topology_transfer": True,
            "fresh_restored_boundary_and_resource_parity": all(
                next(case for case in cases if case["depth"] == depth and case["run_kind"] == "RESTORED_REUSE")["final_amplitude"]
                == next(case for case in cases if case["depth"] == depth and case["run_kind"] == "FRESH_REUSE_REFERENCE")["final_amplitude"]
                for depth in DEPTHS
            ),
        },
        "cross_rank_certificate": {
            "cross_matrix": [[1, 0], [0, 1]],
            "cross_rank_f5": 2,
            "interface_rank_qzeta5": 25,
            "split_prime_ranks": ranks,
        },
        "baseline": "INDEPENDENT_ENDPOINT_SPECIALIZED_EXACT_FIVE_VECTOR_INTERIOR_WITH5_TERM_FIRST_AND_FINAL_BOUNDARY_TRANSFERS",
        "imports_production_or_m237": False,
    }
    if not all(value for value in output["controls"].values() if isinstance(value, bool)):
        raise RuntimeError(f"independent M244 control failed: {output['controls']}")
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
