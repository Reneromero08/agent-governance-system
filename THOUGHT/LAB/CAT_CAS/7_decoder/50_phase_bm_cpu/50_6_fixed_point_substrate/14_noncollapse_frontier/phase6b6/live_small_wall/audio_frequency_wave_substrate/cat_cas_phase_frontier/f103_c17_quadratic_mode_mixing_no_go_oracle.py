#!/usr/bin/env python3
"""Independent oracle for the C17 quadratic mode-mixing diagnostic.

The oracle imports neither the production package nor M150.  It reconstructs
the public schedule, executes a pure-Python 867-coordinate coefficient
recurrence, separately executes the coupled 17-mode recurrence, reconstructs
all final cells, and attacks restoration and the independent-mode sham.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


MODULUS = 103
CYCLE = 17
ROOT = 72
ROOT_INVERSE = pow(ROOT, -1, MODULUS)
CYCLE_INVERSE = pow(CYCLE, -1, MODULUS)
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
ROTATION = tuple(
    tuple(pow(ROOT, mode * shift, MODULUS) for mode in range(CYCLE))
    for shift in range(CYCLE)
)
NTT_MATRIX = tuple(
    tuple(pow(ROOT, mode * coordinate, MODULUS) for coordinate in range(CYCLE))
    for mode in range(CYCLE)
)
INVERSE_NTT_MATRIX = tuple(
    tuple(
        pow(ROOT_INVERSE, mode * coordinate, MODULUS)
        for mode in range(CYCLE)
    )
    for coordinate in range(CYCLE)
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def family_code(family: str) -> int:
    return {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (3 * index + family_code(family) + mutation) % CYCLE


def target_order(hub: int) -> Iterator[int]:
    for offset in range(1, CYCLE):
        yield (hub + offset) % CYCLE


def public_offset(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    mutation: int = 0,
) -> int:
    return (
        5 * controller
        + 7 * target
        + 3 * index
        + 2 * layer
        + family_code(family)
        + mutation
    ) % CYCLE


def quadratic_offset(
    hub: int,
    target: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> int:
    return (
        11 * hub
        + 13 * target
        + 5 * index
        + 3 * family_code(family)
        + mutation
    ) % CYCLE


def ternary_weight(index: int) -> int:
    total = 0
    remaining = index
    while remaining:
        total += remaining % 3
        remaining //= 3
    return total


def phase_exponent(shell: int, index: int, family: str) -> int:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    if family == "PRIMARY":
        values = (
            3 * index + 5 * bit_weight + 1,
            7 * index + 2 * bit_weight + 2,
            11 * index + bit_weight + 4,
        )
    elif family == "REUSE":
        weight = ternary_weight(index)
        values = (
            5 * index + 2 * weight + 3,
            4 * index + 3 * weight + 6,
            9 * index + weight + 8,
        )
    elif family == "ALTERNATE":
        values = (
            7 * index + 3 * gray_weight + 2,
            8 * index + 2 * gray_weight + 5,
            6 * index + gray_weight + 1,
        )
    else:
        fail("unknown oracle family")
    quadratic = values[0] % CYCLE or 1
    linear = values[1] % CYCLE
    cubic = values[2] % CYCLE
    return (
        quadratic * shell**4 + linear * shell**2 + cubic * shell
    ) % CYCLE


def observation(depth: int, family: str) -> tuple[int, int]:
    return (
        (7 * depth + 3 * len(family) + 1) % MODULUS or 1,
        (11 * depth + len(family) + 5) % MODULUS,
    )


def seed_coefficients() -> list[list[list[int]]]:
    state = [
        [[0 for _ in range(CYCLE)] for _ in range(3)]
        for _ in range(CYCLE)
    ]
    for shell in range(CYCLE):
        for slot in range(3):
            positions = (
                (5 * shell + 3 * slot + 1) % CYCLE,
                (7 * shell * shell + 2 * slot + 4) % CYCLE,
                (11 * shell + 5 * slot + 9) % CYCLE,
            )
            amplitudes = (
                1 + (3 * shell + 5 * slot) % 31,
                1 + (7 * shell + 2 * slot) % 29,
                MODULUS - (1 + (11 * shell + slot) % 23),
            )
            for position, amplitude in zip(positions, amplitudes, strict=True):
                state[shell][slot][position] = (
                    state[shell][slot][position] + amplitude
                ) % MODULUS
    return state


def rotate_coefficients(vector: list[int], shift: int) -> list[int]:
    normalized = shift % CYCLE
    if normalized == 0:
        return vector.copy()
    return vector[-normalized:] + vector[:-normalized]


def cyclic_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * CYCLE
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output = (left_index + right_index) % CYCLE
            result[output] = (
                result[output] + left_value * right_value
            ) % MODULUS
    return result


def coefficient_triangular_shear(
    destination: list[list[int]],
    control: list[int],
    *,
    inverse: bool,
) -> None:
    if inverse:
        term_two = cyclic_convolution(control, destination[1])
        destination[2] = [
            (destination[2][coordinate] - term_two[coordinate]) % MODULUS
            for coordinate in range(CYCLE)
        ]
        term_one = cyclic_convolution(control, destination[0])
        destination[1] = [
            (destination[1][coordinate] - term_one[coordinate]) % MODULUS
            for coordinate in range(CYCLE)
        ]
    else:
        term_one = cyclic_convolution(control, destination[0])
        destination[1] = [
            (destination[1][coordinate] + term_one[coordinate]) % MODULUS
            for coordinate in range(CYCLE)
        ]
        term_two = cyclic_convolution(control, destination[1])
        destination[2] = [
            (destination[2][coordinate] + term_two[coordinate]) % MODULUS
            for coordinate in range(CYCLE)
        ]


def apply_coefficient_convolution_layer(
    state: list[list[list[int]]],
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    peers = list(target_order(hub))
    if inverse:
        peers.reverse()
    for peer in peers:
        controller, target = (hub, peer) if layer == 0 else (peer, hub)
        if port_enabled:
            slot = 0 if layer == 0 else 1
            control = rotate_coefficients(
                state[controller][slot],
                public_offset(
                    controller,
                    target,
                    index,
                    family,
                    layer,
                    offset_mutation,
                ),
            )
        else:
            control = [0] * CYCLE
        coefficient_triangular_shear(
            state[target], control, inverse=inverse
        )


def apply_coefficient_quadratic_layer(
    state: list[list[list[int]]],
    index: int,
    family: str,
    *,
    inverse: bool,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    if port_enabled:
        shared_square = [
            value * value % MODULUS for value in state[hub][2]
        ]
    else:
        shared_square = [0] * CYCLE
    peers = list(target_order(hub))
    if inverse:
        peers.reverse()
    sign = -1 if inverse else 1
    for target in peers:
        term = rotate_coefficients(
            shared_square,
            quadratic_offset(
                hub, target, index, family, mutation=offset_mutation
            ),
        )
        state[target][0] = [
            (state[target][0][coordinate] + sign * term[coordinate])
            % MODULUS
            for coordinate in range(CYCLE)
        ]


ORDERS = {
    "OUT_QUADRATIC_IN": ("OUT", "QUADRATIC", "IN"),
    "OUT_IN_QUADRATIC": ("OUT", "IN", "QUADRATIC"),
}


def apply_coefficient_module(
    state: list[list[list[int]]],
    module: str,
    index: int,
    family: str,
    *,
    inverse: bool,
    port_enabled: bool,
    hub_mutation: int,
    linear_offset_mutation: int,
    quadratic_offset_mutation: int,
) -> None:
    if module == "OUT":
        apply_coefficient_convolution_layer(
            state,
            index,
            family,
            0,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            offset_mutation=linear_offset_mutation,
        )
    elif module == "QUADRATIC":
        apply_coefficient_quadratic_layer(
            state,
            index,
            family,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            offset_mutation=quadratic_offset_mutation,
        )
    elif module == "IN":
        apply_coefficient_convolution_layer(
            state,
            index,
            family,
            1,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            offset_mutation=linear_offset_mutation,
        )
    else:
        fail("unknown coefficient module")


def rotate_coefficient_state(
    state: list[list[list[int]]],
    index: int,
    family: str,
    *,
    inverse: bool,
) -> None:
    for shell in range(CYCLE):
        shift = phase_exponent(shell, index, family)
        if inverse:
            shift = -shift
        for slot in range(3):
            state[shell][slot] = rotate_coefficients(
                state[shell][slot], shift
            )


def forward_coefficients(
    state: list[list[list[int]]],
    depth: int,
    family: str,
    *,
    order: str = "OUT_QUADRATIC_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
) -> None:
    for index in range(depth):
        rotate_coefficient_state(
            state, index, family, inverse=False
        )
        for module in ORDERS[order]:
            apply_coefficient_module(
                state,
                module,
                index,
                family,
                inverse=False,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                linear_offset_mutation=0,
                quadratic_offset_mutation=0,
            )


def inverse_coefficients(
    state: list[list[list[int]]],
    depth: int,
    family: str,
    *,
    assumed_order: str = "OUT_QUADRATIC_IN",
    quadratic_offset_mutation: int = 0,
) -> None:
    for index in reversed(range(depth)):
        for module in reversed(ORDERS[assumed_order]):
            apply_coefficient_module(
                state,
                module,
                index,
                family,
                inverse=True,
                port_enabled=True,
                hub_mutation=0,
                linear_offset_mutation=0,
                quadratic_offset_mutation=quadratic_offset_mutation,
            )
        rotate_coefficient_state(state, index, family, inverse=True)


def transform(vector: list[int]) -> list[int]:
    return [
        sum(
            vector[coordinate] * NTT_MATRIX[mode][coordinate]
            for coordinate in range(CYCLE)
        )
        % MODULUS
        for mode in range(CYCLE)
    ]


def inverse_transform(vector: list[int]) -> list[int]:
    return [
        (
            sum(
                vector[mode] * INVERSE_NTT_MATRIX[coordinate][mode]
                for mode in range(CYCLE)
            )
            * CYCLE_INVERSE
        )
        % MODULUS
        for coordinate in range(CYCLE)
    ]


def transform_state(
    state: list[list[list[int]]],
) -> list[list[list[int]]]:
    return [
        [transform(state[shell][slot]) for slot in range(3)]
        for shell in range(CYCLE)
    ]


def inverse_transform_state(
    modes: list[list[list[int]]],
) -> list[list[list[int]]]:
    return [
        [inverse_transform(modes[shell][slot]) for slot in range(3)]
        for shell in range(CYCLE)
    ]


def rotate_modes(vector: list[int], shift: int) -> list[int]:
    factors = ROTATION[shift % CYCLE]
    return [
        vector[mode] * factors[mode] % MODULUS
        for mode in range(CYCLE)
    ]


def spectral_square(modes: list[int]) -> list[int]:
    return [
        (
            sum(
                modes[left] * modes[(output - left) % CYCLE]
                for left in range(CYCLE)
            )
            * CYCLE_INVERSE
        )
        % MODULUS
        for output in range(CYCLE)
    ]


def spectral_triangular_shear(
    destination: list[list[int]],
    control: list[int],
    *,
    inverse: bool,
) -> None:
    if inverse:
        destination[2] = [
            (destination[2][mode] - control[mode] * destination[1][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]
        destination[1] = [
            (destination[1][mode] - control[mode] * destination[0][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]
    else:
        destination[1] = [
            (destination[1][mode] + control[mode] * destination[0][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]
        destination[2] = [
            (destination[2][mode] + control[mode] * destination[1][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]


def apply_spectral_convolution_layer(
    modes: list[list[list[int]]],
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool,
) -> None:
    hub = hub_index(index, family)
    peers = list(target_order(hub))
    if inverse:
        peers.reverse()
    for peer in peers:
        controller, target = (hub, peer) if layer == 0 else (peer, hub)
        control_slot = 0 if layer == 0 else 1
        control = rotate_modes(
            modes[controller][control_slot],
            public_offset(controller, target, index, family, layer),
        )
        spectral_triangular_shear(
            modes[target], control, inverse=inverse
        )


def apply_spectral_quadratic_layer(
    modes: list[list[list[int]]],
    index: int,
    family: str,
    *,
    inverse: bool,
    independent_sham: bool = False,
) -> None:
    hub = hub_index(index, family)
    control = modes[hub][2]
    if independent_sham:
        shared_square = [value * value % MODULUS for value in control]
    else:
        shared_square = spectral_square(control)
    peers = list(target_order(hub))
    if inverse:
        peers.reverse()
    sign = -1 if inverse else 1
    for target in peers:
        term = rotate_modes(
            shared_square, quadratic_offset(hub, target, index, family)
        )
        modes[target][0] = [
            (modes[target][0][mode] + sign * term[mode]) % MODULUS
            for mode in range(CYCLE)
        ]


def rotate_spectral_state(
    modes: list[list[list[int]]],
    index: int,
    family: str,
    *,
    inverse: bool,
) -> None:
    for shell in range(CYCLE):
        shift = phase_exponent(shell, index, family)
        if inverse:
            shift = -shift
        for slot in range(3):
            modes[shell][slot] = rotate_modes(modes[shell][slot], shift)


def forward_modes(
    modes: list[list[list[int]]],
    depth: int,
    family: str,
    *,
    independent_sham: bool = False,
) -> None:
    for index in range(depth):
        rotate_spectral_state(modes, index, family, inverse=False)
        apply_spectral_convolution_layer(
            modes, index, family, 0, inverse=False
        )
        apply_spectral_quadratic_layer(
            modes,
            index,
            family,
            inverse=False,
            independent_sham=independent_sham,
        )
        apply_spectral_convolution_layer(
            modes, index, family, 1, inverse=False
        )


def inverse_modes(
    modes: list[list[list[int]]], depth: int, family: str
) -> None:
    for index in reversed(range(depth)):
        apply_spectral_convolution_layer(
            modes, index, family, 1, inverse=True
        )
        apply_spectral_quadratic_layer(
            modes, index, family, inverse=True
        )
        apply_spectral_convolution_layer(
            modes, index, family, 0, inverse=True
        )
        rotate_spectral_state(modes, index, family, inverse=True)


def flatten_bytes(state: list[list[list[int]]]) -> bytes:
    return bytes(
        value for shell in state for slot in shell for value in slot
    )


def commitment(state: list[list[list[int]]]) -> str:
    return hashlib.sha256(flatten_bytes(state)).hexdigest()


def boundary(
    state: list[list[list[int]]], depth: int, family: str
) -> list[int]:
    quadratic, linear = observation(depth, family)
    result = [0] * CYCLE
    for shell in range(CYCLE):
        for slot in range(3):
            weight = (
                quadratic * shell * shell
                + linear * (slot + 1)
                + 5 * shell * (slot + 1)
                + 1
            ) % MODULUS
            for coordinate in range(CYCLE):
                result[coordinate] = (
                    result[coordinate]
                    + weight * state[shell][slot][coordinate]
                ) % MODULUS
    return result


def execute_case(depth: int, family: str) -> dict[str, Any]:
    seed = seed_coefficients()
    coefficient_state = copy.deepcopy(seed)
    forward_coefficients(coefficient_state, depth, family)
    final_commitment = commitment(coefficient_state)
    final_boundary = boundary(coefficient_state, depth, family)
    support = [
        sum(value != 0 for value in coefficient_state[shell][slot])
        for shell in range(CYCLE)
        for slot in range(3)
    ]

    modes = transform_state(seed)
    forward_modes(modes, depth, family)
    spectral_state = inverse_transform_state(modes)
    coefficient_matches_spectral = spectral_state == coefficient_state
    inverse_modes(modes, depth, family)
    spectral_restored = inverse_transform_state(modes)

    inverse_coefficients(coefficient_state, depth, family)
    phase_work = depth * (32 * 2 * CYCLE * CYCLE + CYCLE)
    spectral_work = depth * (32 * 2 * CYCLE + CYCLE * CYCLE)
    return {
        "depth": depth,
        "family": family,
        "final_commitment": final_commitment,
        "boundary": final_boundary,
        "minimum_support": min(support),
        "maximum_support": max(support),
        "coefficient_inverse_restoration": coefficient_state == seed,
        "spectral_inverse_restoration": spectral_restored == seed,
        "coefficient_equals_coupled_spectral": coefficient_matches_spectral,
        "coupled_spectral_boundary": boundary(spectral_state, depth, family),
        "phase_work": phase_work,
        "spectral_work": spectral_work,
    }


def controls() -> dict[str, bool]:
    seed = seed_coefficients()
    normal = copy.deepcopy(seed)
    forward_coefficients(normal, 4, "PRIMARY")

    correct = copy.deepcopy(normal)
    inverse_coefficients(correct, 4, "PRIMARY")
    wrong = copy.deepcopy(normal)
    inverse_coefficients(
        wrong, 4, "PRIMARY", quadratic_offset_mutation=1
    )
    reordered = copy.deepcopy(normal)
    inverse_coefficients(
        reordered, 4, "PRIMARY", assumed_order="OUT_IN_QUADRATIC"
    )
    null_port = copy.deepcopy(seed)
    forward_coefficients(
        null_port, 4, "PRIMARY", port_enabled=False
    )
    swapped = copy.deepcopy(seed)
    forward_coefficients(
        swapped, 4, "PRIMARY", order="OUT_IN_QUADRATIC"
    )
    mutated = copy.deepcopy(seed)
    forward_coefficients(mutated, 4, "PRIMARY", hub_mutation=1)

    vector = seed[0][0]
    coefficient_square = [value * value % MODULUS for value in vector]
    transformed = transform(vector)
    coupled = spectral_square(transformed)
    independent = [value * value % MODULUS for value in transformed]

    sham_modes = transform_state(seed)
    forward_modes(sham_modes, 4, "PRIMARY", independent_sham=True)
    sham_state = inverse_transform_state(sham_modes)

    witness_modes = [0] * CYCLE
    witness_modes[0] = 1
    witness_modes[1] = 1
    generated = spectral_square(witness_modes)
    return {
        "correct_inverse_restores": correct == seed,
        "wrong_inverse_fails": wrong != seed,
        "reordered_inverse_fails": reordered != seed,
        "null_port_changes_boundary": boundary(normal, 4, "PRIMARY")
        != boundary(null_port, 4, "PRIMARY"),
        "module_order_noncommutes": boundary(normal, 4, "PRIMARY")
        != boundary(swapped, 4, "PRIMARY"),
        "topology_mutation_changes_boundary": boundary(normal, 4, "PRIMARY")
        != boundary(mutated, 4, "PRIMARY"),
        "pointwise_square_transform_law": transform(coefficient_square)
        == coupled,
        "independent_mode_square_rejected": coupled != independent,
        "independent_mode_sham_changes_final_state": sham_state != normal,
        "explicit_cross_mode_generation": generated[0] != 0
        and generated[1] != 0
        and generated[2] != 0,
        "all17_basis_ntt_roundtrip": all(
            inverse_transform(
                transform(
                    [int(index == coordinate) for index in range(CYCLE)]
                )
            )
            == [int(index == coordinate) for index in range(CYCLE)]
            for coordinate in range(CYCLE)
        ),
    }


def repeated_reuse() -> dict[str, Any]:
    seed = seed_coefficients()
    state = copy.deepcopy(seed)
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(64):
        forward_coefficients(state, 8, "REUSE")
        boundaries.add(tuple(boundary(state, 8, "REUSE")))
        inverse_coefficients(state, 8, "REUSE")
    return {
        "cycles": 64,
        "exact_restoration": state == seed,
        "stable_boundary_count": len(boundaries),
    }


def run(package_path: Path, production_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    production_hash = hashlib.sha256(production_path.read_bytes()).hexdigest()
    if package["source_sha256"] != production_hash:
        fail("production source hash does not match package")
    if package["execution_scope"]["depths"] != list(DEPTHS):
        fail("production depth scope changed")
    if package["execution_scope"]["families"] != list(FAMILIES):
        fail("production family scope changed")

    package_cases = {
        (case["depth"], case["family"]): case
        for case in package["cases"]
    }
    oracle_cases: list[dict[str, Any]] = []
    comparisons = 0
    for family in FAMILIES:
        for depth in DEPTHS:
            oracle = execute_case(depth, family)
            production = package_cases[(depth, family)]
            checks = {
                "commitment_match": oracle["final_commitment"]
                == production["final_commitment"],
                "boundary_match": oracle["boundary"]
                == production["boundary"],
                "coupled_boundary_match": oracle["coupled_spectral_boundary"]
                == production["coupled_spectral_boundary"],
                "support_minimum_match": oracle["minimum_support"]
                == production["minimum_final_port_support"],
                "support_maximum_match": oracle["maximum_support"]
                == production["maximum_final_port_support"],
                "coefficient_inverse_restoration": oracle[
                    "coefficient_inverse_restoration"
                ],
                "spectral_inverse_restoration": oracle[
                    "spectral_inverse_restoration"
                ],
                "coefficient_equals_coupled_spectral": oracle[
                    "coefficient_equals_coupled_spectral"
                ],
                "phase_work_match": oracle["phase_work"]
                == production["phase_stats"][
                    "total_nonlinear_core_multiplications"
                ],
                "spectral_work_match": oracle["spectral_work"]
                == production["coupled_spectral_stats"][
                    "total_nonlinear_core_multiplications"
                ],
            }
            if not all(checks.values()):
                fail(
                    f"quadratic oracle mismatch for {family} depth {depth}: {checks}"
                )
            comparisons += len(checks)
            oracle_cases.append(
                {
                    "depth": depth,
                    "family": family,
                    "final_commitment": oracle["final_commitment"],
                    "boundary_commitment": hashlib.sha256(
                        bytes(oracle["boundary"])
                    ).hexdigest(),
                    "checks": checks,
                }
            )

    control_results = controls()
    if not all(control_results.values()):
        fail("one or more independent quadratic controls failed")
    repeated = repeated_reuse()
    if not (
        repeated["exact_restoration"]
        and repeated["stable_boundary_count"] == 1
    ):
        fail("independent repeated reuse failed")

    return {
        "schema": "CAT_CAS_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_source_sha256": production_hash,
        "production_result_sha256": hashlib.sha256(
            package_path.read_bytes()
        ).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "independence": {
            "imports_production": False,
            "imports_m150_dependency": False,
            "uses_numpy": False,
            "implementation": "PURE_PYTHON867_COORDINATE_AND_COUPLED17_MODE_REEXECUTION",
            "shared_inputs": "SEALED_RESULT_AND_PUBLIC_FORMULAS_ONLY",
        },
        "scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(oracle_cases),
            "independent_comparison_count": comparisons,
        },
        "controls": control_results,
        "repeated_reuse": repeated,
        "cases": oracle_cases,
        "observed_resource_law": {
            "coefficient_convolution_multiplications_per_shear": CYCLE
            * CYCLE,
            "coefficient_quadratic_multiplications_per_shared_layer": CYCLE,
            "spectral_convolution_multiplications_per_shear": CYCLE,
            "spectral_quadratic_multiplications_per_shared_layer": CYCLE
            * CYCLE,
            "resident_field_coordinates_all_paths": CYCLE * 3 * CYCLE,
            "independent_spectral_modes_close": False,
            "identical_coefficient_classical_recurrence_executes": True,
            "coupled17_mode_classical_recurrence_executes": True,
        },
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_classification": "NO_RESTORATION_CLAIM",
        "claim_ceiling": package["claim_ceiling"],
        "preserved_subclaims": [
            "GENERAL_MULTI_COORDINATE_F103_C17_SUPERPOSITION",
            "NATIVE_CONVOLUTION_AND_COEFFICIENTWISE_QUADRATIC_INTERLEAVING",
            "NONCOMMUTING_MULTI_CONSUMER_RESIDENT_PORT",
            "FINAL_ONLY_BOUNDARY",
            "EXACT_RESTORATION_AND_REUSE",
            "INDEPENDENT17_MODE_FACTORIZATION_REJECTED",
            "EXACT_COUPLED17_MODE_RECURRENCE",
        ],
        "rejected_interpretations": package["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.package, arguments.production)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
