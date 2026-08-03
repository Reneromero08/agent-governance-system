#!/usr/bin/env python3
"""Independent pure-Python oracle for the F103[C17] superposition package.

The oracle imports neither the production package nor its M149 dependency.  It
reconstructs the public schedule, executes the 17 independent spectral modes,
inverts the NTT to all 867 coefficient cells, and compares commitments and
boundaries from the sealed production result.
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


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


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


def spectral_shear(
    destination: list[list[int]],
    control: list[int],
    *,
    inverse: bool,
) -> None:
    sign = -1 if inverse else 1
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
            (destination[1][mode] + sign * control[mode] * destination[0][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]
        destination[2] = [
            (destination[2][mode] + sign * control[mode] * destination[1][mode])
            % MODULUS
            for mode in range(CYCLE)
        ]


def apply_edge(
    modes: list[list[list[int]]],
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool,
    port_enabled: bool,
    offset_mutation: int,
) -> None:
    if port_enabled:
        control_slot = 0 if layer == 0 else 1
        control = rotate_modes(
            modes[controller][control_slot],
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
    spectral_shear(modes[target], control, inverse=inverse)


def apply_layer(
    modes: list[list[list[int]]],
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool,
    port_enabled: bool,
    hub_mutation: int,
    offset_mutation: int,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    peers = list(target_order(hub))
    if inverse:
        peers.reverse()
    for peer in peers:
        controller, target = (hub, peer) if layer == 0 else (peer, hub)
        apply_edge(
            modes,
            controller,
            target,
            index,
            family,
            layer,
            inverse=inverse,
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
        )


def rotate_all(
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
    order: str = "OUT_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    layers = (0, 1) if order == "OUT_IN" else (1, 0)
    for index in range(depth):
        rotate_all(modes, index, family, inverse=False)
        for layer in layers:
            apply_layer(
                modes,
                index,
                family,
                layer,
                inverse=False,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )


def inverse_modes(
    modes: list[list[list[int]]],
    depth: int,
    family: str,
    *,
    assumed_order: str = "OUT_IN",
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    layers = (1, 0) if assumed_order == "OUT_IN" else (0, 1)
    for index in reversed(range(depth)):
        for layer in layers:
            apply_layer(
                modes,
                index,
                family,
                layer,
                inverse=True,
                port_enabled=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )
        rotate_all(modes, index, family, inverse=True)


def flatten_bytes(state: list[list[list[int]]]) -> bytes:
    return bytes(
        value
        for shell in state
        for slot in shell
        for value in slot
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


def coefficient_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * CYCLE
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[(left_index + right_index) % CYCLE] = (
                result[(left_index + right_index) % CYCLE]
                + left_value * right_value
            ) % MODULUS
    return result


def execute_case(depth: int, family: str) -> dict[str, Any]:
    seed = seed_coefficients()
    modes = transform_state(seed)
    forward_modes(modes, depth, family)
    final_state = inverse_transform_state(modes)
    final_commitment = commitment(final_state)
    final_boundary = boundary(final_state, depth, family)
    support = [
        sum(value != 0 for value in final_state[shell][slot])
        for shell in range(CYCLE)
        for slot in range(3)
    ]
    inverse_modes(modes, depth, family)
    restored = inverse_transform_state(modes)
    return {
        "depth": depth,
        "family": family,
        "final_commitment": final_commitment,
        "boundary": final_boundary,
        "minimum_support": min(support),
        "maximum_support": max(support),
        "exact_spectral_inverse_restoration": restored == seed,
        "modal_multiplications": depth * 32 * 2 * CYCLE,
        "coefficient_multiplications": depth * 32 * 2 * CYCLE * CYCLE,
    }


def controls() -> dict[str, bool]:
    seed = seed_coefficients()
    seed_modes = transform_state(seed)

    normal = copy.deepcopy(seed_modes)
    forward_modes(normal, 4, "PRIMARY")
    correct = copy.deepcopy(normal)
    inverse_modes(correct, 4, "PRIMARY")

    wrong = copy.deepcopy(normal)
    inverse_modes(wrong, 4, "PRIMARY", offset_mutation=1)

    reordered = copy.deepcopy(normal)
    inverse_modes(reordered, 4, "PRIMARY", assumed_order="IN_OUT")

    null_port = copy.deepcopy(seed_modes)
    forward_modes(null_port, 4, "PRIMARY", port_enabled=False)

    swapped = copy.deepcopy(seed_modes)
    forward_modes(swapped, 4, "PRIMARY", order="IN_OUT")

    mutated = copy.deepcopy(seed_modes)
    forward_modes(mutated, 4, "PRIMARY", hub_mutation=1)

    left = [0] * CYCLE
    right = [0] * CYCLE
    left[0] = 1
    left[1] = 1
    right[0] = 1
    right[1] = MODULUS - 1
    product = coefficient_convolution(left, right)

    roundtrip = all(
        inverse_transform(transform([int(index == coordinate) for index in range(CYCLE)]))
        == [int(index == coordinate) for index in range(CYCLE)]
        for coordinate in range(CYCLE)
    )
    convolution_theorem = transform(product) == [
        transform(left)[mode] * transform(right)[mode] % MODULUS
        for mode in range(CYCLE)
    ]
    seed_commitment = commitment(seed)
    return {
        "all17_basis_ntt_roundtrip": roundtrip,
        "coefficient_convolution_theorem": convolution_theorem,
        "correct_inverse_restores": commitment(inverse_transform_state(correct))
        == seed_commitment,
        "wrong_inverse_fails": commitment(inverse_transform_state(wrong))
        != seed_commitment,
        "reordered_inverse_fails": commitment(inverse_transform_state(reordered))
        != seed_commitment,
        "null_port_changes_boundary": boundary(
            inverse_transform_state(normal), 4, "PRIMARY"
        )
        != boundary(inverse_transform_state(null_port), 4, "PRIMARY"),
        "forward_order_noncommutes": boundary(
            inverse_transform_state(normal), 4, "PRIMARY"
        )
        != boundary(inverse_transform_state(swapped), 4, "PRIMARY"),
        "topology_mutation_changes_boundary": boundary(
            inverse_transform_state(normal), 4, "PRIMARY"
        )
        != boundary(inverse_transform_state(mutated), 4, "PRIMARY"),
        "explicit_destructive_interference": (
            product[0] == 1
            and product[1] == 0
            and product[2] == MODULUS - 1
        ),
    }


def repeated_reuse() -> dict[str, Any]:
    seed = seed_coefficients()
    seed_modes = transform_state(seed)
    seed_commitment = commitment(seed)
    boundaries: set[tuple[int, ...]] = set()
    modes = copy.deepcopy(seed_modes)
    for _ in range(100):
        forward_modes(modes, 16, "REUSE")
        final_state = inverse_transform_state(modes)
        boundaries.add(tuple(boundary(final_state, 16, "REUSE")))
        inverse_modes(modes, 16, "REUSE")
    return {
        "cycles": 100,
        "exact_restoration": commitment(inverse_transform_state(modes))
        == seed_commitment,
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
                "boundary_match": oracle["boundary"] == production["boundary"],
                "support_minimum_match": oracle["minimum_support"]
                == production["minimum_final_port_support"],
                "support_maximum_match": oracle["maximum_support"]
                == production["maximum_final_port_support"],
                "inverse_restoration": oracle[
                    "exact_spectral_inverse_restoration"
                ],
                "modal_work_match": oracle["modal_multiplications"]
                == production["spectral_stats"]["modal_multiplications"],
                "coefficient_work_match": oracle["coefficient_multiplications"]
                == production["phase_stats"]["coefficient_multiplications"],
            }
            if not all(checks.values()):
                fail(f"oracle case mismatch for {family} depth {depth}: {checks}")
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
        fail("one or more independent controls failed")
    repeated = repeated_reuse()
    if not (
        repeated["exact_restoration"]
        and repeated["stable_boundary_count"] == 1
    ):
        fail("independent repeated reuse failed")

    return {
        "schema": "CAT_CAS_F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_ORACLE_V1",
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
            "imports_m149_dependency": False,
            "implementation": "PURE_PYTHON17_MODE_NTT_RECURRENCE_AND_FULL867_CELL_RECONSTRUCTION",
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
            "phase_coefficient_convolution_multiplications_per_shear": CYCLE
            * CYCLE,
            "spectral_modal_multiplications_per_shear": CYCLE,
            "resident_field_coordinates_both_paths": CYCLE * 3 * CYCLE,
            "phase_to_spectral_resident_dimension_ratio": 1,
            "phase_to_spectral_convolution_work_ratio": CYCLE,
        },
        "claim_ceiling": package["claim_ceiling"],
        "preserved_subclaims": [
            "GENERAL_MULTI_COORDINATE_F103_C17_SUPERPOSITION",
            "NATIVE_CONVOLUTION_INTERFERENCE",
            "NONCOMMUTING_MULTI_CONSUMER_RESIDENT_PORT",
            "FINAL_ONLY_BOUNDARY",
            "EXACT_RESTORATION_AND_REUSE",
            "EXACT17_MODE_SPECTRAL_FACTORIZATION",
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
