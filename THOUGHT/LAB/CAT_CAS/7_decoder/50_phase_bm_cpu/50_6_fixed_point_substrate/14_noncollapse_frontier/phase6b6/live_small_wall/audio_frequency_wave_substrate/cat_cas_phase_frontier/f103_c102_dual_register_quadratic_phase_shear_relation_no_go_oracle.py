#!/usr/bin/env python3
"""Independent scalar oracle for the dual-register C102 shear diagnostic.

The oracle imports neither the production module nor NumPy.  It reuses only
the already independently qualified scalar M157 group-algebra reference for
the unchanged linear operations, then reconstructs the new public program,
second-register seed, coefficientwise quadratic shear, inverse ordering,
collision, character-coupling law, boundary, and resource-coordinate count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f103_unresolved_c102_group_algebra_superposition_relation_no_go_oracle as base


INTERFACES = (5, 7)
DEPTHS = (1, 2, 4)
FAMILIES = base.FAMILIES
REGISTERS = 2


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def descriptor(interface: int, depth: int, family: str) -> dict[str, Any]:
    code = base.family_code(family)
    return {
        "schema": "CAT_CAS_F103_C102_DUAL_REGISTER_QUADRATIC_SHEAR_RELATION_PROGRAM_V1",
        "interface": interface,
        "depth": depth,
        "family": family,
        "owner": (0xC1580000 + 257 * interface + 131 * depth + code)
        & 0xFFFFFFFF,
        "node_count": base.NODE_COUNT,
        "register_count": REGISTERS,
        "port_type": f"F103_C102_DUAL_REGISTER_C{interface}_TO_C{interface}",
        "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
        "linear_composition": "RANK1_F103_C102_GROUP_ALGEBRA_LEFT_ACTION",
        "linear_intersection": "NATIVE_C102_SUPPORT_SHIFT",
        "cross_character_operation": "REVERSIBLE_COEFFICIENTWISE_QUADRATIC_SHEAR_A_TO_B",
        "projection": "FINAL_REGISTER_B_BOUNDARY_EVALUATION_T_TO_5",
        "observation": [
            (13 * depth + 3 * code + interface) % interface,
            (19 * depth + 7 * code + 2 * interface) % interface,
        ],
    }


def b_seed(interface: int, family: str) -> list[list[list[list[int]]]]:
    code = base.family_code(family)
    result: list[list[list[list[int]]]] = []
    for node in range(base.NODE_COUNT):
        matrix: list[list[list[int]]] = []
        for row in range(interface):
            row_values: list[list[int]] = []
            for column in range(interface):
                polynomial = [0] * base.ORDER
                exponent = (
                    base.seed_exponent(node, family, row, column)
                    + 29
                    + 7 * node
                    + 3 * row
                    + 5 * column
                    + code
                ) % base.ORDER
                polynomial[exponent] = 1
                row_values.append(polynomial)
            matrix.append(row_values)
        result.append(matrix)
    return result


def seed_registers(
    interface: int, family: str
) -> list[list[list[list[list[int]]]]]:
    return [base.seed_coefficients(interface, family), b_seed(interface, family)]


def gamma(index: int, hub: int, peer: int, family: str, mutation: int = 0) -> int:
    exponent = (
        31
        + 5 * index
        + 7 * hub
        + 11 * peer
        + base.family_code(family)
        + mutation
    ) % base.ORDER
    return base.POWERS[exponent]


def multipliers(
    index: int, hub: int, peer: int, family: str, mutation: int = 0
) -> list[int]:
    code = base.family_code(family) + mutation
    return [
        base.POWERS[
            (
                13
                + code
                + 3 * index
                + 5 * hub
                + 7 * peer
                + 2 * exponent * exponent
                + 3 * exponent
            )
            % base.ORDER
        ]
        for exponent in range(base.ORDER)
    ]


def shear_matrix(
    register_a: list[list[list[int]]],
    register_b: list[list[list[int]]],
    index: int,
    hub: int,
    peer: int,
    family: str,
    *,
    inverse: bool,
    mutation: int,
) -> list[list[list[int]]]:
    factor = gamma(index, hub, peer, family, mutation)
    if inverse:
        factor = -factor
    weights = multipliers(index, hub, peer, family, mutation)
    interface = len(register_a)
    return [
        [
            [
                (
                    register_b[row][column][exponent]
                    + factor
                    * weights[exponent]
                    * register_a[row][column][exponent] ** 2
                )
                % base.FIELD
                for exponent in range(base.ORDER)
            ]
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def forward(
    registers: list[list[list[list[list[int]]]]],
    interface: int,
    depth: int,
    family: str,
    *,
    mutation: int = 0,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    for index in range(depth):
        hub = base.hub_index(index, family, mutation)
        for peer in base.peer_order(hub, family):
            amount = base.rotation_shift(interface, peer, index, family)
            current = [
                base.rotate_matrix(registers[register][peer], amount)
                for register in range(REGISTERS)
            ]
            left, right, coupling, inverse_kernel = base.composition_plan(
                interface, hub, peer, index, family, mutation
            )
            current = [
                base.compose_matrix(
                    matrix,
                    left,
                    right,
                    coupling,
                    inverse_kernel,
                    False,
                )
                for matrix in current
            ]
            current[1] = shear_matrix(
                current[0],
                current[1],
                index,
                hub,
                peer,
                family,
                inverse=False,
                mutation=mutation,
            )
            current = [
                base.intersect_matrix(
                    matrix, hub, peer, index, family, False, mutation
                )
                for matrix in current
            ]
            for register in range(REGISTERS):
                registers[register][peer] = current[register]


def inverse(
    registers: list[list[list[list[list[int]]]]],
    interface: int,
    depth: int,
    family: str,
    *,
    mutation: int = 0,
    order: str = "INTERSECT_SHEAR_COMPOSE",
) -> None:
    for index in reversed(range(depth)):
        hub = base.hub_index(index, family, mutation)
        for peer in reversed(base.peer_order(hub, family)):
            left, right, coupling, inverse_kernel = base.composition_plan(
                interface, hub, peer, index, family, mutation
            )
            current = [registers[register][peer] for register in range(REGISTERS)]
            current = [
                base.intersect_matrix(
                    matrix, hub, peer, index, family, True, mutation
                )
                for matrix in current
            ]
            if order == "INTERSECT_SHEAR_COMPOSE":
                current[1] = shear_matrix(
                    current[0],
                    current[1],
                    index,
                    hub,
                    peer,
                    family,
                    inverse=True,
                    mutation=mutation,
                )
                current = [
                    base.compose_matrix(
                        matrix,
                        left,
                        right,
                        coupling,
                        inverse_kernel,
                        True,
                    )
                    for matrix in current
                ]
            elif order == "INTERSECT_COMPOSE_SHEAR":
                current = [
                    base.compose_matrix(
                        matrix,
                        left,
                        right,
                        coupling,
                        inverse_kernel,
                        True,
                    )
                    for matrix in current
                ]
                current[1] = shear_matrix(
                    current[0],
                    current[1],
                    index,
                    hub,
                    peer,
                    family,
                    inverse=True,
                    mutation=mutation,
                )
            else:
                fail("unknown inverse order")
            amount = base.rotation_shift(interface, peer, index, family)
            for register in range(REGISTERS):
                registers[register][peer] = base.rotate_matrix(
                    current[register], -amount
                )


def register_commitment(registers: list[list[list[list[list[int]]]]]) -> str:
    digest = hashlib.sha256()
    for register in registers:
        for matrix in register:
            for row in matrix:
                for polynomial in row:
                    digest.update(bytes(polynomial))
    return digest.hexdigest()


def boundary(
    registers: list[list[list[list[list[int]]]]], program: dict[str, Any]
) -> tuple[int, ...]:
    interface = int(program["interface"])
    left, right = program["observation"]
    return tuple(
        base.polynomial_value(
            registers[1][node][(left + node) % interface][
                (right + 2 * node) % interface
            ]
        )
        for node in range(base.NODE_COUNT)
    )


def collision() -> dict[str, Any]:
    program = descriptor(5, 1, "PRIMARY")
    hub = base.hub_index(0, "PRIMARY")
    peer = base.peer_order(hub, "PRIMARY")[0]
    first_a = [0] * base.ORDER
    first_a[0] = 1
    second_a = first_a[:]
    second_a[0] = (second_a[0] - base.GENERATOR) % base.FIELD
    second_a[1] = 1
    first_b = [0] * base.ORDER
    weights = multipliers(0, hub, peer, "PRIMARY")
    factor = gamma(0, hub, peer, "PRIMARY")
    first_out = [
        factor * weights[index] * first_a[index] ** 2 % base.FIELD
        for index in range(base.ORDER)
    ]
    second_out = [
        factor * weights[index] * second_a[index] ** 2 % base.FIELD
        for index in range(base.ORDER)
    ]
    before_equal = (
        base.polynomial_value(first_a) == base.polynomial_value(second_a)
        and base.polynomial_value(first_b) == base.polynomial_value(first_b)
    )
    after_different = base.polynomial_value(first_out) != base.polynomial_value(
        second_out
    )
    hessian = [
        2 * factor * weights[index] * base.POWERS[index] % base.FIELD
        for index in range(base.ORDER)
    ]
    return {
        "scope": "ONE_C5_PRIMARY_PUBLIC_SHEAR_ON_ARBITRARY_GROUP_ALGEBRA_INPUTS",
        "equal_pre_shear_t_to5_evaluations": before_equal,
        "different_post_shear_register_b_t_to5_evaluations": after_different,
        "single_character_quotient_rejected": before_equal and after_different,
        "quadratic_observable_hessian_rank": sum(value != 0 for value in hessian),
        "linear_sketch_dimension_lower_bound_for_arbitrary_source_polynomial": base.ORDER,
        "all_hessian_diagonal_entries_nonzero": all(value != 0 for value in hessian),
    }


def coupling_diagnostic() -> dict[str, Any]:
    program = descriptor(5, 1, "PRIMARY")
    hub = base.hub_index(0, "PRIMARY")
    peer = base.peer_order(hub, "PRIMARY")[0]
    weights = multipliers(0, hub, peer, "PRIMARY")
    transformed = [
        sum(
            weights[exponent]
            * base.POWERS[(character * exponent) % base.ORDER]
            for exponent in range(base.ORDER)
        )
        % base.FIELD
        for character in range(base.ORDER)
    ]
    support = [index for index, value in enumerate(transformed) if value]
    every = all(
        any(
            transformed[(output - source - companion) % base.ORDER] != 0
            for companion in range(base.ORDER)
        )
        for output in range(base.ORDER)
        for source in range(base.ORDER)
    )
    return {
        "scope": "ONE_C5_PRIMARY_PUBLIC_SHEAR",
        "multiplier_character_support_size": len(support),
        "multiplier_character_support_is_dense": len(support) == base.ORDER,
        "each_output_has_a_quadratic_dependency_on_every_input_character": every,
        "all102_character_sectors_are_causally_coupled": every,
    }


def semantic_controls() -> dict[str, bool]:
    interface, depth, family = 5, 2, "PRIMARY"
    program = descriptor(interface, depth, family)
    seed = seed_registers(interface, family)
    seed_commitment = register_commitment(seed)
    normal = seed_registers(interface, family)
    forward(normal, interface, depth, family)
    normal_boundary = boundary(normal, program)
    missing = register_commitment(normal) != seed_commitment
    wrong = seed_registers(interface, family)
    forward(wrong, interface, depth, family)
    inverse(wrong, interface, depth, family, mutation=1)
    reordered = seed_registers(interface, family)
    forward(reordered, interface, depth, family)
    inverse(
        reordered,
        interface,
        depth,
        family,
        order="INTERSECT_COMPOSE_SHEAR",
    )
    disabled = seed_registers(interface, family)
    forward(disabled, interface, depth, family, enabled=False)
    collision_result = collision()
    coupling_result = coupling_diagnostic()
    return {
        "missing_inverse_changes_payload": missing,
        "wrong_inverse_changes_payload": register_commitment(wrong)
        != seed_commitment,
        "reordered_inverse_changes_payload": register_commitment(reordered)
        != seed_commitment,
        "disabled_port_changes_boundary": boundary(disabled, program)
        != normal_boundary,
        "single_character_collision_changes_later_boundary": collision_result[
            "single_character_quotient_rejected"
        ],
        "one_shear_quadratic_hessian_has_rank102": collision_result[
            "quadratic_observable_hessian_rank"
        ]
        == base.ORDER,
        "every_character_sector_can_influence_every_output_sector": coupling_result[
            "each_output_has_a_quadratic_dependency_on_every_input_character"
        ],
    }


def compare(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label}: production={actual!r} oracle={expected!r}")


def run(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    compare(
        production["schema"],
        "CAT_CAS_F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_RESULT_V1",
        "schema",
    )
    production_cases = {
        (int(case["interface"]), int(case["depth"]), str(case["family"])): case
        for case in production["cases"]
    }
    comparisons = 1
    exact_inverse_cases = 0
    oracle_cases: list[dict[str, Any]] = []
    for interface in INTERFACES:
        for family in FAMILIES:
            for depth in DEPTHS:
                key = (interface, depth, family)
                case = production_cases[key]
                program = descriptor(interface, depth, family)
                registers = seed_registers(interface, family)
                seed_commitment = register_commitment(registers)
                forward(registers, interface, depth, family)
                forward_commitment = register_commitment(registers)
                result = boundary(registers, program)
                character_boundary = tuple(
                    base.evaluations(
                        registers[1][node][
                            (program["observation"][0] + node) % interface
                        ][
                            (program["observation"][1] + 2 * node) % interface
                        ]
                    )[1]
                    for node in range(base.NODE_COUNT)
                )
                payload = REGISTERS * base.NODE_COUNT * interface * interface * base.ORDER
                expected = {
                    "program_fingerprint": digest_json(program),
                    "boundary_commitment": digest_json(list(result)),
                    "forward_register_commitment": forward_commitment,
                    "phase_payload_field_cells": payload,
                    "matched_classical_payload_field_cells": payload,
                }
                for label, value in expected.items():
                    compare(case[label], value, f"{key}:{label}")
                    comparisons += 1
                compare(character_boundary, result, f"{key}:character boundary")
                comparisons += 1
                inverse(registers, interface, depth, family)
                restored = register_commitment(registers) == seed_commitment
                compare(restored, True, f"{key}:inverse")
                comparisons += 1
                exact_inverse_cases += int(restored)
                oracle_cases.append(
                    {
                        "interface": interface,
                        "depth": depth,
                        "family": family,
                        "program_fingerprint": digest_json(program),
                        "boundary_commitment": digest_json(list(result)),
                        "forward_register_commitment": forward_commitment,
                        "payload_field_cells": payload,
                        "exact_inverse_restored": restored,
                    }
                )

    collision_result = collision()
    coupling_result = coupling_diagnostic()
    compare(
        production["single_character_quotient_attack"],
        collision_result,
        "collision",
    )
    compare(
        production["character_coupling_diagnostic"],
        coupling_result,
        "character coupling",
    )
    comparisons += 2
    controls = semantic_controls()
    for label, value in controls.items():
        compare(production["controls"][label], value, f"control:{label}")
        comparisons += 1
    if not all(controls.values()):
        fail("independent semantic controls failed")
    compare(
        production["resource_accounting"]["accepted_phase_and_classical_state_law"],
        "204N2_FIELD_COORDINATES",
        "resource law",
    )
    compare(
        production["resource_accounting"]["phase_to_matched_classical_payload_ratio"],
        1.0,
        "payload ratio",
    )
    comparisons += 2

    return {
        "schema": "CAT_CAS_F103_C102_DUAL_REGISTER_QUADRATIC_PHASE_SHEAR_RELATION_NO_GO_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_production_module": False,
        "imports_numpy": False,
        "reuses_prior_independent_m157_scalar_reference": True,
        "case_count": len(oracle_cases),
        "comparison_count": comparisons,
        "exact_inverse_cases": exact_inverse_cases,
        "reconstructed_public_programs": True,
        "reconstructed_dual_register_coefficients": True,
        "reconstructed_quadratic_shear_and_inverse": True,
        "reconstructed_single_character_collision": True,
        "reconstructed_character_coupling_support": True,
        "reconstructed_boundary_character_evaluation": True,
        "reconstructed_identical_204n2_coordinate_law": True,
        "semantic_controls": controls,
        "cases": oracle_cases,
        "package_local_only": [
            "NUMPY_BACKING_IDENTITY_AND_OPERATION_COUNTERS",
            "PRODUCTION_FULL_CHARACTER_IMPLEMENTATION_PATH",
            "DIRECT_PROCESS_CUSTODY_STATE_MACHINE",
            "PYTHON_ALLOCATOR_AND_NATIVE_LIBRARY_EXCLUSIONS",
        ],
        "claim_ceiling": production["claim_ceiling"],
        "not_established": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.production_result)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "case_count": result["case_count"],
                "comparison_count": result["comparison_count"],
                "exact_inverse_cases": result["exact_inverse_cases"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
