#!/usr/bin/env python3
"""Independent scalar oracle for the M160 column-sign quotient.

The oracle imports neither M160 production, M158 production, nor NumPy.  It
reuses the separately qualified scalar M158 reference for the underlying raw
recurrence and independently reconstructs canonical representatives,
orientation transport, quotient forward/inverse execution, and raw unsealing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import f103_c102_dual_register_quadratic_phase_shear_relation_no_go_oracle as shear


base = shear.base
INTERFACE = 5
DEPTHS = (1, 4, 16, 64)
FAMILIES = shear.FAMILIES
ORIENTATION_BITS = base.NODE_COUNT * INTERFACE


def fail(message: str) -> None:
    raise RuntimeError(message)


def descriptor(depth: int, family: str) -> dict[str, Any]:
    code = base.family_code(family)
    return {
        "interface": INTERFACE,
        "depth": depth,
        "family": family,
        "owner": (0xC1600000 + 131 * depth + code) & 0xFFFFFFFF,
        "observation": [
            (13 * depth + 3 * code + INTERFACE) % INTERFACE,
            (19 * depth + 7 * code + 2 * INTERFACE) % INTERFACE,
        ],
    }


def clone_registers(registers: list[Any]) -> list[Any]:
    return [
        [
            [[polynomial[:] for polynomial in row] for row in matrix]
            for matrix in register
        ]
        for register in registers
    ]


def bit_index(node: int, column: int) -> int:
    return node * INTERFACE + column


def get_signs(mask: int, node: int) -> list[int]:
    return [(mask >> bit_index(node, column)) & 1 for column in range(INTERFACE)]


def set_signs(mask: int, node: int, signs: list[int]) -> int:
    for column, value in enumerate(signs):
        bit = 1 << bit_index(node, column)
        mask = mask | bit if value else mask & ~bit
    return mask


def roll_signs(signs: list[int], amount: int) -> list[int]:
    return [signs[(column - amount) % INTERFACE] for column in range(INTERFACE)]


def canonicalize_columns(
    matrix: list[list[list[int]]], signs: list[int]
) -> tuple[list[list[list[int]]], list[int]]:
    result = [[polynomial[:] for polynomial in row] for row in matrix]
    updated = signs[:]
    for column in range(INTERFACE):
        flat = [
            value
            for row in range(INTERFACE)
            for value in result[row][column]
        ]
        negative = [(-value) % base.FIELD for value in flat]
        if bytes(negative) < bytes(flat):
            offset = 0
            for row in range(INTERFACE):
                result[row][column] = negative[offset : offset + base.ORDER]
                offset += base.ORDER
            updated[column] ^= 1
    return result, updated


def canonicalize_all(registers: list[Any]) -> int:
    mask = 0
    for node in range(base.NODE_COUNT):
        canonical, signs = canonicalize_columns(
            registers[0][node], [0] * INTERFACE
        )
        registers[0][node] = canonical
        mask = set_signs(mask, node, signs)
    return mask


def apply_orientation(registers: list[Any], mask: int) -> None:
    for node in range(base.NODE_COUNT):
        signs = get_signs(mask, node)
        for column in range(INTERFACE):
            if signs[column]:
                for row in range(INTERFACE):
                    registers[0][node][row][column] = [
                        (-value) % base.FIELD
                        for value in registers[0][node][row][column]
                    ]


def negate_column(registers: list[Any], node: int, column: int) -> None:
    for row in range(INTERFACE):
        registers[0][node][row][column] = [
            (-value) % base.FIELD
            for value in registers[0][node][row][column]
        ]


def oriented_seed(family: str) -> list[Any]:
    registers = shear.seed_registers(INTERFACE, family)
    code = base.family_code(family)
    for node in range(base.NODE_COUNT):
        for column in range(INTERFACE):
            if (3 * node + 5 * column + code) % 4 in (1, 2):
                negate_column(registers, node, column)
    return registers


def recanonicalize(
    registers: list[Any], mask: int, node: int, matrix: list[Any], signs: list[int]
) -> tuple[list[Any], int]:
    canonical, updated = canonicalize_columns(matrix, signs)
    return canonical, set_signs(mask, node, updated)


def forward(registers: list[Any], mask: int, depth: int, family: str) -> int:
    for index in range(depth):
        hub = base.hub_index(index, family)
        for peer in base.peer_order(hub, family):
            signs = get_signs(mask, peer)
            amount = base.rotation_shift(INTERFACE, peer, index, family)
            current = [
                base.rotate_matrix(registers[register][peer], amount)
                for register in range(2)
            ]
            signs = roll_signs(signs, amount)
            left, right, coupling, inverse_kernel = base.composition_plan(
                INTERFACE, hub, peer, index, family, 0
            )
            current = [
                base.compose_matrix(
                    matrix, left, right, coupling, inverse_kernel, False
                )
                for matrix in current
            ]
            current[0], mask = recanonicalize(
                registers, mask, peer, current[0], signs
            )
            signs = get_signs(mask, peer)
            current[1] = shear.shear_matrix(
                current[0],
                current[1],
                index,
                hub,
                peer,
                family,
                inverse=False,
                mutation=0,
            )
            current = [
                base.intersect_matrix(matrix, hub, peer, index, family, False, 0)
                for matrix in current
            ]
            current[0], mask = recanonicalize(
                registers, mask, peer, current[0], signs
            )
            for register in range(2):
                registers[register][peer] = current[register]
    return mask


def inverse(registers: list[Any], mask: int, depth: int, family: str) -> int:
    for index in reversed(range(depth)):
        hub = base.hub_index(index, family)
        for peer in reversed(base.peer_order(hub, family)):
            signs = get_signs(mask, peer)
            left, right, coupling, inverse_kernel = base.composition_plan(
                INTERFACE, hub, peer, index, family, 0
            )
            current = [
                base.intersect_matrix(
                    registers[register][peer], hub, peer, index, family, True, 0
                )
                for register in range(2)
            ]
            current[0], mask = recanonicalize(
                registers, mask, peer, current[0], signs
            )
            signs = get_signs(mask, peer)
            current[1] = shear.shear_matrix(
                current[0],
                current[1],
                index,
                hub,
                peer,
                family,
                inverse=True,
                mutation=0,
            )
            current = [
                base.compose_matrix(
                    matrix, left, right, coupling, inverse_kernel, True
                )
                for matrix in current
            ]
            current[0], mask = recanonicalize(
                registers, mask, peer, current[0], signs
            )
            signs = get_signs(mask, peer)
            amount = base.rotation_shift(INTERFACE, peer, index, family)
            current = [base.rotate_matrix(matrix, -amount) for matrix in current]
            signs = roll_signs(signs, -amount)
            current[0], mask = recanonicalize(
                registers, mask, peer, current[0], signs
            )
            for register in range(2):
                registers[register][peer] = current[register]
    return mask


def boundary(registers: list[Any], program: dict[str, Any]) -> tuple[int, ...]:
    return shear.boundary(registers, program)


def case(depth: int, family: str) -> dict[str, Any]:
    program = descriptor(depth, family)
    original = oriented_seed(family)
    original_commitment = shear.register_commitment(original)
    quotient = clone_registers(original)
    mask = canonicalize_all(quotient)
    seal_mask_bits = mask.bit_count()
    sealed_commitment = shear.register_commitment(quotient)
    sealed_mask = mask
    mask = forward(quotient, mask, depth, family)
    quotient_boundary = boundary(quotient, program)
    mask = inverse(quotient, mask, depth, family)
    exact_sealed_restoration = (
        shear.register_commitment(quotient) == sealed_commitment
        and mask == sealed_mask
    )
    apply_orientation(quotient, mask)
    exact_raw_restoration = shear.register_commitment(quotient) == original_commitment

    baseline = oriented_seed(family)
    shear.forward(baseline, INTERFACE, depth, family)
    baseline_boundary = boundary(baseline, program)
    return {
        "depth": depth,
        "family": family,
        "boundary_commitment": hashlib.sha256(bytes(quotient_boundary)).hexdigest(),
        "orientation_bits_set_at_seal": seal_mask_bits,
        "matches_raw_identical_coefficient_recurrence": quotient_boundary
        == baseline_boundary,
        "exact_sealed_restoration": exact_sealed_restoration,
        "exact_raw_payload_restored_after_unseal": exact_raw_restoration,
    }


def generator_control() -> dict[str, Any]:
    family = "PRIMARY"
    program = descriptor(4, family)
    plain = shear.seed_registers(INTERFACE, family)
    plain_representative = clone_registers(plain)
    canonicalize_all(plain_representative)
    shear.forward(plain, INTERFACE, 4, family)
    plain_boundary = boundary(plain, program)
    checks = []
    for node in range(base.NODE_COUNT):
        for column in range(INTERFACE):
            variant = shear.seed_registers(INTERFACE, family)
            negate_column(variant, node, column)
            original_commitment = shear.register_commitment(variant)
            representative = clone_registers(variant)
            mask = canonicalize_all(representative)
            same_representative = representative == plain_representative
            apply_orientation(representative, mask)
            restored = shear.register_commitment(representative) == original_commitment
            shear.forward(variant, INTERFACE, 4, family)
            same_boundary = boundary(variant, program) == plain_boundary
            checks.append(same_representative and restored and same_boundary)
    return {
        "tested_generators": len(checks),
        "all_collapse_preserve_boundary_and_restore": all(checks),
        "free_nonzero_seed_orbit_order": 2**ORIENTATION_BITS,
        "minimum_arbitrary_member_restoration_ledger_bits": ORIENTATION_BITS,
    }


def compare_production(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    production = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        (item["depth"], item["family"]): item for item in production["cases"]
    }
    comparisons = 0
    for observed in cases:
        wanted = expected[(observed["depth"], observed["family"])]
        for field in (
            "boundary_commitment",
            "orientation_bits_set_at_seal",
            "matches_raw_identical_coefficient_recurrence",
            "exact_raw_payload_restored_after_unseal",
        ):
            comparisons += 1
            if observed[field] != wanted[field]:
                fail(f"production mismatch for {observed['family']} depth {observed['depth']} {field}")
    return {"comparisons": comparisons, "all_match": True}


def run(production_path: Path) -> dict[str, Any]:
    cases = [case(depth, family) for family in FAMILIES for depth in DEPTHS]
    if not all(
        item["matches_raw_identical_coefficient_recurrence"]
        and item["exact_sealed_restoration"]
        and item["exact_raw_payload_restored_after_unseal"]
        for item in cases
    ):
        fail("independent quotient case failed")
    generators = generator_control()
    if not generators["all_collapse_preserve_boundary_and_restore"]:
        fail("independent sign generator control failed")
    return {
        "schema": "CAT_CAS_F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_m160_production": False,
        "imports_m158_production": False,
        "imports_numpy": False,
        "reuses_prior_independent_m158_scalar_reference": True,
        "cases": cases,
        "generator_control": generators,
        "production_comparison": compare_production(production_path, cases),
        "resource_reconstruction": {
            "canonical_representative_field_cells": 45900,
            "matched_raw_field_cells": 45900,
            "material_field_cell_reduction": 0,
            "orientation_ledger_bits": ORIENTATION_BITS,
            "net_lossless_information_reduction_bits": 0,
        },
        "claim_ceiling": "INDEPENDENT_SCALAR_PARITY_FOR_M160_C5_TWO_FAMILIES_DEPTHS1_4_16_64_AND_ALL45_SIGN_GENERATORS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.production)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "cases": len(result["cases"]),
                    "generator_control": result["generator_control"],
                    "production_comparison": result["production_comparison"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
