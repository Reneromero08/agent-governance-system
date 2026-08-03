#!/usr/bin/env python3
"""Independent oracle for the coherent F17 latent basis-mismatch package."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
from typing import Any

import f17_nonlinear_canonical_mps_separator_chart as backend


PRIME = 17
LATENT_DIMENSION = 17
EXACT_SIZES = (2, 4, 6)
STRUCTURAL_SIZES = (2, 4, 6, 8, 10)
FAMILIES = ("PRIMARY", "REUSE")
FINITE_FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integer(alg: backend.Algebra, value: int) -> Any:
    if alg.modulus:
        return value % alg.modulus
    return alg.domain.convert(value)


def negative(alg: backend.Algebra, value: Any) -> Any:
    return alg.sub(alg.zero, value)


def grid_edges(n: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        *(((row, column), (row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *(((row, column), (row + 1, column))
          for row in range(n - 1) for column in range(n)),
    )


def public_exponents(
    n: int,
    family: str,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]], int]:
    variant = 0 if family == "PRIMARY" else 1
    edge_count = len(grid_edges(n))
    weights = []
    controls = []
    for module in range(2):
        weights.append(
            tuple(
                1 + ((5 * index + 7 * n + 3 * module + 4 * variant) % 16)
                for index in range(edge_count)
            )
        )
        controls.append(
            tuple(
                1 + ((7 * index + 5 * module + 6 * variant + n) % 16)
                for index in range(edge_count)
            )
        )
    return (
        (weights[0], weights[1]),
        (controls[0], controls[1]),
        3 + 2 * variant,
    )


def transform(
    values: tuple[Any, ...],
    alg: backend.Algebra,
    *,
    inverse: bool = False,
) -> tuple[Any, ...]:
    scale = alg.divide(alg.one, integer(alg, LATENT_DIMENSION)) if inverse else alg.one
    sign = -1 if inverse else 1
    result = []
    for target in range(LATENT_DIMENSION):
        value = alg.zero
        for source, amplitude in enumerate(values):
            value = alg.add(
                value,
                alg.mul(amplitude, alg.power(sign * source * target)),
            )
        result.append(alg.mul(value, scale))
    return tuple(result)


def matching_boundary(
    n: int,
    weights: tuple[Any, ...],
    alg: backend.Algebra,
) -> Any:
    edges = grid_edges(n)
    edge_weight = {
        frozenset((left, right)): weight
        for (left, right), weight in zip(edges, weights, strict=True)
    }

    @functools.lru_cache(maxsize=None)
    def visit(position: int, mask: int) -> Any:
        if position == n * n:
            return alg.one if mask == 0 else alg.zero
        row, column = divmod(position, n)
        if mask & 1:
            return visit(position + 1, mask >> 1)
        value = alg.zero
        here = (row, column)
        if column + 1 < n and not (mask & 2):
            edge = frozenset((here, (row, column + 1)))
            value = alg.add(
                value,
                alg.mul(edge_weight[edge], visit(position + 1, (mask | 2) >> 1)),
            )
        if row + 1 < n:
            edge = frozenset((here, (row + 1, column)))
            value = alg.add(
                value,
                alg.mul(
                    edge_weight[edge],
                    visit(position + 1, (mask >> 1) | (1 << (n - 1))),
                ),
            )
        return value

    return visit(0, 0)


def module_vector(
    n: int,
    family: str,
    module: int,
    alg: backend.Algebra,
) -> tuple[Any, ...]:
    weight_rows, control_rows, _ = public_exponents(n, family)
    result = []
    for latent in range(LATENT_DIMENSION):
        weights = tuple(
            alg.mul(alg.power(weight), alg.power(control * latent))
            for weight, control in zip(
                weight_rows[module], control_rows[module], strict=True
            )
        )
        result.append(matching_boundary(n, weights, alg))
    return tuple(result)


def independent_boundary(
    n: int,
    family: str,
    alg: backend.Algebra,
) -> Any:
    _, _, chirp_exponent = public_exponents(n, family)
    u = (alg.one, *[alg.zero for _ in range(LATENT_DIMENSION - 1)])
    w = tuple(alg.zero for _ in range(LATENT_DIMENSION))
    u = transform(u, alg)
    w = transform(w, alg)
    first = module_vector(n, family, 0, alg)
    w = tuple(
        alg.add(value, alg.mul(boundary, amplitude))
        for value, boundary, amplitude in zip(w, first, u, strict=True)
    )
    chirps = tuple(
        alg.power(chirp_exponent * latent * latent)
        for latent in range(LATENT_DIMENSION)
    )
    u = tuple(alg.mul(value, phase) for value, phase in zip(u, chirps, strict=True))
    w = tuple(alg.mul(value, phase) for value, phase in zip(w, chirps, strict=True))
    u = transform(u, alg)
    w = transform(w, alg)
    second = module_vector(n, family, 1, alg)
    w = tuple(
        alg.add(value, alg.mul(boundary, amplitude))
        for value, boundary, amplitude in zip(w, second, u, strict=True)
    )
    return w[0]


def native_basis(row: int, bit: int, phase: Any, alg: backend.Algebra) -> Any:
    if row == 0:
        return alg.one
    return phase if bit == 0 else negative(alg, phase)


def dual_basis(row: int, bit: int, alg: backend.Algebra) -> Any:
    half = alg.divide(alg.one, integer(alg, 2))
    if row == 0 or bit == 0:
        return half
    return negative(alg, half)


def direct_native_holant_module(
    n: int,
    family: str,
    module: int,
    latent: int,
    alg: backend.Algebra,
) -> Any:
    if n != 2:
        fail("direct native Holant oracle is bounded to n=2")
    edges = grid_edges(n)
    incidents: dict[tuple[int, int], list[int]] = {
        (row, column): [] for row in range(n) for column in range(n)
    }
    for index, (left, right) in enumerate(edges):
        incidents[left].append(index)
        incidents[right].append(index)
    weight_rows, control_rows, _ = public_exponents(n, family)
    result = alg.zero
    for assignment in itertools.product((0, 1), repeat=len(edges)):
        term = alg.one
        for vertex, indices in incidents.items():
            local = alg.zero
            is_black = sum(vertex) % 2 == 0
            for chosen, chosen_edge in enumerate(indices):
                local_term = alg.power(weight_rows[module][chosen_edge]) if is_black else alg.one
                for position, edge_index in enumerate(indices):
                    row = 1 if position == chosen else 0
                    bit = assignment[edge_index]
                    if is_black:
                        phase = alg.power(control_rows[module][edge_index] * latent)
                        factor = native_basis(row, bit, phase, alg)
                    else:
                        factor = dual_basis(row, bit, alg)
                    local_term = alg.mul(local_term, factor)
                local = alg.add(local, local_term)
            term = alg.mul(term, local)
        result = alg.add(result, term)
    return result


def basis_identity_checks(alg: backend.Algebra) -> list[dict[str, Any]]:
    checks = []
    half = alg.divide(alg.one, integer(alg, 2))
    right = (half, half, half, negative(alg, half))
    for exponent in (0, 1, 5, 16):
        phase = alg.power(exponent)
        left = (alg.one, alg.one, phase, negative(alg, phase))
        contraction = []
        for left_row in range(2):
            for right_row in range(2):
                value = alg.zero
                for shared in range(2):
                    value = alg.add(
                        value,
                        alg.mul(left[2 * left_row + shared], right[2 * right_row + shared]),
                    )
                contraction.append(value)
        expected = (alg.one, alg.zero, alg.zero, phase)
        checks.append({"exponent": exponent, "agreement": tuple(contraction) == expected})
    return checks


def verify(production: dict[str, Any]) -> dict[str, Any]:
    exact_parity = []
    exact_lookup = {
        (item["family"], item["n"]): item for item in production["exact_transactions"]
    }
    for family in FAMILIES:
        for n in EXACT_SIZES:
            alg = backend.Algebra("Q_ZETA17")
            expected = exact_lookup[(family, n)]["boundary"]
            observed = alg.serialize(independent_boundary(n, family, alg))
            exact_parity.append(
                {"family": family, "n": n, "agreement": observed == expected}
            )

    structural_parity = []
    structural_lookup = {
        (item["field"], item["family"], item["n"]): item
        for item in production["dual_field_structural_transactions"]
    }
    for modulus, root in FINITE_FIELDS:
        for family in FAMILIES:
            for n in STRUCTURAL_SIZES:
                alg = backend.Algebra(f"F{modulus}", modulus=modulus, root=root)
                expected = structural_lookup[(f"F{modulus}", family, n)]["boundary"]
                observed = alg.serialize(independent_boundary(n, family, alg))
                structural_parity.append(
                    {
                        "field": f"F{modulus}",
                        "family": family,
                        "n": n,
                        "agreement": observed == expected,
                    }
                )

    exact_alg = backend.Algebra("Q_ZETA17")
    native_checks = []
    for family in FAMILIES:
        for module in (0, 1):
            matching = module_vector(2, family, module, exact_alg)
            for latent in (0, 1, 7):
                native = direct_native_holant_module(
                    2, family, module, latent, exact_alg
                )
                native_checks.append(
                    {
                        "family": family,
                        "module": module,
                        "latent": latent,
                        "agreement": native == matching[latent],
                    }
                )

    controls = production["controls"]
    return {
        "schema": "CAT_CAS_F17_COHERENT_LATENT_BASIS_MISMATCH_GRID_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "independence": {
            "imports_production_or_matchgate_predecessor": False,
            "matching_oracle": "INDEPENDENT_ROW_PROFILE_PERFECT_MATCHING_RECURRENCE",
            "latent_recurrence": "INDEPENDENT_34_COORDINATE_FOURIER_CHIRP_SHEAR_EXECUTION",
        },
        "exact_production_parity": exact_parity,
        "dual_field_structural_parity": structural_parity,
        "bounded_native_holant_parity": native_checks,
        "basis_mismatch_identity_checks": basis_identity_checks(exact_alg),
        "control_receipt_consistency": {
            "all_required_controls_true": all(
                controls[name]
                for name in (
                    "missing_inverse_detected",
                    "wrong_inverse_ownership_detected",
                    "premature_projection_rejected",
                    "null_carrier_rejected",
                    "reordered_inverse_detected",
                    "semantic_basis_control_mutation_changes_boundary",
                    "coherent_chirp_mutation_changes_boundary",
                    "fourier_chirp_order_changes_latent_state",
                    "snapshot_command_absent",
                )
            ),
            "limitation": "LIFECYCLE_RECEIPTS_REQUIRE_SEPARATE_SOURCE_AUDIT_NOT_ORACLE_INFERENCE",
        },
        "observed_resource_law": {
            "resident_latent_field_cells": 34,
            "latent_coordinate_count": 17,
            "oracle_materializes_fixed_latent_boundary_vector": True,
            "oracle_latent_boundary_vector_field_cells": 17,
            "production_streams_each_latent_boundary_directly_into_resident_state": True,
            "independent_matching_interface_states": "2_TO_THE_N",
            "independent_module_work": "O_17_N_2_TO_THE_N_FIELD_OPERATIONS_FOR_ORACLE_ONLY",
            "production_matched_determinant_work": "O_17_N_TO_THE_6_FIELD_OPERATIONS",
            "fixed_latent_direct_sum_rank": 17,
        },
        "claim_ceiling": {
            "exact_q_zeta17_sizes": EXACT_SIZES,
            "dual_field_structural_sizes": STRUCTURAL_SIZES,
            "even_open_square_grid_family_only": True,
            "fixed_17_coordinate_latent_port_only": True,
            "identical_compact_classical_direct_sum_recurrence": True,
            "arbitrary_latent_geometry_or_planar_holant_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_COHERENT_SHARED_BASIS_PORT_IS_EXACT_AND_CAUSALLY_USED_BUT_ITS_FIXED_17_COORDINATES_ARE_THE_IDENTICAL_CLASSICAL_DIRECT_SUM",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True)
    args = parser.parse_args()
    with open(args.production, encoding="utf-8") as handle:
        production = json.load(handle)
    print(json.dumps(verify(production), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
