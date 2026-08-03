#!/usr/bin/env python3
"""Independent oracle for the matrix-free anisotropic F17 radial closure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import f17_coherent_veronese_phase_chart_closure as exact
import f17_nonlinear_canonical_mps_separator_chart as backend


P = 17
D = 3
SHELL_COUNTS = (1, *([18] * 16))
EXACT_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
FINITE_FIELDS = ((103, 72), (137, 16))


def field_integer(alg: backend.Algebra, value: int) -> Any:
    return exact.field_integer(alg, value)


def norm(x: int, y: int) -> int:
    return (x * x - D * y * y) % P


def pairing(u: int, v: int, x: int, y: int) -> int:
    return (u * x - D * v * y) % P


def coordinates() -> Iterable[tuple[int, int]]:
    for x in range(P):
        for y in range(P):
            yield x, y


def representatives() -> tuple[tuple[int, int], ...]:
    found: list[tuple[int, int] | None] = [None for _ in range(P)]
    counts = [0 for _ in range(P)]
    for point in coordinates():
        shell = norm(*point)
        counts[shell] += 1
        if found[shell] is None:
            found[shell] = point
    assert tuple(counts) == SHELL_COUNTS
    assert all(point is not None for point in found)
    return tuple(point for point in found if point is not None)


@dataclass(frozen=True)
class Gate:
    quadratic: int
    linear: int
    constant: int

    def exponent(self, shell: int) -> int:
        return (
            self.quadratic * shell * shell
            + self.linear * shell
            + self.constant
        ) % P

    def as_json(self) -> dict[str, int | str]:
        return {
            "kind": "QUARTIC_ANISOTROPIC_NORM_PHASE_THEN_NORMALIZED_PHASE_FOURIER",
            "quadratic_multiplier_mod17": self.quadratic,
            "linear_multiplier_mod17": self.linear,
            "constant_mod17": self.constant,
        }


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    gates: tuple[Gate, ...]
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "coordinate_field": "F17_SQUARED",
            "typed_relation": "ANISOTROPIC_NORM_Q_EQUALS_X2_MINUS_3Y2",
            "unresolved_port": "Q:F17_ANISOTROPIC_NORM_SHELL",
            "depth": self.depth,
            "family": self.family,
            "gates": [gate.as_json() for gate in self.gates],
            "final_observation": {
                "kind": "RADIAL_QUARTIC_SCALAR",
                "quadratic_multiplier_mod17": self.observation_quadratic,
                "linear_multiplier_mod17": self.observation_linear,
            },
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.descriptor(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def gate_parameters(index: int, family: str) -> tuple[int, int, int]:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    ternary_weight = 0
    remaining = index
    while remaining:
        ternary_weight += remaining % 3
        remaining //= 3
    if family == "PRIMARY":
        values = (
            3 * index + 5 * bit_weight + 1,
            7 * index + 2 * bit_weight + 2,
            11 * index + bit_weight + 4,
        )
    elif family == "REUSE":
        values = (
            5 * index + 2 * ternary_weight + 3,
            4 * index + 3 * ternary_weight + 6,
            9 * index + ternary_weight + 8,
        )
    elif family == "ALTERNATE":
        values = (
            7 * index + 3 * gray_weight + 2,
            8 * index + 2 * gray_weight + 5,
            6 * index + gray_weight + 1,
        )
    else:
        raise ValueError(family)
    quadratic = values[0] % P or 1
    return quadratic, values[1] % P, values[2] % P


def compile_program(depth: int, family: str) -> Program:
    return Program(
        depth,
        family,
        tuple(Gate(*gate_parameters(index, family)) for index in range(depth)),
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )


@dataclass(frozen=True)
class OracleGeometry:
    alg: backend.Algebra
    inverse17: Any

    @classmethod
    def compile(cls, alg: backend.Algebra) -> "OracleGeometry":
        assert D not in {value * value % P for value in range(P)}
        assert representatives()
        return cls(alg, alg.inverse(field_integer(alg, P)))

    def entry(self, target: int, source: int) -> Any:
        total = self.alg.zero
        for parameter in range(1, P):
            inverse_four_parameter = pow((4 * parameter) % P, -1, P)
            exponent = (
                -source * parameter - target * inverse_four_parameter
            ) % P
            total = self.alg.add(total, self.alg.power(exponent))
        delta = field_integer(self.alg, P if target == 0 else 0)
        return self.alg.mul(self.inverse17, self.alg.sub(delta, total))

    def factored_fourier(self, state: list[Any]) -> list[Any]:
        spectrum = []
        for parameter in range(1, P):
            total = self.alg.zero
            for source, value in enumerate(state):
                total = self.alg.add(
                    total,
                    self.alg.mul(self.alg.power(-source * parameter), value),
                )
            spectrum.append(total)
        state_sum = self.alg.zero
        for value in state:
            state_sum = self.alg.add(state_sum, value)
        output = []
        for target in range(P):
            total = self.alg.zero
            for offset, parameter in enumerate(range(1, P)):
                inverse_four_parameter = pow((4 * parameter) % P, -1, P)
                total = self.alg.add(
                    total,
                    self.alg.mul(
                        self.alg.power(-target * inverse_four_parameter),
                        spectrum[offset],
                    ),
                )
            scaled = self.alg.mul(self.inverse17, total)
            output.append(
                self.alg.sub(state_sum if target == 0 else self.alg.zero, scaled)
            )
        return output


def commitment(alg: backend.Algebra, state: list[Any]) -> str:
    hasher = hashlib.sha256()
    for value in state:
        record = json.dumps(alg.serialize(value), separators=(",", ":")).encode()
        hasher.update(len(record).to_bytes(8, "big"))
        hasher.update(record)
    return hasher.hexdigest()


def project(alg: backend.Algebra, state: list[Any], program: Program) -> Any:
    total = alg.zero
    for shell, value in enumerate(state):
        phase = alg.power(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        total = alg.add(
            total,
            alg.mul(
                field_integer(alg, SHELL_COUNTS[shell]),
                alg.mul(phase, value),
            ),
        )
    return total


def execute(geometry: OracleGeometry, program: Program) -> dict[str, Any]:
    alg = geometry.alg
    state = [alg.one for _ in range(P)]
    maximum_payload = sum(alg.payload_bits(value) for value in state)
    for gate in program.gates:
        state = [
            alg.mul(value, alg.power(gate.exponent(shell)))
            for shell, value in enumerate(state)
        ]
        state = geometry.factored_fourier(state)
        maximum_payload = max(
            maximum_payload,
            sum(alg.payload_bits(value) for value in state),
        )
    final_commitment = commitment(alg, state)
    boundary = project(alg, state, program)
    for gate in reversed(program.gates):
        state = geometry.factored_fourier(state)
        maximum_payload = max(
            maximum_payload,
            sum(alg.payload_bits(value) for value in state),
        )
        state = [
            alg.mul(value, alg.power(-gate.exponent(shell)))
            for shell, value in enumerate(state)
        ]
        maximum_payload = max(
            maximum_payload,
            sum(alg.payload_bits(value) for value in state),
        )
    restored = state == [alg.one for _ in range(P)]
    return {
        "program_fingerprint": program.fingerprint(),
        "boundary": alg.serialize(boundary),
        "commitment": final_commitment,
        "maximum_resident_payload_bits": maximum_payload,
        "restored_seed_exactly": restored,
    }


def direct_coordinate_entry(
    geometry: OracleGeometry, target: int, source: int
) -> Any:
    u, v = representatives()[target]
    total = geometry.alg.zero
    for x, y in coordinates():
        if norm(x, y) == source:
            total = geometry.alg.add(
                total, geometry.alg.power(pairing(u, v, x, y))
            )
    return geometry.alg.mul(geometry.inverse17, total)


def dense_control(
    geometry: OracleGeometry, program: Program
) -> dict[str, Any]:
    alg = geometry.alg
    state = [[alg.one for _ in range(P)] for _ in range(P)]
    compact = [alg.one for _ in range(P)]
    for gate in program.gates:
        for x, y in coordinates():
            state[x][y] = alg.mul(
                state[x][y], alg.power(gate.exponent(norm(x, y)))
            )
        first = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for y in range(P):
                for x in range(P):
                    first[u][y] = alg.add(
                        first[u][y],
                        alg.mul(alg.power(u * x), state[x][y]),
                    )
        transformed = [[alg.zero for _ in range(P)] for _ in range(P)]
        for u in range(P):
            for v in range(P):
                for y in range(P):
                    transformed[u][v] = alg.add(
                        transformed[u][v],
                        alg.mul(alg.power(-D * v * y), first[u][y]),
                    )
                transformed[u][v] = alg.mul(
                    geometry.inverse17, transformed[u][v]
                )
        state = transformed
        compact = [
            alg.mul(value, alg.power(gate.exponent(shell)))
            for shell, value in enumerate(compact)
        ]
        compact = geometry.factored_fourier(compact)
    reps = representatives()
    dense_shells = [state[x][y] for x, y in reps]
    return {
        "field": alg.kind,
        "family": program.family,
        "depth": program.depth,
        "all_17_shell_values_equal": dense_shells == compact,
        "boundary_equal": (
            alg.serialize(project(alg, dense_shells, program))
            == alg.serialize(project(alg, compact, program))
        ),
    }


def source_shape(source: Path) -> dict[str, bool]:
    tree = ast.parse(source.read_text(encoding="utf-8"), str(source))

    def function(name: str) -> ast.FunctionDef:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(name)

    accepted_text = "\n".join(
        ast.unparse(function(name))
        for name in (
            "apply_fourier",
            "begin_forward",
            "forward",
            "project",
            "inverse",
            "execute_transaction",
        )
    )
    fourier_text = ast.unparse(function("apply_fourier"))
    return {
        "accepted_path_has_no_normalized_fourier_matrix": (
            "normalized_fourier" not in accepted_text
        ),
        "accepted_path_has_no_coordinate_enumeration": (
            "coordinates()" not in accepted_text
        ),
        "accepted_fourier_does_not_call_per_entry_generator": (
            ".entry(" not in fourier_text
        ),
        "accepted_fourier_has_16_cell_spectrum": (
            "range(P - 1)" in fourier_text and "range(1, P)" in fourier_text
        ),
    }


def run(production_path: Path, source_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    exact_geometry = OracleGeometry.compile(backend.Algebra("Q_ZETA17"))
    exact_reexecution = []
    production_exact = {
        item["depth"]: item for item in production["exact_transactions"]
    }
    for depth in EXACT_DEPTHS:
        program = compile_program(depth, "PRIMARY")
        observed = execute(exact_geometry, program)
        expected = production_exact[depth]
        exact_reexecution.append(
            {
                "depth": depth,
                "program_fingerprint_equal": (
                    observed["program_fingerprint"]
                    == expected["program_fingerprint"]
                ),
                "boundary_equal": observed["boundary"] == expected["final_boundary"],
                "commitment_equal": (
                    observed["commitment"] == expected["final_state_commitment"]
                ),
                "resident_payload_equal": (
                    observed["maximum_resident_payload_bits"]
                    == expected["maximum_resident_payload_bits"]
                ),
                "restored_seed_exactly": observed["restored_seed_exactly"],
            }
        )

    structural_reexecution = []
    production_structural = {
        (item["algebra_kind"], item["family"], item["depth"]): item
        for item in production["structural_transactions"]
    }
    geometries = {}
    for modulus, root in FINITE_FIELDS:
        kind = f"F{modulus}"
        geometry = OracleGeometry.compile(
            backend.Algebra(kind, modulus=modulus, root=root)
        )
        geometries[kind] = geometry
        passed = True
        for family in FAMILIES:
            for depth in STRUCTURAL_DEPTHS:
                program = compile_program(depth, family)
                observed = execute(geometry, program)
                expected = production_structural[(kind, family, depth)]
                passed = passed and (
                    observed["program_fingerprint"]
                    == expected["program_fingerprint"]
                    and observed["boundary"] == expected["final_boundary"]
                    and observed["commitment"] == expected["final_state_commitment"]
                    and observed["restored_seed_exactly"]
                )
        structural_reexecution.append(
            {"field": kind, "transaction_count": 24, "all_equal": passed}
        )

    direct_entry_checks = []
    for kind, geometry in geometries.items():
        all_equal = all(
            geometry.entry(target, source)
            == direct_coordinate_entry(geometry, target, source)
            for target in range(P)
            for source in range(P)
        )
        direct_entry_checks.append(
            {"field": kind, "entry_checks": 289, "all_equal": all_equal}
        )
    qzeta_pairs = ((0, 0), (0, 1), (1, 0), (3, 5), (16, 16))
    direct_entry_checks.append(
        {
            "field": "Q_ZETA17",
            "entry_checks": len(qzeta_pairs),
            "all_equal": all(
                exact_geometry.entry(target, source)
                == direct_coordinate_entry(exact_geometry, target, source)
                for target, source in qzeta_pairs
            ),
        }
    )

    dense_controls = [
        dense_control(geometries["F103"], compile_program(2, "ALTERNATE")),
        dense_control(geometries["F137"], compile_program(4, "PRIMARY")),
    ]
    shape = source_shape(source_path)

    mutations = {
        "missing_parameter_term_changes_an_entry": (
            sum(
                1
                for target in range(P)
                for source in range(P)
                if geometries["F137"].entry(target, source)
                != truncated_entry(geometries["F137"], target, source)
            )
            > 0
        ),
        "wrong_reciprocal_sign_changes_an_entry": (
            sum(
                1
                for target in range(P)
                for source in range(P)
                if geometries["F137"].entry(target, source)
                != wrong_sign_entry(geometries["F137"], target, source)
            )
            > 0
        ),
    }

    passed = (
        all(all(item.values()) for item in exact_reexecution)
        and all(item["all_equal"] for item in structural_reexecution)
        and all(item["all_equal"] for item in direct_entry_checks)
        and all(item["all_17_shell_values_equal"] and item["boundary_equal"] for item in dense_controls)
        and all(shape.values())
        and all(mutations.values())
    )
    return {
        "schema": "CAT_CAS_F17_MATRIX_FREE_ANISOTROPIC_RADIAL_PHASE_FOURIER_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "passed": passed,
        "production_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "independence": {
            "imports_production_module": False,
            "reconstructs_public_geometry": True,
            "reconstructs_public_program_schedule": True,
            "reconstructs_factorized_forward_and_inverse": True,
            "reconstructs_final_state_commitment": True,
            "uses_direct_coordinate_character_sums": True,
            "uses_selected_dense_289_coordinate_dfts": True,
        },
        "exact_reexecution": {
            "transaction_count": len(exact_reexecution),
            "depths": list(EXACT_DEPTHS),
            "all_equal": all(all(item.values()) for item in exact_reexecution),
            "resident_payload_tuple": [
                production_exact[depth]["maximum_resident_payload_bits"]
                for depth in EXACT_DEPTHS
            ],
            "transactions": exact_reexecution,
        },
        "structural_reexecution": structural_reexecution,
        "direct_coordinate_entry_checks": direct_entry_checks,
        "dense_coordinate_controls": dense_controls,
        "source_shape": shape,
        "mutations": mutations,
        "resource_law": {
            "resident_exact_field_cells": 17,
            "retained_public_kernel_exact_field_cells": 0,
            "retained_public_generator_exact_field_cells": 1,
            "temporary_spectrum_exact_field_cells": 16,
            "maximum_update_scratch_exact_field_cells": 38,
            "maximum_live_resident_plus_update_scratch_exact_field_cells": 55,
            "maximum_live_including_retained_generator_exact_field_cells": 56,
            "character_products_per_factored_fourier": 544,
            "accepted_assignment_truth_table_coordinate_or_kernel_cells": 0,
        },
        "claim_ceiling": (
            "STRICT_ANISOTROPIC_F17_SQUARED_RADIAL_FUNCTIONS_QUARTIC_NORM_"
            "PHASES_FACTORED16_TERM_KLOOSTERMAN_TYPE_PHASE_FOURIER_Q_ZETA17_"
            "F103_F137_DIRECT_PROCESS_SOFTWARE"
        ),
        "not_established": [
            "CATVM_CUSTODY",
            "GENERAL_NONLINEAR_RELATION_QUOTIENT",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_COMPUTATION",
        ],
    }


def truncated_entry(
    geometry: OracleGeometry, target: int, source: int
) -> Any:
    total = geometry.alg.zero
    for parameter in range(1, P - 1):
        inverse_four_parameter = pow((4 * parameter) % P, -1, P)
        total = geometry.alg.add(
            total,
            geometry.alg.power(
                -source * parameter - target * inverse_four_parameter
            ),
        )
    delta = field_integer(geometry.alg, P if target == 0 else 0)
    return geometry.alg.mul(geometry.inverse17, geometry.alg.sub(delta, total))


def wrong_sign_entry(
    geometry: OracleGeometry, target: int, source: int
) -> Any:
    total = geometry.alg.zero
    for parameter in range(1, P):
        inverse_four_parameter = pow((4 * parameter) % P, -1, P)
        total = geometry.alg.add(
            total,
            geometry.alg.power(
                -source * parameter + target * inverse_four_parameter
            ),
        )
    delta = field_integer(geometry.alg, P if target == 0 else 0)
    return geometry.alg.mul(geometry.inverse17, geometry.alg.sub(delta, total))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.production, args.production_source)
    if not result["passed"]:
        raise SystemExit("independent matrix-free anisotropic oracle failed")
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
