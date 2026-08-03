#!/usr/bin/env python3
"""Exact column-projective quotient for the M158 dual-register shear.

Every linear A-register operation is homogeneous and column-local up to a
public column permutation, while the B shear depends on A only through A^2.
Consequently each A matrix column has a discrete sign symmetry.  This package
executes on canonical sign-orbit representatives and carries the orientation
bits required to restore the actual borrowed carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import f103_c102_dual_register_quadratic_phase_shear_relation_no_go as shear
import f103_unresolved_c102_group_algebra_superposition_relation_no_go as base


INTERFACE = 5
DEPTHS = (1, 4, 16, 64)
FAMILIES = shear.FAMILIES
ORIENTATION_BITS = base.NODE_COUNT * INTERFACE
CLAIM = (
    "BOUNDED_EXACT_F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_QUOTIENT_IS_"
    "CLOSED_UNDER_DUAL_REGISTER_QUADRATIC_SHEAR_RELATION_PROGRAMS_"
    "THROUGH_DEPTH64_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_"
    "REUSE_BUT_THE_CANONICAL_REPRESENTATIVE_RETAINS_ALL45900_FIELD_"
    "CELLS_AND_THE45_BIT_ORIENTATION_LEDGER_EXACTLY_RETURNS_THE_"
    "DISCARDED_ORBIT_INFORMATION_WHILE_THE_IDENTICAL_CLASSICAL_"
    "RECURRENCE_REMAINS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    owner: int
    observation_left: int
    observation_right: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PROGRAM_V1",
            "interface": INTERFACE,
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": base.NODE_COUNT,
            "port_type": "F103_C102_DUAL_REGISTER_COLUMN_PROJECTIVE_C5_TO_C5",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
            "a_register": "COLUMNWISE_SIGN_ORBIT_REPRESENTATIVE",
            "orientation": "PACKABLE45_BIT_PRIVATE_RESTORATION_LEDGER",
            "b_register": "FULL_F103_C102_COEFFICIENT_STATE",
            "projection": "FINAL_REGISTER_B_BOUNDARY_EVALUATION_T_TO_5",
            "observation": [self.observation_left, self.observation_right],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= max(DEPTHS):
        fail("depth outside declared ceiling")
    if family not in FAMILIES:
        fail("family outside declared set")
    code = base.family_code(family)
    return Program(
        depth,
        family,
        (0xC1600000 + 131 * depth + code) & 0xFFFFFFFF,
        (13 * depth + 3 * code + INTERFACE) % INTERFACE,
        (19 * depth + 7 * code + 2 * INTERFACE) % INTERFACE,
    )


def bit_index(node: int, column: int) -> int:
    return node * INTERFACE + column


def get_signs(mask: int, node: int) -> np.ndarray:
    return np.asarray(
        [(mask >> bit_index(node, column)) & 1 for column in range(INTERFACE)],
        dtype=np.uint8,
    )


def set_signs(mask: int, node: int, signs: np.ndarray) -> int:
    for column, value in enumerate(signs):
        bit = 1 << bit_index(node, column)
        if int(value):
            mask |= bit
        else:
            mask &= ~bit
    return mask


def canonicalize_columns(
    matrix: np.ndarray, signs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    result = matrix.copy()
    updated = signs.copy()
    flips = 0
    for column in range(INTERFACE):
        vector = result[:, column, :]
        negative = np.asarray((-vector.astype(np.int64)) % base.FIELD, dtype=np.uint8)
        if negative.tobytes() < vector.tobytes():
            result[:, column, :] = negative
            updated[column] ^= 1
            flips += 1
    return result, updated, flips


def canonicalize_register_a(registers: np.ndarray) -> tuple[int, int]:
    mask = 0
    flips = 0
    for node in range(base.NODE_COUNT):
        signs = np.zeros(INTERFACE, dtype=np.uint8)
        canonical, signs, count = canonicalize_columns(registers[0, node], signs)
        np.copyto(registers[0, node], canonical)
        mask = set_signs(mask, node, signs)
        flips += count
    return mask, flips


def apply_orientation(registers: np.ndarray, mask: int) -> None:
    for node in range(base.NODE_COUNT):
        signs = get_signs(mask, node)
        for column in range(INTERFACE):
            if signs[column]:
                registers[0, node, :, column, :] = np.asarray(
                    (-registers[0, node, :, column, :].astype(np.int64))
                    % base.FIELD,
                    dtype=np.uint8,
                )


def oriented_seed(family: str) -> np.ndarray:
    registers = shear.seed_registers(INTERFACE, family)
    code = base.family_code(family)
    for node in range(base.NODE_COUNT):
        for column in range(INTERFACE):
            if (3 * node + 5 * column + code) % 4 in (1, 2):
                registers[0, node, :, column, :] = np.asarray(
                    (-registers[0, node, :, column, :].astype(np.int64))
                    % base.FIELD,
                    dtype=np.uint8,
                )
    return registers


def negate_a_column(registers: np.ndarray, node: int, column: int) -> None:
    registers[0, node, :, column, :] = np.asarray(
        (-registers[0, node, :, column, :].astype(np.int64)) % base.FIELD,
        dtype=np.uint8,
    )


def raw_forward(registers: np.ndarray, program: Program, *, enabled: bool = True) -> None:
    if not enabled:
        return
    for index in range(program.depth):
        hub = base.hub_index(index, program.family)
        for peer in base.peer_order(hub, program.family):
            amount = base.rotation_shift(INTERFACE, peer, index, program.family)
            current = [
                base.rotate_coefficients(registers[register, peer], amount, None)
                for register in range(shear.REGISTERS)
            ]
            left, right, coupling = base.composition_exponents(
                INTERFACE, hub, peer, index, program.family
            )
            current = [
                base.compose_coefficients(matrix, left, right, coupling, None)
                for matrix in current
            ]
            current[1] = shear.apply_shear(
                current[0],
                current[1],
                index,
                hub,
                peer,
                program.family,
                inverse=False,
                mutation=0,
                stats=None,
            )
            current = [
                base.intersect_coefficients(
                    matrix,
                    hub,
                    peer,
                    index,
                    program.family,
                    inverse=False,
                    mutation=0,
                    stats=None,
                )
                for matrix in current
            ]
            for register in range(shear.REGISTERS):
                np.copyto(registers[register, peer], current[register])


def boundary(registers: np.ndarray, program: Program) -> tuple[int, ...]:
    return tuple(
        base.evaluate_polynomial(
            registers[
                1,
                node,
                (program.observation_left + node) % INTERFACE,
                (program.observation_right + 2 * node) % INTERFACE,
            ]
        )
        for node in range(base.NODE_COUNT)
    )


@dataclass
class QuotientStats:
    canonicalizations: int = 0
    column_scans: int = 0
    representative_flips: int = 0
    consumers: int = 0

    def descriptor(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class Carrier:
    registers: np.ndarray
    family: str
    orientation_mask: int
    restoration_generation: int = 0
    sealed: bool = True

    @classmethod
    def seal(cls, registers: np.ndarray, family: str) -> "Carrier":
        if registers.shape != (
            shear.REGISTERS,
            base.NODE_COUNT,
            INTERFACE,
            INTERFACE,
            base.GROUP_ORDER,
        ):
            fail("wrong carrier shape")
        if registers.dtype != np.uint8:
            fail("wrong carrier dtype")
        mask, _flips = canonicalize_register_a(registers)
        return cls(registers, family, mask)

    def backing_id(self) -> int:
        return id(self.registers)

    def state_commitment(self) -> str:
        return digest_json(
            {
                "registers": hashlib.sha256(self.registers.tobytes()).hexdigest(),
                "family": self.family,
                "orientation_mask_commitment": hashlib.sha256(
                    self.orientation_mask.to_bytes(6, "little")
                ).hexdigest(),
                "sealed": self.sealed,
            }
        )

    def unseal(self) -> np.ndarray:
        if not self.sealed:
            fail("carrier already unsealed")
        apply_orientation(self.registers, self.orientation_mask)
        self.orientation_mask = 0
        self.sealed = False
        return self.registers


def recanonicalize_node(
    carrier: Carrier,
    node: int,
    matrix: np.ndarray,
    signs: np.ndarray,
    stats: QuotientStats | None,
) -> np.ndarray:
    canonical, updated, flips = canonicalize_columns(matrix, signs)
    carrier.orientation_mask = set_signs(carrier.orientation_mask, node, updated)
    if stats is not None:
        stats.canonicalizations += 1
        stats.column_scans += INTERFACE
        stats.representative_flips += flips
    return canonical


def quotient_forward(
    carrier: Carrier,
    program: Program,
    stats: QuotientStats | None = None,
) -> None:
    if not carrier.sealed or carrier.family != program.family:
        fail("carrier/program mismatch")
    registers = carrier.registers
    for index in range(program.depth):
        hub = base.hub_index(index, program.family)
        for peer in base.peer_order(hub, program.family):
            signs = get_signs(carrier.orientation_mask, peer)
            amount = base.rotation_shift(INTERFACE, peer, index, program.family)
            current_a = base.rotate_coefficients(registers[0, peer], amount, None)
            current_b = base.rotate_coefficients(registers[1, peer], amount, None)
            signs = np.roll(signs, amount)
            left, right, coupling = base.composition_exponents(
                INTERFACE, hub, peer, index, program.family
            )
            current_a = base.compose_coefficients(
                current_a, left, right, coupling, None
            )
            current_b = base.compose_coefficients(
                current_b, left, right, coupling, None
            )
            current_a = recanonicalize_node(
                carrier, peer, current_a, signs, stats
            )
            signs = get_signs(carrier.orientation_mask, peer)
            current_b = shear.apply_shear(
                current_a,
                current_b,
                index,
                hub,
                peer,
                program.family,
                inverse=False,
                mutation=0,
                stats=None,
            )
            current_a = base.intersect_coefficients(
                current_a,
                hub,
                peer,
                index,
                program.family,
                inverse=False,
                mutation=0,
                stats=None,
            )
            current_b = base.intersect_coefficients(
                current_b,
                hub,
                peer,
                index,
                program.family,
                inverse=False,
                mutation=0,
                stats=None,
            )
            current_a = recanonicalize_node(
                carrier, peer, current_a, signs, stats
            )
            np.copyto(registers[0, peer], current_a)
            np.copyto(registers[1, peer], current_b)
            if stats is not None:
                stats.consumers += 1


def quotient_inverse(
    carrier: Carrier,
    program: Program,
    stats: QuotientStats | None = None,
    *,
    order: str = "INTERSECT_SHEAR_COMPOSE",
) -> None:
    registers = carrier.registers
    for index in reversed(range(program.depth)):
        hub = base.hub_index(index, program.family)
        for peer in reversed(base.peer_order(hub, program.family)):
            signs = get_signs(carrier.orientation_mask, peer)
            left, right, coupling = base.composition_exponents(
                INTERFACE, hub, peer, index, program.family
            )
            current_a = base.intersect_coefficients(
                registers[0, peer],
                hub,
                peer,
                index,
                program.family,
                inverse=True,
                mutation=0,
                stats=None,
            )
            current_b = base.intersect_coefficients(
                registers[1, peer],
                hub,
                peer,
                index,
                program.family,
                inverse=True,
                mutation=0,
                stats=None,
            )
            current_a = recanonicalize_node(
                carrier, peer, current_a, signs, stats
            )
            signs = get_signs(carrier.orientation_mask, peer)
            if order == "INTERSECT_SHEAR_COMPOSE":
                current_b = shear.apply_shear(
                    current_a,
                    current_b,
                    index,
                    hub,
                    peer,
                    program.family,
                    inverse=True,
                    mutation=0,
                    stats=None,
                )
                current_a = base.inverse_compose_coefficients(
                    current_a, left, right, coupling, None
                )
                current_b = base.inverse_compose_coefficients(
                    current_b, left, right, coupling, None
                )
            elif order == "INTERSECT_COMPOSE_SHEAR":
                current_a = base.inverse_compose_coefficients(
                    current_a, left, right, coupling, None
                )
                current_b = base.inverse_compose_coefficients(
                    current_b, left, right, coupling, None
                )
                current_b = shear.apply_shear(
                    current_a,
                    current_b,
                    index,
                    hub,
                    peer,
                    program.family,
                    inverse=True,
                    mutation=0,
                    stats=None,
                )
            else:
                fail("unknown inverse order")
            current_a = recanonicalize_node(
                carrier, peer, current_a, signs, stats
            )
            signs = get_signs(carrier.orientation_mask, peer)
            amount = base.rotation_shift(INTERFACE, peer, index, program.family)
            current_a = base.rotate_coefficients(current_a, -amount, None)
            current_b = base.rotate_coefficients(current_b, -amount, None)
            signs = np.roll(signs, -amount)
            current_a = recanonicalize_node(
                carrier, peer, current_a, signs, stats
            )
            np.copyto(registers[0, peer], current_a)
            np.copyto(registers[1, peer], current_b)
            if stats is not None:
                stats.consumers += 1


def project_a(_carrier: Carrier) -> None:
    fail("A register is not a public boundary")


def transaction(carrier: Carrier | None, program: Program) -> dict[str, Any]:
    if carrier is None:
        fail("null carrier")
    before = carrier.state_commitment()
    backing = carrier.backing_id()
    stats = QuotientStats()
    quotient_forward(carrier, program, stats)
    final_boundary = boundary(carrier.registers, program)
    final_commitment = hashlib.sha256(bytes(final_boundary)).hexdigest()
    quotient_inverse(carrier, program, stats)
    if carrier.state_commitment() != before:
        fail("quotient carrier failed exact restoration")
    if carrier.backing_id() != backing:
        fail("quotient carrier backing changed")
    carrier.restoration_generation += 1
    return {
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": final_commitment,
        "restoration_generation": carrier.restoration_generation,
        "same_backing_restored": True,
        "stats": stats.descriptor(),
    }


def one_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    raw = oriented_seed(family)
    raw_input_commitment = hashlib.sha256(raw.tobytes()).hexdigest()
    backing = id(raw)
    baseline = raw.copy()
    raw_forward(baseline, program)
    baseline_boundary = boundary(baseline, program)
    carrier = Carrier.seal(raw, family)
    mask_bits = carrier.orientation_mask.bit_count()
    receipt = transaction(carrier, program)
    quotient_boundary_commitment = receipt["boundary_commitment"]
    baseline_boundary_commitment = hashlib.sha256(bytes(baseline_boundary)).hexdigest()
    restored = carrier.unseal()
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "boundary_commitment": quotient_boundary_commitment,
        "matches_raw_identical_coefficient_recurrence": (
            quotient_boundary_commitment == baseline_boundary_commitment
        ),
        "orientation_bits_set_at_seal": mask_bits,
        "exact_raw_payload_restored_after_unseal": (
            hashlib.sha256(restored.tobytes()).hexdigest() == raw_input_commitment
        ),
        "same_backing_restored": id(restored) == backing,
        "restoration_generation": receipt["restoration_generation"],
        "quotient_work": receipt["stats"],
    }


def controls() -> dict[str, bool]:
    family = "PRIMARY"
    program = compile_program(4, family)
    plain = shear.seed_registers(INTERFACE, family)
    oriented = oriented_seed(family)
    plain_carrier = Carrier.seal(plain.copy(), family)
    oriented_carrier = Carrier.seal(oriented.copy(), family)
    same_representative = np.array_equal(
        plain_carrier.registers, oriented_carrier.registers
    )
    distinct_ledger = (
        plain_carrier.orientation_mask != oriented_carrier.orientation_mask
    )

    raw_plain = plain.copy()
    raw_oriented = oriented.copy()
    raw_forward(raw_plain, program)
    raw_forward(raw_oriented, program)
    orbit_boundaries_match = boundary(raw_plain, program) == boundary(
        raw_oriented, program
    )

    generator_checks = []
    plain_representative = plain_carrier.registers
    plain_boundary = boundary(raw_plain, program)
    for node in range(base.NODE_COUNT):
        for column in range(INTERFACE):
            variant = shear.seed_registers(INTERFACE, family)
            negate_a_column(variant, node, column)
            variant_commitment = hashlib.sha256(variant.tobytes()).hexdigest()
            variant_carrier = Carrier.seal(variant, family)
            same_generator_representative = np.array_equal(
                variant_carrier.registers, plain_representative
            )
            variant_raw = shear.seed_registers(INTERFACE, family)
            negate_a_column(variant_raw, node, column)
            raw_forward(variant_raw, program)
            same_generator_boundary = boundary(variant_raw, program) == plain_boundary
            generator_restored = (
                hashlib.sha256(variant_carrier.unseal().tobytes()).hexdigest()
                == variant_commitment
            )
            generator_checks.append(
                same_generator_representative
                and same_generator_boundary
                and generator_restored
            )

    missing = Carrier.seal(oriented.copy(), family)
    before_missing = missing.state_commitment()
    quotient_forward(missing, program)
    missing_inverse_fails = missing.state_commitment() != before_missing

    reordered = Carrier.seal(oriented.copy(), family)
    before_reordered = reordered.state_commitment()
    quotient_forward(reordered, program)
    quotient_inverse(reordered, program, order="INTERSECT_COMPOSE_SHEAR")
    reordered_inverse_fails = reordered.state_commitment() != before_reordered

    dropped = Carrier.seal(oriented.copy(), family)
    raw_commitment = hashlib.sha256(oriented.tobytes()).hexdigest()
    transaction(dropped, program)
    dropped.orientation_mask = 0
    dropped_restoration_fails = (
        hashlib.sha256(dropped.unseal().tobytes()).hexdigest() != raw_commitment
    )

    wrong = Carrier.seal(oriented.copy(), family)
    transaction(wrong, program)
    wrong.orientation_mask ^= 1
    wrong_ledger_fails = (
        hashlib.sha256(wrong.unseal().tobytes()).hexdigest() != raw_commitment
    )

    disabled = oriented.copy()
    raw_forward(disabled, program, enabled=False)
    enabled = oriented.copy()
    raw_forward(enabled, program)
    disabled_path_differs = boundary(disabled, program) != boundary(enabled, program)

    projection_rejected = False
    try:
        project_a(Carrier.seal(oriented.copy(), family))
    except RuntimeError:
        projection_rejected = True

    null_rejected = False
    try:
        transaction(None, program)
    except RuntimeError:
        null_rejected = True

    return {
        "sign_related_inputs_have_same_canonical_representative": same_representative,
        "sign_related_inputs_have_distinct_orientation_ledgers": distinct_ledger,
        "sign_related_inputs_have_identical_public_b_boundaries": orbit_boundaries_match,
        "all45_independent_column_sign_generators_collapse_preserve_boundary_and_restore": all(
            generator_checks
        )
        and len(generator_checks) == ORIENTATION_BITS,
        "missing_inverse_fails_restoration": missing_inverse_fails,
        "reordered_inverse_fails_restoration": reordered_inverse_fails,
        "dropped_orientation_ledger_fails_raw_restoration": dropped_restoration_fails,
        "wrong_orientation_ledger_fails_raw_restoration": wrong_ledger_fails,
        "carrier_disabled_path_changes_boundary": disabled_path_differs,
        "a_register_projection_rejected": projection_rejected,
        "null_carrier_rejected": null_rejected,
    }


def reuse_control() -> dict[str, Any]:
    family = "ALTERNATE"
    raw = oriented_seed(family)
    raw_commitment = hashlib.sha256(raw.tobytes()).hexdigest()
    backing = id(raw)
    carrier = Carrier.seal(raw, family)
    first = transaction(carrier, compile_program(1, family))
    second = transaction(carrier, compile_program(16, family))
    fresh = Carrier.seal(oriented_seed(family), family)
    fresh_second = transaction(fresh, compile_program(16, family))
    restored = carrier.unseal()
    return {
        "same_backing_reused": id(restored) == backing,
        "restoration_generation": second["restoration_generation"],
        "unrelated_second_boundary_matches_fresh": second["boundary_commitment"]
        == fresh_second["boundary_commitment"],
        "exact_raw_payload_restored_after_reuse": hashlib.sha256(
            restored.tobytes()
        ).hexdigest()
        == raw_commitment,
        "first_boundary_commitment": first["boundary_commitment"],
        "snapshot_used": False,
    }


def run() -> dict[str, Any]:
    cases = [one_case(depth, family) for family in FAMILIES for depth in DEPTHS]
    if not all(
        case["matches_raw_identical_coefficient_recurrence"]
        and case["exact_raw_payload_restored_after_unseal"]
        and case["same_backing_restored"]
        for case in cases
    ):
        fail("projective quotient case failed")
    control_results = controls()
    if not all(control_results.values()):
        fail(
            "projective quotient controls failed: "
            + repr([key for key, value in control_results.items() if not value])
        )
    reuse = reuse_control()
    if not all(
        (
            reuse["same_backing_reused"],
            reuse["unrelated_second_boundary_matches_fresh"],
            reuse["exact_raw_payload_restored_after_reuse"],
            not reuse["snapshot_used"],
        )
    ):
        fail("projective quotient reuse failed")
    carrier_cells = int(shear.seed_registers(INTERFACE, "PRIMARY").size)
    fixed_width_bits = carrier_cells * math.ceil(math.log2(base.FIELD))
    return {
        "schema": "CAT_CAS_F103_C102_COLUMN_PROJECTIVE_SIGN_ORBIT_PHASE_QUOTIENT_V1",
        "claim": CLAIM,
        "platform": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": "F103",
            "phase_group": "C102",
            "interface": INTERFACE,
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "quotient": "INDEPENDENT_COLUMNWISE_A_REGISTER_SIGN_ORBITS",
            "orientation_bits": ORIENTATION_BITS,
            "final_boundary_only": True,
            "raw_boundary_values_serialized": False,
        },
        "cases": cases,
        "controls": control_results,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "canonical_representative_field_cells": carrier_cells,
            "matched_raw_coefficient_recurrence_field_cells": carrier_cells,
            "material_field_cell_reduction": 0,
            "raw_fixed_width_payload_bits": fixed_width_bits,
            "projective_orbit_cardinality_reduction_bits_when_all_columns_nonzero": ORIENTATION_BITS,
            "exact_restoration_orientation_ledger_bits": ORIENTATION_BITS,
            "net_lossless_information_reduction_bits_after_restoration_ledger": 0,
            "minimum_packed_orientation_ledger_bytes": (ORIENTATION_BITS + 7) // 8,
            "tested_independent_sign_generators": ORIENTATION_BITS,
            "free_nonzero_seed_sign_orbit_order": 2**ORIENTATION_BITS,
            "information_theoretic_minimum_restoration_ledger_bits_for_arbitrary_orbit_member": ORIENTATION_BITS,
            "actual_numpy_representative_bytes": carrier_cells,
            "python_integer_allocator_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "matched_compact_classical": {
            "strongest_path": "IDENTICAL_DUAL_REGISTER_F103_C102_COEFFICIENT_RECURRENCE",
            "same_field_cells": True,
            "same_public_boundaries": True,
        },
        "claim_ceiling": "C5_TWO_PUBLIC_ROTATING_HUB_FAMILIES_DEPTHS1_4_16_64_COLUMNWISE_GLOBAL_SIGN_ORBITS_OF_EACH_A_MATRIX_COLUMN_WITH45_BIT_PRIVATE_RESTORATION_LEDGER",
        "not_established": [
            "FIELD_CELL_OR_FIXED_WIDTH_PAYLOAD_COMPACTION",
            "QUOTIENT_BY_SCALARS_OTHER_THAN_PLUS_OR_MINUS_ONE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "THE_ONLY_EXECUTED_DISCRETE_PROJECTIVE_QUOTIENT_RETAINS_THE_FULL_FIELD_COORDINATE_BACKING_AND_EXACT_RESTORATION_RETURNS_ALL_REMOVED_ORIENTATION_INFORMATION_SO_THE_NEXT_PHASE_MACHINE_MUST_CHANGE_THE_ALGEBRAIC_STATE_LAW_RATHER_THAN_QUOTIENT_THIS_SHEAR_CARRIER",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    if arguments.summary:
        print(
            json.dumps(
                {
                    "claim": result["claim"],
                    "cases": result["experiment"]["case_count"],
                    "resource_accounting": result["resource_accounting"],
                    "next_obstruction": result["next_obstruction"],
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
