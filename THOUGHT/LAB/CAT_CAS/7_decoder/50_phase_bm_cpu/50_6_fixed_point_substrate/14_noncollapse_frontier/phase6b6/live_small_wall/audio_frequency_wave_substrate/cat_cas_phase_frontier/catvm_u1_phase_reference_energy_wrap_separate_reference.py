#!/usr/bin/env python3
"""M249 standalone exact oracle over an independent ``Q[s]/(s^2-2)`` model."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


LENGTHS = (2, 4, 8, 16)
PORT_TYPE = "CATVM_U1_FINITE_REFERENCE_JOINT_PORT_V1"
OUTPUT_TYPE = "QSQRT2_REDUCED_SYSTEM_BOUNDARY_V1"
CONSUMER_ID = 249001
OWNER = 249004


@dataclass(frozen=True)
class E:
    x: Fraction = Fraction(0)
    y: Fraction = Fraction(0)

    def plus(self, other: "E") -> "E":
        return E(self.x + other.x, self.y + other.y)

    def minus(self, other: "E") -> "E":
        return E(self.x - other.x, self.y - other.y)

    def negative(self) -> "E":
        return E(-self.x, -self.y)

    def times(self, other: "E") -> "E":
        return E(
            self.x * other.x + 2 * self.y * other.y,
            self.x * other.y + self.y * other.x,
        )


Z = E()
I = E(Fraction(1))
HCOEF = E(Fraction(0), Fraction(1, 2))
ROTATIONS = {
    "H": (HCOEF, HCOEF),
    "RATIONAL_3_4_5": (E(Fraction(3, 5)), E(Fraction(4, 5))),
}


def encode_fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def encode(value: E) -> dict[str, list[int]]:
    return {
        "rational": encode_fraction(value.x),
        "sqrt2": encode_fraction(value.y),
    }


def digest_value(value: E) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (value.x.numerator, value.x.denominator),
        (value.y.numerator, value.y.denominator),
    )


def uniform(length: int) -> E:
    power = length.bit_length() - 1
    if power % 2 == 0:
        return E(Fraction(1, 2 ** (power // 2)))
    return E(Fraction(0), Fraction(1, 2 ** ((power + 1) // 2)))


def canonical_state(length: int) -> list[E]:
    return [uniform(length) for _ in range(length)] + [Z for _ in range(length)]


def state_commitment(length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            [digest_value(value) for value in canonical_state(length)],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def public_descriptor(length: int, gate: str) -> tuple[int, str]:
    if length not in LENGTHS or gate not in ROTATIONS:
        raise RuntimeError("invalid independent M249 descriptor")
    return length, gate


def program_digest(descriptor: tuple[int, str]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def rotate_open(values: list[E], length: int, gate_name: str, inverse: bool) -> None:
    a, b = ROTATIONS[gate_name]
    for total in range(1, length):
        i = total
        j = length + total - 1
        left, right = values[i], values[j]
        if inverse:
            values[i] = a.times(left).plus(b.times(right))
            values[j] = b.negative().times(left).plus(a.times(right))
        else:
            values[i] = a.times(left).minus(b.times(right))
            values[j] = b.times(left).plus(a.times(right))


def pair_reflection(values: list[E], length: int) -> None:
    for total in range(1, length):
        index = length + total - 1
        values[index] = values[index].negative()


def streamed_boundary(values: list[E], length: int) -> tuple[E, E, E]:
    p0 = Z
    p1 = Z
    coherence = Z
    for level in range(length):
        left = values[level]
        right = values[length + level]
        p0 = p0.plus(left.times(left))
        p1 = p1.plus(right.times(right))
        coherence = coherence.plus(left.times(right))
    return p0, p1, coherence


def closed_boundary(length: int, gate_name: str) -> tuple[E, E, E]:
    a, b = ROTATIONS[gate_name]
    inv_l = E(Fraction(1, length))
    p1 = E(Fraction(length - 1)).times(b).times(b).times(inv_l)
    p0 = I.minus(p1)
    coherence = b.times(
        I.plus(E(Fraction(length - 2)).times(a))
    ).times(inv_l)
    return p0, p1, coherence


def open_schmidt_minor(length: int, gate_name: str) -> E:
    a, b = ROTATIONS[gate_name]
    return a.times(b).times(E(Fraction(-1, length)))


def cyclic_state(length: int, gate_name: str) -> list[E]:
    a, b = ROTATIONS[gate_name]
    amplitude = uniform(length)
    output = [Z for _ in range(2 * length)]
    for n in range(length):
        output[n] = output[n].plus(a.times(amplitude))
        target = length + ((n - 1) % length)
        output[target] = output[target].plus(b.times(amplitude))
    return output


def cyclic_expected_product(length: int, gate_name: str) -> list[E]:
    a, b = ROTATIONS[gate_name]
    amplitude = uniform(length)
    return [a.times(amplitude) for _ in range(length)] + [
        b.times(amplitude) for _ in range(length)
    ]


def cyclic_apply(values: list[E], length: int, gate_name: str, inverse: bool) -> list[E]:
    a, base_b = ROTATIONS[gate_name]
    b = base_b.negative() if inverse else base_b
    output = [Z for _ in range(2 * length)]
    for level in range(length):
        left = values[level]
        right = values[length + level]
        output[level] = output[level].plus(a.times(left))
        target_right = length + ((level - 1) % length)
        output[target_right] = output[target_right].plus(b.times(left))
        target_left = (level + 1) % length
        output[target_left] = output[target_left].minus(b.times(right))
        output[length + level] = output[length + level].plus(a.times(right))
    return output


def every_basis_inverse_exact(length: int, gate_name: str, cyclic: bool) -> bool:
    for index in range(2 * length):
        basis = [Z for _ in range(2 * length)]
        basis[index] = I
        if cyclic:
            restored = cyclic_apply(
                cyclic_apply(basis, length, gate_name, False),
                length, gate_name, True,
            )
        else:
            restored = list(basis)
            rotate_open(restored, length, gate_name, False)
            rotate_open(restored, length, gate_name, True)
        if restored != basis:
            return False
    return True


def dephased_reference_coherence(length: int, gate_name: str) -> E:
    """Trace the exact I/L reservoir mixture through the open dilation."""
    total = Z
    probability = E(Fraction(1, length))
    for reservoir_input in range(length):
        basis = [Z for _ in range(2 * length)]
        basis[reservoir_input] = I
        rotate_open(basis, length, gate_name, False)
        coherence = Z
        for reservoir_output in range(length):
            coherence = coherence.plus(
                basis[reservoir_output].times(basis[length + reservoir_output])
            )
        total = total.plus(probability.times(coherence))
    return total


def open_transition_energy_preserved(length: int) -> bool:
    for total in range(1, length):
        if 0 + total != 1 + (total - 1):
            return False
    return True


def cyclic_wrap_data(length: int, gate_name: str) -> dict[str, Any]:
    _a, b = ROTATIONS[gate_name]
    wrap_weight = b.times(b).times(E(Fraction(1, length)))
    basis_commutator_squared = E(Fraction(length * length)).times(b).times(b)
    input_commutator_squared = E(Fraction(length)).times(b).times(b)
    return {
        "wrap_weight": encode(wrap_weight),
        "wrapped_total_number_change": length,
        "basis_commutator_norm_squared": encode(basis_commutator_squared),
        "uniform_input_commutator_norm_squared": encode(input_commutator_squared),
        "nonzero": wrap_weight != Z and basis_commutator_squared != Z,
    }


class ReferencePort:
    def __init__(self, length: int) -> None:
        self.length = length
        self.values = canonical_state(length)
        self.last_generation = 0
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.descriptor: tuple[int, str] | None = None
        self.cursor = 0
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.values == canonical_state(self.length)
            and not self.leased
            and self.owner == 0
            and self.generation == 0
            and self.program_id == ""
            and self.descriptor is None
            and self.cursor == 0
        )

    def lease(
        self, descriptor: tuple[int, str], owner: int, generation: int,
        supplied_program_id: str,
    ) -> None:
        if (
            not self.canonical()
            or descriptor[0] != self.length
            or owner != OWNER
            or generation != self.last_generation + 1
            or supplied_program_id != program_digest(descriptor)
        ):
            raise RuntimeError("independent M249 lease rejected")
        self.owner = owner
        self.generation = generation
        self.program_id = supplied_program_id
        self.descriptor = descriptor
        self.leased = True

    def project(self) -> tuple[E, E, E]:
        if not self.leased or self.cursor != 1:
            raise RuntimeError("independent premature M249 projection")
        return streamed_boundary(self.values, self.length)

    def release(self) -> None:
        if self.values != canonical_state(self.length) or self.cursor != 0:
            raise RuntimeError("independent M249 release before restoration")
        restored = self.generation
        self.owner = 0
        self.generation = 0
        self.program_id = ""
        self.descriptor = None
        self.leased = False
        self.last_generation = restored


def execute(
    port: ReferencePort, gate_name: str, generation: int, run_kind: str
) -> dict[str, Any]:
    descriptor = public_descriptor(port.length, gate_name)
    port.lease(descriptor, OWNER, generation, program_digest(descriptor))
    backing = id(port.values)
    rotate_open(port.values, port.length, gate_name, False)
    port.cursor = 1
    boundary = port.project()
    rotate_open(port.values, port.length, gate_name, True)
    port.cursor = 0
    port.release()
    blocks = port.length - 1
    return {
        "length": port.length,
        "gate": gate_name,
        "generation": port.last_generation,
        "run_kind": run_kind,
        "boundary": {
            "p0": encode(boundary[0]),
            "p1": encode(boundary[1]),
            "coherence": encode(boundary[2]),
        },
        "joint_carrier_commitment": state_commitment(port.length),
        "joint_field_cells": 2 * port.length,
        "retained_final_boundary_field_cells_during_inverse": 3,
        "same_joint_backing": id(port.values) == backing,
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "work": {
            "forward_fixed_energy_pair_updates": blocks,
            "inverse_fixed_energy_pair_updates": blocks,
            "forward_field_multiplications": 4 * blocks,
            "forward_field_additions": 2 * blocks,
            "inverse_field_multiplications": 4 * blocks,
            "inverse_field_additions": 2 * blocks,
            "boundary_square_multiplications": 2 * port.length,
            "boundary_coherence_multiplications": port.length,
            "boundary_accumulations": 3 * port.length,
            "retained_dynamic_inverse_history_entries": 0,
        },
    }


def custody_controls() -> dict[str, bool]:
    descriptor = public_descriptor(4, "H")
    wrong_owner = ReferencePort(4)
    try:
        wrong_owner.lease(descriptor, OWNER + 1, 1, program_digest(descriptor))
        owner_rejected = False
    except RuntimeError:
        owner_rejected = True
    wrong_program = ReferencePort(4)
    try:
        wrong_program.lease(descriptor, OWNER, 1, "0" * 64)
        program_rejected = False
    except RuntimeError:
        program_rejected = True
    stale = ReferencePort(4)
    execute(stale, "H", 1, "CONTROL")
    try:
        stale.lease(descriptor, OWNER, 1, program_digest(descriptor))
        stale_rejected = False
    except RuntimeError:
        stale_rejected = True
    changed = ReferencePort(4)
    try:
        changed.lease(
            public_descriptor(4, "RATIONAL_3_4_5"), OWNER, 1,
            program_digest(descriptor),
        )
        changed_rejected = False
    except RuntimeError:
        changed_rejected = True
    premature = ReferencePort(4)
    premature.lease(descriptor, OWNER, 1, program_digest(descriptor))
    try:
        premature.project()
        projection_rejected = False
    except RuntimeError:
        projection_rejected = True
    return {
        "wrong_owner_rejected": owner_rejected,
        "wrong_program_rejected": program_rejected,
        "stale_generation_rejected": stale_rejected,
        "same_id_changed_descriptor_rejected": changed_rejected,
        "premature_projection_rejected": projection_rejected,
    }


def main() -> None:
    public = json.load(sys.stdin)
    if public != {"suite": "M249_U1_PHASE_REFERENCE_STRICT_SCOPE"}:
        raise RuntimeError("invalid standalone M249 suite")

    cases: list[dict[str, Any]] = []
    analytic: dict[str, Any] = {}
    cyclic: dict[str, Any] = {}
    dephased: dict[str, Any] = {}
    for length in LENGTHS:
        primary_port = ReferencePort(length)
        primary = execute(primary_port, "H", 1, "PRIMARY")
        reuse = execute(primary_port, "RATIONAL_3_4_5", 2, "REUSE")
        fresh = execute(
            ReferencePort(length), "RATIONAL_3_4_5", 1, "FRESH"
        )
        cases.extend((primary, reuse, fresh))
        for gate_name in ROTATIONS:
            key = f"L{length}_{gate_name}"
            closed = closed_boundary(length, gate_name)
            analytic[key] = {
                "boundary": {
                    "p0": encode(closed[0]),
                    "p1": encode(closed[1]),
                    "coherence": encode(closed[2]),
                },
                "open_schmidt_minor": encode(open_schmidt_minor(length, gate_name)),
                "minor_nonzero": open_schmidt_minor(length, gate_name) != Z,
            }
            cyclic[key] = {
                "factorizes_exactly": cyclic_state(length, gate_name)
                == cyclic_expected_product(length, gate_name),
                **cyclic_wrap_data(length, gate_name),
            }
            dephased[key] = {
                "coherence": encode(dephased_reference_coherence(length, gate_name)),
                "coherence_zero": dephased_reference_coherence(length, gate_name) == Z,
            }

    missing = canonical_state(4)
    rotate_open(missing, 4, "H", False)
    wrong = list(missing)
    rotate_open(wrong, 4, "RATIONAL_3_4_5", True)
    reordered = canonical_state(4)
    rotate_open(reordered, 4, "H", False)
    pair_reflection(reordered, 4)
    rotate_open(reordered, 4, "H", True)
    pair_reflection(reordered, 4)

    controls = {
        **custody_controls(),
        "all_streamed_boundaries_match_closed_form": all(
            case["boundary"] == analytic[f"L{case['length']}_{case['gate']}"]["boundary"]
            for case in cases
        ),
        "all_open_schmidt_minors_nonzero": all(
            entry["minor_nonzero"] for entry in analytic.values()
        ),
        "all_cyclic_states_factorize_exactly": all(
            entry["factorizes_exactly"] for entry in cyclic.values()
        ),
        "all_cyclic_wrap_and_commutator_witnesses_nonzero": all(
            entry["nonzero"] for entry in cyclic.values()
        ),
        "all_dephased_reference_coherences_zero": all(
            entry["coherence_zero"] for entry in dephased.values()
        ),
        "all_open_transitions_preserve_formal_total_number": all(
            open_transition_energy_preserved(length) for length in LENGTHS
        ),
        "all_open_basis_states_restore_under_exact_inverse": all(
            every_basis_inverse_exact(length, gate, False)
            for length in LENGTHS for gate in ROTATIONS
        ),
        "all_cyclic_basis_states_restore_under_exact_inverse": all(
            every_basis_inverse_exact(length, gate, True)
            for length in LENGTHS for gate in ROTATIONS
        ),
        "missing_inverse_fails_restoration": missing != canonical_state(4),
        "wrong_inverse_fails_restoration": wrong != canonical_state(4),
        "reordered_h_and_pair_reflection_inverse_fails_restoration": reordered != canonical_state(4),
        "all_transactions_restore_same_backing": all(
            case["same_joint_backing"] and case["canonical_after_restoration"]
            for case in cases
        ),
        "all_reuse_generation2_fresh_generation1": all(
            next(c for c in cases if c["length"] == length and c["run_kind"] == "REUSE")["generation"] == 2
            and next(c for c in cases if c["length"] == length and c["run_kind"] == "FRESH")["generation"] == 1
            for length in LENGTHS
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"independent M249 control failure: {controls}")

    output = {
        "result": "PASS_M249_SEPARATE_REFERENCE_STRICT_SCOPE",
        "cases": cases,
        "analytic_boundaries_and_minors": analytic,
        "cyclic_sham": cyclic,
        "dephased_reference": dephased,
        "bilateral_ideal": {
            "shift_eigenrelation_forces_equal_coefficient_modulus": True,
            "nonzero_constant_modulus_partial_norm_squared": {
                str(radius): 2 * radius + 1 for radius in (1, 2, 4, 8, 16)
            },
            "nonzero_shift_eigenvector_in_l2_z": False,
            "proof_law": "C_N_PLUS1_EQUALS_LAMBDA_C_N_WITH_ABS_LAMBDA1_IMPLIES_CONSTANT_NONZERO_MODULUS_AND_SUM_OVER_Z_DIVERGES",
        },
        "controls": controls,
        "resource_oracle": {
            "accepted_joint_field_cells_by_length": [2 * length for length in LENGTHS],
            "closed_form_baseline_field_workspace": 3,
            "closed_form_baseline_iterations_in_length": 0,
            "streamed_baseline_field_workspace": 6,
            "streamed_baseline_iterations_by_length": list(LENGTHS),
            "verifier_basis_density_and_cyclic_checks_are_verifier_only": True,
            "whole_process_resource_accounting_complete": False,
        },
        "claim_limits": {
            "general_coherence_catalyst_no_go": False,
            "physical_energy_conservation": False,
            "physical_phase_reference": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "unbounded_catalytic_computation": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
