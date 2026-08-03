#!/usr/bin/env python3
"""Independent bitmask oracle for the exact C17 fiber-port package.

This source imports neither the production package nor its M145 dependency.
It reconstructs the public schedule, executes the phase carrier as one-hot
17-bit masks, separately executes the compact residue recurrence, rebuilds
the production byte commitments, attacks inverse/order/port controls, and
repeats restoration and reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


P = 17
MASK = (1 << P) - 1
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
COMPLEX_FAILURE_TOLERANCE = 1.0e-8


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def one_hot(exponent: int) -> int:
    return 1 << (exponent % P)


def rotate(bits: int, amount: int) -> int:
    shift = amount % P
    if shift == 0:
        return bits & MASK
    return ((bits << shift) | (bits >> (P - shift))) & MASK


def convolve(left: int, right: int) -> int:
    result = 0
    for coordinate in range(P):
        if (left >> coordinate) & 1:
            result |= rotate(right, coordinate)
    if result.bit_count() != 1:
        fail("bitmask convolution left the one-hot C17 orbit")
    return result


def inverse_phase(value: int) -> int:
    result = value & 1
    for coordinate in range(1, P):
        if (value >> coordinate) & 1:
            result |= 1 << (P - coordinate)
    return result


def seed_exponents() -> list[list[int]]:
    return [
        [
            (5 * shell + 3) % P,
            (7 * shell * shell + 2 * shell + 1) % P,
            (11 * shell * shell + 4 * shell + 6) % P,
        ]
        for shell in range(P)
    ]


def seed_masks() -> list[list[int]]:
    return [[one_hot(value) for value in row] for row in seed_exponents()]


def clone_state(state: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in state]


def family_code(family: str) -> int:
    return {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]


def ternary_weight(index: int) -> int:
    total = 0
    while index:
        total += index % 3
        index //= 3
    return total


def gate_parameters(index: int, family: str) -> tuple[int, int, int]:
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
    return values[0] % P or 1, values[1] % P, values[2] % P


def phase_exponent(shell: int, index: int, family: str) -> int:
    quadratic, linear, cubic = gate_parameters(index, family)
    return (
        quadratic * shell**4 + linear * shell**2 + cubic * shell
    ) % P


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (3 * index + family_code(family) + mutation) % P


def targets(hub: int) -> list[int]:
    return [(hub + offset) % P for offset in range(1, P)]


def offset_exponent(
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
    ) % P


def observation(depth: int, family: str) -> tuple[int, int]:
    return (
        (7 * depth + 3 * len(family) + 1) % P or 1,
        (11 * depth + len(family) + 5) % P,
    )


def program_descriptor(depth: int, family: str) -> dict[str, Any]:
    return {
        "schema": "CAT_CAS_F17_EXACT_C17_FIBER_PORT_PROGRAM_V1",
        "depth": depth,
        "family": family,
        "topology": "PUBLIC_ROTATING_HUB16_OUT16_IN_TRIANGULAR_SCHEDULE",
        "port_type": "ONE_HOT_C17_GROUP_ALGEBRA_PHASE",
        "observation": list(observation(depth, family)),
    }


def shell_weight(shell: int) -> int:
    return 1 + ((shell * shell + 3 * shell + 2) % 7)


def apply_public_masks(
    state: list[list[int]], index: int, family: str, inverse: bool = False
) -> None:
    for shell in range(P):
        exponent = phase_exponent(shell, index, family)
        factor = one_hot(-exponent if inverse else exponent)
        state[shell] = [convolve(value, factor) for value in state[shell]]


def apply_edge_masks(
    state: list[list[int]],
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    offset_mutation: int = 0,
) -> None:
    offset = one_hot(
        offset_exponent(
            controller, target, index, family, layer, offset_mutation
        )
    )
    factor = convolve(state[controller][0], offset) if port_enabled else offset
    if inverse:
        factor = inverse_phase(factor)
    state[target] = [convolve(value, factor) for value in state[target]]


def apply_layer_masks(
    state: list[list[int]],
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    actors = targets(hub)
    if inverse:
        actors.reverse()
    for actor in actors:
        controller, target = (hub, actor) if layer == 0 else (actor, hub)
        apply_edge_masks(
            state,
            controller,
            target,
            index,
            family,
            layer,
            inverse=inverse,
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
        )


def forward_masks(
    depth: int,
    family: str,
    *,
    initial: list[list[int]] | None = None,
    order: str = "OUT_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
) -> list[list[int]]:
    state = clone_state(seed_masks() if initial is None else initial)
    for index in range(depth):
        apply_public_masks(state, index, family)
        layers = (0, 1) if order == "OUT_IN" else (1, 0)
        for layer in layers:
            apply_layer_masks(
                state,
                index,
                family,
                layer,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
            )
    return state


def inverse_masks(
    state: list[list[int]],
    depth: int,
    family: str,
    mode: str = "VALID",
) -> None:
    for index in range(depth - 1, -1, -1):
        if mode == "REORDERED":
            apply_layer_masks(state, index, family, 0, inverse=True)
            apply_layer_masks(state, index, family, 1, inverse=True)
        else:
            apply_layer_masks(state, index, family, 1, inverse=True)
            if mode != "MISSING_OUT":
                apply_layer_masks(
                    state,
                    index,
                    family,
                    0,
                    inverse=True,
                    offset_mutation=1 if mode == "WRONG_OUT" else 0,
                )
        apply_public_masks(state, index, family, inverse=True)


def apply_public_residues(
    state: list[list[int]], index: int, family: str, inverse: bool = False
) -> None:
    sign = -1 if inverse else 1
    for shell in range(P):
        delta = sign * phase_exponent(shell, index, family)
        state[shell] = [(value + delta) % P for value in state[shell]]


def apply_layer_residues(
    state: list[list[int]],
    index: int,
    family: str,
    layer: int,
    inverse: bool = False,
) -> None:
    hub = hub_index(index, family)
    actors = targets(hub)
    if inverse:
        actors.reverse()
    sign = -1 if inverse else 1
    for actor in actors:
        controller, target = (hub, actor) if layer == 0 else (actor, hub)
        factor = (
            state[controller][0]
            + offset_exponent(controller, target, index, family, layer)
        ) % P
        state[target] = [(value + sign * factor) % P for value in state[target]]


def forward_residues(
    depth: int,
    family: str,
    initial: list[list[int]] | None = None,
) -> list[list[int]]:
    state = [list(row) for row in (seed_exponents() if initial is None else initial)]
    for index in range(depth):
        apply_public_residues(state, index, family)
        apply_layer_residues(state, index, family, 0)
        apply_layer_residues(state, index, family, 1)
    return state


def mask_bytes(state: list[list[int]]) -> bytes:
    return bytes(
        (state[shell][slot] >> coordinate) & 1
        for shell in range(P)
        for slot in range(3)
        for coordinate in range(P)
    )


def masks_to_residues(state: list[list[int]]) -> list[list[int]]:
    result = []
    for row in state:
        converted = []
        for value in row:
            if value.bit_count() != 1:
                fail("oracle phase mask is not one-hot")
            converted.append(value.bit_length() - 1)
        result.append(converted)
    return result


def boundary_from_residues(
    state: list[list[int]], depth: int, family: str
) -> tuple[int, ...]:
    quadratic, linear = observation(depth, family)
    coefficients = [0 for _ in range(P)]
    for shell in range(P):
        shift = (quadratic * shell * shell + linear * shell) % P
        for slot, slot_weight in enumerate((3, 1, 1)):
            coordinate = (state[shell][slot] + shift) % P
            coefficients[coordinate] += shell_weight(shell) * slot_weight
    tail = coefficients[-1]
    return tuple(value - tail for value in coefficients[:-1])


def complex_attempt() -> dict[str, Any]:
    roots = [
        complex(math.cos(2.0 * math.pi * value / P), math.sin(2.0 * math.pi * value / P))
        for value in range(P)
    ]
    observed = []
    for depth in (1, 4, 16, 64):
        state = [[roots[value] for value in row] for row in seed_exponents()]
        finite = True
        for index in range(depth):
            try:
                for shell in range(P):
                    factor = roots[phase_exponent(shell, index, "PRIMARY")]
                    state[shell] = [value * factor for value in state[shell]]
                hub = hub_index(index, "PRIMARY")
                for target in targets(hub):
                    factor = state[hub][0] * roots[
                        offset_exponent(hub, target, index, "PRIMARY", 0)
                    ]
                    state[target] = [value * factor for value in state[target]]
                for controller in targets(hub):
                    factor = state[controller][0] * roots[
                        offset_exponent(controller, hub, index, "PRIMARY", 1)
                    ]
                    state[hub] = [value * factor for value in state[hub]]
            except OverflowError:
                finite = False
                break
            finite = all(
                math.isfinite(value.real) and math.isfinite(value.imag)
                for row in state
                for value in row
            )
            if not finite:
                break
        error = (
            max(abs(abs(value) - 1.0) for row in state for value in row)
            if finite
            else None
        )
        observed.append(
            {
                "depth": depth,
                "all_coordinates_finite": finite,
                "maximum_unit_norm_error": error,
                "within_failure_tolerance": finite
                and error is not None
                and error <= COMPLEX_FAILURE_TOLERANCE,
            }
        )
    failures = [item["depth"] for item in observed if not item["within_failure_tolerance"]]
    return {
        "observed": observed,
        "first_failure_depth": failures[0] if failures else None,
    }


def controls() -> dict[str, bool]:
    depth = 4
    family = "ALTERNATE"
    valid = forward_masks(depth, family)
    valid_boundary = boundary_from_residues(
        masks_to_residues(valid), depth, family
    )
    disabled = forward_masks(depth, family, port_enabled=False)
    order = forward_masks(depth, family, order="IN_OUT")
    topology = forward_masks(depth, family, hub_mutation=1)
    inverse_failures = {}
    for mode in ("MISSING_OUT", "WRONG_OUT", "REORDERED"):
        state = forward_masks(depth, family)
        inverse_masks(state, depth, family, mode)
        inverse_failures[mode] = state != seed_masks()
    gauge_shifted = seed_masks()
    for shell in range(P):
        gauge_shifted[shell][0] = convolve(gauge_shifted[shell][0], one_hot(5))
    shifted = forward_masks(4, "PRIMARY", initial=gauge_shifted)
    base = forward_masks(4, "PRIMARY")
    shifted_boundary = boundary_from_residues(
        masks_to_residues(shifted), 4, "PRIMARY"
    )
    base_boundary = boundary_from_residues(
        masks_to_residues(base), 4, "PRIMARY"
    )
    inverse_masks(shifted, 4, "PRIMARY")
    inverse_masks(base, 4, "PRIMARY")
    return {
        "resident_port_factor_changes_boundary": valid_boundary
        != boundary_from_residues(masks_to_residues(disabled), depth, family),
        "out_in_layer_order_changes_boundary": valid_boundary
        != boundary_from_residues(masks_to_residues(order), depth, family),
        "public_hub_topology_mutation_changes_boundary": valid_boundary
        != boundary_from_residues(masks_to_residues(topology), depth, family),
        "missing_out_layer_inverse_changes_actual_carrier": inverse_failures[
            "MISSING_OUT"
        ],
        "wrong_out_layer_inverse_changes_actual_carrier": inverse_failures[
            "WRONG_OUT"
        ],
        "reordered_inverse_changes_actual_carrier": inverse_failures["REORDERED"],
        "gauge_only_port_perturbation_changes_boundary": base_boundary
        != shifted_boundary,
        "gauge_only_port_perturbation_both_restore": base == seed_masks()
        and shifted == gauge_shifted,
    }


def verify(
    package_path: Path,
    production_path: Path,
    predecessor_path: Path,
) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    comparisons = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal comparisons
        comparisons += 1
        if not condition:
            fail(message)

    production_hash = sha256_file(production_path)
    predecessor_hash = sha256_file(predecessor_path)
    expect(package["source_sha256"] == production_hash, "production hash changed")
    expect(
        package["production_dependency"]["sha256"] == predecessor_hash,
        "public schedule dependency hash changed",
    )
    expect(package["execution_scope"]["case_count"] == 21, "case count changed")
    expect(package["execution_scope"]["all_cases_exact"], "package exact gate failed")
    expect(
        not package["execution_scope"]["catvm_machine_boundary_used"],
        "direct process package claimed CATVM",
    )

    case_summaries = []
    package_cases = package["cases"]
    expected_order = [(family, depth) for family in FAMILIES for depth in DEPTHS]
    expect(len(package_cases) == len(expected_order), "package case vector changed")
    for observed, (family, depth) in zip(package_cases, expected_order):
        masks = forward_masks(depth, family)
        residues = forward_residues(depth, family)
        mask_residues = masks_to_residues(masks)
        boundary = boundary_from_residues(residues, depth, family)
        commitment = hashlib.sha256(mask_bytes(masks)).hexdigest()
        expect(observed["family"] == family, "case family changed")
        expect(observed["depth"] == depth, "case depth changed")
        expect(
            observed["program_fingerprint"]
            == digest_json(program_descriptor(depth, family)),
            "program fingerprint changed",
        )
        expect(mask_residues == residues, "mask and residue recurrences diverged")
        expect(observed["final_state_commitment"] == commitment, "commitment changed")
        expect(observed["final_boundary"] == list(boundary), "boundary changed")
        expect(
            observed["matched_residue_boundary"] == list(boundary),
            "matched boundary changed",
        )
        expect(
            observed[
                "final_phase_orbit_bytes_identical_to_expanded_matched_recurrence"
            ],
            "phase/residue parity field failed",
        )
        expect(observed["exact_restoration"], "case restoration field failed")
        expect(observed["same_backing"], "case backing field failed")
        expected_convolutions = 358 * depth + 51
        stats = observed["stats"]
        expect(
            stats["cyclic_convolutions"] == expected_convolutions,
            "convolution count changed",
        )
        expect(
            stats["convolution_coordinate_multiplications"]
            == expected_convolutions * P * P,
            "coordinate work count changed",
        )
        restored = clone_state(masks)
        inverse_masks(restored, depth, family)
        expect(restored == seed_masks(), "oracle inverse failed")
        case_summaries.append(
            {
                "family": family,
                "depth": depth,
                "program_fingerprint": observed["program_fingerprint"],
                "final_state_commitment": commitment,
                "final_boundary_sha256": digest_json(list(boundary)),
                "exact_restoration": restored == seed_masks(),
            }
        )

    independent_controls = controls()
    expect(all(independent_controls.values()), "independent control failed")
    for key, value in independent_controls.items():
        expect(package["controls"][key] == value, f"package control changed: {key}")
    for key in (
        "premature_final_projection_rejected",
        "resident_port_projection_rejected",
        "wrong_program_ownership_rejected",
        "null_carrier_rejected",
    ):
        expect(package["controls"][key], f"service-local guard failed: {key}")

    first = forward_masks(37, "PRIMARY")
    inverse_masks(first, 37, "PRIMARY")
    expect(first == seed_masks(), "pre-reuse inverse failed")
    restored_reuse = forward_masks(REUSE_DEPTH, "REUSE", initial=first)
    fresh_reuse = forward_masks(REUSE_DEPTH, "REUSE")
    expect(restored_reuse == fresh_reuse, "fresh/restored reuse state differs")
    expect(
        boundary_from_residues(
            masks_to_residues(restored_reuse), REUSE_DEPTH, "REUSE"
        )
        == boundary_from_residues(
            masks_to_residues(fresh_reuse), REUSE_DEPTH, "REUSE"
        ),
        "fresh/restored reuse boundary differs",
    )
    inverse_masks(restored_reuse, REUSE_DEPTH, "REUSE")
    expect(restored_reuse == seed_masks(), "unrelated reuse inverse failed")

    repeated = seed_masks()
    for _ in range(REPEATED_REUSE_CYCLES):
        repeated = forward_masks(
            REPEATED_REUSE_DEPTH, "ALTERNATE", initial=repeated
        )
        inverse_masks(repeated, REPEATED_REUSE_DEPTH, "ALTERNATE")
        if repeated != seed_masks():
            fail("repeated reuse inverse failed")
    expect(repeated == seed_masks(), "repeated reuse final state changed")

    numeric = complex_attempt()
    package_numeric = package["rejected_unrenormalized_complex_coordinate_attempt"]
    expect(numeric["first_failure_depth"] == 64, "complex diagnostic changed")
    expect(
        package_numeric["observed_first_unit_norm_failure_depth"]
        == numeric["first_failure_depth"],
        "package complex failure depth changed",
    )
    for package_item, oracle_item in zip(
        package_numeric["observed"], numeric["observed"]
    ):
        expect(
            package_item["depth"] == oracle_item["depth"],
            "complex diagnostic depth changed",
        )
        expect(
            package_item["all_coordinates_finite"]
            == oracle_item["all_coordinates_finite"],
            "complex finiteness changed",
        )
        expect(
            package_item["within_failure_tolerance"]
            == oracle_item["within_failure_tolerance"],
            "complex failure decision changed",
        )

    resources = package["resource_law"]
    baseline = package["matched_classical_recurrence"]
    expect(resources["resident_phase_orbit_bytes"] == 867, "phase bytes changed")
    expect(baseline["resident_bytes"] == 51, "baseline bytes changed")
    expect(
        baseline["phase_to_classical_resident_byte_ratio"] == 17.0,
        "resident ratio changed",
    )
    expect(
        resources["forward_cyclic_convolutions_per_step"] == 179,
        "forward convolution law changed",
    )
    expect(
        resources["forward_convolution_coordinate_multiplications_per_step"]
        == 179 * P * P,
        "forward coordinate work law changed",
    )
    expect(
        package["restoration"]["class"] == "EXACT_ALGEBRAIC_RESTORATION",
        "restoration class changed",
    )
    expect(
        package["restoration"]["transient_buffers"] == "NO_RESTORATION_CLAIM",
        "transient restoration class changed",
    )
    expect(
        not baseline["comparison_establishes_distinct_phase_resource"],
        "package promoted a phase resource",
    )
    expect(
        not baseline["comparison_establishes_computational_advantage"],
        "package promoted an advantage",
    )

    production_text = production_path.read_text(encoding="utf-8")
    expect(
        not re.search(
            r"^(?:from|import)\s+f17_exact_c17_fiber_port_convolution",
            Path(__file__).read_text(encoding="utf-8"),
            re.MULTILINE,
        ),
        "oracle imports production",
    )
    expect("np.argmax" not in production_text, "production decodes phase exponent")
    expect("atan2" not in production_text, "production reads a scalar phase angle")

    return {
        "schema": "CAT_CAS_F17_EXACT_C17_FIBER_PORT_CONVOLUTION_ORACLE_V1",
        "result": "PASS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "source_sha256": sha256_file(Path(__file__)),
        "audited_production_sha256": production_hash,
        "audited_public_schedule_dependency_sha256": predecessor_hash,
        "independence": {
            "imports_production_module": False,
            "imports_public_schedule_dependency": False,
            "representation": "ONE_HOT17_BITMASK_GROUP_ACTION",
            "separate_public_schedule_compiler": True,
            "separate_mask_convolution_forward_inverse": True,
            "separate_compact51_residue_recurrence": True,
            "reconstructs_production867_byte_commitments": True,
            "separate_final_cyclotomic_boundary": True,
            "separate_controls": True,
            "separate_unrelated_and100_cycle_reuse": True,
            "separate_unrenormalized_complex_coordinate_diagnostic": True,
        },
        "case_checks": {
            "case_count": len(case_summaries),
            "all_mask_residue_states_equal": True,
            "all_production_commitments_reconstructed": True,
            "all_boundaries_equal": True,
            "all_exact_restorations_reexecuted": True,
            "cases": case_summaries,
        },
        "controls": independent_controls,
        "reuse": {
            "unrelated_reuse_depth": REUSE_DEPTH,
            "fresh_restored_state_equal": True,
            "fresh_restored_boundary_equal": True,
            "restored_exactly_after_reuse": True,
            "repeated_reuse_cycles": REPEATED_REUSE_CYCLES,
            "repeated_reuse_depth": REPEATED_REUSE_DEPTH,
            "repeated_reuse_exact": True,
        },
        "unrenormalized_complex_coordinate_diagnostic": numeric,
        "observed_resource_law": {
            "accepted_resident_phase_orbit_bytes": 867,
            "matched_resident_residue_bytes": 51,
            "resident_byte_ratio": 17.0,
            "accepted_forward_logical_convolutions_per_step": 179,
            "accepted_forward_coordinate_multiplications_per_step": 179 * P * P,
            "matched_forward_modular_additions_per_step": 179,
            "allocator_native_library_and_whole_process_memory_excluded": True,
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "transient_restoration_class": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "FIXED51_ONE_HOT_C17_PHASE_FACTORS_ON_THE_DECLARED_TRIANGULAR_TOPOLOGY_THROUGH_DEPTH4096",
        "preserved_subclaims": [
            "DIRECT_RESIDENT_PHASE_PORT_CONVOLUTION_WITHOUT_PRODUCTION_EXPONENT_READOUT",
            "MULTIPLE_NONCOMMUTING_PORT_CONSUMERS",
            "FINAL_ONLY_BOUNDARY_PROJECTION",
            "EXACT_SAME_BACKING_RESTORATION_AND_REUSE",
        ],
        "rejected_interpretations": [
            "RESOURCE_BEYOND51_RESIDUE_RECURRENCE",
            "COMPUTATIONAL_ADVANTAGE",
            "GENERAL_RELATIONAL_CONTRACTION",
            "CATVM_MACHINE_ENFORCED_CUSTODY",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "comparison_count": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package, args.production, args.predecessor)
    payload = canonical_json(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
