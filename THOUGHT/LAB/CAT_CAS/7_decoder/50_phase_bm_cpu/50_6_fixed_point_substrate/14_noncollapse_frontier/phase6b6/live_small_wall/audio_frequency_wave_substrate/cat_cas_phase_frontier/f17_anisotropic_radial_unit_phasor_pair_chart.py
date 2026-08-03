#!/usr/bin/env python3
"""Numerical phase-only chart for the M140 anisotropic radial machine.

Each normalized complex radial coefficient is resident only as two phase
angles ``alpha,beta`` with

    z = (exp(i*alpha) + exp(i*beta)) / 2.

Quartic phase gates translate both angles directly.  The radial Fourier is
the M140 matrix-free 16-character contraction followed by the canonical
two-phasor chart map.  This changes the primitive resident state from exact
cyclotomic coefficients to bounded numerical phase coordinates.  It does not
remove the identical compact classical 17-complex recurrence or establish a
physical phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


P = 17
NONSQUARE = 3
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = np.asarray((1, *([18] * 16)), dtype=np.int64)
TAU = 2.0 * math.pi
RESTORATION_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 5.0e-11
CHART_ZERO_FLOOR = 1.0e-7
CHART_RADIUS_TOLERANCE = 2.0e-12
CONTROL_FLOOR = 1.0e-5
REPEATED_REUSE_CYCLES = 100
REPEATED_REUSE_DEPTH = 64
REUSE_DEPTH = 1537
CLAIM = (
    "BOUNDED_NUMERICAL_UNIT_PHASOR_PAIR_CHART_REPRESENTS_NORMALIZED_"
    "ANISOTROPIC_F17_RADIAL_PHASE_FOURIER_STATE_IN_FIXED34_PHASE_"
    "ANGLE_COORDINATES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_"
    "FREE_NUMERICAL_PHYSICAL_STATE_RESTORATION_AND_REUSE_BUT_REQUIRES_"
    "TRANSCENDENTAL_CANONICAL_CHART_ARITHMETIC_AND_HAS_THE_IDENTICAL17_"
    "COMPLEX_COMPACT_CLASSICAL_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def wrap_phase(values: np.ndarray) -> np.ndarray:
    return np.remainder(values + math.pi, TAU) - math.pi


def phasor_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.exp(1j * left) - np.exp(1j * right)))
    )


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


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
        fail("unknown phase-pair public family")
    quadratic = values[0] % P or 1
    return quadratic, values[1] % P, values[2] % P


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_STREAMED_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "gate_generator": "PUBLIC_INDEX_BIT_GRAY_TERNARY_WEIGHT_FORMULA",
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 4096:
        fail("phase-pair program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("phase-pair program family outside declared set")
    program = Program(
        depth,
        family,
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )
    # Compile-time topology work is independent of the answer and does not
    # materialize the depth-length gate sequence.
    gate_parameters(0, family)
    gate_parameters(depth - 1, family)
    return program


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    streamed_gate_rematerializations: int = 0
    matrix_free_fourier_calls: int = 0
    generated_character_products: int = 0
    chart_decodes: int = 0
    chart_encodes: int = 0
    chart_domain_checks: int = 0
    maximum_named_update_complex_cells: int = 0
    maximum_named_update_float_cells: int = 0
    maximum_named_update_bytes: int = 0
    maximum_named_projection_bytes: int = 0

    def observe_update(self, complex_cells: int, float_cells: int) -> None:
        self.maximum_named_update_complex_cells = max(
            self.maximum_named_update_complex_cells, complex_cells
        )
        self.maximum_named_update_float_cells = max(
            self.maximum_named_update_float_cells, float_cells
        )
        self.maximum_named_update_bytes = max(
            self.maximum_named_update_bytes,
            complex_cells * np.dtype(np.complex128).itemsize
            + float_cells * np.dtype(np.float64).itemsize,
        )


@dataclass(frozen=True)
class Geometry:
    roots: np.ndarray
    inverse_four_parameters: np.ndarray
    source_indices: np.ndarray

    @classmethod
    def compile(cls) -> "Geometry":
        squares = {value * value % P for value in range(P)}
        if NONSQUARE in squares:
            fail("anisotropic coefficient became a square")
        roots = np.exp(2j * math.pi * np.arange(P) / P).astype(
            np.complex128
        )
        inverse_four_parameters = np.asarray(
            [pow((4 * value) % P, -1, P) for value in range(1, P)],
            dtype=np.int64,
        )
        source_indices = np.arange(P, dtype=np.int64)
        return cls(roots, inverse_four_parameters, source_indices)


def seed_angles() -> np.ndarray:
    amplitude = np.full(P, 1.0 / P, dtype=np.complex128)
    return encode(amplitude, None)


def decode(angles: np.ndarray, stats: Stats | None) -> np.ndarray:
    left = np.exp(1j * angles[:, 0])
    right = np.exp(1j * angles[:, 1])
    result = 0.5 * (left + right)
    if stats is not None:
        stats.chart_decodes += 1
        stats.observe_update(3 * P, 0)
    return result


def encode(values: np.ndarray, stats: Stats | None) -> np.ndarray:
    magnitudes = np.abs(values)
    if (
        not np.all(np.isfinite(values))
        or float(np.min(magnitudes)) <= CHART_ZERO_FLOOR
        or float(np.max(magnitudes)) > 1.0 + CHART_RADIUS_TOLERANCE
    ):
        fail("phase-pair chart left its declared nonzero unit disk")
    phases = np.angle(values)
    deltas = np.arccos(np.clip(magnitudes, 0.0, 1.0))
    result = wrap_phase(
        np.stack((phases + deltas, phases - deltas), axis=1)
    )
    if stats is not None:
        stats.chart_encodes += 1
        stats.chart_domain_checks += 1
        # Input complex vector, magnitude/phase/delta arrays, and result.
        stats.observe_update(P, 5 * P)
    return result


def phase_exponents(index: int, family: str) -> np.ndarray:
    quadratic, linear, cubic = gate_parameters(index, family)
    shells = np.arange(P, dtype=np.int64)
    return (
        quadratic * shells**4 + linear * shells**2 + cubic * shells
    ) % P


def apply_phase(
    angles: np.ndarray,
    index: int,
    family: str,
    stats: Stats,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    exponents = phase_exponents(index, family)
    delta = sign * (TAU / P) * exponents.astype(np.float64)
    angles[:] = wrap_phase(angles + delta[:, None])
    stats.streamed_gate_rematerializations += 1
    stats.observe_update(0, 4 * P)


def matrix_free_fourier(
    geometry: Geometry, state: np.ndarray, stats: Stats | None
) -> np.ndarray:
    spectrum = np.empty(P - 1, dtype=np.complex128)
    for offset, parameter in enumerate(range(1, P)):
        characters = geometry.roots[
            (-geometry.source_indices * parameter) % P
        ]
        spectrum[offset] = np.dot(characters, state)
        if stats is not None:
            stats.observe_update(2 * P + len(spectrum), 0)
    state_sum = np.sum(state)
    output = np.empty(P, dtype=np.complex128)
    for target in range(P):
        characters = geometry.roots[
            (-target * geometry.inverse_four_parameters) % P
        ]
        accumulator = np.dot(characters, spectrum)
        output[target] = (
            state_sum if target == 0 else 0.0j
        ) - accumulator / P
        if stats is not None:
            stats.observe_update(2 * P + 2 * len(spectrum), 0)
    if stats is not None:
        stats.matrix_free_fourier_calls += 1
        stats.generated_character_products += 2 * P * (P - 1)
    return output


def apply_fourier(
    carrier: "Carrier", geometry: Geometry, stats: Stats
) -> None:
    state = decode(carrier.angles, stats)
    transformed = matrix_free_fourier(geometry, state, stats)
    encoded = encode(transformed, stats)
    carrier.angles[:] = encoded


@dataclass
class Carrier:
    angles: np.ndarray
    stage: str = "RESTORED"
    active_program: str | None = None
    forward_index: int = 0
    inverse_index: int = 0
    projection_calls: int = 0
    restoration_generation: int = 0
    stats: Stats = field(default_factory=Stats)

    @classmethod
    def create(cls) -> "Carrier":
        return cls(seed_angles())

    def backing_identity(self) -> tuple[int, int]:
        return id(self), int(self.angles.__array_interface__["data"][0])

    def restored_error(self) -> float:
        return phasor_distance(self.angles, seed_angles())


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong phase-pair carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("phase-pair carrier owner or stage changed")


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null phase-pair carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or carrier.restored_error() > RESTORATION_TOLERANCE
    ):
        fail("phase-pair carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def forward(carrier: Carrier, program: Program, geometry: Geometry) -> None:
    require_owned(carrier, program, "FORWARD")
    for index in range(program.depth):
        apply_phase(carrier.angles, index, program.family, carrier.stats)
        apply_fourier(carrier, geometry, carrier.stats)
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_PHASE_PAIR_STATE_RESIDENT"


def project(carrier: Carrier, program: Program, geometry: Geometry) -> complex:
    require_owned(carrier, program, "FINAL_PHASE_PAIR_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("phase-pair final projection order changed")
    state = decode(carrier.angles, carrier.stats)
    shells = np.arange(P, dtype=np.int64)
    exponents = (
        program.observation_quadratic * shells * shells
        + program.observation_linear * shells
    ) % P
    phases = geometry.roots[exponents]
    weighted = SHELL_COUNTS * phases * state
    boundary = complex(np.sum(weighted))
    carrier.stats.maximum_named_projection_bytes = max(
        carrier.stats.maximum_named_projection_bytes,
        state.nbytes + phases.nbytes + weighted.nbytes,
    )
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return boundary


def inverse(carrier: Carrier, program: Program, geometry: Geometry) -> float:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("phase-pair inverse requires one final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_fourier(carrier, geometry, carrier.stats)
        apply_phase(
            carrier.angles,
            index,
            program.family,
            carrier.stats,
            inverse=True,
        )
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("phase-pair actual inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(carrier.angles.tobytes(order="C")).hexdigest()


def classical_boundary(
    program: Program, geometry: Geometry
) -> tuple[complex, np.ndarray]:
    state = np.full(P, 1.0 / P, dtype=np.complex128)
    for index in range(program.depth):
        state *= geometry.roots[phase_exponents(index, program.family)]
        state = matrix_free_fourier(geometry, state, None)
    shells = np.arange(P, dtype=np.int64)
    exponents = (
        program.observation_quadratic * shells * shells
        + program.observation_linear * shells
    ) % P
    boundary = complex(
        np.sum(SHELL_COUNTS * geometry.roots[exponents] * state)
    )
    return boundary, state


def execute_case(geometry: Geometry, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program, geometry)
    commitment = state_commitment(carrier)
    boundary = project(carrier, program, geometry)
    classical, classical_state = classical_boundary(program, geometry)
    state_error = float(
        np.max(np.abs(decode(carrier.angles, None) - classical_state))
    )
    boundary_error = abs(boundary - classical)
    if boundary_error > BOUNDARY_TOLERANCE:
        fail("phase-pair boundary exceeded matched classical tolerance")
    restoration_error = inverse(carrier, program, geometry)
    if carrier.backing_identity() != backing:
        fail("phase-pair carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_classical_boundary": complex_pair(classical),
        "maximum_state_error_against_matched_classical": state_error,
        "boundary_error_against_matched_classical": boundary_error,
        "restoration_error": restoration_error,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "resident_restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "transient_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "phase_angle_resident_cells": int(carrier.angles.size),
        "phase_angle_resident_bytes": int(carrier.angles.nbytes),
        "stats": vars(carrier.stats),
    }


def run_transaction(
    carrier: Carrier, program: Program, geometry: Geometry
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program, geometry)
    boundary = project(carrier, program, geometry)
    error = inverse(carrier, program, geometry)
    return boundary, error


def reuse_control(geometry: Geometry) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, compile_program(64, "PRIMARY"), geometry)
    restored_boundary, restoration_error = run_transaction(
        carrier, compile_program(REUSE_DEPTH, "REUSE"), geometry
    )
    fresh = Carrier.create()
    fresh_boundary, _ = run_transaction(
        fresh, compile_program(REUSE_DEPTH, "REUSE"), geometry
    )
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_boundary_error": abs(
            fresh_boundary - restored_boundary
        ),
        "restoration_error": restoration_error,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control(geometry: Geometry) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    program = compile_program(REPEATED_REUSE_DEPTH, "PRIMARY")
    maximum_error = 0.0
    for _ in range(REPEATED_REUSE_CYCLES):
        _, error = run_transaction(carrier, program, geometry)
        maximum_error = max(maximum_error, error)
    return {
        "cycles": REPEATED_REUSE_CYCLES,
        "depth_per_cycle": REPEATED_REUSE_DEPTH,
        "maximum_restoration_error": maximum_error,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def raw_reverse(
    angles: np.ndarray,
    program: Program,
    geometry: Geometry,
    *,
    reordered: bool = False,
) -> np.ndarray:
    carrier = Carrier(np.array(angles, copy=True), stage="RAW")
    stats = carrier.stats
    for index in range(program.depth - 1, -1, -1):
        if reordered:
            apply_phase(
                carrier.angles,
                index,
                program.family,
                stats,
                inverse=True,
            )
            apply_fourier(carrier, geometry, stats)
        else:
            apply_fourier(carrier, geometry, stats)
            apply_phase(
                carrier.angles,
                index,
                program.family,
                stats,
                inverse=True,
            )
    return carrier.angles


def controls(geometry: Geometry) -> dict[str, bool]:
    primary = compile_program(4, "PRIMARY")
    carrier = Carrier.create()
    begin_forward(carrier, primary)
    premature_projection_rejected = False
    try:
        project(carrier, primary, geometry)
    except RuntimeError:
        premature_projection_rejected = True
    forward(carrier, primary, geometry)
    final_angles = np.array(carrier.angles, copy=True)
    missing_inverse_changes_state = (
        phasor_distance(final_angles, seed_angles()) > CONTROL_FLOOR
    )
    wrong = raw_reverse(
        final_angles, compile_program(4, "REUSE"), geometry
    )
    wrong_inverse_changes_state = (
        phasor_distance(wrong, seed_angles()) > CONTROL_FLOOR
    )
    reordered = raw_reverse(
        final_angles, primary, geometry, reordered=True
    )
    reordered_inverse_changes_state = (
        phasor_distance(reordered, seed_angles()) > CONTROL_FLOOR
    )
    reference = project(carrier, primary, geometry)
    inverse(carrier, primary, geometry)

    disabled = np.full(P, 1.0 / P, dtype=np.complex128)
    for _ in range(primary.depth):
        disabled = matrix_free_fourier(geometry, disabled, None)
    shells = np.arange(P, dtype=np.int64)
    observation = geometry.roots[
        (
            primary.observation_quadratic * shells * shells
            + primary.observation_linear * shells
        )
        % P
    ]
    disabled_boundary = complex(
        np.sum(SHELL_COUNTS * observation * disabled)
    )
    phase_disabled_changes_boundary = (
        abs(disabled_boundary - reference) > CONTROL_FLOOR
    )

    null_rejected = False
    try:
        begin_forward(None, primary)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    perturbed = Carrier.create()
    begin_forward(perturbed, primary)
    forward(perturbed, primary, geometry)
    project(perturbed, primary, geometry)
    perturbed.angles[0, 0] += 1.0e-3
    perturbation_rejected = False
    try:
        inverse(perturbed, primary, geometry)
    except RuntimeError:
        perturbation_rejected = True

    result = {
        "premature_projection_rejected": premature_projection_rejected,
        "missing_inverse_changes_state": missing_inverse_changes_state,
        "wrong_inverse_changes_state": wrong_inverse_changes_state,
        "reordered_inverse_changes_state": reordered_inverse_changes_state,
        "phase_disabled_changes_boundary": phase_disabled_changes_boundary,
        "null_carrier_rejected": null_rejected,
        "phase_perturbation_rejected_by_restoration": perturbation_rejected,
    }
    if not all(result.values()):
        fail("phase-pair control failed")
    return result


def run() -> dict[str, Any]:
    geometry = Geometry.compile()
    cases = [
        execute_case(geometry, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    if not all(
        case["boundary_error_against_matched_classical"]
        <= BOUNDARY_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        for case in cases
    ):
        fail("phase-pair declared case failed")
    reuse = reuse_control(geometry)
    repeated = repeated_reuse_control(geometry)
    if not (
        reuse["same_original_backing"]
        and reuse["fresh_restored_boundary_error"] <= BOUNDARY_TOLERANCE
        and reuse["restoration_error"] <= RESTORATION_TOLERANCE
        and reuse["restoration_generation"] == 2
        and repeated["same_backing"]
        and repeated["maximum_restoration_error"]
        <= RESTORATION_TOLERANCE
        and repeated["restoration_generation"] == REPEATED_REUSE_CYCLES
    ):
        fail("phase-pair reuse control failed")
    control_results = controls(geometry)
    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_UNIT_PHASOR_PAIR_CHART_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_SOFTWARE",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
            "chart_zero_floor": CHART_ZERO_FLOOR,
            "chart_radius_slack": CHART_RADIUS_TOLERANCE,
        },
        "source_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "repeated_reuse_cycles": REPEATED_REUSE_CYCLES,
            "unrelated_reuse_depth": REUSE_DEPTH,
        },
        "cases": cases,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "controls": control_results,
        "resource_law": {
            "resident_phase_angle_float64_cells": 34,
            "resident_phase_angle_bytes": 272,
            "matched_classical_resident_complex128_cells": 17,
            "matched_classical_resident_bytes": 272,
            "retained_root_table_complex128_cells": 17,
            "retained_root_table_bytes": 272,
            "retained_inverse_parameter_int64_cells": 16,
            "retained_inverse_parameter_bytes": 128,
            "retained_source_index_int64_cells": 17,
            "retained_source_index_bytes": 136,
            "retained_shell_count_int64_cells": 17,
            "retained_shell_count_bytes": 136,
            "retained_dense_fourier_kernel_cells": 0,
            "compiled_gate_sequence_cells": 0,
            "public_program_descriptor_scalars": 4,
            "maximum_public_program_json_bytes": max(
                case["public_program_json_bytes"] for case in cases
            ),
            "inverse_history_cells": 0,
            "matrix_free_character_products_per_fourier": 544,
            "maximum_named_update_bytes": max(
                case["stats"]["maximum_named_update_bytes"]
                for case in cases
            ),
            "maximum_named_projection_bytes": max(
                case["stats"]["maximum_named_projection_bytes"]
                for case in cases
            ),
            "maximum_named_accepted_live_bytes_including_program_json": (
                272 + 272 + 128 + 136 + 136
                + max(
                    case["stats"]["maximum_named_update_bytes"]
                    for case in cases
                )
                + max(case["public_program_json_bytes"] for case in cases)
            ),
            "verification_baseline_resident_complex128_cells": 17,
            "verification_baseline_is_not_accepted_phase_state": True,
            "chart_transcendentals_per_fourier": (
                "34_COMPLEX_EXP_PLUS17_ABS_PLUS17_ARG_PLUS17_ACOS"
            ),
            "named_temporary_bytes_are_component_maxima_not_process_peak": True,
            "python_numpy_allocator_native_library_hashlib_and_whole_process_memory_excluded": True,
        },
        "matched_classical_baseline": {
            "method": "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE_WITHOUT_PHASE_PAIR_CHART",
            "resident_bytes_equal": True,
            "phase_chart_requires_additional_transcendental_work": True,
            "all_case_boundaries_within_predeclared_tolerance": True,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "restoration": {
            "class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
            "post_inverse_state_reset_or_canonical_reload_used": False,
            "transient_buffers": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "fresh_restored_reuse_equal_within_tolerance": True,
            "repeated_reuse_tested": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "generation_is_package_local_not_catvm_lease": True,
        },
        "claim_boundary": {
            "established": [
                "FIXED34_PHASE_ANGLE_RESIDENCY_FOR21_DECLARED_NUMERICAL_CASES_THROUGH_DEPTH4096",
                "MATRIX_FREE_NONCOMMUTING_QUARTIC_PHASE_AND_RADIAL_FOURIER_CHART_CLOSURE",
                "FINAL_ONLY_BOUNDARY_PROJECTION",
                "HISTORY_FREE_NUMERICAL_INVERSE_RESTORATION_ON_SAME_BACKING",
                "UNRELATED_AND100_CYCLE_REUSE_WITHIN_PREDECLARED_TOLERANCE",
            ],
            "not_established": [
                "EXACT_ALGEBRAIC_SEMANTICS",
                "UNBOUNDED_DEPTH_NUMERICAL_STABILITY",
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_EXECUTION",
                "PHYSICAL_BIT_REPLACEMENT",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "THE_FIXED34_PHASE_ANGLE_CHART_ACCEPTS_TOLERANCE_DEFINED_"
            "CONTINUOUS_STATE_REQUIRES_TRANSCENDENTAL_CANONICAL_CHART_"
            "ARITHMETIC_AND_HAS_THE_IDENTICAL17_COMPLEX_CLASSICAL_"
            "RECURRENCE_WITH_EQUAL_RESIDENT_BYTES"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
