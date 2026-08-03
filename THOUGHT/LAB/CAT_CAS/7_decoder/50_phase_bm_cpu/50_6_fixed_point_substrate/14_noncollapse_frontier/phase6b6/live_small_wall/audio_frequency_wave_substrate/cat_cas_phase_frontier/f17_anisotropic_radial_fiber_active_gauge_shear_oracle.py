#!/usr/bin/env python3
"""Independent oracle for the bounded fiber-active F17 gauge shear.

This file imports neither M147 production nor the M146 predecessor.  It
reimplements the weighted three-phasor chart, gauge shear, local gauge
transport, forward/inverse order, causality witness, and compact scalar
recurrence.  It reuses the independently established M145 public program and
Givens-plan compiler as declared topology input.
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

import f17_anisotropic_radial_local_polar_givens_coupling as m145


P = 17
TAU = 2.0 * math.pi
EPSILON = 1.0 / 32.0
RESIDUAL_WEIGHT = 1.0 - EPSILON
SHEAR_STRENGTH = 3.0 / 16.0
SUPPORTED_MAGNITUDE = 15.0 / 16.0
RADIUS_SLACK = 2.0e-12
ZERO_FLOOR = 1.0e-14
STATE_TOLERANCE = 3.0e-11
GAUGE_TOLERANCE = 3.0e-11
BOUNDARY_TOLERANCE = 2.0e-10
RESTORATION_TOLERANCE = 5.0e-11
CONTROL_FLOOR = 1.0e-6
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def wrap(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def gauge_seed(shell: int) -> float:
    return wrap(TAU * ((5 * shell + 3) % P) / P + 0.137)


def decode(row: np.ndarray) -> complex:
    gauge, left, right = (float(item) for item in row)
    return (
        EPSILON * complex(math.cos(gauge), math.sin(gauge))
        + 0.5
        * RESIDUAL_WEIGHT
        * (
            complex(math.cos(left), math.sin(left))
            + complex(math.cos(right), math.sin(right))
        )
    )


def encode(value: complex, gauge: float) -> tuple[float, float, float]:
    if abs(value) > SUPPORTED_MAGNITUDE + RADIUS_SLACK:
        fail("independent chart left the declared base envelope")
    residual = (
        value - EPSILON * complex(math.cos(gauge), math.sin(gauge))
    ) / RESIDUAL_WEIGHT
    magnitude = abs(residual)
    if magnitude > 1.0 + RADIUS_SLACK:
        fail("independent residual chart left the unit disk")
    if magnitude <= ZERO_FLOOR:
        return wrap(gauge), math.pi / 2.0, -math.pi / 2.0
    phase = math.atan2(residual.imag, residual.real)
    delta = math.acos(min(1.0, max(0.0, magnitude)))
    return wrap(gauge), wrap(phase + delta), wrap(phase - delta)


def seed_angles(gauges: np.ndarray | None = None) -> np.ndarray:
    output = np.empty((P, 3), dtype=np.float64)
    for shell in range(P):
        gauge = gauge_seed(shell) if gauges is None else float(gauges[shell])
        output[shell] = encode(
            complex(m145.shell_scale(shell) / P, 0.0), gauge
        )
    return output


def decoded_state(angles: np.ndarray) -> np.ndarray:
    return np.asarray([decode(row) for row in angles], dtype=np.complex128)


def phasor_error(left: np.ndarray, right: np.ndarray) -> float:
    maximum = 0.0
    for observed, expected in zip(left.flat, right.flat, strict=True):
        maximum = max(
            maximum,
            math.hypot(
                math.cos(float(observed)) - math.cos(float(expected)),
                math.sin(float(observed)) - math.sin(float(expected)),
            ),
        )
    return maximum


def gauge_error(left: np.ndarray, right: np.ndarray) -> float:
    return phasor_error(np.asarray(left), np.asarray(right))


def offset(shell: int, index: int, family: str) -> float:
    code = {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]
    return TAU * ((5 * shell + 3 * index + code) % P) / P


def shear_angle(
    gauge: float,
    shell: int,
    index: int,
    family: str,
    strength: float = SHEAR_STRENGTH,
) -> float:
    return strength * math.sin(gauge + offset(shell, index, family))


def apply_phase(
    angles: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * m145.phase_exponent(shell, index, family) / P
        for slot in range(3):
            angles[shell, slot] = wrap(float(angles[shell, slot]) + delta)


def apply_shear(
    angles: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    strength: float = SHEAR_STRENGTH,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        gauge = float(angles[shell, 0])
        angle = sign * shear_angle(gauge, shell, index, family, strength)
        value = decode(angles[shell])
        value *= complex(math.cos(angle), math.sin(angle))
        angles[shell] = encode(value, gauge)


def apply_signs(
    angles: np.ndarray, plan: m145.GivensPlan
) -> None:
    for shell in range(P):
        if float(plan.diagonal_signs[shell]) < 0.0:
            for slot in range(3):
                angles[shell, slot] = wrap(
                    float(angles[shell, slot]) + math.pi
                )


def apply_pair(
    angles: np.ndarray,
    upper: int,
    lower: int,
    cosine: float,
    sine: float,
    *,
    transpose: bool,
) -> None:
    upper_value = decode(angles[upper])
    lower_value = decode(angles[lower])
    upper_gauge = float(angles[upper, 0])
    lower_gauge = float(angles[lower, 0])
    theta = math.atan2(sine, cosine)
    if transpose:
        next_upper = cosine * upper_value - sine * lower_value
        next_lower = sine * upper_value + cosine * lower_value
        next_upper_gauge = wrap(upper_gauge + theta)
        next_lower_gauge = wrap(lower_gauge - theta)
    else:
        next_upper = cosine * upper_value + sine * lower_value
        next_lower = -sine * upper_value + cosine * lower_value
        next_upper_gauge = wrap(upper_gauge - theta)
        next_lower_gauge = wrap(lower_gauge + theta)
    angles[upper] = encode(next_upper, next_upper_gauge)
    angles[lower] = encode(next_lower, next_lower_gauge)


def apply_givens(
    angles: np.ndarray,
    plan: m145.GivensPlan,
    *,
    inverse: bool = False,
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(m145.elimination_pairs()):
            apply_pair(
                angles,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=False,
            )
        apply_signs(angles, plan)
        return
    apply_signs(angles, plan)
    ordinal = 135
    for upper, lower in m145.reverse_elimination_pairs():
        apply_pair(
            angles,
            upper,
            lower,
            float(plan.cosine_sine[ordinal, 0]),
            float(plan.cosine_sine[ordinal, 1]),
            transpose=True,
        )
        ordinal -= 1
    if ordinal != -1:
        fail("independent phase carrier did not consume public plan")


def forward_angles(
    angles: np.ndarray,
    program: m145.Program,
    plan: m145.GivensPlan,
    *,
    order: str = "STANDARD",
    strength: float = SHEAR_STRENGTH,
) -> None:
    for index in range(program.depth):
        if order == "SHEAR_PHASE_GIVENS":
            apply_shear(angles, index, program.family, strength=strength)
            apply_phase(angles, index, program.family)
            apply_givens(angles, plan)
        elif order == "PHASE_GIVENS_SHEAR":
            apply_phase(angles, index, program.family)
            apply_givens(angles, plan)
            apply_shear(angles, index, program.family, strength=strength)
        elif order == "STANDARD":
            apply_phase(angles, index, program.family)
            apply_shear(angles, index, program.family, strength=strength)
            apply_givens(angles, plan)
        else:
            fail("unknown independent forward order")


def inverse_angles(
    angles: np.ndarray,
    program: m145.Program,
    plan: m145.GivensPlan,
    *,
    mode: str = "STANDARD",
) -> None:
    for index in range(program.depth - 1, -1, -1):
        if mode == "REORDERED":
            apply_shear(angles, index, program.family, inverse=True)
            apply_givens(angles, plan, inverse=True)
        else:
            apply_givens(angles, plan, inverse=True)
            if mode != "MISSING_SHEAR":
                strength = (
                    SHEAR_STRENGTH * 1.125
                    if mode == "WRONG_SHEAR"
                    else SHEAR_STRENGTH
                )
                apply_shear(
                    angles,
                    index,
                    program.family,
                    inverse=True,
                    strength=strength,
                )
        apply_phase(angles, index, program.family, inverse=True)


def boundary(state: np.ndarray, program: m145.Program) -> complex:
    value = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        angle = TAU * exponent / P
        value += (
            m145.shell_scale(shell)
            * complex(math.cos(angle), math.sin(angle))
            * state[shell]
        )
    return complex(value)


def scalar_phase(
    state: np.ndarray,
    gauges: np.ndarray,
    index: int,
    family: str,
) -> None:
    for shell in range(P):
        delta = TAU * m145.phase_exponent(shell, index, family) / P
        state[shell] *= complex(math.cos(delta), math.sin(delta))
        gauges[shell] = wrap(float(gauges[shell]) + delta)


def scalar_shear(
    state: np.ndarray,
    gauges: np.ndarray,
    index: int,
    family: str,
) -> None:
    for shell in range(P):
        angle = shear_angle(float(gauges[shell]), shell, index, family)
        state[shell] *= complex(math.cos(angle), math.sin(angle))


def scalar_signs(
    state: np.ndarray, gauges: np.ndarray, plan: m145.GivensPlan
) -> None:
    for shell in range(P):
        if float(plan.diagonal_signs[shell]) < 0.0:
            state[shell] = -state[shell]
            gauges[shell] = wrap(float(gauges[shell]) + math.pi)


def scalar_givens(
    state: np.ndarray, gauges: np.ndarray, plan: m145.GivensPlan
) -> None:
    scalar_signs(state, gauges, plan)
    ordinal = 135
    for upper, lower in m145.reverse_elimination_pairs():
        cosine = float(plan.cosine_sine[ordinal, 0])
        sine = float(plan.cosine_sine[ordinal, 1])
        theta = math.atan2(sine, cosine)
        upper_value = complex(state[upper])
        lower_value = complex(state[lower])
        state[upper] = cosine * upper_value - sine * lower_value
        state[lower] = sine * upper_value + cosine * lower_value
        gauges[upper] = wrap(float(gauges[upper]) + theta)
        gauges[lower] = wrap(float(gauges[lower]) - theta)
        ordinal -= 1


def scalar_forward(
    program: m145.Program, plan: m145.GivensPlan
) -> tuple[complex, np.ndarray, np.ndarray]:
    state = np.array(m145.seed_state(), dtype=np.complex128, copy=True)
    gauges = np.asarray([gauge_seed(i) for i in range(P)], dtype=np.float64)
    for index in range(program.depth):
        scalar_phase(state, gauges, index, program.family)
        scalar_shear(state, gauges, index, program.family)
        scalar_givens(state, gauges, plan)
    return boundary(state, program), state, gauges


@dataclass
class ComparisonCounter:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            fail(message)


def independent_cases(
    package: dict[str, Any],
    plan: m145.GivensPlan,
    counter: ComparisonCounter,
) -> tuple[list[dict[str, Any]], float, float, float, float]:
    records: list[dict[str, Any]] = []
    state_maximum = 0.0
    gauge_maximum = 0.0
    boundary_maximum = 0.0
    restoration_maximum = 0.0
    package_cases = {
        (item["family"], item["depth"]): item for item in package["cases"]
    }
    for family in FAMILIES:
        for depth in DEPTHS:
            program = m145.compile_program(depth, family)
            initial = seed_angles()
            angles = np.array(initial, copy=True)
            forward_angles(angles, program, plan)
            state = decoded_state(angles)
            gauges = np.array(angles[:, 0], copy=True)
            observed_boundary = boundary(state, program)
            scalar_boundary, scalar_state, scalar_gauges = scalar_forward(
                program, plan
            )
            state_error = float(np.max(np.abs(state - scalar_state)))
            observed_gauge_error = gauge_error(gauges, scalar_gauges)
            observed_boundary_error = abs(observed_boundary - scalar_boundary)
            record = package_cases[(family, depth)]
            package_boundary = complex(*record["final_boundary"])
            counter.require(
                abs(observed_boundary - package_boundary) <= BOUNDARY_TOLERANCE,
                "independent boundary disagreed with package",
            )
            counter.require(
                state_error <= STATE_TOLERANCE,
                "independent phase state disagreed with scalar recurrence",
            )
            counter.require(
                observed_gauge_error <= GAUGE_TOLERANCE,
                "independent gauge state disagreed with scalar recurrence",
            )
            counter.require(
                observed_boundary_error <= BOUNDARY_TOLERANCE,
                "independent boundary disagreed with scalar recurrence",
            )
            inverse_angles(angles, program, plan)
            restoration_error = phasor_error(angles, initial)
            counter.require(
                restoration_error <= RESTORATION_TOLERANCE,
                "independent phase carrier did not restore",
            )
            counter.require(
                record["same_backing"]
                and record["restoration_generation_before"] == 0
                and record["restoration_generation_after"] == 1
                and not record["snapshot_reload_used"]
                and record["inverse_history_cells"] == 0
                and record["retained_restoration_baseline_cells"] == 0,
                "package restoration custody fields changed",
            )
            state_maximum = max(state_maximum, state_error)
            gauge_maximum = max(gauge_maximum, observed_gauge_error)
            boundary_maximum = max(boundary_maximum, observed_boundary_error)
            restoration_maximum = max(restoration_maximum, restoration_error)
            records.append(
                {
                    "family": family,
                    "depth": depth,
                    "boundary_error_against_package": abs(
                        observed_boundary - package_boundary
                    ),
                    "state_error_against_scalar": state_error,
                    "gauge_error_against_scalar": observed_gauge_error,
                    "boundary_error_against_scalar": observed_boundary_error,
                    "restoration_error": restoration_error,
                }
            )
    return (
        records,
        state_maximum,
        gauge_maximum,
        boundary_maximum,
        restoration_maximum,
    )


def independent_witness(
    plan: m145.GivensPlan, counter: ComparisonCounter
) -> dict[str, Any]:
    program = m145.compile_program(4, "PRIMARY")
    gauges = (
        np.asarray([gauge_seed(i) for i in range(P)], dtype=np.float64),
        np.asarray(
            [wrap(gauge_seed(i) + 0.73) for i in range(P)],
            dtype=np.float64,
        ),
    )
    initial = [seed_angles(item) for item in gauges]
    initial_base_error = float(
        np.max(np.abs(decoded_state(initial[0]) - decoded_state(initial[1])))
    )
    final_states: list[np.ndarray] = []
    boundaries: list[complex] = []
    restoration_errors: list[float] = []
    for source in initial:
        angles = np.array(source, copy=True)
        forward_angles(angles, program, plan)
        state = decoded_state(angles)
        final_states.append(state)
        boundaries.append(boundary(state, program))
        inverse_angles(angles, program, plan)
        restoration_errors.append(phasor_error(angles, source))
    final_state_separation = float(
        np.max(np.abs(final_states[0] - final_states[1]))
    )
    boundary_separation = abs(boundaries[0] - boundaries[1])
    counter.require(
        initial_base_error <= STATE_TOLERANCE,
        "witness initial bases differ",
    )
    counter.require(
        gauge_error(gauges[0], gauges[1]) > CONTROL_FLOOR,
        "witness gauges did not differ",
    )
    counter.require(
        final_state_separation > CONTROL_FLOOR,
        "witness gauges did not change final state",
    )
    counter.require(
        boundary_separation > CONTROL_FLOOR,
        "witness gauges did not change final boundary",
    )
    counter.require(
        max(restoration_errors) <= RESTORATION_TOLERANCE,
        "witness carriers did not restore",
    )
    return {
        "initial_base_state_maximum_error": initial_base_error,
        "initial_gauge_phase_separation": gauge_error(gauges[0], gauges[1]),
        "final_base_state_maximum_separation": final_state_separation,
        "final_boundary_separation": boundary_separation,
        "restoration_errors": restoration_errors,
    }


def mutation_controls(
    plan: m145.GivensPlan, counter: ComparisonCounter
) -> dict[str, bool]:
    program = m145.compile_program(4, "ALTERNATE")

    def run_boundary(order: str, strength: float) -> complex:
        angles = seed_angles()
        forward_angles(
            angles, program, plan, order=order, strength=strength
        )
        return boundary(decoded_state(angles), program)

    standard = run_boundary("STANDARD", SHEAR_STRENGTH)
    results = {
        "zero_shear_mutation_detected": abs(
            standard - run_boundary("STANDARD", 0.0)
        )
        > CONTROL_FLOOR,
        "phase_shear_order_mutation_detected": abs(
            standard - run_boundary("SHEAR_PHASE_GIVENS", SHEAR_STRENGTH)
        )
        > CONTROL_FLOOR,
        "shear_givens_order_mutation_detected": abs(
            standard - run_boundary("PHASE_GIVENS_SHEAR", SHEAR_STRENGTH)
        )
        > CONTROL_FLOOR,
    }
    for mode in ("MISSING_SHEAR", "WRONG_SHEAR", "REORDERED"):
        initial = seed_angles()
        angles = np.array(initial, copy=True)
        forward_angles(angles, program, plan)
        inverse_angles(angles, program, plan, mode=mode)
        results[f"{mode.lower()}_detected"] = (
            phasor_error(angles, initial) > CONTROL_FLOOR
        )
    for name, passed in results.items():
        counter.require(passed, f"independent mutation control failed: {name}")
    return results


def independent_reuse(
    plan: m145.GivensPlan, counter: ComparisonCounter
) -> dict[str, Any]:
    original = seed_angles()
    angles = np.array(original, copy=True)
    first = m145.compile_program(37, "PRIMARY")
    forward_angles(angles, first, plan)
    first_boundary = boundary(decoded_state(angles), first)
    inverse_angles(angles, first, plan)
    first_restore = phasor_error(angles, original)
    second = m145.compile_program(1537, "REUSE")
    forward_angles(angles, second, plan)
    restored_boundary = boundary(decoded_state(angles), second)
    inverse_angles(angles, second, plan)
    second_restore = phasor_error(angles, original)
    fresh = seed_angles()
    forward_angles(fresh, second, plan)
    fresh_boundary = boundary(decoded_state(fresh), second)
    reuse_error = abs(restored_boundary - fresh_boundary)
    counter.require(
        max(first_restore, second_restore) <= RESTORATION_TOLERANCE,
        "independent unrelated carrier did not restore",
    )
    counter.require(
        reuse_error <= BOUNDARY_TOLERANCE,
        "independent restored/fresh reuse disagreed",
    )
    return {
        "first_boundary": [first_boundary.real, first_boundary.imag],
        "first_restoration_error": first_restore,
        "second_restoration_error": second_restore,
        "fresh_restored_boundary_error": reuse_error,
    }


def repeated_reuse(
    plan: m145.GivensPlan, counter: ComparisonCounter
) -> dict[str, Any]:
    initial = seed_angles()
    angles = np.array(initial, copy=True)
    program = m145.compile_program(64, "ALTERNATE")
    maximum = 0.0
    for _ in range(100):
        forward_angles(angles, program, plan)
        boundary(decoded_state(angles), program)
        inverse_angles(angles, program, plan)
        maximum = max(maximum, phasor_error(angles, initial))
    counter.require(
        maximum <= RESTORATION_TOLERANCE,
        "independent repeated carrier exceeded restoration tolerance",
    )
    return {"cycles": 100, "depth_per_cycle": 64, "maximum_error": maximum}


def run(
    package_path: Path,
    production_path: Path,
    predecessor_path: Path,
) -> dict[str, Any]:
    package = json.loads(package_path.read_text())
    if package.get("schema") != "CAT_CAS_F17_FIBER_ACTIVE_GAUGE_SHEAR_RESULT_V1":
        fail("unexpected fiber-active package schema")
    counter = ComparisonCounter()
    plan = m145.GivensPlan.compile()
    (
        cases,
        maximum_state,
        maximum_gauge,
        maximum_boundary,
        maximum_restoration,
    ) = independent_cases(package, plan, counter)
    witness = independent_witness(plan, counter)
    mutations = mutation_controls(plan, counter)
    reuse = independent_reuse(plan, counter)
    repeated = repeated_reuse(plan, counter)

    counter.require(
        package["execution_scope"]["case_count"] == 21
        and package["execution_scope"][
            "all_cases_within_predeclared_tolerances"
        ],
        "package execution scope changed",
    )
    counter.require(
        package["carrier_law"][
            "equal_base_different_gauge_boundary_distinguishable"
        ]
        and package["carrier_law"]["phase_and_shear_modules_noncommute"]
        and package["carrier_law"][
            "shear_and_local_givens_modules_noncommute"
        ]
        and package["carrier_law"][
            "no_relation_table_or_assignment_expansion"
        ],
        "package carrier-law fields changed",
    )
    resource = package["resource_law"]
    counter.require(
        resource["resident_phase_angle_float64_cells"] == 51
        and resource["resident_phase_angle_bytes"] == 408
        and resource["retained_public_givens_plan_bytes"] == 2312
        and resource["inverse_history_cells"] == 0
        and resource["retained_restoration_baseline_cells"] == 0
        and resource[
            "maximum_named_warm_execution_live_bytes_including_program_json"
        ]
        == 3223,
        "package accepted resource law changed",
    )
    matched = package["matched_classical_recurrence"]
    counter.require(
        matched["executed_in_every_case"]
        and matched["resident_float64_scalar_equivalent_cells"] == 51
        and matched["resident_bytes"] == 408
        and matched[
            "maximum_named_warm_execution_live_bytes_including_program_json"
        ]
        == 3191
        and not matched["comparison_establishes_distinct_phase_resource"]
        and not matched["comparison_establishes_computational_advantage"]
        and not matched["optimal_compact_classical_recurrence_claimed"],
        "package matched classical law changed",
    )
    counter.require(
        package["restoration"]["class"]
        == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
        and package["restoration"]["transient_buffers"]
        == "NO_RESTORATION_CLAIM"
        and package["restoration"]["same_backing"]
        and not package["restoration"]["snapshot_reload_used"]
        and not package["restoration"][
            "post_inverse_state_reset_or_canonical_reload_used"
        ],
        "package restoration classification changed",
    )
    required_not_established = {
        "OPTIMAL_CLASSICAL_BASELINE",
        "CATVM_MACHINE_ENFORCED_CUSTODY",
        "DISTINCT_PHASE_RESOURCE",
        "COMPUTATIONAL_ADVANTAGE",
        "SMALL_WALL_CROSSING",
        "PHYSICAL_WAVEFORM_EXECUTION",
        "PHYSICAL_BIT_REPLACEMENT",
        "UNBOUNDED_CATALYTIC_COMPUTATION",
    }
    counter.require(
        required_not_established.issubset(
            set(package["claim_boundary"]["not_established"])
        ),
        "package strict claim ceiling changed",
    )

    return {
        "schema": "CAT_CAS_F17_FIBER_ACTIVE_GAUGE_SHEAR_ORACLE_V1",
        "result": "PASS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "production_sha256": hashlib.sha256(
            production_path.read_bytes()
        ).hexdigest(),
        "predecessor_sha256": hashlib.sha256(
            predecessor_path.read_bytes()
        ).hexdigest(),
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "shares_only_established_m145_public_program_and_plan_compiler": True,
            "separate_weighted_three_phasor_chart": True,
            "separate51_angle_forward_inverse": True,
            "separate_gauge_shear_and_transport": True,
            "separate_equal_base_different_gauge_witness": True,
            "separate17_complex_plus17_gauge_scalar_recurrence": True,
            "separate_unrelated_and100_cycle_reuse": True,
        },
        "case_checks": {
            "case_count": len(cases),
            "maximum_state_error_against_scalar": maximum_state,
            "maximum_gauge_error_against_scalar": maximum_gauge,
            "maximum_boundary_error_against_scalar": maximum_boundary,
            "maximum_restoration_error": maximum_restoration,
        },
        "equal_base_different_gauge_witness": witness,
        "mutation_controls": mutations,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "comparison_count": counter.count,
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "claim_ceiling": "21_DECLARED_FIBER_ACTIVE_GAUGE_SHEAR_CASES_THROUGH_DEPTH4096_PLUS_DECLARED_CAUSALITY_AND_REUSE_CONTROLS_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "BASE_ONLY17_COMPLEX_QUOTIENT_INSUFFICIENT_FOR_DECLARED_VARIABLE_GAUGE_CARRIER",
            "EQUAL_BASE_DIFFERENT_GAUGE_FINAL_BOUNDARY_SEPARATION",
            "PHASE_SHEAR_AND_LOCAL_GIVENS_ORDER_CONTROLS",
            "FIXED51_PHASE_ANGLE_CARRIER_WITH_HISTORY_FREE_RESTORATION_AND_REUSE",
            "EXECUTED_MATCHED51_SCALAR_CLASSICAL_RECURRENCE",
        ],
        "not_established": sorted(required_not_established),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.package, args.production, args.predecessor)
    payload = canonical_json(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
