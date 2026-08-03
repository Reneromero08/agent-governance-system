#!/usr/bin/env python3
"""Independent oracle for the bounded direct-angle triangular gauge coupling.

The oracle imports neither M148 production nor the M146 carrier.  It
independently reconstructs the weighted three-phasor seed chart, public
triangular schedule, 51-angle forward/inverse, boundary, causality witness,
mutation controls, reuse, and a second identical-angle recurrence.  It shares
only the previously established M145 public phase exponent and shell weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import f17_anisotropic_radial_local_polar_givens_coupling as m145


P = 17
TAU = 2.0 * math.pi
GAUGE_WEIGHT = 1.0 / 32.0
RESIDUAL_WEIGHT = 1.0 - GAUGE_WEIGHT
COUPLING = 1.0 / 32.0
OVERSTRONG = 5.0 / 32.0
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
RESTORATION_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_FLOOR = 1.0e-6


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def wrap(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def gauge_seed(shell: int) -> float:
    return wrap(TAU * ((5 * shell + 3) % P) / P + 0.137)


def encode(value: complex, gauge: float) -> list[float]:
    residual = (
        value
        - GAUGE_WEIGHT * complex(math.cos(gauge), math.sin(gauge))
    ) / RESIDUAL_WEIGHT
    magnitude = abs(residual)
    if magnitude > 1.0 + 2.0e-12:
        fail("independent seed chart left its unit disk")
    if magnitude <= 1.0e-14:
        left, right = math.pi / 2.0, -math.pi / 2.0
    else:
        phase = math.atan2(residual.imag, residual.real)
        spread = math.acos(min(1.0, max(0.0, magnitude)))
        left, right = wrap(phase + spread), wrap(phase - spread)
    return [wrap(gauge), left, right]


def seed(gauges: list[float] | None = None) -> list[list[float]]:
    return [
        encode(
            complex(m145.shell_scale(shell) / P, 0.0),
            gauge_seed(shell) if gauges is None else gauges[shell],
        )
        for shell in range(P)
    ]


def clone(angles: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in angles]


def decode(row: list[float]) -> complex:
    gauge, left, right = row
    return (
        GAUGE_WEIGHT * complex(math.cos(gauge), math.sin(gauge))
        + 0.5
        * RESIDUAL_WEIGHT
        * (
            complex(math.cos(left), math.sin(left))
            + complex(math.cos(right), math.sin(right))
        )
    )


def decoded_state(angles: list[list[float]]) -> list[complex]:
    return [decode(row) for row in angles]


def phasor_error(
    left: list[list[float]], right: list[list[float]]
) -> float:
    return max(
        math.hypot(math.cos(a) - math.cos(b), math.sin(a) - math.sin(b))
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def flat_bytes(angles: list[list[float]]) -> bytes:
    values = [value for row in angles for value in row]
    return struct.pack("=51d", *values)


def commitment(angles: list[list[float]]) -> str:
    return hashlib.sha256(flat_bytes(angles)).hexdigest()


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    quadratic: int
    linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_DIRECT_ANGLE_GAUGE_COUPLING_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "topology": "PUBLIC_ROTATING_HUB16_OUT16_IN_TRIANGULAR_SCHEDULE",
            "observation": [self.quadratic, self.linear],
        }


def compile_program(depth: int, family: str) -> Program:
    if depth < 1 or depth > 4096 or family not in FAMILIES:
        fail("independent program outside sealed scope")
    return Program(
        depth,
        family,
        (7 * depth + 3 * len(family) + 1) % P or 1,
        (11 * depth + len(family) + 5) % P,
    )


def family_code(family: str) -> int:
    return {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]


def hub(index: int, family: str, mutation: int = 0) -> int:
    return (3 * index + family_code(family) + mutation) % P


def targets(center: int) -> Iterator[int]:
    for displacement in range(1, P):
        yield (center + displacement) % P


def public_offset(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
) -> float:
    exponent = (
        5 * controller
        + 7 * target
        + 3 * index
        + 2 * layer
        + family_code(family)
    ) % P
    return TAU * exponent / P


def rotate(row: list[float], delta: float) -> None:
    row[0] = wrap(row[0] + delta)
    row[1] = wrap(row[1] + delta)
    row[2] = wrap(row[2] + delta)


def edge(
    angles: list[list[float]],
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    strength: float = COUPLING,
) -> None:
    delta = strength * math.sin(
        angles[controller][0]
        + public_offset(controller, target, index, family, layer)
    )
    rotate(angles[target], -delta if inverse else delta)


def phase(
    angles: list[list[float]],
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        rotate(
            angles[shell],
            sign * TAU * m145.phase_exponent(shell, index, family) / P,
        )


def layer(
    angles: list[list[float]],
    index: int,
    family: str,
    layer_number: int,
    *,
    inverse: bool = False,
    strength: float = COUPLING,
    hub_mutation: int = 0,
) -> None:
    center = hub(index, family, hub_mutation)
    actors = list(targets(center))
    if inverse:
        actors.reverse()
    for actor in actors:
        controller, target = (
            (center, actor) if layer_number == 0 else (actor, center)
        )
        edge(
            angles,
            controller,
            target,
            index,
            family,
            layer_number,
            inverse=inverse,
            strength=strength,
        )


def evolve(
    angles: list[list[float]],
    program: Program,
    *,
    strength: float = COUPLING,
    order: str = "OUT_IN",
    hub_mutation: int = 0,
) -> None:
    layers = (0, 1) if order == "OUT_IN" else (1, 0)
    for index in range(program.depth):
        phase(angles, index, program.family)
        for layer_number in layers:
            layer(
                angles,
                index,
                program.family,
                layer_number,
                strength=strength,
                hub_mutation=hub_mutation,
            )


def reverse(
    angles: list[list[float]],
    program: Program,
    *,
    strength: float = COUPLING,
    mode: str = "VALID",
) -> None:
    for index in range(program.depth - 1, -1, -1):
        inverse_layers = (1, 0) if mode != "REORDERED" else (0, 1)
        for layer_number in inverse_layers:
            if mode == "MISSING_OUT" and layer_number == 0:
                continue
            selected_strength = (
                strength * 1.125
                if mode == "WRONG_OUT" and layer_number == 0
                else strength
            )
            layer(
                angles,
                index,
                program.family,
                layer_number,
                inverse=True,
                strength=selected_strength,
            )
        phase(angles, index, program.family, inverse=True)


def boundary(angles: list[list[float]], program: Program) -> complex:
    answer = 0.0j
    for shell, row in enumerate(angles):
        exponent = (program.quadratic * shell * shell + program.linear * shell) % P
        observation = TAU * exponent / P
        answer += (
            m145.shell_scale(shell)
            * complex(math.cos(observation), math.sin(observation))
            * decode(row)
        )
    return answer


def reference_forward(program: Program) -> np.ndarray:
    """Second recurrence using a flat ndarray and separately coded loops."""
    values = np.asarray(seed(), dtype=np.float64)
    for index in range(program.depth):
        for shell in range(P):
            delta = TAU * m145.phase_exponent(shell, index, program.family) / P
            values[shell, :] = [wrap(float(item) + delta) for item in values[shell]]
        center = (3 * index + family_code(program.family)) % P
        for actor in [(center + offset) % P for offset in range(1, P)]:
            delta = COUPLING * math.sin(
                float(values[center, 0])
                + public_offset(center, actor, index, program.family, 0)
            )
            values[actor, :] = [
                wrap(float(item) + delta) for item in values[actor]
            ]
        for actor in [(center + offset) % P for offset in range(1, P)]:
            delta = COUPLING * math.sin(
                float(values[actor, 0])
                + public_offset(actor, center, index, program.family, 1)
            )
            values[center, :] = [
                wrap(float(item) + delta) for item in values[center]
            ]
    return values


@dataclass
class Carrier:
    angles: list[list[float]]
    stage: str = "RESTORED"
    owner: str | None = None
    generation: int = 0

    @classmethod
    def create(cls) -> "Carrier":
        return cls(seed())


def transact(carrier: Carrier, program: Program) -> tuple[complex, float]:
    if carrier.stage != "RESTORED" or carrier.owner is not None:
        fail("independent carrier ownership changed")
    owner = digest_json(program.descriptor())
    carrier.owner = owner
    carrier.stage = "FORWARD"
    evolve(carrier.angles, program)
    carrier.stage = "FINAL_RESIDENT"
    answer = boundary(carrier.angles, program)
    carrier.stage = "INVERSE"
    reverse(carrier.angles, program)
    error = phasor_error(carrier.angles, seed())
    if error > RESTORATION_TOLERANCE:
        fail("independent transaction did not restore")
    carrier.stage = "RESTORED"
    carrier.owner = None
    carrier.generation += 1
    return answer, error


@dataclass
class Counter:
    count: int = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            fail(message)


def compare_cases(package: dict[str, Any], counter: Counter) -> list[dict[str, Any]]:
    package_cases = {
        (case["depth"], case["family"]): case for case in package["cases"]
    }
    results: list[dict[str, Any]] = []
    for family in FAMILIES:
        for depth in DEPTHS:
            program = compile_program(depth, family)
            initial = seed()
            angles = clone(initial)
            evolve(angles, program)
            observed_boundary = boundary(angles, program)
            reference = reference_forward(program)
            reference_bytes_equal = flat_bytes(angles) == reference.tobytes()
            case = package_cases[(depth, family)]
            package_boundary = complex(*case["final_boundary"])
            boundary_error = abs(observed_boundary - package_boundary)
            counter.require(
                digest_json(program.descriptor()) == case["program_fingerprint"],
                "independent program fingerprint changed",
            )
            counter.require(
                commitment(angles) == case["final_state_commitment"],
                "independent final commitment changed",
            )
            counter.require(
                reference_bytes_equal,
                "separate identical-angle recurrence changed bytes",
            )
            counter.require(
                boundary_error <= BOUNDARY_TOLERANCE,
                "independent final boundary changed",
            )
            reverse(angles, program)
            restoration_error = phasor_error(angles, initial)
            counter.require(
                restoration_error <= RESTORATION_TOLERANCE,
                "independent restoration changed",
            )
            results.append(
                {
                    "depth": depth,
                    "family": family,
                    "program_fingerprint": digest_json(program.descriptor()),
                    "final_state_commitment": commitment(
                        [list(row) for row in reference]
                    ),
                    "package_boundary_error": boundary_error,
                    "reference_angle_bytes_equal": reference_bytes_equal,
                    "restoration_error": restoration_error,
                }
            )
    return results


def witness(counter: Counter) -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    first = seed()
    second = seed([wrap(gauge_seed(shell) + 0.73) for shell in range(P)])
    initial_base_error = max(
        abs(left - right)
        for left, right in zip(
            decoded_state(first), decoded_state(second), strict=True
        )
    )
    initial_phase_separation = phasor_error(first, second)
    first_initial, second_initial = clone(first), clone(second)
    evolve(first, program)
    evolve(second, program)
    final_state_separation = max(
        abs(left - right)
        for left, right in zip(
            decoded_state(first), decoded_state(second), strict=True
        )
    )
    boundary_separation = abs(boundary(first, program) - boundary(second, program))
    reverse(first, program)
    reverse(second, program)
    restoration_errors = (
        phasor_error(first, first_initial),
        phasor_error(second, second_initial),
    )
    counter.require(initial_base_error <= 2.0e-12, "witness bases differ")
    counter.require(initial_phase_separation > CONTROL_FLOOR, "witness gauges agree")
    counter.require(boundary_separation > CONTROL_FLOOR, "witness boundary did not separate")
    counter.require(max(restoration_errors) <= RESTORATION_TOLERANCE, "witness did not restore")
    return {
        "initial_base_state_maximum_error": initial_base_error,
        "initial_gauge_phase_separation": initial_phase_separation,
        "final_base_state_maximum_separation": final_state_separation,
        "final_boundary_separation": boundary_separation,
        "restoration_errors": list(restoration_errors),
    }


def mutation_controls(counter: Counter) -> dict[str, bool]:
    program = compile_program(4, "ALTERNATE")
    variants: dict[str, complex] = {}
    for name, strength, order, topology in (
        ("VALID", COUPLING, "OUT_IN", 0),
        ("ZERO", 0.0, "OUT_IN", 0),
        ("ORDER", COUPLING, "IN_OUT", 0),
        ("TOPOLOGY", COUPLING, "OUT_IN", 1),
    ):
        angles = seed()
        evolve(
            angles,
            program,
            strength=strength,
            order=order,
            hub_mutation=topology,
        )
        variants[name] = boundary(angles, program)
    reverse_results: dict[str, float] = {}
    for mode in ("MISSING_OUT", "WRONG_OUT", "REORDERED"):
        initial = seed()
        angles = clone(initial)
        evolve(angles, program)
        reverse(angles, program, mode=mode)
        reverse_results[mode] = phasor_error(angles, initial)
    controls = {
        "zero_coupling_changes_boundary": abs(variants["VALID"] - variants["ZERO"]) > CONTROL_FLOOR,
        "layer_order_changes_boundary": abs(variants["VALID"] - variants["ORDER"]) > CONTROL_FLOOR,
        "hub_topology_changes_boundary": abs(variants["VALID"] - variants["TOPOLOGY"]) > CONTROL_FLOOR,
        "missing_inverse_detected": reverse_results["MISSING_OUT"] > CONTROL_FLOOR,
        "wrong_inverse_detected": reverse_results["WRONG_OUT"] > CONTROL_FLOOR,
        "reordered_inverse_detected": reverse_results["REORDERED"] > CONTROL_FLOOR,
    }
    for name, passed in controls.items():
        counter.require(passed, f"independent mutation failed: {name}")
    return controls


def reuse_checks(counter: Counter) -> tuple[dict[str, Any], dict[str, Any]]:
    carrier = Carrier.create()
    identity = id(carrier.angles)
    transact(carrier, compile_program(37, "PRIMARY"))
    reused_boundary, reused_restoration = transact(
        carrier, compile_program(1537, "REUSE")
    )
    fresh_boundary, _ = transact(
        Carrier.create(), compile_program(1537, "REUSE")
    )
    fresh_error = abs(reused_boundary - fresh_boundary)
    counter.require(fresh_error <= BOUNDARY_TOLERANCE, "reuse boundary changed")
    counter.require(id(carrier.angles) == identity, "reuse backing changed")
    repeated = Carrier.create()
    repeated_identity = id(repeated.angles)
    maximum = 0.0
    for _ in range(100):
        _, error = transact(repeated, compile_program(64, "ALTERNATE"))
        maximum = max(maximum, error)
    counter.require(maximum <= RESTORATION_TOLERANCE, "repeated reuse drifted")
    counter.require(id(repeated.angles) == repeated_identity, "repeated backing changed")
    return (
        {
            "fresh_restored_boundary_error": fresh_error,
            "restoration_error": reused_restoration,
            "same_backing": id(carrier.angles) == identity,
            "restoration_generation": carrier.generation,
        },
        {
            "cycles": 100,
            "depth": 64,
            "maximum_error": maximum,
            "same_backing": id(repeated.angles) == repeated_identity,
            "restoration_generation": repeated.generation,
        },
    )


def conditioning_check(counter: Counter) -> dict[str, Any]:
    program = compile_program(4096, "PRIMARY")
    errors: dict[str, float] = {}
    for name, strength in (("accepted", COUPLING), ("overstrong", OVERSTRONG)):
        initial = seed()
        angles = clone(initial)
        evolve(angles, program, strength=strength)
        reverse(angles, program, strength=strength)
        errors[name] = phasor_error(angles, initial)
    counter.require(errors["accepted"] <= RESTORATION_TOLERANCE, "accepted coupling drifted")
    counter.require(errors["overstrong"] > RESTORATION_TOLERANCE, "overstrong coupling unexpectedly restored")
    return {
        "depth": 4096,
        "accepted_strength": COUPLING,
        "accepted_restoration_error": errors["accepted"],
        "rejected_overstrong_strength": OVERSTRONG,
        "rejected_overstrong_restoration_error": errors["overstrong"],
    }


def run(
    package_path: Path,
    production_path: Path,
    predecessor_path: Path,
) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    counter = Counter()
    counter.require(
        package["schema"]
        == "CAT_CAS_F17_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_RESULT_V1",
        "package schema changed",
    )
    counter.require(
        package["source_sha256"]
        == hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "production source hash changed",
    )
    cases = compare_cases(package, counter)
    witness_result = witness(counter)
    mutations = mutation_controls(counter)
    reuse, repeated = reuse_checks(counter)
    conditioning = conditioning_check(counter)
    counter.require(all(package["controls"].values()), "package control changed")
    counter.require(
        package["execution_scope"]["case_count"] == 21
        and package["execution_scope"]["all_cases_within_predeclared_tolerances"],
        "package execution scope changed",
    )
    resource = package["resource_law"]
    counter.require(
        resource["resident_phase_angle_float64_cells"] == 51
        and resource["resident_phase_angle_bytes"] == 408
        and resource["retained_public_plan_bytes"] == 0
        and resource["maximum_named_update_float64_scratch_cells"] == 4
        and resource["inverse_history_cells"] == 0
        and resource["retained_restoration_baseline_cells"] == 0
        and resource["maximum_named_warm_execution_live_bytes_including_program_json"] == 750,
        "package resource law changed",
    )
    matched = package["matched_classical_recurrence"]
    counter.require(
        matched["method"] == "IDENTICAL51_FLOAT64_ANGLE_TRIANGULAR_RECURRENCE"
        and matched["executed_in_every_case"]
        and matched["final_angle_bytes_identical_in_every_case"]
        and matched["resident_float64_cells"] == 51
        and matched["maximum_named_warm_execution_live_bytes_including_program_json"] == 750
        and not matched["comparison_establishes_distinct_phase_resource"]
        and not matched["comparison_establishes_computational_advantage"]
        and not matched["optimal_compact_classical_recurrence_claimed"],
        "package matched recurrence changed",
    )
    counter.require(
        package["restoration"]["class"] == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
        and package["restoration"]["transient_buffers"] == "NO_RESTORATION_CLAIM"
        and package["restoration"]["same_backing"]
        and not package["restoration"]["snapshot_reload_used"]
        and not package["restoration"]["post_inverse_state_reset_or_canonical_reload_used"],
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
        "package claim ceiling changed",
    )
    return {
        "schema": "CAT_CAS_F17_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_ORACLE_V1",
        "result": "PASS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "production_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "predecessor_sha256": hashlib.sha256(predecessor_path.read_bytes()).hexdigest(),
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "shares_only_established_m145_phase_exponent_and_shell_weights": True,
            "separate_weighted_three_phasor_seed_chart": True,
            "separate51_angle_forward_inverse": True,
            "separate_triangular_gauge_coupling": True,
            "separate_equal_base_different_gauge_witness": True,
            "separate_identical51_angle_recurrence": True,
            "separate_unrelated_and100_cycle_reuse": True,
            "separate_inverse_conditioning_attack": True,
        },
        "case_checks": {
            "case_count": len(cases),
            "maximum_package_boundary_error": max(
                case["package_boundary_error"] for case in cases
            ),
            "maximum_restoration_error": max(
                case["restoration_error"] for case in cases
            ),
            "all_reference_angle_bytes_equal": all(
                case["reference_angle_bytes_equal"] for case in cases
            ),
        },
        "equal_base_different_gauge_witness": witness_result,
        "mutation_controls": mutations,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "inverse_conditioning_control": conditioning,
        "comparison_count": counter.count,
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "claim_ceiling": "21_DECLARED_DIRECT_ANGLE_TRIANGULAR_GAUGE_CASES_THROUGH_DEPTH4096_PLUS_DECLARED_CAUSALITY_CONDITIONING_AND_REUSE_CONTROLS_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "DIRECT_PHASE_ANGLE_SHARED_GAUGE_COUPLING_WITHOUT_COMPLEX_DECODE_OR_RECHART",
            "EQUAL_BASE_DIFFERENT_GAUGE_FINAL_BOUNDARY_SEPARATION",
            "FIXED51_PHASE_ANGLE_CARRIER_WITH_HISTORY_FREE_RESTORATION_AND_REUSE",
            "EXECUTED_IDENTICAL51_ANGLE_CLASSICAL_RECURRENCE",
            "OVERSTRONG_COUPLING_NUMERICAL_RESTORATION_FAILURE",
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
