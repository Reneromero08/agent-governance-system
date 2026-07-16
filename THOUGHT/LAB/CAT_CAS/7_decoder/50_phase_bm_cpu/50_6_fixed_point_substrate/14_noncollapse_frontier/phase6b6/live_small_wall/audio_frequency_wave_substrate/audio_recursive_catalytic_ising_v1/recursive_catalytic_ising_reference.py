#!/usr/bin/env python3
"""Bounded R3 candidate for a recursive phase-native Ising sector.

Each site is a complete established recursive phase tree with one continuous global
orientation on S^1. Native evolution uses continuous phase-difference, field, and
second-harmonic locking terms. No spin, sign, Ising energy, winner, or exact oracle enters
the native recurrence. The antipodal 0/pi projection occurs only at the declared collapse
boundary after evolution.

This ordinary-software candidate establishes no physical oscillator, silicon-phononic
computation, optimization advantage, hardware bit replacement, or Wall crossing.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
R0_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_phase_tree_v1"
    / "recursive_phase_tree_reference.py"
)
_spec = importlib.util.spec_from_file_location("catcas_recursive_phase_tree_r0", R0_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load established R0 recursive phase-tree reference")
r0 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r0
_spec.loader.exec_module(r0)

GENERATOR_ID = "recursive_catalytic_ising_reference_v1"
RESULT_SCHEMA = "recursive_catalytic_ising_candidate_result_v1"
CLAIM_CEILING = "SOFTWARE_RECURSIVE_PHASE_ISING_EMULATOR_ONLY"
ESTABLISHED_TOKEN = "AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED"

SITE_COUNT = 5
STEP_COUNT = 1000
TIME_STEP = 0.03
LOCK_START = 0.0
LOCK_FINAL = 1.2
LOCK_RESIDUAL_MAX = 1e-8
NO_LOCK_RESIDUAL_MIN = 1e-3
ORIENTATION_TOL = 1e-12
ENERGY_TOL = 1e-12
OPTIMUM_GAP_MIN = 1.0

INITIAL_ORIENTATIONS = np.array([0.31, 1.27, -2.11, 2.53, -0.83], dtype=np.float64)
COUPLING_MATRIX = np.array(
    [
        [0.0, 0.0, 1.0, 2.0, -2.0],
        [0.0, 0.0, 2.0, -1.0, -1.0],
        [1.0, 2.0, 0.0, 2.0, -1.0],
        [2.0, -1.0, 2.0, 0.0, -2.0],
        [-2.0, -1.0, -1.0, -2.0, 0.0],
    ],
    dtype=np.float64,
)
FIELD_VECTOR = np.array([-0.5, 0.0, 0.5, -1.0, -0.5], dtype=np.float64)


def _metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def wrap_phase(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    wrapped = (array + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if np.ndim(value) == 0 else wrapped


def phase_lock_residual(phases: np.ndarray) -> float:
    phases = np.asarray(phases, dtype=np.float64)
    zero_distance = np.abs(wrap_phase(phases))
    pi_distance = np.abs(wrap_phase(phases - math.pi))
    return float(np.max(np.minimum(zero_distance, pi_distance)))


def prefix_tree(tree: Any, prefix: str) -> Any:
    """Preserve recursive geometry while making site identities collision-free."""
    document = copy.deepcopy(tree.document())
    mapping = {
        node["node_id"]: f"{prefix}.{node['node_id']}" for node in document["nodes"]
    }
    for node in document["nodes"]:
        node["node_id"] = mapping[node["node_id"]]
    for edge in document["edges"]:
        edge["parent_id"] = mapping[edge["parent_id"]]
        edge["child_id"] = mapping[edge["child_id"]]
    document["root_id"] = mapping[document["root_id"]]
    document["global_spin_phase_rad"] = 0.0
    return r0.beam_from_document(document)


def geometry_signature(tree: Any) -> str:
    document = tree.document()
    local = lambda value: value.split(".", 1)[-1]
    payload = {
        "nodes": sorted(
            [
                {
                    "node_id": local(node["node_id"]),
                    "frequency_hz": node["frequency_hz"],
                    "phase_rad": node["phase_rad"],
                }
                for node in document["nodes"]
            ],
            key=lambda item: item["node_id"],
        ),
        "edges": sorted(
            [
                {
                    "parent_id": local(edge["parent_id"]),
                    "child_id": local(edge["child_id"]),
                    "modulation_index": edge["modulation_index"],
                }
                for edge in document["edges"]
            ],
            key=lambda item: (
                item["parent_id"], item["child_id"], item["modulation_index"]
            ),
        ),
        "root_id": local(document["root_id"]),
    }
    return hashlib.sha256(r0.canonical_json_bytes(payload, pretty=False)).hexdigest()


@dataclass(frozen=True)
class OrientedRecursiveBeam:
    site_id: str
    tree: Any
    orientation_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise ValueError("site_id must be a nonempty string")
        if not isinstance(self.tree, r0.RecursivePhaseBeam):
            raise ValueError("tree must be a complete RecursivePhaseBeam")
        r0.deserialize_tree_bytes(self.tree.canonical_bytes(), require_canonical=True)
        if self.tree.global_spin_phase_rad != 0.0:
            raise ValueError("internal R0 global orientation must remain zero in R3")
        object.__setattr__(
            self,
            "orientation_rad",
            float(wrap_phase(_finite(self.orientation_rad, "orientation_rad"))),
        )

    def with_orientation(self, value: float) -> "OrientedRecursiveBeam":
        return OrientedRecursiveBeam(self.site_id, self.tree, value)

    def render(self, t: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.orientation_rad) * self.tree.render(t)


def initial_state() -> tuple[OrientedRecursiveBeam, ...]:
    template = r0.hierarchy_a()
    return tuple(
        OrientedRecursiveBeam(
            f"site{index}",
            prefix_tree(template, f"site{index}"),
            float(INITIAL_ORIENTATIONS[index]),
        )
        for index in range(SITE_COUNT)
    )


def validate_problem(
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> tuple[np.ndarray, np.ndarray]:
    coupling = np.asarray(coupling, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if coupling.shape != (SITE_COUNT, SITE_COUNT) or field.shape != (SITE_COUNT,):
        raise ValueError("frozen Ising problem has the wrong shape")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(field)):
        raise ValueError("problem coefficients must be finite")
    if not np.array_equal(coupling, coupling.T):
        raise ValueError("coupling matrix must be exactly symmetric")
    if not np.array_equal(np.diag(coupling), np.zeros(SITE_COUNT)):
        raise ValueError("coupling diagonal must be exactly zero")
    return coupling, field


def lock_strength(step_index: int, lock_final: float = LOCK_FINAL) -> float:
    if isinstance(step_index, bool) or not 0 <= step_index < STEP_COUNT:
        raise ValueError("step index is outside the frozen schedule")
    final_value = _finite(lock_final, "lock_final")
    if final_value < 0.0:
        raise ValueError("lock_final must be nonnegative")
    alpha = step_index / (STEP_COUNT - 1)
    return LOCK_START + (final_value - LOCK_START) * alpha


def native_phase_velocity(
    state: Sequence[OrientedRecursiveBeam],
    lock_value: float,
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> np.ndarray:
    """Continuous S1 flow. No binary projection or oracle is reachable here."""
    coupling, field = validate_problem(coupling, field)
    if len(state) != SITE_COUNT:
        raise ValueError("native state has the wrong site count")
    lock_value = _finite(lock_value, "lock_value")
    phases = np.array([site.orientation_rad for site in state], dtype=np.float64)
    velocity = np.zeros(SITE_COUNT, dtype=np.float64)
    for site in range(SITE_COUNT):
        relation = 0.0
        for neighbor in range(SITE_COUNT):
            if site != neighbor:
                relation += coupling[site, neighbor] * math.sin(
                    phases[neighbor] - phases[site]
                )
        velocity[site] = (
            relation
            - field[site] * math.sin(phases[site])
            - lock_value * math.sin(2.0 * phases[site])
        )
    return velocity


def native_phase_step(
    state: Sequence[OrientedRecursiveBeam],
    step_index: int,
    *,
    lock_final: float = LOCK_FINAL,
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> tuple[OrientedRecursiveBeam, ...]:
    velocity = native_phase_velocity(
        state, lock_strength(step_index, lock_final), coupling, field
    )
    return tuple(
        site.with_orientation(
            float(wrap_phase(site.orientation_rad + TIME_STEP * velocity[index]))
        )
        for index, site in enumerate(state)
    )


def evolve_phase_state(
    start: Sequence[OrientedRecursiveBeam],
    *,
    lock_final: float = LOCK_FINAL,
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> tuple[OrientedRecursiveBeam, ...]:
    """Carry complete recursive trees through bounded continuous phase evolution."""
    state = tuple(start)
    if len(state) != SITE_COUNT:
        raise ValueError("evolution requires the frozen site count")
    for step_index in range(STEP_COUNT):
        state = native_phase_step(
            state,
            step_index,
            lock_final=lock_final,
            coupling=coupling,
            field=field,
        )
    return state


def decode_spins(state: Sequence[OrientedRecursiveBeam]) -> tuple[int, ...]:
    """Explicit antipodal collapse boundary, forbidden inside native evolution."""
    return tuple(1 if math.cos(site.orientation_rad) >= 0.0 else -1 for site in state)


def ising_energy(
    spins: Sequence[int],
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> float:
    coupling, field = validate_problem(coupling, field)
    if len(spins) != SITE_COUNT or any(value not in (-1, 1) for value in spins):
        raise ValueError("energy requires an exact antipodal spin vector")
    vector = np.asarray(spins, dtype=np.float64)
    return float(-0.5 * vector @ coupling @ vector - field @ vector)


def exact_ising_oracle() -> list[tuple[float, tuple[int, ...]]]:
    """Adjudication-only enumeration over 2^5 final boundary shadows."""
    values = [
        (ising_energy(spins), tuple(int(value) for value in spins))
        for spins in itertools.product((-1, 1), repeat=SITE_COUNT)
    ]
    return sorted(values, key=lambda item: (item[0], item[1]))


def collapse_boundary(state: Sequence[OrientedRecursiveBeam]) -> dict[str, Any]:
    spins = decode_spins(state)
    oracle = exact_ising_oracle()
    optimum_energy, optimum_spins = oracle[0]
    second_energy, _ = oracle[1]
    return {
        "observed_energy": ising_energy(spins),
        "observed_spins": list(spins),
        "optimum_energy": optimum_energy,
        "optimum_gap": second_energy - optimum_energy,
        "optimum_spins": list(optimum_spins),
        "schema": "recursive_catalytic_ising_collapse_v1",
    }


def _calls(function: Any) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    observed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            observed.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            observed.add(node.func.attr)
    return observed


def assert_native_noncollapse() -> None:
    forbidden = {
        "collapse_boundary", "decode_spins", "exact_ising_oracle", "ising_energy",
        "sign", "argmin", "argmax"
    }
    observed = set().union(
        _calls(native_phase_velocity),
        _calls(native_phase_step),
        _calls(evolve_phase_state),
    )
    reached = forbidden & observed
    if reached:
        raise AssertionError(f"native call graph reaches collapsed operations: {reached}")


def run_reference_tests() -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append(
            {"id": test_id, "observed": observed, "status": "PASS" if passed else "FAIL"}
        )

    start = initial_state()
    start_bytes = [site.tree.canonical_bytes() for site in start]
    start_digests = [site.tree.digest() for site in start]
    start_phases = np.array([site.orientation_rad for site in start])
    node_ids = [node.node_id for site in start for node in site.tree.root.walk()]
    signatures = [geometry_signature(site.tree) for site in start]

    record("initial_state_is_not_binary", phase_lock_residual(start_phases) > 0.1,
           _metric(phase_lock_residual(start_phases)))
    record("site_tree_ids_are_disjoint", len(node_ids) == len(set(node_ids)),
           [site.tree.root.node_id for site in start])
    record("site_recursive_geometries_match", len(set(signatures)) == 1, signatures)

    first = native_phase_step(start, 0)
    first_phases = np.array([site.orientation_rad for site in first])
    record("first_native_step_remains_continuous",
           phase_lock_residual(first_phases) > 0.1 and not np.array_equal(first_phases, start_phases),
           [_metric(value) for value in first_phases])

    final_state = evolve_phase_state(start)
    final_phases = np.array([site.orientation_rad for site in final_state])
    residual = phase_lock_residual(final_phases)
    record("complete_recursive_trees_are_preserved",
           start_bytes == [site.tree.canonical_bytes() for site in final_state]
           and start_digests == [site.tree.digest() for site in final_state],
           start_digests)
    record("final_phase_state_enters_antipodal_sector", residual <= LOCK_RESIDUAL_MAX,
           _metric(residual))
    rerun_phases = np.array([site.orientation_rad for site in evolve_phase_state(initial_state())])
    record("phase_trajectory_is_deterministic", np.array_equal(final_phases, rerun_phases),
           [_metric(value) for value in final_phases])

    t = r0.sample_times()
    beam = final_state[0].render(t)
    rotated = final_state[0].with_orientation(final_state[0].orientation_rad + math.pi).render(t)
    pi_error = float(np.max(np.abs(rotated + beam)))
    amplitude_error = float(np.max(np.abs(np.abs(rotated) - np.abs(beam))))
    record("pi_rotates_the_complete_tree",
           pi_error <= ORIENTATION_TOL and amplitude_error <= ORIENTATION_TOL,
           {"negation_error": _metric(pi_error), "amplitude_error": _metric(amplitude_error)})

    no_lock = evolve_phase_state(initial_state(), lock_final=0.0)
    no_lock_residual = phase_lock_residual(np.array([site.orientation_rad for site in no_lock]))
    record("second_harmonic_lock_is_material", no_lock_residual >= NO_LOCK_RESIDUAL_MIN,
           _metric(no_lock_residual))

    collapsed = collapse_boundary(final_state)
    record("boundary_projection_matches_unique_optimum",
           collapsed["observed_spins"] == collapsed["optimum_spins"]
           and abs(collapsed["observed_energy"] - collapsed["optimum_energy"]) <= ENERGY_TOL
           and collapsed["optimum_gap"] >= OPTIMUM_GAP_MIN,
           collapsed)

    wrong_coupling = decode_spins(evolve_phase_state(initial_state(), coupling=-COUPLING_MATRIX))
    wrong_coupling_energy = ising_energy(wrong_coupling)
    record("wrong_coupling_does_not_solve_frozen_problem",
           list(wrong_coupling) != collapsed["optimum_spins"]
           and wrong_coupling_energy > collapsed["optimum_energy"] + ENERGY_TOL,
           {"spins": list(wrong_coupling), "energy": _metric(wrong_coupling_energy)})

    wrong_field = decode_spins(evolve_phase_state(initial_state(), field=-2.0 * FIELD_VECTOR))
    wrong_field_energy = ising_energy(wrong_field)
    record("wrong_field_does_not_solve_frozen_problem",
           list(wrong_field) != collapsed["optimum_spins"]
           and wrong_field_energy > collapsed["optimum_energy"] + ENERGY_TOL,
           {"spins": list(wrong_field), "energy": _metric(wrong_field_energy)})

    noncollapse_error = None
    try:
        assert_native_noncollapse()
    except AssertionError as exc:
        noncollapse_error = str(exc)
    record("native_call_graph_has_no_binary_or_oracle_feedback",
           noncollapse_error is None, noncollapse_error or "closed")
    record("collapse_boundary_occurs_after_evolution",
           "collapse_boundary" not in _calls(evolve_phase_state),
           "explicit post-evolution boundary")

    passed = sum(test["status"] == "PASS" for test in tests)
    failed = len(tests) - passed
    return {
        "claim_ceiling": CLAIM_CEILING,
        "decision": "SOURCE_CANDIDATE" if failed == 0 else "SOURCE_CANDIDATE_FAILED",
        "generator": GENERATOR_ID,
        "measurements": {
            "final_lock_residual": _metric(residual),
            "final_orientations_rad": [_metric(value) for value in final_phases],
            "final_spins": collapsed["observed_spins"],
            "frozen_optimum_energy": _metric(collapsed["optimum_energy"]),
            "frozen_optimum_gap": _metric(collapsed["optimum_gap"]),
            "no_lock_residual": _metric(no_lock_residual),
            "observed_energy": _metric(collapsed["observed_energy"]),
        },
        "parents": {
            "r0": "AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED",
            "r1": "AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED",
            "r2s": "AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED",
        },
        "result_schema": RESULT_SCHEMA,
        "test_summary": {"failed": failed, "passed": passed, "total": len(tests)},
        "tests": tests,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "emit-result"), nargs="?", default="self-test")
    arguments = parser.parse_args(argv)
    result = run_reference_tests()
    if arguments.command == "emit-result":
        sys.stdout.buffer.write(r0.canonical_json_bytes(result))
    else:
        summary = result["test_summary"]
        print(f"{summary['passed']} PASS / {summary['failed']} FAIL ({summary['total']} total)")
        for test in result["tests"]:
            print(f"{test['status']:4s} {test['id']}: {test['observed']}")
    return 0 if result["test_summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
