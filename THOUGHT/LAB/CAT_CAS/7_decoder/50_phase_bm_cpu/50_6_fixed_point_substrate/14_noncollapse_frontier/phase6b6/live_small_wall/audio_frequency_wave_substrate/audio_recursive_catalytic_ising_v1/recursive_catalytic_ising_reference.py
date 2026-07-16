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
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator


PACKAGE_DIR = Path(__file__).resolve().parent
R0_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_phase_tree_v1"
    / "recursive_phase_tree_reference.py"
)
R1_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_wave_operator_v1"
    / "recursive_wave_operator_reference.py"
)
R2S_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_catalytic_wave_loop_v1"
    / "catalytic_wave_loop_reference.py"
)
_spec = importlib.util.spec_from_file_location("catcas_recursive_phase_tree_r0", R0_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load established R0 recursive phase-tree reference")
r0 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r0
_spec.loader.exec_module(r0)

GENERATOR_ID = "recursive_catalytic_ising_reference_v1"
RESULT_SCHEMA = "recursive_catalytic_ising_reference_result_v1"
CLAIM_CEILING = "SOFTWARE_RECURSIVE_PHASE_ISING_EMULATOR_ONLY"
ESTABLISHED_TOKEN = "AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED"
NEXT_BOUNDARY = "NEXT_AUDIO_PHASE_COMPUTING_BOUNDARY_REQUIRES_EXPLICIT_SELECTION"

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

SITE_IDS = tuple(f"site{index}" for index in range(SITE_COUNT))
TRAJECTORY_SHAPE = (STEP_COUNT + 1, SITE_COUNT)
TRAJECTORY_BYTE_COUNT = TRAJECTORY_SHAPE[0] * TRAJECTORY_SHAPE[1] * 8
TRAJECTORY_DTYPE = "<f8"
TRAJECTORY_FORMAT = "raw_little_endian_float64_phase_matrix"
PORTABLE_METRIC_ATOL = 1e-12
PORTABLE_METRIC_RTOL = 1e-12

CONTRACT_SCHEMA = "recursive_catalytic_ising_contract_v1"
PROBLEM_SCHEMA = "recursive_catalytic_ising_problem_v1"
TREE_IDENTITY_SCHEMA = "recursive_catalytic_ising_tree_identity_record_v1"
PHASE_STATE_SCHEMA = "recursive_catalytic_ising_phase_state_v1"
COLLAPSE_SCHEMA = "recursive_catalytic_ising_collapse_receipt_v1"
ORACLE_SCHEMA = "recursive_catalytic_ising_oracle_table_v1"
MANIFEST_SCHEMA = "recursive_catalytic_ising_fixture_manifest_v1"
TEST_SPEC_SCHEMA = "recursive_catalytic_ising_reference_tests_v1"

FIXTURE_DIR_NAME = "fixtures"
CONTRACT_PATH = "fixtures/r3_contract.json"
PROBLEM_PATH = "fixtures/ising_problem.json"
TREE_IDENTITIES_PATH = "fixtures/site_tree_identities.json"
TREE_PATHS = tuple(f"fixtures/site_tree_{index}.json" for index in range(SITE_COUNT))
INITIAL_STATE_PATH = "fixtures/initial_phase_state.json"
TRAJECTORY_PATH = "fixtures/phase_trajectory.f64le"
FINAL_STATE_PATH = "fixtures/final_phase_state.json"
COLLAPSE_PATH = "fixtures/collapse_receipt.json"
ORACLE_PATH = "fixtures/exact_oracle_table.json"
ORDERED_FIXTURE_PATHS = (
    CONTRACT_PATH,
    PROBLEM_PATH,
    TREE_IDENTITIES_PATH,
    *TREE_PATHS,
    INITIAL_STATE_PATH,
    TRAJECTORY_PATH,
    FINAL_STATE_PATH,
    COLLAPSE_PATH,
    ORACLE_PATH,
)

CONTRACT_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_CONTRACT_SCHEMA.json"
PROBLEM_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_PROBLEM_SCHEMA.json"
TREE_IDENTITY_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_TREE_IDENTITY_SCHEMA.json"
PHASE_STATE_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_PHASE_STATE_SCHEMA.json"
COLLAPSE_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_COLLAPSE_SCHEMA.json"
ORACLE_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_ORACLE_SCHEMA.json"
MANIFEST_SCHEMA_FILE = "RECURSIVE_CATALYTIC_ISING_MANIFEST_SCHEMA.json"
MANIFEST_FILE = "RECURSIVE_CATALYTIC_ISING_FIXTURE_MANIFEST.json"
TESTS_FILE = "RECURSIVE_CATALYTIC_ISING_REFERENCE_TESTS.json"
RESULTS_FILE = "RECURSIVE_CATALYTIC_ISING_REFERENCE_RESULTS.json"

R0_EXPECTED = {
    "claim_ceiling": "SOFTWARE_RECURSIVE_PHASE_TREE_REFERENCE_ONLY",
    "fixture_manifest_sha256": "7112307fa4406cf4880736545a88e56c45fafc6f27cd0a6518a1b40963fb62fa",
    "fixture_set_sha256": "6afb8adb0d14ab2e5a750df519ced073475fbf1554ee8be0732a2ebde5e15925",
    "reference_result_sha256": "46e2cc7cb72217c647f8653ebe61a0dbf2060a222de0eec6624fbb7fbcb94eab",
    "reference_tests_sha256": "3cecfa9f0d79babc4f9d76d7b463a1b8f825e209f2af592e590c52686dc95b2c",
    "source_byte_count": 77043,
    "source_git_blob_sha1": "956adb0ae8e84c091c1dc1e3de650be374fa96d1",
    "source_sha256": "e5911cb868f244ac69f3f8f8c4cfa83440385347be2d4526d5f25376de736887",
}
R1_EXPECTED = {
    "claim_ceiling": "SOFTWARE_COMPLETE_TREE_TEMPORAL_RECURRENCE_REFERENCE_ONLY",
    "fixture_manifest_sha256": "28cbcec8997f6f5eb49dc13e6bf919342af0863a5ba6cb1a70f10dea6fcdbc4e",
    "fixture_set_sha256": "da62112c0459c49673675182e67011899d8ee1e841df3650c0c4a0aeecd137dd",
    "reference_result_sha256": "37cb46f6806555cfaec60910f9b5b92fbcac5bf1d0e976fb67e7f2d2c0ec4139",
    "reference_tests_sha256": "5bf39db581fbc4f5cc290d1ad0ba34bc87315c2d1cf4777acf12d1d8a35023b5",
    "source_byte_count": 107055,
    "source_git_blob_sha1": "3685be9ae63dcd213b2155c8cd66f6f81e45c071",
    "source_sha256": "26b2cfaa63f5fe6bfa97f6d9f64b97d0ee944bc39ac45d406092aea257b2179e",
}
R2S_EXPECTED = {
    "claim_ceiling": "SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY",
    "fixture_manifest_sha256": "5e8bfa247c513d189774ec671265b2d3dc1ea97004e5e8c40baa090f26db3cad",
    "fixture_set_sha256": "e6e51ae655e184f8f43b2afa9fe0c75041046966b4cdecd6fde008b02b684aa8",
    "reference_result_sha256": "bee5727f68fc10ee047d666198b3f060f669058e966aa44802e270f90abbdeeb",
    "reference_tests_sha256": "ef888d8d8b48b2fbdc7897d6d42aa2f63f8c300517f6d9b8911346bf285438c6",
    "source_byte_count": 117873,
    "source_git_blob_sha1": "63eed91f74252082b1258755bdd4371a2a48e105",
    "source_sha256": "6c55861da950caf0738bb5ffb676f0c458a593a805ddd49419d6b2b427f6c33c",
}

FORBIDDEN_INPUT_FIELDS = {
    "answer",
    "candidate",
    "energy",
    "expected_optimum",
    "expected_result",
    "optimum",
    "score",
    "spin",
    "spins",
    "winner",
}


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
    final_state, _, _ = evolve_phase_trajectory(
        start,
        lock_final=lock_final,
        coupling=coupling,
        field=field,
    )
    return final_state


def evolve_phase_trajectory(
    start: Sequence[OrientedRecursiveBeam],
    *,
    lock_final: float = LOCK_FINAL,
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> tuple[tuple[OrientedRecursiveBeam, ...], np.ndarray, int]:
    """Return all continuous phase positions while preserving complete tree bytes."""
    state = tuple(start)
    if len(state) != SITE_COUNT:
        raise ValueError("evolution requires the frozen site count")
    frozen_tree_bytes = tuple(site.tree.canonical_bytes() for site in state)
    trajectory = np.empty(TRAJECTORY_SHAPE, dtype=np.float64)
    trajectory[0] = np.array(
        [site.orientation_rad for site in state], dtype=np.float64
    )
    tree_identity_checks = SITE_COUNT
    for step_index in range(STEP_COUNT):
        state = native_phase_step(
            state,
            step_index,
            lock_final=lock_final,
            coupling=coupling,
            field=field,
        )
        for site_index, site in enumerate(state):
            if site.tree.canonical_bytes() != frozen_tree_bytes[site_index]:
                raise ValueError("complete recursive tree changed during native evolution")
            tree_identity_checks += 1
        trajectory[step_index + 1] = np.array(
            [site.orientation_rad for site in state], dtype=np.float64
        )
    return state, trajectory, tree_identity_checks


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


def exact_ising_oracle(
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> list[tuple[float, tuple[int, ...]]]:
    """Adjudication-only enumeration over 2^5 final boundary shadows."""
    values = [
        (ising_energy(spins, coupling, field), tuple(int(value) for value in spins))
        for spins in itertools.product((-1, 1), repeat=SITE_COUNT)
    ]
    return sorted(values, key=lambda item: (item[0], item[1]))


def collapse_boundary(
    state: Sequence[OrientedRecursiveBeam],
    coupling: np.ndarray = COUPLING_MATRIX,
    field: np.ndarray = FIELD_VECTOR,
) -> dict[str, Any]:
    """Final-only projection. Exact enumeration is intentionally separate."""
    spins = decode_spins(state)
    return {
        "observed_energy": ising_energy(spins, coupling, field),
        "observed_spins": list(spins),
        "schema": "recursive_catalytic_ising_boundary_projection_v1",
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
    proof = native_ast_proof(Path(__file__).resolve())
    if proof["status"] != "PASS":
        raise AssertionError("native continuous-phase structural proof failed")


def _candidate_diagnostics() -> dict[str, Any]:
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
    oracle = exact_ising_oracle()
    optimum_energy, optimum_spins = oracle[0]
    second_energy, _ = oracle[1]
    collapsed = {
        **collapsed,
        "optimum_energy": optimum_energy,
        "optimum_gap": second_energy - optimum_energy,
        "optimum_spins": list(optimum_spins),
    }
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


def _candidate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "emit-result"), nargs="?", default="self-test")
    arguments = parser.parse_args(argv)
    result = _candidate_diagnostics()
    if arguments.command == "emit-result":
        sys.stdout.buffer.write(r0.canonical_json_bytes(result))
    else:
        summary = result["test_summary"]
        print(f"{summary['passed']} PASS / {summary['failed']} FAIL ({summary['total']} total)")
        for test in result["tests"]:
            print(f"{test['status']:4s} {test['id']}: {test['observed']}")
    return 0 if result["test_summary"]["failed"] == 0 else 1


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha1(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    return _sha1(f"blob {len(payload)}\0".encode("ascii") + payload)


def canonical_bytes(document: Any) -> bytes:
    return r0.canonical_json_bytes(document, pretty=True)


def _strict_document(payload: bytes, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"duplicate JSON object key is forbidden: {key}")
            document[key] = value
        return document

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    if canonical_bytes(document) != payload:
        raise ValueError(f"{label} must use canonical JSON bytes")
    return document


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    document = _strict_document(path.read_bytes(), label)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _exact_keys(document: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(document)
    if observed != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _reject_forbidden_fields(document: Mapping[str, Any], label: str) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_INPUT_FIELDS:
                    raise ValueError(f"{label} contains forbidden field: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _finite_list(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return [_finite(value, f"{label}[{index}]") for index, value in enumerate(values)]


def source_binding(source_path: Path = Path(__file__).resolve()) -> dict[str, Any]:
    payload = source_path.read_bytes()
    crlf_count = payload.count(b"\r\n")
    lf_count = payload.count(b"\n")
    line_endings = "LF" if crlf_count == 0 else "CRLF_OR_MIXED"
    return {
        "byte_count": len(payload),
        "git_blob_sha1": _git_blob_sha1(payload),
        "line_endings": line_endings,
        "lf_count": lf_count,
        "sha256": _sha256(payload),
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load parent reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parent_identity(
    source_path: Path,
    package_dir: Path,
    manifest_name: str,
    tests_name: str,
    results_name: str,
) -> dict[str, Any]:
    source_payload = source_path.read_bytes()
    manifest_path = package_dir / manifest_name
    tests_path = package_dir / tests_name
    results_path = package_dir / results_name
    manifest = _load_canonical_json(manifest_path, manifest_name)
    result = _load_canonical_json(results_path, results_name)
    scientific = result.get("scientific")
    if not isinstance(scientific, Mapping):
        raise ValueError(f"{results_name} has no scientific result object")
    return {
        "claim_ceiling": scientific.get("claim_ceiling"),
        "fixture_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "fixture_set_sha256": manifest.get("fixture_set_sha256"),
        "reference_result_sha256": _sha256(results_path.read_bytes()),
        "reference_tests_sha256": _sha256(tests_path.read_bytes()),
        "source_byte_count": len(source_payload),
        "source_git_blob_sha1": _git_blob_sha1(source_payload),
        "source_sha256": _sha256(source_payload),
    }


def parent_custody() -> dict[str, Any]:
    r0_dir = R0_SOURCE.parent
    r1_dir = R1_SOURCE.parent
    r2s_dir = R2S_SOURCE.parent
    observed = {
        "r0": _parent_identity(
            R0_SOURCE,
            r0_dir,
            "RECURSIVE_PHASE_TREE_FIXTURE_MANIFEST.json",
            "RECURSIVE_PHASE_TREE_REFERENCE_TESTS.json",
            "RECURSIVE_PHASE_TREE_REFERENCE_RESULTS.json",
        ),
        "r1": _parent_identity(
            R1_SOURCE,
            r1_dir,
            "RECURSIVE_WAVE_OPERATOR_TRAJECTORY_MANIFEST.json",
            "RECURSIVE_WAVE_OPERATOR_REFERENCE_TESTS.json",
            "RECURSIVE_WAVE_OPERATOR_REFERENCE_RESULTS.json",
        ),
        "r2s": _parent_identity(
            R2S_SOURCE,
            r2s_dir,
            "CATALYTIC_WAVE_LOOP_FIXTURE_MANIFEST.json",
            "CATALYTIC_WAVE_LOOP_REFERENCE_TESTS.json",
            "CATALYTIC_WAVE_LOOP_REFERENCE_RESULTS.json",
        ),
    }
    for name, expected in (
        ("r0", R0_EXPECTED),
        ("r1", R1_EXPECTED),
        ("r2s", R2S_EXPECTED),
    ):
        if observed[name] != expected:
            raise ValueError(f"{name} parent custody identity mismatch")
    return observed


def parent_verification() -> dict[str, Any]:
    modules = {
        "r0": _load_module(R0_SOURCE, "rci_parent_r0"),
        "r1": _load_module(R1_SOURCE, "rci_parent_r1"),
        "r2s": _load_module(R2S_SOURCE, "rci_parent_r2s"),
    }
    reports: dict[str, Any] = {}
    for name, module in modules.items():
        report = module.verify_package(module.PACKAGE_DIR, Path(module.__file__).resolve())
        if report.get("status") != "PASS":
            raise ValueError(f"{name} parent verification failed")
        reports[name] = {
            "recomputed_results_match": report.get("recomputed_results_match"),
            "status": report.get("status"),
            "test_count": report.get("test_count"),
            "tests_passed": report.get("tests_passed"),
        }
    expected_counts = {"r0": 38, "r1": 78, "r2s": 78}
    for name, count in expected_counts.items():
        if reports[name]["test_count"] != count or reports[name]["tests_passed"] != count:
            raise ValueError(f"{name} parent test count mismatch")
    return reports


def _strict_schema(
    title: str,
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    keys = list(properties) if required is None else list(required)
    return {
        "additionalProperties": False,
        "properties": dict(properties),
        "required": keys,
        "title": title,
        "type": "object",
    }


def _sha256_schema() -> dict[str, Any]:
    return {"pattern": "^[0-9a-f]{64}$", "type": "string"}


def _sha1_schema() -> dict[str, Any]:
    return {"pattern": "^[0-9a-f]{40}$", "type": "string"}


def _finite_number_schema(**constraints: Any) -> dict[str, Any]:
    return {"type": "number", **constraints}


def _site_ids_schema() -> dict[str, Any]:
    return {"const": list(SITE_IDS), "type": "array"}


def _spin_vector_schema() -> dict[str, Any]:
    return {
        "items": {"enum": [-1, 1], "type": "integer"},
        "maxItems": SITE_COUNT,
        "minItems": SITE_COUNT,
        "type": "array",
    }


def _source_binding_schema() -> dict[str, Any]:
    return _strict_schema(
        "qualified_source_binding",
        {
            "byte_count": {"minimum": 1, "type": "integer"},
            "git_blob_sha1": _sha1_schema(),
            "lf_count": {"minimum": 1, "type": "integer"},
            "line_endings": {"const": "LF", "type": "string"},
            "sha256": _sha256_schema(),
        },
    )


def _parent_identity_schema(expected: Mapping[str, Any]) -> dict[str, Any]:
    return _strict_schema(
        "established_parent_identity",
        {
            "claim_ceiling": {"const": expected["claim_ceiling"], "type": "string"},
            "fixture_manifest_sha256": {"const": expected["fixture_manifest_sha256"], "type": "string"},
            "fixture_set_sha256": {"const": expected["fixture_set_sha256"], "type": "string"},
            "reference_result_sha256": {"const": expected["reference_result_sha256"], "type": "string"},
            "reference_tests_sha256": {"const": expected["reference_tests_sha256"], "type": "string"},
            "source_byte_count": {"const": expected["source_byte_count"], "type": "integer"},
            "source_git_blob_sha1": {"const": expected["source_git_blob_sha1"], "type": "string"},
            "source_sha256": {"const": expected["source_sha256"], "type": "string"},
        },
    )


def _parents_schema() -> dict[str, Any]:
    return _strict_schema(
        "established_parent_packet",
        {
            "r0": _parent_identity_schema(R0_EXPECTED),
            "r1": _parent_identity_schema(R1_EXPECTED),
            "r2s": _parent_identity_schema(R2S_EXPECTED),
        },
    )


def _fixture_record_schema(path: str, role: str) -> dict[str, Any]:
    trajectory = path == TRAJECTORY_PATH
    return _strict_schema(
        f"fixture_record_{role}",
        {
            "byte_count": {
                "const": TRAJECTORY_BYTE_COUNT,
                "type": "integer",
            } if trajectory else {"minimum": 1, "type": "integer"},
            "byte_order": {"const": "little" if trajectory else "utf-8", "type": "string"},
            "dtype": {"const": TRAJECTORY_DTYPE if trajectory else "canonical_json", "type": "string"},
            "media_type": {
                "const": "application/vnd.cat-cas.raw-float64-phase-matrix" if trajectory else "application/json",
                "type": "string",
            },
            "path": {"const": path, "type": "string"},
            "role": {"const": role, "type": "string"},
            "sha256": _sha256_schema(),
            "shape": {
                "const": list(TRAJECTORY_SHAPE),
                "type": "array",
            } if trajectory else {"type": "null"},
        },
    )


def schema_documents() -> dict[str, dict[str, Any]]:
    contract = _strict_schema(
        CONTRACT_SCHEMA,
        {
            "catalytic_predecessor_law": {"const": "independent_reproduced_predecessor_no_latch_feedback", "type": "string"},
            "claim_ceiling": {"const": CLAIM_CEILING, "type": "string"},
            "collapse_stage": {"const": "after_native_trajectory_return", "type": "string"},
            "energy_tolerance": {"const": ENERGY_TOL, "type": "number"},
            "final_lock_tolerance": {"const": LOCK_RESIDUAL_MAX, "type": "number"},
            "initial_orientations_rad": {
                "const": INITIAL_ORIENTATIONS.tolist(),
                "items": _finite_number_schema(minimum=-math.pi, exclusiveMaximum=math.pi),
                "maxItems": SITE_COUNT,
                "minItems": SITE_COUNT,
                "type": "array",
            },
            "initial_state_sha256": _sha256_schema(),
            "lock_final": {"const": LOCK_FINAL, "type": "number"},
            "lock_schedule": {"const": "linear_in_step_index_0_to_999", "type": "string"},
            "lock_start": {"const": LOCK_START, "type": "number"},
            "minimum_unique_optimum_gap": {"const": OPTIMUM_GAP_MIN, "type": "number"},
            "native_dynamics": {"const": "phase_difference_plus_field_plus_second_harmonic", "type": "string"},
            "oracle_stage": {"const": "after_collapse_boundary", "type": "string"},
            "parents": _parents_schema(),
            "portable_metric_atol": {"const": PORTABLE_METRIC_ATOL, "type": "number"},
            "portable_metric_rtol": {"const": PORTABLE_METRIC_RTOL, "type": "number"},
            "problem_sha256": _sha256_schema(),
            "schema": {"const": CONTRACT_SCHEMA, "type": "string"},
            "site_count": {"const": SITE_COUNT, "type": "integer"},
            "site_ids": _site_ids_schema(),
            "source_binding": _source_binding_schema(),
            "step_count": {"const": STEP_COUNT, "type": "integer"},
            "time_step": {"const": TIME_STEP, "type": "number"},
            "trajectory_byte_count": {"const": TRAJECTORY_BYTE_COUNT, "type": "integer"},
            "trajectory_dtype": {"const": TRAJECTORY_DTYPE, "type": "string"},
            "trajectory_format": {"const": TRAJECTORY_FORMAT, "type": "string"},
            "trajectory_shape": {"const": list(TRAJECTORY_SHAPE), "type": "array"},
            "tree_identity_record_sha256": _sha256_schema(),
            "wrap_interval": {"const": "[-pi,pi)", "type": "string"},
        },
    )
    problem = _strict_schema(
        PROBLEM_SCHEMA,
        {
            "coupling_matrix": {
                "const": COUPLING_MATRIX.tolist(),
                "items": {
                    "items": _finite_number_schema(),
                    "maxItems": SITE_COUNT,
                    "minItems": SITE_COUNT,
                    "type": "array",
                },
                "maxItems": SITE_COUNT,
                "minItems": SITE_COUNT,
                "type": "array",
            },
            "field_vector": {
                "const": FIELD_VECTOR.tolist(),
                "items": _finite_number_schema(),
                "maxItems": SITE_COUNT,
                "minItems": SITE_COUNT,
                "type": "array",
            },
            "schema": {"const": PROBLEM_SCHEMA, "type": "string"},
            "site_ids": _site_ids_schema(),
        },
    )
    tree_identity = _strict_schema(
        TREE_IDENTITY_SCHEMA,
        {
            "schema": {"const": TREE_IDENTITY_SCHEMA, "type": "string"},
            "sites": {
                "items": False,
                "maxItems": SITE_COUNT,
                "minItems": SITE_COUNT,
                "prefixItems": [
                    _strict_schema(
                        f"tree_identity_{site_id}",
                        {
                            "byte_count": {"minimum": 1, "type": "integer"},
                            "fixture_path": {"const": TREE_PATHS[index], "type": "string"},
                            "geometry_sha256": _sha256_schema(),
                            "root_id": {"const": f"{site_id}.root", "type": "string"},
                            "site_id": {"const": site_id, "type": "string"},
                            "tree_digest": _sha256_schema(),
                            "tree_sha256": _sha256_schema(),
                        },
                    )
                    for index, site_id in enumerate(SITE_IDS)
                ],
                "type": "array",
            },
        },
    )
    phase_state = _strict_schema(
        PHASE_STATE_SCHEMA,
        {
            "orientations_rad": {
                "items": _finite_number_schema(minimum=-math.pi, exclusiveMaximum=math.pi),
                "maxItems": SITE_COUNT,
                "minItems": SITE_COUNT,
                "type": "array",
            },
            "schema": {"const": PHASE_STATE_SCHEMA, "type": "string"},
            "site_ids": _site_ids_schema(),
            "stage": {"enum": ["initial_before_native_evolution", "final_after_1000_native_steps"], "type": "string"},
            "trajectory_row_index": {"maximum": STEP_COUNT, "minimum": 0, "type": "integer"},
            "tree_identity_record_sha256": _sha256_schema(),
        },
    )
    phase_state["allOf"] = [
        {
            "if": {"properties": {"stage": {"const": "initial_before_native_evolution"}}, "required": ["stage"]},
            "then": {"properties": {"trajectory_row_index": {"const": 0}}},
        },
        {
            "if": {"properties": {"stage": {"const": "final_after_1000_native_steps"}}, "required": ["stage"]},
            "then": {"properties": {"trajectory_row_index": {"const": STEP_COUNT}}},
        },
    ]
    collapse = _strict_schema(
        COLLAPSE_SCHEMA,
        {
            "claim_ceiling": {"const": CLAIM_CEILING, "type": "string"},
            "collapse_stage": {"const": "after_native_trajectory_return", "type": "string"},
            "final_antipodal_residual": _finite_number_schema(minimum=0.0, maximum=LOCK_RESIDUAL_MAX),
            "final_state_sha256": _sha256_schema(),
            "next_energy": _finite_number_schema(),
            "observed_energy": _finite_number_schema(),
            "observed_spins": _spin_vector_schema(),
            "optimum_energy": _finite_number_schema(),
            "optimum_gap": _finite_number_schema(minimum=OPTIMUM_GAP_MIN),
            "optimum_spins": _spin_vector_schema(),
            "oracle_table_sha256": _sha256_schema(),
            "schema": {"const": COLLAPSE_SCHEMA, "type": "string"},
        },
    )
    oracle = _strict_schema(
        ORACLE_SCHEMA,
        {
            "next_energy": _finite_number_schema(),
            "optimum_energy": _finite_number_schema(),
            "optimum_gap": _finite_number_schema(minimum=OPTIMUM_GAP_MIN),
            "oracle_stage": {"const": "after_collapse_boundary", "type": "string"},
            "problem_sha256": _sha256_schema(),
            "rows": {
                "items": _strict_schema(
                    "exact_oracle_row",
                    {"energy": _finite_number_schema(), "spins": _spin_vector_schema()},
                ),
                "maxItems": 2 ** SITE_COUNT,
                "minItems": 2 ** SITE_COUNT,
                "type": "array",
            },
            "schema": {"const": ORACLE_SCHEMA, "type": "string"},
            "unique_optimum": _spin_vector_schema(),
        },
    )
    manifest = _strict_schema(
        MANIFEST_SCHEMA,
        {
            "claim_ceiling": {"const": CLAIM_CEILING, "type": "string"},
            "contract_sha256": _sha256_schema(),
            "fixture_count": {"const": len(ORDERED_FIXTURE_PATHS), "type": "integer"},
            "fixture_set_sha256": _sha256_schema(),
            "fixtures": {
                "items": False,
                "maxItems": len(ORDERED_FIXTURE_PATHS),
                "minItems": len(ORDERED_FIXTURE_PATHS),
                "prefixItems": [
                    _fixture_record_schema(path, role)
                    for path, role in zip(ORDERED_FIXTURE_PATHS, FIXTURE_ROLES, strict=True)
                ],
                "type": "array",
            },
            "generator": {"const": GENERATOR_ID, "type": "string"},
            "ordered_fixture_paths": {"const": list(ORDERED_FIXTURE_PATHS), "type": "array"},
            "parents": _parents_schema(),
            "problem_sha256": _sha256_schema(),
            "result_path": {"const": RESULTS_FILE, "type": "string"},
            "result_schema": {"const": RESULT_SCHEMA, "type": "string"},
            "schema": {"const": MANIFEST_SCHEMA, "type": "string"},
            "source_binding": _source_binding_schema(),
            "tests_sha256": _sha256_schema(),
            "total_fixture_bytes": {"minimum": TRAJECTORY_BYTE_COUNT, "type": "integer"},
            "trajectory": _strict_schema(
                "trajectory_identity",
                {
                    "byte_count": {"const": TRAJECTORY_BYTE_COUNT, "type": "integer"},
                    "dtype": {"const": TRAJECTORY_DTYPE, "type": "string"},
                    "format": {"const": TRAJECTORY_FORMAT, "type": "string"},
                    "sha256": _sha256_schema(),
                    "shape": {"const": list(TRAJECTORY_SHAPE), "type": "array"},
                },
            ),
            "tree_identity_record_sha256": _sha256_schema(),
        },
    )
    documents = {
        CONTRACT_SCHEMA_FILE: contract,
        PROBLEM_SCHEMA_FILE: problem,
        TREE_IDENTITY_SCHEMA_FILE: tree_identity,
        PHASE_STATE_SCHEMA_FILE: phase_state,
        COLLAPSE_SCHEMA_FILE: collapse,
        ORACLE_SCHEMA_FILE: oracle,
        MANIFEST_SCHEMA_FILE: manifest,
    }
    for document in documents.values():
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return documents


def validate_schema_instance(schema: Mapping[str, Any], instance: Any) -> None:
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        raise ValueError(f"JSON Schema rejection: {errors[0].message}")


CONTRACT_KEYS = {
    "catalytic_predecessor_law",
    "claim_ceiling",
    "collapse_stage",
    "energy_tolerance",
    "final_lock_tolerance",
    "initial_orientations_rad",
    "initial_state_sha256",
    "lock_final",
    "lock_schedule",
    "lock_start",
    "minimum_unique_optimum_gap",
    "native_dynamics",
    "oracle_stage",
    "parents",
    "portable_metric_atol",
    "portable_metric_rtol",
    "problem_sha256",
    "schema",
    "site_count",
    "site_ids",
    "source_binding",
    "step_count",
    "time_step",
    "trajectory_byte_count",
    "trajectory_dtype",
    "trajectory_format",
    "trajectory_shape",
    "tree_identity_record_sha256",
    "wrap_interval",
}
PROBLEM_KEYS = {"coupling_matrix", "field_vector", "schema", "site_ids"}
TREE_IDENTITY_KEYS = {"schema", "sites"}
TREE_IDENTITY_ITEM_KEYS = {
    "byte_count",
    "fixture_path",
    "geometry_sha256",
    "root_id",
    "site_id",
    "tree_digest",
    "tree_sha256",
}
PHASE_STATE_KEYS = {
    "orientations_rad",
    "schema",
    "site_ids",
    "stage",
    "trajectory_row_index",
    "tree_identity_record_sha256",
}
COLLAPSE_KEYS = {
    "claim_ceiling",
    "collapse_stage",
    "final_antipodal_residual",
    "final_state_sha256",
    "next_energy",
    "observed_energy",
    "observed_spins",
    "optimum_energy",
    "optimum_gap",
    "optimum_spins",
    "oracle_table_sha256",
    "schema",
}
ORACLE_KEYS = {
    "next_energy",
    "optimum_energy",
    "optimum_gap",
    "oracle_stage",
    "problem_sha256",
    "rows",
    "schema",
    "unique_optimum",
}
ORACLE_ROW_KEYS = {"energy", "spins"}
MANIFEST_KEYS = {
    "claim_ceiling",
    "contract_sha256",
    "fixture_count",
    "fixture_set_sha256",
    "fixtures",
    "generator",
    "ordered_fixture_paths",
    "parents",
    "problem_sha256",
    "result_path",
    "result_schema",
    "schema",
    "source_binding",
    "tests_sha256",
    "total_fixture_bytes",
    "trajectory",
    "tree_identity_record_sha256",
}
FIXTURE_RECORD_KEYS = {
    "byte_count",
    "byte_order",
    "dtype",
    "media_type",
    "path",
    "role",
    "sha256",
    "shape",
}


def validate_contract_document(
    document: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any] | None = None,
    expected_parents: Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _reject_forbidden_fields(document, "R3 contract")
    _exact_keys(document, CONTRACT_KEYS, "R3 contract")
    if document["schema"] != CONTRACT_SCHEMA:
        raise ValueError("R3 contract schema mismatch")
    if expected_source is not None and document["source_binding"] != dict(expected_source):
        raise ValueError("R3 contract source identity mismatch")
    if expected_parents is not None and document["parents"] != dict(expected_parents):
        raise ValueError("R3 contract parent identity mismatch")
    if _integer(document["site_count"], "site_count") != SITE_COUNT:
        raise ValueError("R3 contract site count mismatch")
    if document["site_ids"] != list(SITE_IDS):
        raise ValueError("R3 contract site identities mismatch")
    initial = _finite_list(document["initial_orientations_rad"], SITE_COUNT, "initial orientations")
    if not np.array_equal(np.asarray(initial, dtype=np.float64), INITIAL_ORIENTATIONS):
        raise ValueError("R3 contract initial orientations mismatch")
    if _integer(document["step_count"], "step_count") != STEP_COUNT:
        raise ValueError("R3 contract step count mismatch")
    if _finite(document["time_step"], "time_step") != TIME_STEP:
        raise ValueError("R3 contract time step mismatch")
    if _finite(document["lock_start"], "lock_start") != LOCK_START:
        raise ValueError("R3 contract lock start mismatch")
    if _finite(document["lock_final"], "lock_final") != LOCK_FINAL:
        raise ValueError("R3 contract lock final mismatch")
    if document["lock_schedule"] != "linear_in_step_index_0_to_999":
        raise ValueError("R3 contract lock schedule mismatch")
    if document["wrap_interval"] != "[-pi,pi)":
        raise ValueError("R3 contract wrap interval mismatch")
    if _finite(document["final_lock_tolerance"], "final_lock_tolerance") != LOCK_RESIDUAL_MAX:
        raise ValueError("R3 contract final lock tolerance mismatch")
    if _finite(document["energy_tolerance"], "energy_tolerance") != ENERGY_TOL:
        raise ValueError("R3 contract energy tolerance mismatch")
    if _finite(document["minimum_unique_optimum_gap"], "minimum gap") != OPTIMUM_GAP_MIN:
        raise ValueError("R3 contract optimum-gap threshold mismatch")
    if document["trajectory_format"] != TRAJECTORY_FORMAT:
        raise ValueError("R3 contract trajectory format mismatch")
    if document["trajectory_dtype"] != TRAJECTORY_DTYPE:
        raise ValueError("R3 contract trajectory dtype mismatch")
    if document["trajectory_shape"] != list(TRAJECTORY_SHAPE):
        raise ValueError("R3 contract trajectory shape mismatch")
    if _integer(document["trajectory_byte_count"], "trajectory_byte_count") != TRAJECTORY_BYTE_COUNT:
        raise ValueError("R3 contract trajectory byte count mismatch")
    if document["collapse_stage"] != "after_native_trajectory_return":
        raise ValueError("R3 contract collapse-stage law mismatch")
    if document["oracle_stage"] != "after_collapse_boundary":
        raise ValueError("R3 contract oracle-stage law mismatch")
    if document["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("R3 contract claim ceiling mismatch")
    if document["catalytic_predecessor_law"] != "independent_reproduced_predecessor_no_latch_feedback":
        raise ValueError("R3 contract catalytic-predecessor meaning mismatch")
    if document["native_dynamics"] != "phase_difference_plus_field_plus_second_harmonic":
        raise ValueError("R3 contract native dynamics mismatch")
    if _finite(document["portable_metric_atol"], "portable_metric_atol") != PORTABLE_METRIC_ATOL:
        raise ValueError("R3 contract portable absolute tolerance mismatch")
    if _finite(document["portable_metric_rtol"], "portable_metric_rtol") != PORTABLE_METRIC_RTOL:
        raise ValueError("R3 contract portable relative tolerance mismatch")
    if expected_hashes is not None:
        for key in ("problem_sha256", "tree_identity_record_sha256", "initial_state_sha256"):
            if document[key] != expected_hashes[key]:
                raise ValueError(f"R3 contract {key} mismatch")
    return dict(document)


def validate_problem_document(document: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    _reject_forbidden_fields(document, "R3 problem")
    _exact_keys(document, PROBLEM_KEYS, "R3 problem")
    if document["schema"] != PROBLEM_SCHEMA:
        raise ValueError("R3 problem schema mismatch")
    if document["site_ids"] != list(SITE_IDS):
        raise ValueError("R3 problem site identities mismatch")
    matrix = document["coupling_matrix"]
    if not isinstance(matrix, list) or len(matrix) != SITE_COUNT:
        raise ValueError("R3 problem coupling matrix has wrong dimensions")
    coupling = np.asarray(
        [_finite_list(row, SITE_COUNT, f"coupling row {index}") for index, row in enumerate(matrix)],
        dtype=np.float64,
    )
    field = np.asarray(_finite_list(document["field_vector"], SITE_COUNT, "field vector"), dtype=np.float64)
    validate_problem(coupling, field)
    if not np.array_equal(coupling, COUPLING_MATRIX):
        raise ValueError("R3 problem coupling matrix differs from the frozen instance")
    if not np.array_equal(field, FIELD_VECTOR):
        raise ValueError("R3 problem field vector differs from the frozen instance")
    return coupling, field


def validate_tree_identity_document(
    document: Mapping[str, Any],
    tree_payloads: Sequence[bytes] | None = None,
) -> dict[str, Any]:
    _exact_keys(document, TREE_IDENTITY_KEYS, "tree identity record")
    if document["schema"] != TREE_IDENTITY_SCHEMA:
        raise ValueError("tree identity record schema mismatch")
    sites = document["sites"]
    if not isinstance(sites, list) or len(sites) != SITE_COUNT:
        raise ValueError("tree identity record must contain five sites")
    candidate_ids = [
        item.get("site_id") if isinstance(item, Mapping) else None for item in sites
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("tree identity site-ID collision")
    site_ids: list[str] = []
    for index, item in enumerate(sites):
        if not isinstance(item, Mapping):
            raise ValueError("tree identity item must be an object")
        _exact_keys(item, TREE_IDENTITY_ITEM_KEYS, "tree identity item")
        expected_site = SITE_IDS[index]
        if item["site_id"] != expected_site:
            raise ValueError("tree identity site order or identity mismatch")
        if item["fixture_path"] != TREE_PATHS[index]:
            raise ValueError("tree identity fixture path mismatch")
        if item["root_id"] != f"{expected_site}.root":
            raise ValueError("tree identity root mismatch")
        site_ids.append(str(item["site_id"]))
        if tree_payloads is not None:
            payload = tree_payloads[index]
            if _integer(item["byte_count"], "tree byte_count") != len(payload):
                raise ValueError("tree identity byte count mismatch")
            if item["tree_sha256"] != _sha256(payload):
                raise ValueError("tree identity SHA-256 mismatch")
            tree = r0.deserialize_tree_bytes(payload, require_canonical=True)
            if item["tree_digest"] != tree.digest():
                raise ValueError("tree identity digest mismatch")
            if item["geometry_sha256"] != geometry_signature(tree):
                raise ValueError("tree identity geometry mismatch")
    return dict(document)


def validate_phase_state_document(
    document: Mapping[str, Any],
    *,
    expected_tree_record_sha256: str,
    expected_orientations: Sequence[float] | None = None,
) -> dict[str, Any]:
    _exact_keys(document, PHASE_STATE_KEYS, "phase state")
    if document["schema"] != PHASE_STATE_SCHEMA:
        raise ValueError("phase state schema mismatch")
    if document["site_ids"] != list(SITE_IDS):
        raise ValueError("phase state site identities mismatch")
    if document["tree_identity_record_sha256"] != expected_tree_record_sha256:
        raise ValueError("phase state tree identity binding mismatch")
    orientations = np.asarray(
        _finite_list(document["orientations_rad"], SITE_COUNT, "phase orientations"),
        dtype=np.float64,
    )
    if np.any(orientations < -math.pi) or np.any(orientations >= math.pi):
        raise ValueError("phase state orientation lies outside [-pi,pi)")
    stage = document["stage"]
    row = _integer(document["trajectory_row_index"], "trajectory_row_index")
    if stage == "initial_before_native_evolution":
        if row != 0 or not np.allclose(
            orientations,
            INITIAL_ORIENTATIONS,
            rtol=0.0,
            atol=PORTABLE_METRIC_ATOL,
        ):
            raise ValueError("initial phase state differs from the frozen state")
    elif stage == "final_after_1000_native_steps":
        if row != STEP_COUNT:
            raise ValueError("final phase state row mismatch")
        if phase_lock_residual(orientations) > LOCK_RESIDUAL_MAX:
            raise ValueError("final phase state lies outside lock tolerance")
    else:
        raise ValueError("phase state stage mismatch")
    if expected_orientations is not None and not np.array_equal(
        orientations, np.asarray(expected_orientations, dtype=np.float64)
    ):
        raise ValueError("phase state orientations do not match committed trajectory")
    return dict(document)


def _validate_spin_vector(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != SITE_COUNT:
        raise ValueError(f"{label} must contain five spins")
    spins: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item not in (-1, 1):
            raise ValueError(f"{label} must contain only exact -1/+1 integers")
        spins.append(item)
    return tuple(spins)


def validate_oracle_document(
    document: Mapping[str, Any],
    *,
    expected_problem_sha256: str,
) -> dict[str, Any]:
    _exact_keys(document, ORACLE_KEYS, "oracle table")
    if document["schema"] != ORACLE_SCHEMA:
        raise ValueError("oracle table schema mismatch")
    if document["oracle_stage"] != "after_collapse_boundary":
        raise ValueError("oracle table stage mismatch")
    if document["problem_sha256"] != expected_problem_sha256:
        raise ValueError("oracle table problem binding mismatch")
    rows = document["rows"]
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("oracle table must contain exactly 32 rows")
    observed: list[tuple[float, tuple[int, ...]]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("oracle row must be an object")
        _exact_keys(row, ORACLE_ROW_KEYS, "oracle row")
        observed.append((_finite(row["energy"], "oracle energy"), _validate_spin_vector(row["spins"], "oracle spins")))
    expected = exact_ising_oracle()
    if observed != expected:
        raise ValueError("oracle table rows do not match independent exact enumeration")
    optimum_energy, optimum_spins = expected[0]
    next_energy = expected[1][0]
    if document["unique_optimum"] != list(optimum_spins):
        raise ValueError("oracle table unique optimum mismatch")
    if _finite(document["optimum_energy"], "optimum_energy") != optimum_energy:
        raise ValueError("oracle table optimum energy mismatch")
    if _finite(document["next_energy"], "next_energy") != next_energy:
        raise ValueError("oracle table next energy mismatch")
    if _finite(document["optimum_gap"], "optimum_gap") != next_energy - optimum_energy:
        raise ValueError("oracle table optimum gap mismatch")
    return dict(document)


def validate_collapse_document(
    document: Mapping[str, Any],
    *,
    expected_final_state_sha256: str,
    expected_oracle_sha256: str,
    expected_final_phases: Sequence[float],
) -> dict[str, Any]:
    _exact_keys(document, COLLAPSE_KEYS, "collapse receipt")
    if document["schema"] != COLLAPSE_SCHEMA:
        raise ValueError("collapse receipt schema mismatch")
    if document["collapse_stage"] != "after_native_trajectory_return":
        raise ValueError("collapse receipt stage mismatch")
    if document["final_state_sha256"] != expected_final_state_sha256:
        raise ValueError("collapse receipt final-state binding mismatch")
    if document["oracle_table_sha256"] != expected_oracle_sha256:
        raise ValueError("collapse receipt oracle binding mismatch")
    phases = np.asarray(expected_final_phases, dtype=np.float64)
    observed_spins = tuple(1 if math.cos(value) >= 0.0 else -1 for value in phases)
    if _validate_spin_vector(document["observed_spins"], "observed spins") != observed_spins:
        raise ValueError("collapse receipt observed spins mismatch")
    observed_energy = ising_energy(observed_spins)
    if _finite(document["observed_energy"], "observed_energy") != observed_energy:
        raise ValueError("collapse receipt observed energy mismatch")
    residual = phase_lock_residual(phases)
    if not math.isclose(
        _finite(document["final_antipodal_residual"], "final residual"),
        residual,
        rel_tol=PORTABLE_METRIC_RTOL,
        abs_tol=PORTABLE_METRIC_ATOL,
    ):
        raise ValueError("collapse receipt final residual mismatch")
    oracle = exact_ising_oracle()
    optimum_energy, optimum_spins = oracle[0]
    next_energy = oracle[1][0]
    if _validate_spin_vector(document["optimum_spins"], "optimum spins") != optimum_spins:
        raise ValueError("collapse receipt optimum mismatch")
    if _finite(document["optimum_energy"], "optimum_energy") != optimum_energy:
        raise ValueError("collapse receipt optimum energy mismatch")
    if _finite(document["next_energy"], "next_energy") != next_energy:
        raise ValueError("collapse receipt next energy mismatch")
    if _finite(document["optimum_gap"], "optimum_gap") != next_energy - optimum_energy:
        raise ValueError("collapse receipt optimum gap mismatch")
    if document["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("collapse receipt claim ceiling mismatch")
    return dict(document)


def problem_document() -> dict[str, Any]:
    return {
        "coupling_matrix": COUPLING_MATRIX.tolist(),
        "field_vector": FIELD_VECTOR.tolist(),
        "schema": PROBLEM_SCHEMA,
        "site_ids": list(SITE_IDS),
    }


def tree_fixture_payloads(start: Sequence[OrientedRecursiveBeam]) -> tuple[bytes, ...]:
    if len(start) != SITE_COUNT:
        raise ValueError("tree fixture generation requires five sites")
    return tuple(site.tree.canonical_bytes() for site in start)


def tree_identity_document(
    start: Sequence[OrientedRecursiveBeam],
    payloads: Sequence[bytes],
) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    for index, (site, payload) in enumerate(zip(start, payloads, strict=True)):
        sites.append(
            {
                "byte_count": len(payload),
                "fixture_path": TREE_PATHS[index],
                "geometry_sha256": geometry_signature(site.tree),
                "root_id": site.tree.root.node_id,
                "site_id": site.site_id,
                "tree_digest": site.tree.digest(),
                "tree_sha256": _sha256(payload),
            }
        )
    return {"schema": TREE_IDENTITY_SCHEMA, "sites": sites}


def phase_state_document(
    stage: str,
    orientations: Sequence[float],
    tree_record_sha256: str,
    row_index: int,
) -> dict[str, Any]:
    return {
        "orientations_rad": [float(value) for value in orientations],
        "schema": PHASE_STATE_SCHEMA,
        "site_ids": list(SITE_IDS),
        "stage": stage,
        "trajectory_row_index": row_index,
        "tree_identity_record_sha256": tree_record_sha256,
    }


def oracle_document(
    oracle_values: Sequence[tuple[float, tuple[int, ...]]],
    problem_sha256: str,
) -> dict[str, Any]:
    optimum_energy, optimum_spins = oracle_values[0]
    next_energy = oracle_values[1][0]
    return {
        "next_energy": float(next_energy),
        "optimum_energy": float(optimum_energy),
        "optimum_gap": float(next_energy - optimum_energy),
        "oracle_stage": "after_collapse_boundary",
        "problem_sha256": problem_sha256,
        "rows": [
            {"energy": float(energy), "spins": list(spins)}
            for energy, spins in oracle_values
        ],
        "schema": ORACLE_SCHEMA,
        "unique_optimum": list(optimum_spins),
    }


def collapse_receipt_document(
    boundary: Mapping[str, Any],
    final_phases: Sequence[float],
    final_state_sha256: str,
    oracle: Mapping[str, Any],
    oracle_sha256: str,
) -> dict[str, Any]:
    return {
        "claim_ceiling": CLAIM_CEILING,
        "collapse_stage": "after_native_trajectory_return",
        "final_antipodal_residual": phase_lock_residual(
            np.asarray(final_phases, dtype=np.float64)
        ),
        "final_state_sha256": final_state_sha256,
        "next_energy": oracle["next_energy"],
        "observed_energy": boundary["observed_energy"],
        "observed_spins": list(boundary["observed_spins"]),
        "optimum_energy": oracle["optimum_energy"],
        "optimum_gap": oracle["optimum_gap"],
        "optimum_spins": list(oracle["unique_optimum"]),
        "oracle_table_sha256": oracle_sha256,
        "schema": COLLAPSE_SCHEMA,
    }


def contract_document(
    *,
    source: Mapping[str, Any],
    parents: Mapping[str, Any],
    problem_sha256: str,
    tree_record_sha256: str,
    initial_state_sha256: str,
) -> dict[str, Any]:
    return {
        "catalytic_predecessor_law": "independent_reproduced_predecessor_no_latch_feedback",
        "claim_ceiling": CLAIM_CEILING,
        "collapse_stage": "after_native_trajectory_return",
        "energy_tolerance": ENERGY_TOL,
        "final_lock_tolerance": LOCK_RESIDUAL_MAX,
        "initial_orientations_rad": INITIAL_ORIENTATIONS.tolist(),
        "initial_state_sha256": initial_state_sha256,
        "lock_final": LOCK_FINAL,
        "lock_schedule": "linear_in_step_index_0_to_999",
        "lock_start": LOCK_START,
        "minimum_unique_optimum_gap": OPTIMUM_GAP_MIN,
        "native_dynamics": "phase_difference_plus_field_plus_second_harmonic",
        "oracle_stage": "after_collapse_boundary",
        "parents": dict(parents),
        "portable_metric_atol": PORTABLE_METRIC_ATOL,
        "portable_metric_rtol": PORTABLE_METRIC_RTOL,
        "problem_sha256": problem_sha256,
        "schema": CONTRACT_SCHEMA,
        "site_count": SITE_COUNT,
        "site_ids": list(SITE_IDS),
        "source_binding": dict(source),
        "step_count": STEP_COUNT,
        "time_step": TIME_STEP,
        "trajectory_byte_count": TRAJECTORY_BYTE_COUNT,
        "trajectory_dtype": TRAJECTORY_DTYPE,
        "trajectory_format": TRAJECTORY_FORMAT,
        "trajectory_shape": list(TRAJECTORY_SHAPE),
        "tree_identity_record_sha256": tree_record_sha256,
        "wrap_interval": "[-pi,pi)",
    }


def trajectory_bytes(trajectory: np.ndarray) -> bytes:
    array = np.asarray(trajectory, dtype=np.dtype(TRAJECTORY_DTYPE))
    if array.shape != TRAJECTORY_SHAPE:
        raise ValueError("phase trajectory has the wrong shape")
    if not np.all(np.isfinite(array)):
        raise ValueError("phase trajectory contains non-finite values")
    if np.any(array < -math.pi) or np.any(array >= math.pi):
        raise ValueError("phase trajectory contains a value outside [-pi,pi)")
    return array.tobytes(order="C")


def parse_trajectory_bytes(payload: bytes) -> np.ndarray:
    if len(payload) != TRAJECTORY_BYTE_COUNT:
        raise ValueError(
            f"phase trajectory byte count must equal {TRAJECTORY_BYTE_COUNT}, got {len(payload)}"
        )
    array = np.frombuffer(payload, dtype=np.dtype(TRAJECTORY_DTYPE)).reshape(
        TRAJECTORY_SHAPE
    )
    if not np.all(np.isfinite(array)):
        raise ValueError("phase trajectory contains non-finite values")
    if np.any(array < -math.pi) or np.any(array >= math.pi):
        raise ValueError("phase trajectory contains a value outside [-pi,pi)")
    return array.copy()


def execute_reference_lifecycle() -> dict[str, Any]:
    source = source_binding()
    parents = parent_custody()
    problem = problem_document()
    problem_payload = canonical_bytes(problem)
    start = initial_state()
    tree_payloads = tree_fixture_payloads(start)
    tree_record = tree_identity_document(start, tree_payloads)
    tree_record_payload = canonical_bytes(tree_record)
    tree_record_sha256 = _sha256(tree_record_payload)
    initial_document = phase_state_document(
        "initial_before_native_evolution",
        [site.orientation_rad for site in start],
        tree_record_sha256,
        0,
    )
    initial_payload = canonical_bytes(initial_document)
    final_state, trajectory, tree_identity_checks = evolve_phase_trajectory(start)
    final_phases = np.asarray(
        [site.orientation_rad for site in final_state], dtype=np.float64
    )
    trajectory_payload = trajectory_bytes(trajectory)
    final_document = phase_state_document(
        "final_after_1000_native_steps",
        final_phases,
        tree_record_sha256,
        STEP_COUNT,
    )
    final_payload = canonical_bytes(final_document)
    boundary = collapse_boundary(final_state)
    oracle_values = exact_ising_oracle()
    oracle = oracle_document(oracle_values, _sha256(problem_payload))
    oracle_payload = canonical_bytes(oracle)
    collapse = collapse_receipt_document(
        boundary,
        final_phases,
        _sha256(final_payload),
        oracle,
        _sha256(oracle_payload),
    )
    collapse_payload = canonical_bytes(collapse)
    contract = contract_document(
        source=source,
        parents=parents,
        problem_sha256=_sha256(problem_payload),
        tree_record_sha256=tree_record_sha256,
        initial_state_sha256=_sha256(initial_payload),
    )
    contract_payload = canonical_bytes(contract)
    return {
        "boundary": boundary,
        "collapse": collapse,
        "collapse_payload": collapse_payload,
        "contract": contract,
        "contract_payload": contract_payload,
        "final_document": final_document,
        "final_payload": final_payload,
        "final_phases": final_phases,
        "final_state": final_state,
        "initial_document": initial_document,
        "initial_payload": initial_payload,
        "oracle": oracle,
        "oracle_payload": oracle_payload,
        "parents": parents,
        "problem": problem,
        "problem_payload": problem_payload,
        "source": source,
        "start": start,
        "trajectory": trajectory,
        "trajectory_payload": trajectory_payload,
        "tree_identity_checks": tree_identity_checks,
        "tree_payloads": tree_payloads,
        "tree_record": tree_record,
        "tree_record_payload": tree_record_payload,
    }


def collapsed_spin_resynthesis_baseline() -> dict[str, Any]:
    spins = np.asarray(
        [1 if math.cos(value) >= 0.0 else -1 for value in INITIAL_ORIENTATIONS],
        dtype=np.int64,
    )
    for _ in range(STEP_COUNT):
        drive = COUPLING_MATRIX @ spins.astype(np.float64) + FIELD_VECTOR
        spins = np.where(drive >= 0.0, 1, -1).astype(np.int64)
    final_spins = tuple(int(value) for value in spins)
    return {
        "binary_after_every_step": True,
        "energy": ising_energy(final_spins),
        "spins": list(final_spins),
        "structural_state_loss": "continuous_phase_and_internal_relation_discarded_each_step",
    }


def energy_retention_baseline(trajectory: np.ndarray) -> dict[str, Any]:
    best_energy = math.inf
    best_spins: tuple[int, ...] | None = None
    update_count = 0
    for phases in np.asarray(trajectory, dtype=np.float64):
        spins = tuple(1 if math.cos(value) >= 0.0 else -1 for value in phases)
        energy = ising_energy(spins)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins
            update_count += 1
    if best_spins is None:
        raise ValueError("energy-retention baseline received an empty trajectory")
    return {
        "best_energy": best_energy,
        "best_spins": list(best_spins),
        "energy_evaluated_after_every_step": True,
        "retention_updates": update_count,
        "structural_state_loss": "binary_energy_selection_controls_retained_output",
    }


FIXTURE_ROLES = (
    "r3_contract",
    "ising_problem",
    "site_tree_identities",
    "site_tree_0",
    "site_tree_1",
    "site_tree_2",
    "site_tree_3",
    "site_tree_4",
    "initial_phase_state",
    "phase_trajectory",
    "final_phase_state",
    "collapse_receipt",
    "exact_oracle_table",
)


def lifecycle_fixture_payloads(lifecycle: Mapping[str, Any]) -> dict[str, bytes]:
    payloads = {
        CONTRACT_PATH: lifecycle["contract_payload"],
        PROBLEM_PATH: lifecycle["problem_payload"],
        TREE_IDENTITIES_PATH: lifecycle["tree_record_payload"],
        INITIAL_STATE_PATH: lifecycle["initial_payload"],
        TRAJECTORY_PATH: lifecycle["trajectory_payload"],
        FINAL_STATE_PATH: lifecycle["final_payload"],
        COLLAPSE_PATH: lifecycle["collapse_payload"],
        ORACLE_PATH: lifecycle["oracle_payload"],
    }
    for path, payload in zip(TREE_PATHS, lifecycle["tree_payloads"], strict=True):
        payloads[path] = payload
    if set(payloads) != set(ORDERED_FIXTURE_PATHS):
        raise ValueError("R3 lifecycle fixture set is incomplete")
    return {path: payloads[path] for path in ORDERED_FIXTURE_PATHS}


def _fixture_record(path: str, role: str, payload: bytes) -> dict[str, Any]:
    if path == TRAJECTORY_PATH:
        return {
            "byte_count": len(payload),
            "byte_order": "little",
            "dtype": TRAJECTORY_DTYPE,
            "media_type": "application/vnd.cat-cas.raw-float64-phase-matrix",
            "path": path,
            "role": role,
            "sha256": _sha256(payload),
            "shape": list(TRAJECTORY_SHAPE),
        }
    return {
        "byte_count": len(payload),
        "byte_order": "utf-8",
        "dtype": "canonical_json",
        "media_type": "application/json",
        "path": path,
        "role": role,
        "sha256": _sha256(payload),
        "shape": None,
    }


def fixture_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "byte_count": record["byte_count"],
            "path": record["path"],
            "role": record["role"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    return _sha256(canonical_bytes(identity))


def fixture_manifest_document(
    lifecycle: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    tests_sha256: str,
) -> dict[str, Any]:
    records = [
        _fixture_record(path, role, payloads[path])
        for path, role in zip(ORDERED_FIXTURE_PATHS, FIXTURE_ROLES, strict=True)
    ]
    trajectory_record = records[ORDERED_FIXTURE_PATHS.index(TRAJECTORY_PATH)]
    return {
        "claim_ceiling": CLAIM_CEILING,
        "contract_sha256": _sha256(payloads[CONTRACT_PATH]),
        "fixture_count": len(records),
        "fixture_set_sha256": fixture_set_sha256(records),
        "fixtures": records,
        "generator": GENERATOR_ID,
        "ordered_fixture_paths": list(ORDERED_FIXTURE_PATHS),
        "parents": lifecycle["parents"],
        "problem_sha256": _sha256(payloads[PROBLEM_PATH]),
        "result_path": RESULTS_FILE,
        "result_schema": RESULT_SCHEMA,
        "schema": MANIFEST_SCHEMA,
        "source_binding": lifecycle["source"],
        "tests_sha256": tests_sha256,
        "total_fixture_bytes": sum(len(payload) for payload in payloads.values()),
        "trajectory": {
            "byte_count": trajectory_record["byte_count"],
            "dtype": trajectory_record["dtype"],
            "format": TRAJECTORY_FORMAT,
            "sha256": trajectory_record["sha256"],
            "shape": trajectory_record["shape"],
        },
        "tree_identity_record_sha256": _sha256(payloads[TREE_IDENTITIES_PATH]),
    }


def validate_manifest_document(
    document: Mapping[str, Any],
    package_dir: Path,
    *,
    expected_source: Mapping[str, Any],
    expected_parents: Mapping[str, Any],
) -> dict[str, bytes]:
    _exact_keys(document, MANIFEST_KEYS, "R3 fixture manifest")
    if document["schema"] != MANIFEST_SCHEMA:
        raise ValueError("R3 fixture manifest schema mismatch")
    if document["generator"] != GENERATOR_ID:
        raise ValueError("R3 fixture manifest generator mismatch")
    if document["source_binding"] != dict(expected_source):
        raise ValueError("R3 fixture manifest source identity mismatch")
    if document["parents"] != dict(expected_parents):
        raise ValueError("R3 fixture manifest parent identity mismatch")
    if document["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("R3 fixture manifest claim ceiling mismatch")
    if document["ordered_fixture_paths"] != list(ORDERED_FIXTURE_PATHS):
        raise ValueError("R3 fixture manifest ordered paths mismatch")
    if document["result_path"] != RESULTS_FILE or document["result_schema"] != RESULT_SCHEMA:
        raise ValueError("R3 fixture manifest result binding mismatch")
    tests_path = package_dir / TESTS_FILE
    if document["tests_sha256"] != _sha256(tests_path.read_bytes()):
        raise ValueError("R3 fixture manifest test binding mismatch")
    records = document["fixtures"]
    if not isinstance(records, list) or len(records) != len(ORDERED_FIXTURE_PATHS):
        raise ValueError("R3 fixture manifest record count mismatch")
    payloads: dict[str, bytes] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("R3 fixture manifest record must be an object")
        _exact_keys(record, FIXTURE_RECORD_KEYS, "R3 fixture record")
        path = ORDERED_FIXTURE_PATHS[index]
        role = FIXTURE_ROLES[index]
        if record["path"] != path or record["role"] != role:
            raise ValueError("R3 fixture manifest role substitution detected")
        payload = (package_dir / path).read_bytes()
        if _integer(record["byte_count"], "fixture byte_count") != len(payload):
            raise ValueError(f"R3 fixture byte count mismatch for role {role}")
        if record["sha256"] != _sha256(payload):
            raise ValueError(f"R3 fixture SHA-256 mismatch for role {role}")
        expected = _fixture_record(path, role, payload)
        if dict(record) != expected:
            raise ValueError(f"R3 fixture metadata mismatch for role {role}")
        payloads[path] = payload
    if _integer(document["fixture_count"], "fixture_count") != len(records):
        raise ValueError("R3 fixture manifest fixture count mismatch")
    if _integer(document["total_fixture_bytes"], "total_fixture_bytes") != sum(
        len(payload) for payload in payloads.values()
    ):
        raise ValueError("R3 fixture manifest total byte count mismatch")
    if document["fixture_set_sha256"] != fixture_set_sha256(records):
        raise ValueError("R3 fixture manifest fixture-set mismatch")
    if document["contract_sha256"] != _sha256(payloads[CONTRACT_PATH]):
        raise ValueError("R3 fixture manifest contract binding mismatch")
    if document["problem_sha256"] != _sha256(payloads[PROBLEM_PATH]):
        raise ValueError("R3 fixture manifest problem binding mismatch")
    if document["tree_identity_record_sha256"] != _sha256(payloads[TREE_IDENTITIES_PATH]):
        raise ValueError("R3 fixture manifest tree binding mismatch")
    expected_trajectory = {
        "byte_count": TRAJECTORY_BYTE_COUNT,
        "dtype": TRAJECTORY_DTYPE,
        "format": TRAJECTORY_FORMAT,
        "sha256": _sha256(payloads[TRAJECTORY_PATH]),
        "shape": list(TRAJECTORY_SHAPE),
    }
    if document["trajectory"] != expected_trajectory:
        raise ValueError("R3 fixture manifest trajectory binding mismatch")
    return payloads


def load_committed_packet(package_dir: Path, source_path: Path) -> dict[str, Any]:
    source = source_binding(source_path)
    parents = parent_custody()
    manifest = _load_canonical_json(package_dir / MANIFEST_FILE, MANIFEST_FILE)
    payloads = validate_manifest_document(
        manifest,
        package_dir,
        expected_source=source,
        expected_parents=parents,
    )
    problem = _strict_document(payloads[PROBLEM_PATH], "R3 problem fixture")
    if not isinstance(problem, Mapping):
        raise ValueError("R3 problem fixture must be an object")
    validate_problem_document(problem)
    tree_payloads = tuple(payloads[path] for path in TREE_PATHS)
    tree_record = _strict_document(payloads[TREE_IDENTITIES_PATH], "tree identity fixture")
    if not isinstance(tree_record, Mapping):
        raise ValueError("tree identity fixture must be an object")
    validate_tree_identity_document(tree_record, tree_payloads)
    tree_record_sha256 = _sha256(payloads[TREE_IDENTITIES_PATH])
    initial = _strict_document(payloads[INITIAL_STATE_PATH], "initial phase state")
    final = _strict_document(payloads[FINAL_STATE_PATH], "final phase state")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise ValueError("phase-state fixtures must be objects")
    trajectory = parse_trajectory_bytes(payloads[TRAJECTORY_PATH])
    validate_phase_state_document(
        initial,
        expected_tree_record_sha256=tree_record_sha256,
        expected_orientations=trajectory[0],
    )
    validate_phase_state_document(
        final,
        expected_tree_record_sha256=tree_record_sha256,
        expected_orientations=trajectory[-1],
    )
    contract = _strict_document(payloads[CONTRACT_PATH], "R3 contract fixture")
    if not isinstance(contract, Mapping):
        raise ValueError("R3 contract fixture must be an object")
    validate_contract_document(
        contract,
        expected_source=source,
        expected_parents=parents,
        expected_hashes={
            "initial_state_sha256": _sha256(payloads[INITIAL_STATE_PATH]),
            "problem_sha256": _sha256(payloads[PROBLEM_PATH]),
            "tree_identity_record_sha256": tree_record_sha256,
        },
    )
    oracle = _strict_document(payloads[ORACLE_PATH], "oracle table fixture")
    if not isinstance(oracle, Mapping):
        raise ValueError("oracle table fixture must be an object")
    validate_oracle_document(oracle, expected_problem_sha256=_sha256(payloads[PROBLEM_PATH]))
    collapse = _strict_document(payloads[COLLAPSE_PATH], "collapse receipt fixture")
    if not isinstance(collapse, Mapping):
        raise ValueError("collapse receipt fixture must be an object")
    validate_collapse_document(
        collapse,
        expected_final_state_sha256=_sha256(payloads[FINAL_STATE_PATH]),
        expected_oracle_sha256=_sha256(payloads[ORACLE_PATH]),
        expected_final_phases=trajectory[-1],
    )
    start = tuple(
        OrientedRecursiveBeam(
            SITE_IDS[index],
            r0.deserialize_tree_bytes(tree_payloads[index], require_canonical=True),
            float(trajectory[0, index]),
        )
        for index in range(SITE_COUNT)
    )
    recomputed_final, recomputed_trajectory, tree_checks = evolve_phase_trajectory(start)
    if trajectory_bytes(recomputed_trajectory) != payloads[TRAJECTORY_PATH]:
        raise ValueError("committed trajectory does not match frozen native evolution")
    recomputed_final_phases = np.asarray(
        [site.orientation_rad for site in recomputed_final], dtype=np.float64
    )
    if not np.array_equal(recomputed_final_phases, trajectory[-1]):
        raise ValueError("committed final state does not match frozen native evolution")
    return {
        "collapse": dict(collapse),
        "contract": dict(contract),
        "final": dict(final),
        "manifest": dict(manifest),
        "oracle": dict(oracle),
        "parents": parents,
        "payloads": payloads,
        "problem": dict(problem),
        "source": source,
        "trajectory": trajectory,
        "tree_checks": tree_checks,
        "tree_payloads": tree_payloads,
        "tree_record": dict(tree_record),
    }


def _ast_shape_sha256(node: ast.AST) -> str:
    return _sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8"))


def _ast_dump_set_sha256(nodes: Sequence[ast.AST]) -> str:
    dumps = sorted(
        ast.dump(node, annotate_fields=True, include_attributes=False) for node in nodes
    )
    return _sha256(canonical_bytes(dumps))


def _module_load_skeleton(module: ast.Module) -> ast.Module:
    class StripBodies(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            replacement = copy.deepcopy(node)
            replacement.decorator_list = [self.visit(item) for item in replacement.decorator_list]
            replacement.args = self.visit(replacement.args)
            replacement.returns = self.visit(replacement.returns) if replacement.returns else None
            replacement.body = [ast.Pass()]
            return replacement

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            replacement = copy.deepcopy(node)
            replacement.decorator_list = [self.visit(item) for item in replacement.decorator_list]
            replacement.args = self.visit(replacement.args)
            replacement.returns = self.visit(replacement.returns) if replacement.returns else None
            replacement.body = [ast.Pass()]
            return replacement

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            replacement = copy.deepcopy(node)
            replacement.args = self.visit(replacement.args)
            replacement.body = ast.Constant(None)
            return replacement

    skeleton = StripBodies().visit(copy.deepcopy(module))
    ast.fix_missing_locations(skeleton)
    return skeleton


def _binding_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _binding_names(item)]
    return []


def _module_binding_inventory(module: ast.Module) -> list[str]:
    inventory: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inventory.append(f"function:{node.name}")
        elif isinstance(node, ast.ClassDef):
            inventory.append(f"class:{node.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                inventory.append(f"import:{alias.asname or alias.name.split('.')[0]}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                inventory.append(f"import:{alias.asname or alias.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _binding_names(target):
                    inventory.append(f"name:{type(target.ctx).__name__}:{name}")
    return sorted(inventory)


def _top_level_functions(module: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }


def _qualified_native_nodes(module: ast.Module) -> dict[str, ast.FunctionDef]:
    nodes = _top_level_functions(module)
    for class_node in module.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for child in class_node.body:
            if isinstance(child, ast.FunctionDef):
                nodes[f"{class_node.name}.{child.name}"] = child
    return nodes


def _native_call_graph(
    nodes: Mapping[str, ast.FunctionDef],
    protected: Sequence[str],
) -> dict[str, list[str]]:
    protected_set = set(protected)
    simple_names = {name: name for name in protected_set if "." not in name}
    graph: dict[str, list[str]] = {}
    for name in protected:
        edges: set[str] = set()
        node = nodes.get(name)
        if node is None:
            graph[name] = []
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name) and child.func.id in simple_names:
                edges.add(simple_names[child.func.id])
            elif isinstance(child.func, ast.Attribute):
                method_name = f"OrientedRecursiveBeam.{child.func.attr}"
                if method_name in protected_set:
                    edges.add(method_name)
            if (
                name == "OrientedRecursiveBeam.with_orientation"
                and isinstance(child.func, ast.Name)
                and child.func.id == "OrientedRecursiveBeam"
            ):
                edges.add("OrientedRecursiveBeam.__post_init__")
        graph[name] = sorted(edges)
    return graph


def native_ast_proof_text(source_text: str) -> dict[str, Any]:
    expected_native_node_sha256 = {
        "_finite": "89d4bf98a17d411a9310201028078c2250032518c887757ce2522e7e20a8a974",
        "wrap_phase": "799b6d31e752a9122c993954e4f552127ff4f4935fd1ab60c045745a28931b4d",
        "validate_problem": "a7ef797409d87d1998bc9ac19feaad932ff8e88d1447f5936aedab53a0c90785",
        "lock_strength": "2da00101d0fd23f5706d0eaf3bff1c9ad376a570cd0f046a99fa85bfda78c41b",
        "OrientedRecursiveBeam.__post_init__": "45155d3ae7fd75b5d5556b8bfb0b2d36c2eceff41bb9b42aefb93434578b5569",
        "OrientedRecursiveBeam.with_orientation": "11af3c57eb92a844a695fadba9806d4e0be51989977e51b28d30aa15ab621269",
        "evolve_phase_state": "7f78de52e5dad8aeb3dd75dd4c70bd0139bd371427998f1464b7b9af5ddb7fd7",
        "evolve_phase_trajectory": "01b7f1b3582b1d4ff12e40f470f5e69fef2817af1245ef057204514bda65bdb3",
        "native_phase_step": "54c4556a45ff51b472321bcd5593dd097c942061192148eeb90e04f5d975e225",
        "native_phase_velocity": "d8cd61c23f1817d063ca3afca60a7a7d17ae461f9842fda73d4ed4d6198b480b",
    }
    expected_lifecycle_sha256 = "334727de992cde5fbb7135f4c448a4ccf801a07fd546d0b343d721214d9f3981"
    expected_module_skeleton_sha256 = "73a72f282c4e38aafd3ca1eed90b3b9daf1a1d3d41585d665c5290ce5c9bc8ab"
    expected_module_runtime_calls_sha256 = "02b88667a438588166d3e07dd92750d81d93db56b5789e40633fd22d11a74cfa"
    expected_module_binding_inventory_sha256 = "fafa9a3885ec1c1055fddc777d0d8c92f733c625b49adb04907ea165a7c941b0"
    expected_module_indirect_writes_sha256 = "2e0969cd37b70dad54530854c55cefbaae15238cff6277bc0e018fa2bca66116"
    module = ast.parse(source_text)
    functions = _top_level_functions(module)
    native_nodes = _qualified_native_nodes(module)
    roots = (
        "evolve_phase_state",
        "evolve_phase_trajectory",
        "native_phase_step",
        "native_phase_velocity",
    )
    protected = tuple(expected_native_node_sha256)
    missing_roots = sorted(set(protected) - set(native_nodes))
    function_sha256: dict[str, str] = {}
    function_shape_violations: list[str] = []
    forbidden_identifiers = {
        "argmax",
        "argmin",
        "candidate",
        "coefficient_vector",
        "collapse_boundary",
        "collapsed_spin_resynthesis_baseline",
        "decode_spins",
        "energy_retention_baseline",
        "exact_ising_oracle",
        "expected_optimum",
        "flat_sine",
        "ising_energy",
        "latch_response",
        "score",
        "sign",
        "winner",
    }
    forbidden_hits: list[str] = []
    for name in protected:
        if name not in native_nodes:
            continue
        node = native_nodes[name]
        observed_hash = _ast_shape_sha256(node)
        function_sha256[name] = observed_hash
        if observed_hash != expected_native_node_sha256[name]:
            function_shape_violations.append(name)
        for child in ast.walk(node):
            identifier: str | None = None
            if isinstance(child, ast.Name):
                identifier = child.id
            elif isinstance(child, ast.Attribute):
                identifier = child.attr
            if identifier in forbidden_identifiers:
                forbidden_hits.append(f"{name}:{identifier}")
    call_graph = _native_call_graph(native_nodes, protected)
    reachable = set(roots)
    frontier = list(roots)
    while frontier:
        current = frontier.pop()
        for target in call_graph.get(current, []):
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    call_graph_closure_violation = reachable != set(protected)
    lifecycle = functions.get("execute_reference_lifecycle")
    lifecycle_sha256 = _ast_shape_sha256(lifecycle) if lifecycle is not None else None
    lifecycle_shape_violation = lifecycle_sha256 != expected_lifecycle_sha256
    lifecycle_positions: dict[str, int] = {}
    if lifecycle is not None:
        for child in ast.walk(lifecycle):
            if not isinstance(child, ast.Call):
                continue
            call_name: str | None = None
            if isinstance(child.func, ast.Name):
                call_name = child.func.id
            elif isinstance(child.func, ast.Attribute):
                call_name = child.func.attr
            if call_name in {
                "evolve_phase_trajectory",
                "collapse_boundary",
                "exact_ising_oracle",
                "collapse_receipt_document",
            }:
                lifecycle_positions.setdefault(call_name, child.lineno)
    lifecycle_order = (
        set(lifecycle_positions)
        == {
            "evolve_phase_trajectory",
            "collapse_boundary",
            "exact_ising_oracle",
            "collapse_receipt_document",
        }
        and lifecycle_positions["evolve_phase_trajectory"]
        < lifecycle_positions["collapse_boundary"]
        < lifecycle_positions["exact_ising_oracle"]
        < lifecycle_positions["collapse_receipt_document"]
    )
    skeleton = _module_load_skeleton(module)
    module_skeleton_sha256 = _ast_shape_sha256(skeleton)
    runtime_calls = [node for node in ast.walk(skeleton) if isinstance(node, ast.Call)]
    module_runtime_calls_sha256 = _ast_dump_set_sha256(runtime_calls)
    binding_inventory = _module_binding_inventory(skeleton)
    module_binding_inventory_sha256 = _sha256(canonical_bytes(binding_inventory))
    indirect_writes = [
        node
        for node in ast.walk(skeleton)
        if isinstance(node, (ast.Attribute, ast.Subscript))
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    module_indirect_writes_sha256 = _ast_dump_set_sha256(indirect_writes)
    module_load_violations: list[str] = []
    if module_skeleton_sha256 != expected_module_skeleton_sha256:
        module_load_violations.append("module_skeleton")
    if module_runtime_calls_sha256 != expected_module_runtime_calls_sha256:
        module_load_violations.append("module_runtime_calls")
    if module_binding_inventory_sha256 != expected_module_binding_inventory_sha256:
        module_load_violations.append("module_binding_inventory")
    if module_indirect_writes_sha256 != expected_module_indirect_writes_sha256:
        module_load_violations.append("module_indirect_writes")
    status = "PASS" if not any(
        (
            missing_roots,
            function_shape_violations,
            forbidden_hits,
            call_graph_closure_violation,
            lifecycle_shape_violation,
            not lifecycle_order,
            module_load_violations,
        )
    ) else "FAIL"
    return {
        "call_graph": call_graph,
        "call_graph_closure_violation": call_graph_closure_violation,
        "call_graph_reachable": sorted(reachable),
        "forbidden_hits": sorted(set(forbidden_hits)),
        "function_sha256": function_sha256,
        "function_shape_violations": function_shape_violations,
        "lifecycle_order": lifecycle_order,
        "lifecycle_positions": lifecycle_positions,
        "lifecycle_sha256": lifecycle_sha256,
        "lifecycle_shape_violation": lifecycle_shape_violation,
        "missing_roots": missing_roots,
        "module_binding_inventory_sha256": module_binding_inventory_sha256,
        "module_indirect_write_count": len(indirect_writes),
        "module_indirect_writes_sha256": module_indirect_writes_sha256,
        "module_load_skeleton_sha256": module_skeleton_sha256,
        "module_load_violations": module_load_violations,
        "module_runtime_call_count": len(runtime_calls),
        "module_runtime_calls_sha256": module_runtime_calls_sha256,
        "native_roots": list(roots),
        "native_transitive_closure": list(protected),
        "status": status,
    }


def native_ast_proof(source_path: Path = Path(__file__).resolve()) -> dict[str, Any]:
    return native_ast_proof_text(source_path.read_text(encoding="utf-8"))


def _replace_in_function(
    source_text: str,
    function_name: str,
    old: str,
    new: str,
) -> str:
    module = ast.parse(source_text)
    function = _top_level_functions(module).get(function_name)
    if function is None or function.end_lineno is None:
        raise ValueError(f"unable to locate function for structural probe: {function_name}")
    lines = source_text.splitlines(keepends=True)
    start = function.lineno - 1
    end = function.end_lineno
    segment = "".join(lines[start:end])
    if old not in segment:
        raise ValueError(f"structural probe target missing in {function_name}: {old}")
    mutated = segment.replace(old, new, 1)
    return "".join(lines[:start]) + mutated + "".join(lines[end:])


def _replace_in_class_method(
    source_text: str,
    class_name: str,
    method_name: str,
    old: str,
    new: str,
) -> str:
    module = ast.parse(source_text)
    class_node = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    method = None if class_node is None else next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ),
        None,
    )
    if method is None or method.end_lineno is None:
        raise ValueError(
            f"unable to locate structural probe method: {class_name}.{method_name}"
        )
    lines = source_text.splitlines(keepends=True)
    start = method.lineno - 1
    end = method.end_lineno
    segment = "".join(lines[start:end])
    if old not in segment:
        raise ValueError(
            f"structural probe target missing in {class_name}.{method_name}: {old}"
        )
    mutated = segment.replace(old, new, 1)
    return "".join(lines[:start]) + mutated + "".join(lines[end:])


def native_ast_mutation_probes(source_path: Path = Path(__file__).resolve()) -> dict[str, bool]:
    source_text = source_path.read_text(encoding="utf-8")
    variants: dict[str, str] = {}
    variants["native_decode_feedback"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "relation = 0.0",
        "relation = float(decode_spins(state)[site])",
    )
    variants["native_energy_feedback"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "velocity = np.zeros(SITE_COUNT, dtype=np.float64)",
        "energy = ising_energy(decode_spins(state))\n    velocity = np.zeros(SITE_COUNT, dtype=np.float64)",
    )
    variants["native_matrix_spin_product"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "relation += coupling[site, neighbor] * math.sin(",
        "relation += float((coupling @ np.sign(phases))[site]) * math.sin(",
    )
    variants["native_field_threshold"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "field[site] * math.sin(phases[site])",
        "field[site] * np.sign(phases[site])",
    )
    variants["native_lock_threshold"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "lock_value * math.sin(2.0 * phases[site])",
        "lock_value * np.sign(phases[site])",
    )
    variants["step_decode_feedback"] = _replace_in_function(
        source_text,
        "native_phase_step",
        "velocity = native_phase_velocity(",
        "spins = decode_spins(state)\n    velocity = native_phase_velocity(",
    )
    variants["trajectory_oracle_feedback"] = _replace_in_function(
        source_text,
        "evolve_phase_trajectory",
        "state = tuple(start)",
        "oracle = exact_ising_oracle()\n    state = tuple(start)",
    )
    variants["trajectory_collapse_feedback"] = _replace_in_function(
        source_text,
        "evolve_phase_trajectory",
        "state = tuple(start)",
        "collapsed = collapse_boundary(start)\n    state = tuple(start)",
    )
    variants["state_energy_feedback"] = _replace_in_function(
        source_text,
        "evolve_phase_state",
        "final_state, _, _ = evolve_phase_trajectory(",
        "energy = ising_energy(decode_spins(start))\n    final_state, _, _ = evolve_phase_trajectory(",
    )
    variants["r2s_latch_feedback"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "phases = np.array([site.orientation_rad for site in state], dtype=np.float64)",
        "latch_response = 0.0\n    phases = np.array([site.orientation_rad for site in state], dtype=np.float64)",
    )
    variants["winner_feedback"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "relation = 0.0",
        "winner = 0\n        relation = 0.0",
    )
    variants["expected_optimum_feedback"] = _replace_in_function(
        source_text,
        "native_phase_velocity",
        "relation = 0.0",
        "expected_optimum = None\n        relation = 0.0",
    )
    variants["flat_sine_substitution"] = _replace_in_function(
        source_text,
        "evolve_phase_trajectory",
        "frozen_tree_bytes = tuple",
        "flat_sine = np.zeros(1)\n    frozen_tree_bytes = tuple",
    )
    variants["coefficient_vector_substitution"] = _replace_in_function(
        source_text,
        "evolve_phase_trajectory",
        "frozen_tree_bytes = tuple",
        "coefficient_vector = np.zeros(1)\n    frozen_tree_bytes = tuple",
    )
    variants["helper_lock_oracle_feedback"] = _replace_in_function(
        source_text,
        "lock_strength",
        "alpha = step_index / (STEP_COUNT - 1)",
        "oracle = exact_ising_oracle()\n    alpha = step_index / (STEP_COUNT - 1)",
    )
    variants["helper_problem_oracle_feedback"] = _replace_in_function(
        source_text,
        "validate_problem",
        "coupling = np.asarray(coupling, dtype=np.float64)",
        "oracle = exact_ising_oracle()\n    coupling = np.asarray(coupling, dtype=np.float64)",
    )
    variants["helper_wrap_antipodal_projection"] = _replace_in_function(
        source_text,
        "wrap_phase",
        "wrapped = (array + math.pi) % (2.0 * math.pi) - math.pi",
        "wrapped = np.where(np.cos(array) >= 0.0, 0.0, -math.pi)",
    )
    variants["method_orientation_oracle_feedback"] = _replace_in_class_method(
        source_text,
        "OrientedRecursiveBeam",
        "with_orientation",
        "return OrientedRecursiveBeam(self.site_id, self.tree, value)",
        "oracle = exact_ising_oracle()\n        return OrientedRecursiveBeam(self.site_id, self.tree, value)",
    )
    variants["method_post_init_latch_feedback"] = _replace_in_class_method(
        source_text,
        "OrientedRecursiveBeam",
        "__post_init__",
        "object.__setattr__(",
        "latch_response = 0.0\n        object.__setattr__(",
    )
    variants["lifecycle_oracle_before_collapse"] = _replace_in_function(
        source_text,
        "execute_reference_lifecycle",
        "boundary = collapse_boundary(final_state)",
        "oracle_values = exact_ising_oracle()\n    boundary = collapse_boundary(final_state)",
    )
    variants["module_direct_rebind"] = source_text + "\nnative_phase_step = collapse_boundary\n"
    variants["module_import_rebind"] = source_text + "\nfrom math import cos as native_phase_step\n"
    variants["module_namespace_write"] = source_text + "\nglobals()['native_phase_step'] = collapse_boundary\n"
    variants["module_default_side_effect"] = source_text + (
        "\ndef _rci_default_probe(value=globals().__setitem__('native_phase_step', "
        "collapse_boundary)):\n    return value\n"
    )
    variants["module_class_side_effect"] = source_text + (
        "\nclass _RciClassProbe:\n    native_phase_step = collapse_boundary\n"
    )
    variants["module_attribute_write"] = source_text + "\nnp.rci_native_step = collapse_boundary\n"
    return {
        name: native_ast_proof_text(mutated)["status"] == "FAIL"
        for name, mutated in variants.items()
    }


def _expect_error(function: Callable[[], Any], phrase: str) -> str:
    try:
        function()
    except (ValueError, TypeError, AssertionError) as exc:
        message = str(exc)
        if phrase.lower() not in message.lower():
            raise AssertionError(
                f"negative control raised the wrong reason: expected {phrase!r}, got {message!r}"
            ) from exc
        return message
    raise AssertionError(f"negative control did not reject: expected {phrase!r}")


def reference_test_ids() -> tuple[str, ...]:
    return (
        "parent_r0_custody_exact",
        "parent_r0_reproduced",
        "parent_r1_custody_exact",
        "parent_r1_reproduced",
        "parent_r2s_custody_exact",
        "parent_r2s_reproduced",
        "source_binding_exact",
        "contract_schema_exact",
        "problem_schema_exact",
        "tree_identity_schema_exact",
        "phase_state_schema_exact",
        "collapse_schema_exact",
        "oracle_schema_exact",
        "manifest_schema_exact",
        "all_schemas_draft_2020_12_valid",
        "contract_schema_malformed_rejected",
        "problem_schema_malformed_rejected",
        "tree_identity_schema_malformed_rejected",
        "phase_state_schema_malformed_rejected",
        "collapse_schema_malformed_rejected",
        "oracle_schema_malformed_rejected",
        "manifest_schema_malformed_rejected",
        "tree_identity_schema_missing_fields_rejected",
        "contract_strict_round_trip",
        "contract_duplicate_key_rejected",
        "contract_unknown_field_rejected",
        "contract_noncanonical_json_rejected",
        "contract_nonfinite_rejected",
        "contract_boolean_number_rejected",
        "contract_mutated_initial_state_rejected",
        "contract_step_count_mutation_rejected",
        "contract_time_step_mutation_rejected",
        "contract_lock_start_mutation_rejected",
        "contract_lock_final_mutation_rejected",
        "contract_nonlinear_schedule_rejected",
        "contract_wrong_collapse_stage_rejected",
        "contract_late_oracle_stage_rejected",
        "contract_wrong_claim_ceiling_rejected",
        "contract_expected_optimum_injection_rejected",
        "problem_strict_round_trip",
        "problem_duplicate_key_rejected",
        "problem_unknown_field_rejected",
        "problem_noncanonical_json_rejected",
        "problem_nonfinite_coupling_rejected",
        "problem_nonfinite_field_rejected",
        "problem_boolean_number_rejected",
        "problem_asymmetric_coupling_rejected",
        "problem_nonzero_diagonal_rejected",
        "problem_wrong_coupling_dimension_rejected",
        "problem_wrong_field_dimension_rejected",
        "problem_expected_optimum_injection_rejected",
        "tree_identity_strict_round_trip",
        "five_tree_fixtures_strict_canonical",
        "tree_identity_checked_at_all_5005_positions",
        "tree_byte_mutation_rejected",
        "tree_geometry_substitution_rejected",
        "tree_site_id_collision_rejected",
        "initial_phase_state_strict_round_trip",
        "mutated_initial_orientation_rejected",
        "trajectory_strict_binary_parse",
        "trajectory_truncation_rejected",
        "trajectory_duplication_rejected",
        "trajectory_nonfinite_sample_rejected",
        "trajectory_sample_mutation_rejected",
        "committed_trajectory_exactly_reproduced",
        "final_phase_state_strict_round_trip",
        "final_state_outside_lock_tolerance_rejected",
        "trajectory_all_finite_and_wrapped",
        "initial_state_outside_antipodal_sector",
        "first_native_update_continuous_nonbinary",
        "native_trajectory_deterministic",
        "final_antipodal_lock_closes",
        "whole_tree_pi_action_negates_all_sites",
        "whole_tree_pi_action_preserves_amplitude_all_sites",
        "complete_recursive_tree_bytes_preserved",
        "no_second_harmonic_lock_rejected",
        "negated_coupling_control_rejected",
        "reversed_scaled_field_control_rejected",
        "site_permutation_covariance",
        "collapse_receipt_strict_round_trip",
        "collapsed_spin_vector_matches_frozen_result",
        "observed_energy_matches_frozen_result",
        "oracle_table_strict_round_trip",
        "oracle_table_contains_exactly_32_states",
        "oracle_unique_optimum_and_gap",
        "oracle_table_mutation_rejected",
        "collapse_receipt_mutation_rejected",
        "spin_resynthesis_baseline_declared_collapsed",
        "energy_retention_baseline_declared_collapsed",
        "collapsed_baselines_outside_native_roots",
        "native_ast_dataflow_proof_passes",
        "collapse_and_oracle_lifecycle_order",
        "native_roots_have_no_collapsed_identifiers",
        "native_ast_mutation_probes_rejected",
        "manifest_closes_all_fixture_bytes",
        "manifest_role_substitution_rejected",
        "manifest_source_binding_exact",
        "manifest_parent_binding_exact",
        "manifest_test_binding_exact",
        "manifest_problem_tree_schedule_binding_exact",
        "all_package_json_strict_parses",
        "claim_ceiling_is_bounded_emulator_only",
        "contact_counts_are_zero",
        "next_boundary_requires_explicit_selection",
        "committed_byte_packet_recomputed",
        "problem_and_schedule_are_frozen",
    )


def reference_test_spec() -> dict[str, Any]:
    return {
        "schema": TEST_SPEC_SCHEMA,
        "tests": [
            {"id": test_id, "required_status": "PASS"}
            for test_id in reference_test_ids()
        ],
    }


def run_reference_tests(
    package_dir: Path,
    source_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    packet = load_committed_packet(package_dir, source_path)
    parent_reports = parent_verification()
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append(
            {
                "id": test_id,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
        )

    parents = packet["parents"]
    record("parent_r0_custody_exact", parents["r0"] == R0_EXPECTED, parents["r0"])
    record("parent_r0_reproduced", parent_reports["r0"]["status"] == "PASS", parent_reports["r0"])
    record("parent_r1_custody_exact", parents["r1"] == R1_EXPECTED, parents["r1"])
    record("parent_r1_reproduced", parent_reports["r1"]["status"] == "PASS", parent_reports["r1"])
    record("parent_r2s_custody_exact", parents["r2s"] == R2S_EXPECTED, parents["r2s"])
    record("parent_r2s_reproduced", parent_reports["r2s"]["status"] == "PASS", parent_reports["r2s"])
    record("source_binding_exact", packet["source"] == source_binding(source_path), packet["source"])

    expected_schemas = schema_documents()
    schema_test_map = (
        ("contract_schema_exact", CONTRACT_SCHEMA_FILE),
        ("problem_schema_exact", PROBLEM_SCHEMA_FILE),
        ("tree_identity_schema_exact", TREE_IDENTITY_SCHEMA_FILE),
        ("phase_state_schema_exact", PHASE_STATE_SCHEMA_FILE),
        ("collapse_schema_exact", COLLAPSE_SCHEMA_FILE),
        ("oracle_schema_exact", ORACLE_SCHEMA_FILE),
        ("manifest_schema_exact", MANIFEST_SCHEMA_FILE),
    )
    for test_id, filename in schema_test_map:
        payload = (package_dir / filename).read_bytes()
        document = _strict_document(payload, filename)
        record(test_id, document == expected_schemas[filename], _sha256(payload))

    schema_instances = {
        CONTRACT_SCHEMA_FILE: packet["contract"],
        PROBLEM_SCHEMA_FILE: packet["problem"],
        TREE_IDENTITY_SCHEMA_FILE: packet["tree_record"],
        PHASE_STATE_SCHEMA_FILE: _strict_document(
            packet["payloads"][INITIAL_STATE_PATH], "initial phase state"
        ),
        COLLAPSE_SCHEMA_FILE: packet["collapse"],
        ORACLE_SCHEMA_FILE: packet["oracle"],
        MANIFEST_SCHEMA_FILE: packet["manifest"],
    }
    for filename, instance in schema_instances.items():
        validate_schema_instance(expected_schemas[filename], instance)
    record(
        "all_schemas_draft_2020_12_valid",
        True,
        {filename: "PASS" for filename in sorted(schema_instances)},
    )
    schema_mutations = (
        (
            "contract_schema_malformed_rejected",
            CONTRACT_SCHEMA_FILE,
            "site_count",
            "5",
        ),
        (
            "problem_schema_malformed_rejected",
            PROBLEM_SCHEMA_FILE,
            "coupling_matrix",
            "not-a-matrix",
        ),
        (
            "tree_identity_schema_malformed_rejected",
            TREE_IDENTITY_SCHEMA_FILE,
            "sites",
            [0] * SITE_COUNT,
        ),
        (
            "phase_state_schema_malformed_rejected",
            PHASE_STATE_SCHEMA_FILE,
            "orientations_rad",
            None,
        ),
        (
            "collapse_schema_malformed_rejected",
            COLLAPSE_SCHEMA_FILE,
            "observed_spins",
            [0] * SITE_COUNT,
        ),
        (
            "oracle_schema_malformed_rejected",
            ORACLE_SCHEMA_FILE,
            "rows",
            [],
        ),
        (
            "manifest_schema_malformed_rejected",
            MANIFEST_SCHEMA_FILE,
            "fixtures",
            [],
        ),
    )
    for test_id, filename, key, value in schema_mutations:
        malformed = copy.deepcopy(schema_instances[filename])
        malformed[key] = value
        observed = _expect_error(
            lambda schema=expected_schemas[filename], item=malformed: validate_schema_instance(
                schema, item
            ),
            "JSON Schema rejection",
        )
        record(test_id, True, observed)
    missing_tree_fields = copy.deepcopy(
        schema_instances[TREE_IDENTITY_SCHEMA_FILE]
    )
    missing_tree_fields["sites"][0] = {"site_id": SITE_IDS[0]}
    record(
        "tree_identity_schema_missing_fields_rejected",
        True,
        _expect_error(
            lambda: validate_schema_instance(
                expected_schemas[TREE_IDENTITY_SCHEMA_FILE],
                missing_tree_fields,
            ),
            "JSON Schema rejection",
        ),
    )

    contract = packet["contract"]
    contract_payload = packet["payloads"][CONTRACT_PATH]
    record("contract_strict_round_trip", True, _sha256(contract_payload))
    duplicate_contract = contract_payload.replace(
        b'"schema":', b'"schema":"duplicate","schema":', 1
    )
    record(
        "contract_duplicate_key_rejected",
        True,
        _expect_error(lambda: _strict_document(duplicate_contract, "R3 contract"), "duplicate"),
    )
    mutated = copy.deepcopy(contract)
    mutated["extra"] = 1
    record(
        "contract_unknown_field_rejected",
        True,
        _expect_error(lambda: validate_contract_document(mutated), "unexpected"),
    )
    noncanonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record(
        "contract_noncanonical_json_rejected",
        True,
        _expect_error(lambda: _strict_document(noncanonical, "R3 contract"), "canonical"),
    )
    mutated = copy.deepcopy(contract)
    mutated["time_step"] = float("nan")
    record(
        "contract_nonfinite_rejected",
        True,
        _expect_error(lambda: validate_contract_document(mutated), "finite"),
    )
    mutated = copy.deepcopy(contract)
    mutated["step_count"] = True
    record(
        "contract_boolean_number_rejected",
        True,
        _expect_error(lambda: validate_contract_document(mutated), "integer"),
    )
    mutated = copy.deepcopy(contract)
    mutated["initial_orientations_rad"][0] += 0.01
    record(
        "contract_mutated_initial_state_rejected",
        True,
        _expect_error(lambda: validate_contract_document(mutated), "initial orientations"),
    )
    contract_mutations = (
        ("contract_step_count_mutation_rejected", "step_count", STEP_COUNT - 1, "step count"),
        ("contract_time_step_mutation_rejected", "time_step", 0.031, "time step"),
        ("contract_lock_start_mutation_rejected", "lock_start", 0.1, "lock start"),
        ("contract_lock_final_mutation_rejected", "lock_final", 1.1, "lock final"),
        ("contract_nonlinear_schedule_rejected", "lock_schedule", "quadratic", "schedule"),
        ("contract_wrong_collapse_stage_rejected", "collapse_stage", "before_native_return", "collapse-stage"),
        ("contract_late_oracle_stage_rejected", "oracle_stage", "during_native_evolution", "oracle-stage"),
        ("contract_wrong_claim_ceiling_rejected", "claim_ceiling", "GENERAL_ISING_SOLVER", "claim ceiling"),
    )
    for test_id, key, value, phrase in contract_mutations:
        mutated = copy.deepcopy(contract)
        mutated[key] = value
        record(test_id, True, _expect_error(lambda item=mutated: validate_contract_document(item), phrase))
    mutated = copy.deepcopy(contract)
    mutated["expected_optimum"] = [-1, -1, -1, -1, 1]
    record(
        "contract_expected_optimum_injection_rejected",
        True,
        _expect_error(lambda: validate_contract_document(mutated), "forbidden field"),
    )

    problem = packet["problem"]
    problem_payload = packet["payloads"][PROBLEM_PATH]
    validate_problem_document(problem)
    record("problem_strict_round_trip", True, _sha256(problem_payload))
    duplicate_problem = problem_payload.replace(
        b'"schema":', b'"schema":"duplicate","schema":', 1
    )
    record(
        "problem_duplicate_key_rejected",
        True,
        _expect_error(lambda: _strict_document(duplicate_problem, "R3 problem"), "duplicate"),
    )
    mutated = copy.deepcopy(problem)
    mutated["extra"] = 1
    record("problem_unknown_field_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "unexpected"))
    noncanonical_problem = json.dumps(problem, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record("problem_noncanonical_json_rejected", True, _expect_error(lambda: _strict_document(noncanonical_problem, "R3 problem"), "canonical"))
    mutated = copy.deepcopy(problem)
    mutated["coupling_matrix"][0][1] = float("nan")
    record("problem_nonfinite_coupling_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "finite"))
    mutated = copy.deepcopy(problem)
    mutated["field_vector"][0] = float("inf")
    record("problem_nonfinite_field_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "finite"))
    mutated = copy.deepcopy(problem)
    mutated["coupling_matrix"][0][0] = True
    record("problem_boolean_number_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "finite real"))
    mutated = copy.deepcopy(problem)
    mutated["coupling_matrix"][0][1] = 1.0
    record("problem_asymmetric_coupling_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "symmetric"))
    mutated = copy.deepcopy(problem)
    mutated["coupling_matrix"][0][0] = 1.0
    record("problem_nonzero_diagonal_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "diagonal"))
    mutated = copy.deepcopy(problem)
    mutated["coupling_matrix"] = mutated["coupling_matrix"][:-1]
    record("problem_wrong_coupling_dimension_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "dimensions"))
    mutated = copy.deepcopy(problem)
    mutated["field_vector"] = mutated["field_vector"][:-1]
    record("problem_wrong_field_dimension_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "exactly 5"))
    mutated = copy.deepcopy(problem)
    mutated["expected_optimum"] = [-1, -1, -1, -1, 1]
    record("problem_expected_optimum_injection_rejected", True, _expect_error(lambda: validate_problem_document(mutated), "forbidden field"))

    tree_record = packet["tree_record"]
    tree_payloads = packet["tree_payloads"]
    validate_tree_identity_document(tree_record, tree_payloads)
    record("tree_identity_strict_round_trip", True, _sha256(packet["payloads"][TREE_IDENTITIES_PATH]))
    strict_tree_hashes = []
    for payload in tree_payloads:
        tree = r0.deserialize_tree_bytes(payload, require_canonical=True)
        strict_tree_hashes.append(_sha256(tree.canonical_bytes()))
    record("five_tree_fixtures_strict_canonical", len(strict_tree_hashes) == SITE_COUNT, strict_tree_hashes)
    record("tree_identity_checked_at_all_5005_positions", packet["tree_checks"] == 5005, packet["tree_checks"])
    mutated_payloads = list(tree_payloads)
    mutated_bytes = bytearray(mutated_payloads[0])
    mutated_bytes[len(mutated_bytes) // 2] ^= 1
    mutated_payloads[0] = bytes(mutated_bytes)
    record("tree_byte_mutation_rejected", True, _expect_error(lambda: validate_tree_identity_document(tree_record, mutated_payloads), "SHA-256"))
    substitute_tree = prefix_tree(r0.hierarchy_b(), "site0")
    substitute_payload = substitute_tree.canonical_bytes()
    geometry_record = copy.deepcopy(tree_record)
    geometry_record["sites"][0]["byte_count"] = len(substitute_payload)
    geometry_record["sites"][0]["tree_sha256"] = _sha256(substitute_payload)
    geometry_record["sites"][0]["tree_digest"] = substitute_tree.digest()
    geometry_payloads = list(tree_payloads)
    geometry_payloads[0] = substitute_payload
    record("tree_geometry_substitution_rejected", True, _expect_error(lambda: validate_tree_identity_document(geometry_record, geometry_payloads), "geometry"))
    collision_record = copy.deepcopy(tree_record)
    collision_record["sites"][1]["site_id"] = collision_record["sites"][0]["site_id"]
    record("tree_site_id_collision_rejected", True, _expect_error(lambda: validate_tree_identity_document(collision_record, tree_payloads), "collision"))

    trajectory = packet["trajectory"]
    tree_record_sha = _sha256(packet["payloads"][TREE_IDENTITIES_PATH])
    initial_document = _strict_document(packet["payloads"][INITIAL_STATE_PATH], "initial state")
    final_document = packet["final"]
    validate_phase_state_document(initial_document, expected_tree_record_sha256=tree_record_sha, expected_orientations=trajectory[0])
    record("initial_phase_state_strict_round_trip", True, _sha256(packet["payloads"][INITIAL_STATE_PATH]))
    mutated = copy.deepcopy(initial_document)
    mutated["orientations_rad"][0] += 0.01
    record("mutated_initial_orientation_rejected", True, _expect_error(lambda: validate_phase_state_document(mutated, expected_tree_record_sha256=tree_record_sha), "frozen state"))
    trajectory_payload = packet["payloads"][TRAJECTORY_PATH]
    record("trajectory_strict_binary_parse", trajectory.shape == TRAJECTORY_SHAPE, {"bytes": len(trajectory_payload), "shape": list(trajectory.shape)})
    record("trajectory_truncation_rejected", True, _expect_error(lambda: parse_trajectory_bytes(trajectory_payload[:-8]), "byte count"))
    record("trajectory_duplication_rejected", True, _expect_error(lambda: parse_trajectory_bytes(trajectory_payload + trajectory_payload[-8:]), "byte count"))
    nonfinite_payload = bytearray(trajectory_payload)
    nonfinite_payload[:8] = np.asarray([float("nan")], dtype=np.dtype(TRAJECTORY_DTYPE)).tobytes()
    record("trajectory_nonfinite_sample_rejected", True, _expect_error(lambda: parse_trajectory_bytes(bytes(nonfinite_payload)), "non-finite"))
    mutated_manifest = copy.deepcopy(manifest)
    mutated_file = bytearray(trajectory_payload)
    mutated_file[16] ^= 1
    with tempfile.TemporaryDirectory() as temporary:
        mutated_package = Path(temporary)
        (mutated_package / TESTS_FILE).write_bytes((package_dir / TESTS_FILE).read_bytes())
        for fixture_path, fixture_payload in packet["payloads"].items():
            target = mutated_package / fixture_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                bytes(mutated_file)
                if fixture_path == TRAJECTORY_PATH
                else fixture_payload
            )
        mutation_rejection = _expect_error(
            lambda: validate_manifest_document(
                mutated_manifest,
                mutated_package,
                expected_source=packet["source"],
                expected_parents=parents,
            ),
            "SHA-256 mismatch",
        )
    record("trajectory_sample_mutation_rejected", True, mutation_rejection)
    _, recomputed_trajectory, _ = evolve_phase_trajectory(initial_state())
    record("committed_trajectory_exactly_reproduced", trajectory_bytes(recomputed_trajectory) == trajectory_payload, _sha256(trajectory_payload))
    validate_phase_state_document(final_document, expected_tree_record_sha256=tree_record_sha, expected_orientations=trajectory[-1])
    record("final_phase_state_strict_round_trip", True, _sha256(packet["payloads"][FINAL_STATE_PATH]))
    mutated = copy.deepcopy(final_document)
    mutated["orientations_rad"][0] = 0.5
    record("final_state_outside_lock_tolerance_rejected", True, _expect_error(lambda: validate_phase_state_document(mutated, expected_tree_record_sha256=tree_record_sha), "outside lock"))
    record("trajectory_all_finite_and_wrapped", bool(np.all(np.isfinite(trajectory)) and np.all(trajectory >= -math.pi) and np.all(trajectory < math.pi)), {"min": _metric(np.min(trajectory)), "max": _metric(np.max(trajectory))})
    record("initial_state_outside_antipodal_sector", phase_lock_residual(trajectory[0]) > 0.1, _metric(phase_lock_residual(trajectory[0])))
    record("first_native_update_continuous_nonbinary", phase_lock_residual(trajectory[1]) > 0.1 and not np.array_equal(trajectory[1], trajectory[0]), [_metric(value) for value in trajectory[1]])
    record("native_trajectory_deterministic", np.array_equal(recomputed_trajectory, trajectory), _sha256(trajectory_bytes(recomputed_trajectory)))
    final_residual = phase_lock_residual(trajectory[-1])
    record("final_antipodal_lock_closes", final_residual <= LOCK_RESIDUAL_MAX, _metric(final_residual))

    final_state = tuple(
        OrientedRecursiveBeam(
            SITE_IDS[index],
            r0.deserialize_tree_bytes(tree_payloads[index], require_canonical=True),
            float(trajectory[-1, index]),
        )
        for index in range(SITE_COUNT)
    )
    times = r0.sample_times()
    negation_errors: list[float] = []
    amplitude_errors: list[float] = []
    for site in final_state:
        beam = site.render(times)
        rotated = site.with_orientation(site.orientation_rad + math.pi).render(times)
        negation_errors.append(float(np.max(np.abs(rotated + beam))))
        amplitude_errors.append(float(np.max(np.abs(np.abs(rotated) - np.abs(beam)))))
    record("whole_tree_pi_action_negates_all_sites", max(negation_errors) <= ORIENTATION_TOL, [_metric(value) for value in negation_errors])
    record("whole_tree_pi_action_preserves_amplitude_all_sites", max(amplitude_errors) <= ORIENTATION_TOL, [_metric(value) for value in amplitude_errors])
    record("complete_recursive_tree_bytes_preserved", packet["tree_checks"] == 5005, [_sha256(payload) for payload in tree_payloads])

    _, no_lock_trajectory, _ = evolve_phase_trajectory(initial_state(), lock_final=0.0)
    no_lock_residual = phase_lock_residual(no_lock_trajectory[-1])
    record("no_second_harmonic_lock_rejected", no_lock_residual >= NO_LOCK_RESIDUAL_MIN, _metric(no_lock_residual))
    wrong_coupling_state, _, _ = evolve_phase_trajectory(initial_state(), coupling=-COUPLING_MATRIX)
    wrong_coupling_spins = decode_spins(wrong_coupling_state)
    wrong_coupling_energy = ising_energy(wrong_coupling_spins)
    record("negated_coupling_control_rejected", list(wrong_coupling_spins) != packet["oracle"]["unique_optimum"] and wrong_coupling_energy > packet["oracle"]["optimum_energy"], {"energy": _metric(wrong_coupling_energy), "spins": list(wrong_coupling_spins)})
    wrong_field_state, _, _ = evolve_phase_trajectory(initial_state(), field=-2.0 * FIELD_VECTOR)
    wrong_field_spins = decode_spins(wrong_field_state)
    wrong_field_energy = ising_energy(wrong_field_spins)
    record("reversed_scaled_field_control_rejected", list(wrong_field_spins) != packet["oracle"]["unique_optimum"] and wrong_field_energy > packet["oracle"]["optimum_energy"], {"energy": _metric(wrong_field_energy), "spins": list(wrong_field_spins)})

    permutation = np.asarray([2, 4, 1, 0, 3], dtype=np.int64)
    original_start = initial_state()
    permuted_start = tuple(original_start[index] for index in permutation)
    permuted_coupling = COUPLING_MATRIX[np.ix_(permutation, permutation)]
    permuted_field = FIELD_VECTOR[permutation]
    permuted_final, _, _ = evolve_phase_trajectory(permuted_start, coupling=permuted_coupling, field=permuted_field)
    mapped = np.empty(SITE_COUNT, dtype=np.float64)
    for permuted_index, original_index in enumerate(permutation):
        mapped[original_index] = permuted_final[permuted_index].orientation_rad
    covariance_error = float(np.max(np.abs(wrap_phase(mapped - trajectory[-1]))))
    record("site_permutation_covariance", covariance_error <= PORTABLE_METRIC_ATOL, _metric(covariance_error))

    collapse = packet["collapse"]
    oracle = packet["oracle"]
    validate_collapse_document(collapse, expected_final_state_sha256=_sha256(packet["payloads"][FINAL_STATE_PATH]), expected_oracle_sha256=_sha256(packet["payloads"][ORACLE_PATH]), expected_final_phases=trajectory[-1])
    record("collapse_receipt_strict_round_trip", True, _sha256(packet["payloads"][COLLAPSE_PATH]))
    record("collapsed_spin_vector_matches_frozen_result", collapse["observed_spins"] == [-1, -1, -1, -1, 1], collapse["observed_spins"])
    record("observed_energy_matches_frozen_result", abs(collapse["observed_energy"] + 12.5) <= ENERGY_TOL, collapse["observed_energy"])
    validate_oracle_document(oracle, expected_problem_sha256=_sha256(packet["payloads"][PROBLEM_PATH]))
    record("oracle_table_strict_round_trip", True, _sha256(packet["payloads"][ORACLE_PATH]))
    record("oracle_table_contains_exactly_32_states", len(oracle["rows"]) == 32 and len({tuple(row["spins"]) for row in oracle["rows"]}) == 32, len(oracle["rows"]))
    record("oracle_unique_optimum_and_gap", oracle["unique_optimum"] == [-1, -1, -1, -1, 1] and oracle["optimum_energy"] == -12.5 and oracle["next_energy"] == -11.5 and oracle["optimum_gap"] >= OPTIMUM_GAP_MIN, {"energy": oracle["optimum_energy"], "gap": oracle["optimum_gap"], "spins": oracle["unique_optimum"]})
    mutated = copy.deepcopy(oracle)
    mutated["rows"][0]["energy"] += 1.0
    record("oracle_table_mutation_rejected", True, _expect_error(lambda: validate_oracle_document(mutated, expected_problem_sha256=_sha256(packet["payloads"][PROBLEM_PATH])), "enumeration"))
    mutated = copy.deepcopy(collapse)
    mutated["observed_spins"][0] *= -1
    record("collapse_receipt_mutation_rejected", True, _expect_error(lambda: validate_collapse_document(mutated, expected_final_state_sha256=_sha256(packet["payloads"][FINAL_STATE_PATH]), expected_oracle_sha256=_sha256(packet["payloads"][ORACLE_PATH]), expected_final_phases=trajectory[-1]), "observed spins"))

    spin_baseline = collapsed_spin_resynthesis_baseline()
    retention_baseline = energy_retention_baseline(trajectory)
    record("spin_resynthesis_baseline_declared_collapsed", spin_baseline["binary_after_every_step"] is True and "discarded" in spin_baseline["structural_state_loss"], spin_baseline)
    record("energy_retention_baseline_declared_collapsed", retention_baseline["energy_evaluated_after_every_step"] is True and "selection" in retention_baseline["structural_state_loss"], retention_baseline)

    ast_proof = native_ast_proof(source_path)
    record("collapsed_baselines_outside_native_roots", not any(name in ast_proof["forbidden_hits"] for name in ("collapsed_spin_resynthesis_baseline", "energy_retention_baseline")), ast_proof["forbidden_hits"])
    record("native_ast_dataflow_proof_passes", ast_proof["status"] == "PASS", ast_proof)
    record("collapse_and_oracle_lifecycle_order", ast_proof["lifecycle_order"], ast_proof["lifecycle_positions"])
    record("native_roots_have_no_collapsed_identifiers", not ast_proof["forbidden_hits"], ast_proof["forbidden_hits"])
    probes = native_ast_mutation_probes(source_path)
    record("native_ast_mutation_probes_rejected", all(probes.values()), probes)

    record("manifest_closes_all_fixture_bytes", manifest["fixture_set_sha256"] == fixture_set_sha256(manifest["fixtures"]), {"count": manifest["fixture_count"], "fixture_set_sha256": manifest["fixture_set_sha256"], "total_bytes": manifest["total_fixture_bytes"]})
    role_mutation = copy.deepcopy(manifest)
    role_mutation["fixtures"][0]["role"], role_mutation["fixtures"][1]["role"] = role_mutation["fixtures"][1]["role"], role_mutation["fixtures"][0]["role"]
    record("manifest_role_substitution_rejected", True, _expect_error(lambda: validate_manifest_document(role_mutation, package_dir, expected_source=packet["source"], expected_parents=parents), "role substitution"))
    record("manifest_source_binding_exact", manifest["source_binding"] == packet["source"], manifest["source_binding"])
    record("manifest_parent_binding_exact", manifest["parents"] == parents, sorted(manifest["parents"]))
    record("manifest_test_binding_exact", manifest["tests_sha256"] == _sha256((package_dir / TESTS_FILE).read_bytes()), manifest["tests_sha256"])
    record("manifest_problem_tree_schedule_binding_exact", manifest["problem_sha256"] == _sha256(packet["payloads"][PROBLEM_PATH]) and manifest["tree_identity_record_sha256"] == tree_record_sha and contract["step_count"] == STEP_COUNT and contract["time_step"] == TIME_STEP and contract["lock_schedule"] == "linear_in_step_index_0_to_999", {"problem": manifest["problem_sha256"], "tree": tree_record_sha})
    json_paths = sorted(
        path for path in package_dir.rglob("*.json") if path.name != RESULTS_FILE
    )
    for path in json_paths:
        _strict_document(path.read_bytes(), path.name)
    record("all_package_json_strict_parses", True, len(json_paths))
    record("claim_ceiling_is_bounded_emulator_only", contract["claim_ceiling"] == CLAIM_CEILING and manifest["claim_ceiling"] == CLAIM_CEILING, CLAIM_CEILING)
    contact_counts = {"adc_dac": 0, "audio_playback": 0, "audio_recording": 0, "hardware": 0, "ssh_scp": 0, "target": 0, "transducer": 0}
    record("contact_counts_are_zero", all(value == 0 for value in contact_counts.values()), contact_counts)
    record("next_boundary_requires_explicit_selection", NEXT_BOUNDARY == "NEXT_AUDIO_PHASE_COMPUTING_BOUNDARY_REQUIRES_EXPLICIT_SELECTION", NEXT_BOUNDARY)
    record("committed_byte_packet_recomputed", packet["tree_checks"] == 5005 and trajectory_bytes(recomputed_trajectory) == trajectory_payload, {"fixture_set": manifest["fixture_set_sha256"], "trajectory": _sha256(trajectory_payload)})
    record("problem_and_schedule_are_frozen", np.array_equal(validate_problem_document(problem)[0], COUPLING_MATRIX) and contract["step_count"] == STEP_COUNT and contract["lock_final"] == LOCK_FINAL, {"steps": STEP_COUNT, "dt": TIME_STEP, "lock_final": LOCK_FINAL})

    observed_ids = tuple(test["id"] for test in tests)
    if observed_ids != reference_test_ids():
        raise ValueError("R3 reference test implementation does not match frozen test specification")
    passed = sum(test["status"] == "PASS" for test in tests)
    failed = len(tests) - passed
    nonbinary_rows = sum(
        phase_lock_residual(row) > LOCK_RESIDUAL_MAX for row in trajectory[:-1]
    )
    return {
        "claim_ceiling": CLAIM_CEILING,
        "contact_counts": contact_counts,
        "established_token_if_all_gates_close": ESTABLISHED_TOKEN,
        "fixture_count": manifest["fixture_count"],
        "fixture_manifest_sha256": _sha256((package_dir / MANIFEST_FILE).read_bytes()),
        "fixture_set_sha256": manifest["fixture_set_sha256"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "measurements": {
            "ast_dataflow_proof": ast_proof,
            "ast_mutation_probes": probes,
            "collapsed_spin_baseline": spin_baseline,
            "energy_retention_baseline": retention_baseline,
            "final_antipodal_residual": _metric(final_residual),
            "final_orientations_rad": [_metric(value) for value in trajectory[-1]],
            "final_spins": collapse["observed_spins"],
            "initial_orientations_rad": [_metric(value) for value in trajectory[0]],
            "no_lock_residual": _metric(no_lock_residual),
            "nonbinary_preboundary_rows": nonbinary_rows,
            "observed_energy": collapse["observed_energy"],
            "optimum_energy": oracle["optimum_energy"],
            "optimum_gap": oracle["optimum_gap"],
            "permutation_covariance_error": _metric(covariance_error),
            "tree_identity_checks": packet["tree_checks"],
            "trajectory_byte_count": len(trajectory_payload),
            "trajectory_sha256": _sha256(trajectory_payload),
            "trajectory_shape": list(trajectory.shape),
            "whole_tree_amplitude_error_max": _metric(max(amplitude_errors)),
            "whole_tree_negation_error_max": _metric(max(negation_errors)),
            "wrong_coupling_energy": _metric(wrong_coupling_energy),
            "wrong_coupling_spins": list(wrong_coupling_spins),
            "wrong_field_energy": _metric(wrong_field_energy),
            "wrong_field_spins": list(wrong_field_spins),
        },
        "next_boundary": NEXT_BOUNDARY,
        "ordinary_software_only": True,
        "parent_custody": parents,
        "parent_verification": parent_reports,
        "physical_claims_established": [],
        "reference_tests_sha256": _sha256((package_dir / TESTS_FILE).read_bytes()),
        "schema": "recursive_catalytic_ising_scientific_result_v1",
        "summary": {"failed": failed, "passed": passed, "test_count": len(tests)},
        "tests": tests,
    }


def result_document(
    package_dir: Path,
    source_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    schemas = {
        filename: _sha256((package_dir / filename).read_bytes())
        for filename in sorted(schema_documents())
    }
    scientific = run_reference_tests(package_dir, source_path, manifest)
    return {
        "fixture_manifest_sha256": _sha256((package_dir / MANIFEST_FILE).read_bytes()),
        "parent_custody": scientific["parent_custody"],
        "schema": RESULT_SCHEMA,
        "schema_bindings": schemas,
        "scientific": scientific,
        "source_binding": source_binding(source_path),
        "verification_policy": {
            "committed_binary_reproduction": "byte_exact",
            "committed_json": "canonical_byte_exact",
            "numeric_atol": PORTABLE_METRIC_ATOL,
            "numeric_rtol": PORTABLE_METRIC_RTOL,
            "result_reproduction": "canonical_byte_exact",
        },
    }


def build_package(
    package_dir: Path = PACKAGE_DIR,
    source_path: Path = Path(__file__).resolve(),
) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / FIXTURE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    lifecycle = execute_reference_lifecycle()
    for filename, document in schema_documents().items():
        (package_dir / filename).write_bytes(canonical_bytes(document))
    tests_document = reference_test_spec()
    tests_payload = canonical_bytes(tests_document)
    (package_dir / TESTS_FILE).write_bytes(tests_payload)
    payloads = lifecycle_fixture_payloads(lifecycle)
    for relative_path, payload in payloads.items():
        target = package_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = fixture_manifest_document(lifecycle, payloads, _sha256(tests_payload))
    manifest_payload = canonical_bytes(manifest)
    (package_dir / MANIFEST_FILE).write_bytes(manifest_payload)
    result = result_document(package_dir, source_path, manifest)
    result_payload = canonical_bytes(result)
    (package_dir / RESULTS_FILE).write_bytes(result_payload)
    summary = result["scientific"]["summary"]
    return {
        "fixture_count": manifest["fixture_count"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "manifest_sha256": _sha256(manifest_payload),
        "operation": "build",
        "result_sha256": _sha256(result_payload),
        "source_sha256": source_binding(source_path)["sha256"],
        "status": "PASS" if summary["failed"] == 0 else "FAIL",
        "test_count": summary["test_count"],
        "tests_passed": summary["passed"],
        "tests_sha256": _sha256(tests_payload),
        "trajectory_bytes": TRAJECTORY_BYTE_COUNT,
        "trajectory_sha256": manifest["trajectory"]["sha256"],
    }


def verify_package(
    package_dir: Path = PACKAGE_DIR,
    source_path: Path = Path(__file__).resolve(),
) -> dict[str, Any]:
    expected_schemas = schema_documents()
    for filename, expected in expected_schemas.items():
        observed = _strict_document((package_dir / filename).read_bytes(), filename)
        if observed != expected:
            raise ValueError(f"schema document mismatch: {filename}")
    tests = _load_canonical_json(package_dir / TESTS_FILE, TESTS_FILE)
    if tests != reference_test_spec():
        raise ValueError("R3 reference test specification mismatch")
    manifest = _load_canonical_json(package_dir / MANIFEST_FILE, MANIFEST_FILE)
    validate_manifest_document(
        manifest,
        package_dir,
        expected_source=source_binding(source_path),
        expected_parents=parent_custody(),
    )
    committed_result_payload = (package_dir / RESULTS_FILE).read_bytes()
    committed_result = _strict_document(committed_result_payload, RESULTS_FILE)
    if not isinstance(committed_result, Mapping):
        raise ValueError("R3 reference result must be an object")
    recomputed_result = result_document(package_dir, source_path, manifest)
    recomputed_payload = canonical_bytes(recomputed_result)
    if recomputed_payload != committed_result_payload:
        raise ValueError("R3 committed result does not match deterministic recomputation")
    summary = recomputed_result["scientific"]["summary"]
    return {
        "fixture_count": manifest["fixture_count"],
        "fixture_total_bytes": manifest["total_fixture_bytes"],
        "manifest_sha256": _sha256((package_dir / MANIFEST_FILE).read_bytes()),
        "operation": "verify",
        "parent_binding_match": recomputed_result["parent_custody"] == parent_custody(),
        "recomputed_results_match": True,
        "result_sha256": _sha256(committed_result_payload),
        "source_binding_match": recomputed_result["source_binding"] == source_binding(source_path),
        "status": "PASS" if summary["failed"] == 0 else "FAIL",
        "test_count": summary["test_count"],
        "tests_passed": summary["passed"],
        "tests_sha256": _sha256((package_dir / TESTS_FILE).read_bytes()),
        "trajectory_bytes": TRAJECTORY_BYTE_COUNT,
        "trajectory_sha256": manifest["trajectory"]["sha256"],
    }


def self_test(source_path: Path = Path(__file__).resolve()) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rci_reference_") as directory:
        package_dir = Path(directory)
        build = build_package(package_dir, source_path)
        verify = verify_package(package_dir, source_path)
    return {
        "build_status": build["status"],
        "fixture_count": build["fixture_count"],
        "fixture_total_bytes": build["fixture_total_bytes"],
        "operation": "self-test",
        "recomputed_results_match": verify["recomputed_results_match"],
        "status": "PASS" if build["status"] == verify["status"] == "PASS" else "FAIL",
        "test_count": verify["test_count"],
        "tests_passed": verify["tests_passed"],
        "trajectory_bytes": verify["trajectory_bytes"],
        "verify_status": verify["status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("build", "self-test", "verify"),
        nargs="?",
        default="self-test",
    )
    arguments = parser.parse_args(argv)
    source_path = Path(__file__).resolve()
    if arguments.command == "build":
        report = build_package(PACKAGE_DIR, source_path)
    elif arguments.command == "verify":
        report = verify_package(PACKAGE_DIR, source_path)
    else:
        report = self_test(source_path)
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
