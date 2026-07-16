#!/usr/bin/env python3
"""Deterministic reference for recursive phase-inside-phase CAT_CAS beams.

This is an ordinary-software, non-hardware reference.  It establishes only that a
nested phase tree can be represented, rendered as a unit-modulus complex beam,
globally Z2-rotated without flattening its internal tree, used as a reversible
pointwise phase operator on a borrowed complex tape, queried by a hierarchy-
sensitive matched beam, and uncomputed.

It does not establish physical audio computing, physical restoration, an Ising
optimizer, an advantage claim, or a Small Wall crossing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


SAMPLE_RATE_HZ = 48_000
FRAME_SECONDS = 0.125
SAMPLE_COUNT = int(SAMPLE_RATE_HZ * FRAME_SECONDS)
ABS_TOL = 1e-12
GENERATOR_ID = "recursive_phase_tree_reference_v1"
CLAIM_CEILING = "SOFTWARE_RECURSIVE_PHASE_TREE_REFERENCE_ONLY"


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


@dataclass(frozen=True)
class PhaseEdge:
    modulation_index: float
    child: "PhaseNode"

    def __post_init__(self) -> None:
        beta = _finite(self.modulation_index, "modulation_index")
        if beta < 0.0:
            raise ValueError("modulation_index must be nonnegative")
        object.__setattr__(self, "modulation_index", beta)

    def canonical(self) -> dict[str, Any]:
        return {
            "modulation_index": self.modulation_index,
            "child": self.child.canonical(),
        }


@dataclass(frozen=True)
class PhaseNode:
    node_id: str
    frequency_hz: float
    phase_rad: float = 0.0
    children: tuple[PhaseEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id or not isinstance(self.node_id, str):
            raise ValueError("node_id must be a nonempty string")
        frequency = _finite(self.frequency_hz, "frequency_hz")
        phase = _finite(self.phase_rad, "phase_rad")
        if frequency <= 0.0 or frequency >= SAMPLE_RATE_HZ / 2:
            raise ValueError("frequency_hz must be inside (0, Nyquist)")
        child_ids = [edge.child.node_id for edge in self.children]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError(f"duplicate direct child IDs under {self.node_id}")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "phase_rad", phase)

    def phase(self, t: np.ndarray) -> np.ndarray:
        """Return recursively nested phase, preserving every child relation."""

        phi = (2.0 * math.pi * self.frequency_hz * t) + self.phase_rad
        for edge in self.children:
            phi = phi + edge.modulation_index * np.sin(edge.child.phase(t))
        return phi

    def canonical(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "frequency_hz": self.frequency_hz,
            "phase_rad": self.phase_rad,
            "children": [edge.canonical() for edge in self.children],
        }

    def walk(self) -> Iterable["PhaseNode"]:
        yield self
        for edge in self.children:
            yield from edge.child.walk()

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(edge.child.max_depth() for edge in self.children)


@dataclass(frozen=True)
class RecursivePhaseBeam:
    root: PhaseNode
    global_spin_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        phase = _finite(self.global_spin_phase_rad, "global_spin_phase_rad")
        object.__setattr__(self, "global_spin_phase_rad", phase)

    def render(self, t: np.ndarray) -> np.ndarray:
        return np.exp(1j * (self.root.phase(t) + self.global_spin_phase_rad))

    def canonical(self) -> dict[str, Any]:
        return {
            "root": self.root.canonical(),
            "global_spin_phase_rad": self.global_spin_phase_rad,
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self.canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def sample_times() -> np.ndarray:
    return np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ


def borrowed_tape(t: np.ndarray) -> np.ndarray:
    """Deterministic dirty complex state unrelated to either phase tree."""

    n = np.arange(t.size, dtype=np.float64)
    phase = 0.173 * n + 0.23 * np.sin(2.0 * math.pi * 41.0 * t)
    amplitude = 0.65 + 0.25 * np.cos(2.0 * math.pi * 29.0 * t)
    return amplitude * np.exp(1j * phase)


def apply_phase_operator(tape: np.ndarray, beam: np.ndarray) -> np.ndarray:
    if tape.shape != beam.shape:
        raise ValueError("tape and beam must have identical shapes")
    if not np.all(np.isfinite(tape)) or not np.all(np.isfinite(beam)):
        raise ValueError("operator inputs must be finite")
    return tape * beam


def uncompute_phase_operator(mutated: np.ndarray, beam: np.ndarray) -> np.ndarray:
    return apply_phase_operator(mutated, np.conjugate(beam))


def matched_response(state_beam: np.ndarray, query_beam: np.ndarray) -> complex:
    if state_beam.shape != query_beam.shape:
        raise ValueError("state and query beams must have identical shapes")
    return complex(np.mean(np.conjugate(query_beam) * state_beam))


def phase_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(a * np.conjugate(b))


def hierarchy_a() -> RecursivePhaseBeam:
    leaf = PhaseNode("leaf", 47.0, phase_rad=0.31)
    middle = PhaseNode(
        "middle",
        233.0,
        phase_rad=-0.22,
        children=(PhaseEdge(0.37, leaf),),
    )
    root = PhaseNode(
        "root",
        1_237.0,
        phase_rad=0.17,
        children=(PhaseEdge(0.71, middle),),
    )
    return RecursivePhaseBeam(root=root)


def hierarchy_b() -> RecursivePhaseBeam:
    middle = PhaseNode("middle", 233.0, phase_rad=-0.22)
    leaf = PhaseNode(
        "leaf",
        47.0,
        phase_rad=0.31,
        children=(PhaseEdge(0.37, middle),),
    )
    root = PhaseNode(
        "root",
        1_237.0,
        phase_rad=0.17,
        children=(PhaseEdge(0.71, leaf),),
    )
    return RecursivePhaseBeam(root=root)


def _metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def run_self_test() -> dict[str, Any]:
    t = sample_times()
    tree_a = hierarchy_a()
    tree_b = hierarchy_b()
    beam_a = tree_a.render(t)
    beam_b = tree_b.render(t)
    beam_a_minus = RecursivePhaseBeam(
        root=tree_a.root,
        global_spin_phase_rad=math.pi,
    ).render(t)

    tape = borrowed_tape(t)
    mutated = apply_phase_operator(tape, beam_a)
    restored = uncompute_phase_operator(mutated, beam_a)
    wrong_restored = uncompute_phase_operator(mutated, beam_b)

    self_response = matched_response(beam_a, beam_a)
    cross_response = matched_response(beam_a, beam_b)

    node_multiset_a = sorted(
        (node.node_id, node.frequency_hz, node.phase_rad) for node in tree_a.root.walk()
    )
    node_multiset_b = sorted(
        (node.node_id, node.frequency_hz, node.phase_rad) for node in tree_b.root.walk()
    )

    measurements = {
        "tree_a_digest": tree_a.digest(),
        "tree_b_digest": tree_b.digest(),
        "tree_a_depth": tree_a.root.max_depth(),
        "tree_b_depth": tree_b.root.max_depth(),
        "node_multisets_equal": node_multiset_a == node_multiset_b,
        "max_unit_modulus_error": _metric(np.max(np.abs(np.abs(beam_a) - 1.0))),
        "max_z2_whole_beam_error": _metric(np.max(np.abs(beam_a_minus + beam_a))),
        "max_amplitude_only_difference": _metric(
            np.max(np.abs(np.abs(beam_a) - np.abs(beam_b)))
        ),
        "max_hierarchy_phase_difference_rad": _metric(
            np.max(np.abs(phase_error(beam_a, beam_b)))
        ),
        "self_query_magnitude": _metric(abs(self_response)),
        "cross_query_magnitude": _metric(abs(cross_response)),
        "forward_mutation_l2": _metric(np.linalg.norm(mutated - tape)),
        "correct_inverse_max_error": _metric(np.max(np.abs(restored - tape))),
        "wrong_inverse_max_error": _metric(np.max(np.abs(wrong_restored - tape))),
    }

    tests = [
        {
            "id": "recursive_depth_present",
            "status": "PASS"
            if measurements["tree_a_depth"] >= 3 and measurements["tree_b_depth"] >= 3
            else "FAIL",
            "observed": [measurements["tree_a_depth"], measurements["tree_b_depth"]],
            "requirement": "both hierarchies have depth >= 3",
        },
        {
            "id": "same_node_multiset_different_parent_child_geometry",
            "status": "PASS" if measurements["node_multisets_equal"] else "FAIL",
            "observed": measurements["node_multisets_equal"],
            "requirement": True,
        },
        {
            "id": "unit_modulus_phase_carrier",
            "status": "PASS"
            if measurements["max_unit_modulus_error"] <= ABS_TOL
            else "FAIL",
            "observed": measurements["max_unit_modulus_error"],
            "tolerance": ABS_TOL,
        },
        {
            "id": "global_z2_rotates_complete_tree",
            "status": "PASS"
            if measurements["max_z2_whole_beam_error"] <= ABS_TOL
            else "FAIL",
            "observed": measurements["max_z2_whole_beam_error"],
            "tolerance": ABS_TOL,
        },
        {
            "id": "amplitude_only_control_is_exact_null",
            "status": "PASS"
            if measurements["max_amplitude_only_difference"] <= ABS_TOL
            else "FAIL",
            "observed": measurements["max_amplitude_only_difference"],
            "tolerance": ABS_TOL,
        },
        {
            "id": "hierarchy_changes_phase_geometry",
            "status": "PASS"
            if measurements["max_hierarchy_phase_difference_rad"] >= 0.10
            else "FAIL",
            "observed": measurements["max_hierarchy_phase_difference_rad"],
            "minimum": 0.10,
        },
        {
            "id": "matched_query_prefers_exact_hierarchy",
            "status": "PASS"
            if measurements["self_query_magnitude"] >= 1.0 - ABS_TOL
            and measurements["cross_query_magnitude"] <= 0.98
            else "FAIL",
            "observed": {
                "self": measurements["self_query_magnitude"],
                "cross": measurements["cross_query_magnitude"],
            },
            "requirement": "self >= 1-tol and cross <= 0.98",
        },
        {
            "id": "borrowed_tape_is_actually_mutated",
            "status": "PASS"
            if measurements["forward_mutation_l2"] >= 1.0
            else "FAIL",
            "observed": measurements["forward_mutation_l2"],
            "minimum": 1.0,
        },
        {
            "id": "correct_inverse_restores",
            "status": "PASS"
            if measurements["correct_inverse_max_error"] <= ABS_TOL
            else "FAIL",
            "observed": measurements["correct_inverse_max_error"],
            "tolerance": ABS_TOL,
        },
        {
            "id": "wrong_hierarchy_inverse_fails",
            "status": "PASS"
            if measurements["wrong_inverse_max_error"] >= 0.05
            else "FAIL",
            "observed": measurements["wrong_inverse_max_error"],
            "minimum": 0.05,
        },
        {
            "id": "canonical_identity_is_deterministic",
            "status": "PASS"
            if tree_a.digest() == tree_a.digest()
            and tree_b.digest() == tree_b.digest()
            and tree_a.digest() != tree_b.digest()
            else "FAIL",
            "observed": {
                "tree_a": tree_a.digest(),
                "tree_b": tree_b.digest(),
            },
            "requirement": "stable per tree and distinct across hierarchies",
        },
    ]

    passed = sum(test["status"] == "PASS" for test in tests)
    result = {
        "schema": "recursive_phase_tree_reference_result_v1",
        "generator": GENERATOR_ID,
        "claim_ceiling": CLAIM_CEILING,
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "collapse_boundary": "diagnostic readout only; no decoded scalar feeds recurrence",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frame_seconds": FRAME_SECONDS,
        "sample_count": SAMPLE_COUNT,
        "measurements": measurements,
        "tests": tests,
        "summary": {
            "passed": passed,
            "failed": len(tests) - passed,
            "test_count": len(tests),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("self-test", "emit-tree-a", "emit-tree-b"),
        nargs="?",
        default="self-test",
    )
    args = parser.parse_args()

    if args.operation == "emit-tree-a":
        payload: Any = hierarchy_a().canonical()
    elif args.operation == "emit-tree-b":
        payload = hierarchy_b().canonical()
    else:
        payload = run_self_test()

    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    if args.operation == "self-test":
        return 0 if payload["summary"]["failed"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
