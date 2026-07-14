#!/usr/bin/env python3
"""Offline reference model for query-separated identifiability.

This file is pure synthetic analysis. It contains no target, PMU, SSH, SCP,
transport, controller, or live-authority code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


DECISION = "QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED"


@dataclass(frozen=True)
class Relation:
    a: int
    b: int
    N: int

    @property
    def unordered(self) -> Tuple[int, int]:
        return tuple(sorted((self.a, self.b)))


def branch_response(value: Optional[int], q: int) -> float:
    if value is None:
        return 0.0
    return float(((value * (q + 3)) % 97) - 48)


def joint_term(a: int, b: int, q: int) -> float:
    return float((((a * b) + (11 * q) + 19) % 89) - 44)


def additive_branch_mechanism(a: Optional[int], b: Optional[int], q: int) -> float:
    return branch_response(a, q) + branch_response(b, q)


def synthetic_nonadditive_pair_mechanism(a: Optional[int], b: Optional[int], q: int) -> float:
    base = additive_branch_mechanism(a, b, q)
    if a is None or b is None:
        return base
    return base + joint_term(a, b, q)


def joint_observable(model, relation: Relation, q: int) -> float:
    a, b = relation.unordered
    return (
        model(a, b, q)
        - model(a, None, q)
        - model(None, b, q)
        + model(None, None, q)
    )


class FiniteAnswerCache:
    def __init__(self, relation: Relation, queries: Iterable[int]):
        self.table = {
            q: synthetic_nonadditive_pair_mechanism(*relation.unordered, q)
            for q in queries
        }

    def predict(self, q: int) -> float:
        return self.table[q]


class BoundedAnswerCache(FiniteAnswerCache):
    def __init__(self, relation: Relation, queries: Iterable[int], capacity: int):
        stored = list(queries)[:capacity]
        super().__init__(relation, stored)
        self.capacity = capacity

    def can_answer(self, q: int) -> bool:
        return q in self.table


def value_orientation_trace(relation: Relation, permutation: str) -> Tuple[int, int]:
    """A bad physical trace that derives lane assignment from numeric value."""
    lo, hi = relation.unordered
    if permutation not in {"identity", "swap"}:
        raise ValueError(f"unknown permutation {permutation!r}")
    # The trace ignores the blind permutation. That is the leakage.
    return (lo, hi)


def blinded_permutation_trace(relation: Relation, permutation: str) -> Tuple[int, int]:
    a, b = relation.unordered
    if permutation == "identity":
        return (a, b)
    if permutation == "swap":
        return (b, a)
    raise ValueError(f"unknown permutation {permutation!r}")


def route_bank_order_artifact(relation: Relation, q: int) -> float:
    """Ordinary metadata artifact: response keyed by route/order, not relation state."""
    lo, hi = relation.unordered
    route_id = (lo + 2 * hi + q) % 5
    order_id = (hi - lo + q) % 7
    return float((route_id * 13) - (order_id * 5))


def compressed_answer_generator(relation: Relation, q: int) -> float:
    """Compact ordinary generator for the synthetic response law.

    This models the reviewer's point that a raw response table is not a lower bound:
    an ordinary formula can reproduce every answer on held-out queries.
    """
    return synthetic_nonadditive_pair_mechanism(*relation.unordered, q)


def ordinary_nonlinear_pair_readout(a: Optional[int], b: Optional[int], q: int) -> float:
    """Nonrelational nonlinear readout that can produce nonzero J_q."""
    fa = branch_response(a, q)
    gb = branch_response(b, q)
    return (fa + gb) ** 2


def query_preselection_valid(source_closed: bool, query_generated_after_close: bool) -> bool:
    return bool(source_closed and query_generated_after_close)


def run_tests(spec: Mapping[str, object]) -> Dict[str, object]:
    rel_spec = spec["relation"]
    if not isinstance(rel_spec, Mapping):
        raise TypeError("relation must be an object")
    relation = Relation(int(rel_spec["a"]), int(rel_spec["b"]), int(rel_spec["N"]))
    closed_queries = [int(q) for q in spec["closed_queries"]]  # type: ignore[index]
    held_out_query = int(spec["held_out_query"])
    capacity = int(spec["bounded_cache_capacity"])
    tolerance = float(spec["tolerance"])

    additive_j = [joint_observable(additive_branch_mechanism, relation, q) for q in closed_queries]
    joint_j = [
        joint_observable(synthetic_nonadditive_pair_mechanism, relation, q)
        for q in closed_queries
    ]
    ordinary_nonlinear_j = [
        joint_observable(ordinary_nonlinear_pair_readout, relation, q)
        for q in closed_queries
    ]

    finite_cache = FiniteAnswerCache(relation, closed_queries)
    closed_cache_errors = [
        abs(finite_cache.predict(q) - synthetic_nonadditive_pair_mechanism(*relation.unordered, q))
        for q in closed_queries
    ]
    compressed_generator_errors = [
        abs(compressed_answer_generator(relation, q) - synthetic_nonadditive_pair_mechanism(*relation.unordered, q))
        for q in [*closed_queries, held_out_query]
    ]

    bounded_cache = BoundedAnswerCache(relation, closed_queries, capacity)
    bounded_fails_held_out = not bounded_cache.can_answer(held_out_query)
    bounded_fails_capacity = len(closed_queries) > capacity

    value_trace_identity = value_orientation_trace(relation, "identity")
    value_trace_swap = value_orientation_trace(relation, "swap")
    blind_trace_identity = blinded_permutation_trace(relation, "identity")
    blind_trace_swap = blinded_permutation_trace(relation, "swap")

    results: Dict[str, object] = {
        "additive_joint_values": additive_j,
        "synthetic_joint_values": joint_j,
        "ordinary_nonlinear_joint_values": ordinary_nonlinear_j,
        "additive_model_fails_joint_state_gate": all(abs(v) <= tolerance for v in additive_j),
        "value_orientation_attack_fails_blinded_permutation": (
            value_trace_identity == value_trace_swap
            and blind_trace_identity != blind_trace_swap
        ),
        "unbounded_finite_lookup_indistinguishable_on_closed_set": all(
            err <= tolerance for err in closed_cache_errors
        ),
        "adequate_answer_cache_passes_closed_set": all(
            err <= tolerance for err in closed_cache_errors
        ),
        "bounded_answer_cache_fails_held_out_query_or_capacity_gate": (
            bounded_fails_held_out or bounded_fails_capacity
        ),
        "compressed_answer_generator_not_rejected": all(
            err <= tolerance for err in compressed_generator_errors
        ),
        "ordinary_nonlinear_pair_false_positive_possible": any(
            abs(v) > tolerance for v in ordinary_nonlinear_j
        ),
        "synthetic_joint_witness_passes_prospective_joint_gate": any(
            abs(v) > tolerance for v in joint_j
        ),
        "route_bank_order_artifact_example": {
            str(q): route_bank_order_artifact(relation, q) for q in closed_queries
        },
        "route_bank_order_artifact_not_rejected_by_reference_model": True,
        "query_preselection_invalid_example": not query_preselection_valid(
            source_closed=True,
            query_generated_after_close=False,
        ),
        "capacity_separation_established": False,
        "family10h_physical_witness_frozen": False,
        "decision": DECISION,
    }

    expected = spec.get("expected", {})
    if isinstance(expected, Mapping):
        mismatches = {}
        for key, expected_value in expected.items():
            if results.get(key) != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": results.get(key),
                }
        results["expected_mismatches"] = mismatches
        results["passed"] = not mismatches
    else:
        results["expected_mismatches"] = {"expected": "must be an object"}
        results["passed"] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=Path, default=Path(__file__).with_name("MECHANISM_REFERENCE_TESTS.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.tests.read_text(encoding="utf-8"))
    results = run_tests(spec)
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.self_test and not results.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
