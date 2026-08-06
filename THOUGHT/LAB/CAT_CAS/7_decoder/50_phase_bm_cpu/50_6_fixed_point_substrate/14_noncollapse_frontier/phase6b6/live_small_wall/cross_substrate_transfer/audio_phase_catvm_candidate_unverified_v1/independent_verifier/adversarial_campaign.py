#!/usr/bin/env python3
"""Seeded adversarial campaign declared in MUTATION_PLAN.json."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cleanroom_verify import (
    DagNode,
    PublicDag,
    SeriesParallelProgram,
    _canonical_json,
    _local_boolean,
    _pebble_model,
    _sha256_json,
    _state_bits,
    dag_analysis,
    enumerate_series_parallel,
    expected_threshold_partition,
    parse_aspr,
    parse_public_dag,
    parse_qanf,
    quotient_partition,
    replay_pebble_actions,
    restoration_controls,
    reversible_schedule_analysis,
    symbolic_qanf_boundary,
)


@dataclass
class Ledger:
    attempted: int = 0
    passed: int = 0
    failed: int = 0
    failures: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    def check(self, condition: bool, family: str, case: object) -> None:
        self.attempted += 1
        if condition:
            self.passed += 1
            return
        self.failed += 1
        assert self.failures is not None
        self.failures.append({"family": family, "case": case})


def _zero_set(coefficients: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    program = SeriesParallelProgram(coefficients, (0, 0, 0, 0), coefficients)
    result = enumerate_series_parallel(program)
    return {tuple(point) for point in result["z_zero_set"]}


def campaign_a(rng: random.Random, source: Path, ledger: Ledger) -> dict[str, object]:
    categories: Counter[str] = Counter()
    output_hashes: list[str] = []
    for index in range(256):
        rows = [
            tuple(rng.randrange(3) for _ in range(4))
            for _row in range(3)
        ]
        program = SeriesParallelProgram(*rows)
        result = enumerate_series_parallel(program)
        output_hashes.append(_sha256_json(result))
        zero_count = len(result["z_zero_set"])
        categories[
            "empty" if zero_count == 0 else "universal" if zero_count == 4 else "partial"
        ] += 1
        ledger.check(
            all(value in (0, 1, 2) for value in result["z_coefficients"]),
            "A:f3_relation_coefficients",
            index,
        )
        reordered = SeriesParallelProgram(program.left, program.right, program.constraint)
        ledger.check(
            enumerate_series_parallel(reordered) == result,
            "A:relation_presentation_reorder",
            index,
        )

    zero = (0, 0, 0, 0)
    targeted_programs = {
        "universal": SeriesParallelProgram(zero, zero, zero),
        "empty": SeriesParallelProgram(zero, zero, (1, 0, 0, 0)),
        "rank_deficient": SeriesParallelProgram(
            (0, 1, 0, 0), zero, zero
        ),
        "contradictory_intersection": SeriesParallelProgram(
            (0, 1, 2, 0), (0, 1, 2, 0), (1, 0, 0, 0)
        ),
    }
    targeted_zero_sets = {
        name: result["z_zero_set"]
        for name, result in (
            (name, enumerate_series_parallel(program))
            for name, program in targeted_programs.items()
        )
    }
    ledger.check(
        len(targeted_zero_sets["universal"]) == 4,
        "A:universal_zero_set",
        targeted_zero_sets["universal"],
    )
    ledger.check(
        len(targeted_zero_sets["empty"]) == 0,
        "A:empty_zero_set",
        targeted_zero_sets["empty"],
    )
    ledger.check(
        0 < len(targeted_zero_sets["rank_deficient"]) < 4,
        "A:rank_deficient",
        targeted_zero_sets["rank_deficient"],
    )
    ledger.check(
        len(targeted_zero_sets["contradictory_intersection"]) == 0,
        "A:contradictory_intersection",
        targeted_zero_sets["contradictory_intersection"],
    )

    primary = parse_aspr(source / "catvm_primary.aspr")
    reuse = parse_aspr(source / "catvm_reuse.aspr")
    primary_controls = restoration_controls(primary)
    reuse_controls = restoration_controls(reuse)
    for cycle in range(128):
        active = primary_controls if cycle % 2 == 0 else reuse_controls
        ledger.check(
            bool(active["nominal_restored"]),
            "A:alternating_reuse",
            cycle,
        )
    stale_primary = enumerate_series_parallel(primary)["z_coefficients"]
    stale_reuse = enumerate_series_parallel(reuse)["z_coefficients"]
    ledger.check(
        stale_primary != stale_reuse,
        "A:stale_G_factor_reuse",
        {"primary": stale_primary, "reuse": stale_reuse},
    )
    return {
        "coefficient_cases": 256,
        "zero_set_categories": dict(sorted(categories.items())),
        "targeted_zero_sets": targeted_zero_sets,
        "case_hash": _sha256_json(output_hashes),
        "alternating_cycles": 128,
        "stale_factor_changes_between_programs": stale_primary != stale_reuse,
    }


def _renumber_dag(dag: PublicDag, mapping: dict[int, int]) -> PublicDag:
    return PublicDag(
        root=mapping[dag.root],
        nodes=tuple(
            DagNode(
                mapping[node.public_id],
                node.kind,
                tuple(mapping[child] for child in node.children),
            )
            for node in dag.nodes
        ),
    )


def _novel_dags() -> dict[str, PublicDag]:
    return {
        "chain": PublicDag(
            4,
            (
                DagNode(1, "leaf", ()),
                DagNode(2, "leaf", ()),
                DagNode(3, "compose", (1, 2)),
                DagNode(4, "intersect", (3, 2)),
            ),
        ),
        "diamond": PublicDag(
            6,
            (
                DagNode(1, "leaf", ()),
                DagNode(2, "leaf", ()),
                DagNode(3, "compose", (1, 2)),
                DagNode(4, "compose", (3, 1)),
                DagNode(5, "intersect", (3, 2)),
                DagNode(6, "intersect", (4, 5)),
            ),
        ),
        "balanced": PublicDag(
            7,
            (
                DagNode(1, "leaf", ()),
                DagNode(2, "leaf", ()),
                DagNode(3, "leaf", ()),
                DagNode(4, "leaf", ()),
                DagNode(5, "compose", (1, 2)),
                DagNode(6, "compose", (3, 4)),
                DagNode(7, "intersect", (5, 6)),
            ),
        ),
    }


def campaign_b(rng: random.Random, source: Path, ledger: Ledger) -> dict[str, object]:
    dag = parse_public_dag(source / "general_multi_dag_affine_topology.txt")
    baseline_graph = dag_analysis(dag)
    baseline_schedule = reversible_schedule_analysis(dag)
    ids = sorted(node.public_id for node in dag.nodes)
    renumber_hashes = []
    for case in range(32):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        mapping = dict(zip(ids, shuffled, strict=True))
        mutated = _renumber_dag(dag, mapping)
        graph = dag_analysis(mutated)
        schedule = reversible_schedule_analysis(mutated)
        ledger.check(
            sorted(graph["fanout"].values())
            == sorted(baseline_graph["fanout"].values()),
            "B:public_id_bijection",
            case,
        )
        ledger.check(
            schedule["capacity_action_tradeoff"]
            == baseline_schedule["capacity_action_tradeoff"],
            "B:renumbered_schedule",
            case,
        )
        renumber_hashes.append(_sha256_json({"graph": graph, "schedule": schedule}))

    novel: dict[str, object] = {}
    for name, graph in _novel_dags().items():
        analysis = dag_analysis(graph)
        schedule = reversible_schedule_analysis(graph)
        ledger.check(
            bool(schedule["selected_witness"]["restored_empty"]),
            f"B:novel_{name}",
            analysis,
        )
        novel[name] = {
            "nodes": analysis["node_count"],
            "fanout": analysis["maximum_fanout"],
            "minimum_capacity": schedule["minimum_reversible_capacity"],
        }

    invalid = {
        "cycle": PublicDag(
            2,
            (
                DagNode(1, "compose", (2, 2)),
                DagNode(2, "compose", (1, 1)),
            ),
        ),
        "disconnected_cycle": PublicDag(
            3,
            (
                DagNode(1, "leaf", ()),
                DagNode(2, "compose", (3, 3)),
                DagNode(3, "compose", (2, 2)),
            ),
        ),
    }
    for name, graph in invalid.items():
        try:
            dag_analysis(graph)
            rejected = False
        except ValueError:
            rejected = True
        ledger.check(rejected, f"B:{name}", name)

    ids_model, requirements, _root = _pebble_model(dag)
    forward_public = baseline_schedule["selected_witness"]["forward_public_nodes"]
    index = {public_id: offset for offset, public_id in enumerate(ids_model)}
    forward = [index[public_id] for public_id in forward_public]
    corruptions = {
        "missing": forward[:-1] + list(reversed(forward)),
        "duplicate": forward + [forward[-1]] + list(reversed(forward)),
        "dependency_reorder": list(reversed(forward)) + forward,
        "wrong_node": [len(ids_model)] + forward[1:] + list(reversed(forward)),
    }
    corruption_results = {}
    for name, actions in corruptions.items():
        try:
            state, _peak = replay_pebble_actions(requirements, actions, 7)
            detected = state != 0
        except ValueError:
            detected = True
        ledger.check(detected, f"B:schedule_{name}", actions[:8])
        corruption_results[name] = detected
    return {
        "renumbered_cases": 32,
        "renumbered_case_hash": _sha256_json(renumber_hashes),
        "novel_topologies": novel,
        "invalid_topologies_rejected": sorted(invalid),
        "schedule_corruptions_detected": corruption_results,
    }


def campaign_c(ledger: Ledger) -> dict[str, object]:
    boundaries: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for bits in (
        tuple(values)
        for values in __import__("itertools").product((0, 1), repeat=6)
    ):
        boundary = symbolic_qanf_boundary(*bits)
        boundaries.setdefault(boundary, []).append(bits)
        direct = (
            1,
            bits[4],
            bits[5] & bits[2],
            bits[5] & bits[3] & bits[0],
            bits[5] & bits[3] & bits[1],
        )
        ledger.check(boundary == direct, "C:all_public_programs", bits)
    collision_sizes = sorted(len(programs) for programs in boundaries.values())
    ledger.check(len(boundaries) == 16, "C:unique_boundary_count", len(boundaries))
    return {
        "programs": 64,
        "unique_boundaries": len(boundaries),
        "equivalent_program_collision_sizes": collision_sizes,
        "maximum_programs_per_boundary": max(collision_sizes),
        "boundary_map_sha256": _sha256_json(
            {
                "".join(map(str, boundary)): [
                    "".join(map(str, program)) for program in programs
                ]
                for boundary, programs in sorted(boundaries.items())
            }
        ),
    }


def _mixed_successors(
    state: int, x: int, z: int, operations: tuple[str, ...]
) -> tuple[int, ...]:
    depth = len(operations)
    left = _state_bits(state, depth)
    if left[0] != x:
        return ()
    successors = []
    for candidate in range(1 << depth):
        right = _state_bits(candidate, depth)
        if any(
            left[layer]
            != _local_boolean(
                left[layer - 1], right[layer - 1], operations[layer - 1]
            )
            for layer in range(1, depth)
        ):
            continue
        if z != _local_boolean(left[-1], right[-1], operations[-1]):
            continue
        successors.append(candidate)
    return tuple(successors)


def _mixed_partition_rank(operations: tuple[str, ...], horizon: int) -> int:
    depth = len(operations)
    states = tuple(range(1 << depth))
    alphabet = ((0, 0), (0, 1), (1, 0), (1, 1))
    signatures = {
        state: tuple(
            int(_state_bits(state, depth)[0] == x) for x, _z in alphabet
        )
        for state in states
    }

    def classify() -> dict[int, int]:
        identifiers: dict[tuple, int] = {}
        result: dict[int, int] = {}
        for state in states:
            signature = signatures[state]
            identifiers.setdefault(signature, len(identifiers))
            result[state] = identifiers[signature]
        return result

    classes = classify()
    edges = {
        (state, x, z): _mixed_successors(state, x, z, operations)
        for state in states
        for x, z in alphabet
    }
    for _step in range(2, horizon + 1):
        signatures = {
            state: tuple(
                tuple(
                    sorted(
                        {
                            classes[successor]
                            for successor in edges[(state, x, z)]
                        }
                    )
                )
                for x, z in alphabet
            )
            for state in states
        }
        classes = classify()
    return len(set(classes.values()))


def campaign_d(rng: random.Random, ledger: Ledger) -> dict[str, object]:
    homogeneous_hashes: list[str] = []
    wrong_horizon_detected = 0
    relabel_checks = 0
    for depth in range(2, 9):
        for horizon in range(1, 17):
            observed_and = quotient_partition(depth, horizon, "and")
            observed_or = quotient_partition(depth, horizon, "or")
            expected = expected_threshold_partition(depth, horizon)
            ledger.check(observed_and == expected, "D:homogeneous_AND", (depth, horizon))
            ledger.check(observed_or == expected, "D:homogeneous_OR", (depth, horizon))
            homogeneous_hashes.append(_sha256_json((observed_and, observed_or)))
            if 2 <= horizon < depth:
                # Moving forward can be observationally identical at the
                # saturation boundary (horizon == depth-1).  Moving backward
                # always crosses a declared class boundary in this range.
                wrong = quotient_partition(depth, horizon - 1, "and")
                detected = wrong != observed_and
                wrong_horizon_detected += int(detected)
                ledger.check(detected, "D:wrong_horizon", (depth, horizon))
            heights = list(range(depth + 1))
            rng.shuffle(heights)
            relabeled = sorted(
                sorted(heights[height] for height in group)
                for group in observed_and
            )
            unrelabeled = sorted(
                sorted(heights[height] for height in group)
                for group in expected
            )
            ledger.check(
                relabeled == unrelabeled,
                "D:state_relabeling",
                (depth, horizon),
            )
            relabel_checks += 1

    mixed: list[dict[str, object]] = []
    patterns = [
        ("alternating", tuple("and" if i % 2 == 0 else "or" for i in range(6))),
        ("nonperiodic", ("and", "and", "or", "and", "or", "or")),
        ("reverse_nonperiodic", ("or", "or", "and", "or", "and", "and")),
    ]
    for name, operations in patterns:
        ranks = [_mixed_partition_rank(operations, horizon) for horizon in range(1, 7)]
        homogeneous_bound = len(operations) + 1
        outside_homogeneous = any(rank > homogeneous_bound for rank in ranks)
        ledger.check(
            outside_homogeneous,
            f"D:mixed_{name}",
            {"operations": operations, "ranks": ranks},
        )
        mixed.append(
            {
                "name": name,
                "operations": list(operations),
                "ranks": ranks,
                "homogeneous_max_rank": homogeneous_bound,
                "outside_homogeneous_law": outside_homogeneous,
            }
        )
    return {
        "homogeneous_cases": 224,
        "homogeneous_case_hash": _sha256_json(homogeneous_hashes),
        "wrong_horizon_detected_cases": wrong_horizon_detected,
        "state_relabel_checks": relabel_checks,
        "mixed_layer_challenges": mixed,
    }


def campaign_input(source: Path, ledger: Ledger) -> dict[str, object]:
    mutations = {
        "empty": "",
        "truncated": "CATCAS_SERIES_PARALLEL_RELATION 1\nTYPE BOOLEAN_F3\n",
        "unknown": "UNKNOWN 1\n",
        "duplicate_relation": (
            "RELATION LEFT A U 0 0 0 0\nRELATION LEFT A U 0 0 0 0\n"
        ),
        "oversized_integer": (
            "RELATION LEFT A U 999999999999999999999999 0 0 0\n"
        ),
    }
    rejected = []
    with tempfile.TemporaryDirectory(prefix="catvm-cleanroom-input-") as temp:
        for name, content in mutations.items():
            path = Path(temp) / f"{name}.aspr"
            path.write_text(content, encoding="utf-8")
            try:
                parse_aspr(path)
                failed_closed = False
            except (ValueError, KeyError):
                failed_closed = True
            ledger.check(failed_closed, f"ALL:input_{name}", content)
            if failed_closed:
                rejected.append(name)
    # The valid source input must remain accepted after the malformed set.
    ledger.check(
        parse_aspr(source / "catvm_primary.aspr")
        == parse_aspr(source / "catvm_primary.aspr"),
        "ALL:deterministic_replay",
        "catvm_primary.aspr",
    )
    return {"malformed_inputs_rejected": rejected, "deterministic_replay": True}


def run(source: Path, plan: Path) -> dict[str, object]:
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    if plan_data.get("status") != "PREDECLARED_BEFORE_CAMPAIGN_EXECUTION":
        raise ValueError("mutation plan was not predeclared")
    seed = int(plan_data["seed"])
    rng = random.Random(seed)
    ledger = Ledger()
    results = {
        "A": campaign_a(rng, source, ledger),
        "B": campaign_b(rng, source, ledger),
        "C": campaign_c(ledger),
        "D": campaign_d(rng, ledger),
        "input": campaign_input(source, ledger),
    }
    payload = {
        "schema_version": "1.0.0",
        "plan_sha256": __import__("hashlib").sha256(plan.read_bytes()).hexdigest(),
        "seed": seed,
        "deterministic": True,
        "results": results,
        "ledger": {
            "attempted": ledger.attempted,
            "passed": ledger.passed,
            "failed": ledger.failed,
            "failures": ledger.failures,
        },
    }
    payload["result_sha256"] = _sha256_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.source, arguments.plan)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result["ledger"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
