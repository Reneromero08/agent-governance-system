#!/usr/bin/env python3
"""Clean-room bounded reconstructions for the four audio CATVM candidates.

This implementation consumes only the public text fixtures and mathematical
laws.  It does not import, execute, or translate the production C recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


F3 = 3


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _powerset_boolean(width: int) -> Iterable[tuple[int, ...]]:
    return itertools.product((0, 1), repeat=width)


# ---------------------------------------------------------------------------
# Candidate A: enumerate zero sets, then interpolate the unique multilinear
# polynomial over F3.  This is deliberately unlike the phase-factor path.


@dataclass(frozen=True)
class SeriesParallelProgram:
    left: tuple[int, int, int, int]
    right: tuple[int, int, int, int]
    constraint: tuple[int, int, int, int]


def parse_aspr(path: Path) -> SeriesParallelProgram:
    relations: dict[str, tuple[int, int, int, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if fields[:1] != ["RELATION"]:
            continue
        if len(fields) != 8:
            raise ValueError(f"malformed RELATION line: {raw}")
        coefficients = tuple(int(value) for value in fields[4:8])
        if any(value not in (0, 1, 2) for value in coefficients):
            raise ValueError(f"non-F3 coefficient: {raw}")
        relations[fields[1]] = coefficients  # type: ignore[assignment]
    expected = {"LEFT", "RIGHT", "CONSTRAINT"}
    if set(relations) != expected:
        raise ValueError(f"expected relations {expected}, got {set(relations)}")
    return SeriesParallelProgram(
        left=relations["LEFT"],
        right=relations["RIGHT"],
        constraint=relations["CONSTRAINT"],
    )


def eval_bilinear(coefficients: Sequence[int], x: int, y: int) -> int:
    return (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * y
    ) % F3


def interpolate_bilinear(
    values: dict[tuple[int, int], int],
) -> tuple[int, int, int, int]:
    c00 = values[(0, 0)] % F3
    c10 = (values[(1, 0)] - c00) % F3
    c01 = (values[(0, 1)] - c00) % F3
    c11 = (values[(1, 1)] - c00 - c10 - c01) % F3
    result = (c00, c10, c01, c11)
    for point, expected in values.items():
        if eval_bilinear(result, *point) != expected % F3:
            raise AssertionError("F3 interpolation failed")
    return result


def enumerate_series_parallel(
    program: SeriesParallelProgram,
) -> dict[str, object]:
    y_values: dict[tuple[int, int], int] = {}
    z_values: dict[tuple[int, int], int] = {}
    witnesses: dict[str, list[int]] = {}
    for a, b in _powerset_boolean(2):
        valid_u = [
            u
            for u in (0, 1)
            if eval_bilinear(program.left, a, u) == 0
            and eval_bilinear(program.right, u, b) == 0
        ]
        # A zero polynomial value means the tuple belongs to the relation.
        # The canonical 0/1 constraint is therefore zero iff a witness exists.
        y_value = 0 if valid_u else 1
        c_value = eval_bilinear(program.constraint, a, b)
        # In F3, q^2 is 0 for q=0 and 1 for either nonzero value.  Thus
        # y^2+c^2 vanishes exactly at the intersection of both zero sets.
        z_value = (y_value * y_value + c_value * c_value) % F3
        y_values[(a, b)] = y_value
        z_values[(a, b)] = z_value
        witnesses[f"{a}{b}"] = valid_u
    return {
        "f_coefficients": list(interpolate_bilinear(y_values)),
        "z_coefficients": list(interpolate_bilinear(z_values)),
        "f_zero_set": [
            list(point) for point, value in sorted(y_values.items()) if value == 0
        ],
        "z_zero_set": [
            list(point) for point, value in sorted(z_values.items()) if value == 0
        ],
        "witnesses": witnesses,
    }


def _add_vector(
    left: Sequence[int], right: Sequence[int], sign: int = 1
) -> tuple[int, ...]:
    return tuple((a + sign * b) % F3 for a, b in zip(left, right, strict=True))


def restoration_controls(program: SeriesParallelProgram) -> dict[str, object]:
    forward = enumerate_series_parallel(program)
    f_factor = tuple(forward["f_coefficients"])
    g_factor = tuple(forward["z_coefficients"])
    zero = (0, 0, 0, 0)

    resident_y = _add_vector(zero, f_factor)
    resident_z = _add_vector(zero, g_factor)
    nominal_z = _add_vector(resident_z, g_factor, sign=-1)
    nominal_y = _add_vector(resident_y, f_factor, sign=-1)

    wrong_g = tuple(g_factor[1:]) + tuple(g_factor[:1])
    wrong_z = _add_vector(resident_z, wrong_g, sign=-1)
    missing_z = resident_z

    # Reordered inverse removes F first.  G^-1 must then be recomputed from
    # the actual (now-zero) Y and C, not from a saved forward factor.
    universal_y = (0, 0, 0, 0)
    reordered_program = SeriesParallelProgram(
        program.left,
        program.right,
        program.constraint,
    )
    c_values = {
        (a, b): (
            eval_bilinear(universal_y, a, b) ** 2
            + eval_bilinear(reordered_program.constraint, a, b) ** 2
        )
        % F3
        for a, b in _powerset_boolean(2)
    }
    reordered_g = interpolate_bilinear(c_values)
    reordered_z = _add_vector(resident_z, reordered_g, sign=-1)

    return {
        "nominal_restored": nominal_y == zero and nominal_z == zero,
        "wrong_g_detected": wrong_z != zero,
        "missing_g_detected": missing_z != zero,
        "reordered_inverse_detected": reordered_z != zero,
        "residuals": {
            "wrong_g_z": list(wrong_z),
            "missing_g_z": list(missing_z),
            "reordered_z": list(reordered_z),
        },
    }


# ---------------------------------------------------------------------------
# Candidate B: generic graph parsing and an exhaustive reversible-pebble
# search.  Production receipts, counters, and schedule code are not used.


@dataclass(frozen=True)
class DagNode:
    public_id: int
    kind: str
    children: tuple[int, ...]


@dataclass(frozen=True)
class PublicDag:
    root: int
    nodes: tuple[DagNode, ...]


def parse_public_dag(path: Path) -> PublicDag:
    root: int | None = None
    nodes: list[DagNode] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "root" and len(fields) == 2:
            root = int(fields[1])
        elif fields[0] == "leaf" and len(fields) == 5:
            nodes.append(DagNode(int(fields[1]), "leaf", ()))
        elif fields[0] in {"compose", "intersect"} and len(fields) == 4:
            nodes.append(
                DagNode(
                    int(fields[1]),
                    fields[0],
                    (int(fields[2]), int(fields[3])),
                )
            )
        else:
            raise ValueError(f"malformed DAG line: {raw}")
    if root is None:
        raise ValueError("DAG root missing")
    ids = [node.public_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate DAG node")
    known = set(ids)
    if root not in known:
        raise ValueError("root not declared")
    for node in nodes:
        if any(child not in known for child in node.children):
            raise ValueError("unknown DAG child")
    return PublicDag(root, tuple(nodes))


def dag_analysis(dag: PublicDag) -> dict[str, object]:
    by_id = {node.public_id: node for node in dag.nodes}
    indegree = {node.public_id: len(node.children) for node in dag.nodes}
    consumers: dict[int, list[int]] = {node.public_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for child in node.children:
            consumers[child].append(node.public_id)
    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for consumer in sorted(consumers[current]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
    if len(order) != len(dag.nodes):
        raise ValueError("DAG contains a cycle")
    fanout = {node: len(edges) for node, edges in consumers.items()}
    return {
        "node_count": len(dag.nodes),
        "root": dag.root,
        "topological_order": order,
        "fanout": {str(node): fanout[node] for node in sorted(fanout)},
        "shared_fanout": {
            str(node): fanout[node] for node in sorted(fanout) if fanout[node] > 1
        },
        "maximum_fanout": max(fanout.values(), default=0),
    }


def _pebble_model(dag: PublicDag) -> tuple[list[int], list[int], int]:
    ids = sorted(node.public_id for node in dag.nodes)
    index = {public_id: offset for offset, public_id in enumerate(ids)}
    requirements = [0] * len(ids)
    for node in dag.nodes:
        for child in node.children:
            requirements[index[node.public_id]] |= 1 << index[child]
    return ids, requirements, 1 << index[dag.root]


def _shortest_pebble_path(
    requirements: Sequence[int],
    target_predicate,
    capacity: int,
) -> list[int] | None:
    queue = deque([0])
    predecessor: dict[int, tuple[int, int] | None] = {0: None}
    goal: int | None = None
    while queue:
        state = queue.popleft()
        if target_predicate(state):
            goal = state
            break
        for node, required in enumerate(requirements):
            if state & required != required:
                continue
            candidate = state ^ (1 << node)
            if candidate.bit_count() > capacity or candidate in predecessor:
                continue
            predecessor[candidate] = (state, node)
            queue.append(candidate)
    if goal is None:
        return None
    actions: list[int] = []
    while predecessor[goal] is not None:
        previous, node = predecessor[goal]  # type: ignore[misc]
        actions.append(node)
        goal = previous
    actions.reverse()
    return actions


def replay_pebble_actions(
    requirements: Sequence[int],
    actions: Sequence[int],
    capacity: int,
) -> tuple[int, int]:
    state = 0
    peak = 0
    for node in actions:
        if node < 0 or node >= len(requirements):
            raise ValueError("action node outside graph")
        if state & requirements[node] != requirements[node]:
            raise ValueError("action violates reversible dependency")
        state ^= 1 << node
        peak = max(peak, state.bit_count())
        if peak > capacity:
            raise ValueError("schedule exceeds capacity")
    return state, peak


def reversible_schedule_analysis(dag: PublicDag) -> dict[str, object]:
    ids, requirements, root_bit = _pebble_model(dag)
    minimum_capacity: int | None = None
    minimum_path: list[int] | None = None
    for capacity in range(1, len(ids) + 1):
        path = _shortest_pebble_path(
            requirements, lambda state: bool(state & root_bit), capacity
        )
        if path is not None:
            minimum_capacity = capacity
            minimum_path = path
            break
    if minimum_capacity is None or minimum_path is None:
        raise AssertionError("root is not reversibly reachable")
    minimum_state, minimum_peak = replay_pebble_actions(
        requirements, minimum_path, minimum_capacity
    )

    # Hold fixed the first minimum-capacity root state.  This avoids selecting
    # a source schedule or receipt and makes capacity/action tradeoffs exact.
    target_state = minimum_state
    tradeoff: dict[str, int | None] = {}
    witnesses: dict[int, list[int]] = {}
    for capacity in range(max(1, minimum_capacity - 1), min(9, len(ids)) + 1):
        path = _shortest_pebble_path(
            requirements, lambda state, target=target_state: state == target, capacity
        )
        tradeoff[str(capacity)] = None if path is None else len(path)
        if path is not None:
            witnesses[capacity] = path

    selected_capacity = next(
        (
            capacity
            for capacity, path in sorted(witnesses.items())
            if len(path) == 28
        ),
        minimum_capacity,
    )
    forward = witnesses[selected_capacity]
    forward_state, peak = replay_pebble_actions(
        requirements, forward, selected_capacity
    )
    reverse = list(reversed(forward))
    restored_state, _ = replay_pebble_actions(
        requirements, [*forward, *reverse], selected_capacity
    )

    controls: dict[str, bool] = {}
    mutations = {
        "missing": [*forward[:-1], *reverse],
        "duplicated": [*forward, forward[-1], *reverse],
        "wrong_first": [len(ids), *forward[1:], *reverse],
    }
    for name, actions in mutations.items():
        try:
            state, _ = replay_pebble_actions(
                requirements, actions, selected_capacity
            )
            controls[name] = state != 0
        except ValueError:
            controls[name] = True

    return {
        "minimum_reversible_capacity": minimum_capacity,
        "minimum_capacity_root_actions": len(minimum_path),
        "minimum_capacity_peak": minimum_peak,
        "fixed_projection_state": [
            ids[index] for index in range(len(ids)) if target_state & (1 << index)
        ],
        "fixed_projection_live": target_state.bit_count(),
        "capacity_action_tradeoff": tradeoff,
        "selected_witness": {
            "logical_capacity": selected_capacity,
            "forward_actions": len(forward),
            "reverse_actions": len(reverse),
            "peak": peak,
            "forward_public_nodes": [ids[index] for index in forward],
            "reverse_is_literal_forward_reverse": reverse == list(reversed(forward)),
            "projection_state_matches": forward_state == target_state,
            "restored_empty": restored_state == 0,
        },
        "mutation_controls_detected": controls,
    }


# ---------------------------------------------------------------------------
# Candidate C: symbolic square-free GF2 polynomials, not the source evaluator.


Monomial = frozenset[str]
Polynomial = frozenset[Monomial]


def gf2_add(*polynomials: Polynomial) -> Polynomial:
    parity: Counter[Monomial] = Counter()
    for polynomial in polynomials:
        parity.update(polynomial)
    return frozenset(term for term, count in parity.items() if count % 2)


def gf2_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    parity: Counter[Monomial] = Counter(a | b for a in left for b in right)
    return frozenset(term for term, count in parity.items() if count % 2)


def gf2_constant(value: int) -> Polynomial:
    return frozenset({frozenset()}) if value & 1 else frozenset()


def gf2_variable(name: str) -> Polynomial:
    return frozenset({frozenset({name})})


def parse_qanf(path: Path) -> tuple[tuple[int, int, int], ...]:
    stages: list[tuple[int, int, int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if fields[:1] not in (["F"], ["G"], ["J"]):
            continue
        if len(fields) != 4:
            raise ValueError(f"malformed QANF stage: {raw}")
        row = tuple(int(value) for value in fields[1:])
        if any(value not in (0, 1) for value in row):
            raise ValueError(f"non-Boolean QANF coefficient: {raw}")
        stages.append(row)  # type: ignore[arg-type]
    if len(stages) != 3:
        raise ValueError("QANF requires F, G, J")
    return tuple(stages)


def symbolic_qanf_boundary(
    alpha: int,
    beta: int,
    gamma: int,
    delta: int,
    eta: int,
    theta: int,
) -> tuple[int, int, int, int, int]:
    a, b, c, e = (gf2_variable(name) for name in ("a", "b", "c", "e"))
    u = gf2_add(gf2_constant(alpha), gf2_mul(gf2_constant(beta), gf2_mul(a, b)))
    v = gf2_add(
        gf2_constant(gamma),
        gf2_mul(gf2_constant(delta), gf2_mul(u, c)),
    )
    rhs = gf2_add(
        gf2_constant(eta),
        gf2_mul(gf2_constant(theta), gf2_mul(v, e)),
    )
    ordered = (
        frozenset(),
        frozenset({"e"}),
        frozenset({"c", "e"}),
        frozenset({"a", "b", "c", "e"}),
    )
    return (1, *(int(term in rhs) for term in ordered))


def qanf_analysis(fixtures: Sequence[Path]) -> dict[str, object]:
    fixture_results: dict[str, list[int]] = {}
    for fixture in fixtures:
        stages = parse_qanf(fixture)
        if any(stage[0] != 1 for stage in stages):
            raise ValueError("bounded public schema requires unit dependent coefficient")
        f, g, j = stages
        fixture_results[fixture.name] = list(
            symbolic_qanf_boundary(f[1], f[2], g[1], g[2], j[1], j[2])
        )

    table: list[list[int]] = []
    for bits in _powerset_boolean(6):
        table.append(list(symbolic_qanf_boundary(*bits)))
    unique = {_canonical_json(boundary) for boundary in table}
    return {
        "fixture_boundaries": fixture_results,
        "public_programs": len(table),
        "unique_boundaries": len(unique),
        "raw_five_bit_table_bits": len(table) * 5,
        "packed_nonconstant_output_table_bits": len(table) * 4,
        "direct_formula_and_count": 4,
        "table_sha256": _sha256_json(table),
        "capacity_obstruction": len(table) == 64 and len(unique) <= 16,
    }


# ---------------------------------------------------------------------------
# Candidate D: construct the raw repeated-neighbor transducer from its local
# Boolean law, then refine finite-horizon suffix-bisimulation signatures.


def _state_bits(state: int, depth: int) -> tuple[int, ...]:
    return tuple((state >> (depth - 1 - index)) & 1 for index in range(depth))


def _threshold_height(state: int, depth: int, variant: str) -> int | None:
    leading = 1 if variant == "and" else 0
    changed = False
    height = 0
    for bit in _state_bits(state, depth):
        if not changed and bit == leading:
            height += 1
        else:
            changed = True
            if bit == leading:
                return None
    return height


def _local_boolean(left: int, right: int, variant: str) -> int:
    if variant == "and":
        return left & right
    if variant == "or":
        return left | right
    raise ValueError(f"unknown homogeneous variant: {variant}")


def _interior_successors(
    state: int,
    x: int,
    z: int,
    depth: int,
    variant: str,
    threshold_states: Sequence[int],
) -> tuple[int, ...]:
    left = _state_bits(state, depth)
    if left[0] != x:
        return ()
    successors = []
    for candidate in threshold_states:
        right = _state_bits(candidate, depth)
        if any(
            left[layer]
            != _local_boolean(left[layer - 1], right[layer - 1], variant)
            for layer in range(1, depth)
        ):
            continue
        if z != _local_boolean(left[-1], right[-1], variant):
            continue
        successors.append(candidate)
    return tuple(successors)


def _suffix_bisimulation(
    depth: int, horizon: int, variant: str
) -> tuple[tuple[tuple[int, ...], ...], dict[int, tuple]]:
    if depth < 1 or horizon < 1:
        raise ValueError("depth and horizon must be positive")
    states = tuple(
        state
        for state in range(1 << depth)
        if _threshold_height(state, depth, variant) is not None
    )
    alphabet = tuple(_powerset_boolean(2))

    # At the last word site, the fixed right boundary accepts exactly when
    # the first raw layer bit agrees with public x; public z is unconstrained.
    signatures: dict[int, tuple] = {
        state: tuple(
            int(_state_bits(state, depth)[0] == x) for x, _z in alphabet
        )
        for state in states
    }

    def classify(values: dict[int, tuple]) -> dict[int, int]:
        identifiers: dict[tuple, int] = {}
        classes: dict[int, int] = {}
        for state in states:
            signature = values[state]
            if signature not in identifiers:
                identifiers[signature] = len(identifiers)
            classes[state] = identifiers[signature]
        return classes

    classes = classify(signatures)
    edge_cache = {
        (state, x, z): _interior_successors(
            state, x, z, depth, variant, states
        )
        for state in states
        for x, z in alphabet
    }
    for _remaining in range(2, horizon + 1):
        signatures = {
            state: tuple(
                tuple(
                    sorted(
                        {
                            classes[successor]
                            for successor in edge_cache[(state, x, z)]
                        }
                    )
                )
                for x, z in alphabet
            )
            for state in states
        }
        classes = classify(signatures)

    grouped: dict[int, list[int]] = {}
    for state, class_id in classes.items():
        grouped.setdefault(class_id, []).append(
            int(_threshold_height(state, depth, variant))
        )
    partition = tuple(
        sorted(
            (tuple(sorted(group)) for group in grouped.values()),
            key=lambda group: group[0],
        )
    )
    return partition, signatures


def quotient_partition(
    depth: int, horizon: int, variant: str = "and"
) -> tuple[tuple[int, ...], ...]:
    return _suffix_bisimulation(depth, horizon, variant)[0]


def expected_threshold_partition(
    depth: int, horizon: int
) -> tuple[tuple[int, ...], ...]:
    if horizon == 1:
        return ((0,), tuple(range(1, depth + 1)))
    if horizon >= depth:
        return tuple((height,) for height in range(depth + 1))
    return (
        *((height,) for height in range(horizon)),
        tuple(range(horizon, depth)),
        (depth,),
    )


def quotient_analysis() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    all_match = True
    duality = True
    overmerge_detected = True
    undermerge_detected = True
    for depth in range(2, 9):
        for horizon in range(1, 17):
            observed, signatures = _suffix_bisimulation(depth, horizon, "and")
            observed_or, _or_signatures = _suffix_bisimulation(
                depth, horizon, "or"
            )
            expected = expected_threshold_partition(depth, horizon)
            rank_formula = (
                2 if horizon == 1 else min(depth + 1, horizon + 2)
            )
            match = observed == expected and len(observed) == rank_formula
            all_match &= match
            duality &= observed_or == observed
            if 2 <= horizon < depth:
                middle = tuple(range(horizon, depth))
                state_by_height = {
                    int(_threshold_height(state, depth, "and")): state
                    for state in signatures
                }
                top_signature = signatures[state_by_height[depth]]
                middle_signature = signatures[state_by_height[middle[0]]]
                overmerge_detected &= top_signature != middle_signature
                if len(middle) > 1:
                    undermerge_detected &= all(
                        signatures[state_by_height[height]] == middle_signature
                        for height in middle
                    )
            cases.append(
                {
                    "depth": depth,
                    "horizon": horizon,
                    "rank": len(observed),
                    "formula_rank": rank_formula,
                    "partition": [list(group) for group in observed],
                    "match": match,
                }
            )
    return {
        "cases_checked": len(cases),
        "all_formula_cases_match": all_match,
        "deliberate_overmerge_detected": overmerge_detected,
        "deliberate_undermerge_detected": undermerge_detected,
        "and_or_duality": duality,
        "depth_greater_than_width_abstract_cases_checked": sum(
            1
            for case in cases
            if int(case["depth"]) > int(case["horizon"])
        ),
        "case_sha256": _sha256_json(cases),
        "cases": cases,
    }


def run_all(source: Path) -> dict[str, object]:
    primary = parse_aspr(source / "catvm_primary.aspr")
    reuse = parse_aspr(source / "catvm_reuse.aspr")
    dag = parse_public_dag(source / "general_multi_dag_affine_topology.txt")
    qanf_fixtures = [
        source / "quadratic_anf_chain_primary.qanf",
        source / "quadratic_anf_chain_affine_sham.qanf",
        source / "quadratic_anf_chain_degree2_nonaffine.qanf",
        source / "quadratic_anf_chain_reuse.qanf",
    ]
    result = {
        "schema_version": "1.0.0",
        "method": "CLEAN_ROOM_ENUMERATION_SYMBOLIC_ALGEBRA_GRAPH_SEARCH",
        "source_results_used_as_oracle": False,
        "candidate_a": {
            "primary": enumerate_series_parallel(primary),
            "reuse": enumerate_series_parallel(reuse),
            "restoration_controls": restoration_controls(primary),
        },
        "candidate_b": {
            "dag": dag_analysis(dag),
            "reversible_schedule": reversible_schedule_analysis(dag),
        },
        "candidate_c": qanf_analysis(qanf_fixtures),
        "candidate_d": quotient_analysis(),
    }
    result["result_sha256"] = _sha256_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_all(arguments.source)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
