#!/usr/bin/env python3
"""Independent full-C17 oracle and compact baselines for Paley DAG pebbling."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any


FIELD = 103
ORDER = 17
ZETA = 72
TARGETS = tuple(range(2, 9))
EDGES = ((2, 0, 1), (3, 0, 1), (4, 2, 3), (5, 2, 3), (6, 4, 5), (7, 3, 6), (8, 5, 7))
RESIDUES = {value * value % ORDER for value in range(1, ORDER)}
NONRESIDUES = set(range(1, ORDER)) - RESIDUES
CLASSES = ({0}, RESIDUES, NONRESIDUES)


def phase(exponent: int) -> int:
    return pow(ZETA, exponent % ORDER, FIELD)


def compact_seed(register: int) -> list[int]:
    offset = 5 if register == 0 else 19
    return [phase(offset + 2 * (coordinate + 1) + coordinate * coordinate) for coordinate in range(3)]


def expand(values: list[int]) -> list[int]:
    return [next(values[index] for index, orbit in enumerate(CLASSES) if position in orbit) for position in range(ORDER)]


def contract(full: list[int]) -> list[int]:
    result = []
    for orbit in CLASSES:
        values = {full[position] for position in orbit}
        if len(values) != 1:
            raise RuntimeError("oracle schedule escaped the Paley algebra")
        result.append(next(iter(values)))
    return result


def full_hadamard(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def full_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[(i + j) % ORDER] = (result[(i + j) % ORDER] + x * y) % FIELD
    return result


def compact_hadamard(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def compact_convolution(left: list[int], right: list[int]) -> list[int]:
    a, b, c = left
    d, e, f = right
    return [
        (a * d + 8 * b * e + 8 * c * f) % FIELD,
        (a * e + b * d + 3 * b * e + 4 * b * f + 4 * c * e + 4 * c * f) % FIELD,
        (a * f + c * d + 4 * b * e + 4 * b * f + 4 * c * e + 3 * c * f) % FIELD,
    ]


def descriptor(topology: int) -> dict[int, tuple[str, int, int]]:
    operations = ("H", "C", "C", "H", "C", "H", "C") if topology == 1 else ("C", "H", "H", "C", "H", "C", "H")
    return {target: (operation, left, right) for (target, left, right), operation in zip(EDGES, operations, strict=True)}


def search(capacity: int) -> tuple[list[int] | None, int]:
    bits = {node: 1 << index for index, node in enumerate(TARGETS)}
    parents = {target: (left, right) for target, left, right in EDGES}
    goal = bits[8]
    queue = deque([0])
    previous: dict[int, tuple[int, int] | None] = {0: None}
    while queue:
        state = queue.popleft()
        if state == goal:
            toggles: list[int] = []
            while previous[state] is not None:
                prior, node = previous[state]
                toggles.append(node)
                state = prior
            return list(reversed(toggles)), len(previous)
        for node in TARGETS:
            if all(parent not in bits or state & bits[parent] for parent in parents[node]):
                successor = state ^ bits[node]
                if successor.bit_count() <= capacity and successor not in previous:
                    previous[successor] = (state, node)
                    queue.append(successor)
    return None, len(previous)


def independent_plan() -> tuple[int, list[dict[str, int | str]], list[int]]:
    unreachable = []
    toggles = None
    capacity = 0
    for candidate in range(1, len(TARGETS) + 1):
        toggles, explored = search(candidate)
        if toggles is not None:
            capacity = candidate
            break
        unreachable.append(explored)
    if toggles is None:
        raise RuntimeError("independent clean pebble search found no schedule")
    free = list(range(capacity))
    live: dict[int, tuple[int, int]] = {}
    epochs = {node: 0 for node in TARGETS}
    steps: list[dict[str, int | str]] = []
    for node in toggles:
        if node in live:
            slot, epoch = live.pop(node)
            steps.append({"kind": "REMOVE", "node": node, "slot": slot, "epoch": epoch})
            free.append(slot)
            free.sort()
        else:
            slot = free.pop(0)
            epoch = epochs[node]
            epochs[node] += 1
            live[node] = (slot, epoch)
            steps.append({"kind": "PLACE", "node": node, "slot": slot, "epoch": epoch})
    if set(live) != {8}:
        raise RuntimeError("independent plan did not reach clean final-only state")
    return capacity, steps, unreachable


def lease_intervals(steps: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    open_leases: dict[tuple[int, int], dict[str, int | str]] = {}
    intervals: list[dict[str, int | str]] = []
    for action, step in enumerate(steps, start=1):
        key = (int(step["node"]), int(step["epoch"]))
        if step["kind"] == "PLACE":
            open_leases[key] = {
                "node": key[0],
                "epoch": key[1],
                "slot": int(step["slot"]),
                "acquire_action": action,
            }
        else:
            lease = open_leases.pop(key)
            lease["release_action"] = action
            intervals.append(lease)
    for lease in open_leases.values():
        lease["release_action"] = "BOUNDARY_THEN_INVERSE_ACTION_1"
        intervals.append(lease)
    return sorted(intervals, key=lambda item: (int(item["acquire_action"]), int(item["node"])))


def full_value(node: int, graph: dict[int, tuple[str, int, int]], sources: dict[int, list[int]], owners: dict[int, int], slots: list[list[int]]) -> list[int]:
    operation, left, right = graph[node]
    left_value = sources[left] if left in sources else slots[owners[left]]
    right_value = sources[right] if right in sources else slots[owners[right]]
    return full_hadamard(left_value, right_value) if operation == "H" else full_convolution(left_value, right_value)


def execute_full(topology: int, capacity: int, steps: list[dict[str, int | str]]) -> dict[str, Any]:
    graph = descriptor(topology)
    sources = {0: expand(compact_seed(0)), 1: expand(compact_seed(1))}
    source_copy = {node: value.copy() for node, value in sources.items()}
    slots = [[0] * ORDER for _ in range(capacity)]
    owners: dict[int, int] = {}
    owner_epochs = [-1] * capacity
    peak = 0

    def apply(step: dict[str, int | str], reverse: bool = False) -> None:
        kind = str(step["kind"])
        if reverse:
            kind = "REMOVE" if kind == "PLACE" else "PLACE"
        node, slot, epoch = int(step["node"]), int(step["slot"]), int(step["epoch"])
        value = full_value(node, graph, sources, owners, slots)
        if kind == "PLACE":
            if slot in owners.values() or any(slots[slot]):
                raise RuntimeError("oracle placed into a live slot")
            slots[slot] = value
            owners[node] = slot
            owner_epochs[slot] = epoch
        else:
            if owners.get(node) != slot or owner_epochs[slot] != epoch:
                raise RuntimeError("oracle lease mismatch")
            slots[slot] = [(stored - expected) % FIELD for stored, expected in zip(slots[slot], value, strict=True)]
            if any(slots[slot]):
                raise RuntimeError("oracle inverse failed to clear a slot")
            del owners[node]
            owner_epochs[slot] = -1

    for step in steps:
        apply(step)
        peak = max(peak, len(owners))
    if set(owners) != {8}:
        raise RuntimeError("oracle forward did not retain only the sink")
    final_classes = contract(slots[owners[8]])
    boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), final_classes, strict=True)) % FIELD
    for step in reversed(steps):
        apply(step, reverse=True)
    restored = sources == source_copy and not owners and owner_epochs == [-1] * capacity and not any(value for slot in slots for value in slot)
    return {
        "topology": topology,
        "boundary": boundary,
        "final_class_values": final_classes,
        "peak_internal_relation_slots": peak,
        "full_relation_carrier_cells": (2 + capacity) * ORDER,
        "forward_then_inverse_relation_evaluations": 2 * len(steps),
        "full136_cells_restore_exactly": restored,
    }


def compact_operation(operation: str, left: list[int], right: list[int]) -> list[int]:
    return compact_hadamard(left, right) if operation == "H" else compact_convolution(left, right)


def retain_all_baseline(topology: int) -> dict[str, Any]:
    graph = descriptor(topology)
    values = {0: compact_seed(0), 1: compact_seed(1)}
    for node in TARGETS:
        operation, left, right = graph[node]
        values[node] = compact_operation(operation, values[left], values[right])
    boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), values[8], strict=True)) % FIELD
    return {"boundary": boundary, "relation_evaluations": 7, "relation_slots": 9, "field_cells": 27}


def occurrence_baseline(topology: int) -> dict[str, Any]:
    graph = descriptor(topology)
    current_live = 0
    peak_live = 0
    evaluations = 0

    def rank(node: int) -> int:
        if node in (0, 1):
            return 0
        _operation, left, right = graph[node]
        left_rank, right_rank = rank(left), rank(right)
        return left_rank + 1 if left_rank == right_rank else max(left_rank, right_rank)

    def evaluate(node: int) -> tuple[list[int], bool]:
        nonlocal current_live, peak_live, evaluations
        if node in (0, 1):
            return compact_seed(node), False
        operation, left, right = graph[node]
        order = (left, right) if rank(left) >= rank(right) else (right, left)
        first_value, first_temporary = evaluate(order[0])
        second_value, second_temporary = evaluate(order[1])
        values = {order[0]: first_value, order[1]: second_value}
        result = compact_operation(operation, values[left], values[right])
        evaluations += 1
        temporaries = int(first_temporary) + int(second_temporary)
        if temporaries == 0:
            current_live += 1
        elif temporaries == 2:
            current_live -= 1
        peak_live = max(peak_live, current_live)
        return result, True

    result, temporary = evaluate(8)
    if not temporary or current_live != 1:
        raise RuntimeError("occurrence baseline register accounting failed")
    boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), result, strict=True)) % FIELD
    current_live -= 1
    return {
        "boundary": boundary,
        "relation_evaluations": evaluations,
        "peak_temporary_relation_slots": peak_live,
        "sealed_leaf_relation_slots": 2,
        "total_relation_slots": 2 + peak_live,
        "field_cells": 3 * (2 + peak_live),
        "final_temporary_released": current_live == 0,
        "relation_vectors_are_atomic_for_this_accounting": True,
        "operation_local_scalar_temporaries_excluded_for_both_paths": True,
    }


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    capacity, steps, unreachable = independent_plan()
    full = [execute_full(topology, capacity, steps) for topology in (1, 2)]
    retain_all = [retain_all_baseline(topology) for topology in (1, 2)]
    occurrence = [occurrence_baseline(topology) for topology in (1, 2)]
    expected = [production["primary"]["first_boundary"], production["primary"]["second_boundary"]]
    if [item["boundary"] for item in full] != expected:
        raise RuntimeError("independent full-C17 boundary mismatch")
    if any(item["boundary"] != expected[index] for index, item in enumerate(retain_all)):
        raise RuntimeError("retain-all compact baseline mismatch")
    if any(item["boundary"] != expected[index] for index, item in enumerate(occurrence)):
        raise RuntimeError("occurrence-expanded compact baseline mismatch")
    if capacity != production["primary"]["internal_relation_capacity"] or len(steps) != production["primary"]["forward_toggle_actions"]:
        raise RuntimeError("independent public scheduler mismatch")
    if capacity != 6 or unreachable != [3, 4, 12, 19, 101] or any(not item["full136_cells_restore_exactly"] for item in full):
        raise RuntimeError("bounded reversible pebble certificate failed")
    if any(item["field_cells"] != 15 or item["relation_evaluations"] != 13 for item in occurrence):
        raise RuntimeError("compact occurrence baseline resource law changed")
    schedule = [{"action": index, **step} for index, step in enumerate(steps, start=1)]
    return {
        "schema": "CATVM_PALEY_RELATION_REMATERIALIZATION_ORACLE_RESULTS_V1",
        "claim_candidate": production["claim_candidate"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_service": False,
        "imports_controller": False,
        "imports_numpy": False,
        "public_topology_scheduler": {
            "minimum_clean_local_reversible_pebble_capacity": capacity,
            "capacities_1_through_5_exhausted_without_clean_sink_state": True,
            "explored_state_counts_for_capacities_1_through_5": unreachable,
            "forward_toggle_count": len(steps),
            "toggle_nodes": [int(step["node"]) for step in steps],
            "schedule": schedule,
            "live_intervals_and_release_points": lease_intervals(steps),
            "forward_reconstruction_obligations": [{"node": 2, "epoch": 1, "place_action": 10, "parents_required_resident": [0, 1]}],
            "inverse_custody": "EXACT_REVERSE_OF_PUBLIC_FORWARD_TOGGLE_AND_SLOT_LEASE_PLAN",
            "compiler_inputs": "PUBLIC_DIRECTED_ACYCLIC_TOPOLOGY_ONLY",
            "compiler_reads_relation_VALUES_or_BOUNDARY": False,
        },
        "full_c17_oracle": full,
        "boundary_parity": {
            "primary": full[0]["boundary"] == expected[0],
            "alternate_and_restored_reuse": full[1]["boundary"] == expected[1] == production["primary"]["fresh_second_boundary"],
        },
        "matched_compact_classical_baselines": {
            "retain_all": retain_all,
            "matched_reversible_pebbling": {
                "relation_slots": 2 + capacity,
                "field_cells": 3 * (2 + capacity),
                "forward_toggle_actions": len(steps),
                "full_lifecycle_relation_evaluations": 2 * len(steps),
                "identical_to_accepted_storage_and_toggle_law": True,
            },
            "occurrence_expanded_compact_recurrence": occurrence,
            "accepted_to_retain_all_field_cell_ratio": 24 / 27,
            "accepted_to_occurrence_expanded_field_cell_ratio": 24 / 15,
            "accepted_storage_advantage_over_strongest_executed_compact_baseline": False,
        },
        "resource_scope": {
            "accepted_compact_relation_field_cells": 24,
            "independent_full_c17_oracle_field_cells": 136,
            "retain_all_compact_relation_field_cells": 27,
            "occurrence_expanded_compact_relation_field_cells": 15,
            "dense17_by17_relation_table_cells": 0,
            "retained_dynamic_inverse_history_records": 0,
            "compiled_public_plan_records": 15,
            "owner_node_records": 6,
            "owner_epoch_records": 6,
            "python_allocator_and_interpreter_overhead_excluded_for_all_paths": True,
            "operation_local_scalar_temporaries_excluded_for_all_compact_relation_paths": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": ["EXACT_MINIMUM_6_UNDER_DECLARED_LOCAL_CLEAN_REVERSIBLE_PEBBLE_LAW", "PUBLIC_TOPOLOGY_DERIVED_LIVE_INTERVALS_RELEASES_AND_RECONSTRUCTION", "EXACT_FULL_C17_BOUNDARY_PARITY", "EXACT_FORWARD_INVERSE_CARRIER_RESTORATION", "FRESH_RESTORED_REUSE_BOUNDARY_PARITY", "RETAIN_ALL_INTERNAL_STORAGE_REDUCED_FROM_7_TO_6_RELATION_SLOTS"],
        "negative_result": "THE_ACCEPTED_24_FIELD_CELL_REMATERIALIZING_PHASE_PATH_MATCHES_THE_REVERSIBLE_CLASSICAL_PEBBLER_AND_EXCEEDS_THE_EXECUTED_15_FIELD_CELL_OCCURRENCE_EXPANDED_COMPACT_CLASSICAL_RECURRENCE_SO_NO_DISTINCT_PHASE_RESOURCE_OR_SMALL_WALL_CROSSING_IS_ESTABLISHED",
        "rejected_interpretations": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_result.read_text(encoding="utf-8"))
    payload = json.dumps(build_result(production), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
