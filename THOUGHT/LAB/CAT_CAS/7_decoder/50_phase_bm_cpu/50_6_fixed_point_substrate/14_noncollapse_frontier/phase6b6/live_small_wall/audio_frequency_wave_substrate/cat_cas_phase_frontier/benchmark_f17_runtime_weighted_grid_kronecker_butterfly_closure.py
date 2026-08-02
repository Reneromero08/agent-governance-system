#!/usr/bin/env python3
"""Warm observational benchmark for the M120 butterfly repair."""

from __future__ import annotations

import gc
import json
import platform
import resource
import statistics
import sys
import time
from typing import Callable

import f17_runtime_weighted_grid_kronecker_butterfly_closure as phase
import f17_runtime_weighted_grid_kronecker_butterfly_closure_oracle as oracle


def distribution(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)
    return {
        "samples": len(ordered),
        "minimum_ns": ordered[0],
        "median_ns": int(statistics.median(ordered)),
        "p90_ns": ordered[min(len(ordered) - 1, 9 * len(ordered) // 10)],
        "maximum_ns": ordered[-1],
    }


def measure(call: Callable[[], object], warmups: int, samples: int) -> dict[str, int]:
    for _ in range(warmups):
        call()
    durations: list[int] = []
    gc.disable()
    try:
        for _ in range(samples):
            start = time.perf_counter_ns()
            call()
            durations.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()
    return distribution(durations)


def main() -> int:
    cases: list[dict[str, object]] = []
    for n in phase.SIZES:
        plan = phase.m119.compile_topology(n)
        program = phase.m119.bind_runtime_program(plan, "PRIMARY")
        resident = phase.GridCarrier.create(plan)
        resident_phase_stats = phase.m119.Stats()
        phase.m119.load_factor_seed(resident, plan, program, resident_phase_stats)
        for ordinal in range(len(plan.operations)):
            phase.m119.apply_operation(resident, plan, program, ordinal, resident_phase_stats)

        def resident_boundary_only() -> tuple[int, ...]:
            value = phase.resident_butterfly_contract(resident, plan, phase.ButterflyStats())
            return tuple(int(coordinate) for coordinate in phase.pair.split_to_full(value))

        def descriptor_boundary_only() -> tuple[int, ...]:
            return phase.compact_butterfly_boundary(plan, program)[0]

        unary, edges = oracle.runtime_weights(n, "PRIMARY")

        def gray_boundary_only() -> tuple[int, ...]:
            histogram, _ = oracle.gray_histogram(n, unary, edges)
            return oracle.canonical_histogram(histogram)

        restoring_carrier = phase.GridCarrier.create(plan)

        def restoring_transaction() -> tuple[int, ...]:
            return phase.execute_transaction(restoring_carrier, plan, "PRIMARY").boundary

        boundaries = (
            resident_boundary_only(),
            descriptor_boundary_only(),
            gray_boundary_only(),
            restoring_transaction(),
        )
        if not all(value == boundaries[0] for value in boundaries):
            raise RuntimeError("M120 benchmark boundary mismatch")
        samples = 31 if n < 4 else 17
        cases.append({
            "n": n,
            "boundary": list(boundaries[0]),
            "all_boundaries_equal": True,
            "resident_phase_factor_butterfly_boundary_only": measure(
                resident_boundary_only, 5, samples
            ),
            "compact_descriptor_butterfly_boundary_only": measure(
                descriptor_boundary_only, 5, samples
            ),
            "gray_delta_global_histogram_boundary_only": measure(
                gray_boundary_only, 2, 11 if n < 4 else 5
            ),
            "full_restoring_phase_transaction": measure(
                restoring_transaction, 5, samples
            ),
            "restoring_carrier_restored_after_timing": restoring_carrier.all_zero(),
        })
        for ordinal in reversed(range(len(plan.operations))):
            phase.m119.apply_operation(
                resident,
                plan,
                program,
                ordinal,
                resident_phase_stats,
                inverse=True,
            )
        phase.m119.unload_factor_seed(resident, plan, resident_phase_stats)
        if not resident.all_zero():
            raise RuntimeError("M120 read-only resident benchmark carrier did not restore")

    result = {
        "experiment": "M120_WARM_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_TIMING",
        "result": "PASS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warm_execution": True,
        "cases": cases,
        "resident_phase_path_scope": "BOUND_RUNTIME_FACTOR_CELLS_READ_BY_TOPOLOGY_DERIVED_BUTTERFLY_AND_FINAL_SCALAR_LIFT",
        "compact_descriptor_path_scope": "PUBLIC_RUNTIME_WEIGHTS_TO_IDENTICAL_KRONECKER_INTERFACE_RECURRENCE_AND_FINAL_SCALAR_LIFT",
        "gray_path_scope": "GLOBAL_GRAY_ASSIGNMENT_17_BIN_CHARACTER_HISTOGRAM",
        "full_transaction_scope": "FACTOR_SEED_LOAD_NATIVE_PHASE_UPDATES_BUTTERFLY_BOUNDARY_REVERSE_AND_EXACT_UNLOAD",
        "resident_and_descriptor_interface_recurrences_are_identical": True,
        "resident_and_descriptor_row_diagonal_generation_are_not_operation_matched": True,
        "timing_is_observational_not_used_for_advantage_claim": True,
        "process_max_rss_kib_after_all_paths": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_is_process_wide_not_path_attributed": True,
        "catvm_boundary_used": False,
        "controller_backend_traffic_bits": 0,
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
