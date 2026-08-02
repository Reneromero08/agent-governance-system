#!/usr/bin/env python3
"""Warm observational timing for the M119 grid phase-factor diagnostic."""

from __future__ import annotations

import gc
import json
import platform
import resource
import statistics
import sys
import time
from typing import Callable

import f17_runtime_weighted_grid_phase_factor_closure as phase
import f17_runtime_weighted_grid_phase_factor_closure_oracle as oracle


def distribution(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)
    return {
        "samples": len(ordered),
        "minimum_ns": ordered[0],
        "median_ns": int(statistics.median(ordered)),
        "p90_ns": ordered[min(len(ordered) - 1, (9 * len(ordered)) // 10)],
        "maximum_ns": ordered[-1],
    }


def measure(call: Callable[[], object], warmups: int, samples: int) -> dict[str, int]:
    for _ in range(warmups):
        call()
    values: list[int] = []
    gc.disable()
    try:
        for _ in range(samples):
            start = time.perf_counter_ns()
            call()
            values.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()
    return distribution(values)


def main() -> int:
    cases: list[dict[str, object]] = []
    for n in phase.SIZES:
        plan = phase.compile_topology(n)
        program = phase.bind_runtime_program(plan, "PRIMARY")
        carrier = phase.GridCarrier.create(plan)

        def phase_call() -> tuple[int, ...]:
            return phase.execute_transaction(carrier, plan, "PRIMARY").boundary

        def compact_call() -> tuple[int, ...]:
            return phase.compact_transfer_boundary(plan, program)[0]

        unary, edge_weights = oracle.runtime_weights(n, "PRIMARY")

        def histogram_call() -> tuple[int, ...]:
            histogram, _ = oracle.gray_delta_histogram(n, unary, edge_weights)
            return oracle.canonical_power_basis(histogram)

        phase_boundary = phase_call()
        compact_boundary = compact_call()
        histogram_boundary = histogram_call()
        if not phase_boundary == compact_boundary == histogram_boundary:
            raise RuntimeError("M119 benchmark boundaries differ")
        samples = 31 if n < 4 else 11
        histogram_samples = 11 if n < 4 else 5
        cases.append({
            "n": n,
            "boundary": list(phase_boundary),
            "all_boundaries_equal": True,
            "phase_restoring_transaction": measure(phase_call, 5, samples),
            "compact_transfer_boundary_only": measure(compact_call, 5, samples),
            "gray_delta_global_histogram_boundary_only": measure(
                histogram_call,
                2,
                histogram_samples,
            ),
            "phase_carrier_generation_after_timing": carrier.generation,
            "phase_carrier_lease_after_timing": carrier.lease,
            "phase_carrier_restored_after_timing": carrier.all_zero(),
        })
    result = {
        "experiment": "M119_WARM_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_TIMING",
        "result": "PASS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warm_execution": True,
        "cases": cases,
        "phase_path_scope": "FACTOR_SEED_LOAD_NATIVE_PREP_UNARY_EDGE_PHASE_COMPOSITION_FINAL_TRANSFER_BOUNDARY_REVERSE_RESTORATION",
        "compact_transfer_path_scope": "FINAL_BOUNDARY_TRANSFER_FROM_RUNTIME_WEIGHTS_ONLY",
        "gray_delta_histogram_path_scope": "FINAL_BOUNDARY_GRAY_CODE_GLOBAL_ASSIGNMENT_CHARACTER_HISTOGRAM_WITH_ONE_CHANGED_BIT_ENERGY_DELTAS",
        "timed_paths_are_not_operation_matched": True,
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
