#!/usr/bin/env python3
"""Warm local timing comparison for the bounded M118 diagnostic."""

from __future__ import annotations

import json
import platform
import resource
import statistics
import sys
import time
from typing import Callable

import f17_variable_rank_nonseparable_tensor_coupling as production


WARMUPS = 20
SAMPLES = 101


def sample(operation: Callable[[], object]) -> tuple[object, list[int]]:
    result: object = None
    for _ in range(WARMUPS):
        result = operation()
    timings: list[int] = []
    for _ in range(SAMPLES):
        started = time.perf_counter_ns()
        result = operation()
        timings.append(time.perf_counter_ns() - started)
    return result, timings


def summarize(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "minimum_ns": ordered[0],
        "median_ns": int(statistics.median(ordered)),
        "p90_ns": ordered[(9 * len(ordered)) // 10],
        "maximum_ns": ordered[-1],
    }


def result() -> dict[str, object]:
    carrier = production.TensorCarrier.create()
    phase_result, phase_timings = sample(
        lambda: production.execute_transaction(carrier, "PRIMARY").boundary
    )
    compact_result, compact_timings = sample(
        lambda: production.compact_factor_boundary("PRIMARY")[0]
    )
    if phase_result != compact_result:
        raise RuntimeError("warm benchmark paths disagree on the final boundary")
    return {
        "experiment": "M118_WARM_DIRECT_PROCESS_PHASE_VERSUS_COMPACT_FACTOR_TIMING",
        "result": "PASS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmups_per_path": WARMUPS,
        "phase_path": summarize(phase_timings),
        "phase_path_scope": "FORWARD_FINAL_BOUNDARY_INVERSE_RESTORATION",
        "compact_factor_path": summarize(compact_timings),
        "compact_factor_path_scope": "FINAL_BOUNDARY_EVALUATION_ONLY",
        "timed_paths_are_not_operation_matched": True,
        "boundaries_equal": True,
        "boundary": list(phase_result),
        "process_max_rss_kib_after_both_paths": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "rss_is_process_wide_not_path_attributed": True,
        "timing_is_local_observational_not_used_for_advantage_claim": True,
        "controller_backend_traffic_bits": 0,
        "catvm_boundary_used": False,
    }


def main() -> None:
    payload = result()
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        return
    if len(sys.argv) != 1:
        raise RuntimeError(
            "usage: benchmark_f17_variable_rank_nonseparable_tensor_coupling.py [--output PATH]"
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
