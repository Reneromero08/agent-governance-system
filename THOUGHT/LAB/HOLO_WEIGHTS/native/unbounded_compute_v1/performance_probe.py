"""Measured phase, compact-DP, and explicit-path timing comparison."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from phase_path_engine import (
    HoloSource,
    borrowed_carrier,
    canonical_bytes,
    classical_path_work,
    collapse_boundary,
    compile_holo,
    compute_leverage,
    execute_catalytic,
    load_holo,
)


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "PERFORMANCE_RESULTS.json"
REPORT_PATH = HERE / "PERFORMANCE_REPORT.md"


def median_ns(action: Callable[[], Any], repeats: int) -> int:
    samples: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        action()
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


def compact_dp(source: HoloSource) -> int:
    modulus = math.prod(source.phase_moduli)
    counts = np.zeros(source.residue_modulus, dtype=np.int64)
    counts[0] = 1
    for weight in source.weights:
        counts = (counts + np.roll(counts, weight)) % modulus
    return int(counts[source.target_residue])


def explicit_gray(source: HoloSource) -> int:
    modulus = source.residue_modulus
    target = source.target_residue
    count = 0
    residue = 0
    previous_gray = 0
    if residue == target:
        count += 1
    for index in range(1, 1 << len(source.weights)):
        gray = index ^ (index >> 1)
        changed = gray ^ previous_gray
        bit = changed.bit_length() - 1
        if gray & changed:
            residue = (residue + source.weights[bit]) % modulus
        else:
            residue = (residue - source.weights[bit]) % modulus
        if residue == target:
            count += 1
        previous_gray = gray
    return count % math.prod(source.phase_moduli)


def prospective_source(size: int) -> HoloSource:
    return load_holo(HERE / "programs" / f"phase_path_alpha_{size}.holo")


def run() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for size in (16, 32, 64, 128, 256):
        source = prospective_source(size)
        compiled = compile_holo(source)
        carrier = borrowed_carrier(compiled, identity=515)

        def phase_action() -> int:
            execution = execute_catalytic(compiled, carrier)
            boundary = collapse_boundary(compiled, execution.result_latch)
            if not boundary.valid:
                raise RuntimeError("phase benchmark boundary invalid")
            return int(boundary.count_mod_crt)

        phase_ns = median_ns(phase_action, repeats=5)
        dp_ns = median_ns(lambda: compact_dp(source), repeats=11)
        execution = execute_catalytic(compiled, carrier)
        boundary = collapse_boundary(compiled, execution.result_latch)
        expected = compact_dp(source)
        if boundary.count_mod_crt != expected:
            raise RuntimeError("performance comparison result mismatch")
        explicit_ns: int | None = None
        explicit_result: int | None = None
        if size == 16:
            explicit_ns = median_ns(lambda: explicit_gray(source), repeats=3)
            explicit_result = explicit_gray(source)
            if explicit_result != expected:
                raise RuntimeError("explicit benchmark result mismatch")
        records.append(
            {
                "compact_dp_ns_median": dp_ns,
                "count_mod_crt": expected,
                "declared_phase_work": execution.total_declared_work,
                "explicit_gray_ns_median": explicit_ns,
                "explicit_path_count": 1 << size,
                "explicit_result": explicit_result,
                "gamma_path_work": compute_leverage(execution),
                "phase_end_to_end_ns_median": phase_ns,
                "phase_over_compact_dp_wall_ratio": phase_ns / dp_ns,
                "steps": size,
            }
        )
    result = {
        "environment_note": "wall timings are local medians, not identities",
        "records": records,
        "resource_conclusion": {
            "advantage_over_compact_dp": False,
            "complete_path_modes": 0,
            "gamma_path_work_grows": all(
                right["gamma_path_work"] > left["gamma_path_work"]
                for left, right in zip(records, records[1:])
            ),
            "native_state_growth": "O(n*k*M) reversible phase relations",
            "phase_wall_faster_than_explicit_at_16": (
                records[0]["phase_end_to_end_ns_median"]
                < records[0]["explicit_gray_ns_median"]
            ),
        },
        "schema": "cat_cas.toroidal_path_sum.performance.v1",
    }
    return result


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.write_bytes(canonical_bytes(result))
    lines = [
        "# Toroidal path-sum performance",
        "",
        "Wall timings are local medians and are not evidence identities.",
        "",
    ]
    for record in result["records"]:
        lines.append(
            f"- n={record['steps']}: phase "
            f"{record['phase_end_to_end_ns_median']} ns; compact DP "
            f"{record['compact_dp_ns_median']} ns; Gamma "
            f"{record['gamma_path_work']:.6g}"
        )
    lines.extend(
        [
            "",
            "Gamma is measured against explicit binary path-work. The compact",
            "integer DP remains faster than this NumPy phase reference, so no",
            "advantage over the best conventional compact algorithm is claimed.",
            "",
        ]
    )
    REPORT_PATH.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    performance = run()
    write_result(performance)
    print(
        json.dumps(
            {
                "gamma_grows": performance["resource_conclusion"][
                    "gamma_path_work_grows"
                ],
                "records": len(performance["records"]),
            },
            sort_keys=True,
        )
    )
