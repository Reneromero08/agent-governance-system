"""Lightweight mutable probe for the fixed-resident torus construction."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from fixed_resident_torus import (
    DEFAULT_PHASE_MODULI,
    RESTORATION_MAX,
    PhaseProgram,
    borrowed_carrier,
    canonical_bytes,
    collapse_boundary,
    engine_fingerprint,
    execute_catalytic,
    maximum_abs_error,
    sha256_bytes,
    source_no_smuggle,
)


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "FIXED_RESIDENT_RESULTS.json"
WIDTH = 31
TARGET = 17
SIZES = (16, 64, 256, 1024, 4096)


def weights(size: int, family: int) -> tuple[int, ...]:
    return tuple(
        1
        + (
            index * (17 + 2 * family)
            + index * index * (5 + family)
            + 7
            + family * 11
        )
        % (WIDTH - 1)
        for index in range(size)
    )


def compact_dp(program: PhaseProgram) -> int:
    modulus = math.prod(program.phase_moduli)
    state = np.zeros(program.residue_modulus, dtype=np.int64)
    state[0] = 1
    for shift in program.weights:
        state = (state + np.roll(state, shift)) % modulus
    return int(state[program.target_residue])


def timed(callable_: Any, repeats: int = 3) -> tuple[Any, int]:
    durations: list[int] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = callable_()
        durations.append(time.perf_counter_ns() - started)
    return result, int(statistics.median(durations))


def run() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    controls: dict[str, Any] | None = None
    reused_carrier: np.ndarray | None = None
    previous_program: PhaseProgram | None = None
    for family in range(2):
        for size in SIZES:
            program = PhaseProgram(
                name=f"fixed-resident-family-{family}-n-{size}",
                residue_modulus=WIDTH,
                phase_moduli=DEFAULT_PHASE_MODULI,
                weights=weights(size, family),
                target_residue=TARGET,
            )
            carrier = (
                borrowed_carrier(program, identity=family * 100 + size)
                if reused_carrier is None
                else reused_carrier
            )
            if carrier.shape != borrowed_carrier(program).shape:
                raise RuntimeError("fixed carrier geometry unexpectedly changed")
            execution, phase_ns = timed(
                lambda: execute_catalytic(program, carrier), repeats=1
            )
            boundary = collapse_boundary(program, execution.result_latch)
            expected, dp_ns = timed(lambda: compact_dp(program))

            if controls is None and size == 64:
                wrong = execute_catalytic(
                    program, carrier, inverse_mode="wrong_program"
                )
                omitted = execute_catalytic(
                    program, carrier, inverse_mode="omitted"
                )
                removed = execute_catalytic(
                    program, carrier, forward_mode="removed"
                )
                scrambled = execute_catalytic(
                    program, carrier, forward_mode="scramble_geometry"
                )
                controls = {
                    "omitted_inverse_error": omitted.restoration_max_abs,
                    "removed_transform_latch_delta": maximum_abs_error(
                        removed.result_latch, execution.result_latch
                    ),
                    "scrambled_geometry_latch_delta": maximum_abs_error(
                        scrambled.result_latch, execution.result_latch
                    ),
                    "steps": size,
                    "wrong_program_inverse_error": wrong.restoration_max_abs,
                }
                controls["passed"] = bool(
                    controls["wrong_program_inverse_error"] > 1.0e-3
                    and controls["omitted_inverse_error"] > 1.0e-3
                    and controls["removed_transform_latch_delta"] > 1.0e-3
                    and controls["scrambled_geometry_latch_delta"] > 1.0e-3
                )
            old_layered_cells = (
                len(program.phase_moduli)
                * (len(program.weights) + 1)
                * 2
                * program.residue_modulus
            )
            record = {
                "boundary_count": boundary.count_mod_crt,
                "boundary_valid": boundary.valid,
                "carrier_reused_from_previous_program": (
                    reused_carrier is not None
                ),
                "compact_dp_count": expected,
                "compact_dp_ns": dp_ns,
                "displacement_l2": execution.displacement_l2,
                "family": family,
                "fixed_resident_cells": execution.resident_complex_cells,
                "fixed_to_layered_cell_ratio": (
                    execution.resident_complex_cells / old_layered_cells
                ),
                "history_factor_count": execution.history_factor_count,
                "phase_end_to_end_ns": phase_ns,
                "phase_over_compact_dp_wall_ratio": phase_ns / dp_ns,
                "program_fingerprint": program.fingerprint,
                "restoration_max_abs": execution.restoration_max_abs,
                "root_distance_max": max(boundary.root_distances),
                "steps": size,
            }
            record["passed"] = bool(
                boundary.valid
                and boundary.count_mod_crt == expected
                and execution.restoration_max_abs <= RESTORATION_MAX
                and execution.displacement_l2 > 1.0
            )
            records.append(record)
            reused_carrier = execution.restored_carrier
            previous_program = program

    if reused_carrier is None or previous_program is None:
        raise RuntimeError("probe produced no carrier")
    fresh = borrowed_carrier(previous_program, identity=999)
    reuse_is_actual_not_fresh = maximum_abs_error(reused_carrier, fresh) > 1.0e-3
    result = {
        "construction": "FIXED_RESIDENT_REVERSIBLE_TORUS_DEVELOPMENT",
        "controls": controls,
        "engine_fingerprint": engine_fingerprint(),
        "fixed_carrier_cells": int(reused_carrier.size),
        "maximum_restoration_error": max(
            record["restoration_max_abs"] for record in records
        ),
        "maximum_steps": max(SIZES),
        "minimum_layer_compression": min(
            1.0 / record["fixed_to_layered_cell_ratio"] for record in records
        ),
        "no_hardcoded_step_limit": True,
        "records": records,
        "reuse_consumed_actual_restored_carrier": reuse_is_actual_not_fresh,
        "schema": "cat_cas.frontier.fixed_resident.v0",
        "source_no_smuggle": source_no_smuggle(),
        "status": (
            "DEVELOPMENT_PASS"
            if all(record["passed"] for record in records)
            and controls is not None
            and controls["passed"]
            and reuse_is_actual_not_fresh
            and source_no_smuggle()["passed"]
            else "DEVELOPMENT_FAIL"
        ),
    }
    result["results_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in result.items() if key != "results_sha256"}
        )
    )
    RESULTS_PATH.write_bytes(canonical_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
