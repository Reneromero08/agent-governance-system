#!/usr/bin/env python3
"""Analyze closed CAT_CAS-owned cache-response ABBA/BAAB pilots."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


class OccupancyAnalysisError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise OccupancyAnalysisError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def session(root: Path, run_id: str, expected_variant: str) -> dict[str, Any]:
    run = root / run_id
    final = load_json(run / "FINAL_RESULT.json")
    require(final["status"] == "GATE_A_FIRST_LIGHT_COMPLETE", f"incomplete run: {run_id}")
    require(final["restoration_complete"], f"restoration incomplete: {run_id}")
    require(final["pilot_variant"] == expected_variant, f"variant mismatch: {run_id}")
    trace = load_jsonl(run / "runtime" / "slot_trace.jsonl")
    analysis = load_json(run / "FIRST_LIGHT_ANALYSIS.json")
    require(len(trace) == len(analysis["ring_slots"]) == 16, "slot coverage mismatch")
    require(
        analysis["measurement_mode"] == "catcas_owned_cache_response_cycles",
        f"measurement mode mismatch: {run_id}",
    )
    expected_footprints = {
        "occupancy-forward": [256 * 1024, 32 * 1024 * 1024, 32 * 1024 * 1024, 256 * 1024],
        "occupancy-reverse": [32 * 1024 * 1024, 256 * 1024, 256 * 1024, 32 * 1024 * 1024],
        "occupancy-equal": [4 * 1024 * 1024] * 4,
    }[expected_variant]
    observed = [int(trace[index]["working_set_bytes"]) for index in range(6, 10)]
    require(observed == expected_footprints, f"working-set schedule mismatch: {run_id}")
    response = [
        float(analysis["ring_slots"][index]["mean_response_cycles"])
        for index in range(6, 10)
    ]
    contrast = (response[0] + response[3] - response[1] - response[2]) / 2.0
    return {
        "run_id": run_id,
        "variant": expected_variant,
        "working_set_bytes": observed,
        "response_cycles_per_touch": response,
        "symmetric_contrast_cycles_per_touch": contrast,
    }


def analyze(root: Path, equal_ids: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    equals = [session(root, run_id, "occupancy-equal") for run_id in equal_ids]
    pair_results: list[dict[str, Any]] = []
    for forward_id, reverse_id in pairs:
        forward = session(root, forward_id, "occupancy-forward")
        reverse = session(root, reverse_id, "occupancy-reverse")
        crossed = (
            forward["symmetric_contrast_cycles_per_touch"]
            - reverse["symmetric_contrast_cycles_per_touch"]
        ) / 2.0
        pair_results.append(
            {
                "forward": forward,
                "reverse": reverse,
                "crossed_occupancy_response_cycles_per_touch": crossed,
            }
        )
    null_ceiling = max(
        abs(row["symmetric_contrast_cycles_per_touch"]) for row in equals
    )
    values = [row["crossed_occupancy_response_cycles_per_touch"] for row in pair_results]
    same_sign = bool(values) and (
        all(value < 0 for value in values) or all(value > 0 for value in values)
    )
    beyond_null = bool(values) and all(abs(value) > null_ceiling for value in values)
    accepted = len(values) >= 3 and same_sign and beyond_null
    return {
        "schema_id": "CAT_CAS_OWNED_CACHE_RESPONSE_ABBA_ANALYSIS_V1",
        "coordinate": "crossed symmetric response of a fixed CAT_CAS-owned measurement buffer",
        "equal_controls": equals,
        "equal_curvature_null_ceiling_cycles_per_touch": null_ceiling,
        "pairs": pair_results,
        "crossed_response_mean_cycles_per_touch": statistics.fmean(values) if values else None,
        "crossed_response_stdev_cycles_per_touch": statistics.pstdev(values) if values else None,
        "all_pair_signs_agree": same_sign,
        "all_pairs_exceed_equal_null": beyond_null,
        "OBSERVABLE_OCCUPANCY_RESPONSE_FOUND": accepted,
        "claim_ceiling": (
            "CAT_CAS-owned shared-cache occupancy changes the aggregate timing response of "
            "a separate CAT_CAS-owned fixed workload; no outside-process observation, "
            "private-data inference, relational memory, or Small Wall claim"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--equal", action="append", required=True)
    parser.add_argument("--pair", action="append", nargs=2, metavar=("FORWARD", "REVERSE"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.runs_root, args.equal, [tuple(pair) for pair in args.pair])
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
