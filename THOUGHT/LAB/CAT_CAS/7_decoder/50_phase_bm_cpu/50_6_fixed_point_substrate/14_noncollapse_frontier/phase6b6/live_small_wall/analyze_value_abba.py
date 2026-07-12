#!/usr/bin/env python3
"""Analyze continuous ABBA/BAAB public-operand preparation pilots."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


class ValueAnalysisError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueAnalysisError(message)


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
    lockin = load_jsonl(run / "runtime" / "LOCKIN_IQ.jsonl")
    analysis = load_json(run / "FIRST_LIGHT_ANALYSIS.json")
    require(len(trace) == len(lockin) == len(analysis["ring_slots"]) == 16, "slot coverage mismatch")
    expected_values = {
        "value-forward": [125, 131, 131, 125],
        "value-reverse": [131, 125, 125, 131],
        "value-equal": [42, 42, 42, 42],
    }[expected_variant]
    observed_values = [trace[index]["orbit_value"] for index in range(6, 10)]
    require(observed_values == expected_values, f"operand schedule mismatch: {run_id}")
    ring = [float(analysis["ring_slots"][index]["mean_ring_period"]) for index in range(6, 10)]
    z = [complex(float(lockin[index]["lockin_i"]), float(lockin[index]["lockin_q"])) for index in range(6, 10)]
    ring_contrast = (ring[0] + ring[3] - ring[1] - ring[2]) / 2.0
    complex_contrast = (z[0] + z[3] - z[1] - z[2]) / 2.0
    return {
        "run_id": run_id,
        "variant": expected_variant,
        "operand_values": observed_values,
        "ring_slots": ring,
        "ring_symmetric_contrast": ring_contrast,
        "complex_symmetric_contrast": {
            "I": complex_contrast.real,
            "Q": complex_contrast.imag,
            "magnitude": abs(complex_contrast),
        },
    }


def analyze(root: Path, equal_ids: list[str], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    equals = [session(root, run_id, "value-equal") for run_id in equal_ids]
    pair_results: list[dict[str, Any]] = []
    for forward_id, reverse_id in pairs:
        forward = session(root, forward_id, "value-forward")
        reverse = session(root, reverse_id, "value-reverse")
        ring_value = (
            forward["ring_symmetric_contrast"] - reverse["ring_symmetric_contrast"]
        ) / 2.0
        fz = complex(
            forward["complex_symmetric_contrast"]["I"],
            forward["complex_symmetric_contrast"]["Q"],
        )
        rz = complex(
            reverse["complex_symmetric_contrast"]["I"],
            reverse["complex_symmetric_contrast"]["Q"],
        )
        zv = (fz - rz) / 2.0
        pair_results.append(
            {
                "forward": forward,
                "reverse": reverse,
                "R_value_lower_minus_mirror_ring": ring_value,
                "R_value_lower_minus_mirror_complex": {
                    "I": zv.real,
                    "Q": zv.imag,
                    "magnitude": abs(zv),
                },
            }
        )
    null_ceiling = max(abs(row["ring_symmetric_contrast"]) for row in equals)
    values = [row["R_value_lower_minus_mirror_ring"] for row in pair_results]
    same_sign = bool(values) and all(value < 0 for value in values) or bool(values) and all(value > 0 for value in values)
    beyond_null = bool(values) and all(abs(value) > null_ceiling for value in values)
    accepted = len(values) >= 3 and same_sign and beyond_null
    return {
        "schema_id": "CAT_CAS_VALUE_ABBA_ANALYSIS_V1",
        "coordinate": "crossed symmetric lower-minus-mirror receiver response",
        "equal_controls": equals,
        "equal_curvature_null_ceiling": null_ceiling,
        "pairs": pair_results,
        "ring_R_value_mean": statistics.fmean(values) if values else None,
        "ring_R_value_stdev": statistics.pstdev(values) if values else None,
        "all_pair_signs_agree": same_sign,
        "all_pairs_exceed_equal_null": beyond_null,
        "OBSERVABLE_OPERAND_STATE_FOUND": accepted,
        "claim_ceiling": "public operand-dependent physical preparation; no relational memory, restoration, orientation, or Small Wall claim",
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
