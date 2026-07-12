#!/usr/bin/env python3
"""Resolve post-drive carrier memory below the 0.5 second slot average."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUNS = (
    "gate_a_first_light_20260711i",
    "gate_a_first_light_20260711j",
    "pilot_anchor_np_20260711b",
    "pilot_anchor_sham_20260711a",
    "pilot_impulse_20260711a",
    "pilot_step_sham_20260711a",
)
RAW_DTYPE = np.dtype([("timestamp_tsc", "<u8"), ("ring_period", "<f8")])
BIN_EDGES_S = (0.0, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5)
TONE_HZ = math.exp(math.log(20.0)) * (1.0 + 0.013 * math.sin(2.399963))


class TransientError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TransientError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def lockin(timestamps: np.ndarray, values: np.ndarray, *, origin_tsc: float, tsc_hz: float) -> complex:
    require(len(values) >= 3, "transient lock-in window too short")
    centered = values - values.mean()
    window = np.hanning(len(values))
    seconds = (timestamps.astype(np.float64) - origin_tsc) / tsc_hz
    weight = window.sum()
    return complex(
        2.0 * np.sum(centered * window * np.cos(2.0 * math.pi * TONE_HZ * seconds)) / weight,
        2.0 * np.sum(centered * window * np.sin(2.0 * math.pi * TONE_HZ * seconds)) / weight,
    )


def summarize_reference(
    raw: np.ndarray,
    *,
    reference_tsc: float,
    phase_origin_tsc: float,
    tsc_hz: float,
    baseline: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamps = raw["timestamp_tsc"]
    for start_s, end_s in zip(BIN_EDGES_S, BIN_EDGES_S[1:]):
        start = reference_tsc + start_s * tsc_hz
        end = reference_tsc + end_s * tsc_hz
        selected = raw[(timestamps >= start) & (timestamps < end)]
        values = selected["ring_period"].astype(np.float64)
        require(len(values) >= 10, f"transient bin lacks samples: {start_s}-{end_s}")
        response = lockin(
            selected["timestamp_tsc"],
            values,
            origin_tsc=phase_origin_tsc,
            tsc_hz=tsc_hz,
        )
        rows.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "sample_count": int(len(values)),
                "mean_ring_delta": float(values.mean() - baseline),
                "median_ring_delta": float(np.median(values) - baseline),
                "ring_stderr": float(values.std() / math.sqrt(len(values))),
                "lockin_i": response.real,
                "lockin_q": response.imag,
                "lockin_magnitude": abs(response),
            }
        )
    return rows


def load_run(root: Path, run_id: str) -> dict[str, Any]:
    run = root / run_id
    final = json.loads((run / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    runtime = json.loads((run / "runtime" / "runtime_result.json").read_text(encoding="utf-8"))
    require(final["status"] == "GATE_A_FIRST_LIGHT_COMPLETE", f"run incomplete: {run_id}")
    raw = np.fromfile(run / "runtime" / "raw_samples.bin", dtype=RAW_DTYPE)
    origin = float(runtime["capture_origin_tsc"])
    tsc_hz = float(runtime["capture_tsc_hz"])
    phase_origin = origin + 3.0 * tsc_hz
    timestamps = raw["timestamp_tsc"]
    pre = raw[(timestamps >= origin + 2.0 * tsc_hz) & (timestamps < origin + 3.0 * tsc_hz)]
    require(len(pre) > 100, "pre-drive baseline incomplete")
    baseline = float(pre["ring_period"].mean())
    variant = final.get("pilot_variant", "pn")
    return {
        "run_id": run_id,
        "variant": variant,
        "baseline_ring_period": baseline,
        "after_impulse_time": summarize_reference(
            raw,
            reference_tsc=origin + 3.5 * tsc_hz,
            phase_origin_tsc=phase_origin,
            tsc_hz=tsc_hz,
            baseline=baseline,
        ),
        "after_full_step_time": summarize_reference(
            raw,
            reference_tsc=origin + 5.0 * tsc_hz,
            phase_origin_tsc=phase_origin,
            tsc_hz=tsc_hz,
            baseline=baseline,
        ),
        "restoration_complete": bool(final["restoration_complete"]),
    }


def analyze(root: Path, run_ids: tuple[str, ...]) -> dict[str, Any]:
    sessions = [load_run(root, run_id) for run_id in run_ids]
    full = [session for session in sessions if session["variant"] in {"pn", "np", "anchor-sham"}]
    impulse = [session for session in sessions if session["variant"] == "impulse"]
    sham = [session for session in sessions if session["variant"] == "step-sham"]
    require(len(full) >= 2 and impulse and sham, "history groups incomplete")

    def group_mean(group: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index in range(len(BIN_EDGES_S) - 1):
            rows = [session[key][index] for session in group]
            complex_values = [complex(row["lockin_i"], row["lockin_q"]) for row in rows]
            mean_complex = sum(complex_values) / len(complex_values)
            result.append(
                {
                    "start_s": rows[0]["start_s"],
                    "end_s": rows[0]["end_s"],
                    "session_count": len(rows),
                    "mean_ring_delta": float(np.mean([row["mean_ring_delta"] for row in rows])),
                    "between_session_ring_stdev": float(np.std([row["mean_ring_delta"] for row in rows])),
                    "mean_lockin_i": mean_complex.real,
                    "mean_lockin_q": mean_complex.imag,
                    "coherent_lockin_magnitude": abs(mean_complex),
                    "mean_lockin_magnitude": float(np.mean([abs(value) for value in complex_values])),
                }
            )
        return result

    full_summary = group_mean(full, "after_full_step_time")
    impulse_summary = group_mean(impulse, "after_impulse_time")
    sham_after_full = group_mean(sham, "after_full_step_time")
    sham_after_impulse = group_mean(sham, "after_impulse_time")

    full_contrast = [
        {
            "start_s": left["start_s"],
            "end_s": left["end_s"],
            "ring_delta_vs_sham": left["mean_ring_delta"] - right["mean_ring_delta"],
            "coherent_lockin_vs_sham": abs(
                complex(left["mean_lockin_i"], left["mean_lockin_q"])
                - complex(right["mean_lockin_i"], right["mean_lockin_q"])
            ),
        }
        for left, right in zip(full_summary, sham_after_full)
    ]
    impulse_contrast = [
        {
            "start_s": left["start_s"],
            "end_s": left["end_s"],
            "ring_delta_vs_sham": left["mean_ring_delta"] - right["mean_ring_delta"],
            "coherent_lockin_vs_sham": abs(
                complex(left["mean_lockin_i"], left["mean_lockin_q"])
                - complex(right["mean_lockin_i"], right["mean_lockin_q"])
            ),
        }
        for left, right in zip(impulse_summary, sham_after_impulse)
    ]

    return {
        "schema_id": "CAT_CAS_TRANSIENT_CARRIER_MEMORY_ANALYSIS_V1",
        "tone_hz": TONE_HZ,
        "bin_edges_s": list(BIN_EDGES_S),
        "sessions": sessions,
        "full_step_group": full_summary,
        "impulse_group": impulse_summary,
        "step_sham_after_full_time": sham_after_full,
        "step_sham_after_impulse_time": sham_after_impulse,
        "full_step_contrast": full_contrast,
        "impulse_contrast": impulse_contrast,
        "claim_ceiling": "transient sensor screen; repeatability and held-out dynamics still required",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.runs_root, tuple(args.runs or DEFAULT_RUNS))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
