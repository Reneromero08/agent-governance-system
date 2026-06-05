#!/usr/bin/env python3
"""
Align external waveform captures to phase2_marker_harness.csv segments.

This is an offline analysis helper. It does not touch hardware, MSRs, PCI config,
or firmware. It accepts a marker CSV from phase2_marker_harness.c and a waveform
CSV exported by a scope or logic analyzer.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_HZ = 2_670_000.0


@dataclass(frozen=True)
class Marker:
    segment: int
    tsc: int
    state: str
    edge: int


@dataclass(frozen=True)
class Sample:
    t: float
    y: float


def parse_marker(path: Path) -> list[Marker]:
    rows: list[Marker] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                Marker(
                    segment=int(row["segment"], 0),
                    tsc=int(row["tsc"], 0),
                    state=row["state"].strip().lower(),
                    edge=int(row["edge"], 0),
                )
            )
    if len(rows) < 3:
        raise SystemExit("marker CSV needs at least 3 segments")
    return rows


def pick_column(fieldnames: list[str], requested: str | None, candidates: Iterable[str]) -> str:
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"column {requested!r} not found; available: {', '.join(fieldnames)}")
        return requested
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise SystemExit(f"could not infer column; available: {', '.join(fieldnames)}")


def parse_wave(path: Path, time_col: str | None, value_col: str | None, sample_rate: float | None) -> list[Sample]:
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        if not reader.fieldnames:
            raise SystemExit("waveform CSV has no header")
        y_col = pick_column(reader.fieldnames, value_col, ("value", "voltage", "ch1", "channel1", "y"))
        t_col = None if sample_rate else pick_column(reader.fieldnames, time_col, ("time", "t", "seconds", "sec"))
        samples: list[Sample] = []
        for i, row in enumerate(reader):
            try:
                t = i / sample_rate if sample_rate else float(row[t_col])  # type: ignore[arg-type]
                y = float(row[y_col])
            except (KeyError, TypeError, ValueError):
                continue
            samples.append(Sample(t, y))
    if len(samples) < 16:
        raise SystemExit("waveform CSV needs at least 16 numeric samples")
    return samples


def segment_windows(markers: list[Marker], wave: list[Sample], segment_us: float, start_offset: float) -> list[tuple[Marker, list[Sample]]]:
    wave_start = wave[0].t + start_offset
    windows: list[tuple[Marker, list[Sample]]] = []
    cursor = 0
    for marker in markers:
        a = wave_start + marker.segment * segment_us / 1_000_000.0
        b = a + segment_us / 1_000_000.0
        while cursor < len(wave) and wave[cursor].t < a:
            cursor += 1
        j = cursor
        bucket: list[Sample] = []
        while j < len(wave) and wave[j].t < b:
            bucket.append(wave[j])
            j += 1
        windows.append((marker, bucket))
    return windows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def goertzel_amplitude(samples: list[Sample], freq_hz: float) -> float:
    if len(samples) < 4:
        return float("nan")
    dt = (samples[-1].t - samples[0].t) / max(1, len(samples) - 1)
    if dt <= 0:
        return float("nan")
    sample_rate = 1.0 / dt
    k = int(0.5 + (len(samples) * freq_hz / sample_rate))
    omega = 2.0 * math.pi * k / len(samples)
    coeff = 2.0 * math.cos(omega)
    q0 = q1 = q2 = 0.0
    dc = mean([s.y for s in samples])
    for sample in samples:
        q0 = coeff * q1 - q2 + (sample.y - dc)
        q2 = q1
        q1 = q0
    power = q1 * q1 + q2 * q2 - coeff * q1 * q2
    return math.sqrt(max(power, 0.0)) / len(samples)


def summarize(windows: list[tuple[Marker, list[Sample]]], shuffle: bool) -> list[dict[str, str]]:
    markers = [m for m, _ in windows]
    buckets = [b for _, b in windows]
    if shuffle:
        rng = random.Random(0x44)
        shuffled = buckets[:]
        rng.shuffle(shuffled)
        windows = list(zip(markers, shuffled))

    by_state: dict[str, list[list[Sample]]] = {}
    for marker, bucket in windows:
        if bucket:
            by_state.setdefault(marker.state, []).append(bucket)

    out: list[dict[str, str]] = []
    for state in sorted(by_state):
        segment_means = [mean([s.y for s in bucket]) for bucket in by_state[state]]
        amps = [goertzel_amplitude(bucket, TARGET_HZ) for bucket in by_state[state]]
        out.append(
            {
                "state": state,
                "segments": str(len(by_state[state])),
                "mean": f"{mean(segment_means):.9g}",
                "mean_std": f"{statistics.pstdev(segment_means):.9g}" if len(segment_means) > 1 else "0",
                "target_hz": f"{TARGET_HZ:.1f}",
                "target_amp_mean": f"{mean(amps):.9g}",
                "target_amp_std": f"{statistics.pstdev(amps):.9g}" if len(amps) > 1 else "0",
                "mode": "shuffled_null" if shuffle else "aligned",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "state", "segments", "mean", "mean_std", "target_hz", "target_amp_mean", "target_amp_std"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, markers: list[Marker], wave: list[Sample], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aligned = [r for r in rows if r["mode"] == "aligned"]
    null = [r for r in rows if r["mode"] == "shuffled_null"]
    lines = [
        "# Phase 2 External Alignment Report",
        "",
        f"Marker CSV: {args.marker}",
        f"Waveform CSV: {args.wave}",
        f"Segments: {len(markers)}",
        f"Waveform samples: {len(wave)}",
        f"Segment length us: {args.segment_us}",
        f"Start offset s: {args.start_offset}",
        f"Target artifact frequency Hz: {TARGET_HZ:.1f}",
        "",
        "## Interpretation Gate",
        "",
        "Accept only if aligned state differences are repeatable and stronger than shuffled-null differences.",
        "Reject the fixed 2.67 MHz line unless its amplitude or phase changes by marker state and survives nulls.",
        "",
        "## Aligned State Summary",
        "",
    ]
    for row in aligned:
        lines.append(
            f"- {row['state']}: mean={row['mean']} std={row['mean_std']} "
            f"target_amp={row['target_amp_mean']} target_amp_std={row['target_amp_std']}"
        )
    lines.extend(["", "## Shuffled Null Summary", ""])
    for row in null:
        lines.append(
            f"- {row['state']}: mean={row['mean']} std={row['mean_std']} "
            f"target_amp={row['target_amp_mean']} target_amp_std={row['target_amp_std']}"
        )
    lines.extend(
        [
            "",
            "## Verdict Rule",
            "",
            "This script does not declare CPU_SINGS by itself. It prepares evidence for the lab report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--wave", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--segment-us", type=float, default=50_000.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--time-column")
    parser.add_argument("--value-column")
    parser.add_argument("--sample-rate", type=float)
    args = parser.parse_args()

    markers = parse_marker(args.marker)
    wave = parse_wave(args.wave, args.time_column, args.value_column, args.sample_rate)
    windows = segment_windows(markers, wave, args.segment_us, args.start_offset)
    rows = summarize(windows, shuffle=False) + summarize(windows, shuffle=True)
    write_csv(args.out_csv, rows)
    write_report(args.out_report, args, markers, wave, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
