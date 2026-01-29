"""
Q32 Phase 7: EEG ingestion for physical coupling harness

Loads OpenNeuro ds005383 (TMNRED Chinese semantic EEG) and converts to M/B CSV
format compatible with q32_physical_force_harness.py.

Dataset: https://openneuro.org/datasets/ds005383
- 30 subjects, ~400 trials each
- Task: fuzzy semantic target recognition in natural Chinese reading
- M variable: trial_type (target=1 semantic match, nontarget=0 mismatch)
- B variable: EEG amplitude in semantic processing window (200-500ms post-stimulus)

This does NOT prove "meaning is a physical force" - it shows the harness
works on real neuroscience data and establishes what ordinary semantic-neural
correlation looks like.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import h5py
    import numpy as np
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}. Install with: pip install h5py numpy")


def _utc_ts() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


@dataclasses.dataclass(frozen=True)
class EEGEpoch:
    onset: float
    sample: int
    trial_type: str
    is_target: bool
    value: int


def load_events_tsv(path: str) -> List[EEGEpoch]:
    """Load BIDS events.tsv and parse into epochs."""
    epochs = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            trial_type = row["trial_type"]
            is_target = trial_type.startswith("target")
            epochs.append(EEGEpoch(
                onset=float(row["onset"]),
                sample=int(row["sample"]),
                trial_type=trial_type,
                is_target=is_target,
                value=int(row["value"]),
            ))
    return epochs


def load_eeg_mat(path: str) -> np.ndarray:
    """Load MATLAB v7.3 .mat file (HDF5 format) containing EEG data."""
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"Expected 'data' key in .mat file, got: {list(f.keys())}")
        data = f["data"][:]
    return data


def extract_epoch_amplitude(
    eeg_data: np.ndarray,
    sample_idx: int,
    window_start_ms: int,
    window_end_ms: int,
    sampling_rate: float,
    channels: Optional[List[int]] = None,
) -> float:
    """
    Extract mean amplitude in a time window after stimulus onset.

    Args:
        eeg_data: (n_samples, n_channels) array
        sample_idx: stimulus onset sample index
        window_start_ms: start of window in ms post-stimulus (e.g., 200 for N400)
        window_end_ms: end of window in ms post-stimulus (e.g., 500 for N400)
        sampling_rate: Hz (e.g., 200)
        channels: list of channel indices to average, or None for all

    Returns:
        Mean amplitude across channels and time window
    """
    start_sample = sample_idx + int(window_start_ms * sampling_rate / 1000)
    end_sample = sample_idx + int(window_end_ms * sampling_rate / 1000)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(eeg_data.shape[0], end_sample)

    if start_sample >= end_sample:
        return float("nan")

    if channels is not None:
        segment = eeg_data[start_sample:end_sample, channels]
    else:
        segment = eeg_data[start_sample:end_sample, :]

    return float(np.mean(segment))


def ingest_session(
    mat_path: str,
    events_path: str,
    window_start_ms: int = 200,
    window_end_ms: int = 500,
    sampling_rate: float = 200.0,
    channels: Optional[List[int]] = None,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Ingest a single session and return M, B series.

    M = 1.0 for target (semantic match), 0.0 for nontarget
    B = mean EEG amplitude in semantic processing window

    Returns:
        (m_series, b_series, metadata)
    """
    epochs = load_events_tsv(events_path)
    eeg_data = load_eeg_mat(mat_path)

    m_series: List[float] = []
    b_series: List[float] = []

    for epoch in epochs:
        m = 1.0 if epoch.is_target else 0.0
        b = extract_epoch_amplitude(
            eeg_data,
            epoch.sample,
            window_start_ms,
            window_end_ms,
            sampling_rate,
            channels,
        )
        m_series.append(m)
        b_series.append(b)

    metadata = {
        "mat_path": mat_path,
        "events_path": events_path,
        "mat_sha256": _sha256_file(mat_path),
        "events_sha256": _sha256_file(events_path),
        "n_epochs": len(epochs),
        "n_targets": sum(1 for e in epochs if e.is_target),
        "n_nontargets": sum(1 for e in epochs if not e.is_target),
        "eeg_shape": list(eeg_data.shape),
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "sampling_rate": sampling_rate,
        "channels": channels,
    }

    return m_series, b_series, metadata


def write_mb_csv(path: str, m: List[float], b: List[float]) -> None:
    """Write M/B series to CSV for physical force harness."""
    if len(m) != len(b):
        raise ValueError("M and B length mismatch")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "b"])
        for mi, bi in zip(m, b):
            w.writerow([f"{mi:.10f}", f"{bi:.10f}"])
    os.replace(tmp, path)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def default_physdata_root() -> str:
    return os.path.join(
        "LAW", "CONTRACTS", "_runs", "q32_public", "physdata", "openneuro", "ds005383"
    )


def default_output_root() -> str:
    return os.path.join(
        "LAW", "CONTRACTS", "_runs", "q32_public", "datatrail"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q32 Phase 7: EEG ingestion for physical coupling harness"
    )
    p.add_argument(
        "--physdata_root",
        type=str,
        default=default_physdata_root(),
        help="Root directory of ds005383 dataset",
    )
    p.add_argument(
        "--subject",
        type=str,
        default="sub-01",
        help="Subject ID (default: sub-01)",
    )
    p.add_argument(
        "--session",
        type=str,
        default="ses-1",
        help="Session ID (default: ses-1)",
    )
    p.add_argument(
        "--window_start_ms",
        type=int,
        default=200,
        help="Start of amplitude window in ms post-stimulus (default: 200, N400 onset)",
    )
    p.add_argument(
        "--window_end_ms",
        type=int,
        default=500,
        help="End of amplitude window in ms post-stimulus (default: 500, N400 peak)",
    )
    p.add_argument(
        "--sampling_rate",
        type=float,
        default=200.0,
        help="EEG sampling rate in Hz (default: 200)",
    )
    p.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel indices (default: all channels)",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=default_output_root(),
        help="Output directory for CSV and receipt",
    )
    p.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Override CSV output path",
    )
    p.add_argument(
        "--receipt_out",
        type=str,
        default=None,
        help="Override receipt output path",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ts = _utc_ts()

    # Build paths
    mat_path = os.path.join(
        args.physdata_root,
        "derivatives", "preproc", args.subject, args.session,
        f"{args.subject}-{args.session}.mat"
    )
    events_path = os.path.join(
        args.physdata_root,
        args.subject, args.session, "eeg",
        f"{args.subject}_{args.session}_task-fuzzysemanticrecognition_events.tsv"
    )

    # Check files exist
    if not os.path.exists(mat_path):
        raise SystemExit(f"EEG .mat file not found: {mat_path}")
    if not os.path.exists(events_path):
        raise SystemExit(f"Events TSV not found: {events_path}")

    # Parse channels
    channels = None
    if args.channels:
        channels = [int(c.strip()) for c in args.channels.split(",")]

    # Ingest
    print(f"Loading EEG data from: {mat_path}")
    print(f"Loading events from: {events_path}")

    m, b, metadata = ingest_session(
        mat_path=mat_path,
        events_path=events_path,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
        sampling_rate=args.sampling_rate,
        channels=channels,
    )

    print(f"Extracted {len(m)} epochs ({metadata['n_targets']} targets, {metadata['n_nontargets']} nontargets)")

    # Output paths
    csv_out = args.csv_out or os.path.join(
        args.output_root,
        f"p7_eeg_{args.subject}_{args.session}_{ts}.csv"
    )
    receipt_out = args.receipt_out or os.path.join(
        args.output_root,
        f"p7_eeg_ingest_receipt_{args.subject}_{args.session}_{ts}.json"
    )

    # Write CSV
    write_mb_csv(csv_out, m, b)
    csv_sha = _sha256_file(csv_out)
    print(f"Wrote CSV: {csv_out}")
    print(f"CSV SHA256: {csv_sha}")

    # Build receipt
    receipt = {
        "type": "EEGIngestReceipt",
        "version": 1,
        "phase": "Q32-Phase7",
        "dataset": "OpenNeuro ds005383 (TMNRED)",
        "dataset_version": "1.0.0",
        "run": {
            "subject": args.subject,
            "session": args.session,
            "window_start_ms": args.window_start_ms,
            "window_end_ms": args.window_end_ms,
            "sampling_rate": args.sampling_rate,
            "channels": channels,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "h5py_version": h5py.__version__,
            "numpy_version": np.__version__,
        },
        "input": metadata,
        "output": {
            "csv_path": csv_out,
            "csv_sha256": csv_sha,
            "n_rows": len(m),
            "m_mean": float(np.mean(m)),
            "m_std": float(np.std(m)),
            "b_mean": float(np.mean(b)),
            "b_std": float(np.std(b)),
        },
    }

    write_json(receipt_out, receipt)
    receipt_sha = _sha256_file(receipt_out)
    print(f"Wrote receipt: {receipt_out}")
    print(f"Receipt SHA256: {receipt_sha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
