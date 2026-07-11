#!/usr/bin/env python3
"""Read-only Gate A CPU-frequency precondition observation.

This module performs no network access, subprocess execution, MSR access, or
filesystem writes. It reads the cpufreq sysfs surface for the frozen sender and
receiver cores and returns a closed JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PRECONDITION_OBSERVATION_V1"
OBSERVATION_MODE = "IDLE_READ_ONLY"
DEFAULT_CORES = (4, 5)
REQUIRED_FREQUENCY_KHZ = 1_600_000
DEFAULT_SAMPLE_COUNT = 200
DEFAULT_INTERVAL_MS = 10
MIN_SAMPLE_COUNT = 2
MAX_SAMPLE_COUNT = 10_000
MIN_INTERVAL_MS = 1
MAX_INTERVAL_MS = 1_000

REQUIRED_POLICY_FILES = (
    "scaling_driver",
    "scaling_governor",
    "cpuinfo_min_freq",
    "cpuinfo_max_freq",
    "scaling_min_freq",
    "scaling_max_freq",
)
OPTIONAL_POLICY_FILES = (
    "scaling_available_governors",
    "scaling_available_frequencies",
    "affected_cpus",
    "related_cpus",
    "transition_latency",
    "bios_limit",
)
INTEGER_POLICY_FILES = frozenset(
    {
        "cpuinfo_min_freq",
        "cpuinfo_max_freq",
        "scaling_min_freq",
        "scaling_max_freq",
        "transition_latency",
        "bios_limit",
    }
)
LIST_INTEGER_POLICY_FILES = frozenset(
    {"scaling_available_frequencies", "affected_cpus", "related_cpus"}
)

ReadBytes = Callable[[Path], bytes]
Sleep = Callable[[float], None]
Clock = Callable[[], int]


class ProbeError(RuntimeError):
    """Raised when the read-only observation surface cannot be closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_ascii(data: bytes, *, label: str) -> str:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProbeError(f"{label} is not ASCII") from exc
    stripped = text.strip()
    require(stripped != "", f"{label} is empty")
    return stripped


def _parse_positive_int(text: str, *, label: str, allow_zero: bool = False) -> int:
    require(text.isdecimal(), f"{label} is not an unsigned decimal integer")
    value = int(text, 10)
    if allow_zero:
        require(value >= 0, f"{label} is negative")
    else:
        require(value > 0, f"{label} must be positive")
    return value


def _parse_policy_value(name: str, text: str) -> Any:
    if name in INTEGER_POLICY_FILES:
        return _parse_positive_int(text, label=name, allow_zero=name == "bios_limit")
    if name in LIST_INTEGER_POLICY_FILES:
        values = text.split()
        require(values, f"{name} list is empty")
        return [_parse_positive_int(value, label=name, allow_zero=True) for value in values]
    return text


def _read_policy_file(
    path: Path,
    *,
    name: str,
    required: bool,
    read_bytes: ReadBytes,
) -> dict[str, Any]:
    try:
        data = read_bytes(path)
    except FileNotFoundError:
        if required:
            raise ProbeError(f"required cpufreq file missing: {path}") from None
        return {
            "present": False,
            "path": str(path),
            "raw_text": None,
            "sha256": None,
            "parsed": None,
            "failure": "NOT_PRESENT",
        }
    except OSError as exc:
        raise ProbeError(f"cpufreq file unreadable: {path}: {exc}") from exc

    text = _decode_ascii(data, label=str(path))
    parsed = _parse_policy_value(name, text)
    return {
        "present": True,
        "path": str(path),
        "raw_text": text,
        "sha256": _sha256(data),
        "parsed": parsed,
        "failure": None,
    }


def _path_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        stat_result = resolved.stat()
    except OSError as exc:
        raise ProbeError(f"cpufreq policy path unobservable: {path}: {exc}") from exc
    require(resolved.is_dir(), f"cpufreq policy path is not a directory: {resolved}")
    return {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
    }


def _policy_path(sysfs_root: Path, core: int) -> Path:
    return sysfs_root / "devices" / "system" / "cpu" / f"cpu{core}" / "cpufreq"


def _metadata_for_core(
    sysfs_root: Path,
    core: int,
    *,
    read_bytes: ReadBytes,
) -> dict[str, Any]:
    path = _policy_path(sysfs_root, core)
    identity_before = _path_identity(path)
    files: dict[str, Any] = {}
    for name in REQUIRED_POLICY_FILES:
        files[name] = _read_policy_file(
            path / name,
            name=name,
            required=True,
            read_bytes=read_bytes,
        )
    for name in OPTIONAL_POLICY_FILES:
        files[name] = _read_policy_file(
            path / name,
            name=name,
            required=False,
            read_bytes=read_bytes,
        )
    identity_after = _path_identity(path)
    require(identity_before == identity_after, f"cpufreq policy identity changed for core {core}")

    minimum = files["cpuinfo_min_freq"]["parsed"]
    maximum = files["cpuinfo_max_freq"]["parsed"]
    scaling_minimum = files["scaling_min_freq"]["parsed"]
    scaling_maximum = files["scaling_max_freq"]["parsed"]
    require(minimum <= maximum, f"cpuinfo frequency bounds inverted for core {core}")
    require(
        minimum <= scaling_minimum <= scaling_maximum <= maximum,
        f"scaling frequency bounds escape cpuinfo bounds for core {core}",
    )

    available = files["scaling_available_frequencies"]["parsed"]
    if available is None:
        required_supported: bool | None = minimum <= REQUIRED_FREQUENCY_KHZ <= maximum
    else:
        required_supported = REQUIRED_FREQUENCY_KHZ in available

    return {
        "core": core,
        "identity": identity_before,
        "files": files,
        "required_frequency_supported": required_supported,
    }


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _sample_frequency(
    sysfs_root: Path,
    core: int,
    *,
    read_bytes: ReadBytes,
) -> int:
    path = _policy_path(sysfs_root, core) / "scaling_cur_freq"
    try:
        data = read_bytes(path)
    except OSError as exc:
        raise ProbeError(f"current frequency unreadable for core {core}: {exc}") from exc
    text = _decode_ascii(data, label=str(path))
    return _parse_positive_int(text, label=f"core {core} scaling_cur_freq")


def observe_frequency_precondition(
    *,
    sysfs_root: Path = Path("/sys"),
    cores: tuple[int, ...] = DEFAULT_CORES,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    read_bytes: ReadBytes = lambda path: path.read_bytes(),
    sleep: Sleep = time.sleep,
    monotonic_ns: Clock = time.monotonic_ns,
    utc_ns: Clock = time.time_ns,
) -> dict[str, Any]:
    """Collect one closed read-only frequency-precondition receipt."""

    require(cores == DEFAULT_CORES, "Gate A frequency observation cores are frozen")
    require(MIN_SAMPLE_COUNT <= sample_count <= MAX_SAMPLE_COUNT, "sample count outside closed bounds")
    require(MIN_INTERVAL_MS <= interval_ms <= MAX_INTERVAL_MS, "sample interval outside closed bounds")
    sysfs_root = sysfs_root.resolve()
    require(sysfs_root.is_dir(), f"sysfs root missing: {sysfs_root}")

    metadata = [_metadata_for_core(sysfs_root, core, read_bytes=read_bytes) for core in cores]
    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        pair = {
            str(core): _sample_frequency(sysfs_root, core, read_bytes=read_bytes)
            for core in cores
        }
        samples.append(
            {
                "index": index,
                "monotonic_ns": monotonic_ns(),
                "utc_ns": utc_ns(),
                "frequency_khz": pair,
                "pair_exact": all(value == REQUIRED_FREQUENCY_KHZ for value in pair.values()),
            }
        )
        if index + 1 < sample_count:
            sleep(interval_ms / 1000.0)

    per_core: dict[str, Any] = {}
    for core in cores:
        values = [sample["frequency_khz"][str(core)] for sample in samples]
        exact = [value == REQUIRED_FREQUENCY_KHZ for value in values]
        per_core[str(core)] = {
            "minimum_khz": min(values),
            "maximum_khz": max(values),
            "unique_khz": sorted(set(values)),
            "exact_sample_count": sum(exact),
            "longest_consecutive_exact_samples": _longest_true_run(exact),
            "all_samples_exact": all(exact),
            "ever_exact": any(exact),
        }

    pair_exact = [bool(sample["pair_exact"]) for sample in samples]
    all_pairs_exact = all(pair_exact)
    any_pair_exact = any(pair_exact)
    if all_pairs_exact:
        status = "PASS_STATIC_PRECONDITION_OBSERVED"
        failure = None
    elif any_pair_exact:
        status = "INCONCLUSIVE_DYNAMIC_PRECONDITION"
        failure = "REQUIRED_FREQUENCY_NOT_STABLE_ACROSS_OBSERVATION"
    else:
        status = "FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED"
        failure = "REQUIRED_FREQUENCY_NEVER_OBSERVED_ON_BOTH_CORES"

    receipt = {
        "schema_id": SCHEMA_ID,
        "observation_mode": OBSERVATION_MODE,
        "sysfs_root": str(sysfs_root),
        "required_frequency_khz": REQUIRED_FREQUENCY_KHZ,
        "cores": list(cores),
        "sample_count_requested": sample_count,
        "sample_interval_ms": interval_ms,
        "control_writes": 0,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "network_operations": 0,
        "policy_metadata": metadata,
        "samples": samples,
        "summary": {
            "per_core": per_core,
            "pair_exact_sample_count": sum(pair_exact),
            "longest_consecutive_exact_pairs": _longest_true_run(pair_exact),
            "all_pairs_exact": all_pairs_exact,
            "any_pair_exact": any_pair_exact,
        },
        "status": status,
        "failure": failure,
    }
    require(math.isfinite(float(receipt["required_frequency_khz"])), "required frequency is not finite")
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Gate A cpufreq precondition observation")
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        receipt = observe_frequency_precondition(
            sysfs_root=args.sysfs_root,
            sample_count=args.sample_count,
            interval_ms=args.interval_ms,
        )
    except ProbeError as exc:
        print(
            json.dumps(
                {
                    "schema_id": SCHEMA_ID,
                    "status": "FAILED_CLOSED_UNOBSERVABLE",
                    "failure": str(exc),
                    "control_writes": 0,
                    "frequency_writes": 0,
                    "voltage_writes": 0,
                    "msr_reads": 0,
                    "msr_writes": 0,
                    "network_operations": 0,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
