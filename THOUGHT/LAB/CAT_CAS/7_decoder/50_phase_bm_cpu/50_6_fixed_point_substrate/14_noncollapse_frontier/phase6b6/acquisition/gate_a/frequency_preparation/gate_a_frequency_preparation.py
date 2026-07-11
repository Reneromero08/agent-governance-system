#!/usr/bin/env python3
"""Fail-closed Gate A CPU-frequency preparation and restoration core.

This source is qualified only against a synthetic null baseline. It exposes no
network, subprocess, voltage, MSR, sender, capture, or Gate A smoke surface.
The CLI refuses the live /sys tree. A future reviewed authority wrapper must
supply any live write capability separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_RESTORATION_V1"
CORES = (4, 5)
REQUIRED_DRIVER = "acpi-cpufreq"
REQUIRED_GOVERNOR = "schedutil"
REQUIRED_FREQUENCY_KHZ = 1_600_000
EXPECTED_BASELINE_MIN_KHZ = 800_000
EXPECTED_BASELINE_MAX_KHZ = 3_200_000
DEFAULT_SAMPLE_COUNT = 200
DEFAULT_INTERVAL_MS = 10
MAX_WRITE_ATTEMPT_COUNT = 8
POLICY_RELATIVE_PATHS = {
    4: "devices/system/cpu/cpufreq/policy4",
    5: "devices/system/cpu/cpufreq/policy5",
}
WRITABLE_SUFFIXES = ("scaling_max_freq", "scaling_min_freq")

ReadBytes = Callable[[Path], bytes]
WriteBytes = Callable[[Path, bytes], None]
Sleep = Callable[[float], None]
Clock = Callable[[], int]


class PreparationError(RuntimeError):
    """Raised when preparation or restoration cannot be closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_ascii(data: bytes, *, label: str) -> str:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PreparationError(f"{label} is not ASCII") from exc
    stripped = text.strip()
    require(stripped != "", f"{label} is empty")
    return stripped


def _parse_uint(data: bytes, *, label: str) -> int:
    text = _decode_ascii(data, label=label)
    require(text.isdecimal(), f"{label} is not an unsigned decimal integer")
    value = int(text, 10)
    require(value > 0, f"{label} must be positive")
    return value


def _canonical_value(value: int) -> bytes:
    require(value > 0, "frequency value must be positive")
    return f"{value}\n".encode("ascii")


def _policy_path(sysfs_root: Path, core: int) -> Path:
    require(core in POLICY_RELATIVE_PATHS, f"unapproved core: {core}")
    return sysfs_root / POLICY_RELATIVE_PATHS[core]


def _identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        stat_result = resolved.stat()
    except OSError as exc:
        raise PreparationError(f"policy path unobservable: {path}: {exc}") from exc
    require(resolved.is_dir(), f"policy path is not a directory: {resolved}")
    return {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
    }


def _read_required(path: Path, *, read_bytes: ReadBytes) -> bytes:
    try:
        value = read_bytes(path)
    except OSError as exc:
        raise PreparationError(f"required sysfs file unreadable: {path}: {exc}") from exc
    require(value != b"", f"required sysfs file empty: {path}")
    return value


def _read_text(path: Path, *, read_bytes: ReadBytes) -> tuple[str, str]:
    raw = _read_required(path, read_bytes=read_bytes)
    return _decode_ascii(raw, label=str(path)), _sha256(raw)


def _read_uint(path: Path, *, read_bytes: ReadBytes) -> tuple[int, bytes]:
    raw = _read_required(path, read_bytes=read_bytes)
    return _parse_uint(raw, label=str(path)), raw


def _snapshot_policy(
    sysfs_root: Path,
    core: int,
    *,
    read_bytes: ReadBytes,
) -> dict[str, Any]:
    policy = _policy_path(sysfs_root, core)
    identity_before = _identity(policy)
    driver, driver_sha = _read_text(policy / "scaling_driver", read_bytes=read_bytes)
    governor, governor_sha = _read_text(
        policy / "scaling_governor", read_bytes=read_bytes
    )
    cpuinfo_min, cpuinfo_min_raw = _read_uint(
        policy / "cpuinfo_min_freq", read_bytes=read_bytes
    )
    cpuinfo_max, cpuinfo_max_raw = _read_uint(
        policy / "cpuinfo_max_freq", read_bytes=read_bytes
    )
    scaling_min, scaling_min_raw = _read_uint(
        policy / "scaling_min_freq", read_bytes=read_bytes
    )
    scaling_max, scaling_max_raw = _read_uint(
        policy / "scaling_max_freq", read_bytes=read_bytes
    )
    available_raw = _read_required(
        policy / "scaling_available_frequencies", read_bytes=read_bytes
    )
    available_text = _decode_ascii(
        available_raw, label=str(policy / "scaling_available_frequencies")
    )
    available = [
        _parse_uint(token.encode("ascii"), label="available frequency")
        for token in available_text.split()
    ]
    affected_raw = _read_required(policy / "affected_cpus", read_bytes=read_bytes)
    related_raw = _read_required(policy / "related_cpus", read_bytes=read_bytes)
    affected = [
        _parse_uint(token.encode("ascii"), label="affected cpu")
        for token in _decode_ascii(affected_raw, label="affected_cpus").split()
    ]
    related = [
        _parse_uint(token.encode("ascii"), label="related cpu")
        for token in _decode_ascii(related_raw, label="related_cpus").split()
    ]
    identity_after = _identity(policy)

    require(
        identity_before == identity_after,
        f"policy identity changed during snapshot: core {core}",
    )
    require(driver == REQUIRED_DRIVER, f"unexpected driver on core {core}: {driver}")
    require(
        governor == REQUIRED_GOVERNOR,
        f"unexpected governor on core {core}: {governor}",
    )
    require(
        affected == [core] and related == [core],
        f"policy ownership mismatch on core {core}",
    )
    require(
        cpuinfo_min == EXPECTED_BASELINE_MIN_KHZ,
        f"cpuinfo minimum drift on core {core}",
    )
    require(
        cpuinfo_max == EXPECTED_BASELINE_MAX_KHZ,
        f"cpuinfo maximum drift on core {core}",
    )
    require(
        scaling_min == EXPECTED_BASELINE_MIN_KHZ,
        f"scaling minimum drift on core {core}",
    )
    require(
        scaling_max == EXPECTED_BASELINE_MAX_KHZ,
        f"scaling maximum drift on core {core}",
    )
    require(
        REQUIRED_FREQUENCY_KHZ in available,
        f"required frequency unsupported on core {core}",
    )

    return {
        "core": core,
        "policy_path": str(policy),
        "identity": identity_before,
        "driver": {"text": driver, "sha256": driver_sha},
        "governor": {"text": governor, "sha256": governor_sha},
        "cpuinfo_min_freq": {
            "value": cpuinfo_min,
            "raw_sha256": _sha256(cpuinfo_min_raw),
        },
        "cpuinfo_max_freq": {
            "value": cpuinfo_max,
            "raw_sha256": _sha256(cpuinfo_max_raw),
        },
        "scaling_min_freq": {
            "value": scaling_min,
            "raw_text": _decode_ascii(scaling_min_raw, label="scaling_min_freq"),
            "raw_sha256": _sha256(scaling_min_raw),
        },
        "scaling_max_freq": {
            "value": scaling_max,
            "raw_text": _decode_ascii(scaling_max_raw, label="scaling_max_freq"),
            "raw_sha256": _sha256(scaling_max_raw),
        },
        "available_frequencies_khz": available,
        "available_frequencies_sha256": _sha256(available_raw),
        "affected_cpus": affected,
        "related_cpus": related,
    }


def _write_and_verify(
    *,
    phase: str,
    core: int,
    field: str,
    value: int,
    sysfs_root: Path,
    read_bytes: ReadBytes,
    write_bytes: WriteBytes,
    ledger: list[dict[str, Any]],
    monotonic_ns: Clock,
    expected_identity: dict[str, Any],
) -> None:
    require(field in WRITABLE_SUFFIXES, f"unapproved writable field: {field}")
    policy = _policy_path(sysfs_root, core)
    require(
        _identity(policy) == expected_identity,
        f"policy identity changed before {phase} write: core {core}",
    )
    path = policy / field
    payload = _canonical_value(value)
    entry = {
        "sequence": len(ledger) + 1,
        "phase": phase,
        "core": core,
        "field": field,
        "path": str(path),
        "requested_value_khz": value,
        "payload_sha256": _sha256(payload),
        "start_monotonic_ns": monotonic_ns(),
        "write_call_returned": False,
        "readback_khz": None,
        "failure": None,
    }
    ledger.append(entry)
    try:
        write_bytes(path, payload)
        entry["write_call_returned"] = True
        readback, _raw = _read_uint(path, read_bytes=read_bytes)
        entry["readback_khz"] = readback
        require(readback == value, f"readback mismatch after {phase} write: {path}")
        require(
            _identity(policy) == expected_identity,
            f"policy identity changed after {phase} write: core {core}",
        )
    except (OSError, PreparationError) as exc:
        entry["failure"] = str(exc)
        raise PreparationError(f"{phase} write failed for {path}: {exc}") from exc
    finally:
        entry["end_monotonic_ns"] = monotonic_ns()


def _observe_pinned(
    *,
    sysfs_root: Path,
    read_bytes: ReadBytes,
    sample_count: int,
    interval_ms: int,
    sleep: Sleep,
    monotonic_ns: Clock,
    expected_identities: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    require(2 <= sample_count <= 10_000, "sample count outside closed bounds")
    require(1 <= interval_ms <= 1_000, "sample interval outside closed bounds")
    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        for core in CORES:
            require(
                _identity(_policy_path(sysfs_root, core))
                == expected_identities[core],
                f"policy identity changed before pinned sample {index}: core {core}",
            )
        values = {
            str(core): _read_uint(
                _policy_path(sysfs_root, core) / "scaling_cur_freq",
                read_bytes=read_bytes,
            )[0]
            for core in CORES
        }
        for core in CORES:
            require(
                _identity(_policy_path(sysfs_root, core))
                == expected_identities[core],
                f"policy identity changed after pinned sample {index}: core {core}",
            )
        exact = all(value == REQUIRED_FREQUENCY_KHZ for value in values.values())
        samples.append(
            {
                "index": index,
                "monotonic_ns": monotonic_ns(),
                "frequency_khz": values,
                "pair_exact": exact,
            }
        )
        if index + 1 < sample_count:
            sleep(interval_ms / 1000.0)
    require(
        all(sample["pair_exact"] for sample in samples),
        "pinned frequency was not static across observation",
    )
    return {
        "sample_count": sample_count,
        "sample_interval_ms": interval_ms,
        "samples": samples,
        "all_pairs_exact": True,
    }


def qualify_preparation_restoration(
    *,
    sysfs_root: Path,
    read_bytes: ReadBytes,
    write_bytes: WriteBytes,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    sleep: Sleep = time.sleep,
    monotonic_ns: Clock = time.monotonic_ns,
) -> dict[str, Any]:
    """Qualify one pin-observe-restore transaction on an injected sysfs root."""

    root = sysfs_root.resolve()
    require(root.is_dir(), f"sysfs root missing: {root}")
    live_root = Path("/sys").resolve()
    require(
        not (root == live_root or live_root in root.parents),
        "live /sys preparation is not authorized",
    )

    receipt: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "required_frequency_khz": REQUIRED_FREQUENCY_KHZ,
        "cores": list(CORES),
        "sysfs_root": str(root),
        "live_sysfs": root == live_root or live_root in root.parents,
        "automatic_retry": False,
        "retry_count": 0,
        "smoke_executions": 0,
        "sender_starts": 0,
        "capture_starts": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "governor_writes": 0,
        "write_ledger": [],
        "snapshot": None,
        "pinned_observation": None,
        "preparation_failure": None,
        "restoration_failure": None,
        "restoration_complete": False,
        "status": None,
    }
    ledger: list[dict[str, Any]] = receipt["write_ledger"]
    snapshots: dict[int, dict[str, Any]] = {}
    writes_started = False

    try:
        snapshots = {
            core: _snapshot_policy(root, core, read_bytes=read_bytes)
            for core in CORES
        }
        receipt["snapshot"] = {str(core): snapshots[core] for core in CORES}
        require(
            snapshots[4]["identity"]["resolved_path"]
            != snapshots[5]["identity"]["resolved_path"],
            "cores 4 and 5 unexpectedly share one cpufreq policy",
        )
        writes_started = True
        for core in CORES:
            _write_and_verify(
                phase="prepare",
                core=core,
                field="scaling_max_freq",
                value=REQUIRED_FREQUENCY_KHZ,
                sysfs_root=root,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                ledger=ledger,
                monotonic_ns=monotonic_ns,
                expected_identity=snapshots[core]["identity"],
            )
            _write_and_verify(
                phase="prepare",
                core=core,
                field="scaling_min_freq",
                value=REQUIRED_FREQUENCY_KHZ,
                sysfs_root=root,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                ledger=ledger,
                monotonic_ns=monotonic_ns,
                expected_identity=snapshots[core]["identity"],
            )
        receipt["pinned_observation"] = _observe_pinned(
            sysfs_root=root,
            read_bytes=read_bytes,
            sample_count=sample_count,
            interval_ms=interval_ms,
            sleep=sleep,
            monotonic_ns=monotonic_ns,
            expected_identities={
                core: snapshots[core]["identity"] for core in CORES
            },
        )
    except (OSError, PreparationError) as exc:
        receipt["preparation_failure"] = str(exc)
    finally:
        if writes_started and snapshots:
            restoration_errors: list[str] = []
            for core in CORES:
                snapshot = snapshots[core]
                for field in ("scaling_min_freq", "scaling_max_freq"):
                    try:
                        _write_and_verify(
                            phase="restore",
                            core=core,
                            field=field,
                            value=snapshot[field]["value"],
                            sysfs_root=root,
                            read_bytes=read_bytes,
                            write_bytes=write_bytes,
                            ledger=ledger,
                            monotonic_ns=monotonic_ns,
                            expected_identity=snapshots[core]["identity"],
                        )
                    except PreparationError as exc:
                        restoration_errors.append(str(exc))
            for core in CORES:
                try:
                    current = _snapshot_policy(root, core, read_bytes=read_bytes)
                    require(
                        current["identity"] == snapshots[core]["identity"],
                        f"policy identity changed after restore: core {core}",
                    )
                except PreparationError as exc:
                    restoration_errors.append(str(exc))
            if restoration_errors:
                receipt["restoration_failure"] = restoration_errors
            else:
                receipt["restoration_complete"] = True

    receipt["frequency_write_attempt_count"] = len(ledger)
    receipt["write_call_returned_count"] = sum(
        bool(entry["write_call_returned"]) for entry in ledger
    )
    require(
        len(ledger) <= MAX_WRITE_ATTEMPT_COUNT,
        "write attempt count exceeded closed bound",
    )
    receipt["written_paths"] = sorted({entry["path"] for entry in ledger})
    expected_paths = sorted(
        str(_policy_path(root, core) / field)
        for core in CORES
        for field in WRITABLE_SUFFIXES
    )
    require(
        set(receipt["written_paths"]).issubset(set(expected_paths)),
        "write path set escaped closed boundary",
    )

    if receipt["restoration_failure"] is not None:
        receipt["status"] = "FAILED_CLOSED_RESTORATION"
    elif receipt["preparation_failure"] is not None and not writes_started:
        receipt["status"] = "FAILED_CLOSED_PRECONDITION_NO_WRITES"
    elif receipt["preparation_failure"] is not None:
        require(
            receipt["restoration_complete"] is True,
            "write-bearing failure lacks restoration",
        )
        receipt["status"] = "FAILED_CLOSED_PREPARATION__RESTORED"
    else:
        require(
            receipt["pinned_observation"] is not None,
            "successful preparation lacks observation",
        )
        require(
            receipt["restoration_complete"] is True,
            "successful preparation lacks restoration",
        )
        receipt["status"] = "QUALIFIED_PREPARATION_AND_RESTORATION"
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic qualification for Gate A frequency preparation/restoration"
    )
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.synthetic_root is None:
        print(
            "LIVE_FREQUENCY_PREPARATION_NOT_AUTHORIZED",
            file=__import__("sys").stderr,
        )
        return 2
    root = args.synthetic_root.resolve()
    if root == Path("/sys").resolve() or Path("/sys").resolve() in root.parents:
        print(
            "LIVE_FREQUENCY_PREPARATION_NOT_AUTHORIZED",
            file=__import__("sys").stderr,
        )
        return 2
    try:
        receipt = qualify_preparation_restoration(
            sysfs_root=root,
            read_bytes=lambda path: path.read_bytes(),
            write_bytes=lambda path, data: path.write_bytes(data),
            sample_count=args.sample_count,
            interval_ms=args.interval_ms,
        )
    except (OSError, PreparationError) as exc:
        print(
            json.dumps(
                {
                    "schema_id": SCHEMA_ID,
                    "status": "FAILED_CLOSED_UNOBSERVABLE",
                    "failure": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "QUALIFIED_PREPARATION_AND_RESTORATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
