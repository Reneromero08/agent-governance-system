#!/usr/bin/env python3
"""Authority-gated live /sys transaction for Gate A frequency preparation.

The reviewed synthetic core intentionally refuses /sys.  This wrapper is the only
source allowed to cross that boundary, and it does so only with an opaque permit
returned by the exact preparation-authority validator.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import gate_a_frequency_preparation as core
import gate_a_frequency_preparation_authority as authority

ReadBytes = Callable[[Path], bytes]
WriteBytes = Callable[[Path, bytes], None]
Sleep = Callable[[float], None]
Clock = Callable[[], int]


class LivePreparationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise LivePreparationError(message)


def execute_authorized_preparation_restoration(
    permit: authority.PreparationPermit,
    *,
    sysfs_root: Path = Path("/sys"),
    read_bytes: ReadBytes,
    write_bytes: WriteBytes,
    sleep: Sleep = time.sleep,
    monotonic_ns: Clock = time.monotonic_ns,
) -> dict[str, Any]:
    """Perform exactly one permit-bound pin/observe/restore transaction."""

    checked = authority.require_permit(permit)
    root = sysfs_root.resolve()
    require(str(root) == checked.sysfs_root == authority.EXPECTED_SYSFS_ROOT, "permit-bound sysfs root mismatch")
    require(checked.required_frequency_khz == core.REQUIRED_FREQUENCY_KHZ, "permit/core frequency mismatch")
    require(checked.sample_count == core.DEFAULT_SAMPLE_COUNT, "permit/core sample count mismatch")
    require(checked.sample_interval_ms == core.DEFAULT_INTERVAL_MS, "permit/core interval mismatch")
    require(checked.maximum_write_attempt_count == core.MAX_WRITE_ATTEMPT_COUNT, "permit/core write cap mismatch")

    receipt: dict[str, Any] = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_AUTHORIZED_FREQUENCY_PREPARATION_V1",
        "authority_id": checked.authority_id,
        "authority_sha256": checked.authority_sha256,
        "bundle_sha256": checked.bundle_sha256,
        "required_frequency_khz": checked.required_frequency_khz,
        "cores": list(core.CORES),
        "sysfs_root": str(root),
        "live_sysfs": True,
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
            core_id: core._snapshot_policy(root, core_id, read_bytes=read_bytes)
            for core_id in core.CORES
        }
        receipt["snapshot"] = {str(core_id): snapshots[core_id] for core_id in core.CORES}
        require(
            snapshots[4]["identity"]["resolved_path"] != snapshots[5]["identity"]["resolved_path"],
            "cores 4 and 5 unexpectedly share one cpufreq policy",
        )
        writes_started = True
        for core_id in core.CORES:
            core._write_and_verify(
                phase="prepare",
                core=core_id,
                field="scaling_max_freq",
                value=checked.required_frequency_khz,
                sysfs_root=root,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                ledger=ledger,
                monotonic_ns=monotonic_ns,
                expected_identity=snapshots[core_id]["identity"],
            )
            core._write_and_verify(
                phase="prepare",
                core=core_id,
                field="scaling_min_freq",
                value=checked.required_frequency_khz,
                sysfs_root=root,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                ledger=ledger,
                monotonic_ns=monotonic_ns,
                expected_identity=snapshots[core_id]["identity"],
            )
        receipt["pinned_observation"] = core._observe_pinned(
            sysfs_root=root,
            read_bytes=read_bytes,
            sample_count=checked.sample_count,
            interval_ms=checked.sample_interval_ms,
            sleep=sleep,
            monotonic_ns=monotonic_ns,
            expected_identities={core_id: snapshots[core_id]["identity"] for core_id in core.CORES},
        )
    except (OSError, core.PreparationError, LivePreparationError) as exc:
        receipt["preparation_failure"] = str(exc)
    finally:
        if writes_started and snapshots:
            restoration_errors: list[str] = []
            for core_id in core.CORES:
                snapshot = snapshots[core_id]
                for field in ("scaling_min_freq", "scaling_max_freq"):
                    try:
                        core._write_and_verify(
                            phase="restore",
                            core=core_id,
                            field=field,
                            value=snapshot[field]["value"],
                            sysfs_root=root,
                            read_bytes=read_bytes,
                            write_bytes=write_bytes,
                            ledger=ledger,
                            monotonic_ns=monotonic_ns,
                            expected_identity=snapshot["identity"],
                        )
                    except (core.PreparationError, OSError) as exc:
                        restoration_errors.append(str(exc))
            for core_id in core.CORES:
                try:
                    current = core._snapshot_policy(root, core_id, read_bytes=read_bytes)
                    require(current["identity"] == snapshots[core_id]["identity"], f"policy identity changed after restore: core {core_id}")
                except (core.PreparationError, LivePreparationError, OSError) as exc:
                    restoration_errors.append(str(exc))
            if restoration_errors:
                receipt["restoration_failure"] = restoration_errors
            else:
                receipt["restoration_complete"] = True

    receipt["frequency_write_attempt_count"] = len(ledger)
    receipt["write_call_returned_count"] = sum(bool(entry["write_call_returned"]) for entry in ledger)
    require(len(ledger) <= checked.maximum_write_attempt_count, "write attempt count exceeded authority cap")
    receipt["written_paths"] = sorted({entry["path"] for entry in ledger})
    expected_paths = sorted(
        str(core._policy_path(root, core_id) / field)
        for core_id in core.CORES
        for field in core.WRITABLE_SUFFIXES
    )
    require(set(receipt["written_paths"]).issubset(set(expected_paths)), "write path escaped exact allowlist")

    if receipt["restoration_failure"] is not None:
        receipt["status"] = "FAILED_CLOSED_RESTORATION"
    elif receipt["preparation_failure"] is not None and not writes_started:
        receipt["status"] = "FAILED_CLOSED_PRECONDITION_NO_WRITES"
    elif receipt["preparation_failure"] is not None:
        require(receipt["restoration_complete"] is True, "write-bearing failure lacks restoration")
        receipt["status"] = "FAILED_CLOSED_PREPARATION__RESTORED"
    else:
        require(receipt["pinned_observation"] is not None, "successful preparation lacks observation")
        require(receipt["restoration_complete"] is True, "successful preparation lacks restoration")
        receipt["status"] = "QUALIFIED_PREPARATION_AND_RESTORATION"
    return receipt
