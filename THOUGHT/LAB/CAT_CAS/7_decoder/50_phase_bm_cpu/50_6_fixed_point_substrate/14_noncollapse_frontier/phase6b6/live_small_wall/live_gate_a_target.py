#!/usr/bin/env python3
"""Direct user-authorized Gate A first-light transaction on the CAT_CAS lab device.

This joins the existing frequency preparation core to the existing Gate A C
runtime.  It creates no authority artifact and contains no voltage or MSR API.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import signal
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gate_a_frequency_preparation as frequency


CORES = (4, 5)
REQUIRED_FREQUENCY_KHZ = 1_600_000
TEMPERATURE_VETO_C = 68.0
MONITOR_INTERVAL_S = 0.1
WORKER_TIMEOUT_S = 20.0
TRANSACTION_TIMEOUT_S = 45
SCHEDULE_SHA256 = "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116"
READONLY_MICRO_SCHEDULE_SHA256 = "57f6aa152d2c099429e7ca2c4d843102739c81b2158e46c4d49f07a96b6f4758"
CODED_PREPROJECTION_SCHEDULE_SHA256 = "35496568999774114af1057ac70fda4b6aeb8a8989e8daf1d1672e508523d07c"
CODED_PREPROJECTION_RESTORED_SCHEDULE_SHA256 = "90538e09de19f90699adabdb2e283a73039f8e5e1e4e71b2501d56e966dbb7cf"
CODED_PREPROJECTION_WARM_RESTORED_SCHEDULE_SHA256 = "94cbace65638dd457983475db0944e37b9e9bf9fec96ae1a8dbb4515db663c3b"
CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_SCHEDULE_SHA256 = "88a93ac2a565f612a3a3789b515a187dbb1e4196519962d56a2be09df2eb0ca7"
CODED_PREPROJECTION_WARM_QUERY_OFF_SCHEDULE_SHA256 = "95d25a543007bdfcdb002ff0ce36642e9f64ef2280d262261e5ea17557482137"
CODED_PREPROJECTION_WARM_DECLARATION_SHAM_SCHEDULE_SHA256 = "89e53ef27c3799cc9c319283821e728e304a8b36a92ac1a76088f28934992310"
CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_SCHEDULE_SHA256 = "51f3fb66cd4f03dff2d3e9aab9196d4f94d85e221cd552b04eda4929669cca2e"
CODED_PREPROJECTION_WARM_PHASE_LOCAL_SCHEDULE_SHA256 = "1144b929905e30f3da1261fdedf5e6393c30d31a41c9a4dd2dc39e8573f4cbc4"
CODED_PREPROJECTION_ACTIVE_QUERY_SCHEDULE_SHA256 = "5a0ac285435ba33a80a3272020f19c85004fc63949f1df4325a3ad90fdcd87f2"
CODED_PREPROJECTION_SOURCE_PHASE_CHOP_SCHEDULE_SHA256 = "0308e6518c6e8e4fd60862f3825750a3865d3f8cb4cbef59d140eefb6d2e0fb1"
READONLY_MICRO_READ_HZ = 2_000
CODED_PREPROJECTION_READ_HZ = 2_000
LEGACY_READ_HZ = 8_000
USER_DIRECTIVE_SHA256 = hashlib.sha256(
    b"CAT_CAS_EXPLICIT_USER_LIVE_AUTHORIZATION__SMALL_WALL_GOAL__2026-07-11"
).hexdigest()
OFF_TOKENS = frozenset({"I", "C0", "D0", "O0", "T"})
SAMPLE_TIMING_RECORD_V1 = struct.Struct("<QQQQQQQQ")
SAMPLE_TIMING_RECORD_V1_BYTES = 64
SAMPLE_TIMING_RECORD_V1_SCHEMA_ID = "CAT_CAS_READONLY_OCCUPANCY_SAMPLE_TIMING_V1"
SAMPLE_TIMING_RECORD_V2 = struct.Struct("<QQQQQQQQQQ")
SAMPLE_TIMING_RECORD_BYTES = 80
SAMPLE_TIMING_SCHEMA_ID = "CAT_CAS_READONLY_OCCUPANCY_SAMPLE_TIMING_V2"
READONLY_VARIANTS = frozenset({
    "readonly-occupancy-forward",
    "readonly-occupancy-reverse",
    "readonly-occupancy-equal",
})
CODED_PREPROJECTION_VARIANTS = frozenset({
    "coded-preprojection-loop",
    "coded-preprojection-restored-loop",
    "coded-preprojection-warm-restored-loop",
    "coded-preprojection-warm-query-scramble-loop",
    "coded-preprojection-warm-query-off-loop",
    "coded-preprojection-warm-declaration-sham-loop",
    "coded-preprojection-warm-phase-local-sham-loop",
    "coded-preprojection-warm-phase-local-loop",
    "coded-preprojection-active-query-loop",
    "coded-preprojection-source-phase-chop-loop",
})
CODED_PREPROJECTION_RESTORED_VARIANTS = frozenset({
    "coded-preprojection-restored-loop",
})
CODED_PREPROJECTION_WARM_RESTORED_VARIANTS = frozenset({
    "coded-preprojection-warm-restored-loop",
    "coded-preprojection-warm-query-scramble-loop",
    "coded-preprojection-warm-query-off-loop",
    "coded-preprojection-warm-declaration-sham-loop",
    "coded-preprojection-warm-phase-local-sham-loop",
    "coded-preprojection-warm-phase-local-loop",
    "coded-preprojection-active-query-loop",
    "coded-preprojection-source-phase-chop-loop",
})
CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS = frozenset({
    "coded-preprojection-warm-query-scramble-loop",
})
CODED_PREPROJECTION_QUERY_OFF_VARIANTS = frozenset({
    "coded-preprojection-warm-query-off-loop",
})
CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS = frozenset({
    "coded-preprojection-warm-declaration-sham-loop",
})
CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS = frozenset({
    "coded-preprojection-warm-phase-local-sham-loop",
})
CODED_PREPROJECTION_PHASE_LOCAL_VARIANTS = frozenset({
    "coded-preprojection-warm-phase-local-sham-loop",
    "coded-preprojection-warm-phase-local-loop",
    "coded-preprojection-active-query-loop",
    "coded-preprojection-source-phase-chop-loop",
})
CODED_PREPROJECTION_ACTIVE_QUERY_VARIANTS = frozenset({
    "coded-preprojection-active-query-loop",
})
CODED_PREPROJECTION_SOURCE_PHASE_CHOP_VARIANTS = frozenset({
    "coded-preprojection-source-phase-chop-loop",
})
CODED_PREPROJECTION_NULL_CONTROL_VARIANTS = (
    CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS |
    CODED_PREPROJECTION_QUERY_OFF_VARIANTS |
    CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS |
    CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS
)
READONLY_TIMING_VARIANTS = READONLY_VARIANTS | CODED_PREPROJECTION_VARIANTS
SOURCE_NAMES = (
    "live_gate_a_target.py",
    "gate_a_frequency_preparation.py",
    "small_wall_worker.c",
    "small_wall_runtime.c",
    "small_wall_runtime.h",
    "combined_pdn_hardware.c",
    "combined_pdn_hardware.h",
    "capture_quality_contract.h",
    "captured_file.c",
    "captured_file.h",
)
FORBIDDEN_PROCESS_MARKERS = (
    "combined_pdn_runner",
    "run_combined_campaign",
    "explicit_slot_runtime",
    "gate_a_worker_live",
)


def coded_preprojection_schedule_sha256(pilot_variant: str) -> str:
    if pilot_variant in CODED_PREPROJECTION_SOURCE_PHASE_CHOP_VARIANTS:
        return CODED_PREPROJECTION_SOURCE_PHASE_CHOP_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_ACTIVE_QUERY_VARIANTS:
        return CODED_PREPROJECTION_ACTIVE_QUERY_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS:
        return CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_QUERY_OFF_VARIANTS:
        return CODED_PREPROJECTION_WARM_QUERY_OFF_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS:
        return CODED_PREPROJECTION_WARM_DECLARATION_SHAM_SCHEDULE_SHA256
    if pilot_variant == "coded-preprojection-warm-phase-local-loop":
        return CODED_PREPROJECTION_WARM_PHASE_LOCAL_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS:
        return CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_WARM_RESTORED_VARIANTS:
        return CODED_PREPROJECTION_WARM_RESTORED_SCHEDULE_SHA256
    if pilot_variant in CODED_PREPROJECTION_RESTORED_VARIANTS:
        return CODED_PREPROJECTION_RESTORED_SCHEDULE_SHA256
    return CODED_PREPROJECTION_SCHEDULE_SHA256


class LiveGateAError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise LiveGateAError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sysfs(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        data = os.read(fd, 4096)
        require(data != b"", f"empty sysfs value: {path}")
        return data
    finally:
        os.close(fd)


def write_sysfs(path: Path, data: bytes) -> None:
    require(data.endswith(b"\n") and data[:-1].isdigit(), "noncanonical frequency payload")
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short sysfs write: {path}")
            view = view[written:]
    finally:
        os.close(fd)


def temperature_path() -> Path:
    candidates: list[Path] = []
    for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (root / "name").read_text(encoding="ascii").strip() == "k10temp":
                candidate = root / "temp1_input"
                if candidate.is_file():
                    candidates.append(candidate)
        except OSError:
            continue
    require(len(candidates) == 1, f"expected one observable k10temp input, found {len(candidates)}")
    return candidates[0]


def read_temperature_c(path: Path) -> float:
    try:
        raw = int(read_sysfs(path).decode("ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LiveGateAError(f"temperature unobservable: {path}: {exc}") from exc
    value = raw / 1000.0
    require(math.isfinite(value), "temperature is not finite")
    return value


def current_frequencies() -> dict[str, int]:
    values: dict[str, int] = {}
    for core in CORES:
        path = Path(f"/sys/devices/system/cpu/cpufreq/policy{core}/scaling_cur_freq")
        try:
            values[str(core)] = int(read_sysfs(path).decode("ascii").strip())
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise LiveGateAError(f"frequency unobservable for policy {core}: {exc}") from exc
    return values


def current_limits() -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}
    for core in CORES:
        policy = Path(f"/sys/devices/system/cpu/cpufreq/policy{core}")
        try:
            values[str(core)] = {
                "min": int(read_sysfs(policy / "scaling_min_freq").decode("ascii").strip()),
                "max": int(read_sysfs(policy / "scaling_max_freq").decode("ascii").strip()),
            }
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise LiveGateAError(f"frequency limits unobservable for policy {core}: {exc}") from exc
    return values


def start_warmup(core: int) -> subprocess.Popen[str]:
    code = (
        "import os,time; "
        f"os.sched_setaffinity(0, {{{core}}}); "
        "end=time.monotonic()+8.0; x=1; "
        "exec('while time.monotonic() < end:\\n x=((x*1664525)+1013904223)&0xffffffff')"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def observe_pinned_exact(snapshots: dict[int, dict[str, Any]]) -> dict[str, Any]:
    settle_samples: list[dict[str, Any]] = []
    observation_samples: list[dict[str, Any]] = []
    warmups = [start_warmup(core) for core in CORES]
    stable_count = 0
    settled = False
    try:
        for index in range(200):
            for core in CORES:
                require(
                    frequency._identity(frequency._policy_path(Path("/sys"), core))
                    == snapshots[core]["identity"],
                    f"policy identity changed during settle observation: {core}",
                )
            values = current_frequencies()
            limits = current_limits()
            exact = all(value == REQUIRED_FREQUENCY_KHZ for value in values.values())
            limits_exact = all(
                pair["min"] == REQUIRED_FREQUENCY_KHZ and pair["max"] == REQUIRED_FREQUENCY_KHZ
                for pair in limits.values()
            )
            settle_samples.append(
                {
                    "index": index,
                    "monotonic_ns": time.monotonic_ns(),
                    "frequency_khz": values,
                    "limits_khz": limits,
                    "pair_exact": exact,
                    "limits_exact": limits_exact,
                }
            )
            require(limits_exact, "frequency limit drift during loaded settle observation")
            stable_count = stable_count + 1 if exact else 0
            if stable_count >= 20:
                settled = True
                break
            time.sleep(0.01)

        if settled:
            for index in range(200):
                for core in CORES:
                    require(
                        frequency._identity(frequency._policy_path(Path("/sys"), core))
                        == snapshots[core]["identity"],
                        f"policy identity changed during exact observation: {core}",
                    )
                values = current_frequencies()
                limits = current_limits()
                observation_samples.append(
                    {
                        "index": index,
                        "monotonic_ns": time.monotonic_ns(),
                        "frequency_khz": values,
                        "limits_khz": limits,
                        "pair_exact": all(value == REQUIRED_FREQUENCY_KHZ for value in values.values()),
                        "limits_exact": all(
                            pair["min"] == REQUIRED_FREQUENCY_KHZ and pair["max"] == REQUIRED_FREQUENCY_KHZ
                            for pair in limits.values()
                        ),
                    }
                )
                if index + 1 < 200:
                    time.sleep(0.01)
    finally:
        for process in warmups:
            stop_process(process)
        for process in warmups:
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
                require(process.returncode in (0, -signal.SIGTERM) and not stderr, f"warmup failed: {stderr}")
    return {
        "settle_sample_count": len(settle_samples),
        "settle_required_consecutive_exact": 20,
        "settled": settled,
        "settle_samples": settle_samples,
        "observation_sample_count": len(observation_samples),
        "sample_interval_ms": 10,
        "samples": observation_samples,
        "all_pairs_exact": settled
        and len(observation_samples) == 200
        and all(sample["pair_exact"] and sample["limits_exact"] for sample in observation_samples),
    }


def process_snapshot(*, allow_worker_pid: int | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
    )
    matches: list[str] = []
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        first = line.split(None, 1)[0]
        pid = int(first) if first.isdigit() else -1
        if allow_worker_pid is not None and pid == allow_worker_pid:
            continue
        if any(marker in line for marker in FORBIDDEN_PROCESS_MARKERS):
            matches.append(line)
    return {"observed_process_count": len(completed.stdout.splitlines()), "forbidden_matches": matches}


def source_digest(source_root: Path) -> tuple[str, dict[str, str]]:
    hashes = {name: sha256_file(source_root / name) for name in SOURCE_NAMES}
    return hashlib.sha256(canonical_bytes(hashes)).hexdigest(), hashes


def compile_worker(source_root: Path, binary: Path, runtime_output: Path) -> tuple[str, list[str]]:
    bundle_sha256, _hashes = source_digest(source_root)
    command = [
        "cc",
        f'-DGATE_A_COMPILED_AUTHORITY_SHA256="{USER_DIRECTIVE_SHA256}"',
        f'-DGATE_A_COMPILED_OUTPUT_ROOT="{runtime_output}"',
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-O2",
        "-march=amdfam10",
        "-mtune=amdfam10",
        "-fno-lto",
        "-pthread",
        "-I",
        str(source_root),
        str(source_root / "small_wall_worker.c"),
        str(source_root / "small_wall_runtime.c"),
        str(source_root / "captured_file.c"),
        "-lm",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, timeout=30)
    return bundle_sha256, command


def write_frequency_and_settle(
    *,
    phase: str,
    core: int,
    field: str,
    value: int,
    expected_identity: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> None:
    require(field in frequency.WRITABLE_SUFFIXES, f"unapproved writable field: {field}")
    policy = frequency._policy_path(Path("/sys"), core)
    require(frequency._identity(policy) == expected_identity, f"policy identity changed before {phase} write: {core}")
    path = policy / field
    payload = frequency._canonical_value(value)
    entry: dict[str, Any] = {
        "sequence": len(ledger) + 1,
        "phase": phase,
        "core": core,
        "field": field,
        "path": str(path),
        "requested_value_khz": value,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "start_monotonic_ns": time.monotonic_ns(),
        "write_call_returned": False,
        "readback_khz": None,
        "readback_samples": [],
        "failure": None,
    }
    ledger.append(entry)
    try:
        write_sysfs(path, payload)
        entry["write_call_returned"] = True
        for attempt in range(10):
            readback = int(read_sysfs(path).decode("ascii").strip())
            entry["readback_khz"] = readback
            entry["readback_samples"].append(
                {"attempt": attempt + 1, "monotonic_ns": time.monotonic_ns(), "value_khz": readback}
            )
            if readback == value:
                break
            time.sleep(0.02)
        require(entry["readback_khz"] == value, f"readback did not settle after {phase} write: {path}")
        require(frequency._identity(policy) == expected_identity, f"policy identity changed after {phase} write: {core}")
    except (OSError, UnicodeDecodeError, ValueError, LiveGateAError, frequency.PreparationError) as exc:
        entry["failure"] = str(exc)
        raise LiveGateAError(f"{phase} write failed for {path}: {exc}") from exc
    finally:
        entry["end_monotonic_ns"] = time.monotonic_ns()


def pin_policies(snapshots: dict[int, dict[str, Any]], ledger: list[dict[str, Any]]) -> None:
    for core in CORES:
        for field in ("scaling_min_freq", "scaling_max_freq"):
            write_frequency_and_settle(
                phase="prepare",
                core=core,
                field=field,
                value=REQUIRED_FREQUENCY_KHZ,
                ledger=ledger,
                expected_identity=snapshots[core]["identity"],
            )


def restore_policies(
    snapshots: dict[int, dict[str, Any]], ledger: list[dict[str, Any]]
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    readback: dict[str, Any] = {}
    for core in CORES:
        snapshot = snapshots[core]
        for field in ("scaling_min_freq", "scaling_max_freq"):
            try:
                write_frequency_and_settle(
                    phase="restore",
                    core=core,
                    field=field,
                    value=int(snapshot[field]["value"]),
                    ledger=ledger,
                    expected_identity=snapshot["identity"],
                )
            except (OSError, frequency.PreparationError, LiveGateAError) as exc:
                errors.append(str(exc))
    for core in CORES:
        try:
            policy = frequency._snapshot_policy(Path("/sys"), core, read_bytes=read_sysfs)
            readback[str(core)] = {
                "identity": policy["identity"],
                "scaling_min_freq": policy["scaling_min_freq"]["value"],
                "scaling_max_freq": policy["scaling_max_freq"]["value"],
            }
            require(policy["identity"] == snapshots[core]["identity"], f"policy {core} identity changed")
            require(
                policy["scaling_min_freq"]["value"] == snapshots[core]["scaling_min_freq"]["value"]
                and policy["scaling_max_freq"]["value"] == snapshots[core]["scaling_max_freq"]["value"],
                f"policy {core} restoration readback mismatch",
            )
        except (OSError, frequency.PreparationError, LiveGateAError) as exc:
            errors.append(str(exc))
    return not errors, errors, readback


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_worker_monitored(
    binary: Path,
    runtime_output: Path,
    bundle_sha256: str,
    temp_path: Path,
    pilot_variant: str,
) -> tuple[int, str, str, list[dict[str, Any]], str | None]:
    if pilot_variant in CODED_PREPROJECTION_VARIANTS:
        schedule_sha256 = coded_preprojection_schedule_sha256(pilot_variant)
        read_hz = CODED_PREPROJECTION_READ_HZ
    elif pilot_variant in READONLY_VARIANTS:
        schedule_sha256 = READONLY_MICRO_SCHEDULE_SHA256
        read_hz = READONLY_MICRO_READ_HZ
    else:
        schedule_sha256 = SCHEDULE_SHA256
        read_hz = LEGACY_READ_HZ
    command = [
        str(binary),
        "--execute-authorized",
        "--authority-sha256",
        USER_DIRECTIVE_SHA256,
        "--schedule-sha256",
        schedule_sha256,
        "--execution-bundle-sha256",
        bundle_sha256,
        "--output-root",
        str(runtime_output),
        "--sender-core",
        "4",
        "--receiver-core",
        "5",
        "--read-hz",
        str(read_hz),
        "--slot-s",
        "0.5",
        "--temperature-veto-c",
        "68.0",
        "--required-frequency-khz",
        "1600000",
        "--pilot-variant",
        pilot_variant,
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observations: list[dict[str, Any]] = []
    veto: str | None = None
    start = time.monotonic()
    receiver_exact_seen = False
    sender_step_exact_seen = False
    sender_anchor_exact_seen = False
    readonly_timing = pilot_variant in READONLY_TIMING_VARIANTS
    coded_preprojection = pilot_variant in CODED_PREPROJECTION_VARIANTS
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - start
            temperature_c = read_temperature_c(temp_path)
            frequencies = current_frequencies()
            limits = current_limits()
            observations.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "elapsed_s": elapsed,
                    "temperature_c": temperature_c,
                    "frequency_khz": frequencies,
                    "limits_khz": limits,
                }
            )
            if temperature_c >= TEMPERATURE_VETO_C:
                veto = f"temperature veto reached: {temperature_c} C"
            elif any(
                pair["min"] != REQUIRED_FREQUENCY_KHZ or pair["max"] != REQUIRED_FREQUENCY_KHZ
                for pair in limits.values()
            ):
                veto = f"frequency limit drift during runtime: {limits}"
            elif elapsed >= 0.5 and frequencies["5"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active receiver frequency drift during runtime: {frequencies}"
            elif readonly_timing and (
                (coded_preprojection and 1.05 <= elapsed <= 7.45)
                or ((not coded_preprojection) and 1.05 <= elapsed <= 2.95)
            ) and frequencies["4"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active micro-stimulus frequency drift during runtime: {frequencies}"
            elif (not readonly_timing) and pilot_variant not in {"step-sham"} and 3.15 <= elapsed <= (
                3.35 if pilot_variant == "impulse" else 4.85
            ) and frequencies["4"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active step-sender frequency drift during runtime: {frequencies}"
            elif (not readonly_timing) and 6.15 <= elapsed <= 6.85 and frequencies["4"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active anchor-sender frequency drift during runtime: {frequencies}"
            elif elapsed > WORKER_TIMEOUT_S:
                veto = "worker timeout"
            receiver_exact_seen = receiver_exact_seen or (
                elapsed >= 0.5 and frequencies["5"] == REQUIRED_FREQUENCY_KHZ
            )
            sender_step_exact_seen = sender_step_exact_seen or (
                (
                    (
                        readonly_timing
                        and (
                            (coded_preprojection and 1.05 <= elapsed <= 7.45)
                            or ((not coded_preprojection) and 1.05 <= elapsed <= 2.95)
                        )
                    )
                    or (
                        (not readonly_timing)
                        and 3.15 <= elapsed <= (3.35 if pilot_variant == "impulse" else 4.85)
                    )
                )
                and frequencies["4"] == REQUIRED_FREQUENCY_KHZ
            )
            sender_anchor_exact_seen = sender_anchor_exact_seen or (
                (readonly_timing or 6.15 <= elapsed <= 6.85)
                and frequencies["4"] == REQUIRED_FREQUENCY_KHZ
            )
            if veto is not None:
                stop_process(process)
                break
            time.sleep(MONITOR_INTERVAL_S)
        stdout, stderr = process.communicate(timeout=3)
    finally:
        stop_process(process)
    anchor_complete = readonly_timing or pilot_variant == "anchor-sham" or sender_anchor_exact_seen
    step_complete = pilot_variant == "step-sham" or sender_step_exact_seen
    if veto is None and not (receiver_exact_seen and step_complete and anchor_complete):
        veto = "scheduled active-frequency observation incomplete"
    return int(process.returncode or 0), stdout, stderr, observations, veto


def exact_permutation_p(step: list[float], off: list[float]) -> float:
    combined = step + off
    observed = abs(statistics.fmean(step) - statistics.fmean(off))
    exceed = 0
    total = 0
    indexes = range(len(combined))
    for chosen in itertools.combinations(indexes, len(step)):
        selected = set(chosen)
        left = [combined[index] for index in indexes if index in selected]
        right = [combined[index] for index in indexes if index not in selected]
        if abs(statistics.fmean(left) - statistics.fmean(right)) >= observed - 1e-15:
            exceed += 1
        total += 1
    return exceed / total


def parse_sample_timing(path: Path) -> list[dict[str, int]]:
    data = path.read_bytes()
    diagnostic_path = path.with_name("TIMING_DIAGNOSTIC_SUMMARY.json")
    declared_schema: str | None = None
    declared_record_bytes: int | None = None
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        declared_schema = diagnostic.get("sample_timing_schema_id")
        declared_record_bytes = diagnostic.get("sample_timing_record_bytes")
    if declared_schema == SAMPLE_TIMING_SCHEMA_ID or declared_record_bytes == SAMPLE_TIMING_RECORD_BYTES:
        schema = SAMPLE_TIMING_SCHEMA_ID
        record_struct = SAMPLE_TIMING_RECORD_V2
    elif declared_schema == SAMPLE_TIMING_RECORD_V1_SCHEMA_ID or declared_record_bytes == SAMPLE_TIMING_RECORD_V1_BYTES:
        schema = SAMPLE_TIMING_RECORD_V1_SCHEMA_ID
        record_struct = SAMPLE_TIMING_RECORD_V1
    elif len(data) % SAMPLE_TIMING_RECORD_BYTES == 0:
        schema = SAMPLE_TIMING_SCHEMA_ID
        record_struct = SAMPLE_TIMING_RECORD_V2
    elif len(data) % SAMPLE_TIMING_RECORD_V1_BYTES == 0:
        schema = SAMPLE_TIMING_RECORD_V1_SCHEMA_ID
        record_struct = SAMPLE_TIMING_RECORD_V1
    else:
        raise LiveGateAError("sample timing file has partial record")
    require(len(data) % record_struct.size == 0, "sample timing file has partial record")
    rows: list[dict[str, int]] = []
    previous_requested: int | None = None
    previous_finished: int | None = None
    for index, record in enumerate(record_struct.iter_unpack(data)):
        values = [int(value) for value in record]
        if schema == SAMPLE_TIMING_SCHEMA_ID:
            (
                requested_sample_index,
                requested_tsc,
                requested_slot,
                started_tsc,
                finished_tsc,
                actual_slot,
                scheduler_lateness_ticks,
                service_ticks,
                missed_deadlines_before_sample,
                valid_measurement,
            ) = values
        else:
            (
                requested_sample_index,
                requested_tsc,
                started_tsc,
                finished_tsc,
                scheduler_lateness_ticks,
                service_ticks,
                _finish_gap_ticks,
                requested_slot,
            ) = values
            actual_slot = requested_slot
            missed_deadlines_before_sample = 0
            valid_measurement = 1
        require(requested_sample_index >= index, "sample timing requested index drift")
        require(requested_slot < 16 and actual_slot < 16, "sample timing slot out of range")
        require(started_tsc >= requested_tsc, "sample timing started before requested")
        require(finished_tsc >= started_tsc, "sample timing finished before started")
        require(
            scheduler_lateness_ticks == max(0, started_tsc - requested_tsc),
            "sample timing lateness mismatch",
        )
        require(service_ticks == finished_tsc - started_tsc, "sample timing service mismatch")
        if previous_requested is not None:
            require(requested_tsc > previous_requested, "sample timing requested timestamp went backward")
            require(finished_tsc >= int(previous_finished), "sample timing finished timestamp went backward")
        rows.append(
            {
                "sample_index": index,
                "requested_sample_index": requested_sample_index,
                "requested_tsc": requested_tsc,
                "requested_slot": requested_slot,
                "started_tsc": started_tsc,
                "finished_tsc": finished_tsc,
                "actual_slot": actual_slot,
                "scheduler_lateness_ticks": scheduler_lateness_ticks,
                "service_ticks": service_ticks,
                "missed_deadlines_before_sample": missed_deadlines_before_sample,
                "valid_measurement": valid_measurement,
                "schema_id": schema,
            }
        )
        previous_requested = requested_tsc
        previous_finished = finished_tsc
    return rows


def sample_timing_summary(runtime_root: Path, expected_count: int) -> dict[str, Any] | None:
    path = runtime_root / "sample_timing.bin"
    diagnostic_path = runtime_root / "TIMING_DIAGNOSTIC_SUMMARY.json"
    if not path.is_file():
        return None
    rows = parse_sample_timing(path)
    require(len(rows) == expected_count, "sample timing count does not match raw samples")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.is_file() else None
    if diagnostic is not None:
        require(
            diagnostic.get("sample_timing_schema_id") in {SAMPLE_TIMING_SCHEMA_ID, SAMPLE_TIMING_RECORD_V1_SCHEMA_ID},
            "sample timing schema mismatch",
        )
    slot_count = len(diagnostic.get("sample_count_per_slot", [])) if diagnostic is not None else 0
    if slot_count <= 0:
        slot_count = max((row["actual_slot"] for row in rows), default=15) + 1
        slot_count = max(slot_count, 16)
    slot_counts = [0 for _ in range(slot_count)]
    for row in rows:
        if 0 <= row["actual_slot"] < slot_count:
            slot_counts[row["actual_slot"]] += 1
    if diagnostic is not None:
        require(slot_counts == diagnostic.get("sample_count_per_slot"), "diagnostic slot counts mismatch")
    lateness = [row["scheduler_lateness_ticks"] for row in rows]
    service_cycles = [row["service_ticks"] / 64.0 for row in rows]
    return {
        "sample_timing_schema_id": rows[0]["schema_id"] if rows else (
            diagnostic.get("sample_timing_schema_id") if diagnostic is not None else None
        ),
        "sample_timing_record_bytes": SAMPLE_TIMING_RECORD_BYTES if rows and rows[0]["schema_id"] == SAMPLE_TIMING_SCHEMA_ID else SAMPLE_TIMING_RECORD_V1_BYTES,
        "sample_count": len(rows),
        "sample_count_per_slot": slot_counts,
        "max_scheduler_lateness_ticks": max(lateness) if lateness else 0,
        "max_service_cycles_per_access": max(service_cycles) if service_cycles else 0.0,
        "skipped_deadline_count": sum(row["missed_deadlines_before_sample"] for row in rows),
        "requested_actual_slot_mismatch_count": sum(
            1 for row in rows if row["requested_slot"] != row["actual_slot"]
        ),
        "diagnostic_classification": (
            diagnostic["capture_quality_classification"] if diagnostic is not None else "BASIC_TIMING_FILE_ONLY"
        ),
        "diagnostic": diagnostic,
    }


def self_test_timing_parser() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gate_a_timing_parser_") as directory:
        root = Path(directory)
        records = [
            (0, 1000, 0, 1000, 1064, 0, 0, 64, 0, 1),
            (1, 2000, 0, 2010, 2074, 1, 10, 64, 0, 1),
            (4, 5000, 1, 5010, 5140, 1, 10, 130, 2, 1),
        ]
        payload = b"".join(SAMPLE_TIMING_RECORD_V2.pack(*record) for record in records)
        (root / "sample_timing.bin").write_bytes(payload)
        (root / "TIMING_DIAGNOSTIC_SUMMARY.json").write_text(
            json.dumps(
                {
                    "sample_timing_schema_id": SAMPLE_TIMING_SCHEMA_ID,
                    "sample_count_per_slot": [1, 2] + [0] * 14,
                    "capture_quality_classification": "CAPTURE_ACCEPTED",
                }
            ),
            encoding="utf-8",
        )
        summary = sample_timing_summary(root, 3)
        require(
            summary is not None
            and summary["sample_count"] == 3
            and summary["skipped_deadline_count"] == 2
            and summary["requested_actual_slot_mismatch_count"] == 1,
            "parser V2 self-test failed",
        )
        legacy = root / "legacy"
        legacy.mkdir()
        legacy_records = [
            (0, 1000, 1000, 1064, 0, 64, 0, 0),
            (1, 2000, 2010, 2074, 10, 64, 1010, 0),
            (2, 3000, 3000, 3130, 0, 130, 1056, 1),
        ]
        legacy_payload = b"".join(SAMPLE_TIMING_RECORD_V1.pack(*record) for record in legacy_records)
        (legacy / "sample_timing.bin").write_bytes(legacy_payload)
        (legacy / "TIMING_DIAGNOSTIC_SUMMARY.json").write_text(
            json.dumps(
                {
                    "sample_timing_schema_id": SAMPLE_TIMING_RECORD_V1_SCHEMA_ID,
                    "sample_count_per_slot": [2, 1] + [0] * 14,
                    "capture_quality_classification": "CAPTURE_ACCEPTED",
                }
            ),
            encoding="utf-8",
        )
        legacy_summary = sample_timing_summary(legacy, 3)
        require(
            legacy_summary is not None
            and legacy_summary["sample_timing_record_bytes"] == SAMPLE_TIMING_RECORD_V1_BYTES,
            "parser V1 self-test failed",
        )
        basic = root / "basic"
        basic.mkdir()
        basic_records = [
            (0, 1000, 0, 1000, 1000, 0, 0, 0, 0, 1),
            (1, 2000, 0, 2010, 2010, 0, 10, 0, 0, 1),
            (2, 3000, 1, 3020, 3020, 1, 20, 0, 0, 1),
        ]
        (basic / "sample_timing.bin").write_bytes(
            b"".join(SAMPLE_TIMING_RECORD_V2.pack(*record) for record in basic_records)
        )
        basic_summary = sample_timing_summary(basic, 3)
        require(
            basic_summary is not None
            and basic_summary["diagnostic_classification"] == "BASIC_TIMING_FILE_ONLY"
            and basic_summary["sample_count_per_slot"][:2] == [2, 1],
            "basic timing parser self-test failed",
        )
    print(json.dumps({"status": "GATE_A_TIMING_PARSER_SELF_TEST_OK", "hardware_executions": 0}))
    return 0


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def analyze_readonly_micro_runtime(runtime_root: Path, pilot_variant: str) -> dict[str, Any]:
    lockin = [json.loads(line) for line in (runtime_root / "LOCKIN_IQ.jsonl").read_text().splitlines() if line]
    require(len(lockin) == 8, "expected 8 micro-schedule lock-in records")
    raw_bytes = (runtime_root / "raw_samples.bin").read_bytes()
    require(len(raw_bytes) % 16 == 0, "raw sample file is not packed Qd records")
    raw = [(int(tsc), float(value)) for tsc, value in struct.iter_unpack("<Qd", raw_bytes)]
    timing_summary = sample_timing_summary(runtime_root, len(raw))
    require(timing_summary is not None, "micro timing summary missing")
    timing_rows = parse_sample_timing(runtime_root / "sample_timing.bin")
    require(len(timing_rows) == len(raw), "micro raw/timing count mismatch")
    diagnostic = timing_summary["diagnostic"]
    burst_by_slot = {
        int(row["slot_index"]): row for row in diagnostic.get("sender_burst_boundaries", [])
    }
    require(all(slot in burst_by_slot for slot in range(2, 6)), "micro burst boundary missing")

    slot_summaries: list[dict[str, Any]] = []
    during_means: dict[int, float | None] = {}
    whole_means: dict[int, float | None] = {}
    sufficient = True
    for slot in range(8):
        slot_values = [
            value for index, (_tsc, value) in enumerate(raw)
            if timing_rows[index]["actual_slot"] == slot and timing_rows[index]["valid_measurement"] == 1
        ]
        burst = burst_by_slot.get(slot)
        before_values: list[float] = []
        during_values: list[float] = []
        after_values: list[float] = []
        if burst is not None:
            burst_start = int(burst["burst_start_tsc"])
            burst_finish = int(burst["burst_finish_tsc"])
            for index, (_tsc, value) in enumerate(raw):
                row = timing_rows[index]
                if row["actual_slot"] != slot or row["valid_measurement"] != 1:
                    continue
                if row["finished_tsc"] <= burst_start:
                    before_values.append(value)
                elif row["started_tsc"] >= burst_finish:
                    after_values.append(value)
                elif row["started_tsc"] < burst_finish and row["finished_tsc"] > burst_start:
                    during_values.append(value)
            if len(during_values) < 10:
                sufficient = False
        whole_means[slot] = _mean(slot_values)
        during_means[slot] = _mean(during_values)
        slot_summaries.append(
            {
                "slot_index": slot,
                "token": lockin[slot]["token"],
                "whole_slot": {
                    "sample_count": len(slot_values),
                    "mean_response_cycles": whole_means[slot],
                },
                "before_burst": {
                    "sample_count": len(before_values),
                    "mean_response_cycles": _mean(before_values),
                },
                "during_burst": {
                    "sample_count": len(during_values),
                    "mean_response_cycles": during_means[slot],
                },
                "after_burst": {
                    "sample_count": len(after_values),
                    "mean_response_cycles": _mean(after_values),
                },
                "burst": burst,
            }
        )

    first_slots = [2, 5]
    second_slots = [3, 4]
    during_first = [during_means[slot] for slot in first_slots if during_means[slot] is not None]
    during_second = [during_means[slot] for slot in second_slots if during_means[slot] is not None]
    whole_first = [whole_means[slot] for slot in first_slots if whole_means[slot] is not None]
    whole_second = [whole_means[slot] for slot in second_slots if whole_means[slot] is not None]
    primary_contrast = None
    whole_slot_contrast = None
    if len(during_first) == 2 and len(during_second) == 2:
        primary_contrast = statistics.fmean(during_second) - statistics.fmean(during_first)
    if len(whole_first) == 2 and len(whole_second) == 2:
        whole_slot_contrast = statistics.fmean(whole_second) - statistics.fmean(whole_first)
    return {
        "schema_id": "CAT_CAS_READONLY_OCCUPANCY_MICRO_ANALYSIS_V1",
        "pilot_variant": pilot_variant,
        "measurement_mode": "catcas_readonly_occupancy_response_cycles",
        "schedule": {
            "schedule_sha256": READONLY_MICRO_SCHEDULE_SHA256,
            "slot_count": 8,
            "slot_duration_s": 0.5,
            "duration_s": 4.0,
            "read_hz": READONLY_MICRO_READ_HZ,
            "tokens": [row["token"] for row in lockin],
            "stimulus_slots": [2, 3, 4, 5],
            "primary_coordinate": "during_burst_response_cycles",
            "minimum_during_burst_samples_per_stimulus_slot": 10,
        },
        "sample_timing": timing_summary,
        "slot_response_summaries": slot_summaries,
        "during_burst_second_minus_first": primary_contrast,
        "whole_slot_second_minus_first": whole_slot_contrast,
        "sufficient_during_burst_samples": sufficient,
        "all_bursts_completed_inside_slots": all(
            bool(burst_by_slot[slot].get("completed_before_slot_end")) for slot in range(2, 6)
        ),
        "engineering_first_light_candidate": False,
        "claim_ceiling": "micro-triad engineering diagnostic only; no final occupancy or Small Wall claim",
    }


CODED_PHASES = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)


def reconstruct_coded_complex(values: list[float]) -> dict[str, float]:
    require(len(values) == len(CODED_PHASES), "coded decoder requires four phases")
    total = 0.0 + 0.0j
    for value, phase in zip(values, CODED_PHASES):
        total += value * complex(math.cos(phase), math.sin(phase))
    z = (2.0 / float(len(CODED_PHASES))) * total
    return {"real": z.real, "imag": z.imag, "abs": abs(z)}


def analyze_coded_preprojection_runtime(runtime_root: Path, pilot_variant: str) -> dict[str, Any]:
    lockin = [json.loads(line) for line in (runtime_root / "LOCKIN_IQ.jsonl").read_text().splitlines() if line]
    require(len(lockin) == 16, "expected 16 coded-loop lock-in records")
    raw_bytes = (runtime_root / "raw_samples.bin").read_bytes()
    require(len(raw_bytes) % 16 == 0, "raw sample file is not packed Qd records")
    raw = [(int(tsc), float(value)) for tsc, value in struct.iter_unpack("<Qd", raw_bytes)]
    timing_summary = sample_timing_summary(runtime_root, len(raw))
    require(timing_summary is not None, "coded-loop timing summary missing")
    timing_rows = parse_sample_timing(runtime_root / "sample_timing.bin")
    require(len(timing_rows) == len(raw), "coded-loop raw/timing count mismatch")
    diagnostic = timing_summary["diagnostic"]
    burst_by_slot = {
        int(row["slot_index"]): row for row in diagnostic.get("sender_burst_boundaries", [])
    }
    if pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_VARIANTS:
        slot_by_token = {str(row["token"]): int(row["slot_index"]) for row in lockin}
        plus_slots = [slot_by_token[f"P{index}"] for index in range(4)]
        minus_slots = [slot_by_token[f"M{index}"] for index in range(4)]
        post_slots = [slot_by_token[f"C{index}"] for index in range(4)]
        source_off_slots = (2, 15)
        neutral_before_slot = 2
        neutral_after_slot = 15
        schedule_sha256 = coded_preprojection_schedule_sha256(pilot_variant)
    elif pilot_variant in CODED_PREPROJECTION_WARM_RESTORED_VARIANTS:
        plus_slots = [3, 4, 5, 6]
        minus_slots = [7, 8, 9, 10]
        post_slots = [11, 12, 13, 14]
        source_off_slots = (2, 15)
        neutral_before_slot = 2
        neutral_after_slot = 15
        schedule_sha256 = coded_preprojection_schedule_sha256(pilot_variant)
    elif pilot_variant in CODED_PREPROJECTION_RESTORED_VARIANTS:
        plus_slots = [2, 3, 4, 5]
        minus_slots = [6, 7, 8, 9]
        post_slots = [10, 11, 12, 13]
        source_off_slots = (1, 14)
        neutral_before_slot = 1
        neutral_after_slot = 14
        schedule_sha256 = CODED_PREPROJECTION_RESTORED_SCHEDULE_SHA256
    else:
        plus_slots = [2, 3, 4, 5]
        minus_slots = [6, 7, 8, 9]
        post_slots = [10, 11, 12, 13]
        source_off_slots = (1, 15)
        neutral_before_slot = 0
        neutral_after_slot = 14
        schedule_sha256 = CODED_PREPROJECTION_SCHEDULE_SHA256
    stimulus_slots = plus_slots + minus_slots + post_slots
    require(all(slot in burst_by_slot for slot in stimulus_slots), "coded-loop burst boundary missing")

    slot_summaries: list[dict[str, Any]] = []
    during_means: dict[int, float | None] = {}
    whole_means: dict[int, float | None] = {}
    sufficient = True
    for slot in range(16):
        slot_values = [
            value for index, (_tsc, value) in enumerate(raw)
            if timing_rows[index]["actual_slot"] == slot and timing_rows[index]["valid_measurement"] == 1
        ]
        burst = burst_by_slot.get(slot)
        before_values: list[float] = []
        during_values: list[float] = []
        after_values: list[float] = []
        if burst is not None:
            burst_start = int(burst["burst_start_tsc"])
            burst_finish = int(burst["burst_finish_tsc"])
            for index, (_tsc, value) in enumerate(raw):
                row = timing_rows[index]
                if row["actual_slot"] != slot or row["valid_measurement"] != 1:
                    continue
                if row["finished_tsc"] <= burst_start:
                    before_values.append(value)
                elif row["started_tsc"] >= burst_finish:
                    after_values.append(value)
                elif row["started_tsc"] < burst_finish and row["finished_tsc"] > burst_start:
                    during_values.append(value)
            if len(during_values) < 10:
                sufficient = False
        whole_means[slot] = _mean(slot_values)
        during_means[slot] = _mean(during_values)
        slot_summaries.append(
            {
                "slot_index": slot,
                "token": lockin[slot]["token"],
                "phase_index": None if slot not in stimulus_slots else (
                    (stimulus_slots.index(slot) % 4) * 2
                ),
                "whole_slot": {
                    "sample_count": len(slot_values),
                    "mean_response_cycles": whole_means[slot],
                },
                "before_burst": {
                    "sample_count": len(before_values),
                    "mean_response_cycles": _mean(before_values),
                },
                "during_burst": {
                    "sample_count": len(during_values),
                    "mean_response_cycles": during_means[slot],
                },
                "after_burst": {
                    "sample_count": len(after_values),
                    "mean_response_cycles": _mean(after_values),
                },
                "burst": burst,
            }
        )

    require(
        all(during_means[slot] is not None for slot in plus_slots + minus_slots + post_slots),
        "coded-loop missing during-burst mean",
    )
    plus_values = [float(during_means[slot]) for slot in plus_slots]
    minus_values = [float(during_means[slot]) for slot in minus_slots]
    post_values = [float(during_means[slot]) for slot in post_slots]
    plus_net = [value - control for value, control in zip(plus_values, post_values)]
    minus_net = [value - control for value, control in zip(minus_values, post_values)]
    post_center = statistics.fmean(post_values)
    post_net = [value - post_center for value in post_values]
    plus_z = reconstruct_coded_complex(plus_net)
    minus_z = reconstruct_coded_complex(minus_net)
    post_z = reconstruct_coded_complex(post_net)
    source_off_values = [
        value for slot in source_off_slots
        for value in [whole_means[slot]]
        if value is not None
    ]
    source_off_range = max(source_off_values) - min(source_off_values) if source_off_values else None
    neutral_before = whole_means[neutral_before_slot]
    neutral_after = whole_means[neutral_after_slot]
    neutral_delta = (
        abs(neutral_after - neutral_before)
        if neutral_before is not None and neutral_after is not None
        else None
    )
    source_phase_chop = pilot_variant in CODED_PREPROJECTION_SOURCE_PHASE_CHOP_VARIANTS
    source_phase_chop_slots: dict[int, dict[str, Any]] = {}
    source_phase_chop_sufficient = True
    source_phase_chop_plus_mean: float | None = None
    source_phase_chop_minus_mean: float | None = None
    source_phase_chop_control_mean: float | None = None
    source_phase_chop_control_floor: float | None = None
    source_phase_chop_opposed = False
    source_phase_chop_exceeds_controls = False
    source_phase_chop_signal = False
    if source_phase_chop:
        for slot in stimulus_slots:
            burst = burst_by_slot[slot]
            burst_start = int(burst["burst_start_tsc"])
            burst_finish = int(burst["burst_finish_tsc"])
            require(burst_finish > burst_start, "source phase-chop burst has nonpositive span")
            segment_values: list[list[float]] = [[] for _ in range(4)]
            span = burst_finish - burst_start
            for index, (_tsc, value) in enumerate(raw):
                row = timing_rows[index]
                if row["actual_slot"] != slot or row["valid_measurement"] != 1:
                    continue
                sample_mid = (row["started_tsc"] + row["finished_tsc"]) // 2
                if sample_mid < burst_start or sample_mid >= burst_finish:
                    continue
                segment = int(((sample_mid - burst_start) * 4) // span)
                segment = max(0, min(3, segment))
                segment_values[segment].append(value)
            segment_counts = [len(values) for values in segment_values]
            segment_means = [_mean(values) for values in segment_values]
            if any(count < 2 for count in segment_counts) or any(value is None for value in segment_means):
                source_phase_chop_sufficient = False
            token = str(lockin[slot]["token"])
            phase_digit = int(token[1]) if len(token) == 2 and token[1].isdigit() else 0
            if all(value is not None for value in segment_means):
                raw_coordinate = reconstruct_coded_complex([float(value) for value in segment_means])
                z = complex(raw_coordinate["real"], raw_coordinate["imag"])
                phase = CODED_PHASES[phase_digit % len(CODED_PHASES)]
                aligned = z * complex(math.cos(-phase), math.sin(-phase))
                aligned_coordinate = {
                    "real": aligned.real,
                    "imag": aligned.imag,
                    "abs": abs(aligned),
                }
            else:
                raw_coordinate = None
                aligned_coordinate = None
            source_phase_chop_slots[slot] = {
                "slot_index": slot,
                "token": token,
                "segment_sample_counts": segment_counts,
                "segment_mean_response_cycles": segment_means,
                "raw_lockin_coordinate": raw_coordinate,
                "phase_aligned_coordinate": aligned_coordinate,
                "phase_aligned_real": (
                    aligned_coordinate["real"] if aligned_coordinate is not None else None
                ),
            }
        plus_aligned = [
            source_phase_chop_slots[slot]["phase_aligned_real"]
            for slot in plus_slots
            if source_phase_chop_slots[slot]["phase_aligned_real"] is not None
        ]
        minus_aligned = [
            source_phase_chop_slots[slot]["phase_aligned_real"]
            for slot in minus_slots
            if source_phase_chop_slots[slot]["phase_aligned_real"] is not None
        ]
        control_aligned = [
            source_phase_chop_slots[slot]["phase_aligned_real"]
            for slot in post_slots
            if source_phase_chop_slots[slot]["phase_aligned_real"] is not None
        ]
        require(
            len(plus_aligned) == 4 and len(minus_aligned) == 4 and len(control_aligned) == 4,
            "source phase-chop missing aligned coordinates",
        )
        source_phase_chop_plus_mean = statistics.fmean(plus_aligned)
        source_phase_chop_minus_mean = statistics.fmean(minus_aligned)
        source_phase_chop_control_mean = statistics.fmean(control_aligned)
        control_spread = max(control_aligned) - min(control_aligned)
        source_phase_chop_control_floor = max(
            abs(source_phase_chop_control_mean),
            control_spread,
            source_off_range or 0.0,
        )
        source_phase_chop_opposed = source_phase_chop_plus_mean * source_phase_chop_minus_mean < 0.0
        source_phase_chop_exceeds_controls = (
            min(abs(source_phase_chop_plus_mean), abs(source_phase_chop_minus_mean)) >
            3.0 * source_phase_chop_control_floor
        )
    fold_odd_opposed = plus_z["imag"] * minus_z["imag"] < 0.0
    fold_odd_balance = abs(abs(plus_z["imag"]) - abs(minus_z["imag"]))
    control_floor = max(abs(post_z["imag"]), source_off_range or 0.0)
    fold_odd_exceeds_controls = min(abs(plus_z["imag"]), abs(minus_z["imag"])) > 3.0 * control_floor
    neutral_restoration_tolerance = 5.0
    neutral_restoration_passed = (
        neutral_delta is not None and neutral_delta <= neutral_restoration_tolerance
    )
    fold_odd_signal_candidate = bool(
        sufficient
        and fold_odd_opposed
        and fold_odd_exceeds_controls
        and all(bool(burst_by_slot[slot].get("completed_before_slot_end")) for slot in stimulus_slots)
    )
    if source_phase_chop:
        fold_odd_opposed = source_phase_chop_opposed
        fold_odd_balance = abs(abs(float(source_phase_chop_plus_mean)) - abs(float(source_phase_chop_minus_mean)))
        control_floor = float(source_phase_chop_control_floor)
        fold_odd_exceeds_controls = source_phase_chop_exceeds_controls
        source_phase_chop_signal = bool(
            source_phase_chop_sufficient
            and source_phase_chop_opposed
            and source_phase_chop_exceeds_controls
            and all(bool(burst_by_slot[slot].get("completed_before_slot_end")) for slot in stimulus_slots)
        )
        fold_odd_signal_candidate = source_phase_chop_signal
    null_control_kind = (
        "query_scramble"
        if pilot_variant in CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS
        else (
            "query_off"
            if pilot_variant in CODED_PREPROJECTION_QUERY_OFF_VARIANTS
            else (
                "declaration_sham"
                if pilot_variant in CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS
                else (
                    "phase_local_sham"
                    if pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS
                    else None
                )
            )
        )
    )
    coded_null_control = pilot_variant in CODED_PREPROJECTION_NULL_CONTROL_VARIANTS
    coded_null_bound = max(5.0, 3.0 * control_floor)
    coded_null_passed = bool(
        coded_null_control
        and neutral_restoration_passed
        and sufficient
        and all(bool(burst_by_slot[slot].get("completed_before_slot_end")) for slot in stimulus_slots)
        and max(abs(plus_z["imag"]), abs(minus_z["imag"])) <= coded_null_bound
    )
    engineering_candidate = bool(
        fold_odd_signal_candidate
        and neutral_restoration_passed
        and not coded_null_control
    )
    active_query = pilot_variant in CODED_PREPROJECTION_ACTIVE_QUERY_VARIANTS
    measurement_mode = (
        "catcas_source_phase_chop_response_cycles"
        if source_phase_chop else (
            "catcas_active_query_delta_cycles"
            if active_query else "catcas_coded_preprojection_response_cycles"
        )
    )
    primary_coordinate = (
        "source-side in-slot phase-chop lock-in aligned by public token phase"
        if source_phase_chop else (
            "active-query balanced subbank delta quadrature fold-odd response"
            if active_query else "post-control-centered quadrature fold-odd response"
        )
    )
    return {
        "schema_id": "CAT_CAS_CODED_PREPROJECTION_LOOP_ANALYSIS_V1",
        "pilot_variant": pilot_variant,
        "measurement_mode": measurement_mode,
        "active_query_receiver_delta": active_query,
        "source_phase_chop": source_phase_chop,
        "schedule": {
            "schedule_sha256": schedule_sha256,
            "slot_count": 16,
            "slot_duration_s": 0.5,
            "duration_s": 8.0,
            "read_hz": CODED_PREPROJECTION_READ_HZ,
            "tokens": [row["token"] for row in lockin],
            "primary_coordinate": primary_coordinate,
            "stimulus_slots": stimulus_slots,
            "pre_plus_slots": plus_slots,
            "pre_minus_slots": minus_slots,
            "post_projection_control_slots": post_slots,
            "neutral_restoration_slots": [neutral_before_slot, neutral_after_slot],
            "source_off_control_slots": list(source_off_slots),
        },
        "sample_timing": timing_summary,
        "slot_response_summaries": slot_summaries,
        "during_burst_means": {
            "pre_projection_private_fold_plus": plus_values,
            "pre_projection_private_fold_minus": minus_values,
            "post_projection_control": post_values,
        },
        "post_control_centered_responses": {
            "pre_projection_private_fold_plus": plus_net,
            "pre_projection_private_fold_minus": minus_net,
            "post_projection_control": post_net,
        },
        "decoded_coordinates": {
            "pre_projection_private_fold_plus": plus_z,
            "pre_projection_private_fold_minus": minus_z,
            "post_projection_control": post_z,
        },
        "source_phase_chop_lockin": (
            {
                "slots": source_phase_chop_slots,
                "sufficient_segment_samples": source_phase_chop_sufficient,
                "pre_projection_private_fold_plus_mean": source_phase_chop_plus_mean,
                "pre_projection_private_fold_minus_mean": source_phase_chop_minus_mean,
                "post_projection_control_mean": source_phase_chop_control_mean,
                "control_floor_cycles": source_phase_chop_control_floor,
                "opposed_sign": source_phase_chop_opposed,
                "exceeds_controls": source_phase_chop_exceeds_controls,
                "signal_candidate": source_phase_chop_signal,
            }
            if source_phase_chop else None
        ),
        "source_off_whole_slot_range_cycles": source_off_range,
        "neutral_restoration": {
            "before_slot": neutral_before_slot,
            "after_slot": neutral_after_slot,
            "before_mean_response_cycles": neutral_before,
            "after_mean_response_cycles": neutral_after,
            "absolute_delta_cycles": neutral_delta,
            "tolerance_cycles": neutral_restoration_tolerance,
            "passed": neutral_restoration_passed,
        },
        "fold_odd_opposed": fold_odd_opposed,
        "fold_odd_balance_error_cycles": fold_odd_balance,
        "control_floor_cycles": control_floor,
        "fold_odd_exceeds_controls": fold_odd_exceeds_controls,
        "fold_odd_signal_candidate": fold_odd_signal_candidate,
        "null_control_kind": null_control_kind,
        "coded_null_control": coded_null_control,
        "coded_null_bound_cycles": coded_null_bound if coded_null_control else None,
        "coded_null_passed": coded_null_passed if coded_null_control else None,
        "query_scramble_control": pilot_variant in CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS,
        "query_scramble_null_bound_cycles": coded_null_bound if pilot_variant in CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS else None,
        "query_scramble_null_passed": coded_null_passed if pilot_variant in CODED_PREPROJECTION_QUERY_SCRAMBLE_VARIANTS else None,
        "query_off_control": pilot_variant in CODED_PREPROJECTION_QUERY_OFF_VARIANTS,
        "query_off_null_bound_cycles": coded_null_bound if pilot_variant in CODED_PREPROJECTION_QUERY_OFF_VARIANTS else None,
        "query_off_null_passed": coded_null_passed if pilot_variant in CODED_PREPROJECTION_QUERY_OFF_VARIANTS else None,
        "declaration_sham_control": pilot_variant in CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS,
        "declaration_sham_null_bound_cycles": coded_null_bound if pilot_variant in CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS else None,
        "declaration_sham_null_passed": coded_null_passed if pilot_variant in CODED_PREPROJECTION_DECLARATION_SHAM_VARIANTS else None,
        "phase_local_sham_control": pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS,
        "phase_local_sham_null_bound_cycles": coded_null_bound if pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS else None,
        "phase_local_sham_null_passed": coded_null_passed if pilot_variant in CODED_PREPROJECTION_PHASE_LOCAL_SHAM_VARIANTS else None,
        "sufficient_during_burst_samples": sufficient,
        "all_bursts_completed_inside_slots": all(
            bool(burst_by_slot[slot].get("completed_before_slot_end")) for slot in stimulus_slots
        ),
        "engineering_candidate": engineering_candidate,
        "engineering_first_light_candidate": engineering_candidate,
        "claim_ceiling": (
            f"{null_control_kind.replace('_', '-')} killing control only; no OrbitState coupling, path memory, holonomy, or Small Wall claim"
            if coded_null_control
            else (
                "source-phase-chop coded-loop physical mapping only; requires matched source-phase sham before any Small Wall claim"
                if source_phase_chop else (
                    "active-query coded-loop physical mapping only; requires matched active-query controls before any Small Wall claim"
                    if active_query
                    else "single coded-loop physical mapping; no OrbitState coupling, path memory, holonomy, or Small Wall claim"
                )
            )
        ),
    }


def analyze_runtime(runtime_root: Path, pilot_variant: str) -> dict[str, Any]:
    if pilot_variant in CODED_PREPROJECTION_VARIANTS:
        return analyze_coded_preprojection_runtime(runtime_root, pilot_variant)
    if pilot_variant in READONLY_VARIANTS:
        return analyze_readonly_micro_runtime(runtime_root, pilot_variant)
    lockin = [json.loads(line) for line in (runtime_root / "LOCKIN_IQ.jsonl").read_text().splitlines() if line]
    require(len(lockin) == 16, "expected 16 lock-in records")
    measurement_modes = {str(row.get("measurement_mode", "ring_period_cycles")) for row in lockin}
    require(len(measurement_modes) == 1, "measurement mode drift")
    measurement_mode = measurement_modes.pop()
    raw_bytes = (runtime_root / "raw_samples.bin").read_bytes()
    require(len(raw_bytes) % 16 == 0, "raw sample file is not packed Qd records")
    raw = [record for record in struct.iter_unpack("<Qd", raw_bytes)]
    timing_summary = sample_timing_summary(runtime_root, len(raw))
    groups: dict[str, list[dict[str, Any]]] = {
        "off": [row for row in lockin if row["token"] in OFF_TOKENS],
        "step": [row for row in lockin if row["token"] == "S0E"],
        "positive_anchor": [row for row in lockin if row["token"] == "A0P"],
        "negative_anchor": [row for row in lockin if row["token"] == "A0N"],
    }
    require(all(groups.values()), "analysis group missing")

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "slot_count": len(rows),
            "mean_i": statistics.fmean(float(row["lockin_i"]) for row in rows),
            "mean_q": statistics.fmean(float(row["lockin_q"]) for row in rows),
            "mean_magnitude": statistics.fmean(float(row["magnitude"]) for row in rows),
            "mean_off_frequency_floor": statistics.fmean(float(row["off_frequency_floor"]) for row in rows),
        }

    summaries = {name: summarize(rows) for name, rows in groups.items()}
    step_magnitudes = [float(row["magnitude"]) for row in groups["step"]]
    off_magnitudes = [float(row["magnitude"]) for row in groups["off"]]
    positive = complex(summaries["positive_anchor"]["mean_i"], summaries["positive_anchor"]["mean_q"])
    negative = complex(summaries["negative_anchor"]["mean_i"], summaries["negative_anchor"]["mean_q"])
    phase_delta = math.atan2((negative / positive).imag, (negative / positive).real) if positive and negative else None

    ring_slots: list[dict[str, Any]] = []
    for row in lockin:
        start = int(row["raw_sample_start_index"])
        end = int(row["raw_sample_end_index"])
        values = [float(value) for _tsc, value in raw[start:end]]
        ring_slots.append(
            {
                "slot_index": int(row["slot_index"]),
                "token": row["token"],
                "sample_count": len(values),
                "mean_ring_period": statistics.fmean(values),
                "mean_response_cycles": statistics.fmean(values),
                "stdev_ring_period": statistics.pstdev(values),
                "stdev_response_cycles": statistics.pstdev(values),
            }
        )

    off_mean = summaries["off"]["mean_magnitude"]
    off_sigma = statistics.pstdev(off_magnitudes)
    step_mean = summaries["step"]["mean_magnitude"]
    anchor_floor = max(off_mean + 3.0 * off_sigma, off_mean * 1.5)
    anchor_opposed = phase_delta is not None and abs(abs(phase_delta) - math.pi) <= math.pi / 4.0
    return {
        "schema_id": "CAT_CAS_GATE_A_FIRST_LIGHT_ANALYSIS_V1",
        "pilot_variant": pilot_variant,
        "measurement_mode": measurement_mode,
        "executed_anchor_order": {
            "pn": ["positive", "negative"],
            "np": ["negative", "positive"],
            "anchor-sham": ["off", "off"],
            "impulse": ["positive", "negative"],
            "step-sham": ["positive", "negative"],
            "phase-forward": ["positive", "negative"],
            "phase-reverse": ["positive", "negative"],
            "value-forward": ["positive", "negative"],
            "value-reverse": ["positive", "negative"],
            "value-equal": ["positive", "negative"],
            "occupancy-forward": ["positive", "negative"],
            "occupancy-reverse": ["positive", "negative"],
            "occupancy-equal": ["positive", "negative"],
            "readonly-occupancy-forward": ["positive", "negative"],
            "readonly-occupancy-reverse": ["positive", "negative"],
            "readonly-occupancy-equal": ["positive", "negative"],
        }[pilot_variant],
        "groups": summaries,
        "sample_timing": timing_summary,
        "step_minus_off_magnitude": step_mean - off_mean,
        "step_vs_off_exact_permutation_p": exact_permutation_p(step_magnitudes, off_magnitudes),
        "anchor_phase_delta_rad": phase_delta,
        "anchor_opposed_within_pi_over_4": anchor_opposed,
        "ring_slots": ring_slots,
        "engineering_first_light_candidate": bool(
            step_mean > off_mean + 3.0 * off_sigma
            and abs(positive) > anchor_floor
            and abs(negative) > anchor_floor
            and anchor_opposed
        ),
        "claim_ceiling": "single-run engineering observation; no carrier-state or Small Wall claim",
    }


def build_file_manifest(output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "FILE_MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_id": "CAT_CAS_LIVE_SMALL_WALL_FILE_MANIFEST_V1", "files": files}


def execute(source_root: Path, output_root: Path, pilot_variant: str) -> dict[str, Any]:
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(not output_root.exists(), f"output root already exists: {output_root}")
    require(
        pilot_variant in {
            "pn", "np", "anchor-sham", "impulse", "step-sham",
            "phase-forward", "phase-reverse",
            "value-forward", "value-reverse", "value-equal",
            "occupancy-forward", "occupancy-reverse", "occupancy-equal",
            "readonly-occupancy-forward", "readonly-occupancy-reverse",
            "readonly-occupancy-equal",
            "coded-preprojection-loop",
            "coded-preprojection-restored-loop",
            "coded-preprojection-warm-restored-loop",
            "coded-preprojection-warm-query-scramble-loop",
            "coded-preprojection-warm-query-off-loop",
            "coded-preprojection-warm-declaration-sham-loop",
            "coded-preprojection-warm-phase-local-sham-loop",
            "coded-preprojection-warm-phase-local-loop",
            "coded-preprojection-active-query-loop",
            "coded-preprojection-source-phase-chop-loop",
        },
        "unknown pilot variant",
    )
    for name in SOURCE_NAMES:
        require((source_root / name).is_file(), f"source missing: {name}")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    runtime_root = output_root / "runtime"
    binary = source_root / "gate_a_worker_live"
    bundle_sha256, compile_command = compile_worker(source_root, binary, runtime_root)
    temp_path = temperature_path()
    preflight_processes = process_snapshot()
    require(not preflight_processes["forbidden_matches"], "forbidden process present before transaction")
    pre_temperature = read_temperature_c(temp_path)
    require(pre_temperature < TEMPERATURE_VETO_C, f"preflight temperature veto: {pre_temperature} C")

    snapshots: dict[int, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    monitor: list[dict[str, Any]] = []
    error: str | None = None
    restoration_complete = False
    restoration_errors: list[str] = []
    restoration_readback: dict[str, Any] = {}
    process_cleanup: dict[str, Any] | None = None
    writes_started = False
    child: subprocess.Popen[str] | None = None

    previous_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"live Gate A transaction interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    signal.alarm(TRANSACTION_TIMEOUT_S)
    try:
        snapshots = {
            core: frequency._snapshot_policy(Path("/sys"), core, read_bytes=read_sysfs)
            for core in CORES
        }
        require(
            snapshots[4]["identity"]["resolved_path"] != snapshots[5]["identity"]["resolved_path"],
            "policies 4 and 5 unexpectedly alias",
        )
        write_json(output_root / "POLICY_SNAPSHOT.json", {str(core): snapshots[core] for core in CORES})
        writes_started = True
        pin_policies(snapshots, ledger)
        pinned = observe_pinned_exact(snapshots)
        write_json(output_root / "PINNED_OBSERVATION.json", pinned)
        require(pinned["settled"], "pinned frequency never established a stable epoch")
        require(pinned["all_pairs_exact"], "pinned frequency drifted during exact observation")
        returncode, stdout, stderr, monitor, veto = run_worker_monitored(
            binary, runtime_root, bundle_sha256, temp_path, pilot_variant
        )
        runtime = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "monitor_veto": veto,
        }
        write_json(output_root / "RUNTIME_PROCESS.json", runtime)
        require(veto is None, veto or "runtime monitor veto")
        require(returncode == 0, f"Gate A worker failed: {stderr.strip()}")
    except Exception as exc:
        error = str(exc)
    finally:
        signal.alarm(0)
        if child is not None:
            stop_process(child)
        try:
            process_cleanup = process_snapshot()
            if process_cleanup["forbidden_matches"]:
                restoration_errors.append("forbidden process remained after runtime")
        except Exception as exc:
            restoration_errors.append(f"process cleanup unobservable: {exc}")
        if writes_started and snapshots:
            restored, errors, readback = restore_policies(snapshots, ledger)
            restoration_complete = restored
            restoration_errors.extend(errors)
            restoration_readback = readback
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    analysis: dict[str, Any] | None = None
    if (runtime_root / "LOCKIN_IQ.jsonl").is_file() and (runtime_root / "raw_samples.bin").is_file():
        try:
            analysis = analyze_runtime(runtime_root, pilot_variant)
            write_json(output_root / "FIRST_LIGHT_ANALYSIS.json", analysis)
        except Exception as exc:
            if error is None:
                error = f"analysis failed: {exc}"

    write_json(output_root / "FREQUENCY_WRITE_LEDGER.json", ledger)
    write_json(output_root / "RUNTIME_MONITOR.json", monitor)
    final = {
        "schema_id": "CAT_CAS_DIRECT_USER_AUTHORIZED_GATE_A_FIRST_LIGHT_V1",
        "status": "GATE_A_FIRST_LIGHT_COMPLETE" if error is None and restoration_complete else "GATE_A_FIRST_LIGHT_FAILED",
        "pilot_variant": pilot_variant,
        "source_bundle_sha256": bundle_sha256,
        "source_hashes": source_digest(source_root)[1],
        "user_directive_sha256": USER_DIRECTIVE_SHA256,
        "schedule_sha256": (
            coded_preprojection_schedule_sha256(pilot_variant)
            if pilot_variant in CODED_PREPROJECTION_VARIANTS
            else (READONLY_MICRO_SCHEDULE_SHA256 if pilot_variant in READONLY_VARIANTS else SCHEDULE_SHA256)
        ),
        "compile_command": compile_command,
        "preflight_temperature_c": pre_temperature,
        "temperature_path": str(temp_path),
        "preflight_processes": preflight_processes,
        "runtime": runtime,
        "process_cleanup": process_cleanup,
        "restoration_complete": restoration_complete,
        "restoration_errors": restoration_errors,
        "restoration_readback": restoration_readback,
        "frequency_write_attempt_count": len(ledger),
        "written_paths": sorted({entry["path"] for entry in ledger}),
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
        "analysis_available": analysis is not None,
        "engineering_first_light_candidate": bool(analysis and analysis["engineering_first_light_candidate"]),
        "error": error,
    }
    if restoration_errors and final["error"] is None:
        final["error"] = "; ".join(restoration_errors)
        final["status"] = "GATE_A_FIRST_LIGHT_FAILED"
    write_json(output_root / "FINAL_RESULT.json", final)
    write_json(output_root / "FILE_MANIFEST.json", build_file_manifest(output_root))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-timing-parser", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--pilot-variant",
        choices=(
            "pn", "np", "anchor-sham", "impulse", "step-sham",
            "phase-forward", "phase-reverse",
            "value-forward", "value-reverse", "value-equal",
            "occupancy-forward", "occupancy-reverse", "occupancy-equal",
            "readonly-occupancy-forward", "readonly-occupancy-reverse",
            "readonly-occupancy-equal",
            "coded-preprojection-loop",
            "coded-preprojection-restored-loop",
            "coded-preprojection-warm-restored-loop",
            "coded-preprojection-warm-query-scramble-loop",
            "coded-preprojection-warm-query-off-loop",
            "coded-preprojection-warm-declaration-sham-loop",
            "coded-preprojection-warm-phase-local-sham-loop",
            "coded-preprojection-warm-phase-local-loop",
            "coded-preprojection-active-query-loop",
            "coded-preprojection-source-phase-chop-loop",
        ),
        default="pn",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test_timing_parser:
        try:
            return self_test_timing_parser()
        except Exception as exc:
            print(f"live_gate_a_target: {exc}", file=sys.stderr)
            return 1
    if args.source_root is None or args.output_root is None:
        print("live_gate_a_target: --source-root and --output-root are required", file=sys.stderr)
        return 2
    try:
        result = execute(args.source_root.resolve(), args.output_root.resolve(), args.pilot_variant)
    except Exception as exc:
        print(f"live_gate_a_target: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "GATE_A_FIRST_LIGHT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
