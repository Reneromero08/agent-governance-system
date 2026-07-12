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
USER_DIRECTIVE_SHA256 = hashlib.sha256(
    b"CAT_CAS_EXPLICIT_USER_LIVE_AUTHORIZATION__SMALL_WALL_GOAL__2026-07-11"
).hexdigest()
OFF_TOKENS = frozenset({"I", "C0", "D0", "O0", "T"})
SOURCE_NAMES = (
    "live_gate_a_target.py",
    "gate_a_frequency_preparation.py",
    "gate_a_worker.c",
    "gate_a_engineering_smoke_runtime.c",
    "gate_a_engineering_smoke_runtime.h",
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
        "-pthread",
        "-I",
        str(source_root),
        str(source_root / "gate_a_worker.c"),
        str(source_root / "gate_a_engineering_smoke_runtime.c"),
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
    command = [
        str(binary),
        "--execute-authorized",
        "--authority-sha256",
        USER_DIRECTIVE_SHA256,
        "--schedule-sha256",
        SCHEDULE_SHA256,
        "--execution-bundle-sha256",
        bundle_sha256,
        "--output-root",
        str(runtime_output),
        "--sender-core",
        "4",
        "--receiver-core",
        "5",
        "--read-hz",
        "8000",
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
            elif pilot_variant not in {"step-sham"} and 3.15 <= elapsed <= (
                3.35 if pilot_variant == "impulse" else 4.85
            ) and frequencies["4"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active step-sender frequency drift during runtime: {frequencies}"
            elif 6.15 <= elapsed <= 6.85 and frequencies["4"] != REQUIRED_FREQUENCY_KHZ:
                veto = f"active anchor-sender frequency drift during runtime: {frequencies}"
            elif elapsed > WORKER_TIMEOUT_S:
                veto = "worker timeout"
            receiver_exact_seen = receiver_exact_seen or (
                elapsed >= 0.5 and frequencies["5"] == REQUIRED_FREQUENCY_KHZ
            )
            sender_step_exact_seen = sender_step_exact_seen or (
                3.15 <= elapsed <= (3.35 if pilot_variant == "impulse" else 4.85)
                and frequencies["4"] == REQUIRED_FREQUENCY_KHZ
            )
            sender_anchor_exact_seen = sender_anchor_exact_seen or (
                6.15 <= elapsed <= 6.85 and frequencies["4"] == REQUIRED_FREQUENCY_KHZ
            )
            if veto is not None:
                stop_process(process)
                break
            time.sleep(MONITOR_INTERVAL_S)
        stdout, stderr = process.communicate(timeout=3)
    finally:
        stop_process(process)
    anchor_complete = pilot_variant == "anchor-sham" or sender_anchor_exact_seen
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


def analyze_runtime(runtime_root: Path, pilot_variant: str) -> dict[str, Any]:
    lockin = [json.loads(line) for line in (runtime_root / "LOCKIN_IQ.jsonl").read_text().splitlines() if line]
    require(len(lockin) == 16, "expected 16 lock-in records")
    measurement_modes = {str(row.get("measurement_mode", "ring_period_cycles")) for row in lockin}
    require(len(measurement_modes) == 1, "measurement mode drift")
    measurement_mode = measurement_modes.pop()
    raw_bytes = (runtime_root / "raw_samples.bin").read_bytes()
    require(len(raw_bytes) % 16 == 0, "raw sample file is not packed Qd records")
    raw = [record for record in struct.iter_unpack("<Qd", raw_bytes)]
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
        }[pilot_variant],
        "groups": summaries,
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
        "schedule_sha256": SCHEDULE_SHA256,
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pilot-variant",
        choices=(
            "pn", "np", "anchor-sham", "impulse", "step-sham",
            "phase-forward", "phase-reverse",
            "value-forward", "value-reverse", "value-equal",
            "occupancy-forward", "occupancy-reverse", "occupancy-equal",
        ),
        default="pn",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(args.source_root.resolve(), args.output_root.resolve(), args.pilot_variant)
    except Exception as exc:
        print(f"live_gate_a_target: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "GATE_A_FIRST_LIGHT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
