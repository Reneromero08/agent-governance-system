#!/usr/bin/env python3
"""Deploy, run, copy back, verify, and clean one Gate A lab-device slice."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parents[1]
FREQUENCY = FRONTIER / "phase6b6" / "acquisition" / "gate_a" / "frequency_preparation"
RUNTIME = FRONTIER / "holo_runtime_v2"
TARGET = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
FILES = {
    HERE / "live_gate_a_target.py": "live_gate_a_target.py",
    FREQUENCY / "gate_a_frequency_preparation.py": "gate_a_frequency_preparation.py",
    HERE / "small_wall_worker.c": "small_wall_worker.c",
    HERE / "small_wall_runtime.c": "small_wall_runtime.c",
    HERE / "small_wall_runtime.h": "small_wall_runtime.h",
    RUNTIME / "combined_pdn_hardware.c": "combined_pdn_hardware.c",
    RUNTIME / "combined_pdn_hardware.h": "combined_pdn_hardware.h",
    RUNTIME / "capture_quality_contract.h": "capture_quality_contract.h",
    RUNTIME / "captured_file.c": "captured_file.c",
    RUNTIME / "captured_file.h": "captured_file.h",
}


class ControllerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ControllerError(message)


def run(command: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise ControllerError(
            f"command failed ({completed.returncode}): {command!r}\n{completed.stderr.strip()}"
        )
    return completed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_copy(local_root: Path) -> dict[str, Any]:
    manifest = json.loads((local_root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema_id"] == "CAT_CAS_LIVE_SMALL_WALL_FILE_MANIFEST_V1", "manifest schema mismatch")
    for entry in manifest["files"]:
        path = local_root / entry["path"]
        require(path.is_file(), f"copied file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"], f"copied size mismatch: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"copied digest mismatch: {entry['path']}")
    return manifest


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("gate_a_first_light_%Y%m%dT%H%M%SZ")


def execute(run_id: str, *, pilot_variant: str, keep_remote: bool) -> dict[str, Any]:
    require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
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
        "pilot variant is not closed",
    )
    for source in FILES:
        require(source.is_file(), f"local source missing: {source}")
    remote_run = f"{REMOTE_BASE}/{run_id}"
    remote_source = f"{remote_run}/source"
    remote_output = f"{remote_run}/output"
    local_run = HERE / "runs" / run_id
    require(not local_run.exists(), f"local run already exists: {local_run}")
    local_run.mkdir(mode=0o700, parents=True, exist_ok=False)

    preflight = (
        f"set -eu; test ! -e {shlex.quote(remote_run)}; "
        f"install -d -m 700 -- {shlex.quote(remote_source)}"
    )
    run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET, preflight], timeout=15)
    for source, remote_name in FILES.items():
        run(["scp", "-q", str(source), f"{TARGET}:{remote_source}/{remote_name}"], timeout=30)

    remote_command = (
        f"timeout --signal=TERM --kill-after=5s 60s python3 "
        f"{shlex.quote(remote_source + '/live_gate_a_target.py')} "
        f"--source-root {shlex.quote(remote_source)} "
        f"--output-root {shlex.quote(remote_output)} "
        f"--pilot-variant {shlex.quote(pilot_variant)}"
    )
    completed = run(["ssh", "-o", "BatchMode=yes", TARGET, remote_command], timeout=75, check=False)
    (local_run / "CONTROLLER_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (local_run / "CONTROLLER_STDERR.txt").write_text(completed.stderr, encoding="utf-8")

    copied = run(
        ["scp", "-q", "-r", f"{TARGET}:{remote_output}/.", str(local_run)],
        timeout=45,
        check=False,
    )
    require(copied.returncode == 0, f"copy-back failed; remote retained at {remote_run}: {copied.stderr.strip()}")
    manifest = verify_copy(local_run)
    final = json.loads((local_run / "FINAL_RESULT.json").read_text(encoding="utf-8"))

    cleaned = False
    if not keep_remote:
        require(remote_run.startswith(REMOTE_BASE + "/") and remote_run != REMOTE_BASE, "unsafe cleanup root")
        cleanup = run(
            ["ssh", "-o", "BatchMode=yes", TARGET, f"rm -rf -- {shlex.quote(remote_run)}"],
            timeout=20,
            check=False,
        )
        require(cleanup.returncode == 0, f"verified copy retained but remote cleanup failed: {cleanup.stderr.strip()}")
        absent = run(
            ["ssh", "-o", "BatchMode=yes", TARGET, f"test ! -e {shlex.quote(remote_run)}"],
            timeout=15,
            check=False,
        )
        require(absent.returncode == 0, "remote run root remained after cleanup")
        cleaned = True

    controller = {
        "schema_id": "CAT_CAS_GATE_A_FIRST_LIGHT_CONTROLLER_V1",
        "run_id": run_id,
        "target": TARGET,
        "pilot_variant": pilot_variant,
        "remote_run": remote_run,
        "local_run": str(local_run),
        "remote_returncode": completed.returncode,
        "verified_file_count": len(manifest["files"]),
        "copy_verified": True,
        "remote_cleaned": cleaned,
        "final_status": final["status"],
        "restoration_complete": final["restoration_complete"],
        "engineering_first_light_candidate": final["engineering_first_light_candidate"],
    }
    (local_run / "CONTROLLER_RESULT.json").write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(controller, sort_keys=True))
    return controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=default_run_id())
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
    parser.add_argument("--keep-remote", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(args.run_id, pilot_variant=args.pilot_variant, keep_remote=args.keep_remote)
    except (ControllerError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"run_gate_a_first_light: {exc}", file=sys.stderr)
        return 1
    return 0 if result["final_status"] == "GATE_A_FIRST_LIGHT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
