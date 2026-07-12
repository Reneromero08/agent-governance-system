#!/usr/bin/env python3
"""Deploy, run, copy back, verify, and clean one OrbitState query first-light."""

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
TARGET = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
FILES = {
    HERE / "orbit_query_target.py": "orbit_query_target.py",
    HERE / "orbit_query_public.py": "orbit_query_public.py",
    HERE / "orbit_query_model.py": "orbit_query_model.py",
    HERE / "orbit_query_runtime.c": "orbit_query_runtime.c",
    HERE / "orbit_query_runtime.h": "orbit_query_runtime.h",
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
    require(manifest["schema_id"] == "CAT_CAS_ORBIT_QUERY_FILE_MANIFEST_V1", "manifest schema mismatch")
    for entry in manifest["files"]:
        path = local_root / entry["path"]
        require(path.is_file(), f"copied file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"], f"copied size mismatch: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"copied digest mismatch: {entry['path']}")
    return manifest


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("orbit_query_first_light_%Y%m%dT%H%M%SZ")


def execute(run_id: str, *, keep_remote: bool) -> dict[str, Any]:
    require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
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
        f"timeout --signal=TERM --kill-after=5s 210s python3 "
        f"{shlex.quote(remote_source + '/orbit_query_target.py')} "
        f"--source-root {shlex.quote(remote_source)} "
        f"--output-root {shlex.quote(remote_output)}"
    )
    completed = run(["ssh", "-o", "BatchMode=yes", TARGET, remote_command], timeout=240, check=False)
    (local_run / "CONTROLLER_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (local_run / "CONTROLLER_STDERR.txt").write_text(completed.stderr, encoding="utf-8")

    copied = run(
        ["scp", "-q", "-r", f"{TARGET}:{remote_output}/.", str(local_run)],
        timeout=60,
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

    adjudication = (
        json.loads((local_run / "ADJUDICATION.json").read_text(encoding="utf-8"))
        if (local_run / "ADJUDICATION.json").is_file()
        else None
    )
    controller = {
        "schema_id": "CAT_CAS_ORBIT_QUERY_FIRST_LIGHT_CONTROLLER_V1",
        "run_id": run_id,
        "target": TARGET,
        "remote_run": remote_run,
        "local_run": str(local_run),
        "remote_returncode": completed.returncode,
        "verified_file_count": len(manifest["files"]),
        "copy_verified": True,
        "remote_cleaned": cleaned,
        "final_status": final["status"],
        "failure": final.get("failure"),
        "adjudication_status": None if adjudication is None else adjudication["status"],
        "features_frozen_sha256": final.get("features_frozen_sha256"),
    }
    (local_run / "CONTROLLER_RESULT.json").write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(controller, sort_keys=True))
    return controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--keep-remote", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(args.run_id, keep_remote=args.keep_remote)
    except (ControllerError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"run_orbit_query_first_light: {exc}", file=sys.stderr)
        return 1
    return 0 if result["final_status"] == "ORBIT_QUERY_TARGET_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
