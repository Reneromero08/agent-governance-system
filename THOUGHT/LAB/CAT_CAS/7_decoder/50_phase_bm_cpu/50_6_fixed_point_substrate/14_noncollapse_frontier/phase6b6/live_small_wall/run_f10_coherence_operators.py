#!/usr/bin/env python3
"""Deploy, run, copy back, verify, and clean one Family 10h coherence-operator probe."""

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
LAB_DEVICE = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
FILES = {
    HERE / "f10_pmc_first_light_target.py": "f10_pmc_first_light_target.py",
    HERE / "f10_pmc_first_light_worker.c": "f10_pmc_first_light_worker.c",
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
    require(manifest["schema_id"] == "CAT_CAS_F10_PMC_FIRST_LIGHT_FILE_MANIFEST_V1", "manifest schema mismatch")
    for entry in manifest["files"]:
        path = local_root / entry["path"]
        require(path.is_file(), f"copied file missing: {entry['path']}")
        require(path.stat().st_size == entry["size"], f"copied size mismatch: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"copied digest mismatch: {entry['path']}")
    return manifest


def default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("f10_coherence_ops_%Y%m%dT%H%M%SZ")


def execute(run_id: str, *, mode: str, keep_remote: bool) -> dict[str, Any]:
    require(re.fullmatch(r"[a-z0-9_]{8,80}", run_id) is not None, "run ID is not closed")
    require(
        mode in {
            "coherence-operators",
            "coherence-operators-route45",
            "coherence-operators-route23",
            "phase-local-pmu",
            "ibs-first-light",
            "wc-flush-order",
            "eviction-sentinel",
            "eviction-phase-local",
            "eviction-phase-bracketed",
            "eviction-phase-bracketed-c2d",
            "eviction-phase-bracketed-duration",
            "history-sentinel",
            "branch-history",
            "indirect-target-history",
            "translation-history",
            "store-load-alias-history",
            "prefetch-stream",
            "process-lifecycle",
        },
        f"unsupported coherence mode: {mode}",
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
    run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", LAB_DEVICE, preflight], timeout=15)
    for source, remote_name in FILES.items():
        run(["scp", "-q", str(source), f"{LAB_DEVICE}:{remote_source}/{remote_name}"], timeout=30)

    remote_command = (
        f"timeout --signal=TERM --kill-after=5s 45s python3 "
        f"{shlex.quote(remote_source + '/f10_pmc_first_light_target.py')} "
        f"--source-root {shlex.quote(remote_source)} "
        f"--output-root {shlex.quote(remote_output)} "
        f"--mode {shlex.quote(mode)}"
    )
    completed = run(["ssh", "-o", "BatchMode=yes", LAB_DEVICE, remote_command], timeout=60, check=False)
    (local_run / "CONTROLLER_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (local_run / "CONTROLLER_STDERR.txt").write_text(completed.stderr, encoding="utf-8")

    copied = run(
        ["scp", "-q", "-r", f"{LAB_DEVICE}:{remote_output}/.", str(local_run)],
        timeout=30,
        check=False,
    )
    require(copied.returncode == 0, f"copy-back failed; remote retained at {remote_run}: {copied.stderr.strip()}")
    manifest = verify_copy(local_run)
    final = json.loads((local_run / "FINAL_RESULT.json").read_text(encoding="utf-8"))

    cleaned = False
    if not keep_remote:
        require(remote_run.startswith(REMOTE_BASE + "/") and remote_run != REMOTE_BASE, "unsafe cleanup root")
        cleanup = run(
            ["ssh", "-o", "BatchMode=yes", LAB_DEVICE, f"rm -rf -- {shlex.quote(remote_run)}"],
            timeout=20,
            check=False,
        )
        require(cleanup.returncode == 0, f"verified copy retained but remote cleanup failed: {cleanup.stderr.strip()}")
        absent = run(
            ["ssh", "-o", "BatchMode=yes", LAB_DEVICE, f"test ! -e {shlex.quote(remote_run)}"],
            timeout=15,
            check=False,
        )
        require(absent.returncode == 0, "remote run root remained after cleanup")
        cleaned = True

    worker_result_files = {
        "phase-local-pmu": "F10_PHASE_LOCAL_PMU_RESULT.json",
        "ibs-first-light": "F10_IBS_FIRST_LIGHT_RESULT.json",
        "wc-flush-order": "F10_WC_FLUSH_ORDER_RESULT.json",
        "eviction-sentinel": "F10_EVICTION_SENTINEL_RESULT.json",
        "eviction-phase-local": "F10_EVICTION_PHASE_LOCAL_RESULT.json",
        "eviction-phase-bracketed": "F10_EVICTION_PHASE_BRACKETED_RESULT.json",
        "eviction-phase-bracketed-c2d": "F10_EVICTION_PHASE_BRACKETED_C2D_RESULT.json",
        "eviction-phase-bracketed-duration": "F10_EVICTION_PHASE_BRACKETED_DURATION_RESULT.json",
        "history-sentinel": "F10_HISTORY_SENTINEL_RESULT.json",
        "branch-history": "F10_BRANCH_HISTORY_RESULT.json",
        "indirect-target-history": "F10_INDIRECT_TARGET_HISTORY_RESULT.json",
        "translation-history": "F10_TRANSLATION_HISTORY_RESULT.json",
        "store-load-alias-history": "F10_STORE_LOAD_ALIAS_HISTORY_RESULT.json",
        "prefetch-stream": "F10_PREFETCH_STREAM_RESULT.json",
        "process-lifecycle": "F10_PROCESS_LIFECYCLE_RESULT.json",
    }
    worker_result_file = worker_result_files.get(mode, "F10_COHERENCE_OPERATOR_RESULT.json")
    worker = json.loads((local_run / worker_result_file).read_text(encoding="utf-8"))
    controller = {
        "schema_id": "CAT_CAS_F10_COHERENCE_OPERATOR_CONTROLLER_V1",
        "run_id": run_id,
        "mode": mode,
        "lab_device": LAB_DEVICE,
        "remote_run": remote_run,
        "local_run": str(local_run),
        "remote_returncode": completed.returncode,
        "verified_file_count": len(manifest["files"]),
        "copy_verified": True,
        "remote_cleaned": cleaned,
        "target_status": final["status"],
        "worker_status": worker["status"],
        "selected_group": worker.get("selected_group", mode),
        "worker_result_file": worker_result_file,
    }
    if mode == "phase-local-pmu":
        controller["phase_local_pmu_captured"] = bool(worker["acceptance"]["phase_local_pmu_captured"])
    elif mode == "ibs-first-light":
        controller["ibs_first_light_available"] = bool(worker["acceptance"]["ibs_first_light_available"])
        controller["ibs_workload_response"] = bool(worker["acceptance"]["any_workload_response"])
    elif mode == "wc-flush-order":
        controller["wc_flush_order_response"] = bool(worker["acceptance"]["wc_flush_order_response"])
    elif mode == "eviction-sentinel":
        controller["eviction_sentinel_response"] = bool(worker["acceptance"]["eviction_sentinel_response"])
    elif mode == "eviction-phase-local":
        controller["eviction_phase_local_captured"] = bool(worker["acceptance"]["eviction_phase_local_captured"])
    elif mode == "eviction-phase-bracketed":
        controller["eviction_phase_bracketed_captured"] = bool(worker["acceptance"]["eviction_phase_bracketed_captured"])
    elif mode == "eviction-phase-bracketed-c2d":
        controller["eviction_phase_bracketed_captured"] = bool(worker["acceptance"]["eviction_phase_bracketed_captured"])
    elif mode == "eviction-phase-bracketed-duration":
        controller["eviction_phase_bracketed_captured"] = bool(worker["acceptance"]["eviction_phase_bracketed_captured"])
        controller["duration_phase_local_signal"] = bool(worker["acceptance"]["duration_phase_local_signal"])
    elif mode == "history-sentinel":
        controller["history_sentinel_response"] = bool(worker["acceptance"]["history_sentinel_response"])
    elif mode == "branch-history":
        controller["branch_history_response"] = bool(worker["acceptance"]["branch_history_response"])
    elif mode == "indirect-target-history":
        controller["indirect_target_history_response"] = bool(worker["acceptance"]["indirect_target_history_response"])
    elif mode == "translation-history":
        controller["translation_history_response"] = bool(worker["acceptance"]["translation_history_response"])
    elif mode == "store-load-alias-history":
        controller["store_load_alias_history_response"] = bool(worker["acceptance"]["store_load_alias_history_response"])
    elif mode == "prefetch-stream":
        controller["prefetch_stream_response"] = bool(worker["acceptance"]["prefetch_stream_response"])
    elif mode == "process-lifecycle":
        controller["process_lifecycle_response"] = bool(worker["acceptance"]["process_lifecycle_response"])
    else:
        controller["controlled_state_found"] = bool(worker["acceptance"]["controlled_state_found"])
    (local_run / "CONTROLLER_RESULT.json").write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(controller, sort_keys=True))
    return controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument(
        "--mode",
        choices=("coherence-operators", "coherence-operators-route45", "coherence-operators-route23", "phase-local-pmu", "ibs-first-light", "wc-flush-order", "eviction-sentinel", "eviction-phase-local", "eviction-phase-bracketed", "eviction-phase-bracketed-c2d", "eviction-phase-bracketed-duration", "history-sentinel", "branch-history", "indirect-target-history", "translation-history", "store-load-alias-history", "prefetch-stream", "process-lifecycle"),
        default="coherence-operators",
    )
    parser.add_argument("--keep-remote", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = execute(args.run_id, mode=args.mode, keep_remote=args.keep_remote)
    except (ControllerError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"run_f10_coherence_operators: {exc}", file=sys.stderr)
        return 1
    return 0 if str(result["target_status"]).endswith("_TARGET_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
