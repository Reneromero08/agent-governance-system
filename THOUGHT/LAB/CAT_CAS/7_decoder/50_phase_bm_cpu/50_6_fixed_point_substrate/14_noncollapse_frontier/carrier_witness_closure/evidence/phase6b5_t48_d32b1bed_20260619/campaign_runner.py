#!/usr/bin/env python3
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CAMPAIGN = Path("/root/catcas_evidence/phase6b5_t48_d32b1bed_20260619")
SRC = Path("/root/carrier_witness_d32b1bed/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate")
CLOSURE = SRC / "14_noncollapse_frontier" / "carrier_witness_closure"
SLOT = SRC / "10_cross_core_wormhole" / "slot2_pdn"
BINARY = CLOSURE / "slot2"
ANALYZER = SLOT / "slot2_pdn_analyze.py"
FINALIZER = CLOSURE / "carrier_witness_finalize.py"
MANIFEST_TOOL = CLOSURE / "carrier_witness_manifest.py"
VALIDATOR_PATH = CLOSURE / "carrier_witness_validate.py"
THERMAL = Path("/sys/class/hwmon/hwmon0/temp1_input")
SOURCE_COMMIT = "d32b1bed0deae1b907a07eeed018b924244e9ea2"

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def temperature():
    raw = THERMAL.read_text().strip()
    value = int(raw) / 1000.0
    if not math.isfinite(value) or value < -100.0 or value >= 68.0:
        raise RuntimeError(f"invalid/veto thermal value {value}")
    return value

def verify_frozen(source_manifest, campaign):
    if socket.gethostname() != "catcas":
        raise RuntimeError("host identity changed")
    flags = set(Path("/proc/cpuinfo").read_text().split())
    if "constant_tsc" not in flags or "nonstop_tsc" not in flags:
        raise RuntimeError("TSC flags missing")
    cmdline = Path("/proc/cmdline").read_text()
    if "isolcpus=2,3,4,5" not in cmdline:
        raise RuntimeError("isolcpus changed")
    if sha(BINARY) != source_manifest["binary_sha256"]:
        raise RuntimeError("binary hash changed")
    for relative, expected in source_manifest["source_files"].items():
        if sha(SRC / relative) != expected:
            raise RuntimeError(f"source hash changed: {relative}")
    if sha(CAMPAIGN / "campaign_runner.py") != source_manifest["runner_sha256"]:
        raise RuntimeError("campaign runner hash changed")
    if shutil.disk_usage(CAMPAIGN).free < 5_000_000_000:
        raise RuntimeError("less than 5 GB free")
    if campaign["status"] != "FROZEN":
        raise RuntimeError("campaign is not frozen")
    temperature()

def build_metadata(run_dir, item, schedule, source_manifest, started, ended):
    with (run_dir / "windows.csv").open(newline="") as handle:
        windows = list(csv.DictReader(handle))
    temps = [float(row[key]) for row in windows for key in ("temp_before_c", "temp_after_c")]
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    frozen = json.loads((CAMPAIGN / "campaign.json").read_text())["frozen_configuration"]
    return {
        "schema_id": "CAT_CAS_PDN_CARRIER_RUN_V1",
        "schema_version": "1.0.0",
        "campaign_id": CAMPAIGN.name,
        "run_id": item["run_id"],
        "condition": item["condition"],
        "route": {"victim": item["victim"], "sender": item["sender"], "label": item["route"]},
        "seed": item["seed"],
        "utc_start": started,
        "utc_end": ended,
        "source_commit": SOURCE_COMMIT,
        "source_files": source_manifest["source_files"],
        "binary_sha256": source_manifest["binary_sha256"],
        "compiler": source_manifest["compiler"],
        "host": {
            "hostname": socket.gethostname(), "cpu": cpu, "kernel": platform.release(),
            "isolcpus": "2,3,4,5", "cmdline": Path("/proc/cmdline").read_text().strip(),
            "constant_tsc": True, "nonstop_tsc": True,
        },
        "timing": {
            "t0_tsc": schedule["t0_tsc"], "tsc_hz": frozen["tsc_hz"],
            "slot_s": frozen["slot_s"], "gap_s": frozen["gap_s"], "read_hz": frozen["read_hz"],
        },
        "drive": {
            "nbin": frozen["nbin"], "f_lo": frozen["f_lo"], "f_hi": frozen["f_hi"],
            "tones_hz": schedule["tones_hz"], "trials_per_family": frozen["trials_per_family"],
            "pstate_target_khz": frozen["pstate_target_khz"],
        },
        "thermal": {
            "source": str(THERMAL), "sensor_name": "k10temp", "veto_c": 68.0,
            "minimum_c": min(temps), "maximum_c": max(temps),
        },
        "files": {
            "schedule": "schedule.json", "windows": "windows.csv", "raw_samples": "raw_samples.bin",
            "summary": "summary.csv", "analysis": "analysis.json", "stdout": "stdout.log",
            "stderr": "stderr.log", "manifest": "run_manifest.json",
        },
        "exit": {
            "sender": 0, "receiver": 0, "orchestrator": 0, "pstate_restored": True,
            "affinity_verified": True, "temperature_veto_triggered": False,
        },
    }

def reconstruct(run_dir):
    spec = importlib.util.spec_from_file_location("carrier_witness_validate", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="cw-run-") as temp:
        return module.reconstruct_run(run_dir, ANALYZER, Path(temp))

def cooldown():
    time.sleep(30)
    deadline = time.monotonic() + 600
    while temperature() > 43.0:
        if time.monotonic() >= deadline:
            raise RuntimeError("cooldown threshold not reached")
        time.sleep(30)

def main():
    campaign = json.loads((CAMPAIGN / "campaign.json").read_text())
    source_manifest = json.loads((CAMPAIGN / "source_manifest.json").read_text())
    plan = json.loads((CAMPAIGN / "run_plan.json").read_text())
    ledger = {"schema_id": "CAT_CAS_PDN_CARRIER_EXECUTION_LEDGER_V1",
              "campaign_id": CAMPAIGN.name, "runs": []}
    atomic_json(CAMPAIGN / "execution_ledger.json", ledger)
    for index, item in enumerate(plan):
        verify_frozen(source_manifest, campaign)
        run_dir = CAMPAIGN / "runs" / item["run_id"]
        if run_dir.exists():
            raise RuntimeError(f"immutable run path already exists: {run_dir}")
        started = datetime.now(timezone.utc).isoformat()
        command = [
            str(BINARY), "--mode", "matrix", "--victim", str(item["victim"]),
            "--sender", str(item["sender"]), "--seed", str(item["seed"]), "--trials", "48",
            "--slot-s", "0.5", "--gap-s", "0.12", "--read-hz", "4000",
            "--tsc-hz", "3214826000", "--pin-khz", "1600000", "--temp-veto", "68",
            "--condition", item["condition"], "--witness-dir", str(run_dir),
            "--run-id", item["run_id"], "--campaign-id", CAMPAIGN.name,
        ]
        process = subprocess.run(command, check=False)
        ended = datetime.now(timezone.utc).isoformat()
        if process.returncode != 0:
            ledger["runs"].append({**item, "exit_status": process.returncode, "valid": False,
                                   "started": started, "ended": ended})
            atomic_json(CAMPAIGN / "execution_ledger.json", ledger)
            raise RuntimeError(f"acquisition failed {item['run_id']} rc={process.returncode}")
        schedule = json.loads((run_dir / "schedule.json").read_text())
        metadata = build_metadata(run_dir, item, schedule, source_manifest, started, ended)
        metadata_path = CAMPAIGN / "metadata" / f"{item['run_id']}.json"
        atomic_json(metadata_path, metadata)
        finalize = subprocess.run([
            sys.executable, str(FINALIZER), str(run_dir), "--metadata", str(metadata_path),
            "--analyzer", str(ANALYZER), "--source-commit", SOURCE_COMMIT,
        ], check=False)
        if finalize.returncode != 0:
            raise RuntimeError(f"finalization failed {item['run_id']} rc={finalize.returncode}")
        verify = subprocess.run([sys.executable, str(MANIFEST_TOOL), "verify",
                                 str(run_dir / "run_manifest.json")], check=False)
        if verify.returncode != 0:
            raise RuntimeError(f"manifest failed {item['run_id']}")
        result = reconstruct(run_dir)
        if not result["valid"]:
            raise RuntimeError(f"reconstruction failed {item['run_id']}: {result['errors']}")
        analysis = json.loads((run_dir / "analysis.json").read_text())
        ledger["runs"].append({
            **item, "exit_status": 0, "valid": True, "started": started, "ended": ended,
            "raw_size": (run_dir / "raw_samples.bin").stat().st_size,
            "raw_sha256": sha(run_dir / "raw_samples.bin"),
            "run_manifest_sha256": sha(run_dir / "run_manifest.json"),
            "raw_records": result["raw_records"], "windows": result["windows"],
            "max_abs_reconstruction_error": result["max_abs_reconstruction_error"],
            "max_rel_reconstruction_error": result["max_rel_reconstruction_error"],
            "scientific_pass": result["scientific_pass"],
            "metrics": {key: analysis[key] for key in (
                "real_accuracy", "real_mode_floor", "real_vs_pseudo_floor",
                "pseudo_reject_floor", "pseudo_declared_match", "wrong_actual_match",
                "wrong_declared_match", "phase_corr_true", "phase_corr_null", "phase_delta")},
            "temperature_min_c": metadata["thermal"]["minimum_c"],
            "temperature_max_c": metadata["thermal"]["maximum_c"],
        })
        atomic_json(CAMPAIGN / "execution_ledger.json", ledger)
        print(f"RUN_COMPLETE {index+1}/{len(plan)} {item['run_id']} scientific={result['scientific_pass']}", flush=True)
        cooldown()
    report_path = CAMPAIGN / "aggregate" / "closure_report.json"
    validator = subprocess.run([
        sys.executable, str(VALIDATOR_PATH), str(CAMPAIGN), "--analyzer", str(ANALYZER),
        "--output", str(report_path),
    ], check=False)
    if validator.returncode not in (0, 1):
        raise RuntimeError(f"campaign validator failed rc={validator.returncode}")
    report = json.loads(report_path.read_text())
    atomic_json(CAMPAIGN / "aggregate" / "aggregate.json", {
        "schema_id": "CAT_CAS_PDN_CARRIER_AGGREGATE_V1",
        "campaign_id": CAMPAIGN.name, "status": report["status"],
        "structural_valid": report["structural_valid"], "runs": report["runs"],
    })
    subprocess.run([sys.executable, str(MANIFEST_TOOL), "campaign", str(CAMPAIGN)], check=True)
    subprocess.run([sys.executable, str(MANIFEST_TOOL), "verify",
                    str(CAMPAIGN / "campaign_manifest.json")], check=True)
    print(f"CAMPAIGN_COMPLETE status={report['status']}", flush=True)

if __name__ == "__main__":
    main()
