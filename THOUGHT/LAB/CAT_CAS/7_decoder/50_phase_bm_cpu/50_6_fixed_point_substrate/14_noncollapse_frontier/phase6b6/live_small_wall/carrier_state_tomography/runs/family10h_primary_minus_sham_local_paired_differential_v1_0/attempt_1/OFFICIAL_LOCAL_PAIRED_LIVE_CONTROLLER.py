#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any


TARGET_HOST = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
TRANSACTION_RUN_ID = "family10h_primary_minus_sham_local_paired_differential_v1_0"
SEGMENTED_TRANSACTION_RUN_ID = "family10h_relation_spatial_pair_readout_v1_1_segmented_0"
SEGMENTED_SOURCE_AUTHORITY_COMMIT = "2cdec9704452669d3651a063b6fb5805913647e3"
SEGMENTED_FREEZE_COMMIT = "8720dd543947f26cbd2af18a6a3dd4870c9adf85"
RUNTIME_AUTHORITY_VALUE = "family10h_relation_spatial_pair_readout_v1_0"
REMOTE_ROOT = f"{REMOTE_BASE}/{TRANSACTION_RUN_ID}"
REMOTE_PACKAGE = f"{REMOTE_BASE}/{TRANSACTION_RUN_ID}_source_package.tar.gz"
REMOTE_ARCHIVE = f"{REMOTE_BASE}/{TRANSACTION_RUN_ID}_attempt1_remote_root.tar.gz"
OWNER_MARKER = f".{TRANSACTION_RUN_ID}_owner"
VARIANTS = [
    "round0_query_relation_pair",
    "round0_relation_sham",
    "round0_scrambled_pair_control",
    "round0_distance_matched_control",
    "round0_route_pressure_control",
    "round1_query_relation_pair",
    "round1_relation_sham",
    "round1_scrambled_pair_control",
    "round1_distance_matched_control",
    "round1_route_pressure_control",
]
EXPECTED_RAW = 2048
EXPECTED_PAIRS = 2048 * 256


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


ATTEMPT_DIR = Path(__file__).resolve().parent
CARRIER_ROOT = ATTEMPT_DIR.parent.parent.parent
PACKAGE_ROOT = CARRIER_ROOT / "family10h_primary_minus_sham_local_paired_differential_v1"
SEGMENTED_ROOT = CARRIER_ROOT / "family10h_relation_spatial_pair_readout_v1_1_segmented"
LOCAL_TMP_PACKAGE = Path("C:/tmp") / f"{TRANSACTION_RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{TRANSACTION_RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = ATTEMPT_DIR / "OFFICIAL_TARGET_ROOT.tar.gz"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_local(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_remote(script: str, *, timeout: int = 120) -> dict[str, Any]:
    return run_local(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET_HOST, script], timeout=timeout)


def run_scp(source: str, dest: str, *, timeout: int = 120) -> dict[str, Any]:
    return run_local(["scp", "-O", "-o", "BatchMode=yes", source, dest], timeout=timeout)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_value(args: list[str]) -> str:
    completed = run_local(["git", *args], cwd=CARRIER_ROOT, timeout=30)
    require(completed["returncode"] == 0, f"git {' '.join(args)} failed: {completed['stderr']}")
    return completed["stdout"].strip()


def local_identity() -> dict[str, Any]:
    package_manifest = json.loads((PACKAGE_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    schedule_manifest = json.loads((PACKAGE_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULE_MANIFEST.json").read_text(encoding="utf-8"))
    threshold_contract = json.loads((PACKAGE_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_THRESHOLD_CONTRACT.json").read_text(encoding="utf-8"))
    head = git_value(["rev-parse", "HEAD"])
    origin = git_value(["rev-parse", "origin/codex/family10h-tomography-repair"])
    return {
        "schema": "FAMILY10H_LOCAL_PAIRED_CONTROLLER_LOCAL_IDENTITY_V1",
        "created_at": utc_now(),
        "branch": git_value(["branch", "--show-current"]),
        "head": head,
        "origin_head": origin,
        "head_equals_origin": head == origin,
        "status_short": run_local(["git", "status", "--short", "--untracked-files=all"], cwd=CARRIER_ROOT, timeout=30)["stdout"],
        "package_manifest_sha256": sha256_file(PACKAGE_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_MANIFEST.json"),
        "package_manifest_canonical_sha256": package_manifest["package_manifest_sha256"],
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "threshold_contract_sha256": threshold_contract["threshold_contract_sha256"],
        "package_decision": package_manifest["package_decision"],
        "transaction_run_id": TRANSACTION_RUN_ID,
        "segmented_source_authority_commit": SEGMENTED_SOURCE_AUTHORITY_COMMIT,
        "segmented_freeze_commit": SEGMENTED_FREEZE_COMMIT,
        "source_package_path": str(LOCAL_TMP_PACKAGE),
        "remote_root": REMOTE_ROOT,
        "remote_package": REMOTE_PACKAGE,
        "remote_archive": REMOTE_ARCHIVE,
        "local_archive": str(LOCAL_ARCHIVE),
    }


def add_file(tf: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tf.gettarinfo(str(source), arcname)
    if source.name in {"relation_spatial_runtime", "relation_spatial_pmu_preflight"} or arcname.endswith(".sh"):
        info.mode |= 0o755
    with source.open("rb") as handle:
        tf.addfile(info, handle)


def add_bytes(tf: tarfile.TarFile, data: bytes, arcname: str, *, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = mode
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


def target_script() -> str:
    variants_json = json.dumps(VARIANTS)
    return f"""#!/usr/bin/env bash
set -u
SOURCE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SOURCE_ROOT"
mkdir -p official_logs official_outputs _relation_spatial_owned_outputs
printf '%s\\n' "$(date -u +%FT%TZ)" > OFFICIAL_TARGET_STARTED_UTC.txt
prepare_cpufreq() {{
python3 - <<'PY'
import json, pathlib, sys, time
cpus = [4, 5]
receipt = {{
    "schema": "FAMILY10H_LOCAL_PAIRED_CPUFREQ_PREPARE_V1",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "requested_governor": "performance",
    "cpus": {{}},
    "failures": [],
}}
for cpu in cpus:
    root = pathlib.Path(f"/sys/devices/system/cpu/cpu{{cpu}}/cpufreq")
    governor = root / "scaling_governor"
    available = root / "scaling_available_governors"
    try:
        before = governor.read_text().strip()
        available_values = available.read_text().strip().split()
        if "performance" not in available_values:
            receipt["failures"].append(f"cpu{{cpu}} performance governor unavailable")
            after = before
        else:
            governor.write_text("performance\\n")
            after = governor.read_text().strip()
            if after != "performance":
                receipt["failures"].append(f"cpu{{cpu}} governor did not switch to performance: {{after}}")
        receipt["cpus"][str(cpu)] = {{
            "governor_path": str(governor),
            "available_governors": available_values,
            "before_governor": before,
            "prepared_governor": after,
            "prepared": after == "performance",
        }}
    except Exception as exc:
        receipt["failures"].append(f"cpu{{cpu}} governor prepare failed: {{exc}}")
receipt["passed"] = not receipt["failures"] and all(item.get("prepared") for item in receipt["cpus"].values())
pathlib.Path("official_logs/cpufreq_prepare.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
sys.exit(0 if receipt["passed"] else 1)
PY
}}
restore_cpufreq() {{
python3 - <<'PY'
import json, pathlib, sys, time
prepare_path = pathlib.Path("official_logs/cpufreq_prepare.json")
receipt = {{
    "schema": "FAMILY10H_LOCAL_PAIRED_CPUFREQ_RESTORE_V1",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "cpus": {{}},
    "failures": [],
}}
try:
    prepare = json.loads(prepare_path.read_text())
except Exception as exc:
    receipt["failures"].append(f"prepare receipt unavailable: {{exc}}")
    prepare = {{"cpus": {{}}}}
for cpu, item in prepare.get("cpus", {{}}).items():
    path = pathlib.Path(item["governor_path"])
    before = item["before_governor"]
    try:
        current = path.read_text().strip()
        path.write_text(before + "\\n")
        after = path.read_text().strip()
        receipt["cpus"][cpu] = {{
            "governor_path": str(path),
            "prepared_governor": item.get("prepared_governor"),
            "restore_target_governor": before,
            "governor_before_restore": current,
            "governor_after_restore": after,
            "restored": after == before,
        }}
        if after != before:
            receipt["failures"].append(f"cpu{{cpu}} governor restore mismatch: {{after}} != {{before}}")
    except Exception as exc:
        receipt["failures"].append(f"cpu{{cpu}} governor restore failed: {{exc}}")
receipt["passed"] = not receipt["failures"] and all(item.get("restored") for item in receipt["cpus"].values())
pathlib.Path("official_logs/cpufreq_restore.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
sys.exit(0 if receipt["passed"] else 1)
PY
}}
prepare_cpufreq
cpufreq_prepare_rc=$?
printf '%s\\n' "$cpufreq_prepare_rc" > official_logs/cpufreq_prepare.rc
trap 'restore_cpufreq; printf "%s\\n" "$?" > official_logs/cpufreq_restore.rc' EXIT
if [ "$cpufreq_prepare_rc" -ne 0 ]; then
  python3 - <<'PY'
import hashlib, json, pathlib
summary = {{
    "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_OFFICIAL_TARGET_RUN_SUMMARY_V1",
    "transaction_run_id": "{TRANSACTION_RUN_ID}",
    "attempt_consumed": False,
    "target_preflight_rc": None,
    "cpufreq_prepare": json.loads(pathlib.Path("official_logs/cpufreq_prepare.json").read_text()),
    "variants": {{}},
    "raw_record_count_total": 0,
    "pair_observation_count_total": 0,
    "source_death_receipt_count_total": 0,
    "all_variant_rc_zero": False,
    "passed": False,
    "small_wall_crossed": False,
}}
summary["summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
pathlib.Path("OFFICIAL_TARGET_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
  exit 22
fi
RELATION_MANIFEST_SHA="$(sha256sum RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json | awk '{{print $1}}')"
FAMILY10H_RELATION_SPATIAL_LIVE_AUTHORITY="{SEGMENTED_TRANSACTION_RUN_ID}" \\
FAMILY10H_RELATION_SPATIAL_SOURCE_AUTHORITY_COMMIT="{SEGMENTED_SOURCE_AUTHORITY_COMMIT}" \\
FAMILY10H_RELATION_SPATIAL_FREEZE_COMMIT="{SEGMENTED_FREEZE_COMMIT}" \\
FAMILY10H_RELATION_SPATIAL_MANIFEST_SHA256="$RELATION_MANIFEST_SHA" \\
python3 -B relation_spatial_target.py --target-preflight --source-root "$SOURCE_ROOT" --output-root "$SOURCE_ROOT/_relation_spatial_owned_outputs/preflight_probe" \\
  > official_logs/target_preflight.json 2> official_logs/target_preflight.stderr
preflight_rc=$?
printf '%s\\n' "$preflight_rc" > official_logs/target_preflight.rc
python3 - "$preflight_rc" <<'PY'
import json, pathlib, sys
rc = int(sys.argv[1])
path = pathlib.Path("official_logs/target_preflight.json")
try:
    data = json.loads(path.read_text())
except Exception as exc:
    data = {{"passed": False, "failures": [f"preflight_json_parse_failed: {{exc}}"], "returncode": rc}}
passed = rc == 0 and data.get("passed") is True
pathlib.Path("official_logs/target_preflight_passed.txt").write_text("true\\n" if passed else "false\\n")
if not passed:
    pathlib.Path("official_logs/target_preflight_failed.txt").write_text(json.dumps(data.get("failures", [])) + "\\n")
sys.exit(0 if passed else 1)
PY
preflight_pass=$?
write_summary() {{
python3 - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path(".")
variants = {variants_json}
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def line_count(path):
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.rstrip(b"\\r\\n"))
def info(path, count=False):
    if not path.exists():
        return {{"present": False, "size_bytes": 0, "sha256": None, "line_count": 0 if count else None}}
    return {{"present": True, "size_bytes": path.stat().st_size, "sha256": sha(path), "line_count": line_count(path) if count else None}}
preflight = {{}}
p = root / "official_logs" / "target_preflight.json"
if p.exists():
    try:
        preflight = json.loads(p.read_text())
    except Exception as exc:
        preflight = {{"passed": False, "parse_error": str(exc)}}
summary = {{
    "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_OFFICIAL_TARGET_RUN_SUMMARY_V1",
    "transaction_run_id": "{TRANSACTION_RUN_ID}",
    "segmented_transaction_run_id": "{SEGMENTED_TRANSACTION_RUN_ID}",
    "runtime_authority": "{RUNTIME_AUTHORITY_VALUE}",
    "attempt_consumed": (root / "ATTEMPT_CONSUMED").exists(),
    "target_preflight": preflight,
    "target_preflight_rc": int((root / "official_logs" / "target_preflight.rc").read_text().strip()) if (root / "official_logs" / "target_preflight.rc").exists() else None,
    "cpufreq_prepare": json.loads((root / "official_logs" / "cpufreq_prepare.json").read_text()) if (root / "official_logs" / "cpufreq_prepare.json").exists() else None,
    "cpufreq_restore": json.loads((root / "official_logs" / "cpufreq_restore.json").read_text()) if (root / "official_logs" / "cpufreq_restore.json").exists() else None,
    "variants": {{}},
    "small_wall_crossed": False,
}}
for variant in variants:
    out = root / "official_outputs" / variant
    rc_path = root / "official_logs" / f"{{variant}}.rc"
    counts_path = root / "official_logs" / f"{{variant}}.counts"
    summary["variants"][variant] = {{
        "rc": int(rc_path.read_text().strip()) if rc_path.exists() else None,
        "counts_text": counts_path.read_text() if counts_path.exists() else "",
        "raw_records.jsonl": info(out / "raw_records.jsonl", True),
        "pair_observations.jsonl": info(out / "pair_observations.jsonl", True),
        "source_death_receipts.jsonl": info(out / "source_death_receipts.jsonl", True),
        "feature_freeze.json": info(out / "feature_freeze.json"),
        "target_execution_receipt.json": info(out / "target_execution_receipt.json"),
    }}
summary["all_variant_rc_zero"] = all(v["rc"] == 0 for v in summary["variants"].values()) if summary["variants"] else False
summary["raw_record_count_total"] = sum(v["raw_records.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["pair_observation_count_total"] = sum(v["pair_observations.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["source_death_receipt_count_total"] = sum(v["source_death_receipts.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["passed"] = (
    summary["attempt_consumed"]
    and summary["target_preflight_rc"] == 0
    and summary["all_variant_rc_zero"]
    and summary["raw_record_count_total"] == 20480
    and summary["pair_observation_count_total"] == 5242880
    and summary["source_death_receipt_count_total"] == 20480
)
payload = json.dumps(summary, indent=2, sort_keys=True)
digest = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
summary["summary_sha256"] = digest
(root / "OFFICIAL_TARGET_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
}}
if [ "$preflight_pass" -ne 0 ]; then
  write_summary
  exit 23
fi
printf '%s\\n' "$(date -u +%FT%TZ)" > ATTEMPT_CONSUMED
overall=0
for variant in {' '.join(VARIANTS)}; do
  schedule="$SOURCE_ROOT/LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES/$variant.tsv"
  output="$SOURCE_ROOT/official_outputs/$variant"
  stdout_path="$SOURCE_ROOT/official_logs/$variant.stdout"
  stderr_path="$SOURCE_ROOT/official_logs/$variant.stderr"
  rc_path="$SOURCE_ROOT/official_logs/$variant.rc"
  counts_path="$SOURCE_ROOT/official_logs/$variant.counts"
  if [ -e "$output" ]; then
    echo 97 > "$rc_path"
    echo "output_preexisting" > "$stderr_path"
    overall=97
    continue
  fi
  FAMILY10H_RELATION_SPATIAL_RUNTIME_AUTHORITY="{RUNTIME_AUTHORITY_VALUE}" \\
    chrt -f 80 "$SOURCE_ROOT/relation_spatial_runtime" --execute-schedule "$schedule" "$output" \\
    > "$stdout_path" 2> "$stderr_path"
  rc=$?
  echo "$rc" > "$rc_path"
  {{
    echo "raw_records=$(test -f "$output/raw_records.jsonl" && wc -l < "$output/raw_records.jsonl" || echo 0)"
    echo "pair_observations=$(test -f "$output/pair_observations.jsonl" && wc -l < "$output/pair_observations.jsonl" || echo 0)"
    echo "source_death_receipts=$(test -f "$output/source_death_receipts.jsonl" && wc -l < "$output/source_death_receipts.jsonl" || echo 0)"
  }} > "$counts_path"
  if [ "$rc" -ne 0 ]; then
    overall="$rc"
  fi
done
write_summary
printf '%s\\n' "$(date -u +%FT%TZ)" > OFFICIAL_TARGET_ENDED_UTC.txt
exit "$overall"
"""


def build_source_package(local_identity_receipt: dict[str, Any]) -> dict[str, Any]:
    if LOCAL_TMP_PACKAGE.exists():
        LOCAL_TMP_PACKAGE.unlink()
    file_receipts: dict[str, Any] = {}
    segmented_files = [path for path in SEGMENTED_ROOT.iterdir() if path.is_file()]
    local_files = [path for path in PACKAGE_ROOT.iterdir() if path.is_file()]
    local_schedule_files = [path for path in (PACKAGE_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES").iterdir() if path.is_file()]
    manifest_file_sha = sha256_file(SEGMENTED_ROOT / "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json")
    deployment = {
        "schema": "FAMILY10H_RELATION_SPATIAL_DEPLOYMENT_CUSTODY_V1",
        "science_package_id": "family10h_relation_spatial_pair_readout_v1_1_segmented",
        "transaction_run_id": SEGMENTED_TRANSACTION_RUN_ID,
        "relation_source_authority_commit": SEGMENTED_SOURCE_AUTHORITY_COMMIT,
        "relation_manifest_freeze_commit": SEGMENTED_FREEZE_COMMIT,
        "manifest_file_sha256": manifest_file_sha,
        "controller_verified_head_equals_origin": local_identity_receipt["head_equals_origin"],
        "controller_verified_clean_worktree": False,
        "controller_status_short": local_identity_receipt["status_short"],
        "fixture_backend_allowed": False,
        "one_attempt_ceiling": 1,
        "local_paired_transaction_run_id": TRANSACTION_RUN_ID,
    }
    with tarfile.open(LOCAL_TMP_PACKAGE, "w:gz") as tf:
        for path in segmented_files:
            arc = f"source/{path.name}"
            add_file(tf, path, arc)
            file_receipts[arc] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in local_files:
            arc = f"source/{path.name}"
            add_file(tf, path, arc)
            file_receipts[arc] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in local_schedule_files:
            arc = f"source/LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES/{path.name}"
            add_file(tf, path, arc)
            file_receipts[arc] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        add_bytes(tf, json.dumps(deployment, indent=2, sort_keys=True).encode("utf-8") + b"\n", "source/RELATION_SPATIAL_DEPLOYMENT_CUSTODY.json")
        add_bytes(tf, target_script().encode("utf-8"), "source/OFFICIAL_TARGET_EXECUTE_LOCAL_PAIRED.sh", mode=0o755)
    return {
        "schema": "FAMILY10H_LOCAL_PAIRED_SOURCE_PACKAGE_RECEIPT_V1",
        "created_at": utc_now(),
        "path": str(LOCAL_TMP_PACKAGE),
        "sha256": sha256_file(LOCAL_TMP_PACKAGE),
        "size_bytes": LOCAL_TMP_PACKAGE.stat().st_size,
        "file_count": len(file_receipts) + 2,
        "files": file_receipts,
        "deployment_custody": deployment,
    }


def deploy_and_execute(package: dict[str, Any]) -> dict[str, Any]:
    absence = run_remote(f"set -eu; test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}", timeout=30)
    require(absence["returncode"] == 0, f"remote canonical path is not fresh: {absence['stderr'] or absence['stdout']}")
    create_base = run_remote(f"set -eu; mkdir -p {REMOTE_BASE}; mkdir -m 0700 {REMOTE_ROOT}; printf '%s\\n' '{TRANSACTION_RUN_ID}' > {REMOTE_ROOT}/{OWNER_MARKER}", timeout=30)
    require(create_base["returncode"] == 0, "remote root creation failed")
    upload = run_scp(str(LOCAL_TMP_PACKAGE), f"{TARGET_HOST}:{REMOTE_PACKAGE}", timeout=240)
    require(upload["returncode"] == 0, "source package upload failed")
    extract = run_remote(
        f"set -eu; tar -xzf {REMOTE_PACKAGE} -C {REMOTE_ROOT}; "
        f"test -x {REMOTE_ROOT}/source/relation_spatial_runtime; "
        f"test -x {REMOTE_ROOT}/source/relation_spatial_pmu_preflight; "
        f"test -x {REMOTE_ROOT}/source/OFFICIAL_TARGET_EXECUTE_LOCAL_PAIRED.sh",
        timeout=120,
    )
    require(extract["returncode"] == 0, "remote source package extraction failed")
    write_json(ATTEMPT_DIR / "OFFICIAL_DEPLOYMENT_RECEIPT.json", {
        "schema": "FAMILY10H_LOCAL_PAIRED_DEPLOYMENT_RECEIPT_V1",
        "created_at": utc_now(),
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "remote_package": REMOTE_PACKAGE,
        "remote_archive": REMOTE_ARCHIVE,
        "source_package": package,
        "remote_absence": absence,
        "remote_root_create": create_base,
        "upload": upload,
        "extract": extract,
        "live_invocation_count": 0,
        "pmu_acquisition_count": 0,
        "small_wall_crossed": False,
    })
    started = utc_now()
    command = f"set +e; cd {REMOTE_ROOT}/source; ./OFFICIAL_TARGET_EXECUTE_LOCAL_PAIRED.sh"
    write_json(ATTEMPT_DIR / "OFFICIAL_LIVE_INVOCATION_STARTED.json", {
        "schema": "FAMILY10H_LOCAL_PAIRED_LIVE_INVOCATION_STARTED_V1",
        "started_at": started,
        "target_host": TARGET_HOST,
        "remote_root": REMOTE_ROOT,
        "command": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", TARGET_HOST, command],
        "authorized_physical_attempt_count": 1,
        "small_wall_crossed": False,
    })
    live = run_remote(command, timeout=7200)
    (ATTEMPT_DIR / "OFFICIAL_TARGET_STDOUT.txt").write_text(live["stdout"], encoding="utf-8")
    (ATTEMPT_DIR / "OFFICIAL_TARGET_STDERR.txt").write_text(live["stderr"], encoding="utf-8")
    completed = {
        "schema": "FAMILY10H_LOCAL_PAIRED_LIVE_INVOCATION_COMPLETED_V1",
        "started_at": started,
        "ended_at": utc_now(),
        "elapsed_seconds": live["elapsed_seconds"],
        "returncode": live["returncode"],
        "stdout_sha256": sha256_bytes(live["stdout"].encode("utf-8")),
        "stderr_sha256": sha256_bytes(live["stderr"].encode("utf-8")),
        "stdout_size": len(live["stdout"].encode("utf-8")),
        "stderr_size": len(live["stderr"].encode("utf-8")),
        "target_stdout_path": str(ATTEMPT_DIR / "OFFICIAL_TARGET_STDOUT.txt"),
        "target_stderr_path": str(ATTEMPT_DIR / "OFFICIAL_TARGET_STDERR.txt"),
        "attempt_consumed_if_target_preflight_passed": True,
        "small_wall_crossed": False,
    }
    write_json(ATTEMPT_DIR / "OFFICIAL_LIVE_INVOCATION_COMPLETED.json", completed)
    return completed


def remote_archive_copyback() -> dict[str, Any]:
    make_archive = run_remote(
        f"set -eu; test -d {REMOTE_ROOT}/source; rm -f {REMOTE_ARCHIVE}; "
        f"tar -czf {REMOTE_ARCHIVE} -C {REMOTE_ROOT} source; "
        f"sha256sum {REMOTE_ARCHIVE}; stat -c%s {REMOTE_ARCHIVE}",
        timeout=900,
    )
    require(make_archive["returncode"] == 0, "remote archive creation failed")
    if LOCAL_TMP_ARCHIVE.exists():
        LOCAL_TMP_ARCHIVE.unlink()
    scp_archive = run_scp(f"{TARGET_HOST}:{REMOTE_ARCHIVE}", str(LOCAL_TMP_ARCHIVE), timeout=900)
    require(scp_archive["returncode"] == 0, "remote archive copy-back failed")
    if LOCAL_ARCHIVE.exists():
        raise RuntimeError(f"local archive already exists: {LOCAL_ARCHIVE}")
    shutil.move(str(LOCAL_TMP_ARCHIVE), str(LOCAL_ARCHIVE))
    remote_lines = [line.strip() for line in make_archive["stdout"].splitlines() if line.strip()]
    remote_sha = remote_lines[0].split()[0] if remote_lines else None
    remote_size = int(remote_lines[1]) if len(remote_lines) > 1 and remote_lines[1].isdigit() else None
    local_sha = sha256_file(LOCAL_ARCHIVE)
    local_size = LOCAL_ARCHIVE.stat().st_size
    return {
        "schema": "FAMILY10H_LOCAL_PAIRED_COPY_BACK_VERIFICATION_V1",
        "created_at": utc_now(),
        "remote_archive": REMOTE_ARCHIVE,
        "local_archive": str(LOCAL_ARCHIVE),
        "remote_archive_sha256": remote_sha,
        "remote_archive_size": remote_size,
        "local_archive_sha256": local_sha,
        "local_archive_size": local_size,
        "archive_sha256_equal": remote_sha == local_sha,
        "archive_size_equal": remote_size == local_size,
        "passed": remote_sha == local_sha and remote_size == local_size,
        "make_archive": make_archive,
        "scp_archive": scp_archive,
        "small_wall_crossed": False,
    }


def inventory_archive() -> dict[str, Any]:
    required = [
        "source/OFFICIAL_TARGET_RUN_SUMMARY.json",
        "source/LOCAL_PAIRED_DIFFERENTIAL_PACKAGE_MANIFEST.json",
        "source/LOCAL_PAIRED_DIFFERENTIAL_SCHEDULE_MANIFEST.json",
        "source/LOCAL_PAIRED_DIFFERENTIAL_THRESHOLD_CONTRACT.json",
        "source/relation_spatial_runtime",
        "source/relation_spatial_pmu_preflight",
        "source/official_logs/target_preflight.json",
    ]
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as tf:
        names = set(tf.getnames())
        missing = [name for name in required if name not in names]
        summary = {}
        if "source/OFFICIAL_TARGET_RUN_SUMMARY.json" in names:
            handle = tf.extractfile("source/OFFICIAL_TARGET_RUN_SUMMARY.json")
            if handle is not None:
                summary = json.loads(handle.read().decode("utf-8"))
        variant_missing = []
        for variant in VARIANTS:
            for filename in ["raw_records.jsonl", "pair_observations.jsonl", "source_death_receipts.jsonl", "feature_freeze.json", "target_execution_receipt.json"]:
                member = f"source/official_outputs/{variant}/{filename}"
                if member not in names:
                    variant_missing.append(member)
    return {
        "schema": "FAMILY10H_LOCAL_PAIRED_ARCHIVE_INVENTORY_V1",
        "created_at": utc_now(),
        "archive_path": str(LOCAL_ARCHIVE),
        "archive_sha256": sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "required_missing": missing,
        "variant_output_missing": variant_missing,
        "target_summary": summary,
        "target_summary_passed": summary.get("passed") is True,
        "raw_record_count_total": summary.get("raw_record_count_total"),
        "pair_observation_count_total": summary.get("pair_observation_count_total"),
        "source_death_receipt_count_total": summary.get("source_death_receipt_count_total"),
        "passed": not missing and not variant_missing and summary.get("passed") is True,
        "small_wall_crossed": False,
    }


def cleanup_remote(copyback: dict[str, Any]) -> dict[str, Any]:
    if not copyback.get("passed"):
        return {
            "schema": "FAMILY10H_LOCAL_PAIRED_REMOTE_CLEANUP_RECEIPT_V1",
            "created_at": utc_now(),
            "skipped": True,
            "reason": "copy-back did not verify",
            "passed": False,
        }
    cleanup = run_remote(
        f"set -eu; test -f {REMOTE_ROOT}/{OWNER_MARKER}; grep -qx '{TRANSACTION_RUN_ID}' {REMOTE_ROOT}/{OWNER_MARKER}; "
        f"rm -rf {REMOTE_ROOT} {REMOTE_ARCHIVE} {REMOTE_PACKAGE}; "
        f"test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}",
        timeout=240,
    )
    return {
        "schema": "FAMILY10H_LOCAL_PAIRED_REMOTE_CLEANUP_RECEIPT_V1",
        "created_at": utc_now(),
        "remote_root": REMOTE_ROOT,
        "remote_archive": REMOTE_ARCHIVE,
        "remote_package": REMOTE_PACKAGE,
        "cleanup": cleanup,
        "passed": cleanup["returncode"] == 0,
    }


def main() -> int:
    require(PACKAGE_ROOT.exists(), f"package root missing: {PACKAGE_ROOT}")
    require(SEGMENTED_ROOT.exists(), f"segmented root missing: {SEGMENTED_ROOT}")
    require(not LOCAL_ARCHIVE.exists(), f"local archive already exists: {LOCAL_ARCHIVE}")
    identity = local_identity()
    write_json(ATTEMPT_DIR / "OFFICIAL_LOCAL_IDENTITY_RECEIPT.json", identity)
    package = build_source_package(identity)
    write_json(ATTEMPT_DIR / "OFFICIAL_SOURCE_PACKAGE_RECEIPT.json", package)
    live = deploy_and_execute(package)
    copyback = remote_archive_copyback()
    write_json(ATTEMPT_DIR / "OFFICIAL_COPY_BACK_VERIFICATION.json", copyback)
    inventory = inventory_archive()
    write_json(ATTEMPT_DIR / "OFFICIAL_ARCHIVE_INVENTORY.json", inventory)
    cleanup = cleanup_remote(copyback)
    write_json(ATTEMPT_DIR / "OFFICIAL_REMOTE_CLEANUP_RECEIPT.json", cleanup)
    result = {
        "schema": "FAMILY10H_LOCAL_PAIRED_LIVE_CONTROLLER_RESULT_V1",
        "created_at": utc_now(),
        "passed": live["returncode"] == 0 and copyback["passed"] and inventory["passed"] and cleanup["passed"],
        "live_returncode": live["returncode"],
        "archive_sha256": copyback["local_archive_sha256"],
        "archive_size": copyback["local_archive_size"],
        "target_summary_passed": inventory["target_summary_passed"],
        "raw_record_count_total": inventory["raw_record_count_total"],
        "pair_observation_count_total": inventory["pair_observation_count_total"],
        "source_death_receipt_count_total": inventory["source_death_receipt_count_total"],
        "cleanup_passed": cleanup["passed"],
        "small_wall_crossed": False,
    }
    write_json(ATTEMPT_DIR / "OFFICIAL_CONTROLLER_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
