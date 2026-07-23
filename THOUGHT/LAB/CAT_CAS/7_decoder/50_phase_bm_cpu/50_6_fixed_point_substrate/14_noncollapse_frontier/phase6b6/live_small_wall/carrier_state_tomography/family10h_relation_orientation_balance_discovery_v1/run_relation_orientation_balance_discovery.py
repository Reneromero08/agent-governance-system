#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_ID = "family10h_relation_orientation_balance_discovery_v1_0"
TARGET_HOST = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
REMOTE_ROOT = f"{REMOTE_BASE}/{RUN_ID}"
REMOTE_PACKAGE = f"{REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
REMOTE_ARCHIVE = f"{REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
OWNER_MARKER = f".{RUN_ID}_owner"
RUNTIME_AUTHORITY_VALUE = "family10h_relation_spatial_pair_readout_v1_0"

VARIANTS = [
    "primary_relation_pair",
    "query_inversion_control",
    "carrier_off_control",
    "carrier_off_inversion_control",
    "relation_sham",
]
QUERY_BY_VARIANT = {
    "primary_relation_pair": "query_relation_pair",
    "query_inversion_control": "query_inversion_control",
    "carrier_off_control": "carrier_off_control",
    "carrier_off_inversion_control": "carrier_off_inversion_control",
    "relation_sham": "relation_sham",
}
ROWS_PER_VARIANT = 512
PAIR_SAMPLE_COUNT = 256
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT
MATCHED_PERMUTATION_COUNT = 63
MATCHED_PERMUTATION_SEED = "family10h-orientation-balance-discovery-v1"

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
SEGMENTED_ROOT = CARRIER_ROOT / "family10h_relation_spatial_pair_readout_v1_1_segmented"
LOCAL_PAIRED_ROOT = CARRIER_ROOT / "family10h_primary_minus_sham_local_paired_differential_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "ORIENTATION_BALANCE_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_ORIENTATION_BALANCE_DISCOVERY_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_ORIENTATION_BALANCE_DISCOVERY_SUMMARY.md"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return sha256_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def wsl_path(path: Path) -> str:
    completed = run_local(["wsl.exe", "wslpath", "-a", str(path.resolve())], timeout=20)
    if completed["returncode"] == 0 and completed["stdout"].strip():
        return completed["stdout"].strip()
    drive = path.resolve().drive.rstrip(":").lower()
    suffix = "/".join(path.resolve().parts[1:])
    return f"/mnt/{drive}/{suffix}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def copy_segmented_source() -> None:
    if SOURCE_ROOT.exists():
        shutil.rmtree(SOURCE_ROOT)
    SOURCE_ROOT.mkdir(parents=True)
    include = [
        "RELATION_GRAMMAR.json",
        "RELATION_GRAMMAR.tsv",
        "RELATION_GRAMMAR.sha256",
        "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.json",
        "RELATION_SPATIAL_IMPLEMENTATION_MANIFEST.sha256",
        "RELATION_SPATIAL_PUBLIC_SCHEDULE.json",
        "RELATION_SPATIAL_PUBLIC_SCHEDULE.tsv",
        "RELATION_SPATIAL_PUBLIC_SCHEDULE.sha256",
        "RELATION_SPATIAL_SENSOR_AUTHORITY_BINDING.json",
        "RELATION_SPATIAL_SOURCE_HASHES.json",
        "relation_spatial_public.py",
        "relation_spatial_adjudication.py",
        "relation_spatial_physical_adjudication.py",
        "relation_spatial_pmu_preflight",
        "relation_spatial_pmu_preflight.c",
        "relation_spatial_runtime.c",
        "relation_spatial_runtime.h",
        "relation_spatial_target.py",
    ]
    for name in include:
        shutil.copy2(SEGMENTED_ROOT / name, SOURCE_ROOT / name)


def patch_runtime_for_transform_kill() -> None:
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    text = header.read_text(encoding="utf-8")
    text = text.replace(
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4",
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4,\n"
        "    RELATION_SPATIAL_CONTROL_QUERY_INVERSION = 5,\n"
        "    RELATION_SPATIAL_CONTROL_CARRIER_OFF = 6,\n"
        "    RELATION_SPATIAL_CONTROL_CARRIER_OFF_QUERY_INVERSION = 7",
    )
    header.write_text(text, encoding="utf-8", newline="\n")

    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")
    text = text.replace(
        '    if (strcmp(query, "distance_matched_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED;\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n",
        '    if (strcmp(query, "distance_matched_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "query_inversion_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_QUERY_INVERSION;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "carrier_off_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_CARRIER_OFF;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "carrier_off_inversion_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_CARRIER_OFF_QUERY_INVERSION;\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n",
    )
    text = text.replace(
        "    if (row->control == RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED) {\n"
        "        return (sample_index & 1u) ? ((a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT)\n"
        "                                  : ((a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT);\n"
        "    }\n"
        "    return relation_spatial_map_index(row->r_query, a_index);\n",
        "    if (row->control == RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED) {\n"
        "        return (sample_index & 1u) ? ((a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT)\n"
        "                                  : ((a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT);\n"
        "    }\n"
        "    if (row->control == RELATION_SPATIAL_CONTROL_QUERY_INVERSION || row->control == RELATION_SPATIAL_CONTROL_CARRIER_OFF_QUERY_INVERSION) {\n"
        "        relation_spatial_relation_id inverted = row->r_query == RELATION_SPATIAL_R0 ? RELATION_SPATIAL_R1 : RELATION_SPATIAL_R0;\n"
        "        return relation_spatial_map_index(inverted, a_index);\n"
        "    }\n"
        "    return relation_spatial_map_index(row->r_query, a_index);\n",
    )
    prep_block = (
        "            relation_spatial_preparation prep;\n"
        "            prep.bank_a_work = row.bank_a_work;\n"
        "            prep.bank_b_work = row.bank_b_work;\n"
        "            prep.relation = row.r_prepare;\n"
        "            prep.source_order = row.source_order;\n"
        "            prep.cyclic_origin = row.cyclic_origin;\n"
        "            shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
    )
    patched_prep_block = (
        "            if (row.control == RELATION_SPATIAL_CONTROL_CARRIER_OFF || row.control == RELATION_SPATIAL_CONTROL_CARRIER_OFF_QUERY_INVERSION) {\n"
        "                relation_spatial_prefault(&shared->state);\n"
        "                flush_state_lines(&shared->state);\n"
        "                shared->preparation_ok = 1;\n"
        "            } else {\n"
        "                relation_spatial_preparation prep;\n"
        "                prep.bank_a_work = row.bank_a_work;\n"
        "                prep.bank_b_work = row.bank_b_work;\n"
        "                prep.relation = row.r_prepare;\n"
        "                prep.source_order = row.source_order;\n"
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "            }\n"
    )
    count = text.count(prep_block)
    require(count == 1, f"runtime execution prep block replacement count unexpected: {count}")
    text = text.replace(prep_block, patched_prep_block)
    runtime.write_text(text, encoding="utf-8", newline="\n")


def compile_runtime() -> dict[str, Any]:
    runtime = SOURCE_ROOT / "relation_spatial_runtime"
    pmu = SOURCE_ROOT / "relation_spatial_pmu_preflight"
    compile_runtime_result = run_local(
        [
            "wsl.exe",
            "--",
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-g",
            "-o",
            wsl_path(runtime),
            wsl_path(SOURCE_ROOT / "relation_spatial_runtime.c"),
            "-lm",
        ],
        timeout=120,
    )
    compile_pmu_result = run_local(
        [
            "wsl.exe",
            "--",
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-g",
            "-o",
            wsl_path(pmu),
            wsl_path(SOURCE_ROOT / "relation_spatial_pmu_preflight.c"),
        ],
        timeout=120,
    )
    self_test = (
        run_local(["wsl.exe", "--", wsl_path(runtime), "--self-test"], timeout=60)
        if runtime.exists()
        else {"returncode": 1, "stdout": "", "stderr": "runtime missing"}
    )
    receipt = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_RUNTIME_BUILD_V1",
        "runtime_compile": compile_runtime_result,
        "pmu_compile": compile_pmu_result,
        "runtime_self_test": self_test,
        "runtime_sha256": sha256_file(runtime) if runtime.exists() else None,
        "pmu_helper_sha256": sha256_file(pmu) if pmu.exists() else None,
        "passed": compile_runtime_result["returncode"] == 0
        and compile_pmu_result["returncode"] == 0
        and self_test["returncode"] == 0
        and runtime.exists()
        and pmu.exists(),
    }
    write_json(HERE / "RUNTIME_BUILD_RECEIPT.json", receipt)
    return receipt


def make_rows() -> dict[str, Any]:
    source_schedule = LOCAL_PAIRED_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES" / "round0_query_relation_pair.tsv"
    with source_schedule.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        base_rows = list(reader)[:ROWS_PER_VARIANT]
    require(len(base_rows) == ROWS_PER_VARIANT, "not enough base rows for discovery schedule")
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    schedules: dict[str, Any] = {}
    for variant in VARIANTS:
        query = QUERY_BY_VARIANT[variant]
        rows = []
        for index, row in enumerate(base_rows):
            copied = dict(row)
            copied["execution_ordinal"] = str(index)
            copied["query"] = query
            copied["operation_semantics_id"] = query
            copied["control_semantics_id"] = "none" if variant == "primary_relation_pair" else query
            copied["tuple_id"] = f"{RUN_ID}:{variant}:{index:06d}:{sha256_bytes((variant + ':' + str(index)).encode())[:16]}"
            copied["block_id"] = f"{variant}_{copied['block_id']}"
            copied["matched_twin_group"] = f"{copied['block_id']}:relation_matrix:{variant}"
            copied["matched_twin_pair"] = f"{copied['block_id']}:relation_pair_{copied['block_local_position']}:{variant}"
            rows.append(copied)
        path = SCHEDULE_DIR / f"{variant}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
        schedules[variant] = {
            "query": query,
            "path": f"ORIENTATION_BALANCE_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_DISCOVERY_SCHEDULE_MANIFEST_V1",
        "run_id": RUN_ID,
        "source_schedule": str(source_schedule),
        "source_schedule_sha256": sha256_file(source_schedule),
        "rows_per_variant": ROWS_PER_VARIANT,
        "pair_sample_count": PAIR_SAMPLE_COUNT,
        "variants": schedules,
        "claim_boundary": {
            "exploratory_only": True,
            "positive_scientific_claim": False,
            "small_wall_crossed": False,
        },
        "mechanism_question": "Does the primary relation-orientation signal survive after subtracting carrier-off receiver-query orientation bias?",
    }
    manifest["schedule_manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    write_json(SOURCE_ROOT / "RELATION_ORIENTATION_BALANCE_SCHEDULE_MANIFEST.json", manifest)
    write_json(HERE / "RELATION_ORIENTATION_BALANCE_SCHEDULE_MANIFEST.json", manifest)
    return manifest


def target_script() -> str:
    variants_json = json.dumps(VARIANTS)
    return f"""#!/usr/bin/env bash
set -u
SOURCE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SOURCE_ROOT"
mkdir -p discovery_logs discovery_outputs
printf '%s\\n' "$(date -u +%FT%TZ)" > DISCOVERY_TARGET_STARTED_UTC.txt
prepare_cpufreq() {{
python3 - <<'PY'
import json, pathlib, sys, time
receipt = {{"schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_CPUFREQ_PREPARE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
for cpu in [4, 5]:
    root = pathlib.Path(f"/sys/devices/system/cpu/cpu{{cpu}}/cpufreq")
    gov = root / "scaling_governor"
    avail = root / "scaling_available_governors"
    try:
        before = gov.read_text().strip()
        available = avail.read_text().strip().split()
        if "performance" in available:
            gov.write_text("performance\\n")
        after = gov.read_text().strip()
        receipt["cpus"][str(cpu)] = {{"governor_path": str(gov), "available_governors": available, "before_governor": before, "prepared_governor": after, "prepared": after == "performance"}}
        if after != "performance":
            receipt["failures"].append(f"cpu{{cpu}} performance governor unavailable: {{after}}")
    except Exception as exc:
        receipt["failures"].append(f"cpu{{cpu}} governor prepare failed: {{exc}}")
receipt["passed"] = not receipt["failures"] and all(item.get("prepared") for item in receipt["cpus"].values())
pathlib.Path("discovery_logs/cpufreq_prepare.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
sys.exit(0 if receipt["passed"] else 1)
PY
}}
restore_cpufreq() {{
python3 - <<'PY'
import json, pathlib, sys, time
prep_path = pathlib.Path("discovery_logs/cpufreq_prepare.json")
receipt = {{"schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_CPUFREQ_RESTORE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
try:
    prep = json.loads(prep_path.read_text())
except Exception as exc:
    prep = {{"cpus": {{}}}}
    receipt["failures"].append(f"prepare receipt unavailable: {{exc}}")
for cpu, item in prep.get("cpus", {{}}).items():
    path = pathlib.Path(item["governor_path"])
    target = item["before_governor"]
    try:
        current = path.read_text().strip()
        path.write_text(target + "\\n")
        after = path.read_text().strip()
        receipt["cpus"][cpu] = {{"governor_before_restore": current, "restore_target_governor": target, "governor_after_restore": after, "restored": after == target}}
        if after != target:
            receipt["failures"].append(f"cpu{{cpu}} restore mismatch: {{after}} != {{target}}")
    except Exception as exc:
        receipt["failures"].append(f"cpu{{cpu}} restore failed: {{exc}}")
receipt["passed"] = not receipt["failures"] and all(item.get("restored") for item in receipt["cpus"].values())
pathlib.Path("discovery_logs/cpufreq_restore.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
sys.exit(0 if receipt["passed"] else 1)
PY
}}
preflight() {{
python3 - <<'PY'
import json, pathlib, subprocess, sys
failures = []
cpuinfo = pathlib.Path("/proc/cpuinfo").read_text(errors="replace")
if "vendor_id\\t: AuthenticAMD" not in cpuinfo:
    failures.append("vendor mismatch")
if "cpu family\\t: 16" not in cpuinfo:
    failures.append("family mismatch")
if "model\\t\\t: 10" not in cpuinfo:
    failures.append("model mismatch")
for cpu in [4, 5]:
    if subprocess.run(["taskset", "-c", str(cpu), "true"], capture_output=True).returncode != 0:
        failures.append(f"cpu{{cpu}} pinning failed")
sensor = pathlib.Path("/sys/class/hwmon/hwmon0/temp1_input")
name = pathlib.Path("/sys/class/hwmon/hwmon0/name")
temp = None
if not sensor.exists() or not name.exists() or name.read_text().strip() != "k10temp":
    failures.append("approved k10temp sensor missing")
else:
    temp = int(sensor.read_text().strip())
    if temp >= 68000:
        failures.append(f"temperature veto {{temp}}")
pmu = subprocess.run(["./relation_spatial_pmu_preflight", "--disabled-group-preflight", "4"], text=True, capture_output=True)
receipt = {{"schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_PREFLIGHT_V1", "passed": not failures and pmu.returncode == 0, "failures": failures, "target_identity": "AuthenticAMD family16 model10 cpus4-5", "sensor": {{"hwmon": "hwmon0", "name": name.read_text().strip() if name.exists() else None, "temp1_input": temp}}, "pmu_preflight": {{"returncode": pmu.returncode, "stdout": pmu.stdout, "stderr": pmu.stderr}}, "small_wall_crossed": False}}
pathlib.Path("discovery_logs/preflight.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\\n")
sys.exit(0 if receipt["passed"] else 1)
PY
}}
prepare_cpufreq
prep_rc=$?
printf '%s\\n' "$prep_rc" > discovery_logs/cpufreq_prepare.rc
trap 'restore_cpufreq; printf "%s\\n" "$?" > discovery_logs/cpufreq_restore.rc' EXIT
if [ "$prep_rc" -ne 0 ]; then
  exit 22
fi
preflight
preflight_rc=$?
printf '%s\\n' "$preflight_rc" > discovery_logs/preflight.rc
if [ "$preflight_rc" -ne 0 ]; then
  exit 23
fi
printf '%s\\n' "$(date -u +%FT%TZ)" > ATTEMPT_CONSUMED
overall=0
for variant in {' '.join(VARIANTS)}; do
  out="$SOURCE_ROOT/discovery_outputs/$variant"
  schedule="$SOURCE_ROOT/ORIENTATION_BALANCE_SCHEDULES/$variant.tsv"
  if [ -e "$out" ]; then
    echo 97 > "discovery_logs/$variant.rc"
    overall=97
    continue
  fi
  FAMILY10H_RELATION_SPATIAL_RUNTIME_AUTHORITY="{RUNTIME_AUTHORITY_VALUE}" \\
    chrt -f 80 "$SOURCE_ROOT/relation_spatial_runtime" --execute-schedule "$schedule" "$out" \\
    > "discovery_logs/$variant.stdout" 2> "discovery_logs/$variant.stderr"
  rc=$?
  echo "$rc" > "discovery_logs/$variant.rc"
  {{
    echo "raw_records=$(test -f "$out/raw_records.jsonl" && wc -l < "$out/raw_records.jsonl" || echo 0)"
    echo "pair_observations=$(test -f "$out/pair_observations.jsonl" && wc -l < "$out/pair_observations.jsonl" || echo 0)"
    echo "source_death_receipts=$(test -f "$out/source_death_receipts.jsonl" && wc -l < "$out/source_death_receipts.jsonl" || echo 0)"
  }} > "discovery_logs/$variant.counts"
  if [ "$rc" -ne 0 ]; then
    overall=$rc
  fi
done
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
def lines(path):
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.rstrip(b"\\r\\n"))
summary = {{"schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_TARGET_SUMMARY_V1", "run_id": "{RUN_ID}", "attempt_consumed": (root / "ATTEMPT_CONSUMED").exists(), "variants": {{}}, "small_wall_crossed": False}}
for variant in variants:
    out = root / "discovery_outputs" / variant
    rc_path = root / "discovery_logs" / f"{{variant}}.rc"
    summary["variants"][variant] = {{"rc": int(rc_path.read_text().strip()) if rc_path.exists() else None, "raw_record_count": lines(out / "raw_records.jsonl"), "pair_observation_count": lines(out / "pair_observations.jsonl"), "source_death_receipt_count": lines(out / "source_death_receipts.jsonl"), "raw_sha256": sha(out / "raw_records.jsonl") if (out / "raw_records.jsonl").exists() else None, "pair_sha256": sha(out / "pair_observations.jsonl") if (out / "pair_observations.jsonl").exists() else None, "death_sha256": sha(out / "source_death_receipts.jsonl") if (out / "source_death_receipts.jsonl").exists() else None}}
summary["raw_record_count_total"] = sum(v["raw_record_count"] for v in summary["variants"].values())
summary["pair_observation_count_total"] = sum(v["pair_observation_count"] for v in summary["variants"].values())
summary["source_death_receipt_count_total"] = sum(v["source_death_receipt_count"] for v in summary["variants"].values())
summary["passed"] = summary["attempt_consumed"] and all(v["rc"] == 0 for v in summary["variants"].values())
summary["summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
(root / "RELATION_ORIENTATION_BALANCE_TARGET_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
printf '%s\\n' "$(date -u +%FT%TZ)" > DISCOVERY_TARGET_ENDED_UTC.txt
exit "$overall"
"""


def build_package() -> dict[str, Any]:
    if LOCAL_PACKAGE.exists():
        LOCAL_PACKAGE.unlink()
    files = {}
    with tarfile.open(LOCAL_PACKAGE, "w:gz") as tf:
        for path in sorted(SOURCE_ROOT.rglob("*")):
            if not path.is_file():
                continue
            arc = f"source/{path.relative_to(SOURCE_ROOT).as_posix()}"
            info = tf.gettarinfo(str(path), arc)
            if path.name in {"relation_spatial_runtime", "relation_spatial_pmu_preflight"}:
                info.mode |= 0o755
            with path.open("rb") as handle:
                tf.addfile(info, handle)
            files[arc] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        data = target_script().encode("utf-8")
        info = tarfile.TarInfo("source/RUN_ORIENTATION_BALANCE_DISCOVERY.sh")
        info.size = len(data)
        info.mode = 0o755
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(data))
        files["source/RUN_ORIENTATION_BALANCE_DISCOVERY.sh"] = {"sha256": sha256_bytes(data), "size_bytes": len(data)}
    receipt = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_SOURCE_PACKAGE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "path": str(LOCAL_PACKAGE),
        "sha256": sha256_file(LOCAL_PACKAGE),
        "size_bytes": LOCAL_PACKAGE.stat().st_size,
        "files": files,
    }
    write_json(HERE / "SOURCE_PACKAGE_RECEIPT.json", receipt)
    return receipt


def deploy_execute_copyback(package: dict[str, Any]) -> dict[str, Any]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCAL_ARCHIVE.exists():
        raise RuntimeError(f"local archive already exists: {LOCAL_ARCHIVE}")
    absence = run_remote(f"set -eu; test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}", timeout=30)
    require(absence["returncode"] == 0, f"remote path not fresh: {absence['stdout']} {absence['stderr']}")
    create = run_remote(f"set -eu; mkdir -p {REMOTE_BASE}; mkdir -m 0700 {REMOTE_ROOT}; printf '%s\\n' '{RUN_ID}' > {REMOTE_ROOT}/{OWNER_MARKER}", timeout=30)
    require(create["returncode"] == 0, "remote root create failed")
    upload = run_scp(str(LOCAL_PACKAGE), f"{TARGET_HOST}:{REMOTE_PACKAGE}", timeout=240)
    require(upload["returncode"] == 0, f"upload failed: {upload['stderr']}")
    extract = run_remote(
        f"set -eu; tar -xzf {REMOTE_PACKAGE} -C {REMOTE_ROOT}; chmod +x {REMOTE_ROOT}/source/relation_spatial_runtime {REMOTE_ROOT}/source/relation_spatial_pmu_preflight {REMOTE_ROOT}/source/RUN_ORIENTATION_BALANCE_DISCOVERY.sh",
        timeout=120,
    )
    require(extract["returncode"] == 0, f"extract failed: {extract['stderr']}")
    live = run_remote(f"set +e; cd {REMOTE_ROOT}/source; ./RUN_ORIENTATION_BALANCE_DISCOVERY.sh", timeout=3600)
    (RUN_ROOT / "DISCOVERY_TARGET_STDOUT.txt").write_text(live["stdout"], encoding="utf-8")
    (RUN_ROOT / "DISCOVERY_TARGET_STDERR.txt").write_text(live["stderr"], encoding="utf-8")
    make_archive = run_remote(
        f"set -eu; rm -f {REMOTE_ARCHIVE}; tar -czf {REMOTE_ARCHIVE} -C {REMOTE_ROOT} source; sha256sum {REMOTE_ARCHIVE}; stat -c%s {REMOTE_ARCHIVE}",
        timeout=900,
    )
    require(make_archive["returncode"] == 0, f"archive failed: {make_archive['stderr']}")
    if LOCAL_TMP_ARCHIVE.exists():
        LOCAL_TMP_ARCHIVE.unlink()
    copy = run_scp(f"{TARGET_HOST}:{REMOTE_ARCHIVE}", str(LOCAL_TMP_ARCHIVE), timeout=900)
    require(copy["returncode"] == 0, f"copyback failed: {copy['stderr']}")
    shutil.move(str(LOCAL_TMP_ARCHIVE), str(LOCAL_ARCHIVE))
    lines = [line.strip() for line in make_archive["stdout"].splitlines() if line.strip()]
    remote_sha = lines[0].split()[0] if lines else None
    remote_size = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else None
    local_sha = sha256_file(LOCAL_ARCHIVE)
    local_size = LOCAL_ARCHIVE.stat().st_size
    copyback = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_COPYBACK_V1",
        "created_at": utc_now(),
        "remote_sha256": remote_sha,
        "remote_size": remote_size,
        "local_sha256": local_sha,
        "local_size": local_size,
        "passed": remote_sha == local_sha and remote_size == local_size,
        "make_archive": make_archive,
        "copy": copy,
    }
    cleanup = {"passed": False, "skipped": True}
    if copyback["passed"]:
        cleanup_cmd = run_remote(
            f"set -eu; test -f {REMOTE_ROOT}/{OWNER_MARKER}; grep -qx '{RUN_ID}' {REMOTE_ROOT}/{OWNER_MARKER}; rm -rf {REMOTE_ROOT} {REMOTE_ARCHIVE} {REMOTE_PACKAGE}; test ! -e {REMOTE_ROOT}; test ! -e {REMOTE_ARCHIVE}; test ! -e {REMOTE_PACKAGE}",
            timeout=240,
        )
        cleanup = {"schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_REMOTE_CLEANUP_V1", "cleanup": cleanup_cmd, "passed": cleanup_cmd["returncode"] == 0}
    receipt = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_LIVE_CONTROLLER_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "package": package,
        "absence": absence,
        "create": create,
        "upload": upload,
        "extract": extract,
        "live": {k: v for k, v in live.items() if k not in {"stdout", "stderr"}},
        "copyback": copyback,
        "cleanup": cleanup,
        "passed": live["returncode"] == 0 and copyback["passed"] and cleanup.get("passed") is True,
        "archive_path": str(LOCAL_ARCHIVE),
        "archive_sha256": local_sha,
        "archive_size": local_size,
        "small_wall_crossed": False,
    }
    write_json(RUN_ROOT / "DISCOVERY_CONTROLLER_RESULT.json", receipt)
    return receipt


def parse_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted((value, index) for index, value in enumerate(values))
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and ordered[end][0] == ordered[pos][0]:
            end += 1
        rank = (pos + end - 1) / 2.0
        for _, index in ordered[pos:end]:
            result[index] = rank
        pos = end
    return result


def pearson(a: list[float], b: list[float]) -> float:
    ma = mean(a)
    mb = mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / math.sqrt(da * db)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def distribution(values: list[float]) -> dict[str, Any]:
    abs_values = [abs(v) for v in values]
    signs = Counter(1 if v > 0 else -1 if v < 0 else 0 for v in values)
    ordered = sorted(abs_values)

    def q(frac: float) -> float:
        if not ordered:
            return 0.0
        idx = (len(ordered) - 1) * frac
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "count": len(values),
        "mean": mean(values),
        "abs_of_mean": abs(mean(values)),
        "abs_mean": mean(abs_values),
        "max_abs": max(abs_values) if abs_values else 0.0,
        "q50_abs": q(0.50),
        "q95_abs": q(0.95),
        "q99_abs": q(0.99),
        "sign_counts": {str(k): signs[k] for k in [-1, 0, 1]},
    }


def pair_groups(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["tuple_id"])].append(pair)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["sample_index"]))
    return grouped


def row_c_pairs(raw: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = pair_groups(pairs)
    rows = []
    for row in raw:
        row_pairs = grouped[row["tuple_id"]]
        a = [float(pair["A_first_touch_cycles"]) for pair in row_pairs]
        b = [float(pair["B_first_touch_cycles"]) for pair in row_pairs]
        rows.append({**row, "C_pair_recomputed": spearman(a, b)})
    return rows


def r_spatial_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    records = []
    for block_id, block in blocks.items():
        cells = {row["relation_cell"]: row for row in block}
        needed = ["prepare_r0__query_r0", "prepare_r0__query_r1", "prepare_r1__query_r0", "prepare_r1__query_r1"]
        if any(name not in cells for name in needed):
            continue
        r = 0.5 * (
            cells["prepare_r0__query_r0"]["C_pair_recomputed"]
            + cells["prepare_r1__query_r1"]["C_pair_recomputed"]
            - cells["prepare_r0__query_r1"]["C_pair_recomputed"]
            - cells["prepare_r1__query_r0"]["C_pair_recomputed"]
        )
        first = block[0]
        records.append(
            {
                "block_id": block_id,
                "R_spatial": r,
                "session": first["session"],
                "replicate": first["replicate"],
                "mapping": first["mapping"],
                "source_order": first["source_order"],
                "query_order": first["query_order"],
                "cyclic_origin": first["cyclic_origin"],
            }
        )
    return records


def analyze_archive(controller: dict[str, Any]) -> dict[str, Any]:
    require(LOCAL_ARCHIVE.exists(), "archive missing for analysis")
    reports = {}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as tf:
        names = set(tf.getnames())
        for variant in VARIANTS:
            prefix = f"source/discovery_outputs/{variant}"
            for suffix in ["raw_records.jsonl", "pair_observations.jsonl", "source_death_receipts.jsonl"]:
                require(f"{prefix}/{suffix}" in names, f"missing archive member {prefix}/{suffix}")
            raw = parse_jsonl_bytes(tf.extractfile(f"{prefix}/raw_records.jsonl").read())  # type: ignore[union-attr]
            pairs = parse_jsonl_bytes(tf.extractfile(f"{prefix}/pair_observations.jsonl").read())  # type: ignore[union-attr]
            deaths = parse_jsonl_bytes(tf.extractfile(f"{prefix}/source_death_receipts.jsonl").read())  # type: ignore[union-attr]
            rows = row_c_pairs(raw, pairs)
            records = r_spatial_records(rows)
            values = [record["R_spatial"] for record in records]
            reports[variant] = {
                "query": QUERY_BY_VARIANT[variant],
                "raw_record_count": len(raw),
                "pair_observation_count": len(pairs),
                "source_death_receipt_count": len(deaths),
                "block_count": len(records),
                "R_spatial": distribution(values),
                "source_alive_valid": all(death.get("source_alive_during_query") is True for death in deaths),
            }
    primary = reports["primary_relation_pair"]["R_spatial"]["mean"]
    inverted = reports["query_inversion_control"]["R_spatial"]["mean"]
    carrier_off = reports["carrier_off_control"]["R_spatial"]["mean"]
    carrier_off_inverted = reports["carrier_off_inversion_control"]["R_spatial"]["mean"]
    sham = reports["relation_sham"]["R_spatial"]["mean"]
    relation_orientation = 0.5 * (primary - inverted)
    carrier_off_orientation = 0.5 * (carrier_off - carrier_off_inverted)
    net_orientation = relation_orientation - carrier_off_orientation
    result = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_DISCOVERY_ANALYSIS_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "variant_reports": reports,
        "coordinates": {
            "R_primary": primary,
            "R_query_inversion": inverted,
            "R_carrier_off": carrier_off,
            "R_carrier_off_inversion": carrier_off_inverted,
            "R_relation_sham": sham,
            "D_primary_minus_sham": primary - sham,
            "query_inversion_sum_primary_plus_inverted": primary + inverted,
            "relation_orientation_half_difference": relation_orientation,
            "carrier_off_orientation_half_difference": carrier_off_orientation,
            "net_relation_minus_carrier_off_orientation": net_orientation,
            "carrier_off_abs_to_primary_abs": abs(carrier_off) / abs(primary) if primary else None,
            "carrier_off_orientation_abs_to_relation_orientation_abs": (
                abs(carrier_off_orientation) / abs(relation_orientation) if relation_orientation else None
            ),
            "net_orientation_to_relation_orientation": net_orientation / relation_orientation if relation_orientation else None,
            "sham_abs_to_primary_abs": abs(sham) / abs(primary) if primary else None,
        },
        "interpretation": {
            "query_inversion_reverses_primary": primary > 0 and inverted < 0,
            "carrier_off_absolute_below_25pct_primary": primary > 0 and abs(carrier_off) <= 0.25 * abs(primary),
            "carrier_off_orientation_below_25pct_relation_orientation": (
                relation_orientation > 0 and abs(carrier_off_orientation) <= 0.25 * abs(relation_orientation)
            ),
            "net_orientation_remains_positive": net_orientation > 0,
            "relation_sham_below_25pct_primary": primary > 0 and abs(sham) <= 0.25 * abs(primary),
            "exploratory_only": True,
            "small_wall_crossed": False,
        },
        "claim_boundary": {
            "positive_scientific_claim": False,
            "small_wall_crossed": False,
            "full_tomography_established": False,
            "r2_restoration_established": False,
        },
    }
    result["analysis_sha256"] = digest({k: v for k, v in result.items() if k != "analysis_sha256"})
    write_json(RUN_ROOT / "RELATION_ORIENTATION_BALANCE_DISCOVERY_ANALYSIS.json", result)
    write_json(SUMMARY_JSON, result)
    lines = [
        "# Relation Orientation Balance Discovery",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "| Variant | R_spatial mean | abs/primary | Blocks |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        mean_value = reports[variant]["R_spatial"]["mean"]
        ratio = abs(mean_value) / abs(primary) if primary else 0.0
        lines.append(f"| `{variant}` | {mean_value:.9f} | {ratio:.3f} | {reports[variant]['block_count']} |")
    lines.extend(
        [
            "",
            "Discovery interpretation:",
            f"- query inversion reverses primary: `{result['interpretation']['query_inversion_reverses_primary']}`",
            f"- carrier-off absolute below 0.25 x primary: `{result['interpretation']['carrier_off_absolute_below_25pct_primary']}`",
            f"- carrier-off orientation below 0.25 x relation orientation: `{result['interpretation']['carrier_off_orientation_below_25pct_relation_orientation']}`",
            f"- net relation-minus-carrier-off orientation remains positive: `{result['interpretation']['net_orientation_remains_positive']}`",
            f"- relation-sham below 0.25 x primary: `{result['interpretation']['relation_sham_below_25pct_primary']}`",
            "",
            "This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare() -> dict[str, Any]:
    copy_segmented_source()
    patch_runtime_for_transform_kill()
    schedule_manifest = make_rows()
    build = compile_runtime()
    result = {
        "schema": "FAMILY10H_RELATION_ORIENTATION_BALANCE_PREPARE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    write_json(HERE / "RELATION_ORIENTATION_BALANCE_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    require(prepare_result["passed"], "prepare failed")
    package = build_package()
    controller = deploy_execute_copyback(package)
    analysis = analyze_archive(controller)
    print(
        json.dumps(
            {
                "controller_passed": controller["passed"],
                "archive_sha256": controller["archive_sha256"],
                "analysis_sha256": analysis["analysis_sha256"],
                "coordinates": analysis["coordinates"],
                "interpretation": analysis["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if controller["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
