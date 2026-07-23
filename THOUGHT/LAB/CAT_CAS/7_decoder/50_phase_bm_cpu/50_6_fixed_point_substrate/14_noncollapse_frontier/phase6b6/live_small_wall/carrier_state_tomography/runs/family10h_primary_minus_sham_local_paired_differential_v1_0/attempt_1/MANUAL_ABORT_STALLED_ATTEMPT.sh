#!/usr/bin/env bash
set +e

ROOT="/root/catcas_live_small_wall/family10h_primary_minus_sham_local_paired_differential_v1_0/source"
cd "$ROOT" || exit 90

mkdir -p official_logs official_outputs

created_utc="$(date -u +%FT%TZ)"
cat > official_logs/manual_abort_due_to_runtime_stall.txt <<EOF
schema=FAMILY10H_LOCAL_PAIRED_MANUAL_ABORT_STALLED_ATTEMPT_V1
created_utc=$created_utc
reason=relation_spatial_runtime_stalled_before_first_raw_row
attempt_consumed=true
action=terminate_stalled_runtime_restore_cpufreq_preserve_partial_evidence
small_wall_crossed=false
EOF

ps -eo pid,ppid,stat,comm,args > official_logs/manual_abort_ps_before.txt

write_counts() {
  variant="$1"
  output="official_outputs/$variant"
  {
    echo "raw_records=$(test -f "$output/raw_records.jsonl" && wc -l < "$output/raw_records.jsonl" || echo 0)"
    echo "pair_observations=$(test -f "$output/pair_observations.jsonl" && wc -l < "$output/pair_observations.jsonl" || echo 0)"
    echo "source_death_receipts=$(test -f "$output/source_death_receipts.jsonl" && wc -l < "$output/source_death_receipts.jsonl" || echo 0)"
  } > "official_logs/${variant}.counts"
}

variants="
round0_query_relation_pair
round0_relation_sham
round0_scrambled_pair_control
round0_distance_matched_control
round0_route_pressure_control
round1_query_relation_pair
round1_relation_sham
round1_scrambled_pair_control
round1_distance_matched_control
round1_route_pressure_control
"

for variant in $variants; do
  write_counts "$variant"
done

if [ ! -f official_logs/round0_query_relation_pair.rc ]; then
  printf '%s\n' "124" > official_logs/round0_query_relation_pair.rc
fi
printf '%s\n' "manual_abort_due_to_runtime_stall" >> official_logs/round0_query_relation_pair.stderr

runtime_pids="$(ps -eo pid=,args= | awk -v root="$ROOT" 'index($0, root "/relation_spatial_runtime --execute-schedule") {print $1}')"
script_pids="$(ps -eo pid=,args= | awk 'index($0, "OFFICIAL_TARGET_EXECUTE_LOCAL_PAIRED.sh") {print $1}')"

{
  echo "runtime_pids=$runtime_pids"
  echo "script_pids=$script_pids"
} > official_logs/manual_abort_identified_pids.txt

if [ -n "$script_pids" ]; then
  kill -STOP $script_pids 2>/dev/null
fi

if [ -n "$runtime_pids" ]; then
  kill -TERM $runtime_pids 2>/dev/null
  sleep 5
  still_running="$(ps -eo pid=,args= | awk -v root="$ROOT" 'index($0, root "/relation_spatial_runtime --execute-schedule") {print $1}')"
  if [ -n "$still_running" ]; then
    kill -KILL $still_running 2>/dev/null
  fi
fi

ps -eo pid,ppid,stat,comm,args > official_logs/manual_abort_ps_after_runtime_kill.txt

python3 - <<'PY'
import json
import pathlib
import sys
import time

prepare_path = pathlib.Path("official_logs/cpufreq_prepare.json")
receipt = {
    "schema": "FAMILY10H_LOCAL_PAIRED_CPUFREQ_RESTORE_V1",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "restore_actor": "manual_abort_stalled_attempt",
    "cpus": {},
    "failures": [],
}
try:
    prepare = json.loads(prepare_path.read_text())
except Exception as exc:
    receipt["failures"].append(f"prepare receipt unavailable: {exc}")
    prepare = {"cpus": {}}

for cpu, item in prepare.get("cpus", {}).items():
    path = pathlib.Path(item["governor_path"])
    target = item["before_governor"]
    try:
        current = path.read_text().strip()
        path.write_text(target + "\n")
        after = path.read_text().strip()
        receipt["cpus"][cpu] = {
            "governor_path": str(path),
            "prepared_governor": item.get("prepared_governor"),
            "restore_target_governor": target,
            "governor_before_restore": current,
            "governor_after_restore": after,
            "restored": after == target,
        }
        if after != target:
            receipt["failures"].append(f"cpu{cpu} governor restore mismatch: {after} != {target}")
    except Exception as exc:
        receipt["failures"].append(f"cpu{cpu} governor restore failed: {exc}")

receipt["passed"] = not receipt["failures"] and all(item.get("restored") for item in receipt["cpus"].values())
pathlib.Path("official_logs/cpufreq_restore.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
pathlib.Path("official_logs/manual_cpufreq_restore.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
sys.exit(0 if receipt["passed"] else 1)
PY
restore_rc="$?"
printf '%s\n' "$restore_rc" > official_logs/cpufreq_restore.rc

python3 - <<'PY'
import hashlib
import json
import pathlib
import time

root = pathlib.Path(".")
variants = [
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

def sha(path: pathlib.Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def line_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.rstrip(b"\r\n"))

def info(path: pathlib.Path, count: bool = False):
    if not path.exists():
        return {"present": False, "size_bytes": 0, "sha256": None, "line_count": 0 if count else None}
    return {
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
        "line_count": line_count(path) if count else None,
    }

def read_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"passed": False, "parse_error": str(exc)}

summary = {
    "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_OFFICIAL_TARGET_RUN_SUMMARY_V1",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "transaction_run_id": "family10h_primary_minus_sham_local_paired_differential_v1_0",
    "segmented_transaction_run_id": "family10h_relation_spatial_pair_readout_v1_1_segmented_0",
    "runtime_authority": "family10h_relation_spatial_pair_readout_v1_0",
    "attempt_consumed": (root / "ATTEMPT_CONSUMED").exists(),
    "manual_abort": {
        "reason": "relation_spatial_runtime_stalled_before_first_raw_row",
        "round0_query_relation_pair_rc": 124,
        "small_wall_crossed": False,
    },
    "target_preflight": read_json(root / "official_logs" / "target_preflight.json"),
    "target_preflight_rc": int((root / "official_logs" / "target_preflight.rc").read_text().strip()) if (root / "official_logs" / "target_preflight.rc").exists() else None,
    "cpufreq_prepare": read_json(root / "official_logs" / "cpufreq_prepare.json"),
    "cpufreq_restore": read_json(root / "official_logs" / "cpufreq_restore.json"),
    "variants": {},
    "small_wall_crossed": False,
}

for variant in variants:
    out = root / "official_outputs" / variant
    rc_path = root / "official_logs" / f"{variant}.rc"
    counts_path = root / "official_logs" / f"{variant}.counts"
    summary["variants"][variant] = {
        "rc": int(rc_path.read_text().strip()) if rc_path.exists() and rc_path.read_text().strip().lstrip("-").isdigit() else None,
        "counts_text": counts_path.read_text() if counts_path.exists() else "",
        "raw_records.jsonl": info(out / "raw_records.jsonl", True),
        "pair_observations.jsonl": info(out / "pair_observations.jsonl", True),
        "source_death_receipts.jsonl": info(out / "source_death_receipts.jsonl", True),
        "feature_freeze.json": info(out / "feature_freeze.json"),
        "target_execution_receipt.json": info(out / "target_execution_receipt.json"),
    }

summary["all_variant_rc_zero"] = all(v["rc"] == 0 for v in summary["variants"].values()) if summary["variants"] else False
summary["raw_record_count_total"] = sum(v["raw_records.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["pair_observation_count_total"] = sum(v["pair_observations.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["source_death_receipt_count_total"] = sum(v["source_death_receipts.jsonl"]["line_count"] or 0 for v in summary["variants"].values())
summary["passed"] = False
summary["summary_sha256"] = hashlib.sha256(
    json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
(root / "OFFICIAL_TARGET_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$(date -u +%FT%TZ)" > OFFICIAL_TARGET_ENDED_UTC.txt
ps -eo pid,ppid,stat,comm,args > official_logs/manual_abort_ps_before_script_kill.txt

if [ -n "$script_pids" ]; then
  kill -KILL $script_pids 2>/dev/null
fi

exit 0
