#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
BASE_PATH = (
    CARRIER_ROOT
    / "family10h_relation_offset2_signed_projection_source_off_screen_discovery_v1"
    / "run_relation_offset2_signed_projection_source_off_screen_discovery.py"
)

spec = importlib.util.spec_from_file_location("offset2_source_off_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load base discovery runner: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
BASE_TARGET_SCRIPT = base.target_script


RUN_ID = "family10h_relation_offset2_handoff_decay_discovery_v1_0"
VARIANTS = [
    "alive_offset2_signed",
    "source_off_no_prep_offset2_signed",
    "dead_after_exit_0ns_offset2_signed",
    "dead_after_exit_10us_offset2_signed",
    "dead_after_exit_100us_offset2_signed",
    "dead_after_exit_1ms_offset2_signed",
    "dead_after_exit_10ms_offset2_signed",
]
QUERY_BY_VARIANT = {
    "alive_offset2_signed": "source_on_offset_2_control",
    "source_off_no_prep_offset2_signed": "source_off_offset_2_control",
    "dead_after_exit_0ns_offset2_signed": "dead_offset_2_control",
    "dead_after_exit_10us_offset2_signed": "dead_offset_2_control",
    "dead_after_exit_100us_offset2_signed": "dead_offset_2_control",
    "dead_after_exit_1ms_offset2_signed": "dead_offset_2_control",
    "dead_after_exit_10ms_offset2_signed": "dead_offset_2_control",
}
LIFECYCLE_BY_VARIANT = {
    "alive_offset2_signed": "alive_during_query",
    "source_off_no_prep_offset2_signed": "source_off_no_preparation",
    "dead_after_exit_0ns_offset2_signed": "prepared_source_exited_before_query_delay_0ns",
    "dead_after_exit_10us_offset2_signed": "prepared_source_exited_before_query_delay_10000ns",
    "dead_after_exit_100us_offset2_signed": "prepared_source_exited_before_query_delay_100000ns",
    "dead_after_exit_1ms_offset2_signed": "prepared_source_exited_before_query_delay_1000000ns",
    "dead_after_exit_10ms_offset2_signed": "prepared_source_exited_before_query_delay_10000000ns",
}
DELAY_NS_BY_VARIANT = {
    "alive_offset2_signed": 0,
    "source_off_no_prep_offset2_signed": 0,
    "dead_after_exit_0ns_offset2_signed": 0,
    "dead_after_exit_10us_offset2_signed": 10_000,
    "dead_after_exit_100us_offset2_signed": 100_000,
    "dead_after_exit_1ms_offset2_signed": 1_000_000,
    "dead_after_exit_10ms_offset2_signed": 10_000_000,
}
ROWS_PER_VARIANT = 512
PAIR_SAMPLE_COUNT = base.PAIR_SAMPLE_COUNT
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT

SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "OFFSET2_HANDOFF_DECAY_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_OFFSET2_HANDOFF_DECAY_DISCOVERY_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_OFFSET2_HANDOFF_DECAY_DISCOVERY_SUMMARY.md"


def configure_base() -> None:
    base.RUN_ID = RUN_ID
    base.REMOTE_ROOT = f"{base.REMOTE_BASE}/{RUN_ID}"
    base.REMOTE_PACKAGE = f"{base.REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
    base.REMOTE_ARCHIVE = f"{base.REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
    base.OWNER_MARKER = f".{RUN_ID}_owner"
    base.VARIANTS = VARIANTS
    base.QUERY_BY_VARIANT = QUERY_BY_VARIANT
    base.LIFECYCLE_BY_VARIANT = LIFECYCLE_BY_VARIANT
    base.ROWS_PER_VARIANT = ROWS_PER_VARIANT
    base.EXPECTED_PAIRS_PER_VARIANT = EXPECTED_PAIRS_PER_VARIANT
    base.HERE = HERE
    base.SOURCE_ROOT = SOURCE_ROOT
    base.SCHEDULE_DIR = SCHEDULE_DIR
    base.RUN_ROOT = RUN_ROOT
    base.LOCAL_PACKAGE = LOCAL_PACKAGE
    base.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    base.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    base.SUMMARY_JSON = SUMMARY_JSON
    base.SUMMARY_MD = SUMMARY_MD


def patch_runtime_for_handoff_decay() -> None:
    base.patch_runtime_for_offset2_signed_projection_source_off_screen()
    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")

    helper_anchor = "static uint64_t monotonic_ns(void) {\n"
    helper_end = "}\n\nstatic int current_cpu_checked"
    sleep_helper = (
        "static void sleep_ns_interval(uint64_t delay_ns) {\n"
        "    struct timespec req;\n"
        "    if (delay_ns == 0u) {\n"
        "        return;\n"
        "    }\n"
        "    req.tv_sec = (time_t)(delay_ns / UINT64_C(1000000000));\n"
        "    req.tv_nsec = (long)(delay_ns % UINT64_C(1000000000));\n"
        "    while (nanosleep(&req, &req) != 0 && errno == EINTR) {\n"
        "    }\n"
        "}\n\n"
    )
    base.require(helper_anchor in text and helper_end in text, "monotonic helper anchors not found")
    text = text.replace(helper_end, "}\n\n" + sleep_helper + "static int current_cpu_checked", 1)

    child_prep_block = (
        "                relation_spatial_preparation prep;\n"
        "                if (!pin_to_core(row.source_cpu_expected)) {\n"
        "                    _exit(11);\n"
        "                }\n"
        "                shared->source_cpu_before = current_cpu_checked();\n"
        "                prep.bank_a_work = row.bank_a_work;\n"
        "                prep.bank_b_work = row.bank_b_work;\n"
        "                prep.relation = row.r_prepare;\n"
        "                prep.source_order = row.source_order;\n"
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
    )
    child_prep_replacement = (
        "                if (!pin_to_core(row.source_cpu_expected)) {\n"
        "                    _exit(11);\n"
        "                }\n"
        "                shared->source_cpu_before = current_cpu_checked();\n"
        "                if (source_off_control(row.control)) {\n"
        "                    relation_spatial_prefault(&shared->state);\n"
        "                    flush_state_lines(&shared->state);\n"
        "                    shared->preparation_ok = 1;\n"
        "                } else {\n"
        "                    relation_spatial_preparation prep;\n"
        "                    prep.bank_a_work = row.bank_a_work;\n"
        "                    prep.bank_b_work = row.bank_b_work;\n"
        "                    prep.relation = row.r_prepare;\n"
        "                    prep.source_order = row.source_order;\n"
        "                    prep.cyclic_origin = row.cyclic_origin;\n"
        "                    shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "                }\n"
    )
    count = text.count(child_prep_block)
    base.require(count == 1, f"physical child prep block replacement count unexpected: {count}")
    text = text.replace(child_prep_block, child_prep_replacement, 1)

    exit_block = (
        "                source_exit_ns = monotonic_ns();\n"
        "                child = -1;\n"
        "                source_alive_at_pair_measurement = 0;\n"
        "                if (reset_double_flush_after_source_dead_control(row.control)) {\n"
    )
    exit_replacement = (
        "                source_exit_ns = monotonic_ns();\n"
        "                child = -1;\n"
        "                source_alive_at_pair_measurement = 0;\n"
        "                if (!source_off_control(row.control)) {\n"
        "                    sleep_ns_interval(row.delay_ns);\n"
        "                }\n"
        "                if (reset_double_flush_after_source_dead_control(row.control)) {\n"
    )
    count = text.count(exit_block)
    base.require(count == 1, f"source-exit delay block replacement count unexpected: {count}")
    text = text.replace(exit_block, exit_replacement, 1)
    runtime.write_text(text, encoding="utf-8", newline="\n")


def make_rows() -> dict[str, Any]:
    source_schedule = base.LOCAL_PAIRED_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES" / "round0_query_relation_pair.tsv"
    with source_schedule.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        base_rows = list(reader)[:ROWS_PER_VARIANT]
    base.require(len(base_rows) == ROWS_PER_VARIANT, "not enough base rows for discovery schedule")
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    schedules: dict[str, Any] = {}
    for variant in VARIANTS:
        query = QUERY_BY_VARIANT[variant]
        rows = []
        for index, row in enumerate(base_rows):
            copied = dict(row)
            copied["execution_ordinal"] = str(index)
            copied["delay_ns"] = str(DELAY_NS_BY_VARIANT[variant])
            copied["query"] = query
            copied["operation_semantics_id"] = query
            copied["control_semantics_id"] = "none" if variant == "alive_offset2_signed" else query
            copied["tuple_id"] = (
                f"{RUN_ID}:{variant}:{index:06d}:"
                f"{base.sha256_bytes((variant + ':' + str(index)).encode())[:16]}"
            )
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
            "source_lifetime": LIFECYCLE_BY_VARIANT[variant],
            "post_source_exit_delay_ns": DELAY_NS_BY_VARIANT[variant],
            "path": f"OFFSET2_HANDOFF_DECAY_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "sha256": base.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_DISCOVERY_SCHEDULE_MANIFEST_V1",
        "run_id": RUN_ID,
        "source_schedule": str(source_schedule),
        "source_schedule_sha256": base.sha256_file(source_schedule),
        "rows_per_variant": ROWS_PER_VARIANT,
        "pair_sample_count": PAIR_SAMPLE_COUNT,
        "variants": schedules,
        "claim_boundary": {
            "exploratory_only": True,
            "positive_scientific_claim": False,
            "small_wall_crossed": False,
        },
        "mechanism_question": (
            "Does the signed offset-2 source-written projection survive for a short post-source-exit "
            "handoff interval, while a true no-preparation source-off baseline remains killed?"
        ),
        "receiver_projection": "matrix contrast of mean(B_first_touch_cycles - A_first_touch_cycles) at signed offset 2",
    }
    manifest["schedule_manifest_sha256"] = base.digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    base.write_json(SOURCE_ROOT / "RELATION_OFFSET2_HANDOFF_DECAY_SCHEDULE_MANIFEST.json", manifest)
    base.write_json(HERE / "RELATION_OFFSET2_HANDOFF_DECAY_SCHEDULE_MANIFEST.json", manifest)
    return manifest


def target_script() -> str:
    script = BASE_TARGET_SCRIPT()
    return (
        script.replace("OFFSET2_SIGNED_PROJECTION_SOURCE_OFF_SCREEN_SCHEDULES", "OFFSET2_HANDOFF_DECAY_SCHEDULES")
        .replace("SOURCE_OFF_SCREEN", "HANDOFF_DECAY")
        .replace("source-off-screen", "handoff-decay")
    )


def build_package() -> dict[str, Any]:
    base.target_script = target_script
    try:
        receipt = base.build_package()
    finally:
        base.target_script = BASE_TARGET_SCRIPT
    receipt["schema"] = "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_SOURCE_PACKAGE_V1"
    base.write_json(HERE / "SOURCE_PACKAGE_RECEIPT.json", receipt)
    return receipt


def deploy_execute_copyback(package: dict[str, Any]) -> dict[str, Any]:
    original_target_script = "RUN_OFFSET2_SIGNED_PROJECTION_SOURCE_OFF_SCREEN_DISCOVERY.sh"
    new_target_script = original_target_script
    # Reuse the proven deploy/copyback path, changing only the script name in-place.
    original = base.deploy_execute_copyback

    def patched_deploy(pkg: dict[str, Any]) -> dict[str, Any]:
        text = Path(base.__file__).read_text(encoding="utf-8")
        if original_target_script not in text:
            return original(pkg)
        return original(pkg)

    _ = patched_deploy
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCAL_ARCHIVE.exists():
        raise RuntimeError(f"local archive already exists: {LOCAL_ARCHIVE}")
    absence = base.run_remote(f"set -eu; test ! -e {base.REMOTE_ROOT}; test ! -e {base.REMOTE_ARCHIVE}; test ! -e {base.REMOTE_PACKAGE}", timeout=30)
    base.require(absence["returncode"] == 0, f"remote path not fresh: {absence['stdout']} {absence['stderr']}")
    create = base.run_remote(f"set -eu; mkdir -p {base.REMOTE_BASE}; mkdir -m 0700 {base.REMOTE_ROOT}; printf '%s\\n' '{RUN_ID}' > {base.REMOTE_ROOT}/{base.OWNER_MARKER}", timeout=30)
    base.require(create["returncode"] == 0, "remote root create failed")
    upload = base.run_scp(str(LOCAL_PACKAGE), f"{base.TARGET_HOST}:{base.REMOTE_PACKAGE}", timeout=240)
    base.require(upload["returncode"] == 0, f"upload failed: {upload['stderr']}")
    extract = base.run_remote(
        f"set -eu; tar -xzf {base.REMOTE_PACKAGE} -C {base.REMOTE_ROOT}; "
        f"chmod +x {base.REMOTE_ROOT}/source/relation_spatial_runtime "
        f"{base.REMOTE_ROOT}/source/relation_spatial_pmu_preflight "
        f"{base.REMOTE_ROOT}/source/{new_target_script}",
        timeout=120,
    )
    base.require(extract["returncode"] == 0, f"extract failed: {extract['stderr']}")
    live = base.run_remote(f"set +e; cd {base.REMOTE_ROOT}/source; ./{new_target_script}", timeout=3600)
    (RUN_ROOT / "DISCOVERY_TARGET_STDOUT.txt").write_text(live["stdout"], encoding="utf-8")
    (RUN_ROOT / "DISCOVERY_TARGET_STDERR.txt").write_text(live["stderr"], encoding="utf-8")
    make_archive = base.run_remote(
        f"set -eu; rm -f {base.REMOTE_ARCHIVE}; tar -czf {base.REMOTE_ARCHIVE} -C {base.REMOTE_ROOT} source; "
        f"sha256sum {base.REMOTE_ARCHIVE}; stat -c%s {base.REMOTE_ARCHIVE}",
        timeout=900,
    )
    base.require(make_archive["returncode"] == 0, f"archive failed: {make_archive['stderr']}")
    if LOCAL_TMP_ARCHIVE.exists():
        LOCAL_TMP_ARCHIVE.unlink()
    copy = base.run_scp(f"{base.TARGET_HOST}:{base.REMOTE_ARCHIVE}", str(LOCAL_TMP_ARCHIVE), timeout=900)
    base.require(copy["returncode"] == 0, f"copyback failed: {copy['stderr']}")
    import shutil

    shutil.move(str(LOCAL_TMP_ARCHIVE), str(LOCAL_ARCHIVE))
    lines = [line.strip() for line in make_archive["stdout"].splitlines() if line.strip()]
    remote_sha = lines[0].split()[0] if lines else None
    remote_size = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else None
    local_sha = base.sha256_file(LOCAL_ARCHIVE)
    local_size = LOCAL_ARCHIVE.stat().st_size
    copyback = {
        "schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_COPYBACK_V1",
        "created_at": base.utc_now(),
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
        cleanup_cmd = base.run_remote(
            f"set -eu; test -f {base.REMOTE_ROOT}/{base.OWNER_MARKER}; grep -qx '{RUN_ID}' {base.REMOTE_ROOT}/{base.OWNER_MARKER}; "
            f"rm -rf {base.REMOTE_ROOT} {base.REMOTE_ARCHIVE} {base.REMOTE_PACKAGE}; "
            f"test ! -e {base.REMOTE_ROOT}; test ! -e {base.REMOTE_ARCHIVE}; test ! -e {base.REMOTE_PACKAGE}",
            timeout=240,
        )
        cleanup = {"schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_REMOTE_CLEANUP_V1", "cleanup": cleanup_cmd, "passed": cleanup_cmd["returncode"] == 0}
    receipt = {
        "schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_LIVE_CONTROLLER_V1",
        "created_at": base.utc_now(),
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
    base.write_json(RUN_ROOT / "DISCOVERY_CONTROLLER_RESULT.json", receipt)
    return receipt


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_archive(controller: dict[str, Any]) -> dict[str, Any]:
    base.require(LOCAL_ARCHIVE.exists(), "archive missing for analysis")
    reports: dict[str, Any] = {}
    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as tf:
        names = set(tf.getnames())
        for variant in VARIANTS:
            prefix = f"source/discovery_outputs/{variant}"
            for suffix in ["raw_records.jsonl", "pair_observations.jsonl", "source_death_receipts.jsonl"]:
                base.require(f"{prefix}/{suffix}" in names, f"missing archive member {prefix}/{suffix}")
            raw = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/raw_records.jsonl").read())  # type: ignore[union-attr]
            pairs = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/pair_observations.jsonl").read())  # type: ignore[union-attr]
            deaths = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/source_death_receipts.jsonl").read())  # type: ignore[union-attr]
            base.require(len(raw) == ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            base.require(len(pairs) == EXPECTED_PAIRS_PER_VARIANT, f"{variant} pair count mismatch: {len(pairs)}")
            base.require(len(deaths) == ROWS_PER_VARIANT, f"{variant} source-death count mismatch: {len(deaths)}")
            rows = base.row_projection_pairs(raw, pairs)
            records = base.relation_matrix_records(rows, "mean_b_minus_a")
            records_by_variant[variant] = records
            values = [record["relation_matrix_contrast"] for record in records]
            delays = [max(0, int(death["query_start_monotonic_ns"]) - int(death["source_exit_monotonic_ns"])) for death in deaths if not death.get("source_alive_at_pair_measurement")]
            reports[variant] = {
                "query": QUERY_BY_VARIANT[variant],
                "source_lifetime": LIFECYCLE_BY_VARIANT[variant],
                "scheduled_post_source_exit_delay_ns": DELAY_NS_BY_VARIANT[variant],
                "observed_post_source_exit_delay_ns": base.distribution([float(value) for value in delays]),
                "raw_record_count": len(raw),
                "pair_observation_count": len(pairs),
                "source_death_receipt_count": len(deaths),
                "block_count": len(records),
                "C_offset2_signed": base.distribution(values),
                "source_alive_at_pair_measurement_counts": dict(Counter(str(death.get("source_alive_at_pair_measurement")) for death in deaths)),
                "source_alive_during_query_counts": dict(Counter(str(death.get("source_alive_during_query")) for death in deaths)),
                "source_lifetime_counts": dict(Counter(str(death.get("source_lifetime")) for death in deaths)),
                "process_custody_counts": dict(Counter(str(death.get("process_custody")) for death in deaths)),
            }

    factors = ["session", "replicate", "mapping", "source_order", "query_order", "cyclic_origin"]

    def value_for(variant: str) -> float:
        return reports[variant]["C_offset2_signed"]["mean"]

    def by_factor(variant: str, factor: str) -> dict[str, float]:
        records = records_by_variant[variant]
        levels = sorted(set(str(record[factor]) for record in records), key=str)
        return {
            level: mean([record["relation_matrix_contrast"] for record in records if str(record[factor]) == level])
            for level in levels
        }

    alive = value_for("alive_offset2_signed")
    source_off = value_for("source_off_no_prep_offset2_signed")
    alive_by_factor = {factor: by_factor("alive_offset2_signed", factor) for factor in factors}
    stratum_reports: dict[str, Any] = {}
    for variant in VARIANTS:
        factor_reports = {}
        ratios: list[float] = []
        values: list[float] = []
        for factor in factors:
            variant_values = by_factor(variant, factor)
            levels = sorted(set(variant_values) & set(alive_by_factor[factor]), key=str)
            level_reports = {}
            for level in levels:
                value = variant_values[level]
                alive_value = alive_by_factor[factor][level]
                ratio = abs(value) / abs(alive_value) if alive_value else None
                if ratio is not None:
                    ratios.append(ratio)
                values.append(value)
                level_reports[level] = {
                    "C_offset2_signed": value,
                    "alive_matched_C_offset2_signed": alive_value,
                    "abs_to_alive_matched_abs": ratio,
                    "same_sign_as_alive_aggregate": (value > 0) == (alive > 0) if alive else False,
                    "preserves_at_least_25pct_alive_matched": ratio is not None and ratio >= 0.25,
                    "below_25pct_alive_matched": ratio is not None and ratio <= 0.25,
                }
            factor_reports[factor] = {
                "levels": level_reports,
                "all_same_sign_as_alive_aggregate": all(item["same_sign_as_alive_aggregate"] for item in level_reports.values()),
                "all_preserve_25pct_alive_matched": all(item["preserves_at_least_25pct_alive_matched"] for item in level_reports.values()),
                "all_below_25pct_alive_matched": all(item["below_25pct_alive_matched"] for item in level_reports.values()),
            }
        stratum_reports[variant] = {
            "factors": factor_reports,
            "all_same_sign_as_alive_aggregate": all((value > 0) == (alive > 0) for value in values) if values and alive else False,
            "all_one_factor_strata_preserve_25pct_alive_matched": all(ratio >= 0.25 for ratio in ratios) if ratios else False,
            "all_one_factor_strata_below_25pct_alive_matched": all(ratio <= 0.25 for ratio in ratios) if ratios else False,
            "max_abs_to_alive_matched_abs": max(ratios) if ratios else None,
            "min_abs_to_alive_matched_abs": min(ratios) if ratios else None,
        }

    def ratio_to_alive(value: float) -> float | None:
        return abs(value) / abs(alive) if alive else None

    coordinates = {"C_alive_offset2_signed": alive, "C_source_off_no_prep_offset2_signed": source_off}
    decay_curve = []
    for variant in VARIANTS:
        value = value_for(variant)
        coordinates[f"C_{variant}"] = value
        coordinates[f"{variant}_abs_to_alive_abs"] = ratio_to_alive(value)
        decay_curve.append(
            {
                "variant": variant,
                "scheduled_post_source_exit_delay_ns": DELAY_NS_BY_VARIANT[variant],
                "C_offset2_signed": value,
                "abs_to_alive_abs": ratio_to_alive(value),
                "all_one_factor_strata_same_sign": stratum_reports[variant]["all_same_sign_as_alive_aggregate"],
                "all_one_factor_strata_preserve_25pct_alive": stratum_reports[variant]["all_one_factor_strata_preserve_25pct_alive_matched"],
            }
        )

    candidate_variants = [
        item["variant"]
        for item in decay_curve
        if item["variant"].startswith("dead_after_exit")
        and item["abs_to_alive_abs"] is not None
        and item["abs_to_alive_abs"] >= 0.25
        and item["all_one_factor_strata_same_sign"]
    ]
    result = {
        "schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_DISCOVERY_ANALYSIS_V1",
        "created_at": base.utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": base.sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "receiver_projection": "relation-matrix contrast of per-row mean(B_first_touch_cycles - A_first_touch_cycles) at signed offset 2",
        "variant_reports": reports,
        "stratum_reports": stratum_reports,
        "coordinates": coordinates,
        "decay_curve": decay_curve,
        "interpretation": {
            "alive_offset2_signed_nonzero": alive != 0.0,
            "true_source_off_no_prep_collapses_below_25pct_alive": abs(source_off) <= 0.25 * abs(alive) if alive else False,
            "true_source_off_no_prep_all_one_factor_strata_below_25pct_alive": stratum_reports["source_off_no_prep_offset2_signed"]["all_one_factor_strata_below_25pct_alive_matched"],
            "post_source_exit_handoff_candidate_variants": candidate_variants,
            "post_source_exit_any_candidate": bool(candidate_variants),
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
    result["analysis_sha256"] = base.digest({k: v for k, v in result.items() if k != "analysis_sha256"})
    base.write_json(RUN_ROOT / "RELATION_OFFSET2_HANDOFF_DECAY_DISCOVERY_ANALYSIS.json", result)
    base.write_json(SUMMARY_JSON, result)
    lines = [
        "# Offset-2 Handoff Decay Discovery",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 2.",
        "",
        "| Variant | Source lifetime | Delay ns | C offset2 signed | abs/abs(alive) | one-factor same sign |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in decay_curve:
        lines.append(
            f"| {item['variant']} | {LIFECYCLE_BY_VARIANT[item['variant']]} | {item['scheduled_post_source_exit_delay_ns']} | "
            f"{item['C_offset2_signed']:.9f} | {((item['abs_to_alive_abs'] or 0.0)):.3f} | "
            f"`{item['all_one_factor_strata_same_sign']}` |"
        )
    lines.extend(
        [
            "",
            "Discovery interpretation:",
            f"- true source-off no-prep collapses below 0.25 x alive: `{result['interpretation']['true_source_off_no_prep_collapses_below_25pct_alive']}`",
            f"- true source-off no-prep all one-factor strata below 0.25 x alive: `{result['interpretation']['true_source_off_no_prep_all_one_factor_strata_below_25pct_alive']}`",
            f"- post-source-exit handoff candidate variants: `{candidate_variants}`",
            "",
            "This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare() -> dict[str, Any]:
    configure_base()
    base.copy_segmented_source()
    patch_runtime_for_handoff_decay()
    schedule_manifest = make_rows()
    build = base.compile_runtime()
    result = {
        "schema": "FAMILY10H_RELATION_OFFSET2_HANDOFF_DECAY_PREPARE_V1",
        "created_at": base.utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    base.write_json(HERE / "RELATION_OFFSET2_HANDOFF_DECAY_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_base()
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    base.require(prepare_result["passed"], "prepare failed")
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
