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


RUN_ID = "family10h_relation_offset16_origin2048_ab_lifecycle_probe_v1_0"
TARGET_HOST = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
REMOTE_ROOT = f"{REMOTE_BASE}/{RUN_ID}"
REMOTE_PACKAGE = f"{REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
REMOTE_ARCHIVE = f"{REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
OWNER_MARKER = f".{RUN_ID}_owner"
RUNTIME_AUTHORITY_VALUE = "family10h_relation_spatial_pair_readout_v1_0"

OFFSETS = [16]
FOCUS_OFFSET = 16
FOCUS_CYCLIC_ORIGIN = "2048"
FOCUS_QUERY_ORDER = "AB"
VARIANTS = [
    "alive_offset16_signed",
    "source_off_offset16_signed",
    "dead_offset16_signed",
    "reset_double_flush_offset16_signed",
]
QUERY_BY_VARIANT = {
    "alive_offset16_signed": "source_on_offset_16_control",
    "source_off_offset16_signed": "source_off_offset_16_control",
    "dead_offset16_signed": "dead_offset_16_control",
    "reset_double_flush_offset16_signed": "reset_double_flush_offset_16_control",
}
LIFECYCLE_BY_VARIANT = {
    "alive_offset16_signed": "alive",
    "source_off_offset16_signed": "source_off",
    "dead_offset16_signed": "dead",
    "reset_double_flush_offset16_signed": "reset_double_flush",
}
OFFSET_BY_VARIANT = {
    variant: FOCUS_OFFSET for variant in VARIANTS
}
ROWS_PER_VARIANT = 480
PAIR_SAMPLE_COUNT = 256
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT
MATCHED_PERMUTATION_COUNT = 63
MATCHED_PERMUTATION_SEED = "family10h-offset16-origin2048-ab-lifecycle-probe-v1"

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
SEGMENTED_ROOT = CARRIER_ROOT / "family10h_relation_spatial_pair_readout_v1_1_segmented"
LOCAL_PAIRED_ROOT = CARRIER_ROOT / "family10h_primary_minus_sham_local_paired_differential_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SUMMARY.md"


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


def patch_runtime_for_offset_signed_projection_source_off_screen() -> None:
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    text = header.read_text(encoding="utf-8")
    text = text.replace(
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4",
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1 = 5,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2 = 6,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2 = 7,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2 = 8,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2 = 9,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4 = 10,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4 = 11,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8 = 12,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8 = 13,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16 = 14,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16 = 15,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024 = 16,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024 = 17,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16 = 18,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16 = 19",
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
        '    if (strcmp(query, "source_off_offset_1_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_on_offset_2_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_offset_2_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_offset_2_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_double_flush_offset_2_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_on_offset_4_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_offset_4_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_on_offset_8_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_offset_8_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_on_offset_16_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_offset_16_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_offset_16_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_double_flush_offset_16_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_on_offset_1024_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_offset_1024_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024;\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n",
    )
    offset_helper_anchor = (
        "static uint32_t control_b_index(const relation_spatial_schedule_row *row, uint32_t a_index, uint32_t sample_index) {\n"
    )
    offset_helper = (
        "static uint32_t signed_offset_index(const relation_spatial_schedule_row *row, uint32_t a_index, uint32_t offset) {\n"
        "    uint32_t bounded = offset % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "    if (row->r_query == RELATION_SPATIAL_R0) {\n"
        "        return (a_index + bounded) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "    }\n"
        "    return (a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - bounded) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "}\n\n"
        + offset_helper_anchor
    )
    require(offset_helper_anchor in text, "control_b_index anchor not found")
    text = text.replace(offset_helper_anchor, offset_helper, 1)
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
        "    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2) {\n"
        "        return signed_offset_index(row, a_index, 2u);\n"
        "    }\n"
        "    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4) {\n"
        "        return signed_offset_index(row, a_index, 4u);\n"
        "    }\n"
        "    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8) {\n"
        "        return signed_offset_index(row, a_index, 8u);\n"
        "    }\n"
        "    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16) {\n"
        "        return signed_offset_index(row, a_index, 16u);\n"
        "    }\n"
        "    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024) {\n"
        "        return signed_offset_index(row, a_index, 1024u);\n"
        "    }\n"
        "    return relation_spatial_map_index(row->r_query, a_index);\n",
    )
    lifecycle_anchor = "static int sample_a_first(const relation_spatial_schedule_row *row, uint32_t sample_index) {\n"
    lifecycle_helpers = (
        "static int source_dead_before_query_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;\n"
        "}\n\n"
        "static int source_off_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024;\n"
        "}\n\n"
        "static int reset_double_flush_after_source_dead_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;\n"
        "}\n\n"
        + lifecycle_anchor
    )
    require(lifecycle_anchor in text, "sample_a_first anchor not found")
    text = text.replace(lifecycle_anchor, lifecycle_helpers, 1)
    text = text.replace(
        "        double c_pair = 0.0;\n",
        "        double c_pair = 0.0;\n"
        "        int source_alive_at_pair_measurement = 1;\n",
        1,
    )
    text = text.replace(
        "            if (!pin_to_core(row.receiver_cpu_expected)) {\n",
        "            if (source_dead_before_query_control(row.control)) {\n"
        "                shared->release_source = 1;\n"
        "                if (waitpid(child, &wait_status, 0) < 0) {\n"
        "                    goto cleanup;\n"
        "                }\n"
        "                source_exit_ns = monotonic_ns();\n"
        "                child = -1;\n"
        "                source_alive_at_pair_measurement = 0;\n"
        "                if (reset_double_flush_after_source_dead_control(row.control)) {\n"
        "                    flush_state_lines(&shared->state);\n"
        "                    flush_state_lines(&shared->state);\n"
        "                }\n"
        "            }\n"
        "            if (!pin_to_core(row.receiver_cpu_expected)) {\n",
        1,
    )
    text = text.replace(
        "            shared->release_source = 1;\n"
        "            if (waitpid(child, &wait_status, 0) < 0) {\n"
        "                goto cleanup;\n"
        "            }\n"
        "            source_exit_ns = monotonic_ns();\n"
        "            close_perf_group(&group);\n"
        "            child = -1;\n",
        "            if (source_alive_at_pair_measurement) {\n"
        "                shared->release_source = 1;\n"
        "                if (waitpid(child, &wait_status, 0) < 0) {\n"
        "                    goto cleanup;\n"
        "                }\n"
        "                source_exit_ns = monotonic_ns();\n"
        "                child = -1;\n"
        "            }\n"
        "            close_perf_group(&group);\n",
        1,
    )
    text = text.replace(
        "    int synthetic\n"
        ") {\n",
        "    int source_alive_at_pair_measurement,\n"
        "    int synthetic\n"
        ") {\n",
        1,
    )
    text = text.replace(
        '\\"source_alive_at_pair_measurement\\":true,\\"physical_measurement\\":%s,',
        '\\"source_alive_at_pair_measurement\\":%s,\\"physical_measurement\\":%s,',
        1,
    )
    text = text.replace(
        "               query_end_ns,\n"
        "               synthetic ? \"false\" : \"true\")\n",
        "               query_end_ns,\n"
        "               source_alive_at_pair_measurement ? \"true\" : \"false\",\n"
        "               synthetic ? \"false\" : \"true\")\n",
        1,
    )
    text = text.replace(
        '\\"source_alive_at_pair_measurement\\":true,\\"source_cpu_expected\\":%d,',
        '\\"source_alive_at_pair_measurement\\":%s,\\"source_cpu_expected\\":%d,',
        1,
    )
    text = text.replace(
        "                    b_cycles[sample],\n"
        "                    row.source_cpu_expected,\n",
        "                    b_cycles[sample],\n"
        "                    source_alive_at_pair_measurement ? \"true\" : \"false\",\n"
        "                    row.source_cpu_expected,\n",
        1,
    )
    text = text.replace(
        "                query_start_ns,\n"
        "                query_end_ns,\n"
        "                synthetic)) {\n",
        "                query_start_ns,\n"
        "                query_end_ns,\n"
        "                source_alive_at_pair_measurement,\n"
        "                synthetic)) {\n",
        1,
    )
    text = text.replace(
        '\\"source_lifetime\\":\\"alive_during_query\\",\\"source_pid\\":%d,',
        '\\"source_lifetime\\":\\"%s\\",\\"source_pid\\":%d,',
        1,
    )
    text = text.replace(
        '\\"source_alive_at_pair_measurement\\":true,\\"source_alive_during_query\\":true,\\"post_observation_query_or_window_selection\\":false,\\"process_custody\\":\\"source_alive_during_spatial_pair_probe\\",',
        '\\"source_alive_at_pair_measurement\\":%s,\\"source_alive_during_query\\":%s,\\"post_observation_query_or_window_selection\\":false,\\"process_custody\\":\\"%s\\",',
        1,
    )
    text = text.replace(
        "                row.execution_ordinal,\n"
        "                synthetic ? 0 : source_pid_record,\n",
        "                row.execution_ordinal,\n"
        "                source_alive_at_pair_measurement ? \"alive_during_query\" : (source_off_control(row.control) ? \"source_off_no_preparation\" : (reset_double_flush_after_source_dead_control(row.control) ? \"dead_before_query_reset_double_flush\" : \"dead_before_query\")),\n"
        "                synthetic ? 0 : source_pid_record,\n",
        1,
    )
    text = text.replace(
        "                query_start_ns,\n"
        "                query_end_ns,\n"
        "                source_cpu_before,\n",
        "                query_start_ns,\n"
        "                query_end_ns,\n"
        "                source_alive_at_pair_measurement ? \"true\" : \"false\",\n"
        "                source_alive_at_pair_measurement ? \"true\" : \"false\",\n"
        "                source_alive_at_pair_measurement ? \"source_alive_during_spatial_pair_probe\" : (source_off_control(row.control) ? \"source_off_no_preparation\" : (reset_double_flush_after_source_dead_control(row.control) ? \"source_dead_before_spatial_pair_probe_reset_double_flush\" : \"source_dead_before_spatial_pair_probe\")),\n"
        "                source_cpu_before,\n",
        1,
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
        "            if (source_off_control(row.control)) {\n"
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
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_RUNTIME_BUILD_V1",
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
        all_rows = list(reader)
    blocks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        blocks[row["block_id"]].append(row)
    selected_blocks = []
    for block in blocks.values():
        if len(block) != 4:
            continue
        first = sorted(block, key=lambda row: int(row["block_local_position"]))[0]
        if first["cyclic_origin"] == FOCUS_CYCLIC_ORIGIN and first["query_order"] == FOCUS_QUERY_ORDER:
            selected_blocks.append(sorted(block, key=lambda row: int(row["block_local_position"])))
    require(selected_blocks, "no complete source blocks matched the focused lifecycle slice")
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    schedules: dict[str, Any] = {}
    for variant in VARIANTS:
        query = QUERY_BY_VARIANT[variant]
        rows = []
        block_index = 0
        while len(rows) < ROWS_PER_VARIANT:
            block = selected_blocks[block_index % len(selected_blocks)]
            for row in block:
                if len(rows) >= ROWS_PER_VARIANT:
                    break
                index = len(rows)
                copied = dict(row)
                copied["block_id"] = f"{variant}_rep{block_index:03d}_{row['block_id']}"
                copied["execution_ordinal"] = str(index)
                copied["query"] = query
                copied["operation_semantics_id"] = query
                copied["control_semantics_id"] = "none" if LIFECYCLE_BY_VARIANT[variant] == "alive" else query
                copied["tuple_id"] = f"{RUN_ID}:{variant}:{index:06d}:{sha256_bytes((variant + ':' + str(index)).encode())[:16]}"
                matched_id = sha256_bytes(copied["block_id"].encode("utf-8"))[:16]
                copied["matched_twin_group"] = f"g_o{OFFSET_BY_VARIANT[variant]}_{LIFECYCLE_BY_VARIANT[variant]}_{matched_id}"
                copied["matched_twin_pair"] = (
                    f"p_o{OFFSET_BY_VARIANT[variant]}_{LIFECYCLE_BY_VARIANT[variant]}_"
                    f"{copied['block_local_position']}_{matched_id}"
                )
                rows.append(copied)
            block_index += 1
        path = SCHEDULE_DIR / f"{variant}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
        schedules[variant] = {
            "query": query,
            "offset": OFFSET_BY_VARIANT[variant],
            "lifecycle": LIFECYCLE_BY_VARIANT[variant],
            "focus_cyclic_origin": FOCUS_CYCLIC_ORIGIN,
            "focus_query_order": FOCUS_QUERY_ORDER,
            "source_complete_block_count": len(selected_blocks),
            "path": f"OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_DISCOVERY_SCHEDULE_MANIFEST_V1",
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
        "mechanism_question": (
            "Does the signed offset receiver projection preserve a source-written relation-matrix "
            "contrast while source-off collapses the same projection before reset/reuse claims?"
        ),
        "receiver_projection": "matrix contrast of mean(B_first_touch_cycles - A_first_touch_cycles) at signed offset sweep",
    }
    manifest["schedule_manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    write_json(SOURCE_ROOT / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SCHEDULE_MANIFEST.json", manifest)
    write_json(HERE / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SCHEDULE_MANIFEST.json", manifest)
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
receipt = {{"schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_CPUFREQ_PREPARE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
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
receipt = {{"schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_CPUFREQ_RESTORE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
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
receipt = {{"schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_PREFLIGHT_V1", "passed": not failures and pmu.returncode == 0, "failures": failures, "target_identity": "AuthenticAMD family16 model10 cpus4-5", "sensor": {{"hwmon": "hwmon0", "name": name.read_text().strip() if name.exists() else None, "temp1_input": temp}}, "pmu_preflight": {{"returncode": pmu.returncode, "stdout": pmu.stdout, "stderr": pmu.stderr}}, "small_wall_crossed": False}}
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
  schedule="$SOURCE_ROOT/OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SCHEDULES/$variant.tsv"
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
summary = {{"schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_TARGET_SUMMARY_V1", "run_id": "{RUN_ID}", "attempt_consumed": (root / "ATTEMPT_CONSUMED").exists(), "variants": {{}}, "small_wall_crossed": False}}
for variant in variants:
    out = root / "discovery_outputs" / variant
    rc_path = root / "discovery_logs" / f"{{variant}}.rc"
    summary["variants"][variant] = {{"rc": int(rc_path.read_text().strip()) if rc_path.exists() else None, "raw_record_count": lines(out / "raw_records.jsonl"), "pair_observation_count": lines(out / "pair_observations.jsonl"), "source_death_receipt_count": lines(out / "source_death_receipts.jsonl"), "raw_sha256": sha(out / "raw_records.jsonl") if (out / "raw_records.jsonl").exists() else None, "pair_sha256": sha(out / "pair_observations.jsonl") if (out / "pair_observations.jsonl").exists() else None, "death_sha256": sha(out / "source_death_receipts.jsonl") if (out / "source_death_receipts.jsonl").exists() else None}}
summary["raw_record_count_total"] = sum(v["raw_record_count"] for v in summary["variants"].values())
summary["pair_observation_count_total"] = sum(v["pair_observation_count"] for v in summary["variants"].values())
summary["source_death_receipt_count_total"] = sum(v["source_death_receipt_count"] for v in summary["variants"].values())
summary["passed"] = summary["attempt_consumed"] and all(v["rc"] == 0 for v in summary["variants"].values())
summary["summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
(root / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_TARGET_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
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
        info = tarfile.TarInfo("source/RUN_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE.sh")
        info.size = len(data)
        info.mode = 0o755
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(data))
        files["source/RUN_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE.sh"] = {"sha256": sha256_bytes(data), "size_bytes": len(data)}
    receipt = {
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_SOURCE_PACKAGE_V1",
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
        f"set -eu; tar -xzf {REMOTE_PACKAGE} -C {REMOTE_ROOT}; chmod +x {REMOTE_ROOT}/source/relation_spatial_runtime {REMOTE_ROOT}/source/relation_spatial_pmu_preflight {REMOTE_ROOT}/source/RUN_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE.sh",
        timeout=120,
    )
    require(extract["returncode"] == 0, f"extract failed: {extract['stderr']}")
    live = run_remote(f"set +e; cd {REMOTE_ROOT}/source; ./RUN_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE.sh", timeout=3600)
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
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_COPYBACK_V1",
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
        cleanup = {"schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_REMOTE_CLEANUP_V1", "cleanup": cleanup_cmd, "passed": cleanup_cmd["returncode"] == 0}
    receipt = {
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_LIVE_CONTROLLER_V1",
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


def row_projection_pairs(raw: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = pair_groups(pairs)
    rows = []
    for row in raw:
        row_pairs = grouped[row["tuple_id"]]
        a = [float(pair["A_first_touch_cycles"]) for pair in row_pairs]
        b = [float(pair["B_first_touch_cycles"]) for pair in row_pairs]
        signed_b_minus_a = [bv - av for av, bv in zip(a, b)]
        rows.append(
            {
                **row,
                "C_pair_recomputed": spearman(a, b),
                "mean_b_minus_a": mean(signed_b_minus_a),
                "mean_abs_b_minus_a": mean([abs(value) for value in signed_b_minus_a]),
            }
        )
    return rows


def relation_matrix_records(rows: list[dict[str, Any]], feature: str) -> list[dict[str, Any]]:
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    records = []
    for block_id, block in blocks.items():
        cells = {row["relation_cell"]: row for row in block}
        needed = ["prepare_r0__query_r0", "prepare_r0__query_r1", "prepare_r1__query_r0", "prepare_r1__query_r1"]
        if any(name not in cells for name in needed):
            continue
        contrast = (
            cells["prepare_r0__query_r0"][feature]
            + cells["prepare_r1__query_r1"][feature]
            - cells["prepare_r0__query_r1"][feature]
            - cells["prepare_r1__query_r0"][feature]
        )
        first = block[0]
        records.append(
            {
                "block_id": block_id,
                "relation_matrix_contrast": contrast,
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
    reports: dict[str, Any] = {}
    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as tf:
        names = set(tf.getnames())
        for variant in VARIANTS:
            prefix = f"source/discovery_outputs/{variant}"
            for suffix in ["raw_records.jsonl", "pair_observations.jsonl", "source_death_receipts.jsonl"]:
                require(f"{prefix}/{suffix}" in names, f"missing archive member {prefix}/{suffix}")
            raw = parse_jsonl_bytes(tf.extractfile(f"{prefix}/raw_records.jsonl").read())  # type: ignore[union-attr]
            pairs = parse_jsonl_bytes(tf.extractfile(f"{prefix}/pair_observations.jsonl").read())  # type: ignore[union-attr]
            deaths = parse_jsonl_bytes(tf.extractfile(f"{prefix}/source_death_receipts.jsonl").read())  # type: ignore[union-attr]
            require(len(raw) == ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            require(len(pairs) == EXPECTED_PAIRS_PER_VARIANT, f"{variant} pair count mismatch: {len(pairs)}")
            require(len(deaths) == ROWS_PER_VARIANT, f"{variant} source-death count mismatch: {len(deaths)}")
            rows = row_projection_pairs(raw, pairs)
            records = relation_matrix_records(rows, "mean_b_minus_a")
            records_by_variant[variant] = records
            values = [record["relation_matrix_contrast"] for record in records]
            reports[variant] = {
                "query": QUERY_BY_VARIANT[variant],
                "lifecycle": LIFECYCLE_BY_VARIANT[variant],
                "raw_record_count": len(raw),
                "pair_observation_count": len(pairs),
                "source_death_receipt_count": len(deaths),
                "block_count": len(records),
                "C_offset_signed": distribution(values),
                "source_alive_at_pair_measurement_counts": dict(
                    Counter(str(death.get("source_alive_at_pair_measurement")) for death in deaths)
                ),
                "source_alive_during_query_counts": dict(
                    Counter(str(death.get("source_alive_during_query")) for death in deaths)
                ),
                "source_lifetime_counts": dict(Counter(str(death.get("source_lifetime")) for death in deaths)),
                "process_custody_counts": dict(Counter(str(death.get("process_custody")) for death in deaths)),
            }

    factors = ["session", "replicate", "mapping", "source_order", "query_order", "cyclic_origin"]
    threshold = 0.25

    def value_for(variant: str) -> float:
        return reports[variant]["C_offset_signed"]["mean"]

    def by_factor(variant: str, factor: str) -> dict[str, float]:
        records = records_by_variant[variant]
        levels = sorted(set(str(record[factor]) for record in records), key=str)
        return {
            level: mean([record["relation_matrix_contrast"] for record in records if str(record[factor]) == level])
            for level in levels
        }

    alive_variant = "alive_offset16_signed"
    source_off_variant = "source_off_offset16_signed"
    dead_variant = "dead_offset16_signed"
    reset_variant = "reset_double_flush_offset16_signed"
    alive = value_for(alive_variant)
    source_off = value_for(source_off_variant)
    dead = value_for(dead_variant)
    reset = value_for(reset_variant)
    alive_by_factor = {factor: by_factor(alive_variant, factor) for factor in factors}
    stratum_reports: dict[str, Any] = {}

    def ratio_to_alive(value: float) -> float | None:
        return abs(value) / abs(alive) if alive else None

    def sign_matches_alive(value: float) -> bool:
        return bool(alive) and bool(value) and ((value > 0) == (alive > 0))

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
                    "C_offset16_signed": value,
                    "matched_alive_C_offset16_signed": alive_value,
                    "abs_to_matched_alive_abs": ratio,
                    "same_sign_as_alive_aggregate": sign_matches_alive(value),
                    "below_25pct_matched_alive": ratio is not None and ratio <= threshold,
                    "above_25pct_matched_alive": ratio is not None and ratio >= threshold,
                }
            factor_reports[factor] = {
                "levels": level_reports,
                "all_same_sign_as_alive_aggregate": all(
                    item["same_sign_as_alive_aggregate"] for item in level_reports.values()
                ),
                "all_below_25pct_matched_alive": all(
                    item["below_25pct_matched_alive"] for item in level_reports.values()
                ),
                "all_above_25pct_matched_alive": all(
                    item["above_25pct_matched_alive"] for item in level_reports.values()
                ),
            }
        stratum_reports[variant] = {
            "offset": FOCUS_OFFSET,
            "lifecycle": LIFECYCLE_BY_VARIANT[variant],
            "factors": factor_reports,
            "all_same_sign_as_alive_aggregate": all(sign_matches_alive(value) for value in values)
            if values and alive
            else False,
            "all_one_factor_strata_below_25pct_alive_matched": all(ratio <= threshold for ratio in ratios)
            if ratios
            else False,
            "all_one_factor_strata_above_25pct_alive_matched": all(ratio >= threshold for ratio in ratios)
            if ratios
            else False,
            "max_abs_to_alive_matched_abs": max(ratios) if ratios else None,
            "min_abs_to_alive_matched_abs": min(ratios) if ratios else None,
        }

    source_off_ratio = ratio_to_alive(source_off)
    dead_ratio = ratio_to_alive(dead)
    reset_ratio = ratio_to_alive(reset)
    alive_nonzero = abs(alive) > 0.0
    alive_same_sign = stratum_reports[alive_variant]["all_same_sign_as_alive_aggregate"]
    source_off_collapse = source_off_ratio is not None and source_off_ratio <= threshold
    source_off_strata_collapse = stratum_reports[source_off_variant]["all_one_factor_strata_below_25pct_alive_matched"]
    dead_preserves = dead_ratio is not None and dead_ratio >= threshold and sign_matches_alive(dead)
    dead_strata_preserve = (
        stratum_reports[dead_variant]["all_one_factor_strata_above_25pct_alive_matched"]
        and stratum_reports[dead_variant]["all_same_sign_as_alive_aggregate"]
    )
    reset_collapse = reset_ratio is not None and reset_ratio <= threshold
    reset_strata_collapse = stratum_reports[reset_variant]["all_one_factor_strata_below_25pct_alive_matched"]
    lifecycle_candidate = (
        alive_nonzero
        and alive_same_sign
        and source_off_collapse
        and source_off_strata_collapse
        and dead_preserves
        and dead_strata_preserve
        and reset_collapse
        and reset_strata_collapse
    )
    lifecycle_reports = {
        "alive": {
            "variant": alive_variant,
            "C_offset16_signed": alive,
            "abs_to_alive_abs": 1.0 if alive_nonzero else None,
            "same_sign_as_alive": True if alive_nonzero else False,
        },
        "source_off": {
            "variant": source_off_variant,
            "C_offset16_signed": source_off,
            "abs_to_alive_abs": source_off_ratio,
            "same_sign_as_alive": sign_matches_alive(source_off),
            "collapses_below_25pct_alive": source_off_collapse,
        },
        "dead": {
            "variant": dead_variant,
            "C_offset16_signed": dead,
            "abs_to_alive_abs": dead_ratio,
            "same_sign_as_alive": sign_matches_alive(dead),
            "preserves_above_25pct_alive": dead_preserves,
        },
        "reset_double_flush": {
            "variant": reset_variant,
            "C_offset16_signed": reset,
            "abs_to_alive_abs": reset_ratio,
            "same_sign_as_alive": sign_matches_alive(reset),
            "collapses_below_25pct_alive": reset_collapse,
        },
    }
    result = {
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_DISCOVERY_ANALYSIS_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "receiver_projection": "relation-matrix contrast of per-row mean(B_first_touch_cycles - A_first_touch_cycles) at signed offset 16",
        "focus": {
            "offset": FOCUS_OFFSET,
            "cyclic_origin": FOCUS_CYCLIC_ORIGIN,
            "query_order": FOCUS_QUERY_ORDER,
            "rows_per_variant": ROWS_PER_VARIANT,
            "pair_samples_per_row": PAIR_SAMPLE_COUNT,
        },
        "variant_reports": reports,
        "stratum_reports": stratum_reports,
        "lifecycle_reports": lifecycle_reports,
        "coordinates": {
            "C_alive_offset16_signed": alive,
            "C_source_off_offset16_signed": source_off,
            "C_dead_offset16_signed": dead,
            "C_reset_double_flush_offset16_signed": reset,
            "source_off_abs_to_alive_abs": source_off_ratio,
            "dead_abs_to_alive_abs": dead_ratio,
            "reset_double_flush_abs_to_alive_abs": reset_ratio,
        },
        "interpretation": {
            "alive_nonzero": alive_nonzero,
            "alive_all_one_factor_strata_same_sign": alive_same_sign,
            "source_off_collapses_below_25pct_alive": source_off_collapse,
            "source_off_all_one_factor_strata_below_25pct_alive": source_off_strata_collapse,
            "dead_preserves_above_25pct_alive_same_sign": dead_preserves,
            "dead_all_one_factor_strata_preserve_same_sign": dead_strata_preserve,
            "reset_double_flush_collapses_below_25pct_alive": reset_collapse,
            "reset_double_flush_all_one_factor_strata_below_25pct_alive": reset_strata_collapse,
            "offset16_origin2048_ab_lifecycle_candidate": lifecycle_candidate,
            "prospective_confirmation_needed": lifecycle_candidate,
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
    write_json(RUN_ROOT / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_DISCOVERY_ANALYSIS.json", result)
    write_json(SUMMARY_JSON, result)
    lines = [
        "# Offset16 Origin2048 AB Lifecycle Probe",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "Receiver projection: relation-matrix contrast of per-row `mean(B_first_touch_cycles - A_first_touch_cycles)` at signed offset 16.",
        "",
        "| Variant | Offset | Lifecycle | C offset signed | abs/abs(matched alive) | max stratum ratio |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = value_for(variant)
        offset = OFFSET_BY_VARIANT[variant]
        ratio = lifecycle_reports[LIFECYCLE_BY_VARIANT[variant]]["abs_to_alive_abs"] or 0.0
        lines.append(
            f"| {variant} | {offset} | {LIFECYCLE_BY_VARIANT[variant]} | {value:.9f} | "
            f"{ratio:.3f} | "
            f"{(stratum_reports[variant]['max_abs_to_alive_matched_abs'] or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "Discovery interpretation:",
            f"- alive nonzero: `{result['interpretation']['alive_nonzero']}`",
            f"- alive one-factor strata same sign: `{result['interpretation']['alive_all_one_factor_strata_same_sign']}`",
            f"- source-off collapses below 25pct alive: `{result['interpretation']['source_off_collapses_below_25pct_alive']}`",
            f"- source-off one-factor strata collapse: `{result['interpretation']['source_off_all_one_factor_strata_below_25pct_alive']}`",
            f"- dead preserves above 25pct alive same-sign: `{result['interpretation']['dead_preserves_above_25pct_alive_same_sign']}`",
            f"- dead one-factor strata preserve same-sign: `{result['interpretation']['dead_all_one_factor_strata_preserve_same_sign']}`",
            f"- reset double flush collapses below 25pct alive: `{result['interpretation']['reset_double_flush_collapses_below_25pct_alive']}`",
            f"- reset double flush one-factor strata collapse: `{result['interpretation']['reset_double_flush_all_one_factor_strata_below_25pct_alive']}`",
            f"- lifecycle candidate: `{result['interpretation']['offset16_origin2048_ab_lifecycle_candidate']}`",
            "",
            "This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare() -> dict[str, Any]:
    copy_segmented_source()
    patch_runtime_for_offset_signed_projection_source_off_screen()
    schedule_manifest = make_rows()
    build = compile_runtime()
    result = {
        "schema": "FAMILY10H_RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_PREPARE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    write_json(HERE / "RELATION_OFFSET16_ORIGIN2048_AB_LIFECYCLE_PROBE_PREPARE_RESULT.json", result)
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
