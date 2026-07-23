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


RUN_ID = "family10h_relation_rspatial_phase_mutation_discovery_v1_0"
TARGET_HOST = "root@192.168.137.100"
REMOTE_BASE = "/root/catcas_live_small_wall"
REMOTE_ROOT = f"{REMOTE_BASE}/{RUN_ID}"
REMOTE_PACKAGE = f"{REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
REMOTE_ARCHIVE = f"{REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
OWNER_MARKER = f".{RUN_ID}_owner"
RUNTIME_AUTHORITY_VALUE = "family10h_relation_spatial_pair_readout_v1_0"

VARIANTS = [
    "alive_primary",
    "alive_sham",
    "dead_primary",
    "dead_sham",
    "reset_double_flush_primary",
    "reset_double_flush_sham",
    "mutate_same_primary",
    "mutate_same_sham",
    "mutate_opposite_primary",
    "mutate_opposite_sham",
]
QUERY_BY_VARIANT = {
    "alive_primary": "query_relation_pair",
    "alive_sham": "relation_sham",
    "dead_primary": "dead_query_relation_pair_control",
    "dead_sham": "dead_relation_sham_control",
    "reset_double_flush_primary": "reset_double_flush_query_relation_pair_control",
    "reset_double_flush_sham": "reset_double_flush_relation_sham_control",
    "mutate_same_primary": "mutate_same_query_relation_pair_control",
    "mutate_same_sham": "mutate_same_relation_sham_control",
    "mutate_opposite_primary": "mutate_opposite_query_relation_pair_control",
    "mutate_opposite_sham": "mutate_opposite_relation_sham_control",
}
ROWS_PER_VARIANT = 1024
PAIR_SAMPLE_COUNT = 256
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT
MATCHED_PERMUTATION_COUNT = 63
MATCHED_PERMUTATION_SEED = "family10h-rspatial-phase-mutation-discovery-v1"

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
SEGMENTED_ROOT = CARRIER_ROOT / "family10h_relation_spatial_pair_readout_v1_1_segmented"
LOCAL_PAIRED_ROOT = CARRIER_ROOT / "family10h_primary_minus_sham_local_paired_differential_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "RSPATIAL_PHASE_MUTATION_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_RSPATIAL_PHASE_MUTATION_DISCOVERY_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_RSPATIAL_PHASE_MUTATION_DISCOVERY_SUMMARY.md"


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


def patch_runtime_for_rspatial_full_schedule_lifetime_reset() -> None:
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    text = header.read_text(encoding="utf-8")
    text = text.replace(
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4",
        "RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_PRIMARY = 5,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_SHAM = 6,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_FLUSH_PRIMARY = 7,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_FLUSH_SHAM = 8,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_PRIMARY = 9,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM = 10,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY = 11,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM = 12,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY = 13,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM = 14,\n"
        "    RELATION_SPATIAL_CONTROL_RESTORE_SAME_PRIMARY = 15,\n"
        "    RELATION_SPATIAL_CONTROL_RESTORE_SAME_SHAM = 16,\n"
        "    RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_PRIMARY = 17,\n"
        "    RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_SHAM = 18",
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
        '    if (strcmp(query, "dead_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_flush_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_FLUSH_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_flush_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_FLUSH_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_prefault_flush_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_prefault_flush_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_double_flush_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_double_flush_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_lane_a_flush_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_lane_a_flush_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "mutate_same_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESTORE_SAME_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "mutate_same_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESTORE_SAME_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "mutate_opposite_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "mutate_opposite_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_SHAM;\n"
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
        "    if (row->control == RELATION_SPATIAL_CONTROL_DEAD_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_FLUSH_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESTORE_SAME_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_SHAM) {\n"
        "        return (a_index + 1024u + ((uint32_t)row->execution_ordinal & 31u)) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "    }\n"
        "    return relation_spatial_map_index(row->r_query, a_index);\n",
    )
    lane_a_anchor = "static int rdtscp_available(void) {\n"
    lane_a_helper = (
        "static void flush_lane_a_lines(relation_spatial_carrier_state *state) {\n"
        "#if defined(__x86_64__) || defined(__i386__)\n"
        "    uint32_t i = 0u;\n"
        "    for (i = 0u; i < FAMILY10H_RELATION_SPATIAL_LINE_COUNT; ++i) {\n"
        "        _mm_clflush(&state->lane_a[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES]);\n"
        "    }\n"
        "    _mm_mfence();\n"
        "#else\n"
        "    (void)state;\n"
        "#endif\n"
        "}\n\n"
        + lane_a_anchor
    )
    require(lane_a_anchor in text, "rdtscp_available anchor not found")
    text = text.replace(lane_a_anchor, lane_a_helper, 1)
    lifecycle_anchor = "static int sample_a_first(const relation_spatial_schedule_row *row, uint32_t sample_index) {\n"
    lifecycle_helpers = (
        "static int source_dead_before_query_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_DEAD_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_SAME_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_SAME_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_SHAM;\n"
        "}\n\n"
        "static int reset_after_source_dead_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESET_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM;\n"
        "}\n\n"
        "static int reset_prefault_flush_after_source_dead_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_PREFAULT_FLUSH_SHAM;\n"
        "}\n\n"
        "static int reset_double_flush_after_source_dead_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM;\n"
        "}\n\n"
        "static int mutate_same_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESTORE_SAME_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_SAME_SHAM;\n"
        "}\n\n"
        "static int mutate_opposite_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_SHAM;\n"
        "}\n\n"
        "static relation_spatial_relation_id opposite_relation(relation_spatial_relation_id relation) {\n"
        "    return relation == RELATION_SPATIAL_R0 ? RELATION_SPATIAL_R1 : RELATION_SPATIAL_R0;\n"
        "}\n\n"
        "static int reset_lane_a_flush_after_source_dead_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM;\n"
        "}\n\n"
        + lifecycle_anchor
    )
    require(lifecycle_anchor in text, "sample_a_first anchor not found")
    text = text.replace(lifecycle_anchor, lifecycle_helpers, 1)
    text = text.replace(
        "        double c_pair = 0.0;\n",
        "        double c_pair = 0.0;\n"
        "        int source_alive_at_pair_measurement = 1;\n",
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
        "                if (reset_after_source_dead_control(row.control)) {\n"
        "                    if (reset_prefault_flush_after_source_dead_control(row.control)) {\n"
        "                        relation_spatial_prefault(&shared->state);\n"
        "                        flush_state_lines(&shared->state);\n"
        "                    } else if (reset_double_flush_after_source_dead_control(row.control)) {\n"
        "                        flush_state_lines(&shared->state);\n"
        "                        flush_state_lines(&shared->state);\n"
        "                    } else if (reset_lane_a_flush_after_source_dead_control(row.control)) {\n"
        "                        flush_lane_a_lines(&shared->state);\n"
        "                    } else {\n"
        "                        flush_state_lines(&shared->state);\n"
        "                    }\n"
        "                }\n"
        "                if (mutate_same_control(row.control)\n"
        "                    || mutate_opposite_control(row.control)) {\n"
        "                    relation_spatial_preparation restore;\n"
        "                    if (!pin_to_core(row.source_cpu_expected)) {\n"
        "                        goto cleanup;\n"
        "                    }\n"
        "                    restore.bank_a_work = row.bank_a_work;\n"
        "                    restore.bank_b_work = row.bank_b_work;\n"
        "                    restore.relation = mutate_opposite_control(row.control)\n"
        "                        ? opposite_relation(row.r_prepare)\n"
        "                        : row.r_prepare;\n"
        "                    restore.source_order = row.source_order;\n"
        "                    restore.cyclic_origin = row.cyclic_origin;\n"
        "                    if (!relation_spatial_prepare(restore, &shared->state)) {\n"
        "                        goto cleanup;\n"
        "                    }\n"
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
        "    uint64_t query_end_ns,\n"
        "    int synthetic\n"
        ") {\n",
        "    uint64_t query_end_ns,\n"
        "    int source_alive_at_pair_measurement,\n"
        "    int synthetic\n"
        ") {\n",
        1,
    )
    text = text.replace(
        '"source_alive_at_pair_measurement\\":true,\\"physical_measurement\\":%s,\\"positive_physical_claim\\":false}\\n",\n',
        '"source_alive_at_pair_measurement\\":%s,\\"physical_measurement\\":%s,\\"positive_physical_claim\\":false}\\n",\n',
        1,
    )
    text = text.replace(
        "               query_start_ns,\n"
        "               query_end_ns,\n"
        "               synthetic ? \"false\" : \"true\")\n",
        "               query_start_ns,\n"
        "               query_end_ns,\n"
        "               source_alive_at_pair_measurement ? \"true\" : \"false\",\n"
        "               synthetic ? \"false\" : \"true\")\n",
        1,
    )
    text = text.replace(
        '"source_alive_at_pair_measurement\\":true,\\"source_cpu_expected\\":%d,\\"receiver_cpu_expected\\":%d,\\"timer_method\\":\\"rdtscp_lfence_serialized\\",\\"physical_measurement\\":%s,\\"positive_physical_claim\\":false}\\n",\n',
        '"source_alive_at_pair_measurement\\":%s,\\"source_cpu_expected\\":%d,\\"receiver_cpu_expected\\":%d,\\"timer_method\\":\\"rdtscp_lfence_serialized\\",\\"physical_measurement\\":%s,\\"positive_physical_claim\\":false}\\n",\n',
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
        "        if (!write_raw_record(\n",
        "        const char *source_lifetime_text = source_alive_at_pair_measurement ? \"alive_during_query\" : \"dead_before_query\";\n"
        "        const char *process_custody_text = source_alive_at_pair_measurement\n"
        "            ? \"source_alive_during_spatial_pair_probe\"\n"
        "            : (mutate_same_control(row.control)\n"
        "                ? \"source_dead_then_source_cpu_mutate_same_relation_before_spatial_pair_probe\"\n"
        "                : (mutate_opposite_control(row.control)\n"
        "                    ? \"source_dead_then_source_cpu_mutate_opposite_relation_before_spatial_pair_probe\"\n"
        "                    : (reset_prefault_flush_after_source_dead_control(row.control)\n"
        "                ? \"source_dead_then_prefault_flush_before_spatial_pair_probe\"\n"
        "                : (reset_double_flush_after_source_dead_control(row.control)\n"
        "                    ? \"source_dead_then_double_flush_before_spatial_pair_probe\"\n"
        "                    : (reset_lane_a_flush_after_source_dead_control(row.control)\n"
        "                        ? \"source_dead_then_lane_a_flush_before_spatial_pair_probe\"\n"
        "                        : (reset_after_source_dead_control(row.control)\n"
        "                            ? \"source_dead_then_full_flush_before_spatial_pair_probe\"\n"
        "                            : \"source_dead_before_spatial_pair_probe\"))))));\n"
        "        if (!write_raw_record(\n",
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
        '"source_lifetime\\":\\"alive_during_query\\",\\"source_pid\\":%d,',
        '"source_lifetime\\":\\"%s\\",\\"source_pid\\":%d,',
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
        "                source_lifetime_text,\n"
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
        "                process_custody_text,\n"
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
    count = text.count(prep_block)
    require(count == 1, f"runtime execution prep block count unexpected: {count}")
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
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_RUNTIME_BUILD_V1",
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
            copied["control_semantics_id"] = "none" if variant == "alive_primary" else query
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
            "path": f"RSPATIAL_PHASE_MUTATION_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_DISCOVERY_SCHEDULE_MANIFEST_V1",
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
            "Which active reset method most consistently kills the source-dead R-spatial "
            "primary-minus-sham relation coordinate across matched strata?"
        ),
    }
    manifest["schedule_manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    write_json(SOURCE_ROOT / "RELATION_RSPATIAL_PHASE_MUTATION_SCHEDULE_MANIFEST.json", manifest)
    write_json(HERE / "RELATION_RSPATIAL_PHASE_MUTATION_SCHEDULE_MANIFEST.json", manifest)
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
receipt = {{"schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_CPUFREQ_PREPARE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
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
receipt = {{"schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_CPUFREQ_RESTORE_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": {{}}, "failures": []}}
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
receipt = {{"schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_PREFLIGHT_V1", "passed": not failures and pmu.returncode == 0, "failures": failures, "target_identity": "AuthenticAMD family16 model10 cpus4-5", "sensor": {{"hwmon": "hwmon0", "name": name.read_text().strip() if name.exists() else None, "temp1_input": temp}}, "pmu_preflight": {{"returncode": pmu.returncode, "stdout": pmu.stdout, "stderr": pmu.stderr}}, "small_wall_crossed": False}}
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
  schedule="$SOURCE_ROOT/RSPATIAL_PHASE_MUTATION_SCHEDULES/$variant.tsv"
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
summary = {{"schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_TARGET_SUMMARY_V1", "run_id": "{RUN_ID}", "attempt_consumed": (root / "ATTEMPT_CONSUMED").exists(), "variants": {{}}, "small_wall_crossed": False}}
for variant in variants:
    out = root / "discovery_outputs" / variant
    rc_path = root / "discovery_logs" / f"{{variant}}.rc"
    summary["variants"][variant] = {{"rc": int(rc_path.read_text().strip()) if rc_path.exists() else None, "raw_record_count": lines(out / "raw_records.jsonl"), "pair_observation_count": lines(out / "pair_observations.jsonl"), "source_death_receipt_count": lines(out / "source_death_receipts.jsonl"), "raw_sha256": sha(out / "raw_records.jsonl") if (out / "raw_records.jsonl").exists() else None, "pair_sha256": sha(out / "pair_observations.jsonl") if (out / "pair_observations.jsonl").exists() else None, "death_sha256": sha(out / "source_death_receipts.jsonl") if (out / "source_death_receipts.jsonl").exists() else None}}
summary["raw_record_count_total"] = sum(v["raw_record_count"] for v in summary["variants"].values())
summary["pair_observation_count_total"] = sum(v["pair_observation_count"] for v in summary["variants"].values())
summary["source_death_receipt_count_total"] = sum(v["source_death_receipt_count"] for v in summary["variants"].values())
summary["passed"] = summary["attempt_consumed"] and all(v["rc"] == 0 for v in summary["variants"].values())
summary["summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
(root / "RELATION_RSPATIAL_PHASE_MUTATION_TARGET_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
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
        info = tarfile.TarInfo("source/RUN_RSPATIAL_PHASE_MUTATION_DISCOVERY.sh")
        info.size = len(data)
        info.mode = 0o755
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(data))
        files["source/RUN_RSPATIAL_PHASE_MUTATION_DISCOVERY.sh"] = {"sha256": sha256_bytes(data), "size_bytes": len(data)}
    receipt = {
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_SOURCE_PACKAGE_V1",
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
        f"set -eu; tar -xzf {REMOTE_PACKAGE} -C {REMOTE_ROOT}; chmod +x {REMOTE_ROOT}/source/relation_spatial_runtime {REMOTE_ROOT}/source/relation_spatial_pmu_preflight {REMOTE_ROOT}/source/RUN_RSPATIAL_PHASE_MUTATION_DISCOVERY.sh",
        timeout=120,
    )
    require(extract["returncode"] == 0, f"extract failed: {extract['stderr']}")
    live = run_remote(f"set +e; cd {REMOTE_ROOT}/source; ./RUN_RSPATIAL_PHASE_MUTATION_DISCOVERY.sh", timeout=3600)
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
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_COPYBACK_V1",
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
        cleanup = {"schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_REMOTE_CLEANUP_V1", "cleanup": cleanup_cmd, "passed": cleanup_cmd["returncode"] == 0}
    receipt = {
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_LIVE_CONTROLLER_V1",
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


def relation_cell_axes(rows: list[dict[str, Any]]) -> dict[str, float]:
    cells = {
        "prepare_r0__query_r0": [],
        "prepare_r0__query_r1": [],
        "prepare_r1__query_r0": [],
        "prepare_r1__query_r1": [],
    }
    for row in rows:
        cell = row.get("relation_cell")
        if cell in cells:
            cells[cell].append(float(row["C_pair"]))
    require(all(cells[cell] for cell in cells), "missing relation cell for phase-axis analysis")
    rr = mean(cells["prepare_r0__query_r0"])
    rf = mean(cells["prepare_r0__query_r1"])
    fr = mean(cells["prepare_r1__query_r0"])
    ff = mean(cells["prepare_r1__query_r1"])
    return {
        "cell_prepare_r0_query_r0": rr,
        "cell_prepare_r0_query_r1": rf,
        "cell_prepare_r1_query_r0": fr,
        "cell_prepare_r1_query_r1": ff,
        "common_axis": 0.25 * (rr + rf + fr + ff),
        "diag_R_axis": 0.5 * (rr + ff - rf - fr),
        "prep_axis": 0.5 * (rr + rf - fr - ff),
        "query_axis": 0.5 * (rr - rf + fr - ff),
        "offdiag_skew_axis": 0.5 * (rf - fr),
    }


def analyze_archive(controller: dict[str, Any]) -> dict[str, Any]:
    require(LOCAL_ARCHIVE.exists(), "archive missing for analysis")
    reports = {}
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
            rows = row_c_pairs(raw, pairs)
            records = r_spatial_records(rows)
            axes = relation_cell_axes(rows)
            records_by_variant[variant] = records
            values = [record["R_spatial"] for record in records]
            reports[variant] = {
                "query": QUERY_BY_VARIANT[variant],
                "raw_record_count": len(raw),
                "pair_observation_count": len(pairs),
                "source_death_receipt_count": len(deaths),
                "block_count": len(records),
                "R_spatial": distribution(values),
                "relation_cell_axes": axes,
                "source_alive_at_pair_measurement_counts": dict(
                    Counter(str(death.get("source_alive_at_pair_measurement")) for death in deaths)
                ),
                "source_alive_during_query_counts": dict(
                    Counter(str(death.get("source_alive_during_query")) for death in deaths)
                ),
                "source_lifetime_counts": dict(Counter(str(death.get("source_lifetime")) for death in deaths)),
                "process_custody_counts": dict(Counter(str(death.get("process_custody")) for death in deaths)),
            }

    method_pairs = {
        "alive": ("alive_primary", "alive_sham"),
        "dead": ("dead_primary", "dead_sham"),
        "reset_double_flush": ("reset_double_flush_primary", "reset_double_flush_sham"),
        "mutate_same": ("mutate_same_primary", "mutate_same_sham"),
        "mutate_opposite": ("mutate_opposite_primary", "mutate_opposite_sham"),
    }
    method_reports: dict[str, Any] = {}
    for method, (primary_variant, sham_variant) in method_pairs.items():
        primary = reports[primary_variant]["R_spatial"]["mean"]
        sham = reports[sham_variant]["R_spatial"]["mean"]
        delta = primary - sham
        primary_axes = reports[primary_variant]["relation_cell_axes"]
        sham_axes = reports[sham_variant]["relation_cell_axes"]
        axis_delta = {
            key: primary_axes[key] - sham_axes[key]
            for key in [
                "common_axis",
                "diag_R_axis",
                "prep_axis",
                "query_axis",
                "offdiag_skew_axis",
            ]
        }
        method_reports[method] = {
            "primary_variant": primary_variant,
            "sham_variant": sham_variant,
            "primary_R": primary,
            "sham_R": sham,
            "D_primary_minus_sham": delta,
            "D_positive": delta > 0,
            "primary_positive": primary > 0,
            "sham_positive": sham > 0,
            "sham_abs_to_primary_abs": abs(sham) / abs(primary) if primary else None,
            "axis_primary_minus_sham": axis_delta,
            "prep_query_phase_radians": math.atan2(axis_delta["query_axis"], axis_delta["prep_axis"]),
            "prep_query_magnitude": math.hypot(axis_delta["prep_axis"], axis_delta["query_axis"]),
            "diag_skew_phase_radians": math.atan2(axis_delta["offdiag_skew_axis"], axis_delta["diag_R_axis"]),
            "diag_skew_magnitude": math.hypot(axis_delta["diag_R_axis"], axis_delta["offdiag_skew_axis"]),
        }
    alive_d = method_reports["alive"]["D_primary_minus_sham"]
    dead_d = method_reports["dead"]["D_primary_minus_sham"]
    alive_primary = method_reports["alive"]["primary_R"]
    dead_primary = method_reports["dead"]["primary_R"]

    def ratio_to_alive(value: float) -> float | None:
        return abs(value) / abs(alive_d) if alive_d else None

    for report in method_reports.values():
        report["D_abs_to_alive_D_abs"] = ratio_to_alive(report["D_primary_minus_sham"])
        report["prep_query_magnitude_abs_to_alive"] = (
            report["prep_query_magnitude"] / method_reports["alive"]["prep_query_magnitude"]
            if method_reports["alive"]["prep_query_magnitude"]
            else None
        )
        report["diag_skew_magnitude_abs_to_alive"] = (
            report["diag_skew_magnitude"] / method_reports["alive"]["diag_skew_magnitude"]
            if method_reports["alive"]["diag_skew_magnitude"]
            else None
        )

    factors = ["session", "replicate", "mapping", "source_order", "query_order", "cyclic_origin"]

    def d_by_factor(method: str, factor: str) -> dict[str, float]:
        primary_variant, sham_variant = method_pairs[method]
        primary_records = records_by_variant[primary_variant]
        sham_records = records_by_variant[sham_variant]
        levels = sorted(
            set(str(record[factor]) for record in primary_records) & set(str(record[factor]) for record in sham_records),
            key=str,
        )
        values: dict[str, float] = {}
        for level in levels:
            p = mean([record["R_spatial"] for record in primary_records if str(record[factor]) == level])
            s = mean([record["R_spatial"] for record in sham_records if str(record[factor]) == level])
            values[level] = p - s
        return values

    stratum_reports: dict[str, Any] = {}
    alive_strata = {factor: d_by_factor("alive", factor) for factor in factors}
    for method in method_pairs:
        factor_reports = {}
        ratios: list[float] = []
        d_values: list[float] = []
        for factor in factors:
            method_values = d_by_factor(method, factor)
            level_reports = {}
            for level, value in method_values.items():
                alive_value = alive_strata.get(factor, {}).get(level)
                ratio = abs(value) / abs(alive_value) if alive_value else None
                if ratio is not None:
                    ratios.append(ratio)
                d_values.append(value)
                level_reports[level] = {
                    "D_primary_minus_sham": value,
                    "alive_matched_D": alive_value,
                    "abs_to_alive_matched_abs": ratio,
                    "D_positive": value > 0,
                    "below_25pct_alive_matched": ratio is not None and ratio <= 0.25,
                }
            factor_reports[factor] = {
                "levels": level_reports,
                "all_D_positive": all(item["D_positive"] for item in level_reports.values()),
                "all_below_25pct_alive_matched": all(
                    item["below_25pct_alive_matched"] for item in level_reports.values()
                ),
            }
        stratum_reports[method] = {
            "factors": factor_reports,
            "all_one_factor_strata_D_positive": all(value > 0 for value in d_values),
            "all_one_factor_strata_below_25pct_alive_matched": all(ratio <= 0.25 for ratio in ratios)
            if ratios
            else False,
            "max_abs_to_alive_matched_abs": max(ratios) if ratios else None,
        }

    reset_methods = ["reset_double_flush"]
    reuse_methods = ["mutate_same", "mutate_opposite"]
    best_reset_method = min(
        reset_methods,
        key=lambda method: (
            stratum_reports[method]["max_abs_to_alive_matched_abs"]
            if stratum_reports[method]["max_abs_to_alive_matched_abs"] is not None
            else float("inf"),
            method_reports[method]["D_abs_to_alive_D_abs"]
            if method_reports[method]["D_abs_to_alive_D_abs"] is not None
            else float("inf"),
        ),
    )
    best_reset = method_reports[best_reset_method]
    best_reset_strata = stratum_reports[best_reset_method]

    result = {
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_DISCOVERY_ANALYSIS_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "variant_reports": reports,
        "method_reports": method_reports,
        "stratum_reports": stratum_reports,
        "coordinates": {
            "R_alive_primary": alive_primary,
            "R_alive_sham": method_reports["alive"]["sham_R"],
            "D_alive_primary_minus_sham": alive_d,
            "R_dead_primary": dead_primary,
            "R_dead_sham": method_reports["dead"]["sham_R"],
            "D_dead_primary_minus_sham": dead_d,
            "dead_D_abs_to_alive_D_abs": ratio_to_alive(dead_d),
            "dead_primary_abs_to_alive_primary_abs": abs(dead_primary) / abs(alive_primary) if alive_primary else None,
            "best_reset_method": best_reset_method,
            "best_reset_D_primary_minus_sham": best_reset["D_primary_minus_sham"],
            "best_reset_D_abs_to_alive_D_abs": best_reset["D_abs_to_alive_D_abs"],
            "best_reset_max_stratum_abs_to_alive_matched_abs": best_reset_strata["max_abs_to_alive_matched_abs"],
            "mutate_same_D_primary_minus_sham": method_reports["mutate_same"]["D_primary_minus_sham"],
            "mutate_same_D_abs_to_alive_D_abs": method_reports["mutate_same"]["D_abs_to_alive_D_abs"],
            "mutate_opposite_D_primary_minus_sham": method_reports["mutate_opposite"]["D_primary_minus_sham"],
            "mutate_opposite_D_abs_to_alive_D_abs": method_reports["mutate_opposite"]["D_abs_to_alive_D_abs"],
            "reset_methods_by_stratum_ratio": sorted(
                [
                    {
                        "method": method,
                        "D_primary_minus_sham": method_reports[method]["D_primary_minus_sham"],
                        "D_abs_to_alive_D_abs": method_reports[method]["D_abs_to_alive_D_abs"],
                        "max_stratum_abs_to_alive_matched_abs": stratum_reports[method][
                            "max_abs_to_alive_matched_abs"
                        ],
                    }
                    for method in reset_methods
                ],
                key=lambda item: (
                    item["max_stratum_abs_to_alive_matched_abs"]
                    if item["max_stratum_abs_to_alive_matched_abs"] is not None
                    else float("inf"),
                    item["D_abs_to_alive_D_abs"] if item["D_abs_to_alive_D_abs"] is not None else float("inf"),
                ),
            ),
            "reuse_methods": [
                {
                    "method": method,
                    "D_primary_minus_sham": method_reports[method]["D_primary_minus_sham"],
                    "D_abs_to_alive_D_abs": method_reports[method]["D_abs_to_alive_D_abs"],
                    "all_one_factor_strata_D_positive": stratum_reports[method]["all_one_factor_strata_D_positive"],
                    "max_stratum_abs_to_alive_matched_abs": stratum_reports[method]["max_abs_to_alive_matched_abs"],
                }
                for method in reuse_methods
            ],
        },
        "interpretation": {
            "alive_D_positive": alive_d > 0,
            "dead_D_positive": dead_d > 0,
            "dead_preserves_at_least_25pct_alive_D": abs(dead_d) >= 0.25 * abs(alive_d) if alive_d else False,
            "source_death_kills_below_25pct_alive_D": abs(dead_d) <= 0.25 * abs(alive_d) if alive_d else False,
            "dead_all_one_factor_strata_D_positive": stratum_reports["dead"]["all_one_factor_strata_D_positive"],
            "best_reset_method": best_reset_method,
            "best_reset_kills_aggregate_below_25pct_alive_D": (
                best_reset["D_abs_to_alive_D_abs"] is not None and best_reset["D_abs_to_alive_D_abs"] <= 0.25
            ),
            "best_reset_kills_all_one_factor_strata_below_25pct_alive_D": best_reset_strata[
                "all_one_factor_strata_below_25pct_alive_matched"
            ],
            "best_reset_reduces_dead_D_abs": abs(best_reset["D_primary_minus_sham"]) < abs(dead_d),
            "dead_primary_same_sign_as_alive_primary": (dead_primary > 0) == (alive_primary > 0),
            "candidate_persistent_source_written_state_with_reset_method": (
                alive_d > 0
                and dead_d > 0
                and abs(dead_d) >= 0.25 * abs(alive_d)
                and best_reset["D_abs_to_alive_D_abs"] is not None
                and best_reset["D_abs_to_alive_D_abs"] <= 0.25
                and best_reset_strata["all_one_factor_strata_below_25pct_alive_matched"]
            )
            if alive_d
            else False,
            "mutate_same_rebuilds_positive_D": method_reports["mutate_same"]["D_primary_minus_sham"] > 0,
            "mutate_same_rebuilds_at_least_50pct_alive_D": (
                method_reports["mutate_same"]["D_abs_to_alive_D_abs"] is not None
                and method_reports["mutate_same"]["D_abs_to_alive_D_abs"] >= 0.50
            ),
            "mutate_opposite_flips_D_negative": method_reports["mutate_opposite"]["D_primary_minus_sham"] < 0,
            "mutate_opposite_magnitude_at_least_50pct_alive_D": (
                method_reports["mutate_opposite"]["D_abs_to_alive_D_abs"] is not None
                and method_reports["mutate_opposite"]["D_abs_to_alive_D_abs"] >= 0.50
            ),
            "phase_mutation_candidate": (
                best_reset["D_abs_to_alive_D_abs"] is not None
                and best_reset["D_abs_to_alive_D_abs"] <= 0.25
                and method_reports["mutate_same"]["D_primary_minus_sham"] > 0
                and method_reports["mutate_opposite"]["D_primary_minus_sham"] < 0
            ),
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
    write_json(RUN_ROOT / "RELATION_RSPATIAL_PHASE_MUTATION_DISCOVERY_ANALYSIS.json", result)
    write_json(SUMMARY_JSON, result)
    lines = [
        "# Relation R-Spatial Phase Mutation Discovery",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "| Method | Primary R | Sham R | D primary-minus-sham | abs(D)/abs(alive D) | max stratum ratio | all strata <= 0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["alive", "dead", *reset_methods, *reuse_methods]:
        report = method_reports[method]
        stratum = stratum_reports[method]
        lines.append(
            f"| {method} | {report['primary_R']:.9f} | {report['sham_R']:.9f} | "
            f"{report['D_primary_minus_sham']:.9f} | "
            f"{(report['D_abs_to_alive_D_abs'] or 0.0):.3f} | "
            f"{(stratum['max_abs_to_alive_matched_abs'] or 0.0):.3f} | "
            f"{stratum['all_one_factor_strata_below_25pct_alive_matched']} |"
        )
    lines.extend(
        [
            "",
            "| Method | prep/query phase rad | prep/query mag ratio | diag/skew phase rad | diag/skew mag ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ["alive", "dead", *reset_methods, *reuse_methods]:
        report = method_reports[method]
        lines.append(
            f"| {method} | {report['prep_query_phase_radians']:.6f} | "
            f"{(report['prep_query_magnitude_abs_to_alive'] or 0.0):.3f} | "
            f"{report['diag_skew_phase_radians']:.6f} | "
            f"{(report['diag_skew_magnitude_abs_to_alive'] or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "Discovery interpretation:",
            f"- alive D positive: `{result['interpretation']['alive_D_positive']}`",
            f"- dead D positive: `{result['interpretation']['dead_D_positive']}`",
            f"- source death kills below 0.25 x alive D: `{result['interpretation']['source_death_kills_below_25pct_alive_D']}`",
            f"- dead all one-factor strata positive: `{result['interpretation']['dead_all_one_factor_strata_D_positive']}`",
            f"- best reset method: `{result['interpretation']['best_reset_method']}`",
            f"- best reset kills aggregate below 0.25 x alive D: `{result['interpretation']['best_reset_kills_aggregate_below_25pct_alive_D']}`",
            f"- best reset kills all one-factor strata below 0.25 x alive D: `{result['interpretation']['best_reset_kills_all_one_factor_strata_below_25pct_alive_D']}`",
            f"- best reset reduces dead D abs: `{result['interpretation']['best_reset_reduces_dead_D_abs']}`",
            f"- persistent source-written-state candidate with reset method: `{result['interpretation']['candidate_persistent_source_written_state_with_reset_method']}`",
            f"- mutate-same rebuilds positive D: `{result['interpretation']['mutate_same_rebuilds_positive_D']}`",
            f"- mutate-opposite flips D negative: `{result['interpretation']['mutate_opposite_flips_D_negative']}`",
            f"- phase mutation candidate: `{result['interpretation']['phase_mutation_candidate']}`",
            "",
            "This is exploratory evidence only. It emits no prospective scientific claim and does not promote `SMALL_WALL_CROSSED`.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare() -> dict[str, Any]:
    copy_segmented_source()
    patch_runtime_for_rspatial_full_schedule_lifetime_reset()
    schedule_manifest = make_rows()
    build = compile_runtime()
    result = {
        "schema": "FAMILY10H_RELATION_RSPATIAL_PHASE_MUTATION_PREPARE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    write_json(HERE / "RELATION_RSPATIAL_PHASE_MUTATION_PREPARE_RESULT.json", result)
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
