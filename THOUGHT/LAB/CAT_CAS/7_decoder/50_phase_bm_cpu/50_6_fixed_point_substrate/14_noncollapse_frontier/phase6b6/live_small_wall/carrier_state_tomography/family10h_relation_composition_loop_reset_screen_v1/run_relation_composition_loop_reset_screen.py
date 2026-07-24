#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


RUN_ID = "family10h_relation_composition_loop_reset_screen_v1_2"
PACKAGE_ID = "family10h_relation_composition_loop_reset_screen_v1"
ROWS_PER_VARIANT = 512
PAIR_SAMPLE_COUNT = 256
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT

VARIANTS = [
    "r0r1_primary",
    "r0r1_sham",
    "r1r0_primary",
    "r1r0_sham",
    "dead_r0r1_primary",
    "dead_r0r1_sham",
    "dead_r1r0_primary",
    "dead_r1r0_sham",
    "reset_r0r1_primary",
    "reset_r0r1_sham",
    "reset_r1r0_primary",
    "reset_r1r0_sham",
    "source_off_r0r1_primary",
    "source_off_r0r1_sham",
    "source_off_r1r0_primary",
    "source_off_r1r0_sham",
    "neutral_primary",
    "neutral_sham",
    "random_primary",
    "random_sham",
]

QUERY_BY_VARIANT = {
    "r0r1_primary": "compose_r0_then_r1_query_relation_pair_control",
    "r0r1_sham": "compose_r0_then_r1_relation_sham_control",
    "r1r0_primary": "compose_r1_then_r0_query_relation_pair_control",
    "r1r0_sham": "compose_r1_then_r0_relation_sham_control",
    "dead_r0r1_primary": "dead_compose_r0_then_r1_query_relation_pair_control",
    "dead_r0r1_sham": "dead_compose_r0_then_r1_relation_sham_control",
    "dead_r1r0_primary": "dead_compose_r1_then_r0_query_relation_pair_control",
    "dead_r1r0_sham": "dead_compose_r1_then_r0_relation_sham_control",
    "reset_r0r1_primary": "reset_c_r0r1_primary_control",
    "reset_r0r1_sham": "reset_c_r0r1_sham_control",
    "reset_r1r0_primary": "reset_c_r1r0_primary_control",
    "reset_r1r0_sham": "reset_c_r1r0_sham_control",
    "source_off_r0r1_primary": "source_off_compose_r0_then_r1_query_relation_pair_control",
    "source_off_r0r1_sham": "source_off_compose_r0_then_r1_relation_sham_control",
    "source_off_r1r0_primary": "source_off_compose_r1_then_r0_query_relation_pair_control",
    "source_off_r1r0_sham": "source_off_compose_r1_then_r0_relation_sham_control",
    "neutral_primary": "neutral_compose_query_relation_pair_control",
    "neutral_sham": "neutral_compose_relation_sham_control",
    "random_primary": "random_compose_query_relation_pair_control",
    "random_sham": "random_compose_relation_sham_control",
}

COMPOSITION_BY_VARIANT = {
    "r0r1_primary": "r0_then_r1",
    "r0r1_sham": "r0_then_r1",
    "r1r0_primary": "r1_then_r0",
    "r1r0_sham": "r1_then_r0",
    "dead_r0r1_primary": "r0_then_r1",
    "dead_r0r1_sham": "r0_then_r1",
    "dead_r1r0_primary": "r1_then_r0",
    "dead_r1r0_sham": "r1_then_r0",
    "reset_r0r1_primary": "r0_then_r1_after_double_flush",
    "reset_r0r1_sham": "r0_then_r1_after_double_flush",
    "reset_r1r0_primary": "r1_then_r0_after_double_flush",
    "reset_r1r0_sham": "r1_then_r0_after_double_flush",
    "source_off_r0r1_primary": "r0_then_r1",
    "source_off_r0r1_sham": "r0_then_r1",
    "source_off_r1r0_primary": "r1_then_r0",
    "source_off_r1r0_sham": "r1_then_r0",
    "neutral_primary": "neutral_alternating",
    "neutral_sham": "neutral_alternating",
    "random_primary": "deterministic_randomized",
    "random_sham": "deterministic_randomized",
}

COMPOSITION_TAG_BY_VARIANT = {
    "r0r1_primary": "r0r1",
    "r0r1_sham": "r0r1",
    "r1r0_primary": "r1r0",
    "r1r0_sham": "r1r0",
    "dead_r0r1_primary": "r0r1",
    "dead_r0r1_sham": "r0r1",
    "dead_r1r0_primary": "r1r0",
    "dead_r1r0_sham": "r1r0",
    "reset_r0r1_primary": "r0r1_reset",
    "reset_r0r1_sham": "r0r1_reset",
    "reset_r1r0_primary": "r1r0_reset",
    "reset_r1r0_sham": "r1r0_reset",
    "source_off_r0r1_primary": "r0r1_source_off",
    "source_off_r0r1_sham": "r0r1_source_off",
    "source_off_r1r0_primary": "r1r0_source_off",
    "source_off_r1r0_sham": "r1r0_source_off",
    "neutral_primary": "neutral",
    "neutral_sham": "neutral",
    "random_primary": "random",
    "random_sham": "random",
}

HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
BASE_RUNNER = CARRIER_ROOT / "family10h_relation_temporal_source_off_discovery_v1" / "run_relation_temporal_source_off_discovery.py"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("temporal_source_off_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
BASE_PATCH_RUNTIME = base.patch_runtime_for_rspatial_full_schedule_lifetime_reset
BASE_TARGET_SCRIPT = base.target_script


SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "COMPOSITION_LOOP_RESET_SCREEN_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_SUMMARY.md"


def configure_base() -> None:
    base.RUN_ID = RUN_ID
    base.REMOTE_ROOT = f"{base.REMOTE_BASE}/{RUN_ID}"
    base.REMOTE_PACKAGE = f"{base.REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
    base.REMOTE_ARCHIVE = f"{base.REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
    base.OWNER_MARKER = f".{RUN_ID}_owner"
    base.VARIANTS = VARIANTS
    base.QUERY_BY_VARIANT = QUERY_BY_VARIANT
    base.ROWS_PER_VARIANT = ROWS_PER_VARIANT
    base.PAIR_SAMPLE_COUNT = PAIR_SAMPLE_COUNT
    base.EXPECTED_PAIRS_PER_VARIANT = EXPECTED_PAIRS_PER_VARIANT
    base.HERE = HERE
    base.CARRIER_ROOT = CARRIER_ROOT
    base.SOURCE_ROOT = SOURCE_ROOT
    base.SCHEDULE_DIR = SCHEDULE_DIR
    base.RUN_ROOT = RUN_ROOT
    base.LOCAL_PACKAGE = LOCAL_PACKAGE
    base.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    base.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    base.SUMMARY_JSON = SUMMARY_JSON
    base.SUMMARY_MD = SUMMARY_MD
    base.target_script = target_script


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return sha256_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    base.write_json(path, payload)


def require(condition: bool, message: str) -> None:
    base.require(condition, message)


def patch_runtime_for_composition_loop() -> None:
    BASE_PATCH_RUNTIME()
    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    htext = header.read_text(encoding="utf-8")
    htext = htext.replace(
        "RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM = 20",
        "RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM = 20,\n"
        "    RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_PRIMARY = 21,\n"
        "    RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_SHAM = 22,\n"
        "    RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_PRIMARY = 23,\n"
        "    RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_SHAM = 24,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_PRIMARY = 25,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_SHAM = 26,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_PRIMARY = 27,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_SHAM = 28,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY = 29,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM = 30,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY = 31,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM = 32,\n"
        "    RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_PRIMARY = 33,\n"
        "    RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_SHAM = 34,\n"
        "    RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_PRIMARY = 35,\n"
        "    RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM = 36,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY = 37,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM = 38,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY = 39,\n"
        "    RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM = 40",
    )
    header.write_text(htext, encoding="utf-8", newline="\n")

    text = runtime.read_text(encoding="utf-8")
    temporal_map_function = (
        "uint32_t relation_spatial_map_index(relation_spatial_relation_id relation, uint32_t logical_a_index) {\n"
        "    (void)relation;\n"
        "    return (logical_a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "}\n"
        "\n"
        "static relation_spatial_order_id relation_spatial_opposite_order(relation_spatial_order_id order) {\n"
        "    return order == RELATION_SPATIAL_ORDER_AB ? RELATION_SPATIAL_ORDER_BA : RELATION_SPATIAL_ORDER_AB;\n"
        "}\n"
    )
    native_map_function = (
        "uint32_t relation_spatial_map_index(relation_spatial_relation_id relation, uint32_t logical_a_index) {\n"
        "    if (relation == RELATION_SPATIAL_R0) {\n"
        "        return (logical_a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "    }\n"
        "    return (logical_a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;\n"
        "}\n"
    )
    require(temporal_map_function in text, "temporal map-erasure anchor not found")
    text = text.replace(temporal_map_function, native_map_function, 1)
    text = text.replace(
        '    if (strcmp(query, "source_off_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n",
        '    if (strcmp(query, "source_off_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "compose_r0_then_r1_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "compose_r0_then_r1_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "compose_r1_then_r0_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "compose_r1_then_r0_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_compose_r0_then_r1_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_compose_r0_then_r1_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_compose_r1_then_r0_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_compose_r1_then_r0_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_c_r0r1_primary_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_c_r0r1_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_c_r1r0_primary_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "reset_c_r1r0_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_compose_r0_then_r1_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_compose_r0_then_r1_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_compose_r1_then_r0_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_compose_r1_then_r0_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "neutral_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "neutral_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "random_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "random_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n",
        1,
    )
    text = text.replace(
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM) {\n",
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM) {\n",
        1,
    )
    text = text.replace(
        "static int rdtscp_available(void) {\n",
        "static void relation_spatial_touch_pair(relation_spatial_carrier_state *state, uint32_t a_index, relation_spatial_relation_id relation, relation_spatial_order_id order) {\n"
        "    uint32_t b_index = relation_spatial_map_index(relation, a_index);\n"
        "    volatile uint8_t *a = &state->lane_a[a_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];\n"
        "    volatile uint8_t *b = &state->lane_b[b_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];\n"
        "    if (order == RELATION_SPATIAL_ORDER_AB) {\n"
        "        *a = (uint8_t)(*a + 1u);\n"
        "        *b = (uint8_t)(*b + 1u);\n"
        "    } else {\n"
        "        *b = (uint8_t)(*b + 1u);\n"
        "        *a = (uint8_t)(*a + 1u);\n"
        "    }\n"
        "}\n\n"
        "static int relation_spatial_prepare_composition(relation_spatial_preparation prep, relation_spatial_carrier_state *state, relation_spatial_relation_id first, relation_spatial_relation_id second) {\n"
        "    uint32_t step = 0u;\n"
        "    if (state == NULL || prep.cyclic_origin >= FAMILY10H_RELATION_SPATIAL_LINE_COUNT) {\n"
        "        return 0;\n"
        "    }\n"
        "    relation_spatial_prefault(state);\n"
        "    flush_state_lines(state);\n"
        "    for (step = 0u; step < (FAMILY10H_RELATION_SPATIAL_TOTAL_WORK / 2u); ++step) {\n"
        "        uint32_t a_index = relation_spatial_origin_index(prep.cyclic_origin, step);\n"
        "        relation_spatial_touch_pair(state, a_index, first, prep.source_order);\n"
        "        relation_spatial_touch_pair(state, a_index, second, prep.source_order);\n"
        "    }\n"
        "    return 1;\n"
        "}\n\n"
        "static int relation_spatial_prepare_composition_neutral(relation_spatial_preparation prep, relation_spatial_carrier_state *state) {\n"
        "    uint32_t step = 0u;\n"
        "    if (state == NULL || prep.cyclic_origin >= FAMILY10H_RELATION_SPATIAL_LINE_COUNT) {\n"
        "        return 0;\n"
        "    }\n"
        "    relation_spatial_prefault(state);\n"
        "    flush_state_lines(state);\n"
        "    for (step = 0u; step < (FAMILY10H_RELATION_SPATIAL_TOTAL_WORK / 2u); ++step) {\n"
        "        uint32_t a_index = relation_spatial_origin_index(prep.cyclic_origin, step);\n"
        "        if ((step & 1u) == 0u) {\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R0, prep.source_order);\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R1, prep.source_order);\n"
        "        } else {\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R1, prep.source_order);\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R0, prep.source_order);\n"
        "        }\n"
        "    }\n"
        "    return 1;\n"
        "}\n\n"
        "static int relation_spatial_prepare_composition_randomized(relation_spatial_preparation prep, relation_spatial_carrier_state *state) {\n"
        "    uint32_t step = 0u;\n"
        "    uint32_t x = prep.cyclic_origin ^ UINT32_C(0x9e3779b9);\n"
        "    if (state == NULL || prep.cyclic_origin >= FAMILY10H_RELATION_SPATIAL_LINE_COUNT) {\n"
        "        return 0;\n"
        "    }\n"
        "    relation_spatial_prefault(state);\n"
        "    flush_state_lines(state);\n"
        "    for (step = 0u; step < (FAMILY10H_RELATION_SPATIAL_TOTAL_WORK / 2u); ++step) {\n"
        "        uint32_t a_index = relation_spatial_origin_index(prep.cyclic_origin, step);\n"
        "        x ^= x << 13;\n"
        "        x ^= x >> 17;\n"
        "        x ^= x << 5;\n"
        "        if ((x & 1u) == 0u) {\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R0, prep.source_order);\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R1, prep.source_order);\n"
        "        } else {\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R1, prep.source_order);\n"
        "            relation_spatial_touch_pair(state, a_index, RELATION_SPATIAL_R0, prep.source_order);\n"
        "        }\n"
        "    }\n"
        "    return 1;\n"
        "}\n\n"
        "static int rdtscp_available(void) {\n",
        1,
    )
    text = text.replace(
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM;\n"
        "}\n\n"
        "static int source_off_control(relation_spatial_control_id control) {\n",
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int source_off_control(relation_spatial_control_id control) {\n",
        1,
    )
    text = text.replace(
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM;\n"
        "}\n\n"
        "static int reset_prefault_flush_after_source_dead_control(relation_spatial_control_id control) {\n",
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_LANE_A_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int reset_prefault_flush_after_source_dead_control(relation_spatial_control_id control) {\n",
        1,
    )
    text = text.replace(
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM;\n"
        "}\n\n"
        "static int reset_lane_a_flush_after_source_dead_control(relation_spatial_control_id control) {\n",
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int reset_lane_a_flush_after_source_dead_control(relation_spatial_control_id control) {\n",
        1,
    )
    text = text.replace(
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM;\n"
        "}\n\n"
        "static int mutate_same_control(relation_spatial_control_id control) {\n",
        "    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int mutate_same_control(relation_spatial_control_id control) {\n",
        1,
    )
    text = text.replace(
        "static int source_off_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM;\n"
        "}\n\n",
        "static int source_off_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int compose_r0_r1_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R0_R1_SHAM;\n"
        "}\n\n"
        "static int compose_r1_r0_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM;\n"
        "}\n\n"
        "static int neutral_compose_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_NEUTRAL_COMPOSE_SHAM;\n"
        "}\n\n"
        "static int random_compose_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM;\n"
        "}\n\n",
        1,
    )
    text = text.replace(
        "        relation_spatial_order_id effective_source_order = prep.source_order;\n"
        "        if (prep.relation == RELATION_SPATIAL_R1) {\n"
        "            effective_source_order = relation_spatial_opposite_order(effective_source_order);\n"
        "        }\n"
        "        if (effective_source_order == RELATION_SPATIAL_ORDER_AB) {\n",
        "        if (prep.source_order == RELATION_SPATIAL_ORDER_AB) {\n",
        1,
    )
    text = text.replace(
        "            : \"source_alive_during_spatial_pair_probe\"\n",
        "            ? (compose_r0_r1_control(row.control)\n"
        "                ? \"source_alive_compose_r0_then_r1_before_spatial_pair_probe\"\n"
        "                : (compose_r1_r0_control(row.control)\n"
        "                    ? \"source_alive_compose_r1_then_r0_before_spatial_pair_probe\"\n"
        "                    : (neutral_compose_control(row.control)\n"
        "                        ? \"source_alive_neutral_composition_before_spatial_pair_probe\"\n"
        "                        : (random_compose_control(row.control)\n"
        "                            ? \"source_alive_random_composition_before_spatial_pair_probe\"\n"
        "                            : \"source_alive_during_spatial_pair_probe\"))))\n",
        1,
    )
    text = text.replace(
        "            shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n",
        "            if (compose_r0_r1_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "            } else if (compose_r1_r0_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "            } else if (neutral_compose_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition_neutral(prep, &shared->state);\n"
        "            } else if (random_compose_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition_randomized(prep, &shared->state);\n"
        "            } else {\n"
        "                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "            }\n",
        1,
    )
    text = text.replace(
        "    if (!rdtscp_available() || m0 != 1u || m1 != 1u) {\n",
        "    if (!rdtscp_available() || m0 != 1u || m1 != (FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u)) {\n",
        1,
    )
    runtime.write_text(text, encoding="utf-8", newline="\n")


def source_order_for_variant(variant: str) -> str:
    return "A_then_B"


def target_script() -> str:
    return (
        BASE_TARGET_SCRIPT()
        .replace("RELATION_TEMPORAL_SOURCE_OFF", "RELATION_COMPOSITION_LOOP_RESET_SCREEN")
        .replace("TEMPORAL_SOURCE_OFF_SCHEDULES", "COMPOSITION_LOOP_RESET_SCREEN_SCHEDULES")
        .replace("family10h_relation_temporal_source_off_discovery_v1", PACKAGE_ID)
    )


def make_rows() -> dict[str, Any]:
    source_schedule = base.LOCAL_PAIRED_ROOT / "LOCAL_PAIRED_DIFFERENTIAL_SCHEDULES" / "round0_query_relation_pair.tsv"
    with source_schedule.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        base_rows = list(reader)[:ROWS_PER_VARIANT]
    require(len(base_rows) == ROWS_PER_VARIANT, "not enough base rows for discovery schedule")
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    schedules: dict[str, Any] = {}
    marginal_fields = [
        "block_local_position",
        "row_role",
        "q",
        "bank_A_work",
        "bank_B_work",
        "total_work",
        "r_prepare",
        "r_query",
        "relation_match",
        "relation_cell",
        "session",
        "replicate",
        "mapping",
        "delay_label",
        "delay_ns",
        "query_order",
        "cyclic_origin",
        "route_pressure_class",
        "distance_control_class",
        "allocation_order_class",
        "prefault_class",
        "source_cpu_expected",
        "receiver_cpu_expected",
        "source_loop_count",
        "receiver_loop_count",
        "read_count",
        "write_count",
        "page_count_A",
        "page_count_B",
        "line_count_A",
        "line_count_B",
        "expected_pmu_group",
        "requires_pmu",
        "post_observation_scheduling",
    ]
    marginal_digests: dict[str, str] = {}
    for variant in VARIANTS:
        query = QUERY_BY_VARIANT[variant]
        rows = []
        for index, row in enumerate(base_rows):
            copied = dict(row)
            copied["execution_ordinal"] = str(index)
            copied["query"] = query
            copied["source_order"] = source_order_for_variant(variant)
            copied["operation_semantics_id"] = f"cloop:{COMPOSITION_TAG_BY_VARIANT[variant]}:{variant}"
            copied["control_semantics_id"] = "none" if query in {"query_relation_pair", "relation_sham"} else query
            copied["tuple_id"] = f"{RUN_ID}:{variant}:{index:06d}:{sha256_bytes((variant + ':' + str(index)).encode())[:16]}"
            copied["block_id"] = f"{variant}_{copied['block_id']}"
            copied["matched_twin_group"] = f"{copied['block_id']}:cloop:{variant}"
            copied["matched_twin_pair"] = f"{copied['block_id']}:rp_{copied['block_local_position']}:{variant}"
            rows.append(copied)
        path = SCHEDULE_DIR / f"{variant}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
        marginal_digests[variant] = digest([{field: row[field] for field in marginal_fields} for row in rows])
        schedules[variant] = {
            "query": query,
            "composition_sequence": COMPOSITION_BY_VARIANT[variant],
            "source_order_inside_each_relation_pair": source_order_for_variant(variant),
            "path": f"COMPOSITION_LOOP_RESET_SCREEN_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "marginal_digest_excluding_composition_and_query": marginal_digests[variant],
            "sha256": base.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_COMPOSITION_LOOP_RESET_SCREEN_SCHEDULE_MANIFEST_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "source_schedule": str(source_schedule),
        "source_schedule_sha256": base.sha256_file(source_schedule),
        "rows_per_variant": ROWS_PER_VARIANT,
        "pair_sample_count": PAIR_SAMPLE_COUNT,
        "variants": schedules,
        "receiver_schedule_frozen_before_source_relation": True,
        "matched_marginal_law": {
            "spatial_addresses": "native R0 +1 and R1 -1 maps preserved; each composition variant uses the same A/B marginal line coverage",
            "work_counts": "copied unchanged from the same base rows for every variant; composition prep halves loop steps and touches two relation pairs per step",
            "receiver_query": "same four-cell held-out boundary projection in every source-order variant",
            "tested_relation": "relation composition sequence only: R0_then_R1 versus R1_then_R0, with double-flush carrier-off reset variants",
            "composition_sequence_excluded_from_marginal_digest": True,
            "query_string_excluded_from_marginal_digest": True,
        },
        "predeclared_patterns": {
            "H1_composition_sensitive_relational_carrier": [
                "R0_then_R1 and R1_then_R0 differ in held-out boundary coordinate after primary-minus-sham subtraction",
                "source-off composition difference collapses",
                "neutral and randomized composition controls do not replay the directed composition difference",
                "dead composition difference preserves a nontrivial fraction of alive composition difference",
            ],
            "H2_amplitude_or_occupancy_only": [
                "R0_then_R1 and R1_then_R0 are equivalent or both replay the same local differential",
                "source-off geometry is comparable",
            ],
            "H3_receiver_or_query_artifact": [
                "composition separation appears in source-off rows or tracks receiver query_order strata",
            ],
            "H4_timing_artifact": [
                "composition separation follows execution ordinal, duration, or delay rather than source composition",
            ],
        },
        "claim_boundary": {
            "exploratory_only": True,
            "positive_scientific_claim": False,
            "small_wall_crossed": False,
        },
    }
    manifest["schedule_manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "schedule_manifest_sha256"})
    write_json(SOURCE_ROOT / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_SCHEDULE_MANIFEST.json", manifest)
    write_json(HERE / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_SCHEDULE_MANIFEST.json", manifest)
    return manifest


def mean(values: list[float]) -> float:
    return base.mean(values)


def report_pair(reports: dict[str, Any], primary: str, sham: str) -> dict[str, Any]:
    primary_r = reports[primary]["R_spatial"]["mean"]
    sham_r = reports[sham]["R_spatial"]["mean"]
    return {
        "primary_variant": primary,
        "sham_variant": sham,
        "primary_R": primary_r,
        "sham_R": sham_r,
        "D_primary_minus_sham": primary_r - sham_r,
        "primary_positive": primary_r > 0,
        "sham_positive": sham_r > 0,
    }


def contrast_by_factor(
    records_by_variant: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
    factor: str,
) -> dict[str, Any]:
    left_records = records_by_variant[left]
    right_records = records_by_variant[right]
    levels = sorted(set(str(row[factor]) for row in left_records) & set(str(row[factor]) for row in right_records))
    values = {}
    for level in levels:
        lval = mean([row["R_spatial"] for row in left_records if str(row[factor]) == level])
        rval = mean([row["R_spatial"] for row in right_records if str(row[factor]) == level])
        values[level] = {
            "left_R": lval,
            "right_R": rval,
            "left_minus_right": lval - rval,
        }
    return {
        "levels": values,
        "all_same_sign_as_aggregate": None,
    }


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
            raw = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/raw_records.jsonl").read())  # type: ignore[union-attr]
            pairs = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/pair_observations.jsonl").read())  # type: ignore[union-attr]
            deaths = base.parse_jsonl_bytes(tf.extractfile(f"{prefix}/source_death_receipts.jsonl").read())  # type: ignore[union-attr]
            require(len(raw) == ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            require(len(pairs) == EXPECTED_PAIRS_PER_VARIANT, f"{variant} pair count mismatch: {len(pairs)}")
            require(len(deaths) == ROWS_PER_VARIANT, f"{variant} source-death count mismatch: {len(deaths)}")
            rows = base.row_c_pairs(raw, pairs)
            records = base.r_spatial_records(rows)
            axes = base.relation_cell_axes(rows)
            records_by_variant[variant] = records
            values = [record["R_spatial"] for record in records]
            reports[variant] = {
                "query": QUERY_BY_VARIANT[variant],
                "composition_sequence": COMPOSITION_BY_VARIANT[variant],
                "source_order_inside_each_relation_pair": source_order_for_variant(variant),
                "raw_record_count": len(raw),
                "pair_observation_count": len(pairs),
                "source_death_receipt_count": len(deaths),
                "block_count": len(records),
                "R_spatial": base.distribution(values),
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

    pair_reports = {
        "r0r1": report_pair(reports, "r0r1_primary", "r0r1_sham"),
        "r1r0": report_pair(reports, "r1r0_primary", "r1r0_sham"),
        "dead_r0r1": report_pair(reports, "dead_r0r1_primary", "dead_r0r1_sham"),
        "dead_r1r0": report_pair(reports, "dead_r1r0_primary", "dead_r1r0_sham"),
        "reset_r0r1": report_pair(reports, "reset_r0r1_primary", "reset_r0r1_sham"),
        "reset_r1r0": report_pair(reports, "reset_r1r0_primary", "reset_r1r0_sham"),
        "source_off_r0r1": report_pair(reports, "source_off_r0r1_primary", "source_off_r0r1_sham"),
        "source_off_r1r0": report_pair(reports, "source_off_r1r0_primary", "source_off_r1r0_sham"),
        "neutral": report_pair(reports, "neutral_primary", "neutral_sham"),
        "random": report_pair(reports, "random_primary", "random_sham"),
    }
    alive_primary_order = reports["r0r1_primary"]["R_spatial"]["mean"] - reports["r1r0_primary"]["R_spatial"]["mean"]
    alive_composition_order = (
        pair_reports["r0r1"]["D_primary_minus_sham"] - pair_reports["r1r0"]["D_primary_minus_sham"]
    )
    dead_composition_order = (
        pair_reports["dead_r0r1"]["D_primary_minus_sham"] - pair_reports["dead_r1r0"]["D_primary_minus_sham"]
    )
    reset_composition_order = (
        pair_reports["reset_r0r1"]["D_primary_minus_sham"] - pair_reports["reset_r1r0"]["D_primary_minus_sham"]
    )
    source_off_composition_order = (
        pair_reports["source_off_r0r1"]["D_primary_minus_sham"]
        - pair_reports["source_off_r1r0"]["D_primary_minus_sham"]
    )
    denom = abs(alive_composition_order)
    factors = ["session", "replicate", "mapping", "query_order", "cyclic_origin"]
    stratum_composition_contrasts = {
        factor: contrast_by_factor(records_by_variant, "r0r1_primary", "r1r0_primary", factor)
        for factor in factors
    }
    reset_stratum_composition_contrasts = {
        factor: contrast_by_factor(records_by_variant, "reset_r0r1_primary", "reset_r1r0_primary", factor)
        for factor in factors
    }
    for factor, report in stratum_composition_contrasts.items():
        aggregate_sign = 1 if alive_primary_order > 0 else -1 if alive_primary_order < 0 else 0
        report["all_same_sign_as_aggregate"] = all(
            (1 if item["left_minus_right"] > 0 else -1 if item["left_minus_right"] < 0 else 0) == aggregate_sign
            for item in report["levels"].values()
        )
    for factor, report in reset_stratum_composition_contrasts.items():
        report["all_below_25pct_alive_matched"] = all(
            (
                abs(item["left_minus_right"])
                <= 0.25
                * abs(stratum_composition_contrasts[factor]["levels"][level]["left_minus_right"])
                if stratum_composition_contrasts[factor]["levels"][level]["left_minus_right"]
                else False
            )
            for level, item in report["levels"].items()
            if level in stratum_composition_contrasts[factor]["levels"]
        )

    result = {
        "schema": "FAMILY10H_RELATION_COMPOSITION_LOOP_RESET_SCREEN_ANALYSIS_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": base.sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "variant_reports": reports,
        "pair_reports": pair_reports,
        "stratum_composition_contrasts": stratum_composition_contrasts,
        "reset_stratum_composition_contrasts": reset_stratum_composition_contrasts,
        "coordinates": {
            "R_r0r1_primary": reports["r0r1_primary"]["R_spatial"]["mean"],
            "R_r1r0_primary": reports["r1r0_primary"]["R_spatial"]["mean"],
            "R_alive_primary_composition_r0r1_minus_r1r0": alive_primary_order,
            "D_r0r1_primary_minus_sham": pair_reports["r0r1"]["D_primary_minus_sham"],
            "D_r1r0_primary_minus_sham": pair_reports["r1r0"]["D_primary_minus_sham"],
            "Omega_alive_composition_r0r1_minus_r1r0": alive_composition_order,
            "Omega_dead_composition_r0r1_minus_r1r0": dead_composition_order,
            "Omega_reset_double_flush_composition_r0r1_minus_r1r0": reset_composition_order,
            "Omega_source_off_composition_r0r1_minus_r1r0": source_off_composition_order,
            "source_off_composition_abs_to_alive_composition_abs": abs(source_off_composition_order) / denom if denom else None,
            "dead_composition_abs_to_alive_composition_abs": abs(dead_composition_order) / denom if denom else None,
            "reset_composition_abs_to_alive_composition_abs": abs(reset_composition_order) / denom if denom else None,
            "D_neutral_primary_minus_sham": pair_reports["neutral"]["D_primary_minus_sham"],
            "D_random_primary_minus_sham": pair_reports["random"]["D_primary_minus_sham"],
            "neutral_abs_to_alive_composition_abs": abs(pair_reports["neutral"]["D_primary_minus_sham"]) / denom if denom else None,
            "random_abs_to_alive_composition_abs": abs(pair_reports["random"]["D_primary_minus_sham"]) / denom if denom else None,
        },
        "interpretation": {
            "alive_composition_order_nonzero": abs(alive_composition_order) > 0.0,
            "dead_preserves_at_least_25pct_alive_composition_order": (
                abs(dead_composition_order) >= 0.25 * denom if denom else False
            ),
            "source_off_collapses_below_25pct_alive_composition_order": (
                abs(source_off_composition_order) <= 0.25 * denom if denom else False
            ),
            "reset_double_flush_collapses_below_25pct_alive_composition_order": (
                abs(reset_composition_order) <= 0.25 * denom if denom else False
            ),
            "reset_double_flush_all_one_factor_strata_below_25pct_alive_matched": all(
                report["all_below_25pct_alive_matched"] for report in reset_stratum_composition_contrasts.values()
            ),
            "composition_contrast_all_one_factor_strata_same_sign": all(
                report["all_same_sign_as_aggregate"] for report in stratum_composition_contrasts.values()
            ),
            "composition_loop_reset_candidate": (
                denom > 0.0
                and abs(dead_composition_order) >= 0.25 * denom
                and abs(source_off_composition_order) <= 0.25 * denom
                and abs(reset_composition_order) <= 0.25 * denom
                and all(report["all_same_sign_as_aggregate"] for report in stratum_composition_contrasts.values())
                and all(
                    report["all_below_25pct_alive_matched"]
                    for report in reset_stratum_composition_contrasts.values()
                )
            ),
            "exploratory_only": True,
            "small_wall_crossed": False,
        },
        "claim_boundary": {
            "positive_scientific_claim": False,
            "small_wall_crossed": False,
            "full_tomography_established": False,
            "holographic_relational_invariant_established": False,
            "r2_restoration_established": False,
        },
    }
    result["analysis_sha256"] = digest({k: v for k, v in result.items() if k != "analysis_sha256"})
    write_json(RUN_ROOT / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_ANALYSIS.json", result)
    write_json(SUMMARY_JSON, result)
    lines = [
        "# Relation Composition Loop Reset Screen",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "| Pair | Primary R | Sham R | D primary-minus-sham |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "r0r1",
        "r1r0",
        "dead_r0r1",
        "dead_r1r0",
        "reset_r0r1",
        "reset_r1r0",
        "source_off_r0r1",
        "source_off_r1r0",
        "neutral",
        "random",
    ]:
        report = pair_reports[key]
        lines.append(
            f"| {key} | {report['primary_R']:.9f} | {report['sham_R']:.9f} | {report['D_primary_minus_sham']:.9f} |"
        )
    lines.extend(
        [
            "",
            "Composition coordinates:",
            f"- alive Omega composition R0->R1 minus R1->R0: `{alive_composition_order:.9f}`",
            f"- dead Omega composition R0->R1 minus R1->R0: `{dead_composition_order:.9f}`",
            f"- reset double-flush Omega composition R0->R1 minus R1->R0: `{reset_composition_order:.9f}`",
            f"- source-off Omega composition R0->R1 minus R1->R0: `{source_off_composition_order:.9f}`",
            f"- source-off abs/alive abs: `{(abs(source_off_composition_order) / denom if denom else 0.0):.3f}`",
            f"- dead abs/alive abs: `{(abs(dead_composition_order) / denom if denom else 0.0):.3f}`",
            f"- reset abs/alive abs: `{(abs(reset_composition_order) / denom if denom else 0.0):.3f}`",
            "",
            "Interpretation:",
            f"- composition-loop reset candidate: `{result['interpretation']['composition_loop_reset_candidate']}`",
            f"- source-off collapses below 0.25 x alive composition: `{result['interpretation']['source_off_collapses_below_25pct_alive_composition_order']}`",
            f"- dead preserves at least 0.25 x alive composition: `{result['interpretation']['dead_preserves_at_least_25pct_alive_composition_order']}`",
            f"- reset double-flush collapses below 0.25 x alive composition: `{result['interpretation']['reset_double_flush_collapses_below_25pct_alive_composition_order']}`",
            f"- reset double-flush all one-factor strata below 0.25 x matched alive: `{result['interpretation']['reset_double_flush_all_one_factor_strata_below_25pct_alive_matched']}`",
            f"- one-factor strata same sign: `{result['interpretation']['composition_contrast_all_one_factor_strata_same_sign']}`",
            "",
            "This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare() -> dict[str, Any]:
    configure_base()
    base.copy_segmented_source()
    patch_runtime_for_composition_loop()
    schedule_manifest = make_rows()
    build = base.compile_runtime()
    result = {
        "schema": "FAMILY10H_RELATION_COMPOSITION_LOOP_RESET_SCREEN_PREPARE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    write_json(HERE / "RELATION_COMPOSITION_LOOP_RESET_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_base()
    base.make_rows = make_rows
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    require(prepare_result["passed"], "prepare failed")
    package = base.build_package()
    controller = base.deploy_execute_copyback(package)
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
