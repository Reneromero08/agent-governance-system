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


RUN_ID = "family10h_relation_source_map_odd_composition_lifecycle_probe_v1_0"
PACKAGE_ID = "family10h_relation_source_map_odd_composition_lifecycle_probe_v1"
ROWS_PER_VARIANT = 512
PAIR_SAMPLE_COUNT = 256
EXPECTED_PAIRS_PER_VARIANT = ROWS_PER_VARIANT * PAIR_SAMPLE_COUNT

PRESENTATIONS = ["native", "swapped"]
LIFECYCLES = ["alive", "source_off", "dead"]
COMPOSITION_ORDERS = ["r0r1", "r1r0"]
ROLES = ["primary", "sham"]
VARIANTS = [
    f"{presentation}_{lifecycle}_{order}_{role}"
    for presentation in PRESENTATIONS
    for lifecycle in LIFECYCLES
    for order in COMPOSITION_ORDERS
    for role in ROLES
]


def variant_parts(variant: str) -> tuple[str, str, str, str]:
    for presentation in PRESENTATIONS:
        prefix = f"{presentation}_"
        if not variant.startswith(prefix):
            continue
        after_presentation = variant[len(prefix) :]
        for lifecycle in LIFECYCLES:
            life_prefix = f"{lifecycle}_"
            if not after_presentation.startswith(life_prefix):
                continue
            after_lifecycle = after_presentation[len(life_prefix) :]
            for order in COMPOSITION_ORDERS:
                order_prefix = f"{order}_"
                if not after_lifecycle.startswith(order_prefix):
                    continue
                role = after_lifecycle[len(order_prefix) :]
                if role in ROLES:
                    return presentation, lifecycle, order, role
    raise ValueError(f"invalid source-map-odd composition variant: {variant}")


def compact_variant_tag(variant: str) -> str:
    presentation, lifecycle, order, role = variant_parts(variant)
    pres_tag = {"native": "n", "swapped": "s"}[presentation]
    life_tag = {"alive": "alv", "source_off": "off", "dead": "ded"}[lifecycle]
    role_tag = {"primary": "p", "sham": "h"}[role]
    return f"{pres_tag}_{life_tag}_{order}_{role_tag}"


def control_enum_name(variant: str) -> str:
    presentation, lifecycle, order, role = variant_parts(variant)
    life_tag = {"alive": "ALIVE", "source_off": "SOURCE_OFF", "dead": "DEAD"}[lifecycle]
    return (
        "RELATION_SPATIAL_CONTROL_MAPODD_"
        f"{presentation.upper()}_{life_tag}_{order.upper()}_{role.upper()}"
    )


def query_for_variant(variant: str) -> str:
    return f"mop_{compact_variant_tag(variant)}_control"


QUERY_BY_VARIANT = {variant: query_for_variant(variant) for variant in VARIANTS}
COMPOSITION_BY_VARIANT = {
    variant: f"{variant_parts(variant)[0]}_{variant_parts(variant)[2]}_{variant_parts(variant)[1]}"
    for variant in VARIANTS
}
COMPOSITION_TAG_BY_VARIANT = {
    variant: f"{variant_parts(variant)[0]}_{variant_parts(variant)[1]}_{variant_parts(variant)[2]}"
    for variant in VARIANTS
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
SCHEDULE_DIR = SOURCE_ROOT / "SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULES"
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / "attempt_1"
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SUMMARY.md"


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
    mapodd_enum_lines = [
        f"    {control_enum_name(variant)} = {41 + index}"
        for index, variant in enumerate(VARIANTS)
    ]
    require(
        "RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM = 40" in htext,
        "map-odd enum anchor not found",
    )
    htext = htext.replace(
        "RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM = 40",
        "RELATION_SPATIAL_CONTROL_RESET_COMPOSE_R1_R0_SHAM = 40,\n" + ",\n".join(mapodd_enum_lines),
        1,
    )
    header.write_text(htext, encoding="utf-8", newline="\n")

    text = runtime.read_text(encoding="utf-8")
    mapodd_parse_cases = "".join(
        f'    if (strcmp(query, "{QUERY_BY_VARIANT[variant]}") == 0) {{\n'
        f"        *out = {control_enum_name(variant)};\n"
        "        return 1;\n"
        "    }\n"
        for variant in VARIANTS
    )

    def control_lines(variants: list[str], subject: str) -> str:
        return "".join(f"        || {subject} == {control_enum_name(variant)}\n" for variant in variants)

    def control_return_function(name: str, variants: list[str]) -> str:
        lines = [f"static int {name}(relation_spatial_control_id control) {{\n"]
        first, *rest = variants
        lines.append(f"    return control == {control_enum_name(first)}\n")
        lines.extend(f"        || control == {control_enum_name(variant)}\n" for variant in rest[:-1])
        if rest:
            lines.append(f"        || control == {control_enum_name(rest[-1])};\n")
        else:
            lines[-1] = lines[-1].rstrip("\n") + ";\n"
        lines.append("}\n\n")
        return "".join(lines)

    sham_variants = [variant for variant in VARIANTS if variant_parts(variant)[3] == "sham"]
    source_dead_variants = [
        variant for variant in VARIANTS if variant_parts(variant)[1] in {"source_off", "dead"}
    ]
    source_off_variants = [variant for variant in VARIANTS if variant_parts(variant)[1] == "source_off"]
    native_r0r1_variants = [
        variant
        for variant in VARIANTS
        if variant_parts(variant)[0] == "native" and variant_parts(variant)[2] == "r0r1"
    ]
    native_r1r0_variants = [
        variant
        for variant in VARIANTS
        if variant_parts(variant)[0] == "native" and variant_parts(variant)[2] == "r1r0"
    ]
    swapped_r0r1_variants = [
        variant
        for variant in VARIANTS
        if variant_parts(variant)[0] == "swapped" and variant_parts(variant)[2] == "r0r1"
    ]
    swapped_r1r0_variants = [
        variant
        for variant in VARIANTS
        if variant_parts(variant)[0] == "swapped" and variant_parts(variant)[2] == "r1r0"
    ]
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
    parse_anchor = "    return 0;\n}\n\nstatic int split_tsv"
    require(parse_anchor in text, "map-odd parse insertion anchor not found")
    text = text.replace(
        parse_anchor,
        mapodd_parse_cases + "    return 0;\n}\n\nstatic int split_tsv",
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
    sham_anchor = "        || row->control == RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM) {\n"
    require(sham_anchor in text, "map-odd sham B-index insertion anchor not found")
    text = text.replace(
        sham_anchor,
        "        || row->control == RELATION_SPATIAL_CONTROL_RANDOM_COMPOSE_SHAM\n"
        + "".join(f"        || row->control == {control_enum_name(variant)}\n" for variant in sham_variants[:-1])
        + f"        || row->control == {control_enum_name(sham_variants[-1])}) {{\n",
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
    dead_anchor = "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
    require(dead_anchor in text, "map-odd source-dead lifecycle insertion anchor not found")
    text = text.replace(
        dead_anchor,
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
        + "".join(f"        || control == {control_enum_name(variant)}\n" for variant in source_dead_variants[:-1])
        + f"        || control == {control_enum_name(source_dead_variants[-1])};\n",
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
    source_off_anchor = "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
    require(source_off_anchor in text, "map-odd source-off lifecycle insertion anchor not found")
    text = text.replace(
        source_off_anchor,
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
        + "".join(f"        || control == {control_enum_name(variant)}\n" for variant in source_off_variants[:-1])
        + f"        || control == {control_enum_name(source_off_variants[-1])};\n",
        1,
    )
    helper_anchor = "static int neutral_compose_control(relation_spatial_control_id control) {\n"
    require(helper_anchor in text, "map-odd helper insertion anchor not found")
    text = text.replace(
        helper_anchor,
        control_return_function("mapodd_native_r0_r1_control", native_r0r1_variants)
        + control_return_function("mapodd_native_r1_r0_control", native_r1r0_variants)
        + control_return_function("mapodd_swapped_r0_r1_control", swapped_r0r1_variants)
        + control_return_function("mapodd_swapped_r1_r0_control", swapped_r1r0_variants)
        + helper_anchor,
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
        "            if (mapodd_native_r0_r1_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "            } else if (mapodd_native_r1_r0_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "            } else if (mapodd_swapped_r0_r1_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "            } else if (mapodd_swapped_r1_r0_control(row.control)) {\n"
        "                shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "            } else if (compose_r0_r1_control(row.control)) {\n"
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
    live_prepare_anchor = (
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
    )
    require(live_prepare_anchor in text, "map-odd live source-preparation anchor not found")
    text = text.replace(
        live_prepare_anchor,
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                if (mapodd_native_r0_r1_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "                } else if (mapodd_native_r1_r0_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "                } else if (mapodd_swapped_r0_r1_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "                } else if (mapodd_swapped_r1_r0_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "                } else if (compose_r0_r1_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "                } else if (compose_r1_r0_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "                } else if (neutral_compose_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_neutral(prep, &shared->state);\n"
        "                } else if (random_compose_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_randomized(prep, &shared->state);\n"
        "                } else {\n"
        "                    shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "                }\n",
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
        .replace("RELATION_TEMPORAL_SOURCE_OFF", "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE")
        .replace("TEMPORAL_SOURCE_OFF_SCHEDULES", "SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULES")
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
        tag = compact_variant_tag(variant)
        rows = []
        for index, row in enumerate(base_rows):
            copied = dict(row)
            copied["execution_ordinal"] = str(index)
            copied["query"] = query
            copied["source_order"] = source_order_for_variant(variant)
            copied["operation_semantics_id"] = f"mop:{tag}"
            copied["control_semantics_id"] = "none" if query in {"query_relation_pair", "relation_sham"} else query
            row_digest = sha256_bytes((variant + ":" + str(index)).encode())[:16]
            copied["tuple_id"] = f"{RUN_ID}:{tag}:{index:06d}:{row_digest}"
            copied["block_id"] = f"{tag}_{copied['block_id']}"
            matched_id = sha256_bytes(copied["block_id"].encode())[:16]
            copied["matched_twin_group"] = f"mop_{tag}_{matched_id}"
            copied["matched_twin_pair"] = f"mop_{tag}_{copied['block_local_position']}_{matched_id}"
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
            "compact_variant_tag": tag,
            "runtime_control": control_enum_name(variant),
            "source_order_inside_each_relation_pair": source_order_for_variant(variant),
            "path": f"SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULES/{variant}.tsv",
            "row_count": len(rows),
            "expected_pair_observation_count": len(rows) * PAIR_SAMPLE_COUNT,
            "marginal_digest_excluding_composition_and_query": marginal_digests[variant],
            "sha256": base.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "FAMILY10H_RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULE_MANIFEST_V1",
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
            "tested_relation": "source-map-odd composition orientation only: native versus swapped presentation, R0_then_R1 versus R1_then_R0, alive/source-off/dead lifecycle",
            "composition_sequence_excluded_from_marginal_digest": True,
            "query_string_excluded_from_marginal_digest": True,
        },
        "predeclared_patterns": {
            "H1_source_map_odd_composition_sensitive_carrier": [
                "native Omega = D(R0_then_R1) - D(R1_then_R0) is nonzero after primary-minus-sham subtraction",
                "swapped presentation reverses the native Omega sign under identical marginal work",
                "Gamma = (Omega_native - Omega_swapped) / 2 is stable in one-factor strata",
                "source-off Gamma collapses below 0.25 x alive Gamma",
                "dead Gamma preserves sign and at least 0.25 x alive Gamma",
            ],
            "H2_amplitude_or_occupancy_only": [
                "native and swapped presentations replay the same sign or a common even-mode amplitude",
                "source-off geometry is comparable",
            ],
            "H3_receiver_or_query_artifact": [
                "Gamma separation appears in source-off rows or tracks receiver query_order strata",
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
    write_json(SOURCE_ROOT / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULE_MANIFEST.json", manifest)
    write_json(HERE / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_SCHEDULE_MANIFEST.json", manifest)
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

    def variant_name(presentation: str, lifecycle: str, order: str, role: str) -> str:
        return f"{presentation}_{lifecycle}_{order}_{role}"

    def pair_key(presentation: str, lifecycle: str, order: str) -> str:
        return f"{presentation}_{lifecycle}_{order}"

    pair_reports = {
        pair_key(presentation, lifecycle, order): report_pair(
            reports,
            variant_name(presentation, lifecycle, order, "primary"),
            variant_name(presentation, lifecycle, order, "sham"),
        )
        for presentation in PRESENTATIONS
        for lifecycle in LIFECYCLES
        for order in COMPOSITION_ORDERS
    }

    def d_value(presentation: str, lifecycle: str, order: str) -> float:
        return pair_reports[pair_key(presentation, lifecycle, order)]["D_primary_minus_sham"]

    def omega(presentation: str, lifecycle: str) -> float:
        return d_value(presentation, lifecycle, "r0r1") - d_value(presentation, lifecycle, "r1r0")

    def gamma(lifecycle: str) -> float:
        return (omega("native", lifecycle) - omega("swapped", lifecycle)) / 2.0

    def even_mode(lifecycle: str) -> float:
        return (omega("native", lifecycle) + omega("swapped", lifecycle)) / 2.0

    def sign(value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    gamma_by_lifecycle = {lifecycle: gamma(lifecycle) for lifecycle in LIFECYCLES}
    omega_by_presentation_lifecycle = {
        f"{presentation}_{lifecycle}": omega(presentation, lifecycle)
        for presentation in PRESENTATIONS
        for lifecycle in LIFECYCLES
    }
    even_by_lifecycle = {lifecycle: even_mode(lifecycle) for lifecycle in LIFECYCLES}
    denom = abs(gamma_by_lifecycle["alive"])
    aggregate_sign = sign(gamma_by_lifecycle["alive"])

    def d_by_factor(presentation: str, lifecycle: str, order: str, factor: str) -> dict[str, Any]:
        primary = variant_name(presentation, lifecycle, order, "primary")
        sham = variant_name(presentation, lifecycle, order, "sham")
        primary_records = records_by_variant[primary]
        sham_records = records_by_variant[sham]
        levels = sorted(set(str(row[factor]) for row in primary_records) & set(str(row[factor]) for row in sham_records))
        result: dict[str, Any] = {}
        for level in levels:
            primary_r = mean([row["R_spatial"] for row in primary_records if str(row[factor]) == level])
            sham_r = mean([row["R_spatial"] for row in sham_records if str(row[factor]) == level])
            result[level] = {
                "primary_R": primary_r,
                "sham_R": sham_r,
                "D_primary_minus_sham": primary_r - sham_r,
            }
        return result

    def gamma_by_factor(lifecycle: str, factor: str) -> dict[str, Any]:
        d_maps = {
            (presentation, order): d_by_factor(presentation, lifecycle, order, factor)
            for presentation in PRESENTATIONS
            for order in COMPOSITION_ORDERS
        }
        common_levels = sorted(set.intersection(*(set(values.keys()) for values in d_maps.values())))
        levels: dict[str, Any] = {}
        for level in common_levels:
            native_omega = d_maps[("native", "r0r1")][level]["D_primary_minus_sham"] - d_maps[("native", "r1r0")][level]["D_primary_minus_sham"]
            swapped_omega = d_maps[("swapped", "r0r1")][level]["D_primary_minus_sham"] - d_maps[("swapped", "r1r0")][level]["D_primary_minus_sham"]
            local_gamma = (native_omega - swapped_omega) / 2.0
            local_even = (native_omega + swapped_omega) / 2.0
            levels[level] = {
                "omega_native": native_omega,
                "omega_swapped": swapped_omega,
                "gamma": local_gamma,
                "even_mode": local_even,
                "native_swapped_sign_reversal": sign(native_omega) != 0 and sign(native_omega) == -sign(swapped_omega),
                "even_abs_to_gamma_abs": abs(local_even) / abs(local_gamma) if local_gamma else None,
            }
        return {"levels": levels}

    factors = ["session", "replicate", "mapping", "query_order", "cyclic_origin"]
    stratum_gamma = {
        factor: {lifecycle: gamma_by_factor(lifecycle, factor) for lifecycle in LIFECYCLES}
        for factor in factors
    }
    for factor, by_lifecycle in stratum_gamma.items():
        alive_levels = by_lifecycle["alive"]["levels"]
        off_levels = by_lifecycle["source_off"]["levels"]
        dead_levels = by_lifecycle["dead"]["levels"]
        common_levels = sorted(set(alive_levels) & set(off_levels) & set(dead_levels))
        by_lifecycle["summary"] = {
            "common_level_count": len(common_levels),
            "alive_gamma_all_same_sign_as_aggregate": all(
                sign(alive_levels[level]["gamma"]) == aggregate_sign for level in common_levels
            ),
            "source_off_all_below_25pct_matched_alive_gamma": all(
                abs(off_levels[level]["gamma"]) <= 0.25 * abs(alive_levels[level]["gamma"])
                for level in common_levels
                if alive_levels[level]["gamma"]
            ),
            "dead_all_preserve_same_sign_and_at_least_25pct_alive_gamma": all(
                sign(dead_levels[level]["gamma"]) == sign(alive_levels[level]["gamma"])
                and abs(dead_levels[level]["gamma"]) >= 0.25 * abs(alive_levels[level]["gamma"])
                for level in common_levels
            ),
            "alive_native_swapped_reversal_all_levels": all(
                alive_levels[level]["native_swapped_sign_reversal"] for level in common_levels
            ),
            "alive_even_mode_below_25pct_gamma_all_levels": all(
                alive_levels[level]["even_abs_to_gamma_abs"] is not None
                and alive_levels[level]["even_abs_to_gamma_abs"] <= 0.25
                for level in common_levels
            ),
        }

    source_off_ratio = abs(gamma_by_lifecycle["source_off"]) / denom if denom else None
    dead_ratio = abs(gamma_by_lifecycle["dead"]) / denom if denom else None
    alive_even_ratio = abs(even_by_lifecycle["alive"]) / denom if denom else None
    aggregate_swapped_reversal = (
        sign(omega_by_presentation_lifecycle["native_alive"]) != 0
        and sign(omega_by_presentation_lifecycle["native_alive"])
        == -sign(omega_by_presentation_lifecycle["swapped_alive"])
    )
    aggregate_dead_preserves = (
        sign(gamma_by_lifecycle["dead"]) == aggregate_sign and abs(gamma_by_lifecycle["dead"]) >= 0.25 * denom
        if denom
        else False
    )
    aggregate_source_off_collapses = abs(gamma_by_lifecycle["source_off"]) <= 0.25 * denom if denom else False
    aggregate_even_small = abs(even_by_lifecycle["alive"]) <= 0.25 * denom if denom else False
    strata_alive_sign = all(
        report["summary"]["alive_gamma_all_same_sign_as_aggregate"] for report in stratum_gamma.values()
    )
    strata_source_off_collapse = all(
        report["summary"]["source_off_all_below_25pct_matched_alive_gamma"] for report in stratum_gamma.values()
    )
    strata_dead_preserve = all(
        report["summary"]["dead_all_preserve_same_sign_and_at_least_25pct_alive_gamma"]
        for report in stratum_gamma.values()
    )
    strata_swapped_reversal = all(
        report["summary"]["alive_native_swapped_reversal_all_levels"] for report in stratum_gamma.values()
    )
    strata_even_small = all(
        report["summary"]["alive_even_mode_below_25pct_gamma_all_levels"] for report in stratum_gamma.values()
    )

    result = {
        "schema": "FAMILY10H_RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_ANALYSIS_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller.get("passed") is True,
        "archive_sha256": base.sha256_file(LOCAL_ARCHIVE),
        "archive_size": LOCAL_ARCHIVE.stat().st_size,
        "variant_reports": reports,
        "pair_reports": pair_reports,
        "stratum_gamma": stratum_gamma,
        "coordinates": {
            "omega_by_presentation_lifecycle": omega_by_presentation_lifecycle,
            "gamma_by_lifecycle": gamma_by_lifecycle,
            "even_mode_by_lifecycle": even_by_lifecycle,
            "Gamma_alive": gamma_by_lifecycle["alive"],
            "Gamma_source_off": gamma_by_lifecycle["source_off"],
            "Gamma_dead": gamma_by_lifecycle["dead"],
            "source_off_gamma_abs_to_alive_gamma_abs": source_off_ratio,
            "dead_gamma_abs_to_alive_gamma_abs": dead_ratio,
            "alive_even_mode_abs_to_gamma_abs": alive_even_ratio,
        },
        "interpretation": {
            "alive_gamma_nonzero": abs(gamma_by_lifecycle["alive"]) > 0.0,
            "aggregate_native_swapped_reversal": aggregate_swapped_reversal,
            "aggregate_alive_even_mode_below_25pct_gamma": aggregate_even_small,
            "aggregate_source_off_gamma_collapses_below_25pct_alive_gamma": aggregate_source_off_collapses,
            "aggregate_dead_gamma_preserves_same_sign_and_at_least_25pct_alive_gamma": aggregate_dead_preserves,
            "strata_alive_gamma_all_same_sign_as_aggregate": strata_alive_sign,
            "strata_native_swapped_reversal_all_levels": strata_swapped_reversal,
            "strata_alive_even_mode_below_25pct_gamma_all_levels": strata_even_small,
            "strata_source_off_gamma_collapses_below_25pct_matched_alive_gamma": strata_source_off_collapse,
            "strata_dead_gamma_preserves_same_sign_and_at_least_25pct_matched_alive_gamma": strata_dead_preserve,
            "source_map_odd_composition_lifecycle_candidate": (
                denom > 0.0
                and aggregate_swapped_reversal
                and aggregate_even_small
                and aggregate_source_off_collapses
                and aggregate_dead_preserves
                and strata_alive_sign
                and strata_swapped_reversal
                and strata_even_small
                and strata_source_off_collapse
                and strata_dead_preserve
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
    write_json(RUN_ROOT / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_ANALYSIS.json", result)
    write_json(SUMMARY_JSON, result)
    lines = [
        "# Relation Source Map Odd Composition Lifecycle Probe",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{result['archive_sha256']}`",
        f"Analysis SHA-256: `{result['analysis_sha256']}`",
        "",
        "| Pair | Primary R | Sham R | D primary-minus-sham |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(pair_reports):
        report = pair_reports[key]
        lines.append(
            f"| {key} | {report['primary_R']:.9f} | {report['sham_R']:.9f} | {report['D_primary_minus_sham']:.9f} |"
        )
    lines.extend(
        [
            "",
            "Source-map-odd composition coordinates:",
            f"- Omega native alive: `{omega_by_presentation_lifecycle['native_alive']:.9f}`",
            f"- Omega swapped alive: `{omega_by_presentation_lifecycle['swapped_alive']:.9f}`",
            f"- Gamma alive: `{gamma_by_lifecycle['alive']:.9f}`",
            f"- Gamma source-off: `{gamma_by_lifecycle['source_off']:.9f}`",
            f"- Gamma dead: `{gamma_by_lifecycle['dead']:.9f}`",
            f"- source-off |Gamma| / alive |Gamma|: `{(source_off_ratio if source_off_ratio is not None else 0.0):.3f}`",
            f"- dead |Gamma| / alive |Gamma|: `{(dead_ratio if dead_ratio is not None else 0.0):.3f}`",
            f"- alive even-mode |E| / |Gamma|: `{(alive_even_ratio if alive_even_ratio is not None else 0.0):.3f}`",
            "",
            "Interpretation:",
            f"- source-map-odd composition lifecycle candidate: `{result['interpretation']['source_map_odd_composition_lifecycle_candidate']}`",
            f"- native/swapped reversal aggregate: `{result['interpretation']['aggregate_native_swapped_reversal']}`",
            f"- source-off Gamma collapses below 0.25 x alive Gamma: `{result['interpretation']['aggregate_source_off_gamma_collapses_below_25pct_alive_gamma']}`",
            f"- dead Gamma preserves same sign and >= 0.25 x alive Gamma: `{result['interpretation']['aggregate_dead_gamma_preserves_same_sign_and_at_least_25pct_alive_gamma']}`",
            f"- alive Gamma same sign in one-factor strata: `{result['interpretation']['strata_alive_gamma_all_same_sign_as_aggregate']}`",
            f"- source-off Gamma collapses in one-factor strata: `{result['interpretation']['strata_source_off_gamma_collapses_below_25pct_matched_alive_gamma']}`",
            f"- dead Gamma preserves in one-factor strata: `{result['interpretation']['strata_dead_gamma_preserves_same_sign_and_at_least_25pct_matched_alive_gamma']}`",
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
        "schema": "FAMILY10H_RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_PREPARE_V1",
        "created_at": utc_now(),
        "run_id": RUN_ID,
        "schedule_manifest_sha256": schedule_manifest["schedule_manifest_sha256"],
        "runtime_sha256": build["runtime_sha256"],
        "pmu_helper_sha256": build["pmu_helper_sha256"],
        "passed": build["passed"],
        "small_wall_crossed": False,
    }
    write_json(HERE / "RELATION_SOURCE_MAP_ODD_COMPOSITION_LIFECYCLE_PROBE_PREPARE_RESULT.json", result)
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
