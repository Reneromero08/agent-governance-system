#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
TRUE_RUNNER = (
    CARRIER_ROOT
    / "family10h_relation_true_composition_replay_screen_v1"
    / "run_relation_true_composition_replay_screen.py"
)

spec = importlib.util.spec_from_file_location("true_composition_base", TRUE_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load true-composition runner: {TRUE_RUNNER}")
true = importlib.util.module_from_spec(spec)
spec.loader.exec_module(true)

RUN_ID = "family10h_relation_palindrome_holonomy_source_off_screen_v1_0"
PACKAGE_ID = "family10h_relation_palindrome_holonomy_source_off_screen_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_SCHEDULES"
ATTEMPT_LABEL = os.environ.get("FAMILY10H_DISCOVERY_ATTEMPT_LABEL", "attempt_1")
if any(sep in ATTEMPT_LABEL for sep in ("/", "\\", ":")) or not ATTEMPT_LABEL:
    raise RuntimeError(f"invalid attempt label: {ATTEMPT_LABEL!r}")
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / ATTEMPT_LABEL
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_SUMMARY.md"

VARIANTS = [
    "palindrome_a_primary",
    "palindrome_a_sham",
    "palindrome_b_primary",
    "palindrome_b_sham",
    "dead_palindrome_a_primary",
    "dead_palindrome_a_sham",
    "dead_palindrome_b_primary",
    "dead_palindrome_b_sham",
    "source_off_palindrome_a_primary",
    "source_off_palindrome_a_sham",
    "source_off_palindrome_b_primary",
    "source_off_palindrome_b_sham",
    "block_replay_a_primary",
    "block_replay_a_sham",
    "block_replay_b_primary",
    "block_replay_b_sham",
]

QUERY_BY_VARIANT = {
    "palindrome_a_primary": "palindrome_a_compose_query_relation_pair_control",
    "palindrome_a_sham": "palindrome_a_compose_relation_sham_control",
    "palindrome_b_primary": "palindrome_b_compose_query_relation_pair_control",
    "palindrome_b_sham": "palindrome_b_compose_relation_sham_control",
    "dead_palindrome_a_primary": "dead_palindrome_a_compose_query_relation_pair_control",
    "dead_palindrome_a_sham": "dead_palindrome_a_compose_relation_sham_control",
    "dead_palindrome_b_primary": "dead_palindrome_b_compose_query_relation_pair_control",
    "dead_palindrome_b_sham": "dead_palindrome_b_compose_relation_sham_control",
    "source_off_palindrome_a_primary": "source_off_palindrome_a_compose_query_relation_pair_control",
    "source_off_palindrome_a_sham": "source_off_palindrome_a_compose_relation_sham_control",
    "source_off_palindrome_b_primary": "source_off_palindrome_b_compose_query_relation_pair_control",
    "source_off_palindrome_b_sham": "source_off_palindrome_b_compose_relation_sham_control",
    "block_replay_a_primary": "block_replay_a_compose_query_relation_pair_control",
    "block_replay_a_sham": "block_replay_a_compose_relation_sham_control",
    "block_replay_b_primary": "block_replay_b_compose_query_relation_pair_control",
    "block_replay_b_sham": "block_replay_b_compose_relation_sham_control",
}

COMPOSITION_BY_VARIANT = {
    "palindrome_a_primary": "palindrome_holonomy_r0_r1_r1_r0",
    "palindrome_a_sham": "palindrome_holonomy_r0_r1_r1_r0",
    "palindrome_b_primary": "palindrome_holonomy_r1_r0_r0_r1",
    "palindrome_b_sham": "palindrome_holonomy_r1_r0_r0_r1",
    "dead_palindrome_a_primary": "dead_palindrome_holonomy_r0_r1_r1_r0",
    "dead_palindrome_a_sham": "dead_palindrome_holonomy_r0_r1_r1_r0",
    "dead_palindrome_b_primary": "dead_palindrome_holonomy_r1_r0_r0_r1",
    "dead_palindrome_b_sham": "dead_palindrome_holonomy_r1_r0_r0_r1",
    "source_off_palindrome_a_primary": "source_off_palindrome_holonomy_r0_r1_r1_r0",
    "source_off_palindrome_a_sham": "source_off_palindrome_holonomy_r0_r1_r1_r0",
    "source_off_palindrome_b_primary": "source_off_palindrome_holonomy_r1_r0_r0_r1",
    "source_off_palindrome_b_sham": "source_off_palindrome_holonomy_r1_r0_r0_r1",
    "block_replay_a_primary": "block_replay_r0_r0_r1_r1",
    "block_replay_a_sham": "block_replay_r0_r0_r1_r1",
    "block_replay_b_primary": "block_replay_r1_r1_r0_r0",
    "block_replay_b_sham": "block_replay_r1_r1_r0_r0",
}

CONDITION_PAIRS = {
    "alive": ("palindrome_a", "palindrome_b"),
    "dead": ("dead_palindrome_a", "dead_palindrome_b"),
    "source_off": ("source_off_palindrome_a", "source_off_palindrome_b"),
    "block_replay": ("block_replay_a", "block_replay_b"),
}
STRATUM_FACTORS = ["session", "replicate", "mapping", "query_order", "cyclic_origin"]

CONTROL_DEFS = [
    (45, "RELATION_SPATIAL_CONTROL_PALINDROME_A_PRIMARY", "palindrome_a_compose_query_relation_pair_control"),
    (46, "RELATION_SPATIAL_CONTROL_PALINDROME_A_SHAM", "palindrome_a_compose_relation_sham_control"),
    (47, "RELATION_SPATIAL_CONTROL_PALINDROME_B_PRIMARY", "palindrome_b_compose_query_relation_pair_control"),
    (48, "RELATION_SPATIAL_CONTROL_PALINDROME_B_SHAM", "palindrome_b_compose_relation_sham_control"),
    (49, "RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_PRIMARY", "dead_palindrome_a_compose_query_relation_pair_control"),
    (50, "RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_SHAM", "dead_palindrome_a_compose_relation_sham_control"),
    (51, "RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_PRIMARY", "dead_palindrome_b_compose_query_relation_pair_control"),
    (52, "RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_SHAM", "dead_palindrome_b_compose_relation_sham_control"),
    (53, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY", "source_off_palindrome_a_compose_query_relation_pair_control"),
    (54, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM", "source_off_palindrome_a_compose_relation_sham_control"),
    (55, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY", "source_off_palindrome_b_compose_query_relation_pair_control"),
    (56, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM", "source_off_palindrome_b_compose_relation_sham_control"),
    (57, "RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY", "block_replay_a_compose_query_relation_pair_control"),
    (58, "RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM", "block_replay_a_compose_relation_sham_control"),
    (59, "RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY", "block_replay_b_compose_query_relation_pair_control"),
    (60, "RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM", "block_replay_b_compose_relation_sham_control"),
]


def require(condition: bool, message: str) -> None:
    true.replay.require(condition, message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    require(count == 1, f"{label} replacement count unexpected: {count}")
    return text.replace(old, new, 1)


def configure_true_runner() -> None:
    true.RUN_ID = RUN_ID
    true.PACKAGE_ID = PACKAGE_ID
    true.HERE = HERE
    true.SOURCE_ROOT = SOURCE_ROOT
    true.SCHEDULE_DIR = SCHEDULE_DIR
    true.ATTEMPT_LABEL = ATTEMPT_LABEL
    true.RUN_ROOT = RUN_ROOT
    true.LOCAL_PACKAGE = LOCAL_PACKAGE
    true.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    true.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    true.SUMMARY_JSON = SUMMARY_JSON
    true.SUMMARY_MD = SUMMARY_MD
    true.target_script = target_script
    true.configure_replay()
    true.replay.VARIANTS = list(VARIANTS)
    true.replay.QUERY_BY_VARIANT = dict(QUERY_BY_VARIANT)
    true.replay.COMPOSITION_BY_VARIANT = dict(COMPOSITION_BY_VARIANT)


def target_script() -> str:
    return (
        true.REPLAY_TARGET_SCRIPT()
        .replace(
            "for variant in alive_primary alive_sham source_off_primary source_off_sham; do",
            "for variant in " + " ".join(VARIANTS) + "; do",
        )
        .replace("COMPOSITION_REPLAY_EXCLUSION_SCHEDULES", "PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_SCHEDULES")
        .replace("RELATION_COMPOSITION_REPLAY_EXCLUSION", "RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN")
    )


def patch_palindrome_holonomy_controls() -> None:
    true.patch_runtime_for_true_physical_composition()
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    htext = header.read_text(encoding="utf-8")
    control_lines = [
        f"    {name} = {value}{',' if value != CONTROL_DEFS[-1][0] else ''}\n"
        for value, name, _query in CONTROL_DEFS
    ]
    htext = replace_once(
        htext,
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY = 41,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM = 42,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY = 43,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM = 44\n",
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY = 41,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM = 42,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY = 43,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM = 44,\n"
        + "".join(control_lines),
        "header control enum",
    )
    header.write_text(htext, encoding="utf-8", newline="\n")

    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")
    parse_insert = "".join(
        f'    if (strcmp(query, "{query}") == 0) {{\n'
        f"        *out = {name};\n"
        "        return 1;\n"
        "    }\n"
        for _value, name, query in CONTROL_DEFS
    )
    text = replace_once(
        text,
        "    return 0;\n}\n\nstatic int split_tsv",
        parse_insert + "    return 0;\n}\n\nstatic int split_tsv",
        "parse control insert",
    )
    text = replace_once(
        text,
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM) {\n",
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_PALINDROME_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_PALINDROME_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM) {\n",
        "control b-index sham list",
    )
    text = replace_once(
        text,
        "static int rdtscp_available(void) {\n",
        (
            "static int relation_spatial_prepare_composition4(relation_spatial_preparation prep, relation_spatial_carrier_state *state, relation_spatial_relation_id r0, relation_spatial_relation_id r1, relation_spatial_relation_id r2, relation_spatial_relation_id r3) {\n"
            "    uint32_t step = 0u;\n"
            "    if (state == NULL || prep.cyclic_origin >= FAMILY10H_RELATION_SPATIAL_LINE_COUNT) {\n"
            "        return 0;\n"
            "    }\n"
            "    relation_spatial_prefault(state);\n"
            "    flush_state_lines(state);\n"
            "    for (step = 0u; step < (FAMILY10H_RELATION_SPATIAL_TOTAL_WORK / 4u); ++step) {\n"
            "        uint32_t a_index = relation_spatial_origin_index(prep.cyclic_origin, step);\n"
            "        relation_spatial_touch_pair(state, a_index, r0, prep.source_order);\n"
            "        relation_spatial_touch_pair(state, a_index, r1, prep.source_order);\n"
            "        relation_spatial_touch_pair(state, a_index, r2, prep.source_order);\n"
            "        relation_spatial_touch_pair(state, a_index, r3, prep.source_order);\n"
            "    }\n"
            "    return 1;\n"
            "}\n\n"
            "static int rdtscp_available(void) {\n"
        ),
        "composition4 function insert",
    )
    text = replace_once(
        text,
        "static int balanced_alt_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM;\n"
        "}\n\n",
        "static int balanced_alt_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM;\n"
        "}\n\n"
        "static int palindrome_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM;\n"
        "}\n\n"
        "static int palindrome_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_PALINDROME_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM;\n"
        "}\n\n"
        "static int block_replay_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM;\n"
        "}\n\n"
        "static int block_replay_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM;\n"
        "}\n\n",
        "control predicates insert",
    )
    source_dead_old = (
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n"
    )
    text = replace_once(
        text,
        source_dead_old + "}\n\nstatic int source_off_control",
        (
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_A_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_DEAD_PALINDROME_B_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM;\n"
            "}\n\nstatic int source_off_control"
        ),
        "source-dead controls",
    )
    text = replace_once(
        text,
        source_dead_old,
        (
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
            "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM;\n"
        ),
        "source-off controls",
    )
    patched_branches = 0
    for indent in ("            ", "                "):
        inner = indent + "    "
        branch_anchor = (
            f"{indent}}} else if (balanced_alt_b_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition_balanced_alt(prep, &shared->state, 1u);\n"
            f"{indent}}} else if (neutral_compose_control(row.control)) {{\n"
        )
        branch_new = (
            f"{indent}}} else if (balanced_alt_b_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition_balanced_alt(prep, &shared->state, 1u);\n"
            f"{indent}}} else if (palindrome_a_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition4(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
            f"{indent}}} else if (palindrome_b_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition4(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
            f"{indent}}} else if (block_replay_a_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition4(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1, RELATION_SPATIAL_R1);\n"
            f"{indent}}} else if (block_replay_b_control(row.control)) {{\n"
            f"{inner}shared->preparation_ok = relation_spatial_prepare_composition4(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0, RELATION_SPATIAL_R0);\n"
            f"{indent}}} else if (neutral_compose_control(row.control)) {{\n"
        )
        count = text.count(branch_anchor)
        require(count == 1, f"composition branch replacement count unexpected for indent {len(indent)}: {count}")
        text = text.replace(branch_anchor, branch_new, 1)
        patched_branches += count
    require(patched_branches == 2, f"composition branches patched mismatch: {patched_branches}")
    runtime.write_text(text, encoding="utf-8", newline="\n")


def read_jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    member = archive.extractfile(name)
    require(member is not None, f"{name} not extractable")
    return [json.loads(line) for line in member]


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    avg = mean(ordered)
    return {
        "count": len(ordered),
        "mean": avg,
        "abs_of_mean": abs(avg),
        "abs_mean": mean(abs(value) for value in ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "sign_counts": {
            "-1": sum(1 for value in ordered if value < 0.0),
            "0": sum(1 for value in ordered if value == 0.0),
            "1": sum(1 for value in ordered if value > 0.0),
        },
    }


def block_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[str(row["block_id"])].append(row)
    records: list[dict[str, Any]] = []
    for block_id, block in by_block.items():
        cells = {row["relation_cell"]: row for row in block if row["row_role"] == "relation_matrix"}
        require(len(cells) == 4, f"{block_id} relation cell count mismatch: {len(cells)}")
        c00 = float(cells["prepare_r0__query_r0"]["C_pair"])
        c01 = float(cells["prepare_r0__query_r1"]["C_pair"])
        c10 = float(cells["prepare_r1__query_r0"]["C_pair"])
        c11 = float(cells["prepare_r1__query_r1"]["C_pair"])
        first = block[0]
        records.append(
            {
                "block_id": block_id,
                "R_spatial": 0.5 * (c00 + c11 - c01 - c10),
                "prep_axis": 0.5 * (c00 + c01 - c10 - c11),
                "offdiag_skew_axis": 0.5 * (c01 - c10),
                "query_axis": 0.5 * (c00 - c01 + c10 - c11),
                "common_axis": 0.25 * (c00 + c01 + c10 + c11),
                "session": first["session"],
                "replicate": first["replicate"],
                "mapping": first["mapping"],
                "query_order": first["query_order"],
                "cyclic_origin": first["cyclic_origin"],
            }
        )
    return records


def mean_field(records: list[dict[str, Any]], field: str, factor: str | None = None, level: Any | None = None) -> float | None:
    selected = records if factor is None else [record for record in records if record[factor] == level]
    if not selected:
        return None
    return mean(float(record[field]) for record in selected)


def variant_report(raw_rows: list[dict[str, Any]], pair_count: int, death_count: int) -> dict[str, Any]:
    records = block_records(raw_rows)
    return {
        "raw_record_count": len(raw_rows),
        "pair_observation_count": pair_count,
        "source_death_receipt_count": death_count,
        "block_count": len(records),
        "R_spatial": distribution([record["R_spatial"] for record in records]),
        "axes": {
            field: distribution([record[field] for record in records])
            for field in ("prep_axis", "offdiag_skew_axis", "query_axis", "common_axis")
        },
        "block_records": records,
    }


def condition_report(reports: dict[str, Any], left: str, right: str, field: str = "R_spatial") -> dict[str, Any]:
    left_primary = mean_field(reports[f"{left}_primary"]["block_records"], field)
    left_sham = mean_field(reports[f"{left}_sham"]["block_records"], field)
    right_primary = mean_field(reports[f"{right}_primary"]["block_records"], field)
    right_sham = mean_field(reports[f"{right}_sham"]["block_records"], field)
    require(None not in (left_primary, left_sham, right_primary, right_sham), f"{left}/{right} field missing")
    left_delta = left_primary - left_sham
    right_delta = right_primary - right_sham
    return {
        "field": field,
        "left_variant": left,
        "right_variant": right,
        "left_primary": left_primary,
        "left_sham": left_sham,
        "right_primary": right_primary,
        "right_sham": right_sham,
        "left_primary_minus_sham": left_delta,
        "right_primary_minus_sham": right_delta,
        "coordinate_left_minus_right": left_delta - right_delta,
    }


def stratum_coordinate(
    reports: dict[str, Any],
    left: str,
    right: str,
    field: str,
    factor: str,
    level: Any,
) -> float | None:
    left_primary = mean_field(reports[f"{left}_primary"]["block_records"], field, factor, level)
    left_sham = mean_field(reports[f"{left}_sham"]["block_records"], field, factor, level)
    right_primary = mean_field(reports[f"{right}_primary"]["block_records"], field, factor, level)
    right_sham = mean_field(reports[f"{right}_sham"]["block_records"], field, factor, level)
    if None in (left_primary, left_sham, right_primary, right_sham):
        return None
    return (left_primary - left_sham) - (right_primary - right_sham)


def stratum_reports(reports: dict[str, Any], aggregate_alive: float, field: str = "R_spatial") -> dict[str, Any]:
    output: dict[str, Any] = {}
    aggregate_sign = 1 if aggregate_alive > 0 else -1 if aggregate_alive < 0 else 0
    for factor in STRATUM_FACTORS:
        levels = sorted({record[factor] for record in reports["palindrome_a_primary"]["block_records"]}, key=str)
        rows = []
        for level in levels:
            values = {
                condition: stratum_coordinate(reports, left, right, field, factor, level)
                for condition, (left, right) in CONDITION_PAIRS.items()
            }
            if values["alive"] is None:
                continue
            denom = abs(float(values["alive"]))
            ratios = {
                key: (abs(float(values[key])) / denom if denom else None)
                for key in ("dead", "source_off", "block_replay")
            }
            sign = 1 if values["alive"] > 0 else -1 if values["alive"] < 0 else 0
            rows.append(
                {
                    "level": level,
                    "coordinates": values,
                    "ratios_to_alive_abs": ratios,
                    "alive_same_sign_as_aggregate": sign == aggregate_sign and sign != 0,
                }
            )
        output[factor] = {
            "levels": rows,
            "all_alive_same_sign_as_aggregate": all(item["alive_same_sign_as_aggregate"] for item in rows),
            "max_source_off_ratio": max(item["ratios_to_alive_abs"]["source_off"] for item in rows),
            "min_dead_ratio": min(item["ratios_to_alive_abs"]["dead"] for item in rows),
            "max_block_replay_ratio": max(item["ratios_to_alive_abs"]["block_replay"] for item in rows),
        }
    return output


def analyze_archive(controller: dict[str, Any]) -> dict[str, Any]:
    require(LOCAL_ARCHIVE.exists(), "archive missing for analysis")
    reports: dict[str, Any] = {}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as archive:
        names = set(archive.getnames())
        for variant in VARIANTS:
            prefix = f"source/discovery_outputs/{variant}"
            raw_name = f"{prefix}/raw_records.jsonl"
            pair_name = f"{prefix}/pair_observations.jsonl"
            death_name = f"{prefix}/source_death_receipts.jsonl"
            for name in (raw_name, pair_name, death_name):
                require(name in names, f"missing archive member {name}")
            raw = read_jsonl_member(archive, raw_name)
            pairs = read_jsonl_member(archive, pair_name)
            deaths = read_jsonl_member(archive, death_name)
            require(len(raw) == true.replay.ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            require(
                len(pairs) == true.replay.ROWS_PER_VARIANT * true.replay.PAIR_SAMPLE_COUNT,
                f"{variant} pair count mismatch: {len(pairs)}",
            )
            require(len(deaths) == true.replay.ROWS_PER_VARIANT, f"{variant} death count mismatch: {len(deaths)}")
            reports[variant] = variant_report(raw, len(pairs), len(deaths))

    r_reports = {condition: condition_report(reports, left, right, "R_spatial") for condition, (left, right) in CONDITION_PAIRS.items()}
    prep_reports = {condition: condition_report(reports, left, right, "prep_axis") for condition, (left, right) in CONDITION_PAIRS.items()}
    alive = r_reports["alive"]["coordinate_left_minus_right"]
    denom = abs(alive)
    coordinates = {
        "Omega_palindrome_holonomy_alive_a_minus_b": alive,
        "Omega_palindrome_holonomy_dead_a_minus_b": r_reports["dead"]["coordinate_left_minus_right"],
        "Omega_palindrome_holonomy_source_off_a_minus_b": r_reports["source_off"]["coordinate_left_minus_right"],
        "Omega_block_replay_a_minus_b": r_reports["block_replay"]["coordinate_left_minus_right"],
        "source_off_abs_to_alive_abs": abs(r_reports["source_off"]["coordinate_left_minus_right"]) / denom if denom else None,
        "dead_abs_to_alive_abs": abs(r_reports["dead"]["coordinate_left_minus_right"]) / denom if denom else None,
        "block_replay_abs_to_alive_abs": abs(r_reports["block_replay"]["coordinate_left_minus_right"]) / denom if denom else None,
        "K_prep_axis_palindrome_alive_a_minus_b": prep_reports["alive"]["coordinate_left_minus_right"],
        "K_prep_axis_palindrome_dead_a_minus_b": prep_reports["dead"]["coordinate_left_minus_right"],
        "K_prep_axis_palindrome_source_off_a_minus_b": prep_reports["source_off"]["coordinate_left_minus_right"],
        "K_prep_axis_block_replay_a_minus_b": prep_reports["block_replay"]["coordinate_left_minus_right"],
    }
    strata = stratum_reports(reports, alive, "R_spatial")
    max_source_off = max(report["max_source_off_ratio"] for report in strata.values())
    min_dead = min(report["min_dead_ratio"] for report in strata.values())
    max_block = max(report["max_block_replay_ratio"] for report in strata.values())
    interpretation = {
        "palindrome_holonomy_coordinate_nonzero": denom > 0.0,
        "aggregate_source_off_collapses_below_25pct": coordinates["source_off_abs_to_alive_abs"] <= 0.25 if denom else False,
        "aggregate_dead_preserves_at_least_25pct": coordinates["dead_abs_to_alive_abs"] >= 0.25 if denom else False,
        "aggregate_block_replay_below_25pct": coordinates["block_replay_abs_to_alive_abs"] <= 0.25 if denom else False,
        "all_one_factor_alive_strata_same_sign": all(report["all_alive_same_sign_as_aggregate"] for report in strata.values()),
        "all_one_factor_source_off_below_25pct": max_source_off <= 0.25,
        "all_one_factor_dead_preserves_25pct": min_dead >= 0.25,
        "all_one_factor_block_replay_below_25pct": max_block <= 0.25,
        "max_one_factor_source_off_ratio": max_source_off,
        "min_one_factor_dead_ratio": min_dead,
        "max_one_factor_block_replay_ratio": max_block,
        "palindrome_holonomy_source_off_dead_candidate": False,
        "exploratory_only": True,
        "small_wall_crossed": False,
    }
    interpretation["palindrome_holonomy_source_off_dead_candidate"] = (
        interpretation["palindrome_holonomy_coordinate_nonzero"]
        and interpretation["aggregate_source_off_collapses_below_25pct"]
        and interpretation["aggregate_dead_preserves_at_least_25pct"]
        and interpretation["aggregate_block_replay_below_25pct"]
        and interpretation["all_one_factor_alive_strata_same_sign"]
        and interpretation["all_one_factor_source_off_below_25pct"]
        and interpretation["all_one_factor_dead_preserves_25pct"]
        and interpretation["all_one_factor_block_replay_below_25pct"]
    )
    result = {
        "schema": "FAMILY10H_RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_ANALYSIS_V1",
        "created_at": true.replay.utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller["passed"],
        "archive_sha256": controller["archive_sha256"],
        "archive_size": controller["archive_size"],
        "variant_reports": {name: {k: v for k, v in report.items() if k != "block_records"} for name, report in reports.items()},
        "condition_reports": r_reports,
        "prep_axis_condition_reports": prep_reports,
        "stratum_reports": strata,
        "coordinates": coordinates,
        "interpretation": interpretation,
        "claim_boundary": {
            "positive_scientific_claim": False,
            "holographic_relational_invariant_established": False,
            "full_tomography_established": False,
            "r2_restoration_established": False,
            "small_wall_crossed": False,
        },
    }
    result["analysis_sha256"] = true.replay.digest({k: v for k, v in result.items() if k != "analysis_sha256"})
    return result


def write_summary(analysis: dict[str, Any]) -> None:
    c = analysis["coordinates"]
    i = analysis["interpretation"]
    lines = [
        "# Relation Palindrome-Holonomy Source-Off Screen",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{analysis['archive_sha256']}`",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`",
        "",
        "Palindrome holonomy coordinates:",
        f"- alive Omega palindrome A minus B: `{c['Omega_palindrome_holonomy_alive_a_minus_b']:.9f}`",
        f"- dead Omega palindrome A minus B: `{c['Omega_palindrome_holonomy_dead_a_minus_b']:.9f}`",
        f"- source-off Omega palindrome A minus B: `{c['Omega_palindrome_holonomy_source_off_a_minus_b']:.9f}`",
        f"- grouped block-replay Omega A minus B: `{c['Omega_block_replay_a_minus_b']:.9f}`",
        f"- source-off abs/alive abs: `{c['source_off_abs_to_alive_abs']:.3f}`",
        f"- dead abs/alive abs: `{c['dead_abs_to_alive_abs']:.3f}`",
        f"- block-replay abs/alive abs: `{c['block_replay_abs_to_alive_abs']:.3f}`",
        "",
        "Interpretation:",
        f"- palindrome holonomy source-off/dead candidate: `{i['palindrome_holonomy_source_off_dead_candidate']}`",
        f"- aggregate source-off below 0.25 x alive: `{i['aggregate_source_off_collapses_below_25pct']}`",
        f"- aggregate dead at least 0.25 x alive: `{i['aggregate_dead_preserves_at_least_25pct']}`",
        f"- aggregate block replay below 0.25 x alive: `{i['aggregate_block_replay_below_25pct']}`",
        f"- all one-factor alive strata same sign: `{i['all_one_factor_alive_strata_same_sign']}`",
        f"- all one-factor source-off below 0.25: `{i['all_one_factor_source_off_below_25pct']}`",
        f"- all one-factor dead preserves 0.25: `{i['all_one_factor_dead_preserves_25pct']}`",
        f"- all one-factor block replay below 0.25: `{i['all_one_factor_block_replay_below_25pct']}`",
        "",
        "This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def prepare() -> dict[str, Any]:
    configure_true_runner()
    true.replay.patch_runtime_for_composition_loop = patch_palindrome_holonomy_controls
    result = true.replay.prepare()
    result["schema"] = "FAMILY10H_RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_PREPARE_V1"
    true.replay.write_json(HERE / "RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_true_runner()
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    require(prepare_result["passed"], "prepare failed")
    package = true.replay.base.build_package()
    controller = true.replay.base.deploy_execute_copyback(package)
    analysis = analyze_archive(controller)
    true.replay.write_json(RUN_ROOT / "RELATION_PALINDROME_HOLONOMY_SOURCE_OFF_SCREEN_ANALYSIS.json", analysis)
    true.replay.write_json(SUMMARY_JSON, analysis)
    write_summary(analysis)
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
