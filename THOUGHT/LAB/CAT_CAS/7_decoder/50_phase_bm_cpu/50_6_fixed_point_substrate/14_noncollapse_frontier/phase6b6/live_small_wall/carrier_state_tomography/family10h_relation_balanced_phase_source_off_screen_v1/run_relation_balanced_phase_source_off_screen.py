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

RUN_ID = "family10h_relation_balanced_phase_source_off_screen_v1_0"
PACKAGE_ID = "family10h_relation_balanced_phase_source_off_screen_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "BALANCED_PHASE_SOURCE_OFF_SCREEN_SCHEDULES"
ATTEMPT_LABEL = os.environ.get("FAMILY10H_DISCOVERY_ATTEMPT_LABEL", "attempt_1")
if any(sep in ATTEMPT_LABEL for sep in ("/", "\\", ":")) or not ATTEMPT_LABEL:
    raise RuntimeError(f"invalid attempt label: {ATTEMPT_LABEL!r}")
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / ATTEMPT_LABEL
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_SUMMARY.md"

VARIANTS = [
    "balanced_alt_a_primary",
    "balanced_alt_a_sham",
    "balanced_alt_b_primary",
    "balanced_alt_b_sham",
    "dead_balanced_alt_a_primary",
    "dead_balanced_alt_a_sham",
    "dead_balanced_alt_b_primary",
    "dead_balanced_alt_b_sham",
    "source_off_balanced_alt_a_primary",
    "source_off_balanced_alt_a_sham",
    "source_off_balanced_alt_b_primary",
    "source_off_balanced_alt_b_sham",
]

QUERY_BY_VARIANT = {
    "balanced_alt_a_primary": "balanced_alt_a_compose_query_relation_pair_control",
    "balanced_alt_a_sham": "balanced_alt_a_compose_relation_sham_control",
    "balanced_alt_b_primary": "balanced_alt_b_compose_query_relation_pair_control",
    "balanced_alt_b_sham": "balanced_alt_b_compose_relation_sham_control",
    "dead_balanced_alt_a_primary": "dead_balanced_alt_a_compose_query_relation_pair_control",
    "dead_balanced_alt_a_sham": "dead_balanced_alt_a_compose_relation_sham_control",
    "dead_balanced_alt_b_primary": "dead_balanced_alt_b_compose_query_relation_pair_control",
    "dead_balanced_alt_b_sham": "dead_balanced_alt_b_compose_relation_sham_control",
    "source_off_balanced_alt_a_primary": "source_off_balanced_alt_a_compose_query_relation_pair_control",
    "source_off_balanced_alt_a_sham": "source_off_balanced_alt_a_compose_relation_sham_control",
    "source_off_balanced_alt_b_primary": "source_off_balanced_alt_b_compose_query_relation_pair_control",
    "source_off_balanced_alt_b_sham": "source_off_balanced_alt_b_compose_relation_sham_control",
}

COMPOSITION_BY_VARIANT = {
    "balanced_alt_a_primary": "balanced_alternating_phase_a",
    "balanced_alt_a_sham": "balanced_alternating_phase_a",
    "balanced_alt_b_primary": "balanced_alternating_phase_b",
    "balanced_alt_b_sham": "balanced_alternating_phase_b",
    "dead_balanced_alt_a_primary": "dead_balanced_alternating_phase_a",
    "dead_balanced_alt_a_sham": "dead_balanced_alternating_phase_a",
    "dead_balanced_alt_b_primary": "dead_balanced_alternating_phase_b",
    "dead_balanced_alt_b_sham": "dead_balanced_alternating_phase_b",
    "source_off_balanced_alt_a_primary": "source_off_balanced_alternating_phase_a",
    "source_off_balanced_alt_a_sham": "source_off_balanced_alternating_phase_a",
    "source_off_balanced_alt_b_primary": "source_off_balanced_alternating_phase_b",
    "source_off_balanced_alt_b_sham": "source_off_balanced_alternating_phase_b",
}

CONDITION_PAIRS = {
    "alive": ("balanced_alt_a", "balanced_alt_b"),
    "dead": ("dead_balanced_alt_a", "dead_balanced_alt_b"),
    "source_off": ("source_off_balanced_alt_a", "source_off_balanced_alt_b"),
}
STRATUM_FACTORS = ["session", "replicate", "mapping", "query_order", "cyclic_origin"]


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
        .replace("COMPOSITION_REPLAY_EXCLUSION_SCHEDULES", "BALANCED_PHASE_SOURCE_OFF_SCREEN_SCHEDULES")
        .replace("RELATION_COMPOSITION_REPLAY_EXCLUSION", "RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN")
    )


def patch_balanced_phase_controls() -> None:
    true.patch_runtime_for_true_physical_composition()
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    htext = header.read_text(encoding="utf-8")
    htext = htext.replace(
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY = 41,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM = 42,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY = 43,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM = 44\n",
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY = 41,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM = 42,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY = 43,\n"
        "    RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM = 44,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_PRIMARY = 45,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_SHAM = 46,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_PRIMARY = 47,\n"
        "    RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_SHAM = 48,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_PRIMARY = 49,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM = 50,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_PRIMARY = 51,\n"
        "    RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM = 52\n",
        1,
    )
    header.write_text(htext, encoding="utf-8", newline="\n")

    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")
    parse_insert = (
        '    if (strcmp(query, "dead_balanced_alt_a_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_balanced_alt_a_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_balanced_alt_b_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "dead_balanced_alt_b_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_balanced_alt_a_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_balanced_alt_a_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_balanced_alt_b_compose_query_relation_pair_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_PRIMARY;\n"
        "        return 1;\n"
        "    }\n"
        '    if (strcmp(query, "source_off_balanced_alt_b_compose_relation_sham_control") == 0) {\n'
        "        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM;\n"
        "        return 1;\n"
        "    }\n"
    )
    text = text.replace("    return 0;\n}\n\nstatic int split_tsv", parse_insert + "    return 0;\n}\n\nstatic int split_tsv", 1)
    text = text.replace(
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM) {\n",
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM) {\n",
        1,
    )
    text = text.replace(
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n",
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM;\n",
        1,
    )
    text = text.replace(
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM;\n",
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R0_R1_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_COMPOSE_R1_R0_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM;\n",
        1,
    )
    text = text.replace(
        "static int balanced_alt_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM;\n"
        "}\n\n"
        "static int balanced_alt_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM;\n"
        "}\n",
        "static int balanced_alt_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_A_SHAM;\n"
        "}\n\n"
        "static int balanced_alt_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BALANCED_ALT_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BALANCED_ALT_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BALANCED_ALT_B_SHAM;\n"
        "}\n",
        1,
    )
    runtime.write_text(text, encoding="utf-8", newline="\n")


def read_jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    member = archive.extractfile(name)
    true.replay.require(member is not None, f"{name} not extractable")
    return [json.loads(line) for line in member]


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "abs_of_mean": abs(mean(ordered)),
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
        true.replay.require(len(cells) == 4, f"{block_id} relation cell count mismatch: {len(cells)}")
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


def variant_report(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = block_records(raw_rows)
    return {
        "raw_record_count": len(raw_rows),
        "block_count": len(records),
        "pair_observation_count": len(raw_rows) * true.replay.PAIR_SAMPLE_COUNT,
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
    true.replay.require(None not in (left_primary, left_sham, right_primary, right_sham), f"{left}/{right} field missing")
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
        "phase_coordinate_left_minus_right": left_delta - right_delta,
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
        levels = sorted({record[factor] for record in reports["balanced_alt_a_primary"]["block_records"]}, key=str)
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
                for key in ("dead", "source_off")
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
        }
    return output


def analyze_archive(controller: dict[str, Any]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    death_counts: dict[str, int] = {}
    with tarfile.open(LOCAL_ARCHIVE, "r:gz") as archive:
        names = set(archive.getnames())
        for variant in VARIANTS:
            raw_name = f"source/discovery_outputs/{variant}/raw_records.jsonl"
            death_name = f"source/discovery_outputs/{variant}/source_death_receipts.jsonl"
            true.replay.require(raw_name in names, f"{variant} raw_records missing")
            true.replay.require(death_name in names, f"{variant} source_death_receipts missing")
            raw = read_jsonl_member(archive, raw_name)
            deaths = read_jsonl_member(archive, death_name)
            true.replay.require(len(raw) == true.replay.ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            true.replay.require(len(deaths) == true.replay.ROWS_PER_VARIANT, f"{variant} death count mismatch: {len(deaths)}")
            reports[variant] = variant_report(raw)
            death_counts[variant] = len(deaths)

    r_reports = {condition: condition_report(reports, left, right, "R_spatial") for condition, (left, right) in CONDITION_PAIRS.items()}
    prep_reports = {condition: condition_report(reports, left, right, "prep_axis") for condition, (left, right) in CONDITION_PAIRS.items()}
    alive = r_reports["alive"]["phase_coordinate_left_minus_right"]
    denom = abs(alive)
    coordinates = {
        "Omega_balanced_phase_alive_a_minus_b": alive,
        "Omega_balanced_phase_dead_a_minus_b": r_reports["dead"]["phase_coordinate_left_minus_right"],
        "Omega_balanced_phase_source_off_a_minus_b": r_reports["source_off"]["phase_coordinate_left_minus_right"],
        "source_off_abs_to_alive_abs": abs(r_reports["source_off"]["phase_coordinate_left_minus_right"]) / denom if denom else None,
        "dead_abs_to_alive_abs": abs(r_reports["dead"]["phase_coordinate_left_minus_right"]) / denom if denom else None,
        "K_prep_axis_balanced_phase_alive_a_minus_b": prep_reports["alive"]["phase_coordinate_left_minus_right"],
        "K_prep_axis_balanced_phase_dead_a_minus_b": prep_reports["dead"]["phase_coordinate_left_minus_right"],
        "K_prep_axis_balanced_phase_source_off_a_minus_b": prep_reports["source_off"]["phase_coordinate_left_minus_right"],
    }
    strata = stratum_reports(reports, alive, "R_spatial")
    max_source_off = max(report["max_source_off_ratio"] for report in strata.values())
    min_dead = min(report["min_dead_ratio"] for report in strata.values())
    interpretation = {
        "balanced_phase_coordinate_nonzero": denom > 0.0,
        "aggregate_source_off_collapses_below_25pct": coordinates["source_off_abs_to_alive_abs"] <= 0.25 if denom else False,
        "aggregate_dead_preserves_at_least_25pct": coordinates["dead_abs_to_alive_abs"] >= 0.25 if denom else False,
        "all_one_factor_alive_strata_same_sign": all(report["all_alive_same_sign_as_aggregate"] for report in strata.values()),
        "all_one_factor_source_off_below_25pct": max_source_off <= 0.25,
        "all_one_factor_dead_preserves_25pct": min_dead >= 0.25,
        "max_one_factor_source_off_ratio": max_source_off,
        "min_one_factor_dead_ratio": min_dead,
        "balanced_phase_source_off_dead_candidate": False,
        "exploratory_only": True,
        "small_wall_crossed": False,
    }
    interpretation["balanced_phase_source_off_dead_candidate"] = (
        interpretation["balanced_phase_coordinate_nonzero"]
        and interpretation["aggregate_source_off_collapses_below_25pct"]
        and interpretation["aggregate_dead_preserves_at_least_25pct"]
        and interpretation["all_one_factor_alive_strata_same_sign"]
        and interpretation["all_one_factor_source_off_below_25pct"]
        and interpretation["all_one_factor_dead_preserves_25pct"]
    )
    result = {
        "schema": "FAMILY10H_RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_ANALYSIS_V1",
        "created_at": true.replay.utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller["passed"],
        "archive_sha256": controller["archive_sha256"],
        "archive_size": controller["archive_size"],
        "variant_reports": {name: {k: v for k, v in report.items() if k != "block_records"} for name, report in reports.items()},
        "source_death_receipt_counts": death_counts,
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
        "# Relation Balanced-Phase Source-Off Screen",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{analysis['archive_sha256']}`",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`",
        "",
        "Balanced phase coordinates:",
        f"- alive Omega phase A minus B: `{c['Omega_balanced_phase_alive_a_minus_b']:.9f}`",
        f"- dead Omega phase A minus B: `{c['Omega_balanced_phase_dead_a_minus_b']:.9f}`",
        f"- source-off Omega phase A minus B: `{c['Omega_balanced_phase_source_off_a_minus_b']:.9f}`",
        f"- source-off abs/alive abs: `{c['source_off_abs_to_alive_abs']:.3f}`",
        f"- dead abs/alive abs: `{c['dead_abs_to_alive_abs']:.3f}`",
        f"- alive prep-axis K phase A minus B: `{c['K_prep_axis_balanced_phase_alive_a_minus_b']:.9f}`",
        "",
        "Interpretation:",
        f"- balanced phase source-off/dead candidate: `{i['balanced_phase_source_off_dead_candidate']}`",
        f"- aggregate source-off below 0.25 x alive: `{i['aggregate_source_off_collapses_below_25pct']}`",
        f"- aggregate dead at least 0.25 x alive: `{i['aggregate_dead_preserves_at_least_25pct']}`",
        f"- all one-factor alive strata same sign: `{i['all_one_factor_alive_strata_same_sign']}`",
        f"- all one-factor source-off below 0.25: `{i['all_one_factor_source_off_below_25pct']}`",
        f"- all one-factor dead preserves 0.25: `{i['all_one_factor_dead_preserves_25pct']}`",
        "",
        "This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def prepare() -> dict[str, Any]:
    configure_true_runner()
    true.replay.patch_runtime_for_composition_loop = patch_balanced_phase_controls
    result = true.replay.prepare()
    result["schema"] = "FAMILY10H_RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_PREPARE_V1"
    true.replay.write_json(HERE / "RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_true_runner()
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    true.replay.require(prepare_result["passed"], "prepare failed")
    package = true.replay.base.build_package()
    controller = true.replay.base.deploy_execute_copyback(package)
    analysis = analyze_archive(controller)
    true.replay.write_json(RUN_ROOT / "RELATION_BALANCED_PHASE_SOURCE_OFF_SCREEN_ANALYSIS.json", analysis)
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
