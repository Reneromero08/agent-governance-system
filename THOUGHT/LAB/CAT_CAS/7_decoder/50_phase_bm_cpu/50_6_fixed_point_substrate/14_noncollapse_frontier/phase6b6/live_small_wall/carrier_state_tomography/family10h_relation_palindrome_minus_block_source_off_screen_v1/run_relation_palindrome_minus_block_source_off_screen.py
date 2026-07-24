#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import tarfile
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
PAL_RUNNER = (
    CARRIER_ROOT
    / "family10h_relation_palindrome_holonomy_source_off_screen_v1"
    / "run_relation_palindrome_holonomy_source_off_screen.py"
)

spec = importlib.util.spec_from_file_location("palindrome_base", PAL_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load palindrome runner: {PAL_RUNNER}")
pal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pal)

RUN_ID = "family10h_relation_palindrome_minus_block_source_off_screen_v1_0"
PACKAGE_ID = "family10h_relation_palindrome_minus_block_source_off_screen_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_SCHEDULES"
ATTEMPT_LABEL = os.environ.get("FAMILY10H_DISCOVERY_ATTEMPT_LABEL", "attempt_1")
if any(sep in ATTEMPT_LABEL for sep in ("/", "\\", ":")) or not ATTEMPT_LABEL:
    raise RuntimeError(f"invalid attempt label: {ATTEMPT_LABEL!r}")
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / ATTEMPT_LABEL
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_SUMMARY.md"

VARIANTS = [
    "palindrome_a_primary",
    "palindrome_a_sham",
    "palindrome_b_primary",
    "palindrome_b_sham",
    "block_replay_a_primary",
    "block_replay_a_sham",
    "block_replay_b_primary",
    "block_replay_b_sham",
    "dead_palindrome_a_primary",
    "dead_palindrome_a_sham",
    "dead_palindrome_b_primary",
    "dead_palindrome_b_sham",
    "dead_block_replay_a_primary",
    "dead_block_replay_a_sham",
    "dead_block_replay_b_primary",
    "dead_block_replay_b_sham",
    "source_off_palindrome_a_primary",
    "source_off_palindrome_a_sham",
    "source_off_palindrome_b_primary",
    "source_off_palindrome_b_sham",
    "source_off_block_replay_a_primary",
    "source_off_block_replay_a_sham",
    "source_off_block_replay_b_primary",
    "source_off_block_replay_b_sham",
]

QUERY_BY_VARIANT = {
    "palindrome_a_primary": "palindrome_a_compose_query_relation_pair_control",
    "palindrome_a_sham": "palindrome_a_compose_relation_sham_control",
    "palindrome_b_primary": "palindrome_b_compose_query_relation_pair_control",
    "palindrome_b_sham": "palindrome_b_compose_relation_sham_control",
    "block_replay_a_primary": "block_replay_a_compose_query_relation_pair_control",
    "block_replay_a_sham": "block_replay_a_compose_relation_sham_control",
    "block_replay_b_primary": "block_replay_b_compose_query_relation_pair_control",
    "block_replay_b_sham": "block_replay_b_compose_relation_sham_control",
    "dead_palindrome_a_primary": "dead_palindrome_a_compose_query_relation_pair_control",
    "dead_palindrome_a_sham": "dead_palindrome_a_compose_relation_sham_control",
    "dead_palindrome_b_primary": "dead_palindrome_b_compose_query_relation_pair_control",
    "dead_palindrome_b_sham": "dead_palindrome_b_compose_relation_sham_control",
    "dead_block_replay_a_primary": "dead_block_replay_a_compose_query_relation_pair_control",
    "dead_block_replay_a_sham": "dead_block_replay_a_compose_relation_sham_control",
    "dead_block_replay_b_primary": "dead_block_replay_b_compose_query_relation_pair_control",
    "dead_block_replay_b_sham": "dead_block_replay_b_compose_relation_sham_control",
    "source_off_palindrome_a_primary": "source_off_palindrome_a_compose_query_relation_pair_control",
    "source_off_palindrome_a_sham": "source_off_palindrome_a_compose_relation_sham_control",
    "source_off_palindrome_b_primary": "source_off_palindrome_b_compose_query_relation_pair_control",
    "source_off_palindrome_b_sham": "source_off_palindrome_b_compose_relation_sham_control",
    "source_off_block_replay_a_primary": "source_off_block_replay_a_compose_query_relation_pair_control",
    "source_off_block_replay_a_sham": "source_off_block_replay_a_compose_relation_sham_control",
    "source_off_block_replay_b_primary": "source_off_block_replay_b_compose_query_relation_pair_control",
    "source_off_block_replay_b_sham": "source_off_block_replay_b_compose_relation_sham_control",
}

COMPOSITION_BY_VARIANT = {
    "palindrome_a_primary": "palindrome_holonomy_r0_r1_r1_r0",
    "palindrome_a_sham": "palindrome_holonomy_r0_r1_r1_r0",
    "palindrome_b_primary": "palindrome_holonomy_r1_r0_r0_r1",
    "palindrome_b_sham": "palindrome_holonomy_r1_r0_r0_r1",
    "block_replay_a_primary": "block_replay_r0_r0_r1_r1",
    "block_replay_a_sham": "block_replay_r0_r0_r1_r1",
    "block_replay_b_primary": "block_replay_r1_r1_r0_r0",
    "block_replay_b_sham": "block_replay_r1_r1_r0_r0",
    "dead_palindrome_a_primary": "dead_palindrome_holonomy_r0_r1_r1_r0",
    "dead_palindrome_a_sham": "dead_palindrome_holonomy_r0_r1_r1_r0",
    "dead_palindrome_b_primary": "dead_palindrome_holonomy_r1_r0_r0_r1",
    "dead_palindrome_b_sham": "dead_palindrome_holonomy_r1_r0_r0_r1",
    "dead_block_replay_a_primary": "dead_block_replay_r0_r0_r1_r1",
    "dead_block_replay_a_sham": "dead_block_replay_r0_r0_r1_r1",
    "dead_block_replay_b_primary": "dead_block_replay_r1_r1_r0_r0",
    "dead_block_replay_b_sham": "dead_block_replay_r1_r1_r0_r0",
    "source_off_palindrome_a_primary": "source_off_palindrome_holonomy_r0_r1_r1_r0",
    "source_off_palindrome_a_sham": "source_off_palindrome_holonomy_r0_r1_r1_r0",
    "source_off_palindrome_b_primary": "source_off_palindrome_holonomy_r1_r0_r0_r1",
    "source_off_palindrome_b_sham": "source_off_palindrome_holonomy_r1_r0_r0_r1",
    "source_off_block_replay_a_primary": "source_off_block_replay_r0_r0_r1_r1",
    "source_off_block_replay_a_sham": "source_off_block_replay_r0_r0_r1_r1",
    "source_off_block_replay_b_primary": "source_off_block_replay_r1_r1_r0_r0",
    "source_off_block_replay_b_sham": "source_off_block_replay_r1_r1_r0_r0",
}

CONDITION_PREFIXES = {
    "alive": ("palindrome", "block_replay"),
    "dead": ("dead_palindrome", "dead_block_replay"),
    "source_off": ("source_off_palindrome", "source_off_block_replay"),
}
STRATUM_FACTORS = ["session", "replicate", "mapping", "query_order", "cyclic_origin"]

EXTRA_CONTROL_DEFS = [
    (61, "RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_PRIMARY", "dead_block_replay_a_compose_query_relation_pair_control"),
    (62, "RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_SHAM", "dead_block_replay_a_compose_relation_sham_control"),
    (63, "RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_PRIMARY", "dead_block_replay_b_compose_query_relation_pair_control"),
    (64, "RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_SHAM", "dead_block_replay_b_compose_relation_sham_control"),
    (65, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_PRIMARY", "source_off_block_replay_a_compose_query_relation_pair_control"),
    (66, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_SHAM", "source_off_block_replay_a_compose_relation_sham_control"),
    (67, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_PRIMARY", "source_off_block_replay_b_compose_query_relation_pair_control"),
    (68, "RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_SHAM", "source_off_block_replay_b_compose_relation_sham_control"),
]


def require(condition: bool, message: str) -> None:
    pal.require(condition, message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    require(count == 1, f"{label} replacement count unexpected: {count}")
    return text.replace(old, new, 1)


def configure_pal_runner() -> None:
    pal.RUN_ID = RUN_ID
    pal.PACKAGE_ID = PACKAGE_ID
    pal.HERE = HERE
    pal.CARRIER_ROOT = CARRIER_ROOT
    pal.SOURCE_ROOT = SOURCE_ROOT
    pal.SCHEDULE_DIR = SCHEDULE_DIR
    pal.ATTEMPT_LABEL = ATTEMPT_LABEL
    pal.RUN_ROOT = RUN_ROOT
    pal.LOCAL_PACKAGE = LOCAL_PACKAGE
    pal.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    pal.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    pal.SUMMARY_JSON = SUMMARY_JSON
    pal.SUMMARY_MD = SUMMARY_MD
    pal.VARIANTS = list(VARIANTS)
    pal.QUERY_BY_VARIANT = dict(QUERY_BY_VARIANT)
    pal.COMPOSITION_BY_VARIANT = dict(COMPOSITION_BY_VARIANT)
    pal.target_script = target_script
    pal.configure_true_runner()


def target_script() -> str:
    return (
        pal.true.REPLAY_TARGET_SCRIPT()
        .replace(
            "for variant in alive_primary alive_sham source_off_primary source_off_sham; do",
            "for variant in " + " ".join(VARIANTS) + "; do",
        )
        .replace("COMPOSITION_REPLAY_EXCLUSION_SCHEDULES", "PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_SCHEDULES")
        .replace("RELATION_COMPOSITION_REPLAY_EXCLUSION", "RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN")
    )


def patch_palindrome_minus_block_controls() -> None:
    pal.patch_palindrome_holonomy_controls()
    header = SOURCE_ROOT / "relation_spatial_runtime.h"
    htext = header.read_text(encoding="utf-8")
    extra_lines = [
        f"    {name} = {value}{',' if value != EXTRA_CONTROL_DEFS[-1][0] else ''}\n"
        for value, name, _query in EXTRA_CONTROL_DEFS
    ]
    htext = replace_once(
        htext,
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY = 57,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM = 58,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY = 59,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM = 60\n",
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY = 57,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM = 58,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY = 59,\n"
        "    RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM = 60,\n"
        + "".join(extra_lines),
        "header extra control enum",
    )
    header.write_text(htext, encoding="utf-8", newline="\n")

    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")
    parse_insert = "".join(
        f'    if (strcmp(query, "{query}") == 0) {{\n'
        f"        *out = {name};\n"
        "        return 1;\n"
        "    }\n"
        for _value, name, query in EXTRA_CONTROL_DEFS
    )
    text = replace_once(
        text,
        "    return 0;\n}\n\nstatic int split_tsv",
        parse_insert + "    return 0;\n}\n\nstatic int split_tsv",
        "parse extra controls",
    )
    text = replace_once(
        text,
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM) {\n",
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_SHAM\n"
        "        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_SHAM) {\n",
        "extra block sham b-index",
    )
    text = replace_once(
        text,
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM;\n"
        "}\n\nstatic int source_off_control",
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_SHAM;\n"
        "}\n\nstatic int source_off_control",
        "extra source-dead block controls",
    )
    text = replace_once(
        text,
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM;\n",
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_PALINDROME_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_SHAM;\n",
        "extra source-off block controls",
    )
    text = replace_once(
        text,
        "static int block_replay_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM;\n"
        "}\n\n"
        "static int block_replay_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM;\n"
        "}\n",
        "static int block_replay_a_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_A_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_A_SHAM;\n"
        "}\n\n"
        "static int block_replay_b_control(relation_spatial_control_id control) {\n"
        "    return control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_BLOCK_REPLAY_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_DEAD_BLOCK_REPLAY_B_SHAM\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_PRIMARY\n"
        "        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_BLOCK_REPLAY_B_SHAM;\n"
        "}\n",
        "extra block predicates",
    )
    runtime.write_text(text, encoding="utf-8", newline="\n")


def condition_coordinate(reports: dict[str, Any], prefix: str, field: str = "R_spatial") -> dict[str, Any]:
    return pal.condition_report(reports, f"{prefix}_a", f"{prefix}_b", field)


def differential_for_condition(reports: dict[str, Any], condition: str, field: str = "R_spatial") -> dict[str, Any]:
    palindrome_prefix, block_prefix = CONDITION_PREFIXES[condition]
    palindrome = condition_coordinate(reports, palindrome_prefix, field)
    block = condition_coordinate(reports, block_prefix, field)
    return {
        "field": field,
        "condition": condition,
        "palindrome": palindrome,
        "block": block,
        "xi_palindrome_minus_block": palindrome["coordinate_left_minus_right"] - block["coordinate_left_minus_right"],
    }


def stratum_coordinate(
    reports: dict[str, Any],
    prefix: str,
    field: str,
    factor: str,
    level: Any,
) -> float | None:
    return pal.stratum_coordinate(reports, f"{prefix}_a", f"{prefix}_b", field, factor, level)


def differential_strata(reports: dict[str, Any], aggregate_alive: float, field: str = "R_spatial") -> dict[str, Any]:
    output: dict[str, Any] = {}
    aggregate_sign = 1 if aggregate_alive > 0 else -1 if aggregate_alive < 0 else 0
    for factor in STRATUM_FACTORS:
        levels = sorted({record[factor] for record in reports["palindrome_a_primary"]["block_records"]}, key=str)
        rows = []
        for level in levels:
            values: dict[str, Any] = {}
            for condition, (pal_prefix, block_prefix) in CONDITION_PREFIXES.items():
                pal_value = stratum_coordinate(reports, pal_prefix, field, factor, level)
                block_value = stratum_coordinate(reports, block_prefix, field, factor, level)
                values[condition] = {
                    "palindrome": pal_value,
                    "block": block_value,
                    "xi": None if pal_value is None or block_value is None else pal_value - block_value,
                }
            if values["alive"]["xi"] is None:
                continue
            denom = abs(float(values["alive"]["xi"]))
            ratios = {
                key: (abs(float(values[key]["xi"])) / denom if denom else None)
                for key in ("dead", "source_off")
            }
            sign = 1 if values["alive"]["xi"] > 0 else -1 if values["alive"]["xi"] < 0 else 0
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
            raw = pal.read_jsonl_member(archive, raw_name)
            pairs = pal.read_jsonl_member(archive, pair_name)
            deaths = pal.read_jsonl_member(archive, death_name)
            require(len(raw) == pal.true.replay.ROWS_PER_VARIANT, f"{variant} raw count mismatch: {len(raw)}")
            require(
                len(pairs) == pal.true.replay.ROWS_PER_VARIANT * pal.true.replay.PAIR_SAMPLE_COUNT,
                f"{variant} pair count mismatch: {len(pairs)}",
            )
            require(len(deaths) == pal.true.replay.ROWS_PER_VARIANT, f"{variant} death count mismatch: {len(deaths)}")
            reports[variant] = pal.variant_report(raw, len(pairs), len(deaths))

    r_reports = {condition: differential_for_condition(reports, condition, "R_spatial") for condition in CONDITION_PREFIXES}
    prep_reports = {condition: differential_for_condition(reports, condition, "prep_axis") for condition in CONDITION_PREFIXES}
    alive = r_reports["alive"]["xi_palindrome_minus_block"]
    denom = abs(alive)
    coordinates = {
        "Xi_alive_palindrome_minus_block": alive,
        "Xi_dead_palindrome_minus_block": r_reports["dead"]["xi_palindrome_minus_block"],
        "Xi_source_off_palindrome_minus_block": r_reports["source_off"]["xi_palindrome_minus_block"],
        "source_off_abs_to_alive_abs": abs(r_reports["source_off"]["xi_palindrome_minus_block"]) / denom if denom else None,
        "dead_abs_to_alive_abs": abs(r_reports["dead"]["xi_palindrome_minus_block"]) / denom if denom else None,
        "Omega_alive_palindrome": r_reports["alive"]["palindrome"]["coordinate_left_minus_right"],
        "Omega_alive_block": r_reports["alive"]["block"]["coordinate_left_minus_right"],
        "Omega_dead_palindrome": r_reports["dead"]["palindrome"]["coordinate_left_minus_right"],
        "Omega_dead_block": r_reports["dead"]["block"]["coordinate_left_minus_right"],
        "Omega_source_off_palindrome": r_reports["source_off"]["palindrome"]["coordinate_left_minus_right"],
        "Omega_source_off_block": r_reports["source_off"]["block"]["coordinate_left_minus_right"],
        "K_prep_axis_Xi_alive": prep_reports["alive"]["xi_palindrome_minus_block"],
        "K_prep_axis_Xi_dead": prep_reports["dead"]["xi_palindrome_minus_block"],
        "K_prep_axis_Xi_source_off": prep_reports["source_off"]["xi_palindrome_minus_block"],
    }
    strata = differential_strata(reports, alive, "R_spatial")
    max_source_off = max(report["max_source_off_ratio"] for report in strata.values())
    min_dead = min(report["min_dead_ratio"] for report in strata.values())
    interpretation = {
        "palindrome_minus_block_coordinate_nonzero": denom > 0.0,
        "aggregate_source_off_collapses_below_25pct": coordinates["source_off_abs_to_alive_abs"] <= 0.25 if denom else False,
        "aggregate_dead_preserves_at_least_25pct": coordinates["dead_abs_to_alive_abs"] >= 0.25 if denom else False,
        "all_one_factor_alive_strata_same_sign": all(report["all_alive_same_sign_as_aggregate"] for report in strata.values()),
        "all_one_factor_source_off_below_25pct": max_source_off <= 0.25,
        "all_one_factor_dead_preserves_25pct": min_dead >= 0.25,
        "max_one_factor_source_off_ratio": max_source_off,
        "min_one_factor_dead_ratio": min_dead,
        "palindrome_minus_block_source_off_dead_candidate": False,
        "exploratory_only": True,
        "small_wall_crossed": False,
    }
    interpretation["palindrome_minus_block_source_off_dead_candidate"] = (
        interpretation["palindrome_minus_block_coordinate_nonzero"]
        and interpretation["aggregate_source_off_collapses_below_25pct"]
        and interpretation["aggregate_dead_preserves_at_least_25pct"]
        and interpretation["all_one_factor_alive_strata_same_sign"]
        and interpretation["all_one_factor_source_off_below_25pct"]
        and interpretation["all_one_factor_dead_preserves_25pct"]
    )
    result = {
        "schema": "FAMILY10H_RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_ANALYSIS_V1",
        "created_at": pal.true.replay.utc_now(),
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
    result["analysis_sha256"] = pal.true.replay.digest({k: v for k, v in result.items() if k != "analysis_sha256"})
    return result


def write_summary(analysis: dict[str, Any]) -> None:
    c = analysis["coordinates"]
    i = analysis["interpretation"]
    lines = [
        "# Relation Palindrome-Minus-Block Source-Off Screen",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{analysis['archive_sha256']}`",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`",
        "",
        "Differential coordinates:",
        f"- alive Xi palindrome-minus-block: `{c['Xi_alive_palindrome_minus_block']:.9f}`",
        f"- dead Xi palindrome-minus-block: `{c['Xi_dead_palindrome_minus_block']:.9f}`",
        f"- source-off Xi palindrome-minus-block: `{c['Xi_source_off_palindrome_minus_block']:.9f}`",
        f"- source-off abs/alive abs: `{c['source_off_abs_to_alive_abs']:.3f}`",
        f"- dead abs/alive abs: `{c['dead_abs_to_alive_abs']:.3f}`",
        "",
        "Raw components:",
        f"- alive palindrome Omega: `{c['Omega_alive_palindrome']:.9f}`",
        f"- alive block Omega: `{c['Omega_alive_block']:.9f}`",
        f"- dead palindrome Omega: `{c['Omega_dead_palindrome']:.9f}`",
        f"- dead block Omega: `{c['Omega_dead_block']:.9f}`",
        f"- source-off palindrome Omega: `{c['Omega_source_off_palindrome']:.9f}`",
        f"- source-off block Omega: `{c['Omega_source_off_block']:.9f}`",
        "",
        "Interpretation:",
        f"- palindrome-minus-block source-off/dead candidate: `{i['palindrome_minus_block_source_off_dead_candidate']}`",
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
    configure_pal_runner()
    pal.true.replay.patch_runtime_for_composition_loop = patch_palindrome_minus_block_controls
    result = pal.true.replay.prepare()
    result["schema"] = "FAMILY10H_RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_PREPARE_V1"
    pal.true.replay.write_json(HERE / "RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_pal_runner()
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    require(prepare_result["passed"], "prepare failed")
    package = pal.true.replay.base.build_package()
    controller = pal.true.replay.base.deploy_execute_copyback(package)
    analysis = analyze_archive(controller)
    pal.true.replay.write_json(RUN_ROOT / "RELATION_PALINDROME_MINUS_BLOCK_SOURCE_OFF_SCREEN_ANALYSIS.json", analysis)
    pal.true.replay.write_json(SUMMARY_JSON, analysis)
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
