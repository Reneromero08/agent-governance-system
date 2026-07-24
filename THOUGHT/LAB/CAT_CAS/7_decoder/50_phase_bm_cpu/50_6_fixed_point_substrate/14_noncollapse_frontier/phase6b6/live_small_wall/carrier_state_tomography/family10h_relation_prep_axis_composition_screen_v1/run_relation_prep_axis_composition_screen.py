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

RUN_ID = "family10h_relation_prep_axis_composition_screen_v1_0"
PACKAGE_ID = "family10h_relation_prep_axis_composition_screen_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "PREP_AXIS_COMPOSITION_SCREEN_SCHEDULES"
ATTEMPT_LABEL = os.environ.get("FAMILY10H_DISCOVERY_ATTEMPT_LABEL", "attempt_1")
if any(sep in ATTEMPT_LABEL for sep in ("/", "\\", ":")) or not ATTEMPT_LABEL:
    raise RuntimeError(f"invalid attempt label: {ATTEMPT_LABEL!r}")
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / ATTEMPT_LABEL
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_PREP_AXIS_COMPOSITION_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_PREP_AXIS_COMPOSITION_SCREEN_SUMMARY.md"

AXIS = "prep_axis"
VARIANT_PAIRS = {
    "adjacent": ("adjacent_r0r1", "adjacent_r1r0"),
    "dead": ("dead_adjacent_r0r1", "dead_adjacent_r1r0"),
    "source_off": ("source_off_adjacent_r0r1", "source_off_adjacent_r1r0"),
    "gapped": ("gapped_r0r1", "gapped_r1r0"),
    "balanced": ("balanced_alt_a", "balanced_alt_b"),
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


def target_script() -> str:
    return (
        true.REPLAY_TARGET_SCRIPT()
        .replace("COMPOSITION_REPLAY_EXCLUSION_SCHEDULES", "PREP_AXIS_COMPOSITION_SCREEN_SCHEDULES")
        .replace("RELATION_COMPOSITION_REPLAY_EXCLUSION", "RELATION_PREP_AXIS_COMPOSITION_SCREEN")
    )


def variant_axis(base_analysis: dict[str, Any], variant: str, role: str) -> float:
    return float(base_analysis["variant_reports"][f"{variant}_{role}"]["relation_cell_axes"][AXIS])


def pair_report(base_analysis: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    left_primary = variant_axis(base_analysis, left, "primary")
    left_sham = variant_axis(base_analysis, left, "sham")
    right_primary = variant_axis(base_analysis, right, "primary")
    right_sham = variant_axis(base_analysis, right, "sham")
    left_delta = left_primary - left_sham
    right_delta = right_primary - right_sham
    return {
        "axis": AXIS,
        "left_variant": left,
        "right_variant": right,
        "left_primary_axis": left_primary,
        "left_sham_axis": left_sham,
        "right_primary_axis": right_primary,
        "right_sham_axis": right_sham,
        "left_primary_minus_sham": left_delta,
        "right_primary_minus_sham": right_delta,
        "order_coordinate_left_minus_right": left_delta - right_delta,
    }


def load_raw_records(archive_path: Path) -> dict[str, list[dict[str, Any]]]:
    variants = [f"{variant}_{role}" for pair in VARIANT_PAIRS.values() for variant in pair for role in ("primary", "sham")]
    records: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        for variant in variants:
            name = f"source/discovery_outputs/{variant}/raw_records.jsonl"
            true.replay.require(name in names, f"{variant} raw_records member missing")
            member = archive.extractfile(name)
            true.replay.require(member is not None, f"{variant} raw_records member not extractable")
            records[variant] = [json.loads(line) for line in member]
    return records


def block_axis_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                AXIS: 0.5 * (c00 + c01 - c10 - c11),
                "session": first["session"],
                "replicate": first["replicate"],
                "mapping": first["mapping"],
                "query_order": first["query_order"],
                "cyclic_origin": first["cyclic_origin"],
            }
        )
    return records


def mean_axis(records: list[dict[str, Any]], factor: str | None = None, level: Any | None = None) -> float | None:
    selected = records if factor is None else [record for record in records if record[factor] == level]
    if not selected:
        return None
    return mean(float(record[AXIS]) for record in selected)


def stratum_coordinate(
    axis_records: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
    factor: str | None = None,
    level: Any | None = None,
) -> float | None:
    left_primary = mean_axis(axis_records[f"{left}_primary"], factor, level)
    left_sham = mean_axis(axis_records[f"{left}_sham"], factor, level)
    right_primary = mean_axis(axis_records[f"{right}_primary"], factor, level)
    right_sham = mean_axis(axis_records[f"{right}_sham"], factor, level)
    if None in (left_primary, left_sham, right_primary, right_sham):
        return None
    return (left_primary - left_sham) - (right_primary - right_sham)


def stratum_reports(axis_records: dict[str, list[dict[str, Any]]], aggregate_adjacent: float) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    aggregate_sign = 1 if aggregate_adjacent > 0 else -1 if aggregate_adjacent < 0 else 0
    for factor in STRATUM_FACTORS:
        levels = sorted({record[factor] for record in axis_records["adjacent_r0r1_primary"]}, key=str)
        level_reports = []
        for level in levels:
            values = {
                label: stratum_coordinate(axis_records, left, right, factor, level)
                for label, (left, right) in VARIANT_PAIRS.items()
            }
            if values["adjacent"] is None:
                continue
            denom = abs(float(values["adjacent"]))
            ratios = {
                label: (abs(float(values[label])) / denom if denom else None)
                for label in ("dead", "source_off", "gapped", "balanced")
            }
            sign = 1 if values["adjacent"] > 0 else -1 if values["adjacent"] < 0 else 0
            level_reports.append(
                {
                    "level": level,
                    "coordinates": values,
                    "ratios_to_adjacent_abs": ratios,
                    "same_sign_as_aggregate": sign == aggregate_sign and sign != 0,
                }
            )
        reports[factor] = {
            "levels": level_reports,
            "all_adjacent_same_sign_as_aggregate": all(item["same_sign_as_aggregate"] for item in level_reports),
            "max_source_off_ratio": max(
                item["ratios_to_adjacent_abs"]["source_off"] for item in level_reports if item["ratios_to_adjacent_abs"]["source_off"] is not None
            ),
            "max_gapped_ratio": max(
                item["ratios_to_adjacent_abs"]["gapped"] for item in level_reports if item["ratios_to_adjacent_abs"]["gapped"] is not None
            ),
            "max_balanced_ratio": max(
                item["ratios_to_adjacent_abs"]["balanced"] for item in level_reports if item["ratios_to_adjacent_abs"]["balanced"] is not None
            ),
            "min_dead_ratio": min(
                item["ratios_to_adjacent_abs"]["dead"] for item in level_reports if item["ratios_to_adjacent_abs"]["dead"] is not None
            ),
        }
    return reports


def build_prep_axis_analysis(controller: dict[str, Any], base_analysis: dict[str, Any]) -> dict[str, Any]:
    pair_reports = {label: pair_report(base_analysis, left, right) for label, (left, right) in VARIANT_PAIRS.items()}
    coordinates = {
        "K_prep_axis_adjacent_r0r1_minus_r1r0": pair_reports["adjacent"]["order_coordinate_left_minus_right"],
        "K_prep_axis_dead_r0r1_minus_r1r0": pair_reports["dead"]["order_coordinate_left_minus_right"],
        "K_prep_axis_source_off_r0r1_minus_r1r0": pair_reports["source_off"]["order_coordinate_left_minus_right"],
        "K_prep_axis_gapped_r0r1_minus_r1r0": pair_reports["gapped"]["order_coordinate_left_minus_right"],
        "K_prep_axis_balanced_alt_a_minus_b": pair_reports["balanced"]["order_coordinate_left_minus_right"],
    }
    denom = abs(coordinates["K_prep_axis_adjacent_r0r1_minus_r1r0"])
    coordinates.update(
        {
            "source_off_abs_to_adjacent_abs": abs(coordinates["K_prep_axis_source_off_r0r1_minus_r1r0"]) / denom if denom else None,
            "dead_abs_to_adjacent_abs": abs(coordinates["K_prep_axis_dead_r0r1_minus_r1r0"]) / denom if denom else None,
            "gapped_abs_to_adjacent_abs": abs(coordinates["K_prep_axis_gapped_r0r1_minus_r1r0"]) / denom if denom else None,
            "balanced_abs_to_adjacent_abs": abs(coordinates["K_prep_axis_balanced_alt_a_minus_b"]) / denom if denom else None,
        }
    )
    raw_records = load_raw_records(LOCAL_ARCHIVE)
    axis_records = {variant: block_axis_records(rows) for variant, rows in raw_records.items()}
    strata = stratum_reports(axis_records, coordinates["K_prep_axis_adjacent_r0r1_minus_r1r0"])
    max_source_off = max(report["max_source_off_ratio"] for report in strata.values())
    max_gapped = max(report["max_gapped_ratio"] for report in strata.values())
    max_balanced = max(report["max_balanced_ratio"] for report in strata.values())
    min_dead = min(report["min_dead_ratio"] for report in strata.values())
    interpretation = {
        "prep_axis_coordinate_nonzero": denom > 0.0,
        "aggregate_source_off_collapses_below_25pct": coordinates["source_off_abs_to_adjacent_abs"] <= 0.25 if denom else False,
        "aggregate_dead_preserves_at_least_25pct": coordinates["dead_abs_to_adjacent_abs"] >= 0.25 if denom else False,
        "aggregate_gapped_collapses_below_25pct": coordinates["gapped_abs_to_adjacent_abs"] <= 0.25 if denom else False,
        "aggregate_balanced_collapses_below_25pct": coordinates["balanced_abs_to_adjacent_abs"] <= 0.25 if denom else False,
        "all_one_factor_adjacent_strata_same_sign": all(
            report["all_adjacent_same_sign_as_aggregate"] for report in strata.values()
        ),
        "all_one_factor_controls_pass_25pct": (
            max_source_off <= 0.25 and max_gapped <= 0.25 and max_balanced <= 0.25 and min_dead >= 0.25
        ),
        "max_one_factor_source_off_ratio": max_source_off,
        "max_one_factor_gapped_ratio": max_gapped,
        "max_one_factor_balanced_ratio": max_balanced,
        "min_one_factor_dead_ratio": min_dead,
        "prep_axis_replay_exclusion_candidate": False,
        "exploratory_only": True,
        "small_wall_crossed": False,
    }
    interpretation["prep_axis_replay_exclusion_candidate"] = (
        interpretation["prep_axis_coordinate_nonzero"]
        and interpretation["aggregate_source_off_collapses_below_25pct"]
        and interpretation["aggregate_dead_preserves_at_least_25pct"]
        and interpretation["aggregate_gapped_collapses_below_25pct"]
        and interpretation["aggregate_balanced_collapses_below_25pct"]
        and interpretation["all_one_factor_adjacent_strata_same_sign"]
        and interpretation["all_one_factor_controls_pass_25pct"]
    )
    result = {
        "schema": "FAMILY10H_RELATION_PREP_AXIS_COMPOSITION_SCREEN_ANALYSIS_V1",
        "created_at": true.replay.utc_now(),
        "run_id": RUN_ID,
        "controller_passed": controller["passed"],
        "archive_sha256": controller["archive_sha256"],
        "archive_size": controller["archive_size"],
        "base_true_composition_analysis_sha256": base_analysis["analysis_sha256"],
        "axis": AXIS,
        "pair_reports": pair_reports,
        "coordinates": coordinates,
        "stratum_reports": strata,
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
        "# Relation Prep-Axis Composition Screen",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Archive SHA-256: `{analysis['archive_sha256']}`",
        f"Analysis SHA-256: `{analysis['analysis_sha256']}`",
        "",
        "Prep-axis coordinates:",
        f"- adjacent K prep-axis R0->R1 minus R1->R0: `{c['K_prep_axis_adjacent_r0r1_minus_r1r0']:.9f}`",
        f"- dead K prep-axis R0->R1 minus R1->R0: `{c['K_prep_axis_dead_r0r1_minus_r1r0']:.9f}`",
        f"- source-off K prep-axis R0->R1 minus R1->R0: `{c['K_prep_axis_source_off_r0r1_minus_r1r0']:.9f}`",
        f"- gapped K prep-axis R0->R1 minus R1->R0: `{c['K_prep_axis_gapped_r0r1_minus_r1r0']:.9f}`",
        f"- balanced-alt K prep-axis phase A minus B: `{c['K_prep_axis_balanced_alt_a_minus_b']:.9f}`",
        f"- source-off abs/adjacent abs: `{c['source_off_abs_to_adjacent_abs']:.3f}`",
        f"- dead abs/adjacent abs: `{c['dead_abs_to_adjacent_abs']:.3f}`",
        f"- gapped abs/adjacent abs: `{c['gapped_abs_to_adjacent_abs']:.3f}`",
        f"- balanced abs/adjacent abs: `{c['balanced_abs_to_adjacent_abs']:.3f}`",
        "",
        "Interpretation:",
        f"- prep-axis replay-exclusion candidate: `{i['prep_axis_replay_exclusion_candidate']}`",
        f"- aggregate source-off below 0.25 x adjacent: `{i['aggregate_source_off_collapses_below_25pct']}`",
        f"- aggregate dead at least 0.25 x adjacent: `{i['aggregate_dead_preserves_at_least_25pct']}`",
        f"- aggregate gapped below 0.25 x adjacent: `{i['aggregate_gapped_collapses_below_25pct']}`",
        f"- aggregate balanced below 0.25 x adjacent: `{i['aggregate_balanced_collapses_below_25pct']}`",
        f"- all one-factor adjacent strata same sign: `{i['all_one_factor_adjacent_strata_same_sign']}`",
        f"- all one-factor controls pass 0.25 envelope: `{i['all_one_factor_controls_pass_25pct']}`",
        "",
        "This is exploratory evidence only. It emits no positive scientific claim and does not promote `SMALL_WALL_CROSSED`.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def prepare() -> dict[str, Any]:
    configure_true_runner()
    result = true.prepare()
    result["schema"] = "FAMILY10H_RELATION_PREP_AXIS_COMPOSITION_SCREEN_PREPARE_V1"
    true.replay.write_json(HERE / "RELATION_PREP_AXIS_COMPOSITION_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_true_runner()
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    true.replay.require(prepare_result["passed"], "prepare failed")
    package = true.replay.base.build_package()
    controller = true.replay.base.deploy_execute_copyback(package)
    base_analysis = true.replay.analyze_archive(controller)
    base_analysis["schema"] = "FAMILY10H_RELATION_PREP_AXIS_BASE_TRUE_COMPOSITION_ANALYSIS_V1"
    base_analysis["physical_source_composition_prep_repaired"] = True
    base_analysis["analysis_sha256"] = true.replay.digest({k: v for k, v in base_analysis.items() if k != "analysis_sha256"})
    true.replay.write_json(RUN_ROOT / "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_ANALYSIS.json", base_analysis)
    analysis = build_prep_axis_analysis(controller, base_analysis)
    true.replay.write_json(RUN_ROOT / "RELATION_PREP_AXIS_COMPOSITION_SCREEN_ANALYSIS.json", analysis)
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
