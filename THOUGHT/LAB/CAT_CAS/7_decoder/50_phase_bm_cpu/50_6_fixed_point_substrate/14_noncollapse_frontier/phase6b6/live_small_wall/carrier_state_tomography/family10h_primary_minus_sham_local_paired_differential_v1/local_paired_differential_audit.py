from __future__ import annotations

import gc
import hashlib
import itertools
import json
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
CARRIER_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = CARRIER_ROOT / "family10h_primary_minus_sham_differential_v1"
RUN_ROOT = CARRIER_ROOT / "runs"
DISCOVERY_ROOT = (
    CARRIER_ROOT
    / "family10h_relation_spatial_pair_readout_v1_1_segmented_candidate"
    / "primary_minus_sham_differential_v1"
)
POSITION_ROOT = RUN_ROOT / "family10h_relation_sham_position_exploration_0" / "attempt_1"

sys.path.insert(0, str(PACKAGE_ROOT))
import primary_minus_sham_differential_adjudication as official_adj  # type: ignore  # noqa: E402
import primary_minus_sham_differential_public as law  # type: ignore  # noqa: E402


FACTOR_STRATA = tuple(law.FACTOR_STRATA)
PRIMARY_QUERY = law.PRIMARY_QUERY
SHAM_QUERY = law.ANTI_RELATION_QUERY
GENERIC_CONTROL_QUERIES = tuple(law.GENERIC_CONTROL_QUERIES)
ENVELOPE_FRACTION = law.GENERIC_CONTROL_ENVELOPE_FRACTION


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def digest_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average empty sequence")
    return sum(items) / len(items)


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "median": None,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "median": statistics.median(ordered),
        "positive_count": sum(1 for value in ordered if value > 0),
        "negative_count": sum(1 for value in ordered if value < 0),
        "zero_count": sum(1 for value in ordered if value == 0),
    }


def parse_variant_records(tf: tarfile.TarFile, output_prefix: str, variant: str) -> dict[str, Any]:
    prefix = f"{output_prefix}/{variant}"
    raw = official_adj.spatial_adj.parse_jsonl_bytes(
        official_adj.read_member(tf, f"{prefix}/raw_records.jsonl")
    )
    pairs = official_adj.spatial_adj.parse_jsonl_bytes(
        official_adj.read_member(tf, f"{prefix}/pair_observations.jsonl")
    )
    deaths = official_adj.spatial_adj.parse_jsonl_bytes(
        official_adj.read_member(tf, f"{prefix}/source_death_receipts.jsonl")
    )
    packet = {
        "raw_records": raw,
        "pair_observations": pairs,
        "source_death_receipts": deaths,
    }
    rows = official_adj.spatial_adj.row_c_pairs(packet)
    records, failures = official_adj.spatial_adj.r_spatial_records(rows)
    lifecycle_valid = all(
        death.get("source_alive_during_query") is True
        and death.get("source_alive_at_pair_measurement") is True
        for death in deaths
    )
    result = {
        "raw_record_count": len(raw),
        "pair_observation_count": len(pairs),
        "source_death_receipt_count": len(deaths),
        "source_lifecycle_valid": lifecycle_valid,
        "block_count": len(records),
        "failures": failures,
        "records": records,
    }
    del raw, pairs, deaths, packet, rows
    gc.collect()
    return result


def load_primary_sham_records(archive_path: Path, output_prefix: str) -> dict[str, Any]:
    rounds: dict[str, Any] = {}
    with tarfile.open(archive_path, "r:gz") as tf:
        for round_index in range(law.ROUND_COUNT):
            round_key = str(round_index)
            rounds[round_key] = {}
            for query in (PRIMARY_QUERY, SHAM_QUERY):
                variant = f"round{round_index}_{query}"
                rounds[round_key][query] = parse_variant_records(tf, output_prefix, variant)
    return rounds


def group_records(records: list[dict[str, Any]], factors: tuple[str, ...]) -> dict[tuple[str, ...], list[float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record[factor]) for factor in factors)].append(float(record["R_spatial"]))
    return grouped


def compare_groups(
    primary_records: list[dict[str, Any]],
    sham_records: list[dict[str, Any]],
    factors: tuple[str, ...],
) -> dict[str, Any]:
    primary_groups = group_records(primary_records, factors)
    sham_groups = group_records(sham_records, factors)
    shared_keys = sorted(set(primary_groups) & set(sham_groups))
    cells = []
    for key in shared_keys:
        primary_mean = mean(primary_groups[key])
        sham_mean = mean(sham_groups[key])
        d_value = primary_mean - sham_mean
        cells.append(
            {
                "levels": dict(zip(factors, key)),
                "primary_mean": primary_mean,
                "sham_mean": sham_mean,
                "D_primary_minus_sham": d_value,
                "D_gt_0": d_value > 0,
                "primary_gt_0": primary_mean > 0,
                "sham_lt_0": sham_mean < 0,
                "primary_count": len(primary_groups[key]),
                "sham_count": len(sham_groups[key]),
            }
        )
    d_values = [cell["D_primary_minus_sham"] for cell in cells]
    return {
        "factors": list(factors),
        "cell_count": len(cells),
        "D_distribution": distribution(d_values),
        "D_positive_count": sum(1 for cell in cells if cell["D_gt_0"]),
        "primary_positive_count": sum(1 for cell in cells if cell["primary_gt_0"]),
        "sham_negative_count": sum(1 for cell in cells if cell["sham_lt_0"]),
        "D_positive_fraction": (sum(1 for cell in cells if cell["D_gt_0"]) / len(cells)) if cells else None,
        "primary_positive_fraction": (sum(1 for cell in cells if cell["primary_gt_0"]) / len(cells)) if cells else None,
        "sham_negative_fraction": (sum(1 for cell in cells if cell["sham_lt_0"]) / len(cells)) if cells else None,
        "weakest_cells": sorted(cells, key=lambda cell: cell["D_primary_minus_sham"])[:8],
    }


def round_record_audit(records_by_round: dict[str, Any]) -> dict[str, Any]:
    round_reports: dict[str, Any] = {}
    for round_key, round_records in records_by_round.items():
        primary_records = round_records[PRIMARY_QUERY]["records"]
        sham_records = round_records[SHAM_QUERY]["records"]
        primary_mean = mean(float(record["R_spatial"]) for record in primary_records)
        sham_mean = mean(float(record["R_spatial"]) for record in sham_records)
        d_value = primary_mean - sham_mean
        by_granularity: dict[str, Any] = {}
        one_factor_details: dict[str, Any] = {}
        for granularity in range(1, len(FACTOR_STRATA) + 1):
            combo_reports = []
            for combo in itertools.combinations(FACTOR_STRATA, granularity):
                combo_report = compare_groups(primary_records, sham_records, combo)
                combo_reports.append(combo_report)
                if granularity == 1:
                    one_factor_details[combo[0]] = combo_report
            total_cells = sum(item["cell_count"] for item in combo_reports)
            by_granularity[str(granularity)] = {
                "combo_count": len(combo_reports),
                "total_cell_count": total_cells,
                "D_positive_cell_count": sum(item["D_positive_count"] for item in combo_reports),
                "primary_positive_cell_count": sum(item["primary_positive_count"] for item in combo_reports),
                "sham_negative_cell_count": sum(item["sham_negative_count"] for item in combo_reports),
                "D_positive_fraction": (
                    sum(item["D_positive_count"] for item in combo_reports) / total_cells
                ),
                "primary_positive_fraction": (
                    sum(item["primary_positive_count"] for item in combo_reports) / total_cells
                ),
                "sham_negative_fraction": (
                    sum(item["sham_negative_count"] for item in combo_reports) / total_cells
                ),
                "minimum_D": min(item["D_distribution"]["min"] for item in combo_reports),
                "weakest_combo_cells": sorted(
                    [
                        {
                            "factors": item["factors"],
                            "cell": cell,
                        }
                        for item in combo_reports
                        for cell in item["weakest_cells"][:2]
                    ],
                    key=lambda item: item["cell"]["D_primary_minus_sham"],
                )[:8],
            }
        round_reports[round_key] = {
            "R_primary": primary_mean,
            "R_sham": sham_mean,
            "D_primary_minus_sham": d_value,
            "R_primary_gt_0": primary_mean > 0,
            "R_sham_lt_0": sham_mean < 0,
            "D_gt_0": d_value > 0,
            "record_counts": {
                "primary_blocks": len(primary_records),
                "sham_blocks": len(sham_records),
                "primary_raw_records": round_records[PRIMARY_QUERY]["raw_record_count"],
                "sham_raw_records": round_records[SHAM_QUERY]["raw_record_count"],
                "primary_pair_observations": round_records[PRIMARY_QUERY]["pair_observation_count"],
                "sham_pair_observations": round_records[SHAM_QUERY]["pair_observation_count"],
                "primary_source_death_receipts": round_records[PRIMARY_QUERY]["source_death_receipt_count"],
                "sham_source_death_receipts": round_records[SHAM_QUERY]["source_death_receipt_count"],
            },
            "source_lifecycle_valid": (
                round_records[PRIMARY_QUERY]["source_lifecycle_valid"]
                and round_records[SHAM_QUERY]["source_lifecycle_valid"]
            ),
            "block_failures": {
                "primary": round_records[PRIMARY_QUERY]["failures"],
                "sham": round_records[SHAM_QUERY]["failures"],
            },
            "one_factor_strata": one_factor_details,
            "granularity_summary": by_granularity,
        }
    return round_reports


def round_generic_envelope_from_existing(name: str, existing_rounds: dict[str, Any]) -> dict[str, Any]:
    reports = {}
    for round_key, item in existing_rounds.items():
        d_value = item["D_primary_minus_sham"]
        max_generic = item.get("max_abs_generic_control")
        if max_generic is None:
            max_generic = item.get("max_abs_generic_control_mean")
        ratio = item.get("max_abs_generic_control_to_D")
        if ratio is None:
            ratio = item.get("max_generic_abs_to_differential_ratio")
        reports[round_key] = {
            "D_primary_minus_sham": d_value,
            "max_abs_generic_control": max_generic,
            "max_abs_generic_control_to_D": ratio,
            "passed_0_25_D_envelope": bool(d_value > 0 and max_generic <= ENVELOPE_FRACTION * d_value),
            "generic_control_means": item.get("generic_control_means", {}),
            "generic_q99_warning_queries": item.get("generic_q99_warning_queries", []),
            "source": name,
        }
    return reports


def extract_single_factor_counts(round_reports: dict[str, Any]) -> dict[str, Any]:
    total = 0
    d_positive = 0
    primary_positive = 0
    sham_negative = 0
    weakest = []
    for round_key, report in round_reports.items():
        for factor, factor_report in report["one_factor_strata"].items():
            total += factor_report["cell_count"]
            d_positive += factor_report["D_positive_count"]
            primary_positive += factor_report["primary_positive_count"]
            sham_negative += factor_report["sham_negative_count"]
            for cell in factor_report["weakest_cells"]:
                weakest.append({"round": round_key, "factor": factor, "cell": cell})
    return {
        "cell_count": total,
        "D_positive_count": d_positive,
        "primary_positive_count": primary_positive,
        "sham_negative_count": sham_negative,
        "D_positive_fraction": d_positive / total,
        "primary_positive_fraction": primary_positive / total,
        "sham_negative_fraction": sham_negative / total,
        "weakest_cells": sorted(weakest, key=lambda item: item["cell"]["D_primary_minus_sham"])[:12],
    }


def compact_rounds(round_reports: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "R_primary",
        "R_sham",
        "D_primary_minus_sham",
        "R_primary_gt_0",
        "R_sham_lt_0",
        "D_gt_0",
        "record_counts",
        "source_lifecycle_valid",
        "block_failures",
    )
    return {round_key: {key: report[key] for key in keys} for round_key, report in round_reports.items()}


def position_differentials(position_analysis: dict[str, Any]) -> dict[str, Any]:
    summary = position_analysis["diagnostic_summary"]
    comparisons = {
        "primary_alone_minus_sham_alone": summary["primary_alone_mean"] - summary["sham_alone_mean"],
        "primary_after_sham_minus_sham_first": summary["primary_after_sham_mean"] - summary["sham_first_mean"],
        "primary_before_sham_minus_sham_after_primary_original_offset": (
            summary["primary_before_sham_mean"] - summary["sham_after_primary_original_offset_mean"]
        ),
    }
    return {
        "sham_sign_samples": {
            "sham_alone": summary["sham_alone_mean"],
            "sham_first": summary["sham_first_mean"],
            "sham_after_primary_original_offset": summary["sham_after_primary_original_offset_mean"],
        },
        "primary_sign_samples": {
            "primary_alone": summary["primary_alone_mean"],
            "primary_after_sham": summary["primary_after_sham_mean"],
            "primary_before_sham": summary["primary_before_sham_mean"],
        },
        "paired_differentials": comparisons,
        "all_position_differentials_positive": all(value > 0 for value in comparisons.values()),
        "position_disposition": position_analysis["diagnostic_disposition"],
    }


def render_law(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Prospective Local Paired Differential Law",
            "",
            "Package identity:",
            "",
            "```text",
            "family10h_primary_minus_sham_local_paired_differential_v1",
            "```",
            "",
            "Purpose:",
            "",
            "Freeze the smallest prospective repair to the primary-minus-sham differential law. The repaired coordinate treats `relation_sham` as a local anti-relation comparator whose absolute sign may drift across fresh carrier resets. The prospective coordinate is the paired differential/order statistic:",
            "",
            "```text",
            "D_local = R_primary - R_sham",
            "R_primary = mean R_spatial(query_relation_pair)",
            "R_sham = mean R_spatial(relation_sham)",
            "```",
            "",
            "The law does not require `R_sham < 0`. The old absolute-sham-sign gate is retired for this coordinate because retained evidence shows that the sham baseline can cross zero while the local primary-minus-sham ordering persists.",
            "",
            "## Frozen Acquisition Shape",
            "",
            "- Reuse the existing Family 10h relation-spatial runtime, PMU helper, source/receiver boundary, target identity, sensor identity, factors, and five query variants.",
            "- Use fresh carrier state and a fresh runtime process for each segment.",
            "- Run the same two independently reset rounds.",
            "- Preserve the existing factor strata: session, replicate, mapping, source order, query order, and cyclic origin.",
            "- Do not add new queries, factors, thresholds, post-run feature selection, or schedule edits.",
            "- Preserve generic matched-permutation q99 as a reported diagnostic only.",
            "",
            "## Prospective Success Law",
            "",
            "All of the following must pass in a fresh prospective physical attempt:",
            "",
            "1. All custody, source lifecycle, target identity, sensor identity, runtime hash, PMU preflight, manifest, schedule, and no-fixture gates pass.",
            "2. Every round has `D_local > 0`.",
            "3. Every round has `R_primary > 0` as a source-orientation sanity gate.",
            "4. Every one-factor matched stratum for session, replicate, mapping, source order, query order, and cyclic origin has `D_local > 0`.",
            "5. Every one-factor matched stratum has `R_primary > 0`.",
            "6. No stratum or round requires `R_sham < 0`.",
            "7. Generic controls pass the existing envelope in every round:",
            "",
            "```text",
            "max(abs(R_scrambled), abs(R_distance), abs(R_route)) <= 0.25 * D_local",
            "```",
            "",
            "8. `relation_sham` remains excluded from the generic-null set.",
            "9. No post-observation threshold revision or target-derived fitting is permitted.",
            "",
            "The six-factor crossed-cell audit is diagnostic only for this law. Each complete crossed cell has four block samples in the retained schedule; freezing a full crossed-cell sign gate would overfit sparse cells and change the experiment more than this repair requires.",
            "",
            "## Result Vocabulary",
            "",
            "```text",
            "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CONFIRMED_PROSPECTIVE",
            "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_NOT_CONFIRMED_PROSPECTIVE",
            "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CUSTODY_INVALID",
            "```",
            "",
            "Maximum scientific claim:",
            "",
            "```text",
            "PUBLIC_POST_SOURCE_LOCAL_PRIMARY_MINUS_SHAM_DIFFERENTIAL_CONFIRMED",
            "```",
            "",
            "This law cannot establish full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing by itself.",
            "",
            "## Freeze Basis",
            "",
            f"This law is frozen from offline audit `{report['audit_sha256']}` of the retained exploratory and official physical evidence. The audit supports a minimal prospective confirmation package, not retrospective promotion.",
            "",
        ]
    ) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Local Paired Differential Audit",
        "",
        f"Audit SHA-256: `{report['audit_sha256']}`",
        f"Decision: `{decision['audit_result']}`",
        "",
        "## Diagnosis",
        "",
        "The sham baseline crosses zero because its absolute mean is a weak near-zero baseline that is sensitive to fresh-process carrier offset. The position exploration shows the negative sham response reproduces when sham is alone, first, or after primary, so the official round-0 positive sham is not explained by primary-before-sham positioning alone.",
        "",
        "The durable candidate is the local ordering: `query_relation_pair` stays above `relation_sham`, and the generic controls stay small relative to that separation. The repaired law therefore gates on `D = R_primary - R_sham`, not on the absolute sign of `R_sham`.",
        "",
        "## Round Evidence",
        "",
        "| Evidence | Round | R_primary | R_sham | D | sham<0 | D>0 | generic/D | envelope |",
        "|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for evidence_name in ("exploratory_differential", "official_differential"):
        rounds = report["evidence"][evidence_name]["rounds"]
        generic = report["evidence"][evidence_name]["generic_control_envelope_by_round"]
        for round_key in sorted(rounds, key=int):
            item = rounds[round_key]
            ratio = generic[round_key]["max_abs_generic_control_to_D"]
            lines.append(
                f"| {evidence_name} | {round_key} | {item['R_primary']:.9f} | {item['R_sham']:.9f} | "
                f"{item['D_primary_minus_sham']:.9f} | `{item['R_sham_lt_0']}` | `{item['D_gt_0']}` | "
                f"{ratio:.3f} | `{generic[round_key]['passed_0_25_D_envelope']}` |"
            )
    lines.extend(
        [
            "",
            "## Local Strata",
            "",
            "| Evidence | one-factor cells | D>0 | primary>0 | sham<0 | weakest D |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for evidence_name in ("exploratory_differential", "official_differential"):
        single = report["evidence"][evidence_name]["single_factor_strata_summary"]
        weakest = single["weakest_cells"][0]["cell"]["D_primary_minus_sham"]
        lines.append(
            f"| {evidence_name} | {single['cell_count']} | {single['D_positive_count']} | "
            f"{single['primary_positive_count']} | {single['sham_negative_count']} | {weakest:.9f} |"
        )
    lines.extend(
        [
            "",
            "Complete six-factor crossed cells are retained as diagnostics, not gates. The retained schedule gives only four block samples per crossed cell, and those sparse cells invert in both exploratory and official evidence even while round and one-factor strata are stable.",
            "",
            "## Generic Controls",
            "",
            "All four differential rounds pass the retained `0.25 * D` generic-control envelope. q99 warnings remain diagnostics and are not removed.",
            "",
            "## Frozen Next Step",
            "",
            "The audit supports freezing the revised prospective local paired differential law. It does not retroactively confirm the official failed attempt, and it does not authorize a new live attempt. The smallest confirmation package is the existing segmented shape plus the repaired adjudication law that removes only `R_sham < 0`.",
            "",
            "## Claim Boundary",
            "",
            f"Retrospective audit result: `{decision['audit_result']}`",
            f"Prospective law frozen: `{decision['prospective_law_frozen']}`",
            "Small Wall crossed: `false`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report() -> dict[str, Any]:
    official_adjudication_path = RUN_ROOT / "family10h_primary_minus_sham_differential_v1_0" / "attempt_1" / "PRIMARY_MINUS_SHAM_DIFFERENTIAL_ADJUDICATION.json"
    official_copyback_path = RUN_ROOT / "family10h_primary_minus_sham_differential_v1_0" / "attempt_1" / "OFFICIAL_COPY_BACK_VERIFICATION.json"
    official_archive_path = RUN_ROOT / "family10h_primary_minus_sham_differential_v1_0" / "attempt_1" / "OFFICIAL_TARGET_ROOT.tar.gz"
    discovery_summary_path = DISCOVERY_ROOT / "PRIMARY_MINUS_SHAM_DIFFERENTIAL_SUMMARY.json"
    discovery_archive_path = RUN_ROOT / "family10h_primary_minus_sham_differential_0" / "attempt_1" / "DIFFERENTIAL_TARGET_ROOT.tar.gz"
    discovery_copyback_path = RUN_ROOT / "family10h_primary_minus_sham_differential_0" / "attempt_1" / "DIFFERENTIAL_COPY_BACK_VERIFICATION.json"
    position_analysis_path = POSITION_ROOT / "EXPLORATORY_POSITION_ANALYSIS.json"

    official_adjudication = read_json(official_adjudication_path)
    discovery_summary = read_json(discovery_summary_path)
    position_analysis = read_json(position_analysis_path)

    exploratory_records = load_primary_sham_records(discovery_archive_path, "source/differential_outputs")
    official_records = load_primary_sham_records(official_archive_path, "source/official_outputs")

    exploratory_rounds = round_record_audit(exploratory_records)
    official_rounds = round_record_audit(official_records)
    exploratory_single = extract_single_factor_counts(exploratory_rounds)
    official_single = extract_single_factor_counts(official_rounds)

    exploratory_generic = round_generic_envelope_from_existing(
        "PRIMARY_MINUS_SHAM_DIFFERENTIAL_SUMMARY.json",
        discovery_summary["differential_by_round"],
    )
    official_generic = round_generic_envelope_from_existing(
        "PRIMARY_MINUS_SHAM_DIFFERENTIAL_ADJUDICATION.json",
        official_adjudication["differential_by_round"],
    )

    position_report = position_differentials(position_analysis)
    absolute_sham_samples = {
        "position_exploration": position_report["sham_sign_samples"],
        "exploratory_differential_rounds": {
            round_key: round_report["R_sham"] for round_key, round_report in exploratory_rounds.items()
        },
        "official_differential_rounds": {
            round_key: round_report["R_sham"] for round_key, round_report in official_rounds.items()
        },
    }
    all_sham_values = [
        value for sample_group in absolute_sham_samples.values() for value in sample_group.values()
    ]
    d_values = [
        round_report["D_primary_minus_sham"]
        for round_report in itertools.chain(exploratory_rounds.values(), official_rounds.values())
    ]
    all_generic_envelopes = list(exploratory_generic.values()) + list(official_generic.values())

    evidence_supports_revised_law = (
        all(value > 0 for value in d_values)
        and exploratory_single["D_positive_count"] == exploratory_single["cell_count"]
        and official_single["D_positive_count"] == official_single["cell_count"]
        and all(item["passed_0_25_D_envelope"] for item in all_generic_envelopes)
        and min(all_sham_values) < 0
        and max(all_sham_values) > 0
    )

    report: dict[str, Any] = {
        "schema": "FAMILY10H_PRIMARY_MINUS_SHAM_LOCAL_PAIRED_DIFFERENTIAL_AUDIT_V1",
        "package_id": "family10h_primary_minus_sham_local_paired_differential_v1",
        "question": "Why does relation_sham cross zero while primary-minus-sham ordering persists?",
        "input_evidence": {
            "official_adjudication": {
                "path": str(official_adjudication_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(official_adjudication_path),
                "result_class": official_adjudication["result_class"],
                "scientific_claim": official_adjudication["scientific_claim"],
                "archive_sha256": official_adjudication["archive_sha256"],
            },
            "official_copyback": {
                "path": str(official_copyback_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(official_copyback_path),
                "passed": read_json(official_copyback_path)["passed"],
            },
            "official_archive": {
                "path": str(official_archive_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(official_archive_path),
                "size_bytes": official_archive_path.stat().st_size,
            },
            "exploratory_differential_summary": {
                "path": str(discovery_summary_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(discovery_summary_path),
                "diagnostic_disposition": discovery_summary["diagnostic_disposition"],
                "archive_sha256": discovery_summary["archive_sha256"],
            },
            "exploratory_differential_copyback": {
                "path": str(discovery_copyback_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(discovery_copyback_path),
                "passed": read_json(discovery_copyback_path)["passed"],
            },
            "exploratory_differential_archive": {
                "path": str(discovery_archive_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(discovery_archive_path),
                "size_bytes": discovery_archive_path.stat().st_size,
            },
            "position_exploration_analysis": {
                "path": str(position_analysis_path.relative_to(CARRIER_ROOT)),
                "sha256": file_sha256(position_analysis_path),
                "diagnostic_disposition": position_analysis["diagnostic_disposition"],
            },
        },
        "diagnosis": {
            "absolute_sham_sign_instability": {
                "observed": True,
                "crosses_zero": min(all_sham_values) < 0 and max(all_sham_values) > 0,
                "samples": absolute_sham_samples,
                "distribution": distribution(all_sham_values),
                "interpretation": (
                    "relation_sham is a weak near-zero local comparator. Its absolute sign is not stable "
                    "across fresh carrier resets, while the primary-minus-sham separation is stable at "
                    "round and one-factor matched-stratum levels."
                ),
            },
            "position_after_primary_not_sufficient_explanation": position_report,
            "paired_ordering_stability": {
                "round_D_distribution": distribution(d_values),
                "all_round_D_positive": all(value > 0 for value in d_values),
                "single_factor_cells_total": exploratory_single["cell_count"] + official_single["cell_count"],
                "single_factor_D_positive_total": exploratory_single["D_positive_count"] + official_single["D_positive_count"],
                "single_factor_primary_positive_total": exploratory_single["primary_positive_count"] + official_single["primary_positive_count"],
                "single_factor_sham_negative_total": exploratory_single["sham_negative_count"] + official_single["sham_negative_count"],
            },
            "generic_control_relative_envelope_stability": {
                "round_count": len(all_generic_envelopes),
                "passed_count": sum(1 for item in all_generic_envelopes if item["passed_0_25_D_envelope"]),
                "envelope_fraction": ENVELOPE_FRACTION,
                "q99_warnings_preserved_as_diagnostics": True,
            },
            "factor_levels_where_ordering_weakens": {
                "exploratory_weakest_one_factor_cells": exploratory_single["weakest_cells"],
                "official_weakest_one_factor_cells": official_single["weakest_cells"],
                "crossed_cell_warning": (
                    "Six-factor crossed cells are sparse with four block samples per cell and do not "
                    "support a prospective crossed-cell sign gate."
                ),
            },
        },
        "evidence": {
            "exploratory_differential": {
                "rounds": compact_rounds(exploratory_rounds),
                "single_factor_strata_summary": exploratory_single,
                "granularity_summary_by_round": {
                    round_key: round_report["granularity_summary"] for round_key, round_report in exploratory_rounds.items()
                },
                "generic_control_envelope_by_round": exploratory_generic,
            },
            "official_differential": {
                "rounds": compact_rounds(official_rounds),
                "single_factor_strata_summary": official_single,
                "granularity_summary_by_round": {
                    round_key: round_report["granularity_summary"] for round_key, round_report in official_rounds.items()
                },
                "generic_control_envelope_by_round": official_generic,
            },
            "position_exploration": position_report,
        },
        "decision": {
            "audit_result": (
                "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_LAW_SUPPORTED_FOR_PROSPECTIVE_FREEZE"
                if evidence_supports_revised_law
                else "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_LAW_NOT_SUPPORTED"
            ),
            "existing_evidence_supports_revised_law": evidence_supports_revised_law,
            "prospective_law_frozen": evidence_supports_revised_law,
            "retrospective_promotion": False,
            "new_live_attempt_authorized": False,
            "small_wall_crossed": False,
            "minimal_confirmation_package_identified": evidence_supports_revised_law,
            "minimal_confirmation_package": (
                "Reuse the existing primary-minus-sham segmented acquisition shape and replace only "
                "the R_sham < 0 adjudication gate with local paired D > 0 ordering gates."
            ),
        },
    }
    report["audit_sha256"] = digest_payload(report)
    return report


def main() -> int:
    report = build_report()
    write_json(SCRIPT_DIR / "LOCAL_PAIRED_DIFFERENTIAL_AUDIT.json", report)
    (SCRIPT_DIR / "LOCAL_PAIRED_DIFFERENTIAL_AUDIT.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    (SCRIPT_DIR / "PROSPECTIVE_LOCAL_PAIRED_DIFFERENTIAL_LAW.md").write_text(
        render_law(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit_result": report["decision"]["audit_result"],
                "audit_sha256": report["audit_sha256"],
                "prospective_law_frozen": report["decision"]["prospective_law_frozen"],
                "small_wall_crossed": report["decision"]["small_wall_crossed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
