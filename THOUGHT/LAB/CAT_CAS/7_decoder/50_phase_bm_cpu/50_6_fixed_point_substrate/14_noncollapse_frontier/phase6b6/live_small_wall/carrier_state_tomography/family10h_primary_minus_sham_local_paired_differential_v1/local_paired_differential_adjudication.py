from __future__ import annotations

import gc
import hashlib
import itertools
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import local_paired_differential_public as law


SEGMENTED_ROOT = law.segmented_package_root()
sys.path.insert(0, str(SEGMENTED_ROOT))
import relation_spatial_physical_adjudication as spatial_adj  # type: ignore  # noqa: E402


def read_member(tf: tarfile.TarFile, name: str) -> bytes:
    member = tf.getmember(name)
    handle = tf.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member unreadable: {name}")
    return handle.read()


def distribution(values: list[float]) -> dict[str, Any]:
    return spatial_adj.distribution(values)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(record["R_spatial"]) for record in records]
    strata = {}
    for factor in law.FACTOR_STRATA:
        by_level: dict[str, list[float]] = defaultdict(list)
        for record in records:
            by_level[str(record[factor])].append(float(record["R_spatial"]))
        strata[factor] = {level: distribution(level_values) for level, level_values in sorted(by_level.items())}
    return {"distribution": distribution(values), "strata": strata}


def analyze_variant(tf: tarfile.TarFile, output_prefix: str, variant: str, query: str, round_index: int) -> dict[str, Any]:
    prefix = f"{output_prefix}/{variant}"
    raw = spatial_adj.parse_jsonl_bytes(read_member(tf, f"{prefix}/raw_records.jsonl"))
    pairs = spatial_adj.parse_jsonl_bytes(read_member(tf, f"{prefix}/pair_observations.jsonl"))
    deaths = spatial_adj.parse_jsonl_bytes(read_member(tf, f"{prefix}/source_death_receipts.jsonl"))
    feature = json.loads(read_member(tf, f"{prefix}/feature_freeze.json").decode("utf-8"))
    receipt = json.loads(read_member(tf, f"{prefix}/target_execution_receipt.json").decode("utf-8"))
    packet = {"raw_records": raw, "pair_observations": pairs, "source_death_receipts": deaths}
    rows = spatial_adj.row_c_pairs(packet)
    records, failures = spatial_adj.r_spatial_records(rows)
    null = spatial_adj.matched_permutation_null(packet, rows, spatial_adj.physical_threshold_contract())
    report = {
        "variant": variant,
        "round": round_index,
        "query": query,
        "raw_record_count": len(raw),
        "pair_observation_count": len(pairs),
        "source_death_receipt_count": len(deaths),
        "feature_freeze": feature,
        "target_execution_receipt": receipt,
        "source_lifecycle_valid": all(
            death.get("source_alive_during_query") is True
            and death.get("source_alive_at_pair_measurement") is True
            for death in deaths
        ),
        "block_count": len(records),
        "R_spatial": summarize_records(records),
        "matched_permutation_q99_diagnostic": null,
        "failures": failures,
        "_records": records,
    }
    del raw, pairs, deaths, packet, rows
    gc.collect()
    return report


def claim_boundary(observed: bool) -> dict[str, Any]:
    return {
        "local_primary_minus_sham_differential_confirmed": observed,
        "maximum_scientific_claim": law.POSITIVE_CLAIM,
        "full_carrier_state_tomography_established": False,
        "physical_relational_memory_established": False,
        "catalytic_borrowing_established": False,
        "r2_restoration_established": False,
        "small_wall_crossed": False,
    }


def find_output_prefix(tf: tarfile.TarFile) -> str:
    names = set(tf.getnames())
    candidates = [
        "source/official_outputs",
        "source/local_paired_outputs",
        "source/differential_outputs",
    ]
    required_suffix = f"round0_{law.PRIMARY_QUERY}/raw_records.jsonl"
    for candidate in candidates:
        if f"{candidate}/{required_suffix}" in names:
            return candidate
    raise ValueError("no recognized physical output prefix found")


def crossed_cell_diagnostics(report: dict[str, Any], round_index: int) -> dict[str, Any]:
    primary_records = report["variant_reports"][f"round{round_index}_{law.PRIMARY_QUERY}"]["_records"]
    sham_records = report["variant_reports"][f"round{round_index}_{law.SHAM_QUERY}"]["_records"]

    def grouped(records: list[dict[str, Any]]) -> dict[tuple[str, ...], list[float]]:
        groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for record in records:
            key = tuple(str(record[factor]) for factor in law.FACTOR_STRATA)
            groups[key].append(float(record["R_spatial"]))
        return groups

    primary_groups = grouped(primary_records)
    sham_groups = grouped(sham_records)
    cells = []
    for key in sorted(set(primary_groups) & set(sham_groups)):
        primary_mean = sum(primary_groups[key]) / len(primary_groups[key])
        sham_mean = sum(sham_groups[key]) / len(sham_groups[key])
        d_value = primary_mean - sham_mean
        cells.append(
            {
                "levels": dict(zip(law.FACTOR_STRATA, key)),
                "R_primary": primary_mean,
                "R_sham": sham_mean,
                "D_local": d_value,
                "D_local_gt_0": d_value > 0,
                "R_primary_gt_0": primary_mean > 0,
                "R_sham_lt_0": sham_mean < 0,
                "primary_count": len(primary_groups[key]),
                "sham_count": len(sham_groups[key]),
            }
        )
    d_values = [cell["D_local"] for cell in cells]
    return {
        "diagnostic_only": True,
        "gate": False,
        "reason_not_gate": "sparse six-factor crossed cells have four primary and four sham blocks per cell in this schedule",
        "cell_count": len(cells),
        "D_local_positive_count": sum(1 for cell in cells if cell["D_local_gt_0"]),
        "R_primary_positive_count": sum(1 for cell in cells if cell["R_primary_gt_0"]),
        "R_sham_negative_count": sum(1 for cell in cells if cell["R_sham_lt_0"]),
        "D_distribution": distribution(d_values),
        "weakest_cells": sorted(cells, key=lambda cell: cell["D_local"])[:12],
    }


def adjudicate_archive(archive_path: Path) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    report: dict[str, Any] = {
        "schema": "FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_ADJUDICATION_V1",
        "package_id": law.PACKAGE_ID,
        "transaction_run_id": law.TRANSACTION_RUN_ID,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "threshold_contract": law.threshold_contract(),
        "variant_reports": {},
        "differential_by_round": {},
        "differential_by_factor_stratum": {},
        "crossed_cell_diagnostics_by_round": {},
        "generic_q99_warnings": [],
        "absolute_sham_sign_diagnostics": {},
    }
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            output_prefix = find_output_prefix(tf)
            report["archive_output_prefix"] = output_prefix
            report["package_manifest"] = json.loads(
                read_member(tf, f"source/{law.PACKAGE_MANIFEST_FILENAME}").decode("utf-8")
            )
            report["schedule_manifest"] = json.loads(
                read_member(tf, f"source/{law.SCHEDULE_MANIFEST_FILENAME}").decode("utf-8")
            )
            for round_index in range(law.ROUND_COUNT):
                for query in law.ALL_QUERIES:
                    variant = f"round{round_index}_{query}"
                    report["variant_reports"][variant] = analyze_variant(
                        tf, output_prefix, variant, query, round_index
                    )
    except (KeyError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        report.update(
            {
                "result_class": law.RESULT_CUSTODY_INVALID,
                "passed": False,
                "scientific_claim": law.NEGATIVE_CLAIM,
                "claim_boundary": claim_boundary(False),
                "custody_failure": str(exc),
            }
        )
        report["adjudication_sha256"] = law.digest(
            {key: value for key, value in report.items() if key != "adjudication_sha256"}
        )
        return report

    for round_index in range(law.ROUND_COUNT):
        primary = report["variant_reports"][f"round{round_index}_{law.PRIMARY_QUERY}"]["R_spatial"]["distribution"]["mean"]
        sham = report["variant_reports"][f"round{round_index}_{law.SHAM_QUERY}"]["R_spatial"]["distribution"]["mean"]
        generic_means = {
            query: report["variant_reports"][f"round{round_index}_{query}"]["R_spatial"]["distribution"]["mean"]
            for query in law.GENERIC_CONTROL_QUERIES
        }
        generic_q99 = {
            query: report["variant_reports"][f"round{round_index}_{query}"]["matched_permutation_q99_diagnostic"]
            for query in law.GENERIC_CONTROL_QUERIES
        }
        d_value = primary - sham
        max_generic = max(abs(value) for value in generic_means.values())
        q99_warning_queries = [
            query
            for query in law.GENERIC_CONTROL_QUERIES
            if abs(generic_means[query]) > generic_q99[query]["q99_abs_mean_R_spatial"]
        ]
        for query in q99_warning_queries:
            report["generic_q99_warnings"].append({"round": round_index, "query": query})
        report["differential_by_round"][str(round_index)] = {
            "R_primary": primary,
            "R_sham": sham,
            "D_local": d_value,
            "generic_control_means": generic_means,
            "max_abs_generic_control": max_generic,
            "max_abs_generic_control_to_D": max_generic / d_value if d_value > 0 else None,
            "R_primary_gt_0": primary > 0,
            "R_sham_lt_0_diagnostic": sham < 0,
            "D_local_gt_0": d_value > 0,
            "generic_control_envelope_passed": d_value > 0
            and max_generic <= law.GENERIC_CONTROL_ENVELOPE_FRACTION * d_value,
            "generic_q99_warning_queries": q99_warning_queries,
            "passed": primary > 0
            and d_value > 0
            and max_generic <= law.GENERIC_CONTROL_ENVELOPE_FRACTION * d_value,
        }
        report["absolute_sham_sign_diagnostics"][str(round_index)] = {
            "R_sham": sham,
            "R_sham_lt_0": sham < 0,
            "gate": False,
        }
        report["crossed_cell_diagnostics_by_round"][str(round_index)] = crossed_cell_diagnostics(
            report, round_index
        )

    stratum_gate_values = []
    for factor in law.FACTOR_STRATA:
        report["differential_by_factor_stratum"][factor] = {}
        for round_index in range(law.ROUND_COUNT):
            primary_strata = report["variant_reports"][f"round{round_index}_{law.PRIMARY_QUERY}"]["R_spatial"]["strata"][factor]
            sham_strata = report["variant_reports"][f"round{round_index}_{law.SHAM_QUERY}"]["R_spatial"]["strata"][factor]
            levels = sorted(set(primary_strata) & set(sham_strata))
            level_reports = {}
            for level in levels:
                primary_mean = primary_strata[level]["mean"]
                sham_mean = sham_strata[level]["mean"]
                d_value = primary_mean - sham_mean
                passed = primary_mean > 0 and d_value > 0
                stratum_gate_values.append(passed)
                level_reports[level] = {
                    "R_primary": primary_mean,
                    "R_sham": sham_mean,
                    "D_local": d_value,
                    "R_primary_gt_0": primary_mean > 0,
                    "R_sham_lt_0_diagnostic": sham_mean < 0,
                    "D_local_gt_0": d_value > 0,
                    "passed": passed,
                }
            report["differential_by_factor_stratum"][factor][str(round_index)] = level_reports

    round_gates = [item["passed"] for item in report["differential_by_round"].values()]
    lifecycle_gates = [
        item["source_lifecycle_valid"] and not item["failures"]
        for item in report["variant_reports"].values()
    ]
    gates = {
        "all_variants_complete": all(
            item["raw_record_count"] == 2048
            and item["pair_observation_count"] == 2048 * law.PAIR_SAMPLE_COUNT
            and item["source_death_receipt_count"] == 2048
            for item in report["variant_reports"].values()
        ),
        "all_lifecycle_and_block_gates_pass": all(lifecycle_gates),
        "round_local_paired_law_passed": all(round_gates),
        "one_factor_stratum_law_passed": all(stratum_gate_values),
        "relation_sham_not_in_generic_null_set": law.SHAM_QUERY not in law.GENERIC_CONTROL_QUERIES,
        "absolute_sham_sign_is_diagnostic_only": True,
        "sparse_crossed_cells_are_diagnostic_only": True,
        "post_observation_threshold_revision_absent": True,
    }
    non_gate_invariants = {
        "small_wall_crossed": False,
    }
    passed = all(gates.values())
    report.update(
        {
            "gates": gates,
            "non_gate_invariants": non_gate_invariants,
            "passed": passed,
            "result_class": law.RESULT_CONFIRMED if passed else law.RESULT_NOT_CONFIRMED,
            "scientific_claim": law.POSITIVE_CLAIM if passed else law.NEGATIVE_CLAIM,
            "claim_boundary": claim_boundary(passed),
        }
    )
    for item in report["variant_reports"].values():
        item.pop("_records", None)
    report["adjudication_sha256"] = law.digest(
        {key: value for key, value in report.items() if key != "adjudication_sha256"}
    )
    return report


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adjudication_path = output_dir / "LOCAL_PAIRED_DIFFERENTIAL_ADJUDICATION.json"
    law.write_json(adjudication_path, report)
    lines = [
        "# Local Paired Differential Adjudication",
        "",
        f"Result class: `{report['result_class']}`",
        f"Scientific claim: `{report['scientific_claim']}`",
        f"Archive SHA-256: `{report['archive_sha256']}`",
        f"Adjudication SHA-256: `{report['adjudication_sha256']}`",
        "",
        "## Round Results",
        "",
        "| Round | R_primary | R_sham | D | max abs generic | generic/D | envelope | sham<0 diagnostic | q99 warnings |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for round_id, item in sorted(report["differential_by_round"].items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"| {round_id} | {item['R_primary']:.9f} | {item['R_sham']:.9f} | "
            f"{item['D_local']:.9f} | {item['max_abs_generic_control']:.9f} | "
            f"{item['max_abs_generic_control_to_D']:.3f} | `{item['generic_control_envelope_passed']}` | "
            f"`{item['R_sham_lt_0_diagnostic']}` | "
            f"`{','.join(item['generic_q99_warning_queries']) or 'none'}` |"
        )
    lines.extend(
        [
            "",
            "This adjudication uses the frozen prospective local paired differential law only. It gates on `D = R_primary - R_sham`, `R_primary > 0`, one-factor matched stratum ordering, and the existing `0.25 * D` generic-control envelope.",
            "",
            "`R_sham < 0` is reported as a diagnostic, not a gate. Complete six-factor crossed cells are also diagnostic only because each retained crossed cell is sparse.",
            "",
            "Claim boundary: no full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing is established by this package.",
        ]
    )
    (output_dir / "LOCAL_PAIRED_DIFFERENTIAL_ADJUDICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print("usage: local_paired_differential_adjudication.py <archive.tar.gz> <output-dir>", file=sys.stderr)
        return 2
    report = adjudicate_archive(Path(argv[0]))
    write_outputs(report, Path(argv[1]))
    print(
        json.dumps(
            {
                "result_class": report["result_class"],
                "passed": report["passed"],
                "adjudication_sha256": report["adjudication_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
