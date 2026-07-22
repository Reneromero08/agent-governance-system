from __future__ import annotations

import gc
import hashlib
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import primary_minus_sham_differential_public as law


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


def analyze_variant(tf: tarfile.TarFile, variant: str, query: str, round_index: int) -> dict[str, Any]:
    prefix = f"source/official_outputs/{variant}"
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
            death.get("source_alive_during_query") is True and death.get("source_alive_at_pair_measurement") is True
            for death in deaths
        ),
        "block_count": len(records),
        "R_spatial": summarize_records(records),
        "matched_permutation_q99_diagnostic": null,
        "failures": failures,
    }
    del raw, pairs, deaths, packet, rows
    gc.collect()
    return report


def claim_boundary(observed: bool) -> dict[str, Any]:
    return {
        "primary_minus_sham_relation_coordinate_confirmed": observed,
        "maximum_scientific_claim": law.POSITIVE_CLAIM,
        "full_carrier_state_tomography_established": False,
        "physical_relational_memory_established": False,
        "catalytic_borrowing_established": False,
        "r2_restoration_established": False,
        "small_wall_crossed": False,
    }


def adjudicate_archive(archive_path: Path) -> dict[str, Any]:
    archive_bytes = archive_path.read_bytes()
    report: dict[str, Any] = {
        "schema": "FAMILY10H_PRIMARY_MINUS_SHAM_DIFFERENTIAL_ADJUDICATION_V1",
        "package_id": law.PACKAGE_ID,
        "transaction_run_id": law.TRANSACTION_RUN_ID,
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "threshold_contract": law.threshold_contract(),
        "variant_reports": {},
        "differential_by_round": {},
        "differential_by_factor_stratum": {},
        "generic_q99_warnings": [],
    }
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            report["package_manifest"] = json.loads(
                read_member(tf, f"source/{law.PACKAGE_MANIFEST_FILENAME}").decode("utf-8")
            )
            report["schedule_manifest"] = json.loads(
                read_member(tf, f"source/{law.SCHEDULE_MANIFEST_FILENAME}").decode("utf-8")
            )
            report["target_run_summary"] = json.loads(read_member(tf, "source/OFFICIAL_TARGET_RUN_SUMMARY.json").decode("utf-8"))
            for round_index in range(law.ROUND_COUNT):
                for query in law.ALL_QUERIES:
                    variant = f"round{round_index}_{query}"
                    report["variant_reports"][variant] = analyze_variant(tf, variant, query, round_index)
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
        report["adjudication_sha256"] = law.digest({k: v for k, v in report.items() if k != "adjudication_sha256"})
        return report

    for round_index in range(law.ROUND_COUNT):
        primary = report["variant_reports"][f"round{round_index}_{law.PRIMARY_QUERY}"]["R_spatial"]["distribution"]["mean"]
        anti = report["variant_reports"][f"round{round_index}_{law.ANTI_RELATION_QUERY}"]["R_spatial"]["distribution"]["mean"]
        generic_means = {
            query: report["variant_reports"][f"round{round_index}_{query}"]["R_spatial"]["distribution"]["mean"]
            for query in law.GENERIC_CONTROL_QUERIES
        }
        generic_q99 = {
            query: report["variant_reports"][f"round{round_index}_{query}"]["matched_permutation_q99_diagnostic"]
            for query in law.GENERIC_CONTROL_QUERIES
        }
        d_value = primary - anti
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
            "R_anti": anti,
            "D_primary_minus_sham": d_value,
            "generic_control_means": generic_means,
            "max_abs_generic_control": max_generic,
            "max_abs_generic_control_to_D": max_generic / d_value if d_value > 0 else None,
            "R_primary_gt_0": primary > 0,
            "R_anti_lt_0": anti < 0,
            "D_primary_minus_sham_gt_0": d_value > 0,
            "generic_control_envelope_passed": d_value > 0
            and max_generic <= law.GENERIC_CONTROL_ENVELOPE_FRACTION * d_value,
            "generic_q99_warning_queries": q99_warning_queries,
        }

    stratum_gate_values = []
    for factor in law.FACTOR_STRATA:
        report["differential_by_factor_stratum"][factor] = {}
        for round_index in range(law.ROUND_COUNT):
            primary_strata = report["variant_reports"][f"round{round_index}_{law.PRIMARY_QUERY}"]["R_spatial"]["strata"][factor]
            anti_strata = report["variant_reports"][f"round{round_index}_{law.ANTI_RELATION_QUERY}"]["R_spatial"]["strata"][factor]
            levels = sorted(set(primary_strata) & set(anti_strata))
            level_reports = {}
            for level in levels:
                primary_mean = primary_strata[level]["mean"]
                anti_mean = anti_strata[level]["mean"]
                d_value = primary_mean - anti_mean
                passed = primary_mean > 0 and anti_mean < 0 and d_value > 0
                stratum_gate_values.append(passed)
                level_reports[level] = {
                    "R_primary": primary_mean,
                    "R_anti": anti_mean,
                    "D_primary_minus_sham": d_value,
                    "R_primary_gt_0": primary_mean > 0,
                    "R_anti_lt_0": anti_mean < 0,
                    "D_primary_minus_sham_gt_0": d_value > 0,
                    "passed": passed,
                }
            report["differential_by_factor_stratum"][factor][str(round_index)] = level_reports

    round_gates = [
        item["R_primary_gt_0"]
        and item["R_anti_lt_0"]
        and item["D_primary_minus_sham_gt_0"]
        and item["generic_control_envelope_passed"]
        for item in report["differential_by_round"].values()
    ]
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
        "round_coordinate_law_passed": all(round_gates),
        "factor_stratum_law_passed": all(stratum_gate_values),
        "relation_sham_not_in_generic_null_set": law.ANTI_RELATION_QUERY not in law.GENERIC_CONTROL_QUERIES,
        "post_observation_threshold_revision": False,
        "small_wall_crossed": False,
    }
    passed = all(gates.values())
    report.update(
        {
            "gates": gates,
            "passed": passed,
            "result_class": law.RESULT_CONFIRMED if passed else law.RESULT_NOT_CONFIRMED,
            "scientific_claim": law.POSITIVE_CLAIM if passed else law.NEGATIVE_CLAIM,
            "claim_boundary": claim_boundary(passed),
        }
    )
    report["adjudication_sha256"] = law.digest({k: v for k, v in report.items() if k != "adjudication_sha256"})
    return report


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adjudication_path = output_dir / "PRIMARY_MINUS_SHAM_DIFFERENTIAL_ADJUDICATION.json"
    law.write_json(adjudication_path, report)
    lines = [
        "# Primary-Minus-Sham Differential Official Adjudication",
        "",
        f"Result class: `{report['result_class']}`",
        f"Scientific claim: `{report['scientific_claim']}`",
        f"Archive SHA-256: `{report['archive_sha256']}`",
        f"Adjudication SHA-256: `{report['adjudication_sha256']}`",
        "",
        "## Round Results",
        "",
        "| Round | R_primary | R_anti | D | max abs generic | generic/D | envelope | q99 warnings |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for round_id, item in sorted(report["differential_by_round"].items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"| {round_id} | {item['R_primary']:.9f} | {item['R_anti']:.9f} | "
            f"{item['D_primary_minus_sham']:.9f} | {item['max_abs_generic_control']:.9f} | "
            f"{item['max_abs_generic_control_to_D']:.3f} | `{item['generic_control_envelope_passed']}` | "
            f"`{','.join(item['generic_q99_warning_queries']) or 'none'}` |"
        )
    lines.extend(
        [
            "",
            "This adjudication uses the frozen prospective primary-minus-sham law only. Generic matched-permutation q99 is reported as a diagnostic and is not the generic-control gate. Relation-sham is the anti-relation comparator, not a null control.",
            "",
            "Claim boundary: no full carrier-state tomography, physical relational memory, catalytic borrowing, R2 restoration, or Small Wall crossing is established by this package.",
        ]
    )
    (output_dir / "PRIMARY_MINUS_SHAM_DIFFERENTIAL_ADJUDICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print("usage: primary_minus_sham_differential_adjudication.py <archive.tar.gz> <output-dir>", file=sys.stderr)
        return 2
    report = adjudicate_archive(Path(argv[0]))
    write_outputs(report, Path(argv[1]))
    print(json.dumps({"result_class": report["result_class"], "passed": report["passed"], "adjudication_sha256": report["adjudication_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
