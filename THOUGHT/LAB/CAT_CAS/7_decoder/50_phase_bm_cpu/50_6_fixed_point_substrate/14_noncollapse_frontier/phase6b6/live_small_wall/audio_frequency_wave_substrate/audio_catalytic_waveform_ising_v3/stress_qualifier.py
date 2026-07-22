from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
DEVELOPMENT_SOURCE = PACKAGE_DIR / "development_qualifier.py"
OUTPUT_FILE = PACKAGE_DIR / "STRESS_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "STRESS_REPORT.md"

PUBLIC_SEED = "CATCAS-V3-DEVELOPMENT-STRESS-2026-07-22"
STRESS_CASE_COUNT = 512
COUPLING_VALUES = (-2.0, -1.0, 1.0, 2.0)
FIELD_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_stress_machine")
development = load_module(
    DEVELOPMENT_SOURCE, "catcas_waveform_ising_v3_stress_development"
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def derived_byte(index: int, coordinate: str) -> int:
    return hashlib.sha256(
        f"{PUBLIC_SEED}|{index:04d}|{coordinate}".encode("ascii")
    ).digest()[0]


def generated_problem(index: int) -> tuple[np.ndarray, np.ndarray]:
    coupling = np.zeros((machine.SITE_COUNT, machine.SITE_COUNT), dtype=np.float64)
    edge = 0
    for left in range(machine.SITE_COUNT):
        for right in range(left + 1, machine.SITE_COUNT):
            value = COUPLING_VALUES[
                derived_byte(index, f"J{edge:02d}") % len(COUPLING_VALUES)
            ]
            coupling[left, right] = value
            coupling[right, left] = value
            edge += 1
    field = np.asarray(
        [
            FIELD_VALUES[
                derived_byte(index, f"h{site:02d}") % len(FIELD_VALUES)
            ]
            for site in range(machine.SITE_COUNT)
        ],
        dtype=np.float64,
    )
    return coupling, field


def stress_corpus() -> list[dict[str, Any]]:
    known = development.development_corpus()
    excluded = {record["problem_sha256"] for record in known}
    records: list[dict[str, Any]] = []
    index = 0
    while len(records) < STRESS_CASE_COUNT:
        coupling, field = generated_problem(index)
        identity = development.problem_sha256(coupling, field)
        if identity not in excluded:
            excluded.add(identity)
            records.append(
                {
                    "coupling": coupling,
                    "field": field,
                    "generator_index": index,
                    "label": f"stress_{len(records):03d}",
                    "problem_sha256": identity,
                }
            )
        index += 1
    return records


def execute_record(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    coupling = np.asarray(record["coupling"], dtype=np.float64)
    field = np.asarray(record["field"], dtype=np.float64)
    execution = machine.execute_native_cycle(borrowed, coupling, field)
    boundary = machine.project_boundary(execution, record["label"])
    oracle = development.exact_classification(
        coupling, field, boundary.raw_spins, boundary.valid
    )
    restored = machine.restore_carrier(execution)
    restoration_error = machine.maximum_abs_error(
        restored, machine.as_carrier_bank(borrowed)
    )
    return {
        "accepted": boundary.valid,
        "best_mode_concentration": metric(boundary.best_mode_concentration),
        "coupling_matrix_J": np.asarray(coupling, dtype=np.float64).tolist(),
        "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
        "energy_gap": oracle["energy_gap"],
        "generator_index": record["generator_index"],
        "label": record["label"],
        "minimum_coherence": metric(min(boundary.coherence)),
        "minimum_geometry_coherence": metric(boundary.minimum_geometry_coherence),
        "optimum_count": oracle["optimum_count"],
        "outcome": oracle["outcome"],
        "problem_sha256": record["problem_sha256"],
        "raw_matches_optimum": oracle["raw_matches_optimum"],
        "raw_spins": list(boundary.raw_spins),
        "restoration_max_abs_error": metric(restoration_error),
        "second_mode_gap": metric(boundary.second_mode_gap),
    }


def build_document() -> dict[str, Any]:
    corpus = stress_corpus()
    borrowed = machine.borrowed_carrier()
    records = [execute_record(record, borrowed) for record in corpus]
    unique = [record for record in records if record["optimum_count"] == 1]
    non_unique = [record for record in records if record["optimum_count"] != 1]
    summary = {
        "accepted_correct": sum(record["outcome"] == "ACCEPTED_CORRECT" for record in records),
        "accepted_incorrect": sum(record["outcome"] == "ACCEPTED_INCORRECT" for record in records),
        "case_count": len(records),
        "maximum_restoration_error": metric(
            max(record["restoration_max_abs_error"] for record in records)
        ),
        "minimum_unique_energy_gap": metric(
            min(record["energy_gap"] for record in unique)
        ),
        "minimum_unique_response_coherence": metric(
            min(record["minimum_coherence"] for record in unique)
        ),
        "non_unique_count": len(non_unique),
        "non_unique_rejected": sum(not record["accepted"] for record in non_unique),
        "rejected_correct": sum(record["outcome"] == "REJECTED_CORRECT" for record in records),
        "rejected_incorrect": sum(record["outcome"] == "REJECTED_INCORRECT" for record in records),
        "unique_count": len(unique),
        "unique_raw_correct": sum(record["raw_matches_optimum"] for record in unique),
    }
    stress_pass = bool(
        summary["accepted_incorrect"] == 0
        and summary["rejected_correct"] == 0
        and summary["rejected_incorrect"] == 0
        and summary["unique_raw_correct"] == summary["unique_count"]
        and summary["non_unique_rejected"] == summary["non_unique_count"]
        and summary["maximum_restoration_error"] <= machine.RESTORATION_MAX
    )
    ordered_hash = sha256_bytes(
        canonical_bytes([record["problem_sha256"] for record in corpus])
    )
    return {
        "generator": {
            "coupling_values": list(COUPLING_VALUES),
            "field_values": list(FIELD_VALUES),
            "identity_only_duplicate_and_known_exclusion": True,
            "public_seed": PUBLIC_SEED,
            "requested_case_count": STRESS_CASE_COUNT,
        },
        "machine_fingerprint": machine.machine_fingerprint(),
        "ordered_problem_sha256": ordered_hash,
        "records": records,
        "schema": "catalytic_waveform_ising_v3_stress_v1",
        "stress_pass": stress_pass,
        "summary": summary,
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    summary = document["summary"]
    lines = [
        "# Catalytic Waveform-Ising V3 Deterministic Stress Qualification",
        "",
        f"Public seed: `{PUBLIC_SEED}`",
        f"Ordered problem hash: `{document['ordered_problem_sha256']}`",
        f"Stress pass: `{document['stress_pass']}`",
        "",
        "```text",
        f"cases                 {summary['case_count']}",
        f"unique                {summary['unique_count']}",
        f"unique raw correct    {summary['unique_raw_correct']}",
        f"accepted incorrect    {summary['accepted_incorrect']}",
        f"non-unique rejected   {summary['non_unique_rejected']} / {summary['non_unique_count']}",
        "```",
        "",
        "These cases are development stress data, not unseen evidence. No outcome or",
        "uniqueness filtering was used; only complete problem-identity duplicates and",
        "the prior 115 development identities were excluded.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = build_document()
    if OUTPUT_FILE.read_bytes() != canonical_bytes(document):
        raise ValueError("stress results do not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise ValueError("stress report does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "ordered_problem_sha256": document["ordered_problem_sha256"],
                "stress_pass": document["stress_pass"],
                "summary": document["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["stress_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
