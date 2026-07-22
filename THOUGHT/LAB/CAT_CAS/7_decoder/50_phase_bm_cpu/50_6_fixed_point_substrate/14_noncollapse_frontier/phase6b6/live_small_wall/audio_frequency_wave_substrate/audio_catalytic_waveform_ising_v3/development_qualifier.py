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
SUBSTRATE_DIR = PACKAGE_DIR.parent
MACHINE_SOURCE = PACKAGE_DIR / "v3_machine.py"
STABILITY_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2_stability_gate"
STABILITY_DEVELOPMENT_SOURCE = STABILITY_DIR / "development_qualifier.py"
STABILITY_BATCH = STABILITY_DIR / "STABILITY_BATCH_CUSTODY.json"
OUTPUT_FILE = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "DEVELOPMENT_REPORT.md"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_machine")
development_reference = load_module(
    STABILITY_DEVELOPMENT_SOURCE, "catcas_waveform_ising_v3_development_reference"
)
v2 = development_reference.v2


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def problem_sha256(coupling: np.ndarray, field: np.ndarray) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "coupling_matrix_J": np.asarray(coupling, dtype=np.float64).tolist(),
                "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
            }
        )
    )


def development_corpus() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in development_reference.known_instances():
        coupling = np.asarray(record["coupling"], dtype=np.float64)
        field = np.asarray(record["field"], dtype=np.float64)
        records.append(
            {
                "coupling": coupling,
                "field": field,
                "label": str(record["label"]),
                "problem_sha256": problem_sha256(coupling, field),
                "source_group": str(record["source_group"]),
            }
        )
    stability_batch = json.loads(STABILITY_BATCH.read_text(encoding="utf-8"))
    for record in stability_batch["ordered_instances"]:
        coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(record["field_vector_h"], dtype=np.float64)
        records.append(
            {
                "coupling": coupling,
                "field": field,
                "label": f"stability_batch_{int(record['index']):02d}",
                "problem_sha256": str(record["problem_sha256"]),
                "source_group": "stability_batch",
            }
        )
    identities = [record["problem_sha256"] for record in records]
    if len(records) != 115 or len(set(identities)) != 115:
        raise RuntimeError("V3 development corpus must contain 115 unique instances")
    return records


def exact_classification(
    coupling: np.ndarray,
    field: np.ndarray,
    raw_spins: tuple[int, ...],
    accepted: bool,
) -> dict[str, Any]:
    rows = v2.exact_oracle(coupling, field)
    optimum_energy = float(rows[0][0])
    optimum_states = tuple(
        state for energy, state in rows if abs(float(energy) - optimum_energy) <= 1.0e-12
    )
    raw_match = raw_spins in optimum_states
    unique = len(optimum_states) == 1
    if not unique:
        outcome = "NON_UNIQUE"
    elif accepted and raw_match:
        outcome = "ACCEPTED_CORRECT"
    elif accepted and not raw_match:
        outcome = "ACCEPTED_INCORRECT"
    elif not accepted and raw_match:
        outcome = "REJECTED_CORRECT"
    else:
        outcome = "REJECTED_INCORRECT"
    second_energy = next(
        (float(energy) for energy, _ in rows if float(energy) > optimum_energy + 1.0e-12),
        None,
    )
    return {
        "energy_gap": None if second_energy is None else metric(second_energy - optimum_energy),
        "optimum_count": len(optimum_states),
        "optimum_energy": metric(optimum_energy),
        "optimum_states": [list(state) for state in optimum_states],
        "outcome": outcome,
        "raw_matches_optimum": raw_match,
        "unique": unique,
    }


def response_delta(left: Any, right: Any) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.responses, dtype=np.complex128)
            - np.asarray(right.responses, dtype=np.complex128)
        )
    )


def execute_record(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    coupling = record["coupling"]
    field = record["field"]
    execution = machine.execute_native_cycle(borrowed, coupling, field)
    boundary = machine.project_boundary(execution, record["label"])
    oracle = exact_classification(
        coupling, field, boundary.raw_spins, boundary.valid
    )
    restored = machine.restore_carrier(execution)
    restoration_error = machine.maximum_abs_error(
        restored, machine.as_carrier_bank(borrowed)
    )
    reuse = machine.execute_native_cycle(restored, coupling, field)
    reuse_boundary = machine.project_boundary(reuse, record["label"] + "_reuse")
    reuse_restored = machine.restore_carrier(reuse)
    reuse_restoration_error = machine.maximum_abs_error(
        reuse_restored, machine.as_carrier_bank(borrowed)
    )
    return {
        "accepted": boundary.valid,
        "best_mode_concentration": metric(boundary.best_mode_concentration),
        "best_mode_index": boundary.best_mode_index,
        "coupling_matrix_J": np.asarray(coupling, dtype=np.float64).tolist(),
        "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
        "displacement_l2": metric(execution.displacement_l2),
        "energy_gap": oracle["energy_gap"],
        "label": record["label"],
        "minimum_coherence": metric(min(boundary.coherence)),
        "minimum_geometry_coherence": metric(boundary.minimum_geometry_coherence),
        "operator_count": int(execution.relation_history.shape[0]) + 1,
        "optimum_count": oracle["optimum_count"],
        "optimum_states": oracle["optimum_states"],
        "outcome": oracle["outcome"],
        "problem_sha256": record["problem_sha256"],
        "raw_matches_optimum": oracle["raw_matches_optimum"],
        "raw_spins": list(boundary.raw_spins),
        "response_reuse_delta_l2": metric(response_delta(boundary, reuse_boundary)),
        "restoration_max_abs_error": metric(restoration_error),
        "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        "second_mode_gap": metric(boundary.second_mode_gap),
        "source_group": record["source_group"],
        "accepted_result_reproduced_on_reuse": bool(
            boundary.valid == reuse_boundary.valid
            and (
                not boundary.valid
                or boundary.raw_spins == reuse_boundary.raw_spins
            )
        ),
    }


def group_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique = [record for record in records if record["optimum_count"] == 1]
    non_unique = [record for record in records if record["optimum_count"] != 1]
    return {
        "accepted_correct": sum(record["outcome"] == "ACCEPTED_CORRECT" for record in records),
        "accepted_incorrect": sum(record["outcome"] == "ACCEPTED_INCORRECT" for record in records),
        "case_count": len(records),
        "non_unique_count": len(non_unique),
        "non_unique_rejected": sum(not record["accepted"] for record in non_unique),
        "rejected_correct": sum(record["outcome"] == "REJECTED_CORRECT" for record in records),
        "rejected_incorrect": sum(record["outcome"] == "REJECTED_INCORRECT" for record in records),
        "unique_count": len(unique),
        "unique_raw_correct": sum(record["raw_matches_optimum"] for record in unique),
    }


def build_document() -> dict[str, Any]:
    corpus = development_corpus()
    borrowed = machine.borrowed_carrier()
    records = [execute_record(record, borrowed) for record in corpus]
    groups = {
        group: group_summary(
            [record for record in records if record["source_group"] == group]
        )
        for group in sorted({record["source_group"] for record in records})
    }
    summary = group_summary(records)
    summary.update(
        {
            "all_restoration_pass": all(
                record["restoration_max_abs_error"] <= machine.RESTORATION_MAX
                for record in records
            ),
            "all_reuse_pass": all(
                record["accepted_result_reproduced_on_reuse"]
                and record["response_reuse_delta_l2"]
                <= machine.REUSE_RESPONSE_MAX
                and record["reuse_restoration_max_abs_error"] <= machine.RESTORATION_MAX
                for record in records
            ),
            "known_failure_count_before_v3": 10,
            "known_failures_remaining": sum(
                not record["raw_matches_optimum"]
                for record in records
                if record["optimum_count"] == 1
            ),
            "maximum_restoration_error": metric(
                max(record["restoration_max_abs_error"] for record in records)
            ),
            "maximum_reuse_restoration_error": metric(
                max(record["reuse_restoration_max_abs_error"] for record in records)
            ),
            "maximum_reuse_response_delta_l2": metric(
                max(record["response_reuse_delta_l2"] for record in records)
            ),
            "minimum_unique_mode_gap": metric(
                min(
                    record["second_mode_gap"]
                    for record in records
                    if record["optimum_count"] == 1
                )
            ),
            "minimum_unique_response_coherence": metric(
                min(
                    record["minimum_coherence"]
                    for record in records
                    if record["optimum_count"] == 1
                )
            ),
        }
    )
    pass_development = bool(
        summary["known_failures_remaining"] == 0
        and summary["accepted_incorrect"] == 0
        and summary["rejected_correct"] == 0
        and summary["non_unique_rejected"] == summary["non_unique_count"]
        and summary["all_restoration_pass"]
        and summary["all_reuse_pass"]
    )
    return {
        "claim_ceiling": machine.CLAIM_CEILING,
        "development_pass": pass_development,
        "groups": groups,
        "machine_contract": machine.machine_contract(),
        "machine_fingerprint": machine.machine_fingerprint(),
        "records": records,
        "schema": "catalytic_waveform_ising_v3_development_v1",
        "summary": summary,
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    summary = document["summary"]
    lines = [
        "# Catalytic Waveform-Ising V3 Development",
        "",
        f"Machine fingerprint: `{document['machine_fingerprint']}`",
        f"Development pass: `{document['development_pass']}`",
        "",
        "```text",
        f"known cases                  {summary['case_count']}",
        f"unique cases                 {summary['unique_count']}",
        f"unique raw correct           {summary['unique_raw_correct']}",
        f"known failures remaining     {summary['known_failures_remaining']}",
        f"accepted incorrect           {summary['accepted_incorrect']}",
        f"non-unique rejected          {summary['non_unique_rejected']} / {summary['non_unique_count']}",
        f"restoration                  {summary['all_restoration_pass']}",
        f"reuse                        {summary['all_reuse_pass']}",
        "```",
        "",
        "All oracle information is development-only and is absent from the V3 native",
        "machine. The machine performs a bounded 32-mode recursive spectral phase",
        "evaluation and makes no scaling or computational-advantage claim.",
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
        raise ValueError("development results do not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise ValueError("development report does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "development_pass": document["development_pass"],
                "machine_fingerprint": document["machine_fingerprint"],
                "summary": document["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["development_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
