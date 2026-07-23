from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np


sys.dont_write_bytecode = True

PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
MACHINE_SOURCE = PACKAGE_DIR / "dimension_general_machine.py"
V3_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v3"
V3_MACHINE_SOURCE = V3_DIR / "v3_machine.py"
V3_DEVELOPMENT_RESULTS = V3_DIR / "DEVELOPMENT_RESULTS.json"
OUTPUT_FILE = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
REPORT_FILE = PACKAGE_DIR / "DEVELOPMENT_REPORT.md"

PUBLIC_SEED = "CATCAS-V3-SIX-SITE-DEVELOPMENT-2026-07-23"
DEVELOPMENT_CASE_COUNT = 512
COUPLING_VALUES = (-2.0, -1.0, 1.0, 2.0)
FIELD_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def load_module_bytes(source: bytes, path: Path, name: str) -> Any:
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


def load_module(path: Path, name: str) -> Any:
    return load_module_bytes(path.read_bytes(), path, name)


machine = load_module(MACHINE_SOURCE, "catcas_v3_six_development_machine")
v3 = load_module(V3_MACHINE_SOURCE, "catcas_v3_five_preservation_reference")


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


def development_corpus() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    generator_index = 0
    while len(records) < DEVELOPMENT_CASE_COUNT:
        coupling, field = generated_problem(generator_index)
        identity = machine.problem_identity_sha256(coupling, field)
        if identity not in identities:
            identities.add(identity)
            records.append(
                {
                    "coupling": coupling,
                    "field": field,
                    "generator_index": generator_index,
                    "label": f"six_development_{len(records):03d}",
                    "problem_sha256": identity,
                }
            )
        generator_index += 1
    return records


def exact_oracle(coupling: np.ndarray, field: np.ndarray) -> dict[str, Any]:
    rows: list[tuple[float, tuple[int, ...]]] = []
    for state in itertools.product((-1, 1), repeat=machine.SITE_COUNT):
        spins = np.asarray(state, dtype=np.float64)
        energy = float(-0.5 * spins @ coupling @ spins - field @ spins)
        rows.append((energy, tuple(int(value) for value in state)))
    rows.sort(key=lambda item: (item[0], item[1]))
    optimum_energy = rows[0][0]
    optima = tuple(
        state
        for energy, state in rows
        if abs(float(energy) - optimum_energy) <= 1.0e-12
    )
    second_energy = next(
        (
            float(energy)
            for energy, _ in rows
            if float(energy) > optimum_energy + 1.0e-12
        ),
        None,
    )
    return {
        "energy_gap": (
            None if second_energy is None else metric(second_energy - optimum_energy)
        ),
        "optimum_count": len(optima),
        "optimum_energy": metric(optimum_energy),
        "optimum_states": optima,
    }


def response_delta(left: Any, right: Any) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.responses, dtype=np.complex128)
            - np.asarray(right.responses, dtype=np.complex128)
        )
    )


def execute_record(record: dict[str, Any], borrowed: np.ndarray) -> dict[str, Any]:
    coupling = np.asarray(record["coupling"], dtype=np.float64)
    field = np.asarray(record["field"], dtype=np.float64)
    execution = machine.execute_native_cycle(borrowed, coupling, field)
    boundary = machine.project_boundary(execution, record["label"])
    oracle = exact_oracle(coupling, field)
    raw_match = tuple(boundary.raw_spins) in oracle["optimum_states"]
    unique = oracle["optimum_count"] == 1
    if not unique:
        outcome = "NON_UNIQUE_REJECTED" if not boundary.valid else "NON_UNIQUE_ACCEPTED"
    elif boundary.valid and raw_match:
        outcome = "UNIQUE_ACCEPTED_CORRECT"
    elif boundary.valid:
        outcome = "UNIQUE_ACCEPTED_INCORRECT"
    elif raw_match:
        outcome = "UNIQUE_REJECTED_CORRECT"
    else:
        outcome = "UNIQUE_REJECTED_INCORRECT"

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
        "best_mode_index": boundary.best_mode_index,
        "coupling_matrix_J": coupling.tolist(),
        "displacement_l2": metric(execution.displacement_l2),
        "energy_gap": oracle["energy_gap"],
        "field_vector_h": field.tolist(),
        "generator_index": record["generator_index"],
        "label": record["label"],
        "minimum_response_coherence": metric(min(boundary.coherence)),
        "minimum_geometry_coherence": metric(boundary.minimum_geometry_coherence),
        "operator_count": int(execution.relation_history.shape[0]) + 1,
        "optimum_count": oracle["optimum_count"],
        "outcome": outcome,
        "problem_sha256": record["problem_sha256"],
        "raw_matches_optimum": raw_match,
        "raw_spins": list(boundary.raw_spins),
        "response_reuse_delta_l2": metric(response_delta(boundary, reuse_boundary)),
        "restoration_max_abs_error": metric(restoration_error),
        "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        "reuse_result_reproduced": bool(
            boundary.valid == reuse_boundary.valid
            and (
                not boundary.valid
                or boundary.raw_spins == reuse_boundary.raw_spins
            )
        ),
        "second_mode_gap": metric(boundary.second_mode_gap),
    }


def five_site_variant() -> Any:
    source = MACHINE_SOURCE.read_bytes()
    anchor = b"TARGET_SITE_COUNT = 6"
    if source.count(anchor) != 1:
        raise RuntimeError("dimension target anchor drift")
    return load_module_bytes(
        source.replace(anchor, b"TARGET_SITE_COUNT = 5"),
        MACHINE_SOURCE,
        "catcas_v3_dimension_general_five_variant",
    )


def five_site_preservation() -> dict[str, Any]:
    variant = five_site_variant()
    saved = json.loads(V3_DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    exact_primitives = bool(
        np.array_equal(v3.borrowed_carrier(), variant.borrowed_carrier())
        and np.array_equal(v3.PHASE_MODES, variant.PHASE_MODES)
        and np.array_equal(v3.canonical_geometry(), variant.canonical_geometry())
        and v3.SAMPLE_COUNT == variant.SAMPLE_COUNT == 256
        and v3.MODE_COUNT == variant.MODE_COUNT == 32
        and v3.DEFAULT_LAW.__dict__ == variant.DEFAULT_LAW.__dict__
    )
    borrowed = v3.borrowed_carrier()
    for record in saved["records"]:
        coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(record["field_vector_h"], dtype=np.float64)
        label = str(record["label"])
        reference = v3.execute_native_cycle(borrowed, coupling, field)
        candidate = variant.execute_native_cycle(borrowed, coupling, field)
        reference_boundary = v3.project_boundary(reference, label)
        candidate_boundary = variant.project_boundary(candidate, label)
        reference_restored = v3.restore_carrier(reference)
        candidate_restored = variant.restore_carrier(candidate)
        if not (
            np.array_equal(reference.displaced, candidate.displaced)
            and reference_boundary.document() == candidate_boundary.document()
            and np.array_equal(reference_restored, candidate_restored)
        ):
            mismatches.append(label)
    return {
        "case_count": len(saved["records"]),
        "exact_primitives": exact_primitives,
        "mismatch_labels": mismatches,
        "pass": exact_primitives and not mismatches and len(saved["records"]) == 115,
        "reference_machine_fingerprint": str(saved["machine_fingerprint"]),
        "variant_phase_denominator": int(variant.RELATION_PHASE_DENOMINATOR),
        "variant_sample_count": int(variant.SAMPLE_COUNT),
    }


def build_document() -> dict[str, Any]:
    preservation = five_site_preservation()
    corpus = development_corpus()
    borrowed = machine.borrowed_carrier()
    records = [execute_record(record, borrowed) for record in corpus]
    unique = [record for record in records if record["optimum_count"] == 1]
    non_unique = [record for record in records if record["optimum_count"] != 1]
    summary = {
        "accepted_incorrect": sum(
            record["outcome"] == "UNIQUE_ACCEPTED_INCORRECT" for record in records
        ),
        "case_count": len(records),
        "maximum_restoration_error": metric(
            max(record["restoration_max_abs_error"] for record in records)
        ),
        "maximum_reuse_response_delta_l2": metric(
            max(record["response_reuse_delta_l2"] for record in records)
        ),
        "maximum_reuse_restoration_error": metric(
            max(record["reuse_restoration_max_abs_error"] for record in records)
        ),
        "minimum_unique_energy_gap": metric(
            min(record["energy_gap"] for record in unique)
        ),
        "minimum_unique_response_coherence": metric(
            min(record["minimum_response_coherence"] for record in unique)
        ),
        "non_unique_count": len(non_unique),
        "non_unique_rejected": sum(not record["accepted"] for record in non_unique),
        "rejected_unique": sum(
            record["outcome"].startswith("UNIQUE_REJECTED") for record in records
        ),
        "reuse_pass_count": sum(record["reuse_result_reproduced"] for record in records),
        "unique_count": len(unique),
        "unique_raw_correct": sum(record["raw_matches_optimum"] for record in unique),
    }
    development_pass = bool(
        preservation["pass"]
        and summary["accepted_incorrect"] == 0
        and summary["rejected_unique"] == 0
        and summary["unique_raw_correct"] == summary["unique_count"]
        and summary["non_unique_rejected"] == summary["non_unique_count"]
        and summary["reuse_pass_count"] == summary["case_count"]
        and summary["maximum_restoration_error"] <= machine.RESTORATION_MAX
        and summary["maximum_reuse_restoration_error"] <= machine.RESTORATION_MAX
        and summary["maximum_reuse_response_delta_l2"] <= machine.REUSE_RESPONSE_MAX
    )
    ordered_hash = sha256_bytes(
        canonical_bytes([record["problem_sha256"] for record in records])
    )
    return {
        "claim_ceiling": machine.CLAIM_CEILING,
        "development_pass": development_pass,
        "five_site_preservation": preservation,
        "generator": {
            "case_count": DEVELOPMENT_CASE_COUNT,
            "coupling_values": list(COUPLING_VALUES),
            "field_values": list(FIELD_VALUES),
            "public_seed": PUBLIC_SEED,
        },
        "machine_contract": machine.machine_contract(),
        "machine_fingerprint": machine.machine_fingerprint(),
        "ordered_problem_sha256": ordered_hash,
        "records": records,
        "schema": "catalytic_waveform_ising_v3_six_site_development_v1",
        "summary": summary,
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    summary = document["summary"]
    preservation = document["five_site_preservation"]
    lines = [
        "# Catalytic Waveform-Ising V3 Six-Site Development",
        "",
        f"Machine fingerprint: `{document['machine_fingerprint']}`",
        f"Development pass: `{document['development_pass']}`",
        f"Five-site exact preservation: `{preservation['pass']}`",
        "",
        "```text",
        f"five-site exact cases       {preservation['case_count']}",
        f"six-site cases              {summary['case_count']}",
        f"six-site unique correct     {summary['unique_raw_correct']} / {summary['unique_count']}",
        f"six-site non-unique rejected {summary['non_unique_rejected']} / {summary['non_unique_count']}",
        f"accepted incorrect          {summary['accepted_incorrect']}",
        f"rejected unique             {summary['rejected_unique']}",
        f"reuse passes                {summary['reuse_pass_count']} / {summary['case_count']}",
        "```",
        "",
        "Oracle information is development-only. The native machine contains no oracle,",
        "energy calculation, decoded-spin recurrence, or candidate selection. This",
        "development evidence is excluded from the prospective batch.",
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
        raise ValueError("six-site development results do not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise ValueError("six-site development report does not reproduce")
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
                "five_site_preservation": document["five_site_preservation"],
                "machine_fingerprint": document["machine_fingerprint"],
                "ordered_problem_sha256": document["ordered_problem_sha256"],
                "summary": document["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["development_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
