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
STRESS_SOURCE = PACKAGE_DIR / "stress_qualifier.py"
CONTROL_RESULTS = PACKAGE_DIR / "CONTROL_RESULTS.json"
DEVELOPMENT_RESULTS = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
STRESS_RESULTS = PACKAGE_DIR / "STRESS_RESULTS.json"
CONTRACT_FILE = PACKAGE_DIR / "V3_EXPERIMENT_CONTRACT.md"
FREEZE_FILE = PACKAGE_DIR / "V3_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V3_BATCH_CUSTODY.json"

STARTING_REMOTE_HEAD = "8c44761ba48736e20786256ca3441ea99c36004b"
PUBLIC_BATCH_SEED = "CATCAS-V3-PROSPECTIVE-BATCH-2026-07-22"
BATCH_SIZE = 256
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


machine = load_module(MACHINE_SOURCE, "catcas_waveform_ising_v3_freeze_machine")
development = load_module(
    DEVELOPMENT_SOURCE, "catcas_waveform_ising_v3_freeze_development"
)
stress = load_module(STRESS_SOURCE, "catcas_waveform_ising_v3_freeze_stress")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_byte(index: int, coordinate: str) -> int:
    return hashlib.sha256(
        f"{PUBLIC_BATCH_SEED}|{index:04d}|{coordinate}".encode("ascii")
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


def excluded_development_identities() -> set[str]:
    identities = {
        record["problem_sha256"] for record in development.development_corpus()
    }
    identities.update(record["problem_sha256"] for record in stress.stress_corpus())
    if len(identities) != 627:
        raise RuntimeError("complete 627-case development identity set required")
    return identities


def batch_document() -> dict[str, Any]:
    excluded = excluded_development_identities()
    records: list[dict[str, Any]] = []
    generator_index = 0
    while len(records) < BATCH_SIZE:
        coupling, field = generated_problem(generator_index)
        identity = development.problem_sha256(coupling, field)
        if identity not in excluded:
            excluded.add(identity)
            records.append(
                {
                    "coupling_matrix_J": coupling.tolist(),
                    "field_vector_h": field.tolist(),
                    "generator_index": generator_index,
                    "index": len(records),
                    "problem_sha256": identity,
                }
            )
        generator_index += 1
    ordered_hash = sha256_bytes(
        canonical_bytes([record["problem_sha256"] for record in records])
    )
    return {
        "batch_size": BATCH_SIZE,
        "coupling_values": list(COUPLING_VALUES),
        "development_identity_count_excluded": 627,
        "field_values": list(FIELD_VALUES),
        "generation_rule": (
            "SHA256(public_seed|generator_index|coordinate), modulo frozen value list; "
            "skip only complete problem-identity duplicates or development identities"
        ),
        "no_difficulty_or_outcome_filtering": True,
        "ordered_batch_sha256": ordered_hash,
        "ordered_instances": records,
        "public_seed": PUBLIC_BATCH_SEED,
        "schema": "catalytic_waveform_ising_v3_batch_custody_v1",
    }


def frozen_promotion_criterion() -> dict[str, Any]:
    return {
        "accepted_incorrect_maximum": 0,
        "batch_size_required": BATCH_SIZE,
        "minimum_unique_optimum_cases": 160,
        "native_no_smuggle_must_pass": True,
        "non_unique_accepted_maximum": 0,
        "oracle_calls_before_preoracle_seal": 0,
        "rejected_unique_correct_maximum": 0,
        "restoration_all_cases": True,
        "reuse_all_cases": True,
        "strict_controls_all_cases": True,
        "unique_raw_correct_rate_minimum": 1.0,
        "uninterpretable_maximum": 0,
    }


def load_qualified_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    development_result = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    stress_result = json.loads(STRESS_RESULTS.read_text(encoding="utf-8"))
    controls = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    if not development_result["development_pass"]:
        raise RuntimeError("115-case development qualification did not pass")
    if not stress_result["stress_pass"]:
        raise RuntimeError("512-case development stress qualification did not pass")
    if not controls["overall_pass"]:
        raise RuntimeError("V3 controls did not pass")
    fingerprints = {
        development_result["machine_fingerprint"],
        stress_result["machine_fingerprint"],
        controls["machine_fingerprint"],
        machine.machine_fingerprint(),
    }
    if len(fingerprints) != 1:
        raise RuntimeError("qualified evidence does not bind one machine")
    return development_result, stress_result, controls


def freeze_document(batch: dict[str, Any]) -> dict[str, Any]:
    development_result, stress_result, controls = load_qualified_inputs()
    return {
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "batch_size": batch["batch_size"],
        "claim_ceiling": machine.CLAIM_CEILING,
        "control_results_sha256": sha256_file(CONTROL_RESULTS),
        "development_results_sha256": sha256_file(DEVELOPMENT_RESULTS),
        "development_summary": development_result["summary"],
        "experiment_contract_bytes": CONTRACT_FILE.stat().st_size,
        "experiment_contract_sha256": sha256_file(CONTRACT_FILE),
        "freeze_before_waveform_execution_or_oracle": True,
        "machine_contract": machine.machine_contract(),
        "machine_fingerprint": machine.machine_fingerprint(),
        "machine_source_bytes": MACHINE_SOURCE.stat().st_size,
        "machine_source_sha256": sha256_file(MACHINE_SOURCE),
        "predecessor_starting_remote_head": STARTING_REMOTE_HEAD,
        "promotion_criterion": frozen_promotion_criterion(),
        "schema": "catalytic_waveform_ising_v3_freeze_v1",
        "stress_results_sha256": sha256_file(STRESS_RESULTS),
        "stress_summary": stress_result["summary"],
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    batch = batch_document()
    freeze = freeze_document(batch)
    write_atomic(BATCH_FILE, canonical_bytes(batch))
    write_atomic(FREEZE_FILE, canonical_bytes(freeze))
    return batch, freeze


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    batch = batch_document()
    freeze = freeze_document(batch)
    if BATCH_FILE.read_bytes() != canonical_bytes(batch):
        raise ValueError("V3 batch custody does not reproduce")
    if FREEZE_FILE.read_bytes() != canonical_bytes(freeze):
        raise ValueError("V3 freeze does not reproduce")
    return batch, freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    batch, freeze = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "batch_ordered_sha256": batch["ordered_batch_sha256"],
                "batch_size": batch["batch_size"],
                "machine_fingerprint": freeze["machine_fingerprint"],
                "status": "V3_FROZEN_BEFORE_PROSPECTIVE_EXECUTION_OR_ORACLE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
