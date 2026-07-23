from __future__ import annotations

import argparse
import hashlib
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
CONTROL_RESULTS = PACKAGE_DIR / "CONTROL_RESULTS.json"
DEVELOPMENT_RESULTS = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
CONTRACT_FILE = PACKAGE_DIR / "SIX_SITE_EXPERIMENT_CONTRACT.md"
FREEZE_FILE = PACKAGE_DIR / "SIX_SITE_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "SIX_SITE_BATCH_CUSTODY.json"
FIVE_SITE_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v3"
FIVE_SITE_SOURCE = FIVE_SITE_DIR / "v3_machine.py"
FIVE_SITE_MANIFEST = FIVE_SITE_DIR / "V3_FINAL_MANIFEST.json"

EXECUTION_SOURCE_NAMES = (
    "control_qualifier.py",
    "development_qualifier.py",
    "dimension_general_machine.py",
    "finalize_package.py",
    "freeze_builder.py",
    "independent_verifier.py",
    "oracle_adjudicator.py",
    "preoracle_runner.py",
    "qualify_package.py",
    "resource_accounting.py",
)
TRANSITIVE_DEPENDENCY_PATHS = {
    "audio_catalytic_waveform_ising_v3/V3_FINAL_MANIFEST.json": FIVE_SITE_MANIFEST,
    "audio_catalytic_waveform_ising_v3/v3_machine.py": FIVE_SITE_SOURCE,
}
SOURCE_EXECUTION_CONTRACT = {
    "bytecode_cache_inputs_forbidden_under_package": True,
    "bytecode_cache_writes_disabled": True,
    "compile_dont_inherit": True,
    "compile_optimization": 0,
    "independent_compiled_bytecode_policy_required": True,
    "local_module_loader": "compile_exact_source_bytes",
    "selection_semantics_policy_required": True,
    "semantic_mutation_fixture_count": 16,
    "sole_selection_capable_region": "project_boundary:body",
    "whole_module_ast_policy_required": True,
}

STARTING_REMOTE_HEAD = "12eb88bb9131d956ea7712e7620b97f9802f93bb"
FIVE_SITE_MACHINE_FINGERPRINT = (
    "1bb3d9c8677c9f9677e5c4d650d27db690c26490f7001401d758217207ba2025"
)
FIVE_SITE_SOURCE_SHA256 = (
    "fb12a0a7c54f4b1fac26b10a2041bfe7205c26d8136b2395c75a6b1d7c99694a"
)
FIVE_SITE_MANIFEST_SHA256 = (
    "2e3d4efb7927a4217e40dd8eabbbb0cc317975485e3da6194e8df83621255e4e"
)
PUBLIC_BATCH_SEED = "CATCAS-V3-SIX-SITE-PROSPECTIVE-BATCH-2026-07-23"
BATCH_SIZE = 256
COUPLING_VALUES = (-2.0, -1.0, 1.0, 2.0)
FIELD_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def load_module(path: Path, name: str) -> Any:
    source = path.read_bytes()
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


machine = load_module(
    MACHINE_SOURCE, "catcas_waveform_ising_v3_six_site_freeze_machine"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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
            FIELD_VALUES[derived_byte(index, f"h{site:02d}") % len(FIELD_VALUES)]
            for site in range(machine.SITE_COUNT)
        ],
        dtype=np.float64,
    )
    return coupling, field


def development_identities() -> set[str]:
    document = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    records = document["records"]
    if len(records) != 512:
        raise RuntimeError("complete 512-case six-site development set required")
    identities: set[str] = set()
    for record in records:
        coupling = np.asarray(record["coupling_matrix_J"], dtype=np.float64)
        field = np.asarray(record["field_vector_h"], dtype=np.float64)
        machine.validate_problem(coupling, field, machine.DEFAULT_LAW)
        identity = machine.problem_identity_sha256(coupling, field)
        if identity != record["problem_sha256"]:
            raise RuntimeError("development J/h identity mismatch")
        identities.add(identity)
    if len(identities) != 512:
        raise RuntimeError("development identities are not unique")
    return identities


def batch_document() -> dict[str, Any]:
    excluded = development_identities()
    records: list[dict[str, Any]] = []
    generator_index = 0
    while len(records) < BATCH_SIZE:
        coupling, field = generated_problem(generator_index)
        identity = machine.problem_identity_sha256(coupling, field)
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
        "development_identity_count_excluded": 512,
        "field_values": list(FIELD_VALUES),
        "generation_rule": (
            "SHA256(public_seed|generator_index|coordinate), modulo frozen value "
            "list; skip only complete problem-identity duplicates or development "
            "identities"
        ),
        "no_difficulty_outcome_or_uniqueness_filtering": True,
        "ordered_batch_sha256": ordered_hash,
        "ordered_instances": records,
        "public_seed": PUBLIC_BATCH_SEED,
        "schema": "catalytic_waveform_ising_v3_six_site_batch_custody_v1",
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


def load_qualified_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    development = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    controls = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    if not development["development_pass"]:
        raise RuntimeError("six-site development qualification did not pass")
    if not development["five_site_preservation"]["pass"]:
        raise RuntimeError("exact five-site preservation did not pass")
    summary = development["summary"]
    if not (
        summary["unique_raw_correct"] == summary["unique_count"]
        and summary["accepted_incorrect"] == 0
        and summary["rejected_unique"] == 0
        and summary["non_unique_rejected"] == summary["non_unique_count"]
        and summary["reuse_pass_count"] == summary["case_count"]
    ):
        raise RuntimeError("known six-site development failures remain")
    if not controls["overall_pass"]:
        raise RuntimeError("six-site controls did not pass")
    fingerprints = {
        development["machine_fingerprint"],
        controls["machine_fingerprint"],
        machine.machine_fingerprint(),
    }
    if len(fingerprints) != 1:
        raise RuntimeError("qualified evidence does not bind one machine")
    return development, controls


def assert_five_site_identity() -> None:
    if sha256_file(FIVE_SITE_SOURCE) != FIVE_SITE_SOURCE_SHA256:
        raise RuntimeError("frozen five-site V3 source changed")
    if sha256_file(FIVE_SITE_MANIFEST) != FIVE_SITE_MANIFEST_SHA256:
        raise RuntimeError("frozen five-site V3 manifest changed")


def freeze_document(batch: dict[str, Any]) -> dict[str, Any]:
    development, controls = load_qualified_inputs()
    assert_five_site_identity()
    return {
        "batch_file_sha256": sha256_bytes(canonical_bytes(batch)),
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "batch_size": batch["batch_size"],
        "claim_ceiling": machine.CLAIM_CEILING,
        "control_results_sha256": sha256_file(CONTROL_RESULTS),
        "control_summary": controls["summary"],
        "development_results_sha256": sha256_file(DEVELOPMENT_RESULTS),
        "development_summary": development["summary"],
        "dimension_general_law": {
            "active_bin_count": machine.MODE_COUNT,
            "active_bin_start": machine.ACTIVE_BIN_START,
            "complex_samples_per_mode": machine.COMPLEX_SAMPLES_PER_MODE,
            "maximum_penalty_bound": machine.maximum_penalty_bound(
                machine.SITE_COUNT, 2.0, 2.0
            ),
            "mode_count": machine.MODE_COUNT,
            "phase_denominator": machine.RELATION_PHASE_DENOMINATOR,
            "sample_count": machine.SAMPLE_COUNT,
            "site_count": machine.SITE_COUNT,
        },
        "execution_source_sha256": {
            name: sha256_file(PACKAGE_DIR / name) for name in EXECUTION_SOURCE_NAMES
        },
        "experiment_contract_bytes": CONTRACT_FILE.stat().st_size,
        "experiment_contract_sha256": sha256_file(CONTRACT_FILE),
        "five_site_preservation": development["five_site_preservation"],
        "five_site_v3_manifest_sha256": FIVE_SITE_MANIFEST_SHA256,
        "five_site_v3_source_sha256": FIVE_SITE_SOURCE_SHA256,
        "freeze_before_waveform_execution_or_oracle": True,
        "machine_contract": machine.machine_contract(),
        "machine_fingerprint": machine.machine_fingerprint(),
        "machine_source_bytes": MACHINE_SOURCE.stat().st_size,
        "machine_source_sha256": sha256_file(MACHINE_SOURCE),
        "predecessor_machine_fingerprint": FIVE_SITE_MACHINE_FINGERPRINT,
        "promotion_criterion": frozen_promotion_criterion(),
        "schema": "catalytic_waveform_ising_v3_six_site_freeze_v1",
        "source_execution_contract": SOURCE_EXECUTION_CONTRACT,
        "starting_remote_head": STARTING_REMOTE_HEAD,
        "transitive_dependency_sha256": {
            name: sha256_file(path)
            for name, path in sorted(TRANSITIVE_DEPENDENCY_PATHS.items())
        },
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
        raise ValueError("six-site batch custody does not reproduce")
    if FREEZE_FILE.read_bytes() != canonical_bytes(freeze):
        raise ValueError("six-site freeze does not reproduce")
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
                "status": "SIX_SITE_FROZEN_BEFORE_PROSPECTIVE_EXECUTION_OR_ORACLE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
