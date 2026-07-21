from __future__ import annotations

import argparse
import ast
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
GATE_SOURCE = PACKAGE_DIR / "stability_gate.py"
DEVELOPMENT_FILE = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.json"
NO_SMUGGLE_FILE = PACKAGE_DIR / "NO_SMUGGLE_PROOF.json"
AUTHORITY_FILE = PACKAGE_DIR / "EXPERIMENT_AUTHORITY.txt"
FREEZE_FILE = PACKAGE_DIR / "SUCCESSOR_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "STABILITY_BATCH_CUSTODY.json"
REPORT_FILE = PACKAGE_DIR / "SUCCESSOR_AND_BATCH_FREEZE.md"
V2_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2"
V2_SOURCE = V2_DIR / "successor_machine.py"
V2_FREEZE = V2_DIR / "SUCCESSOR_FREEZE.json"
V2_BATCH = V2_DIR / "V2_BATCH_CUSTODY.json"

STARTING_HEAD = "a186eea796dad433143b42c50fb3dbb6513a0572"
V2_MACHINE_FINGERPRINT = (
    "c20f2cd4068ca32528bc52671793bca04d897456944e33ad8571083428c48930"
)
V2_SOURCE_SHA256 = (
    "48f16ddf33e635b7f58881a0e31486dd0ec5deb181ab0fab8e0a2657a8fa5ce7"
)
AUTHORITY_BYTES = 14547
AUTHORITY_SHA256 = "5fa4c5af4ab4820a47d6acdadbbf421b8c844f008c72d349db246278e25758cf"
PUBLIC_SEED = (
    "CATCAS_AUDIO_V2_STABILITY_GATE_BATCH|LATE_COMPLEX_KINETIC_DISCRIMINATOR|"
    "a186eea796dad433143b42c50fb3dbb6513a0572|20260721|R0"
)
BATCH_SIZE = 64
J_VALUES = (-2.0, -1.0, 1.0, 2.0)
H_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
CLAIM_CEILING = "BOUNDED_SOFTWARE_REJECT_ONLY_WAVEFORM_STABILITY_REFERENCE_ONLY"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gate = load_module(GATE_SOURCE, "catcas_stability_freeze_gate")
v2 = gate.v2


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def source_execution_absence() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "evaluate_stability",
        "exact_oracle",
        "execute_native_cycle",
        "ising_energy",
        "project_boundary",
    }
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden:
            findings.append(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
            findings.append(node.func.attr)
    return {
        "forbidden_execution_or_adjudication_calls": sorted(findings),
        "pass": not findings,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def machine_identity() -> dict[str, Any]:
    frozen = json.loads(V2_FREEZE.read_text(encoding="utf-8"))
    source_sha = sha256_file(V2_SOURCE)
    if source_sha != V2_SOURCE_SHA256:
        raise RuntimeError("frozen V2 source identity changed")
    if frozen["machine"]["machine_sha256"] != V2_MACHINE_FINGERPRINT:
        raise RuntimeError("frozen V2 machine fingerprint changed")
    if frozen["machine"]["successor_source_sha256"] != source_sha:
        raise RuntimeError("V2 source does not match its frozen evidence")
    return {
        "machine_fingerprint": V2_MACHINE_FINGERPRINT,
        "source_bytes": V2_SOURCE.stat().st_size,
        "source_path": "../audio_catalytic_waveform_ising_v2/successor_machine.py",
        "source_sha256": source_sha,
    }


def discriminator_document() -> dict[str, Any]:
    development = json.loads(DEVELOPMENT_FILE.read_text(encoding="utf-8"))
    no_smuggle = json.loads(NO_SMUGGLE_FILE.read_text(encoding="utf-8"))
    if not development["pass"] or not no_smuggle["pass"]:
        raise RuntimeError("stability discriminator did not pass development qualification")
    document = {
        "acceptance_law": gate.gate_contract()["acceptance_law"],
        "claim_ceiling": CLAIM_CEILING,
        "development_qualification_sha256": sha256_file(DEVELOPMENT_FILE),
        "diagnostic_restoration_law": (
            "replay the exact recorded V2 complex phase operators, transport masks, "
            "and circular shifts; query only the nine frozen late checkpoints; then "
            "traverse the same inverse history and require max error <= 2e-12"
        ),
        "fixed_diagnostic_schedule": gate.gate_contract()[
            "checkpoint_steps_zero_based_after_update"
        ],
        "mechanism": (
            "reject only when the same nominal V2 complex trajectory jointly exhibits "
            "peak late phase velocity above 0.008 rad/step and mean late complex-response "
            "drift above 0.08 L2"
        ),
        "null_baseline": (
            "disable the stability gate and reproduce the unchanged nominal V2 "
            "acceptance decision and raw result"
        ),
        "no_smuggle_proof_sha256": sha256_file(NO_SMUGGLE_FILE),
        "raw_result_effect": "REJECT_ONLY_NEVER_ALTERS_OR_SELECTS_A_RESULT",
        "schema": "catalytic_waveform_ising_v2_stability_discriminator_v1",
        "source_bytes": GATE_SOURCE.stat().st_size,
        "source_sha256": sha256_file(GATE_SOURCE),
        "thresholds": {
            "diagnostic_restoration_max_abs_error": gate.DIAGNOSTIC_RESTORATION_MAX,
            "late_mean_response_drift_l2_max": gate.MAX_LATE_MEAN_RESPONSE_DRIFT_L2,
            "late_phase_velocity_rad_per_step_max": (
                gate.MAX_LATE_PHASE_VELOCITY_RAD_PER_STEP
            ),
            "replay_max_abs_delta": gate.REPLAY_TOLERANCE,
        },
        "v2_machine_fingerprint": V2_MACHINE_FINGERPRINT,
    }
    document["discriminator_fingerprint"] = sha256_bytes(canonical_bytes(document))
    return document


def problem_identity(coupling: Sequence[Sequence[float]], field: Sequence[float]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {"coupling_matrix_J": coupling, "field_vector_h": field}
        )
    )


def prior_instances() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, record in enumerate(v2.development_instances()):
        if index < 2:
            source_group = "original_primary_and_reuse"
        elif index == 2:
            source_group = "first_heldout"
        else:
            source_group = "v1_batch"
        records.append(
            {
                "coupling_matrix_J": np.asarray(record["coupling"]).tolist(),
                "field_vector_h": np.asarray(record["field"]).tolist(),
                "source_group": source_group,
            }
        )
    v2_batch = json.loads(V2_BATCH.read_text(encoding="utf-8"))
    for record in v2_batch["ordered_instances"]:
        records.append(
            {
                "coupling_matrix_J": record["coupling_matrix_J"],
                "field_vector_h": record["field_vector_h"],
                "source_group": "v2_batch",
            }
        )
    if len(records) != 51:
        raise RuntimeError("excluded development corpus is not exactly 51 instances")
    for record in records:
        record["problem_sha256"] = problem_identity(
            record["coupling_matrix_J"], record["field_vector_h"]
        )
    return records


def coefficient(label: str, values: Sequence[float]) -> float:
    digest = hashlib.sha256((PUBLIC_SEED + "|" + label).encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(values)
    return float(values[index])


def candidate(candidate_index: int) -> dict[str, Any]:
    coupling = np.zeros((v2.SITE_COUNT, v2.SITE_COUNT), dtype=np.float64)
    edge_index = 0
    for row in range(v2.SITE_COUNT):
        for column in range(row + 1, v2.SITE_COUNT):
            value = coefficient(
                f"candidate={candidate_index}|J|edge={edge_index}|{row},{column}",
                J_VALUES,
            )
            coupling[row, column] = value
            coupling[column, row] = value
            edge_index += 1
    field = [
        coefficient(f"candidate={candidate_index}|h|site={site}", H_VALUES)
        for site in range(v2.SITE_COUNT)
    ]
    return {
        "candidate_index": candidate_index,
        "coupling_matrix_J": coupling.tolist(),
        "field_vector_h": field,
    }


def promotion_criterion() -> dict[str, Any]:
    return {
        "accepted_correct_count_min": 32,
        "accepted_correct_rate_among_unique_min": 0.80,
        "accepted_incorrect_count_max": 0,
        "batch_size_required": 64,
        "diagnostic_restoration_all_instances": True,
        "native_no_smuggle_must_pass": True,
        "nominal_restoration_all_instances": True,
        "reuse_all_instances": True,
        "stability_controls_all_instances": True,
        "stability_gate_no_smuggle_must_pass": True,
        "strict_v2_controls_all_instances": True,
        "unique_optimum_instance_count_min": 32,
        "uninterpretable_count_max": 0,
    }


def batch_document() -> dict[str, Any]:
    previous = prior_instances()
    excluded = {record["problem_sha256"] for record in previous}
    accepted: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    skipped: list[dict[str, Any]] = []
    candidate_index = 0
    while len(accepted) < BATCH_SIZE:
        record = candidate(candidate_index)
        identity = problem_identity(
            record["coupling_matrix_J"], record["field_vector_h"]
        )
        if identity in excluded:
            skipped.append(
                {
                    "candidate_index": candidate_index,
                    "problem_sha256": identity,
                    "reason": "complete_J_h_pair_in_51_case_development_corpus",
                }
            )
        elif identity in accepted_hashes:
            skipped.append(
                {
                    "candidate_index": candidate_index,
                    "problem_sha256": identity,
                    "reason": "duplicate_inside_new_batch",
                }
            )
        else:
            record["index"] = len(accepted)
            record["problem_sha256"] = identity
            accepted.append(record)
            accepted_hashes.add(identity)
        candidate_index += 1
    return {
        "batch_size": BATCH_SIZE,
        "coefficient_sets": {"J": list(J_VALUES), "h": list(H_VALUES)},
        "excluded_development_instances": previous,
        "freeze_order": {
            "batch_waveform_executed_before_freeze": False,
            "discriminator_tuned_after_batch_generation": False,
            "oracle_or_uniqueness_inspected_before_freeze": False,
            "result_based_replacement": False,
        },
        "generation_rule": (
            "Enumerate candidate_index from zero. For each upper-triangle J edge and "
            "each h coordinate, SHA-256 the public seed plus the coordinate label, map "
            "the first eight big-endian digest bytes modulo the frozen coefficient "
            "list, mirror J, and keep its diagonal zero. Skip only complete J,h pair "
            "identity collisions with the 51-case development corpus or within this "
            "new batch. Stop at 64 without waveform execution or oracle inspection."
        ),
        "ordered_batch_sha256": sha256_bytes(canonical_bytes(accepted)),
        "ordered_instances": accepted,
        "promotion_criterion": promotion_criterion(),
        "public_seed": PUBLIC_SEED,
        "schema": "catalytic_waveform_ising_v2_stability_batch_custody_v1",
        "selection_law": "IDENTITY_ONLY_NO_WAVEFORM_EXECUTION_NO_ORACLE",
        "skipped_candidates": skipped,
    }


def freeze_document() -> dict[str, Any]:
    authority = AUTHORITY_FILE.read_bytes()
    if len(authority) != AUTHORITY_BYTES or sha256_bytes(authority) != AUTHORITY_SHA256:
        raise RuntimeError("experiment authority identity mismatch")
    absence = source_execution_absence()
    if not absence["pass"]:
        raise RuntimeError("freeze builder crossed execution or adjudication boundary")
    machine = machine_identity()
    discriminator = discriminator_document()
    batch = batch_document()
    return {
        "authority": {
            "bytes": len(authority),
            "path": AUTHORITY_FILE.name,
            "sha256": sha256_bytes(authority),
        },
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "batch_size": batch["batch_size"],
        "claim_ceiling": CLAIM_CEILING,
        "discriminator": discriminator,
        "existing_v2_thresholds_unchanged": {
            "coherence_min": v2.COHERENCE_MIN,
            "lock_residual_max_rad": v2.LOCK_RESIDUAL_MAX,
            "restoration_max_abs_error": v2.RESTORATION_MAX,
            "wrong_restoration_min_abs_error": v2.WRONG_RESTORATION_MIN,
        },
        "freeze_source_absence_proof": absence,
        "machine": machine,
        "preserved_results": [
            "CATALYTIC_WAVEFORM_ISING_COMPUTATION_VERIFIED",
            "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_PARTIAL",
            "CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL",
            "CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_PARTIAL",
        ],
        "schema": "catalytic_waveform_ising_v2_stability_successor_freeze_v1",
        "starting_head": STARTING_HEAD,
        "status": "FROZEN_BEFORE_64_INSTANCE_WAVEFORM_EXECUTION_AND_ORACLE",
    }


def report_bytes(freeze: dict[str, Any], batch: dict[str, Any]) -> bytes:
    return (
        "# V2 waveform-native stability discriminator and batch freeze\n\n"
        f"Status: `{freeze['status']}`\n\n"
        f"V2 machine fingerprint: `{freeze['machine']['machine_fingerprint']}`\n"
        f"Discriminator fingerprint: `{freeze['discriminator']['discriminator_fingerprint']}`\n"
        f"Authority: `{freeze['authority']['sha256']}` ({freeze['authority']['bytes']} bytes)\n"
        f"Batch size: `{batch['batch_size']}`\n"
        f"Ordered batch SHA-256: `{batch['ordered_batch_sha256']}`\n\n"
        "The gate is reject-only and observes only nine fixed complex checkpoints from "
        "the unchanged nominal V2 trajectory. The complete 64-instance batch was "
        "generated by public SHA-256 coordinate mapping and identity-only exclusion. "
        "No new-batch waveform execution, oracle, uniqueness check, difficulty filter, "
        "or result-based replacement occurred before this freeze.\n"
    ).encode("utf-8")


def build() -> dict[str, Any]:
    freeze = freeze_document()
    batch = batch_document()
    write_atomic(FREEZE_FILE, canonical_bytes(freeze))
    write_atomic(BATCH_FILE, canonical_bytes(batch))
    write_atomic(REPORT_FILE, report_bytes(freeze, batch))
    return freeze


def verify() -> dict[str, Any]:
    freeze = freeze_document()
    batch = batch_document()
    expected = {
        FREEZE_FILE: canonical_bytes(freeze),
        BATCH_FILE: canonical_bytes(batch),
        REPORT_FILE: report_bytes(freeze, batch),
    }
    for path, payload in expected.items():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen artifact does not reproduce: {path.name}")
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    freeze = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "batch_ordered_sha256": freeze["batch_ordered_sha256"],
                "batch_size": freeze["batch_size"],
                "discriminator_fingerprint": freeze["discriminator"][
                    "discriminator_fingerprint"
                ],
                "machine_fingerprint": freeze["machine"]["machine_fingerprint"],
                "status": freeze["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
