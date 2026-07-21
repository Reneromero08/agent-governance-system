from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
SUCCESSOR_SOURCE = PACKAGE_DIR / "successor_machine.py"
DEVELOPMENT_RESULT = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.json"
AUTHORITY_FILE = PACKAGE_DIR / "EXPERIMENT_AUTHORITY.txt"
FREEZE_FILE = PACKAGE_DIR / "SUCCESSOR_FREEZE.json"
BATCH_FILE = PACKAGE_DIR / "V2_BATCH_CUSTODY.json"
REPORT_FILE = PACKAGE_DIR / "SUCCESSOR_AND_BATCH_FREEZE.md"
HELDOUT_CUSTODY = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_heldout_v1"
    / "HELD_OUT_INSTANCE_CUSTODY.json"
)
PREVIOUS_BATCH_CUSTODY = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_batch_v1"
    / "BATCH_INSTANCE_CUSTODY.json"
)

STARTING_HEAD = "32df773f7d2ee3641a577c0352cc6430d5e0a5d2"
AUTHORITY_BYTES = 12444
AUTHORITY_SHA256 = "028faaf10c4705e6cb811e3a9326b61a24265e718bcde34caba533cf98a4f777"
PUBLIC_SEED = (
    "CATCAS_AUDIO_V2_BATCH|COHERENCE_TRANSFORM_CAUSAL|"
    "32df773f7d2ee3641a577c0352cc6430d5e0a5d2|20260720|R0"
)
BATCH_SIZE = 32
J_VALUES = (-2.0, -1.0, 1.0, 2.0)
H_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(SUCCESSOR_SOURCE, "catcas_v2_freeze_machine")


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
    temporary.replace(path)


def source_call_absence() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"exact_oracle", "optimum_states", "ising_energy"}
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden:
            findings.append(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
            findings.append(node.func.attr)
    return {
        "forbidden_adjudication_calls": sorted(findings),
        "pass": not findings,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def machine_document() -> dict[str, Any]:
    development = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    if not development["pass"]:
        raise ValueError("successor development qualification did not pass")
    document = {
        "acceptance_thresholds": {
            "coherence_min": v2.COHERENCE_MIN,
            "lock_residual_max_rad": v2.LOCK_RESIDUAL_MAX,
            "restoration_max_abs_error": v2.RESTORATION_MAX,
            "wrong_restoration_min_abs_error": v2.WRONG_RESTORATION_MIN,
        },
        "claim_ceiling": v2.CLAIM_CEILING,
        "development_qualification_sha256": sha256_file(DEVELOPMENT_RESULT),
        "inverse_law": (
            "reverse each recorded phase update, transport mask, and circular shift "
            "in exact reverse order; then remove the actual recursive transform and "
            "initial site phase"
        ),
        "machine_law": asdict(v2.DEFAULT_LAW),
        "mechanism": {
            "carrier_power_conditioning": (
                "predecessor law for 1000 solve steps, then five native reversible "
                "consolidation steps"
            ),
            "coherence": (
                "native complex within-site spatial flow plus transported mean-wave "
                "coupling; no boundary cleanup"
            ),
            "recursive_transform": (
                "actual recursive waveform transform determines the recurrent complex "
                "pair-channel phase at half predecessor depth"
            ),
        },
        "predecessor_results_preserved": [
            "CATALYTIC_WAVEFORM_ISING_COMPUTATION_VERIFIED",
            "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_PARTIAL",
            "CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL",
        ],
        "restoration_law": {
            "result_latch_outside_reversal": True,
            "reuse_consumes_exact_restored_carrier": True,
            "wrong_order_and_omission_controls_required": True,
        },
        "schema": "catalytic_waveform_ising_v2_machine_freeze_v1",
        "starting_head": STARTING_HEAD,
        "strict_control_law": (
            "causal replacements pass only when both native operator-history L2 and "
            "final complex-response L2 are at least 1e-3; validity-flag changes alone "
            "never pass"
        ),
        "successor_source_bytes": SUCCESSOR_SOURCE.stat().st_size,
        "successor_source_sha256": sha256_file(SUCCESSOR_SOURCE),
    }
    document["machine_sha256"] = sha256_bytes(canonical_bytes(document))
    return document


def problem_identity(coupling: Sequence[Sequence[float]], field: Sequence[float]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "coupling_matrix_J": coupling,
                "field_vector_h": field,
            }
        )
    )


def prior_instances() -> list[dict[str, Any]]:
    heldout = json.loads(HELDOUT_CUSTODY.read_text(encoding="utf-8"))
    batch = json.loads(PREVIOUS_BATCH_CUSTODY.read_text(encoding="utf-8"))
    records = [
        {
            "coupling_matrix_J": v2.r4.COUPLING.tolist(),
            "field_vector_h": v2.r4.PRIMARY_FIELD.tolist(),
            "label": "verified_primary",
        },
        {
            "coupling_matrix_J": v2.r4.COUPLING.tolist(),
            "field_vector_h": v2.r4.REUSE_FIELD.tolist(),
            "label": "verified_reuse",
        },
        {
            "coupling_matrix_J": heldout["held_out_instance"]["coupling_matrix_J"],
            "field_vector_h": heldout["held_out_instance"]["field_vector_h"],
            "label": "first_heldout",
        },
    ]
    for record in batch["ordered_instances"]:
        records.append(
            {
                "coupling_matrix_J": record["coupling_matrix_J"],
                "field_vector_h": record["field_vector_h"],
                "label": f"prior_batch_{int(record['index']):02d}",
            }
        )
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
                    "reason": "complete_J_h_pair_previously_used",
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
    ordered_hash = sha256_bytes(canonical_bytes(accepted))
    return {
        "batch_size": BATCH_SIZE,
        "coefficient_sets": {
            "h": list(H_VALUES),
            "J": list(J_VALUES),
        },
        "excluded_prior_instances": previous,
        "freeze_order": {
            "batch_executed_before_freeze": False,
            "machine_tuned_after_new_batch_generation": False,
            "oracle_uniqueness_inspected_before_freeze": False,
            "result_based_instance_selection": False,
        },
        "generation_rule": (
            "Enumerate candidate_index from zero. For every upper-triangle J edge and "
            "h coordinate, SHA-256 the public seed and coordinate label, map the first "
            "eight big-endian digest bytes modulo the frozen coefficient list, mirror J, "
            "and keep its diagonal zero. Skip only complete-pair identity duplicates or "
            "pairs used in prior primary, reuse, held-out, or batch evidence. Stop at 32."
        ),
        "ordered_batch_sha256": ordered_hash,
        "ordered_instances": accepted,
        "promotion_criterion": {
            "accepted_correct_count_min": 12,
            "accepted_correct_rate_among_unique_min": 0.60,
            "accepted_incorrect_count_max": 0,
            "all_other_strict_controls_must_pass": True,
            "batch_size_required": 32,
            "native_no_smuggle_must_pass": True,
            "restoration_success_all_instances": True,
            "reuse_success_all_instances": True,
            "strict_removed_transform_all_instances": True,
            "unique_optimum_instance_count_min": 16,
            "uninterpretable_count_max": 0,
        },
        "public_seed": PUBLIC_SEED,
        "schema": "catalytic_waveform_ising_v2_batch_custody_v1",
        "selection_law": "identity_only_no_waveform_execution_no_oracle",
        "skipped_candidates": skipped,
    }


def freeze_document(authority: bytes) -> dict[str, Any]:
    if len(authority) != AUTHORITY_BYTES or sha256_bytes(authority) != AUTHORITY_SHA256:
        raise ValueError("experiment authority identity mismatch")
    machine = machine_document()
    batch = batch_document()
    absence = source_call_absence()
    if not absence["pass"]:
        raise ValueError("freeze source contains an adjudication call")
    return {
        "authority": {
            "bytes": len(authority),
            "path": AUTHORITY_FILE.name,
            "sha256": sha256_bytes(authority),
        },
        "batch_ordered_sha256": batch["ordered_batch_sha256"],
        "batch_size": batch["batch_size"],
        "claim_ceiling": v2.CLAIM_CEILING,
        "freeze_source_absence_proof": absence,
        "machine": machine,
        "schema": "catalytic_waveform_ising_v2_successor_and_batch_freeze_v1",
        "status": "FROZEN_BEFORE_NEW_BATCH_EXECUTION_AND_ORACLE",
    }


def report_bytes(freeze: dict[str, Any], batch: dict[str, Any]) -> bytes:
    lines = [
        "# Catalytic Waveform-Ising V2 Successor and Batch Freeze",
        "",
        "Status: `FROZEN_BEFORE_NEW_BATCH_EXECUTION_AND_ORACLE`",
        "",
        f"Machine SHA-256: `{freeze['machine']['machine_sha256']}`",
        f"Authority: `{freeze['authority']['sha256']}` ({freeze['authority']['bytes']} bytes)",
        f"Batch size: `{batch['batch_size']}`",
        f"Ordered batch SHA-256: `{batch['ordered_batch_sha256']}`",
        "",
        "The acceptance thresholds remain 0.90 coherence, 0.15-radian lock residual, "
        "2e-12 restoration error, and 1e-3 wrong-restoration/control materiality.",
        "",
        "The new instances were generated and excluded by complete-pair identity only. "
        "No native execution, exact oracle, uniqueness check, or result-based selection "
        "occurred before this freeze.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def build(authority_source: Path) -> dict[str, Any]:
    authority = authority_source.read_bytes()
    freeze = freeze_document(authority)
    batch = batch_document()
    write_atomic(AUTHORITY_FILE, authority)
    write_atomic(FREEZE_FILE, canonical_bytes(freeze))
    write_atomic(BATCH_FILE, canonical_bytes(batch))
    write_atomic(REPORT_FILE, report_bytes(freeze, batch))
    return freeze


def verify() -> dict[str, Any]:
    authority = AUTHORITY_FILE.read_bytes()
    freeze = freeze_document(authority)
    batch = batch_document()
    expected = {
        FREEZE_FILE: canonical_bytes(freeze),
        BATCH_FILE: canonical_bytes(batch),
        REPORT_FILE: report_bytes(freeze, batch),
    }
    for path, payload in expected.items():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen artifact does not reproduce: {path.name}")
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--authority-source", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.authority_source is None:
            raise ValueError("build requires --authority-source")
        freeze = build(args.authority_source)
    else:
        freeze = verify()
    print(
        json.dumps(
            {
                "batch_ordered_sha256": freeze["batch_ordered_sha256"],
                "batch_size": freeze["batch_size"],
                "machine_sha256": freeze["machine"]["machine_sha256"],
                "status": freeze["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
