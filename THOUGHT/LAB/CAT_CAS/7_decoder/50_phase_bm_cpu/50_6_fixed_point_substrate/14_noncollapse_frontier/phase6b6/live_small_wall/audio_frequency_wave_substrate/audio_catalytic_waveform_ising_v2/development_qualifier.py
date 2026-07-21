from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
SUCCESSOR_SOURCE = PACKAGE_DIR / "successor_machine.py"
PREDECESSOR_ADAPTER_SOURCE = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_heldout_v1"
    / "heldout_generalization_reference.py"
)
RESULT_FILE = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.json"
REPORT_FILE = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.md"


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v2 = load_module(SUCCESSOR_SOURCE, "catcas_v2_development_machine")
held = load_module(PREDECESSOR_ADAPTER_SOURCE, "catcas_v2_development_predecessor")
r4 = held.r4


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "max": metric(np.max(array)),
        "mean": metric(np.mean(array)),
        "median": metric(np.median(array)),
        "min": metric(np.min(array)),
    }


def optimum_states(
    coupling: np.ndarray, field: np.ndarray
) -> tuple[float, set[tuple[int, ...]]]:
    rows = v2.exact_oracle(coupling, field)
    energy = rows[0][0]
    states = {spins for value, spins in rows if abs(value - energy) <= 1.0e-12}
    return energy, states


def predecessor_cycle(
    borrowed: np.ndarray, coupling: np.ndarray, field: np.ndarray
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    execution = held.execute_native_cycle_problem(borrowed, coupling, field)
    latch = held.project_boundary_problem(execution, coupling, field, "predecessor")
    restored = r4.restore_carrier(execution, "correct")
    controls = held.run_controls(borrowed, coupling, field, execution, latch)
    return execution, latch, restored, controls


def successor_cycle(
    borrowed: np.ndarray, coupling: np.ndarray, field: np.ndarray
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    execution = v2.execute_native_cycle(borrowed, coupling, field)
    latch = v2.project_boundary(execution, "successor")
    restored = v2.restore_carrier(execution, "correct")
    controls = v2.run_strict_controls(
        borrowed, coupling, field, execution, latch, v2.DEFAULT_LAW
    )
    return execution, latch, restored, controls


def version_record(
    *,
    name: str,
    execution: Any,
    latch: Any,
    restored: np.ndarray,
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    controls: dict[str, Any],
    reuse_execute: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
    reuse_project: Callable[[Any, np.ndarray, np.ndarray, str], Any],
    reuse_restore: Callable[[Any], np.ndarray],
    optimum: set[tuple[int, ...]],
) -> dict[str, Any]:
    raw_match = tuple(latch.raw_spins) in optimum
    restoration_error = float(np.max(np.abs(restored - borrowed)))
    reuse_execution = reuse_execute(restored, r4.COUPLING, r4.REUSE_FIELD)
    reuse_latch = reuse_project(
        reuse_execution, r4.COUPLING, r4.REUSE_FIELD, f"{name}_reuse"
    )
    reuse_restored = reuse_restore(reuse_execution)
    reuse_input_error = float(np.max(np.abs(reuse_execution.borrowed - restored)))
    reuse_restore_error = float(np.max(np.abs(reuse_restored - restored)))
    residual = v2.lock_residual(latch.phases)
    return {
        "accepted": bool(latch.valid),
        "accepted_correct": bool(latch.valid and raw_match),
        "accepted_incorrect": bool(latch.valid and not raw_match),
        "carrier_displacement_l2": metric(execution.displacement_l2),
        "coherence": [metric(value) for value in latch.coherence],
        "minimum_coherence": metric(min(latch.coherence)),
        "raw_match": raw_match,
        "raw_spins": list(latch.raw_spins),
        "residual_rad": metric(residual),
        "restoration_max_abs_error": metric(restoration_error),
        "reuse": {
            "input_max_abs_error": metric(reuse_input_error),
            "result_valid": bool(reuse_latch.valid),
            "restoration_max_abs_error": metric(reuse_restore_error),
        },
        "strict_controls": {
            "all_pass": bool(controls["all_pass"]),
            "measurements": controls["measurements"],
            "outcomes": controls["outcomes"],
        },
    }


def predecessor_reuse_execute(
    borrowed: np.ndarray, coupling: np.ndarray, field: np.ndarray
) -> Any:
    return held.execute_native_cycle_problem(borrowed, coupling, field)


def predecessor_reuse_project(
    execution: Any, coupling: np.ndarray, field: np.ndarray, label: str
) -> Any:
    return held.project_boundary_problem(execution, coupling, field, label)


def predecessor_reuse_restore(execution: Any) -> np.ndarray:
    return r4.restore_carrier(execution, "correct")


def successor_reuse_execute(
    borrowed: np.ndarray, coupling: np.ndarray, field: np.ndarray
) -> Any:
    return v2.execute_native_cycle(borrowed, coupling, field)


def successor_reuse_project(
    execution: Any, coupling: np.ndarray, field: np.ndarray, label: str
) -> Any:
    del coupling, field
    return v2.project_boundary(execution, label)


def successor_reuse_restore(execution: Any) -> np.ndarray:
    return v2.restore_carrier(execution, "correct")


def aggregate(records: Sequence[dict[str, Any]], version: str) -> dict[str, Any]:
    values = [record[version] for record in records]
    unique = [
        record[version]
        for record in records
        if record["optimum_state_count"] == 1
    ]
    response_key = (
        "no_transform"
        if version == "predecessor"
        else "no_transform"
    )
    if version == "predecessor":
        carrier_key = "carrier_content_causal"
        flat_key = "flat_geometry_changed_or_destroyed"
        scramble_key = "scrambled_geometry_changed_or_destroyed"
        transform_key = "no_transform_changed_or_destroyed"
        rank_key = "samplewise_non_rank_one_residual"
    else:
        carrier_key = "uniform_carrier_replacement"
        flat_key = "flat_geometry"
        scramble_key = "parent_child_geometry_scramble"
        transform_key = "removed_waveform_transform"
        rank_key = "samplewise_non_rank_one_residual"
    return {
        "accepted_correct_unique": sum(item["accepted_correct"] for item in unique),
        "accepted_incorrect_unique": sum(
            item["accepted_incorrect"] for item in unique
        ),
        "carrier_content_causality_pass_count": sum(
            item["strict_controls"]["outcomes"][carrier_key] for item in values
        ),
        "flat_geometry_sensitivity_pass_count": sum(
            item["strict_controls"]["outcomes"][flat_key] for item in values
        ),
        "minimum_coherence_distribution": distribution(
            [item["minimum_coherence"] for item in values]
        ),
        "raw_optimum_agreement_unique": sum(item["raw_match"] for item in unique),
        "residual_rad_distribution": distribution(
            [item["residual_rad"] for item in values]
        ),
        "restoration_error_distribution": distribution(
            [item["restoration_max_abs_error"] for item in values]
        ),
        "reuse_input_error_distribution": distribution(
            [item["reuse"]["input_max_abs_error"] for item in values]
        ),
        "reuse_restoration_error_distribution": distribution(
            [item["reuse"]["restoration_max_abs_error"] for item in values]
        ),
        "samplewise_non_rank_one_distribution": distribution(
            [
                item["strict_controls"]["measurements"][rank_key]
                for item in values
            ]
        ),
        "scrambled_geometry_sensitivity_pass_count": sum(
            item["strict_controls"]["outcomes"][scramble_key]
            for item in values
        ),
        "strict_all_controls_pass_count": sum(
            item["strict_controls"]["all_pass"] for item in values
        ),
        "strict_removed_transform_pass_count": sum(
            item["strict_controls"]["outcomes"][transform_key]
            for item in values
        ),
        "strict_removed_transform_response_delta_distribution": distribution(
            [
                item["strict_controls"]["measurements"]["response_deltas_l2"][
                    response_key
                ]
                for item in values
            ]
        ),
        "total_instance_count": len(values),
        "unique_instance_count": len(unique),
    }


def build_document() -> dict[str, Any]:
    carrier = r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)
    records: list[dict[str, Any]] = []
    for instance in v2.development_instances():
        coupling = instance["coupling"]
        field = instance["field"]
        _, optimum = optimum_states(coupling, field)

        pred_execution, pred_latch, pred_restored, pred_controls = predecessor_cycle(
            borrowed, coupling, field
        )
        candidate_execution, candidate_latch, candidate_restored, candidate_controls = (
            successor_cycle(borrowed, coupling, field)
        )
        records.append(
            {
                "coupling_matrix_J": coupling.tolist(),
                "field_vector_h": field.tolist(),
                "label": instance["label"],
                "optimum_state_count": len(optimum),
                "predecessor": version_record(
                    name="predecessor",
                    execution=pred_execution,
                    latch=pred_latch,
                    restored=pred_restored,
                    borrowed=borrowed,
                    coupling=coupling,
                    field=field,
                    controls=pred_controls,
                    reuse_execute=predecessor_reuse_execute,
                    reuse_project=predecessor_reuse_project,
                    reuse_restore=predecessor_reuse_restore,
                    optimum=optimum,
                ),
                "successor": version_record(
                    name="successor",
                    execution=candidate_execution,
                    latch=candidate_latch,
                    restored=candidate_restored,
                    borrowed=borrowed,
                    coupling=coupling,
                    field=field,
                    controls=candidate_controls,
                    reuse_execute=successor_reuse_execute,
                    reuse_project=successor_reuse_project,
                    reuse_restore=successor_reuse_restore,
                    optimum=optimum,
                ),
            }
        )
    predecessor = aggregate(records, "predecessor")
    successor = aggregate(records, "successor")
    improvement = {
        "accepted_correct_unique_delta": (
            successor["accepted_correct_unique"]
            - predecessor["accepted_correct_unique"]
        ),
        "raw_optimum_agreement_unique_delta": (
            successor["raw_optimum_agreement_unique"]
            - predecessor["raw_optimum_agreement_unique"]
        ),
        "strict_all_controls_pass_count_delta": (
            successor["strict_all_controls_pass_count"]
            - predecessor["strict_all_controls_pass_count"]
        ),
        "strict_removed_transform_pass_count_delta": (
            successor["strict_removed_transform_pass_count"]
            - predecessor["strict_removed_transform_pass_count"]
        ),
    }
    pass_conditions = {
        "accepted_correct_improved": improvement["accepted_correct_unique_delta"] > 0,
        "accepted_incorrect_zero": successor["accepted_incorrect_unique"] == 0,
        "all_successor_strict_controls_pass": (
            successor["strict_all_controls_pass_count"] == len(records)
        ),
        "raw_optimum_behavior_not_regressed": (
            improvement["raw_optimum_agreement_unique_delta"] >= 0
        ),
        "restoration_within_frozen_tolerance": (
            successor["restoration_error_distribution"]["max"]
            <= v2.RESTORATION_MAX
        ),
        "reuse_within_frozen_tolerance": (
            successor["reuse_input_error_distribution"]["max"] == 0.0
            and successor["reuse_restoration_error_distribution"]["max"]
            <= v2.RESTORATION_MAX
        ),
    }
    return {
        "claim_ceiling": v2.CLAIM_CEILING,
        "development_corpus_law": (
            "all prior primary, reuse, held-out, and 16-batch instances are known "
            "development cases and cannot support a new generalization claim"
        ),
        "improvement": improvement,
        "machine_law": as_plain_dict(v2.DEFAULT_LAW),
        "pass": all(pass_conditions.values()),
        "pass_conditions": pass_conditions,
        "predecessor": predecessor,
        "records": records,
        "schema": "catalytic_waveform_ising_v2_development_qualification_v1",
        "source_custody": {
            "predecessor_adapter_sha256": sha256_file(PREDECESSOR_ADAPTER_SOURCE),
            "successor_machine_sha256": sha256_file(SUCCESSOR_SOURCE),
        },
        "successor": successor,
    }


def as_plain_dict(law: Any) -> dict[str, Any]:
    return {
        name: getattr(law, name)
        for name in law.__dataclass_fields__
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    predecessor = document["predecessor"]
    successor = document["successor"]
    lines = [
        "# Catalytic Waveform-Ising V2 Development Qualification",
        "",
        "Status: `PASS`" if document["pass"] else "Status: `FAIL`",
        "",
        "All 19 cases are a known development corpus. This report makes no held-out claim.",
        "",
        "```text",
        f"raw optimum agreement, unique      {predecessor['raw_optimum_agreement_unique']} -> {successor['raw_optimum_agreement_unique']} / {successor['unique_instance_count']}",
        f"accepted correct, unique           {predecessor['accepted_correct_unique']} -> {successor['accepted_correct_unique']}",
        f"accepted incorrect, unique         {predecessor['accepted_incorrect_unique']} -> {successor['accepted_incorrect_unique']}",
        f"strict removed-transform passes    {predecessor['strict_removed_transform_pass_count']} -> {successor['strict_removed_transform_pass_count']} / {successor['total_instance_count']}",
        f"strict all-control passes           {predecessor['strict_all_controls_pass_count']} -> {successor['strict_all_controls_pass_count']} / {successor['total_instance_count']}",
        f"successor coherence min             {successor['minimum_coherence_distribution']['min']}",
        f"successor transform response min    {successor['strict_removed_transform_response_delta_distribution']['min']}",
        f"successor restoration max           {successor['restoration_error_distribution']['max']}",
        f"successor reuse restoration max     {successor['reuse_restoration_error_distribution']['max']}",
        "```",
        "",
        "The successor keeps the 0.90 coherence, 0.15-radian lock, and 2e-12 restoration gates unchanged.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(RESULT_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = build_document()
    if RESULT_FILE.read_bytes() != canonical_bytes(document):
        raise ValueError("development qualification JSON does not reproduce")
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise ValueError("development qualification report does not reproduce")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "pass": document["pass"],
                "predecessor": document["predecessor"],
                "successor": document["successor"],
            },
            sort_keys=True,
        )
    )
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
