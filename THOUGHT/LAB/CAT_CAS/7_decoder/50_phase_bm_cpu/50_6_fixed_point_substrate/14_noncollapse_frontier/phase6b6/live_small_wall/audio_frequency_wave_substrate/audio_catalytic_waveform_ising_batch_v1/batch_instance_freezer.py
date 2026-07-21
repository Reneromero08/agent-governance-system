from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PACKAGE_DIR / "BATCH_INSTANCE_CUSTODY.json"
SCHEMA = "catalytic_waveform_ising_batch_custody_v1"
STARTING_HEAD = "eb1835b6304730f3b1bd0107f1f7c3fc9f9aa275"
PREDECESSOR_SOURCE_SHA256 = (
    "50b6db77e2602e18356636ddb892f6d51aedb0573c6b2418afc8e5cc174991cc"
)
FROZEN_MACHINE_SHA256 = (
    "cf95d0cd364af38d47a2f2784aa489ab5a52dc8aea62131c1a8545ff4978203a"
)
PUBLIC_SEED = (
    "CATCAS_AUDIO_HELD_OUT_BATCH_V1|UNCHANGED_MACHINE|"
    "eb1835b6304730f3b1bd0107f1f7c3fc9f9aa275|BATCH_16"
)
BATCH_SIZE = 16
SITE_COUNT = 5
J_VALUES = (-2.0, -1.0, 1.0, 2.0)
H_VALUES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

VERIFIED_J = (
    (0.0, 0.0, 1.0, 2.0, -2.0),
    (0.0, 0.0, 2.0, -1.0, -1.0),
    (1.0, 2.0, 0.0, 2.0, -1.0),
    (2.0, -1.0, 2.0, 0.0, -2.0),
    (-2.0, -1.0, -1.0, -2.0, 0.0),
)
VERIFIED_PRIMARY_H = (-2.0, 1.0, -2.0, -2.0, -2.0)
VERIFIED_REUSE_H = (1.0, -1.0, 0.5, 0.5, -1.0)
PRIOR_HELDOUT_J = (
    (0.0, 1.0, 1.0, -2.0, 2.0),
    (1.0, 0.0, -2.0, 2.0, 1.0),
    (1.0, -2.0, 0.0, -2.0, 1.0),
    (-2.0, 2.0, -2.0, 0.0, -1.0),
    (2.0, 1.0, 1.0, -1.0, 0.0),
)
PRIOR_HELDOUT_H = (-2.0, -1.0, 0.5, 0.5, -0.5)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def coefficient(instance_index: int, label: str, index: int, values: Sequence[float]) -> float:
    payload = f"{PUBLIC_SEED}|INSTANCE|{instance_index}|{label}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return float(values[int.from_bytes(digest[:8], "big") % len(values)])


def problem_document(instance_index: int) -> dict[str, Any]:
    coupling = [[0.0 for _ in range(SITE_COUNT)] for _ in range(SITE_COUNT)]
    edge_index = 0
    for left in range(SITE_COUNT):
        for right in range(left + 1, SITE_COUNT):
            value = coefficient(instance_index, "J", edge_index, J_VALUES)
            coupling[left][right] = value
            coupling[right][left] = value
            edge_index += 1
    field = [coefficient(instance_index, "h", site, H_VALUES) for site in range(SITE_COUNT)]
    return {
        "coupling_matrix_J": coupling,
        "field_vector_h": field,
        "index": instance_index,
    }


def excluded_problem_keys() -> set[bytes]:
    return {
        canonical_bytes(
            {
                "coupling_matrix_J": [list(row) for row in VERIFIED_J],
                "field_vector_h": list(VERIFIED_PRIMARY_H),
            }
        ),
        canonical_bytes(
            {
                "coupling_matrix_J": [list(row) for row in VERIFIED_J],
                "field_vector_h": list(VERIFIED_REUSE_H),
            }
        ),
        canonical_bytes(
            {
                "coupling_matrix_J": [list(row) for row in PRIOR_HELDOUT_J],
                "field_vector_h": list(PRIOR_HELDOUT_H),
            }
        ),
    }


def problem_key(problem: dict[str, Any]) -> bytes:
    return canonical_bytes(
        {
            "coupling_matrix_J": problem["coupling_matrix_J"],
            "field_vector_h": problem["field_vector_h"],
        }
    )


def ordered_instances() -> list[dict[str, Any]]:
    exclusions = excluded_problem_keys()
    seen: set[bytes] = set()
    records: list[dict[str, Any]] = []
    for index in range(BATCH_SIZE):
        problem = problem_document(index)
        key = problem_key(problem)
        if key in exclusions:
            raise ValueError(f"batch instance {index} duplicates a predecessor instance")
        if key in seen:
            raise ValueError(f"batch instance {index} duplicates an earlier batch instance")
        seen.add(key)
        records.append({**problem, "instance_sha256": sha256_bytes(key)})
    return records


def ordered_batch_hash(instances: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes(list(instances)))


def custody_document() -> dict[str, Any]:
    instances = ordered_instances()
    return {
        "batch_generation": {
            "batch_size": BATCH_SIZE,
            "generation_rule": (
                "For each zero-based batch index, upper-triangle edge, and field coordinate, "
                "SHA-256 the public seed plus the domain-separated instance index, coefficient "
                "label, and coordinate. Interpret the first eight digest bytes as an unsigned "
                "big-endian integer and reduce modulo the prospectively declared coefficient "
                "list. Mirror J exactly and set its diagonal to exact zero. Abort rather than "
                "replace an instance if any complete (J,h) pair duplicates a predecessor or "
                "earlier batch pair. No oracle or waveform result participates."
            ),
            "h_coefficient_values": list(H_VALUES),
            "j_coefficient_values": list(J_VALUES),
            "public_seed": PUBLIC_SEED,
            "site_count": SITE_COUNT,
        },
        "freeze_order": {
            "any_batch_native_evolution_run_before_freeze": False,
            "any_oracle_consulted_before_freeze": False,
            "any_result_based_instance_selection": False,
            "machine_tuned_after_batch_generation": False,
        },
        "freezer_source_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "frozen_machine": {
            "machine_sha256": FROZEN_MACHINE_SHA256,
            "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
            "variables_allowed_to_change": ["coupling_matrix_J", "field_vector_h"],
        },
        "ordered_batch_sha256": ordered_batch_hash(instances),
        "ordered_instances": instances,
        "predecessor_exclusions": {
            "count": 3,
            "prior_heldout_instance_sha256": (
                "49db989fd525366867cf9c6866ebc7000b531b438b0227d7bb919e0ff3bf2704"
            ),
        },
        "promotion_criterion_frozen_before_execution": {
            "accepted_correct_count_min": 8,
            "accepted_correct_rate_among_unique_min": 0.5,
            "accepted_incorrect_count_max": 0,
            "all_causality_restoration_reuse_and_controls_must_pass": True,
            "batch_size_required": BATCH_SIZE,
            "unique_optimum_instance_count_min": 12,
            "uninterpretable_count_max": 0,
        },
        "schema": SCHEMA,
        "starting_head": STARTING_HEAD,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> None:
    write_atomic(OUTPUT_PATH, canonical_bytes(custody_document()))


def verify() -> None:
    expected = canonical_bytes(custody_document())
    if OUTPUT_PATH.read_bytes() != expected:
        raise ValueError("committed batch custody does not reproduce")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    if args.command == "build":
        build()
    else:
        verify()
    document = custody_document()
    print(json.dumps({
        "batch_size": len(document["ordered_instances"]),
        "ordered_batch_sha256": document["ordered_batch_sha256"],
        "schema": document["schema"],
        "status": "FROZEN_BEFORE_EXECUTION_AND_ORACLE",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
