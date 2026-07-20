from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PACKAGE_DIR / "HELD_OUT_INSTANCE_CUSTODY.json"
SCHEMA = "catalytic_waveform_ising_heldout_custody_v1"
STARTING_HEAD = "62e8dab8c8631d112122a6e43cb9dcd7a4985bee"
PREDECESSOR_SOURCE_SHA256 = (
    "50b6db77e2602e18356636ddb892f6d51aedb0573c6b2418afc8e5cc174991cc"
)
PREDECESSOR_SOURCE_BYTES = 46388
PUBLIC_SEED = (
    "CATCAS_AUDIO_HELD_OUT_V1|UNCHANGED_MACHINE|"
    "62e8dab8c8631d112122a6e43cb9dcd7a4985bee"
)
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def coefficient(label: str, index: int, values: Sequence[float]) -> float:
    payload = f"{PUBLIC_SEED}|{label}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return float(values[int.from_bytes(digest[:8], "big") % len(values)])


def instance_document() -> dict[str, Any]:
    coupling = [[0.0 for _ in range(SITE_COUNT)] for _ in range(SITE_COUNT)]
    edge_index = 0
    for left in range(SITE_COUNT):
        for right in range(left + 1, SITE_COUNT):
            value = coefficient("J", edge_index, J_VALUES)
            coupling[left][right] = value
            coupling[right][left] = value
            edge_index += 1
    field = [coefficient("h", site, H_VALUES) for site in range(SITE_COUNT)]
    if tuple(tuple(row) for row in coupling) == VERIFIED_J:
        raise ValueError("public rule duplicated the verified coupling matrix")
    if tuple(field) in (VERIFIED_PRIMARY_H, VERIFIED_REUSE_H):
        raise ValueError("public rule duplicated a verified field vector")
    return {
        "coupling_matrix_J": coupling,
        "field_vector_h": field,
        "generation_rule": (
            "For each upper-triangle edge and field coordinate, SHA-256 the public "
            "seed, coefficient label, and zero-based index; interpret the first eight "
            "digest bytes as an unsigned big-endian integer and reduce modulo the "
            "prospectively declared coefficient list. Mirror J exactly and set its "
            "diagonal to exact zero. No oracle or waveform result participates."
        ),
        "h_coefficient_values": list(H_VALUES),
        "j_coefficient_values": list(J_VALUES),
        "public_seed": PUBLIC_SEED,
        "site_count": SITE_COUNT,
    }


def custody_document() -> dict[str, Any]:
    instance = instance_document()
    return {
        "freeze_order": {
            "expected_optimum_observed_before_freeze": False,
            "mechanism_parameter_tuned_after_instance_generation": False,
            "native_evolution_run_before_freeze": False,
            "oracle_consulted_before_freeze": False,
        },
        "freezer_source_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "held_out_instance": instance,
        "held_out_instance_sha256": sha256_bytes(canonical_bytes(instance)),
        "predecessor_machine": {
            "claim_ceiling": (
                "BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY"
            ),
            "decision": "CATALYTIC_WAVEFORM_ISING_COMPUTATION_VERIFIED",
            "source_bytes": PREDECESSOR_SOURCE_BYTES,
            "source_sha256": PREDECESSOR_SOURCE_SHA256,
        },
        "schema": SCHEMA,
        "starting_head": STARTING_HEAD,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> None:
    write_atomic(OUTPUT_PATH, canonical_bytes(custody_document()))


def verify() -> None:
    expected = canonical_bytes(custody_document())
    if OUTPUT_PATH.read_bytes() != expected:
        raise ValueError("committed held-out custody does not reproduce")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    if args.command == "build":
        build()
    else:
        verify()
    document = custody_document()
    print(
        json.dumps(
            {
                "held_out_instance_sha256": document["held_out_instance_sha256"],
                "schema": document["schema"],
                "status": "FROZEN_BEFORE_EXECUTION_AND_ORACLE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
