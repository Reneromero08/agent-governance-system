"""Build the deterministic prospective .holo corpus without executing it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from phase_path_engine import (
    DEFAULT_PHASE_MODULI,
    RESTORATION_MAX,
    HoloSource,
    canonical_bytes,
    compile_holo,
    engine_fingerprint,
    sha256_bytes,
)


HERE = Path(__file__).resolve().parent
PROGRAM_DIR = HERE / "programs"
CONTRACT_PATH = HERE / "PROSPECTIVE_CONTRACT.json"
RESIDUE_MODULUS = 31
SIZES = (16, 32, 64, 128, 256)
FAMILIES = (
    "phase_path_alpha",
    "phase_path_beta",
    "phase_path_gamma",
    "phase_path_delta",
)


def deterministic_values(label: str, count: int) -> tuple[int, ...]:
    output: list[int] = []
    counter = 0
    while len(output) < count:
        digest = hashlib.sha256(f"{label}:{counter}".encode("utf-8")).digest()
        output.extend(1 + byte % (RESIDUE_MODULUS - 1) for byte in digest)
        counter += 1
    return tuple(output[:count])


def source_for(family: str, steps: int) -> HoloSource:
    target_digest = hashlib.sha256(
        f"{family}:{steps}:target".encode("utf-8")
    ).digest()
    return HoloSource(
        name=f"{family}_{steps}",
        residue_modulus=RESIDUE_MODULUS,
        phase_moduli=DEFAULT_PHASE_MODULI,
        weights=deterministic_values(f"{family}:{steps}", steps),
        target_residue=target_digest[0] % RESIDUE_MODULUS,
        max_steps=steps,
    )


def build() -> dict[str, Any]:
    PROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for size in SIZES:
        for family in FAMILIES:
            source = source_for(family, size)
            compiled = compile_holo(source)
            relative_path = f"programs/{source.name}.holo"
            payload = canonical_bytes(source.document())
            (HERE / relative_path).write_bytes(payload)
            entries.append(
                {
                    "family": family,
                    "holo_byte_count": len(payload),
                    "holo_sha256": sha256_bytes(payload),
                    "path": relative_path,
                    "program_sha256": compiled.program_sha256,
                    "steps": size,
                }
            )
    ordered_batch_sha256 = sha256_bytes(canonical_bytes(entries))
    source_paths = (
        "phase_path_engine.py",
        "prospective_builder.py",
        "prospective_runner.py",
    )
    source_hashes = {
        path: sha256_bytes((HERE / path).read_bytes()) for path in source_paths
    }
    contract = {
        "authority": {
            "mission": "CAT_CAS_UNBOUNDED_COMPUTE_V1",
            "physical_contact": False,
            "starting_remote_head": (
                "ebbf1e64ccffb23d2d801ff147c75ab927da7ff4"
            ),
        },
        "batch": {
            "case_count": len(entries),
            "entries": entries,
            "families": list(FAMILIES),
            "ordered_batch_sha256": ordered_batch_sha256,
            "sizes": list(SIZES),
        },
        "claim_ceiling": (
            "BOUNDED_SOFTWARE_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_ONLY"
        ),
        "engine_fingerprint": engine_fingerprint(),
        "experiment": "CAT_CAS_COMPACT_TOROIDAL_PATH_SUM_V1",
        "frozen_before_raw_execution": True,
        "mechanism": {
            "classical_path_materialization": False,
            "compiler_executes_target": False,
            "complete_path_modes": 0,
            "native_state": "relative phase roots on a product torus",
            "operator": "global triangular torus path shear",
            "program_derived_inverse": True,
            "result_latch_survives_uncompute": True,
        },
        "promotion_criteria": {
            "actual_restored_carrier_reused": True,
            "all_boundaries_valid": True,
            "all_cases_external_exact": True,
            "all_cases_required": len(entries),
            "all_controls_pass": True,
            "gamma_grows_with_size_in_every_family": True,
            "gamma_greater_than_one_from_size": 16,
            "history_factor_count": 0,
            "maximum_restoration_error": RESTORATION_MAX,
            "no_hidden_path_materialization": True,
            "zero_uninterpretable": True,
        },
        "schema": "cat_cas.toroidal_path_sum.contract.v1",
        "source_hashes": source_hashes,
    }
    return contract


def write_contract(contract: dict[str, Any]) -> None:
    CONTRACT_PATH.write_bytes(canonical_bytes(contract))


if __name__ == "__main__":
    frozen = build()
    write_contract(frozen)
    print(
        json.dumps(
            {
                "cases": frozen["batch"]["case_count"],
                "engine_fingerprint": frozen["engine_fingerprint"],
                "ordered_batch_sha256": frozen["batch"][
                    "ordered_batch_sha256"
                ],
            },
            sort_keys=True,
        )
    )
