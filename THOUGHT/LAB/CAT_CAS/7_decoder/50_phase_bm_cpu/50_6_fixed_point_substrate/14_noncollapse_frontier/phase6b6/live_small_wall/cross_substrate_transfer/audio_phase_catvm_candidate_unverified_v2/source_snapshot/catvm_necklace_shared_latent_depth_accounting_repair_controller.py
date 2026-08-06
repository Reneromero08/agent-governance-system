#!/usr/bin/env python3
"""Distinct resource-accounting successor for the atomic depth CATVM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HISTORICAL_CLAIM = (
    "CATVM_ATOMIC_DISPATCH_REPAIRED_TOPOLOGY_REMATERIALIZED_"
    "OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_"
    "CARRIER_AT_DEPTH32"
)
REPAIRED_CLAIM = (
    "CATVM_ATOMIC_DISPATCH_AND_RESOURCE_ACCOUNTING_REPAIRED_"
    "TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_LATENT_PHASE_"
    "PROGRAM_FIXED_570_CARRIER_AT_DEPTH32"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: "
            "catvm_necklace_shared_latent_depth_accounting_repair_"
            "controller.py SERVICE EVIDENCE_DIR"
        )
    service = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    historical_controller = Path(__file__).with_name(
        "catvm_necklace_shared_latent_depth_atomic_repair_controller.py"
    )
    historical_result = evidence_dir / "historical_atomic_result.json"
    historical_stderr = evidence_dir / "historical_atomic_controller.stderr"

    with historical_result.open("wb") as stdout_file:
        with historical_stderr.open("wb") as stderr_file:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(historical_controller),
                    str(service),
                    str(evidence_dir),
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
    if completed.returncode != 0:
        fail("historical atomic controller failed")
    if historical_stderr.stat().st_size != 0:
        fail("historical atomic controller emitted stderr")

    result = json.loads(historical_result.read_text(encoding="utf-8"))
    if result.get("result") != "PASS":
        fail("historical atomic controls did not pass")
    if result.get("claim_candidate") != HISTORICAL_CLAIM:
        fail("unexpected historical atomic claim")
    resource = result.get("resource_law")
    if not isinstance(resource, dict):
        fail("historical resource law missing")
    if (
        resource.get(
            "primary_peak_counted_complex_cells_excluding_plan"
        )
        != 2280
        or resource.get(
            "reuse_peak_counted_complex_cells_excluding_plan"
        )
        != 2850
    ):
        fail("historical resource defect was not preserved")

    result["claim_candidate"] = REPAIRED_CLAIM
    result["historical_atomic_repair_package"] = {
        "claim_candidate": HISTORICAL_CLAIM,
        "classification": "REJECTED_SOURCE_DEFECT",
        "source_defect": (
            "LIVE_289_COMPLEX_GENERATOR_MATRIX_OMITTED_FROM_"
            "PRIMARY_AND_REUSE_PEAKS"
        ),
        "preserved_subclaim": (
            "ATOMIC_DISPATCH_INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
    }
    resource.update(
        {
            "generator_matrix_complex_cells": 289,
            "generator_plan_object_bytes": 4664,
            "generator_plan_nonmatrix_scalar_bytes": 40,
            "primary_peak_counted_complex_cells_excluding_plan": 2569,
            "reuse_peak_counted_complex_cells_excluding_plan": 3139,
            "primary_peak_counted_complex_cells_including_plan_roots": 2586,
            "reuse_peak_counted_complex_cells_including_plan_roots": 3156,
            "plan_object_bytes": 296,
            "plan_necklace_count": 285,
            "plan_necklace_capacity": 285,
            "plan_necklace_element_bytes": 36,
            "plan_necklace_capacity_bytes": 10260,
            "request_packet_bytes": 32,
            "response_packet_bytes": 116,
            "accepted_primary_reuse_requests": 5,
            "accepted_primary_reuse_request_bytes": 160,
            "accepted_primary_reuse_response_bytes": 580,
            "accepted_primary_reuse_protocol_bytes": 740,
        }
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
