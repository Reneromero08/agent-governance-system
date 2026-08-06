#!/usr/bin/env python3
"""Independent protocol attack for shared-latent module-owner binding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import catvm_necklace_shared_latent_protocol as protocol
from catvm_necklace_shared_latent_controller import (
    Service,
    fail,
    require,
)


WRONG_MODULE_OWNER = 13


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: "
            "catvm_necklace_shared_latent_owner_repair_controller.py "
            "SERVICE EVIDENCE_DIR"
        )
    executable = Path(sys.argv[1]).resolve()
    evidence_dir = Path(sys.argv[2]).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    service = Service(executable, evidence_dir, "module-owner")
    service.initialize()
    _, wrong_module_owner = service.call(WRONG_MODULE_OWNER)
    require(
        wrong_module_owner,
        protocol.STATUS_DENIED,
        "wrong nonzero module owner",
    )
    if int(wrong_module_owner["flags"]) & protocol.BOUNDARY_VALID:
        fail("module-owner denial exposed a boundary")

    _, stage = service.call(protocol.BEGIN)
    _, final = service.call(protocol.CONTINUE)
    _, reuse = service.call(protocol.REUSE)
    require(stage, protocol.STATUS_OK, "post-attack stage")
    require(final, protocol.STATUS_OK, "post-attack final")
    require(reuse, protocol.STATUS_OK, "post-attack reuse")
    if int(stage["flags"]) & protocol.BOUNDARY_VALID:
        fail("post-attack stage exposed a boundary")
    if not int(stage["flags"]) & protocol.STAGE_RESIDENT:
        fail("post-attack latent state was not resident")
    if (
        int(final["flags"])
        & (protocol.BOUNDARY_VALID | protocol.RESTORED)
        != (protocol.BOUNDARY_VALID | protocol.RESTORED)
    ):
        fail("post-attack final response preceded restoration")
    if float(final["restoration_error"]) > 6e-11:
        fail("post-attack primary restoration failed")
    if float(reuse["restoration_error"]) > 6e-11:
        fail("post-attack reuse restoration failed")
    if float(reuse["norm_error"]) > 6e-11:
        fail("post-attack fresh/restored reuse parity failed")
    service.stop()

    result = {
        "claim_candidate": (
            "CATVM_ENFORCED_OWNER_BOUND_COHERENT_SHARED_LATENT_"
            "OBSERVATION_PORT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER"
        ),
        "result": "PASS",
        "claim_ceiling": (
            "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
            "NONCE_DERIVED_OUTER_LEASE_EXACT_GENERATION_GRID17_FOUR_"
            "EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_"
            "NECKLACES_570_COMPLEX_CELLS_TWO_CELL_LATENT_FIBER_FIXED_"
            "FOUR_MODULE_PRIMARY_THREE_MODULE_REUSE_STATIC_OWNER_"
            "0X4C415431_SEVEN_BIN_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
        ),
        "module_port_owner": {
            "expected": 0x4C415431,
            "wrong_nonzero_owner": "DENIED_BEFORE_CARRIER_OPERATION",
            "boundary_values_released": 0,
        },
        "transaction_custody": {
            "exact_outer_lease": True,
            "exact_outer_generation": True,
            "post_attack_stage_resident": True,
            "post_attack_final_generation": final["generation"],
            "post_attack_reuse_generation": reuse["generation"],
        },
        "primary_restoration_error": final["restoration_error"],
        "reuse_restoration_error": reuse["restoration_error"],
        "fresh_restored_reuse_boundary_error": reuse["norm_error"],
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "predecessor_source_defect_preserved": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
