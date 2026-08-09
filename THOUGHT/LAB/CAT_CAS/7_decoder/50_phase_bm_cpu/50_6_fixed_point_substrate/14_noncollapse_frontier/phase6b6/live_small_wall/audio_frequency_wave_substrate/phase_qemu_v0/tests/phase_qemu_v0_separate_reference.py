#!/usr/bin/env python3
"""Independent fixed-point/reference and boundary audit for Phase-QEMU V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


Q30_ONE = 1 << 30
DECAY_Q30 = 1_063_004_406
PREPARE_STEPS = 256
CYCLES_PER_STEP = 64
RINGDOWN_STEPS = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q30_mul(left: int, right: int) -> int:
    product = left * right
    if product >= 0:
        return product >> 30
    return -((-product + (1 << 30) - 1) >> 30)


def energy(i: int, q: int) -> int:
    return (i * i + q * q) >> 30


@dataclass
class ReferenceState:
    carrier_present: bool
    source_i: int = 0
    source_q: int = 0
    carrier_i: int = 0
    carrier_q: int = 0
    detector_i: int = 0
    detector_q: int = 0
    virtual_cycles: int = 0
    dissipated_energy: int = 0

    def prepare(self, phase_arm: int) -> None:
        sign = -1 if phase_arm else 1
        self.source_i = sign * Q30_ONE
        for _ in range(PREPARE_STEPS):
            if self.carrier_present:
                self.carrier_i += (self.source_i - self.carrier_i) >> 3
                self.carrier_q += (self.source_q - self.carrier_q) >> 3
            self.detector_i += (self.carrier_i - self.detector_i) >> 2
            self.detector_q += (self.carrier_q - self.detector_q) >> 2
            self.virtual_cycles += CYCLES_PER_STEP

    def isolate(self) -> None:
        if self.carrier_present:
            self.carrier_q += self.source_i >> 12

    def evolve(self, steps: int) -> None:
        for _ in range(steps):
            before = energy(self.carrier_i, self.carrier_q)
            old_i, old_q = self.carrier_i, self.carrier_q
            rotated_i = old_i - (old_q >> 12)
            rotated_q = old_q + (old_i >> 12)
            self.carrier_i = q30_mul(rotated_i, DECAY_Q30)
            self.carrier_q = q30_mul(rotated_q, DECAY_Q30)
            after = energy(self.carrier_i, self.carrier_q)
            if before > after:
                self.dissipated_energy += before - after
            target_i = self.carrier_i + (self.source_i >> 20)
            target_q = self.carrier_q + (self.source_q >> 20)
            self.detector_i += (target_i - self.detector_i) >> 2
            self.detector_q += (target_q - self.detector_q) >> 2
            self.virtual_cycles += CYCLES_PER_STEP

    def boundary(self) -> dict[str, int]:
        return {
            "i": self.detector_i,
            "q": self.detector_q,
            "energy_q30": energy(self.detector_i, self.detector_q),
            "barrier_receipt": 8,
            "virtual_cycles": self.virtual_cycles,
        }


def simulate(phase_arm: int, carrier_present: bool) -> tuple[dict[str, int], int]:
    state = ReferenceState(carrier_present=carrier_present)
    state.prepare(phase_arm)
    state.isolate()
    state.evolve(RINGDOWN_STEPS)
    return state.boundary(), state.dissipated_energy


def audit_source(path: Path) -> dict[str, bool | str]:
    source = path.read_text(encoding="utf-8")
    read_body_match = re.search(
        r"static uint64_t phase_mmio_read.*?\n}\n\nstatic void phase_mmio_write",
        source,
        re.S,
    )
    if read_body_match is None:
        raise AssertionError("could not isolate MMIO read implementation")
    read_body = read_body_match.group(0)
    hidden_names = (
        "source_i",
        "source_q",
        "carrier_i",
        "carrier_q",
        "detector_i",
        "detector_q",
        "dissipated_energy_q30",
    )
    hidden_register_absent = all(name not in read_body for name in hidden_names)
    vmstate_hidden_domains_present = all(
        f"VMSTATE_{kind}({name}" in source
        for kind, name in (
            ("INT64", "source_i"),
            ("INT64", "carrier_i"),
            ("INT64", "detector_i"),
            ("UINT64", "dissipated_energy_q30"),
        )
    )
    return {
        "source_sha256": sha256(path),
        "backend_ops_interface_present": "typedef struct PhaseBackendOps" in source,
        "p0_is_one_backend_not_frontend_contract": "static const PhaseBackendOps p0_ops" in source,
        "hidden_process_coordinates_absent_from_mmio_reads": hidden_register_absent,
        "boundary_lock_sentinel_present": "s->response_ready ? s->boundary_i : UINT64_MAX" in source,
        "native_p0_inverse_explicitly_unavailable": "ERR_P0_INVERSE_UNAVAILABLE" in source,
        "native_p0_restoration_explicitly_unavailable": "ERR_P0_RESTORATION_UNAVAILABLE" in source,
        "qemu_vmstate_includes_hidden_process_domains": vmstate_hidden_domains_present,
        "canonical_model_reinitialization_is_separate_command": (
            "CMD_CANONICAL_MODEL_REINITIALIZE" in source
        ),
        "owner_program_custody_capability_is_narrowly_named": (
            "CAP_OWNER_PROGRAM_CUSTODY_LEASE" in source
            and "CAP_TYPED_LEASE" not in source
        ),
        "migration_post_load_validates_topology_and_machine_state": (
            ".post_load = phase_qemu_v0_post_load" in source
            and "configured_carrier_present" in source
        ),
        "no_answer_table_tokens": all(
            token not in source
            for token in ("truth_table", "assignment_table", "precomputed_answer")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--device-source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    observed = json.loads(args.result.read_text(encoding="utf-8"))
    expected_0, dissipation_0 = simulate(0, True)
    expected_pi, dissipation_pi = simulate(1, True)
    expected_removed, dissipation_removed = simulate(0, False)

    source_audit = audit_source(args.device_source)
    controls = {
        "primary_0_exact_reference_parity": observed["primary_0"] == expected_0,
        "primary_pi_exact_reference_parity": observed["primary_pi"] == expected_pi,
        "removed_carrier_exact_reference_parity": observed["removed_carrier"] == expected_removed,
        "reference_antipodal_quadratures_within_fixed_point_quantization": (
            abs(expected_pi["i"] + expected_0["i"]) <= 16
            and abs(expected_pi["q"] + expected_0["q"]) <= 128
        ),
        "reference_matched_energy_within_fixed_point_quantization": abs(
            expected_pi["energy_q30"] - expected_0["energy_q30"]
        ) <= 32,
        "reference_removed_carrier_strictly_smaller": expected_removed["energy_q30"] < expected_0["energy_q30"],
        "reference_carrier_dissipates_positive_energy": dissipation_0 > 0 and dissipation_pi > 0,
        "removed_carrier_has_zero_mechanical_dissipation": dissipation_removed == 0,
        "all_source_boundary_audits_pass": all(
            value for key, value in source_audit.items() if key != "source_sha256"
        ),
        "production_controls_all_pass": all(observed["controls"].values()),
        "p0_capabilities_omit_inverse_restore_reuse": observed["device"]["capabilities"] & 0xE0 == 0,
        "canonical_model_reinitialization_not_restoration": (
            observed["classification"]["restoration_classification"]
            == "NO_RESTORATION_CLAIM"
            and observed["classification"]["model_reinitialization_classification"]
            == "NO_RESTORATION_CLAIM"
        ),
        "migration_stream_no_smuggle_not_claimed": (
            observed["boundary_security_scope"]
            ["migration_and_snapshot_streams_are_trusted_backend_state"]
            is True
            and observed["boundary_security_scope"]
            ["host_or_migration_stream_no_smuggle_enforcement_established"]
            is False
        ),
        "m257_not_escaped_by_deterministic_model": (
            observed["claim_limits"]["distinct_phase_resource"] is False
            and observed["claim_limits"]["computational_advantage"] is False
        ),
    }
    failed = [key for key, value in controls.items() if not value]
    if failed:
        raise AssertionError(f"failed independent controls: {failed}")

    output = {
        "schema": "phase-qemu-v0-separate-reference-v1",
        "production_result_sha256": sha256(args.result),
        "device_source_audit": source_audit,
        "expected_primary_0": expected_0,
        "expected_primary_pi": expected_pi,
        "expected_removed_carrier": expected_removed,
        "dissipated_energy_q30": {
            "primary_0": dissipation_0,
            "primary_pi": dissipation_pi,
            "removed_carrier": dissipation_removed,
        },
        "controls": controls,
        "verification_classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "migration_topology_guard_verification_level": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "claim_ceiling": "DETERMINISTIC_FIXED_POINT_QEMU_PCI_MODEL_OF_SELECTED_P0_PROCESS_GEOMETRY_ONLY",
        "next_obstruction": "P0_RINGDOWN_IS_DISSIPATIVE_AND_HAS_NO_NATIVE_INVERT_RESTORE_REUSE_LAW_WHILE_THE_IDENTICAL_FIXED_POINT_STATE_RECURRENCE_IS_AN_EQUAL_ACCESS_CLASSICAL_SHADOW",
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
