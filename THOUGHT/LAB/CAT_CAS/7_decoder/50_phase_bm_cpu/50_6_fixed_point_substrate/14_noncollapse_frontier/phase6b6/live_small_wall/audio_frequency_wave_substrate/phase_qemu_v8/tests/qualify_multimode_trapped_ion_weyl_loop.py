#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M266 trapped-ion SDF digital twin."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "multimode_trapped_ion_weyl_loop.py"
REFERENCE = PACKAGE / "tests" / "multimode_trapped_ion_weyl_loop_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP.json"
REFERENCE_SEAL = (
    PACKAGE
    / "evidence"
    / "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_SEPARATE_REFERENCE.json"
)

CLAIM = (
    "THREE_MODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_NULLSPACE_PULSES_CLOSE_ALL_"
    "NOMINAL_MODE_DISPLACEMENTS_AND_IMPLEMENT_TWO_DISTINCT_ZZ_PHASE_PROGRAMS_"
    "ON_ONE_LOGICAL_MULTIMODE_BACKING_WHILE_DECLARED_NONZERO_MARKOVIAN_HEATING_"
    "MONOTONICALLY_BREAKS_EXACT_INITIAL_MODE_STATE_RETURN_WITHOUT_RECOOLING"
)
CEILING = (
    "DETERMINISTIC_COMPLEX128_FLOAT64_GAUSSIAN_MOMENT_SOFTWARE_DIGITAL_TWIN_"
    "WITH_DECLARED_LINEAR_HARMONIC_STATE_DEPENDENT_FORCE_AND_MARKOVIAN_ADDITIVE_"
    "HEATING_LAWS_NO_PHYSICAL_ION_CUSTODY_AND_DIRECT_COMPILED_CLIENT_CHANNEL_SHADOW"
)
RESTORATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "NOMINAL_ZERO_HEATING_LOGICAL_GAUSSIAN_MODE_RETURN_ONLY_WITH_NONZERO_HEATING_"
    "EXACT_SAME_MODE_RETURN_REJECTED_AND_FRESH_MODE_SWAP_OR_RECOOLING_CLASSIFIED_"
    "AS_EXTERNAL_RESET"
)
DISPOSITION = (
    "DIRECT_COMPILED_ZZ_AND_DEPHASING_CHANNEL_SHADOW_OMITS_THE_THREE_MODE_"
    "CONTROLLER_LOOP_WHILE_HEATING_PREVENTS_CATALYTIC_RETURN_SO_NO_RESOURCE_"
    "ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
SUCCESSOR = (
    "CONDITIONAL_GAUSSIAN_CLOSED_LOOP_FORWARD_SHADOW_AND_IRREVERSIBLE_DIFFUSION_"
    "NO_RETURN"
)

EXPECTED_HASHES = {
    PRODUCTION: "c2b181ce6dc6689731e651745e5ae76708d86163ffffbfedb88453704c284928",
    REFERENCE: "3916e4b4a68da5641d7bac78b96195d3e6131a770f94b693ea7a0e3c23ab58a7",
    CONTRACT: "785c5c4bc3ab4ec4d7abc037088dc23f4ce51fb45b40e2fb7b4149bcacc92603",
    FINDINGS: "31cb6732c9fa8eff0fa6b84fa0ebc624ce3819e950915ff77ef04d69e5016220",
}

PULSE_A = (
    875322.2363528529,
    -1017217.5496960702,
    1149417.5349432244,
    -128702.55695608277,
    -128702.55695606946,
    1149417.5349432132,
    -1017217.549696064,
    875322.2363528487,
)
PULSE_B = (
    -229497.22240917346,
    725694.6296712102,
    -1293755.9826773852,
    1628861.1574510091,
    -1628861.1574510091,
    1293755.9826774026,
    -725694.6296712208,
    229497.22240918515,
)
PROGRAMS = (
    ("A_Z0_Z1_MINUS_PI_OVER_8", (0, 1), -math.pi / 8.0, PULSE_A),
    ("B_Z1_Z2_MINUS_PI_OVER_6", (1, 2), -math.pi / 6.0, PULSE_B),
)
PRODUCTION_FIXTURES = {
    "vacuum_product",
    "thermal_product_nbar_0p05_0p10_0p20",
    "coherent_product_alpha_0p30_plus_0p15i_minus_0p20_plus_0p10i_0p10_minus_0p25i",
    "squeezed_mode0_r0p35_phi0p20_with_thermal_spectators",
    "tmsv_bus0_inert_reference_r0p30",
}
REFERENCE_FIXTURES = {
    "vacuum",
    "thermal",
    "coherent",
    "squeezed_mode0_r0p35_phi0p20_with_thermal_spectators",
    "tmsv_bus0_inert_reference_r0p30",
}
BOUNDARY_NAMES = {
    "X0",
    "X1",
    "X2",
    "Y0Z1",
    "Z0Y1",
    "Z1Y2",
    "X0X1",
    "X1X2",
}
IDEAL_BOUNDARY = {
    "X0": 1.0 / math.sqrt(2.0),
    "X1": 1.0 / (2.0 * math.sqrt(2.0)),
    "X2": 0.5,
    "Y0Z1": 1.0 / math.sqrt(2.0),
    "Z0Y1": 1.0 / (2.0 * math.sqrt(2.0)),
    "Z1Y2": math.sqrt(3.0) / 2.0,
    "X0X1": 0.5,
    "X1X2": 1.0 / math.sqrt(2.0),
}
PRODUCTION_CHECKS = {
    "program_a_exact_self_area_phase",
    "program_b_exact_self_area_phase",
    "program_a_all_mode_closure",
    "program_b_all_mode_closure",
    "program_a_midloop_witness",
    "program_b_midloop_witness",
    "program_a_heating_exposure_witness",
    "program_b_heating_exposure_witness",
    "nominal_direct_shadow_parity",
    "heated_direct_channel_shadow_parity",
    "heated_primary_detectably_differs_from_ideal",
    "heated_primary_rejects_exact_mode_return",
    "heating_covariance_drift_monotone",
    "same_logical_backing_a_then_b",
    "spectator_omission_detected",
    "omitted_final_segment_detected",
    "segment_area_error_detected",
    "detuning_error_detected",
    "opposite_orientation_flips_phase",
    "zero_force_has_zero_area",
    "frozen_control_receipts",
    "tmsv_reference_detects_heating",
    "no_physical_claim",
    "m257_intact",
}
REFERENCE_CHECKS = {
    "exact_closure_max_le_5e_minus_12",
    "target_phase_abs_error_le_2e_minus_12",
    "rk4_strictly_converges_1024_2048_4096",
    "finest_rk4_phase_abs_error_le_2e_minus_10",
    "intra_segment_self_area_omission_detected",
    "heated_client_channel_differs",
    "heated_client_purity_decreases",
    "heated_client_channel_is_positive",
    "heating_rejects_exact_mode_return",
    "pulse_omission_detected",
    "spectator_omission_detected",
    "detuning_drift_detected",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(
    first: float,
    second: float,
    *,
    absolute: float,
    relative: float = 0.0,
    label: str,
) -> None:
    require(
        math.isclose(float(first), float(second), abs_tol=absolute, rel_tol=relative),
        f"{label}: {first!r} != {second!r}",
    )


def close_sequence(
    first: Sequence[float],
    second: Sequence[float],
    *,
    absolute: float,
    label: str,
) -> None:
    require(len(first) == len(second), f"{label} length changed")
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        close(left, right, absolute=absolute, label=f"{label}[{index}]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate(script: Path) -> bytes:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=PACKAGE,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode == 0, f"{script.name} exited {result.returncode}")
    require(result.stderr == b"", f"unexpected stderr from {script.name}")
    require(result.stdout.endswith(b"\n"), f"unterminated JSON from {script.name}")
    return result.stdout


def scrub_nonclaim_diagnostics(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                0.0
                if key.endswith("_not_used_for_claim")
                else scrub_nonclaim_diagnostics(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_nonclaim_diagnostics(item) for item in value]
    return value


def seal_bytes(raw: bytes) -> bytes:
    normalized = scrub_nonclaim_diagnostics(json.loads(raw))
    return (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def guest_visible_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "guest_visible" and item is True:
                paths.append(path)
            paths.extend(guest_visible_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(guest_visible_paths(item, f"{prefix}[{index}]"))
    return paths


def source_and_document_audit() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing dependency: {path.name}")
        require(sha256(path) == expected, f"dependency changed: {path.name}")

    production_source = PRODUCTION.read_text(encoding="utf-8")
    for anchor in (
        "return float((delta * dt - math.sin(delta * dt)) / (delta * delta))",
        "phase += couplings[mode_index] ** 2 * _self_area(",
        "attenuation = math.exp(-exposure)",
        "fullscale = 1.8e6",
        '"full_joint_heated_gaussian_process_retained": False',
        '"frequency_eigenvector_table_physical_trap_mapping_established": False',
    ):
        require(anchor in production_source, f"production source anchor missing: {anchor}")

    reference_source = REFERENCE.read_text(encoding="utf-8")
    require(PRODUCTION.name not in reference_source, "reference names production")
    require(
        "import multimode_trapped_ion_weyl_loop" not in reference_source,
        "reference imports production",
    )
    for anchor in (
        "self_area = q**2 / detuning**2 * (angle - np.sin(angle))",
        "k1 = derivative(instant, alpha, omega)",
        "k4 = derivative(instant + dt, alpha + dt * k3[0], omega)",
        "exponent = float(np.sum(GAMMA * exposure * difference**2))",
    ):
        require(anchor in reference_source, f"reference source anchor missing: {anchor}")

    for document in (CONTRACT, FINDINGS):
        text = document.read_text(encoding="utf-8")
        for value, label in (
            (CLAIM, "claim"),
            (CEILING, "ceiling"),
            (RESTORATION, "restoration class"),
            (RESTORATION_SCOPE, "restoration scope"),
            (DISPOSITION, "resource disposition"),
            (SUCCESSOR, "successor"),
        ):
            require(value in text, f"{label} missing from {document.name}")
        lowered = text.lower().replace("_", " ")
        for required in ("m257", "heating", "physical", "synthetic", "collective"):
            require(required in lowered, f"{required} scope missing from {document.name}")
        require(
            "direct compiled" in lowered or ("direct" in lowered and "shadow" in lowered),
            f"direct-shadow scope missing from {document.name}",
        )

    contract = CONTRACT.read_text(encoding="utf-8")
    for receipt in (
        "0.7125305458876734",
        "0.8557657738719043",
        "0.008551211055284244",
        "0.004805936264387415",
        "0.004249949772572",
        "0.991962603457778",
    ):
        require(receipt in contract, f"contract receipt missing: {receipt}")
    findings = FINDINGS.read_text(encoding="utf-8")
    for receipt in ("[0.0036, 0.0072, 0.0144]", "0.0233306665142683", "3.2097743958157e-29"):
        require(receipt in findings, f"findings receipt missing: {receipt}")


def metadata_audit(production: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    require(production["schema"] == "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_RESULT_V1", "production schema changed")
    require(reference["schema"] == "PHASE_QEMU_V8_TRAPPED_ION_WEYL_LOOP_SEPARATE_REFERENCE_V1", "reference schema changed")
    require(production["milestone"] == reference["milestone"] == "M266", "milestone changed")
    require(production["status"] == "PASS_INTERNAL_SELF_CHECK_HEATED_NONRESTORATION", "production internal status changed")
    require(production["source_self_assertion"] == "PASS_INTERNAL_CONSISTENCY_ONLY", "production self assertion changed")
    require("INDEPENDENT" not in production["status"], "production self-check promoted to independent")
    require(reference["reference_id"] == "M266_MULTIMODE_TRAPPED_ION_WEYL_LOOP_SEPARATE_REFERENCE_V1", "reference id changed")
    require(reference["status"] == "PASS_INDEPENDENT_CONTINUUM_REFERENCE_SELF_CHECK", "reference status changed")
    require(reference["reference_self_assertion"] == reference["status"], "reference self assertion differs")
    require(production["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "embedded production hash differs")
    require(reference["source_sha256"] == EXPECTED_HASHES[REFERENCE], "embedded reference hash differs")

    for key, expected in (
        ("claim", CLAIM),
        ("restoration_classification", RESTORATION),
        ("restoration_scope", RESTORATION_SCOPE),
        ("resource_disposition", DISPOSITION),
        ("next_mechanism", SUCCESSOR),
    ):
        require(production[key] == expected, f"production {key} changed")
        require(reference[key] == expected, f"reference {key} changed")
    require(production["claim_ceiling"] == CEILING, "production ceiling changed")
    require(reference["ceiling"] == CEILING, "reference ceiling changed")
    require(not production["terminal"] and not reference["terminal"], "long-term goal terminated")

    require(set(production["checks"]) == PRODUCTION_CHECKS, "production check schema changed")
    require(len(production["checks"]) == 24 and all(production["checks"].values()), "production 24/24 checks failed")
    require(set(reference["checks"]) == REFERENCE_CHECKS, "reference check schema changed")
    require(len(reference["checks"]) == 12 and all(reference["checks"].values()), "reference 12/12 checks failed")


def descriptor_and_program_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    geometry = production["hardware_geometry"]
    require(geometry["ions"] == 3 and geometry["physical_modes"] == 3, "production device size changed")
    require(geometry["mode_names"] == ["COM", "STRETCH", "WOBBLE"], "mode names changed")
    close_sequence(geometry["mode_frequencies_hz"], (1_900_000.0, 1_918_000.0, 1_930_000.0), absolute=0.0, label="mode frequencies")
    close(geometry["drive_frequency_hz"], 1_911_000.0, absolute=0.0, label="drive frequency")
    close_sequence(geometry["signed_drive_minus_mode_detunings_rad_s"], tuple(2.0 * math.pi * value for value in (11_000.0, -7_000.0, -19_000.0)), absolute=1e-10, label="detunings")
    close_sequence(geometry["heating_quanta_per_s"], (15.0, 30.0, 60.0), absolute=0.0, label="heating rates")
    close(geometry["program_duration_s"], 120e-6, absolute=0.0, label="duration")
    require(geometry["segments_per_program"] == 8, "segment count changed")
    close(geometry["segment_duration_s"], 15e-6, absolute=1e-18, label="segment duration")
    require(len(geometry["eta_matrix"]) == 3 and all(len(row) == 3 for row in geometry["eta_matrix"]), "eta geometry changed")

    descriptor = reference["public_descriptor"]
    require(descriptor["segment_count"] == 8, "reference segment count changed")
    close(descriptor["duration_seconds"], 120e-6, absolute=0.0, label="reference duration")
    close_sequence(descriptor["detuning_hz"], (11_000.0, -7_000.0, -19_000.0), absolute=0.0, label="reference detunings")
    close_sequence(descriptor["heating_quanta_per_second"], (15.0, 30.0, 60.0), absolute=0.0, label="reference heating")
    require(descriptor["covariance_and_occupation_law"] == "V_DOT_M=GAMMA_M*I2_NBAR_DOT_M=GAMMA_M", "reference covariance law changed")
    require(descriptor["coherence_law"] == "EXP_MINUS_SUM_M_GAMMA_M_INTEGRAL_ABS_DELTA_BETA_M_T_SQUARED_DT", "reference coherence law changed")

    production_programs = production["programs"]
    reference_programs = reference["program_records"]
    reference_descriptors = descriptor["programs"]
    require(len(production_programs) == len(reference_programs) == len(reference_descriptors) == 2, "program count changed")
    for index, (name, pair, target, pulses) in enumerate(PROGRAMS):
        prod = production_programs[index]
        ref = reference_programs[index]
        ref_descriptor = reference_descriptors[index]
        require(prod["name"] == ref["name"] == ref_descriptor["name"] == name, f"program {index} name changed")
        require(tuple(prod["target_pair"]) == tuple(ref["ions"]) == tuple(ref_descriptor["ions"]) == pair, f"program {name} pair changed")
        close(prod["target_theta"], target, absolute=2e-15, label=f"{name} production target")
        close(ref["target_zz_exponent_coefficient_rad"], target, absolute=2e-15, label=f"{name} reference target")
        close(ref_descriptor["target_zz_exponent_coefficient_rad"], target, absolute=2e-15, label=f"{name} descriptor target")
        close_sequence(prod["amplitudes_rad_s"], pulses, absolute=1e-9, label=f"{name} production pulses")
        close_sequence(ref_descriptor["pulses_rad_per_second"], pulses, absolute=1e-9, label=f"{name} reference pulses")
        close(prod["walsh_zz_phase"], target, absolute=2e-12, label=f"{name} production phase")
        close(ref["exact_segment_recurrence"]["zz_exponent_coefficient_rad"], target, absolute=2e-12, label=f"{name} reference phase")
        require(prod["phase_error"] <= 2e-12, f"{name} phase error")
        require(prod["maximum_final_branch_displacement"] <= 1e-12, f"{name} closure")
        require(ref["exact_segment_recurrence"]["maximum_base_displacement_abs"] <= 5e-12, f"{name} reference closure")
        require(prod["self_area_included"], f"{name} self-area flag lost")
        require(prod["cross_segment_area_convention"] == "IM_INCREMENT_TIMES_CONJUGATE_PRIOR_DISPLACEMENT", f"{name} area convention changed")
        require(ref["intra_segment_self_area_control"]["abs_phase_error_when_self_area_is_omitted"] >= 1e-3, f"{name} self-area omission not detected")

        errors = []
        for grid in (1024, 2048, 4096):
            record = ref["rk4_continuum_grids"][str(grid)]
            require(record["steps"] == grid, f"{name} RK grid changed")
            errors.append(abs(float(record["zz_exponent_coefficient_rad"]) - float(ref["exact_segment_recurrence"]["zz_exponent_coefficient_rad"])))
        require(errors[2] < errors[1] < errors[0], f"{name} RK4 phase does not strictly converge")
        require(errors[-1] <= 2e-10, f"{name} RK4 finest error too large")
        close(prod["maximum_heating_exposure"], ref["maximum_heating_dephasing_exponent"], absolute=2e-12, label=f"{name} heating exposure parity")

    dot = sum(left * right for left, right in zip(PULSE_A, PULSE_B, strict=True))
    norm_a = math.sqrt(sum(value * value for value in PULSE_A))
    norm_b = math.sqrt(sum(value * value for value in PULSE_B))
    require(abs(dot / (norm_a * norm_b)) <= 1e-12, "A/B waveforms are not distinct orthogonal controls")
    close(production_programs[0]["maximum_midloop_branch_displacement"], 0.7125305458876734, absolute=2e-12, label="A midloop")
    close(production_programs[1]["maximum_midloop_branch_displacement"], 0.8557657738719043, absolute=2e-12, label="B midloop")
    close(production_programs[0]["maximum_heating_exposure"], 0.008551211055284244, absolute=2e-14, label="A exposure")
    close(production_programs[1]["maximum_heating_exposure"], 0.004805936264387415, absolute=2e-14, label="B exposure")


def boundary_heating_and_fixture_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    client = production["client"]
    released = client["combined_heated_released_boundary"]
    require(set(released) == {"boundary", "guest_visible"}, "released boundary wrapper changed")
    require(released["guest_visible"] is True, "combined heated boundary is not guest visible")
    require(client["guest_visible_boundary_path"] == "client.combined_heated_released_boundary.boundary", "guest boundary path changed")
    require(guest_visible_paths(production) == ["client.combined_heated_released_boundary.guest_visible"], "guest visibility escaped the one nested boundary")
    require(set(released["boundary"]) == BOUNDARY_NAMES, "released boundary field set changed")
    require(set(client["combined_nominal_boundary"]) == BOUNDARY_NAMES, "nominal boundary field set changed")

    direct = reference["direct_compiled_client_channel"]
    require(set(direct["ideal_boundary"]) == set(direct["heated_boundary"]) == BOUNDARY_NAMES, "reference boundary field set changed")
    for name, exact in IDEAL_BOUNDARY.items():
        close(client["combined_nominal_boundary"][name], exact, absolute=2e-12, label=f"production ideal {name}")
        close(direct["ideal_boundary"][name], exact, absolute=2e-12, label=f"reference ideal {name}")
        close(released["boundary"][name], direct["heated_boundary"][name], absolute=2e-12, label=f"heated boundary {name}")
    require(client["combined_nominal_trace_distance_to_direct_ideal"] <= 1e-12, "nominal direct parity failed")
    require(client["combined_heated_trace_distance_to_direct_heated_channel"] <= 1e-12, "heated direct parity failed")
    close(client["combined_heated_trace_distance_to_direct_ideal"], direct["heated_client_trace_distance_to_ideal"], absolute=2e-12, label="heated trace distance parity")
    close(client["heated_density_integrity"]["purity"], direct["heated_client_purity"], absolute=2e-12, label="heated purity parity")
    close(client["combined_heated_trace_distance_to_direct_ideal"], 0.004249949772572, absolute=2e-12, label="heated trace distance receipt")
    close(client["heated_density_integrity"]["purity"], 0.991962603457778, absolute=2e-12, label="heated purity receipt")
    require(client["heated_density_integrity"]["minimum_eigenvalue"] >= -1e-12, "heated client is not positive")

    fixtures = production["same_logical_multimode_backing_heated_primary"]
    require(len(fixtures) == 5, "production fixture count changed")
    require({record["fixture"] for record in fixtures} == PRODUCTION_FIXTURES, "production fixture set changed")
    reference_ledger = reference["gaussian_fixture_ledger"]
    require(set(reference_ledger["named_initial_fixtures"]) == REFERENCE_FIXTURES, "reference fixture set changed")
    expected_delta_a = (0.0018, 0.0036, 0.0072)
    expected_delta_ab = (0.0036, 0.0072, 0.0144)
    tmsv_count = 0
    for record in fixtures:
        close_sequence(record["heating_delta_n_after_a"], expected_delta_a, absolute=1e-15, label=f"{record['fixture']} delta A")
        close_sequence(record["heating_delta_n_after_ab"], expected_delta_ab, absolute=1e-15, label=f"{record['fixture']} delta AB")
        require(record["mean_return_max_abs"] <= 1e-15, "mode means failed nominal closure")
        require(not record["after_a_exact_initial_mode_state_return"] and not record["after_ab_exact_initial_mode_state_return"], "heated mode return promoted")
        require(record["covariance_drift_strictly_monotone"], "heating drift is not monotone")
        close(record["after_ab_covariance_frobenius_drift"], 0.023330666514268, absolute=2e-15, label=f"{record['fixture']} covariance drift")
        custody = record["logical_custody"]
        require(custody["allocation_object_unchanged"] and custody["allocation_base_pointer_unchanged"], "logical allocation changed")
        require(custody["generation_sequence"] == [0, 1, 2], "generation sequence changed")
        require(custody["carrier_supply_count"] == 1, "carrier supplied more than once")
        require(custody["snapshot_count"] == custody["reload_count"] == custody["recooling_count"] == custody["carrier_replacement_count"] == 0, "accepted path used reset or replacement")
        require(not custody["physical_same_mode_custody_established"], "logical custody promoted to physical custody")
        coherence = record["tmsv_reference_coherence"]
        if coherence is not None:
            tmsv_count += 1
            require(record["fixture"] == "tmsv_bus0_inert_reference_r0p30", "coherence record attached to wrong fixture")
            require(coherence["reference_is_inert"], "reference became active")
            require(not coherence["exact_reference_coherence_return"], "heated reference return promoted")
            require(coherence["after_ab_entanglement_infidelity"] > 0.0, "TMSV coherence did not detect heating")
    require(tmsv_count == 1, "TMSV coherence applicability changed")

    declared = reference_ledger["declared_heating_after_a_then_b"]
    close_sequence(declared["delta_n_by_mode"], expected_delta_ab, absolute=1e-15, label="reference delta AB")
    close(declared["covariance_drift_frobenius"], 0.023330666514268298, absolute=2e-15, label="reference covariance drift")
    require(not declared["exact_initial_mode_state_return"], "reference heated return promoted")
    nominal = reference_ledger["nominal_zero_heating"]
    require(nominal["means_return"] and nominal["covariances_return"] and nominal["bus_reference_cross_covariance_returns"], "nominal Gaussian return failed")
    require(nominal["logical_gaussian_return_only"], "nominal scope caveat lost")


def controls_resources_and_nonclaims_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    controls = production["controls"]
    expected_controls = {
        "omitted_final_segment",
        "target_middle_mode_only_spectator_omission",
        "detuning_plus_250hz_all_modes",
        "amplitude_quantization_12bit_fullscale_1p8e6",
        "segment_index3_plus_1_percent",
        "opposite_phase_orientation",
        "zero_force_zero_area",
        "snapshot_reload",
        "recooling_external_reset",
        "fresh_mode_swap_carrier_replacement",
        "tmsv_reference_coherence",
        "nominal_reference",
    }
    require(set(controls) == expected_controls, "production control set changed")
    omitted = controls["omitted_final_segment"]
    require(omitted["segments_executed"] == omitted["program_b"]["segments_executed"] == 7, "omission schedule changed")
    require(omitted["maximum_closure"] >= 0.4 and omitted["program_b"]["maximum_closure"] >= 0.1, "omitted segment not detected")
    require(controls["target_middle_mode_only_spectator_omission"]["maximum_closure"] >= 1.0, "spectator omission not detected")
    detuned = controls["detuning_plus_250hz_all_modes"]
    require(detuned["maximum_closure"] >= 0.07 and detuned["program_b"]["maximum_closure"] >= 0.02, "detuning control collapsed")
    quantized = controls["amplitude_quantization_12bit_fullscale_1p8e6"]
    require(quantized["maximum_closure"] >= 3e-4 and quantized["program_b"]["maximum_closure"] >= 3e-4, "12-bit quantization not detected")
    area = controls["segment_index3_plus_1_percent"]
    require(area["maximum_closure"] >= 3e-4 and area["program_b"]["maximum_closure"] >= 3e-4, "segment area error not detected")
    opposite = controls["opposite_phase_orientation"]
    close(opposite["walsh_theta"], math.pi / 8.0, absolute=2e-12, label="opposite A")
    close(opposite["program_b"]["walsh_theta"], math.pi / 6.0, absolute=2e-12, label="opposite B")
    zero = controls["zero_force_zero_area"]
    require(zero["maximum_closure"] == zero["program_b"]["maximum_closure"] == 0.0, "zero-force closure changed")
    require(zero["walsh_theta"] == zero["program_b"]["walsh_theta"] == 0.0, "zero-force phase appeared")
    snapshot = controls["snapshot_reload"]
    require(snapshot["classification"] == "SNAPSHOT_RELOAD" and snapshot["snapshot_count"] == snapshot["reload_count"] == 1, "snapshot sham changed")
    require(not snapshot["accepted_as_catalytic_restoration"], "snapshot promoted")
    recooling = controls["recooling_external_reset"]
    require(recooling["classification"] == "EXTERNAL_RESET" and recooling["recooling_count"] == 1, "recooling sham changed")
    require(not recooling["accepted_as_same_mode_reuse"], "recooling promoted")
    swap = controls["fresh_mode_swap_carrier_replacement"]
    require(swap["classification"] == "CARRIER_REPLACEMENT" and swap["carrier_replacement_count"] == 1, "fresh swap sham changed")
    require(not swap["accepted_as_same_mode_reuse"], "fresh swap promoted")
    require(controls["tmsv_reference_coherence"]["checked_in_named_fixture"] and not controls["tmsv_reference_coherence"]["bus_marginal_only_is_sufficient"], "TMSV control changed")

    ref_controls = reference["named_controls"]
    require(set(ref_controls) == {PROGRAMS[0][0], PROGRAMS[1][0], "architecture_shams"}, "reference controls changed")
    for name, _, _, _ in PROGRAMS:
        record = ref_controls[name]
        require(record["last_segment_omission"]["maximum_base_displacement_abs"] >= 1.0, f"reference {name} omission collapsed")
        require(record["mode2_spectator_omission"]["abs_phase_error"] >= 1e-2, f"reference {name} spectator control collapsed")
        require(record["detuning_drift_hz_50_minus75_plus100"]["maximum_base_displacement_abs"] >= 1e-2, f"reference {name} detuning control collapsed")
    shams = ref_controls["architecture_shams"]
    require(shams["recooling"] == "EXTERNAL_RESET_NOT_CATALYTIC_RESTORATION", "reference recooling classification changed")
    require(shams["snapshot_reload"] == "FORBIDDEN_HISTORY_BASED_RESET", "reference snapshot classification changed")

    model = production["model"]
    require(model["state"] == "THREE_SYNTHETIC_COLLECTIVE_HARMONIC_MODES_PLUS_THREE_QUBIT_CLIENT_GAUSSIAN_MOMENTS_AND_EXACT_CLIENT_CHANNEL", "model state changed")
    require(not model["frequency_eigenvector_table_physical_trap_mapping_established"], "synthetic geometry promoted to trap mapping")
    require(not model["full_joint_heated_gaussian_process_retained"], "full joint process falsely retained")
    require(model["heating_coherence_law"] == "EXP_MINUS_SUM_M_GAMMA_M_INTEGRAL_ABS_DELTA_BETA_M_SQUARED_DT", "production heating law changed")

    ledger = production["resource_ledger"]
    for key, expected in (
        ("physical_modes", 3),
        ("client_qubits", 3),
        ("programs", 2),
        ("force_segments", 16),
        ("single_qubit_basis_rotation_pulses", 8),
        ("public_amplitude_descriptor_values", 16),
        ("amplitude_descriptor_bytes", 128),
        ("closure_real_constraint_rank", 6),
        ("closure_nullspace_dimension", 2),
        ("controller_quantization_control_bits", 12),
        ("retained_dynamic_trajectory_history", 0),
        ("retained_public_schedule_entries", 16),
        ("privileged_initial_gaussian_baselines", 5),
    ):
        require(ledger[key] == expected, f"resource {key} changed")
    close(ledger["controller_quantization_fullscale_rad_s"], 1.8e6, absolute=0.0, label="quantization fullscale")
    require(ledger["basis_rotation_accounting"] == "FOUR_PER_PROGRAM_FOR_PHYSICAL_XX_TO_EFFECTIVE_ZZ_MAPPING", "basis rotation accounting changed")
    close_sequence(ledger["heating_delta_n_after_two_programs"], (0.0036, 0.0072, 0.0144), absolute=1e-15, label="resource delta n")
    close(ledger["maximum_named_covariance_frobenius_drift"], 0.023330666514268, absolute=2e-15, label="resource covariance drift")
    close(ledger["added_heating_energy_joules_after_two_programs"], 3.20977439581571e-29, absolute=1e-41, relative=1e-12, label="heating energy")
    require(ledger["sum_of_per_mode_peak_coherent_energy_upper_bound_joules"] > 0.0, "coherent energy upper bound absent")
    require(not ledger["verifier_baselines_readable_by_dynamics"], "dynamics gained verifier baselines")
    require(ledger["software_forward_shadow_omits_modes_and_return"], "forward-shadow caveat lost")

    comparator = production["strongest_honest_classical_comparator"]
    require(comparator["bus_coordinates_retained"] == comparator["controller_segments_executed"] == 0, "direct shadow retained bus execution")
    require(comparator["exposure_compilation_force_segments_processed"] == 16, "exposure compilation not charged")
    require(comparator["direct_zz_phase_gates"] == 2 and comparator["complex_channel_descriptor_cells"] == 128, "direct shadow ledger changed")
    require(comparator["matches_nominal_client"] and comparator["matches_heated_client"], "direct shadow parity claim changed")
    require(not comparator["restoration_stage_executed"] and not comparator["m257_escape_established"], "direct shadow restoration/M257 changed")

    for key in (
        "physical_execution",
        "physical_same_mode_custody",
        "physical_restoration",
        "phase_native_client_architecture",
        "unbounded_compute",
        "resource_advantage",
    ):
        require(not production[key], f"production promoted {key}")
    require(not production.get("replace_the_bit_with_pi_established", False), "production promoted bit replacement")
    require(production["m257_intact"], "production weakened M257")

    claims = reference["claims"]
    for key in (
        "nonzero_heating_exact_initial_mode_state_return",
        "same_backing_restoration",
        "physical_execution",
        "physical_same_mode_custody",
        "physical_restoration",
        "fresh_mode_swap_is_restoration",
        "recooling_is_catalytic_restoration",
        "computational_advantage",
        "m257_escape",
        "unbounded_compute",
        "bit_replaced_with_pi",
    ):
        require(not claims[key], f"reference promoted {key}")
    require(claims["nominal_zero_heating_logical_gaussian_mode_return"], "nominal logical return lost")
    authority = reference["architecture_authority"]
    require(not authority["reference_is_physical_evidence"], "reference promoted to physical evidence")
    require(authority["equal_access_direct_compiled_shadow_is_allowed"], "direct comparator access removed")
    require(not authority["retained_dynamic_trajectory_history"] and not authority["retained_inverse_history"], "reference retained hidden history")

    fock = production["finite_fock_nonuniformity_witness"]
    require(not fock["numerically_executed"], "finite Fock witness execution invented")
    require(not fock["top_fock_or_uniform_arbitrary_state_claim"], "uniform finite-Fock claim promoted")
    require(fock["energy_constrained_reference_required_for_ANY_future_fock_promotion"], "future Fock guardrail lost")


def seal_audit(production_bytes: bytes, reference_bytes: bytes, write: bool) -> None:
    production_seal = seal_bytes(production_bytes)
    reference_seal = seal_bytes(reference_bytes)
    if write:
        PRODUCTION_SEAL.parent.mkdir(parents=True, exist_ok=True)
        PRODUCTION_SEAL.write_bytes(production_seal)
        REFERENCE_SEAL.write_bytes(reference_seal)
    require(PRODUCTION_SEAL.is_file(), "production seal missing")
    require(REFERENCE_SEAL.is_file(), "reference seal missing")
    require(PRODUCTION_SEAL.read_bytes() == production_seal, "production seal drift")
    require(REFERENCE_SEAL.read_bytes() == reference_seal, "reference seal drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-seals", action="store_true")
    arguments = parser.parse_args()

    source_and_document_audit()
    production_bytes = regenerate(PRODUCTION)
    reference_bytes = regenerate(REFERENCE)
    production = json.loads(production_bytes)
    reference = json.loads(reference_bytes)

    metadata_audit(production, reference)
    descriptor_and_program_audit(production, reference)
    boundary_heating_and_fixture_audit(production, reference)
    controls_resources_and_nonclaims_audit(production, reference)
    seal_audit(production_bytes, reference_bytes, arguments.write_seals)

    print(
        "PASS_STRICT_SCOPE M266_MULTIMODE_TRAPPED_ION_WEYL_LOOP "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "RESTORATION=NO_RESTORATION_CLAIM "
        "SCOPE=NOMINAL_CLOSURE_HEATED_RETURN_REJECTED "
        "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
