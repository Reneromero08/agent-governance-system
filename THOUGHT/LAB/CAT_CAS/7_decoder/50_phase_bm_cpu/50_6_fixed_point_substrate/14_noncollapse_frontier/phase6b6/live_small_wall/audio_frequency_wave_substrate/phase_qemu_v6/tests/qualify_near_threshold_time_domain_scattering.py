#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M264 time-domain obstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "near_threshold_time_domain_scattering.py"
REFERENCE = PACKAGE / "tests" / "near_threshold_time_domain_scattering_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING.json"
REFERENCE_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V6_TIME_DOMAIN_SCATTERING_SEPARATE_REFERENCE.json"

CLAIM = (
    "FROZEN_L641_SIGMA50_NEAR_THRESHOLD_TIME_DOMAIN_SCATTERING_EXECUTES_"
    "TRANSIENT_BORROW_DRAIN_AND_COMPLETE_RETURNED_DENSITY_HANDOFF_BUT_THE_"
    "PUBLIC_T120_ADIABATIC_PREPARATION_AND_FINITE_PACKET_T340_RETURN_FAIL_"
    "DECLARED_1E_MINUS_7_MATCHED_FREE_TARGET_TRACE_DISTANCE_GATES"
)
CEILING = (
    "FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_SINGLE_PROBE_L641_FOUR_SPIN_"
    "TIME_DOMAIN_MODEL_WITH_RETURNED_TARGET_DENSITY_REMATERIALIZATION_AND_"
    "NO_SAME_BACKING_OR_PHYSICAL_RESTORATION"
)
DISPOSITION = (
    "STRICT_PREPARATION_AND_FINITE_PACKET_RESTORATION_OBSTRUCTION_RETAINS_"
    "REAL_TRANSIENT_INTERACTION_DRAIN_AND_APPROXIMATE_FUNCTIONAL_HANDOFF_BUT_"
    "REQUIRES_A_CHANGED_RETURN_PREPARATION_LAW_NOT_POST_HOC_FIXTURE_TUNING"
)
NEXT = (
    "RESIDENT_OPEN_DRAIN_OR_ECHO_RETURN_WITH_PAID_GROUND_STATE_SUPPLY_"
    "PREDECLARED_FINITE_PACKET_ERROR_AND_SAME_BACKING_TARGET_CUSTODY"
)
RETURN_SCOPE = "FAILED_DECLARED_TIME_DOMAIN_RETURN_OR_REUSE_THRESHOLDS"

EXPECTED_HASHES = {
    PRODUCTION: "fc065c1814b2a9914df3a6547d3cb59e01feb172ff5784d3f62c3ba76af19f27",
    REFERENCE: "d2a488a391012013f549faa17cf2ad39772c8388438ecdd7c874ddd7df40580e",
    CONTRACT: "30eb3932a976e7ba2dbbea7c990c93ae5371097e94ccbffc35d14502346c9099",
    FINDINGS: "cedb3d6f0d2e4876b25c2a442117631bada4962a5ebf920382f34be1e7e057f2",
}

FALSE_RETURN_CHECKS = {"query_a_target_return", "query_b_reuse_target_return"}
GLOBAL_NOT_ESTABLISHED = {
    "QEMU_DEVICE_EXECUTION",
    "PHYSICAL_SCATTERING_OR_OBSERVATION",
    "SOURCE_ISOLATION_IN_HARDWARE",
    "SAME_BACKING_RESTORATION_OR_REUSE",
    "PERMANENT_RESTORATION_ON_A_FINITE_LEAD",
    "ASYMPTOTIC_OR_GROWING_TARGET_RESOURCE_LAW",
    "TENSOR_NETWORK_CROSSOVER",
    "DISTINCT_PHASE_RESOURCE",
    "COMPUTATIONAL_ADVANTAGE",
    "M257_ESCAPE",
    "SMALL_WALL_CROSSING",
    "UNBOUNDED_COMPUTE",
    "PHYSICAL_BIT_REPLACEMENT_WITH_PI",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def complex_value(record: Mapping[str, object]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


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


def _scrub_nondeterministic_wall_time(value: Any) -> Any:
    """Remove only diagnostic wall clocks before durable byte comparison.

    Production intentionally serializes measured ``*_wall_seconds_not_used_for_claim``
    values.  Raw output bytes therefore cannot be stable across regenerations.  The
    scientific seal retains every key while replacing only those explicitly
    nonclaim-bearing measurements with zero.
    """

    if isinstance(value, dict):
        return {
            key: (
                0.0
                if key.endswith("wall_seconds_not_used_for_claim")
                else _scrub_nondeterministic_wall_time(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_nondeterministic_wall_time(item) for item in value]
    return value


def seal_bytes(raw: bytes) -> bytes:
    parsed = json.loads(raw)
    normalized = _scrub_nondeterministic_wall_time(parsed)
    return (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def source_and_document_audit() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing dependency: {path.name}")
        require(sha256(path) == expected, f"dependency changed: {path.name}")

    reference_text = REFERENCE.read_text(encoding="utf-8")
    require(PRODUCTION.name not in reference_text, "reference names production source")
    require("import near_threshold_time_domain_scattering" not in reference_text, "reference imports production")

    production_text = PRODUCTION.read_text(encoding="utf-8")
    reuse_start = production_text.index("query_b_reuse, query_b_reuse_primary")
    reuse_end = production_text.index("# Clean-B executes", reuse_start)
    reuse_call = production_text[reuse_start:reuse_end]
    require(
        "query_b_reuse_continuation" in reuse_call
        and "continue_to_delay_control=True" in reuse_call,
        "production B-reuse does not execute the T370 continuation",
    )
    above_start = production_text.index("above_threshold, above_primary")
    above_end = production_text.index("# Static boundary potential", above_start)
    require(
        "exact_ground_column" in production_text[above_start:above_end],
        "production above-threshold control does not use the exact ground target",
    )
    acceptance_start = production_text.index("acceptance_checks = {")
    acceptance_end = production_text.index("all_pass =", acceptance_start)
    acceptance_source = production_text[acceptance_start:acceptance_end]
    for required in (
        '"query_b_transient_interaction": query_b_reuse[',
        '"query_b_contact_drained": query_b_reuse[',
        '"query_b_detector_flux_370": min(',
        "b_reuse_centroid_370[\"incoming_flux\"]",
        "b_reuse_centroid_370[\"outgoing_flux\"]",
    ):
        require(required in acceptance_source, f"B-reuse source gate missing: {required}")

    for path in (CONTRACT, FINDINGS):
        text = path.read_text(encoding="utf-8")
        for value, label in (
            (CLAIM, "claim"),
            (CEILING, "ceiling"),
            (DISPOSITION, "disposition"),
            (NEXT, "successor"),
        ):
            require(value in text, f"{label} missing from {path.name}")
        lowered = text.lower()
        require("no_restoration_claim" in lowered, f"no-restoration class missing from {path.name}")
        require("m257" in lowered, f"M257 guardrail missing from {path.name}")
        require("same-backing" in lowered, f"same-backing ceiling missing from {path.name}")
        require("rematerialization" in lowered, f"rematerialization disclosure missing from {path.name}")
        require(
            ("t480" in lowered or "tprep=480" in lowered)
            and ("diagnostic" in lowered or "control" in lowered),
            f"T480 diagnostic scope missing from {path.name}",
        )

    findings = FINDINGS.read_text(encoding="utf-8")
    require("8.038931410866084e-05" in findings, "query A obstruction value missing")
    require("2.8167715493073277e-05" in findings, "query B obstruction value missing")
    require("strict obstruction, not a promotion" in findings.lower(), "obstruction status missing")


def metadata_and_obstruction(production: dict[str, Any], reference: dict[str, Any]) -> None:
    for evidence, label in ((production, "production"), (reference, "reference")):
        require(evidence["milestone"] == "M264", f"{label} milestone changed")
        require(evidence["source_self_assertion"] == "SOURCE_SELF_CHECK_PASS", f"{label} source self-check failed")
        require(evidence["claim"] == CLAIM, f"{label} claim changed")
        require(evidence["claim_ceiling"] == CEILING, f"{label} ceiling changed")
        require(evidence["disposition"] == DISPOSITION, f"{label} disposition changed")
        require(evidence["next_mechanism"] == NEXT, f"{label} successor changed")
        restoration = evidence["restoration"]
        require(restoration["classification"] == "NO_RESTORATION_CLAIM", f"{label} restoration promoted")
        require(restoration["scope"] == RETURN_SCOPE, f"{label} restoration scope changed")
        require(restoration["executed_time_domain_target_excursion_and_drain"], f"{label} lost executed transient")
        require(restoration["returned_state_rematerialization_used"], f"{label} hides rematerialization")
        require(restoration["generic_target_state_repreparation_via_density_rematerialization_used"], f"{label} hides repreparation")
        for key in (
            "same_backing_restoration_established",
            "same_backing_reuse_established",
            "physical_restoration_established",
            "permanent_restoration_established",
        ):
            require(not restoration[key], f"{label} promoted {key}")

    require(
        production["verification_scope"]
        == {
            "science": "PACKAGE_SELF_REVIEW_PENDING_SEPARATE_REFERENCE_PARITY",
            "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource": "PACKAGE_SELF_REVIEW",
        },
        "production verification scope changed",
    )
    require(
        reference["verification_scope"]
        == {
            "science": "SEPARATE_REFERENCE_PARITY",
            "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource": "PACKAGE_SELF_REVIEW",
        },
        "reference verification scope changed",
    )
    require(production["backend"]["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "production self hash changed")
    require(not production["acceptance_all"], "failed fixture was promoted")
    require(not production["restoration_promotion_all"], "restoration promotion returned")
    require(production["execution_integrity_all"], "production execution integrity failed")
    require(all(production["execution_integrity_checks"].values()), "an execution-integrity check failed")
    failed = {key for key, value in production["acceptance_checks"].items() if not value}
    require(failed == FALSE_RETURN_CHECKS, f"unexpected production failures: {sorted(failed)}")

    qualification = reference["package_qualification"]
    require(qualification["result"] == "FAIL_MATCHED_FREE_TARGET_RETURN_GATE", "reference result changed")
    require(qualification["promotion"] == "DENIED", "reference promoted package")
    require(qualification["source_self_check_pass_does_not_imply_package_pass"], "reference conflates source and package pass")
    assertions = reference["assertions"]
    require(not assertions["matched_free_target_return_gate_pass"], "reference return gate promoted")
    require(assertions["matched_free_target_return_obstruction_detected"], "reference obstruction missing")
    require(assertions["package_qualification"] == "FAIL_MATCHED_FREE_TARGET_RETURN_GATE", "reference package status changed")


def geometry_locality_and_packets(production: dict[str, Any], reference: dict[str, Any]) -> None:
    configuration = production["public_configuration"]
    lead = configuration["lead"]
    require(
        (lead["length"], lead["packet_center"], lead["packet_sigma"])
        == (641, 320, 50),
        "production lead geometry changed",
    )
    require(lead["detector_bond"] == [192, 193], "production detector changed")
    require(lead["contact_site"] == 0, "production contact changed")
    require((lead["primary_stop_time"], lead["delay_control_stop_time"], lead["observation_dt"]) == (340.0, 370.0, 2.0), "production timing changed")
    require(configuration["target"]["dimension"] == 16, "production target dimension changed")
    close(configuration["target"]["gap"], reference["target"]["gap"], absolute=1.0e-12, label="target gap parity")

    fixture = reference["public_fixture"]
    require(
        (
            fixture["lead_length"],
            fixture["joint_state_dimension"],
            fixture["packet_center_site"],
            fixture["packet_sigma_x"],
            fixture["detector_bond"],
            fixture["final_time"],
            fixture["observation_grid_dt"],
        )
        == (641, 10256, 320, 50, [192, 193], 340.0, 2.0),
        "reference public fixture changed",
    )
    require(fixture["source_off_after_packet_initialization"], "reference source replayed")

    locality = production["locality_fail_closed_evidence"]
    require(locality["contact_projector_nnz"] == 1, "production projector nnz changed")
    require(locality["contact_projector_coordinate"] == [0, 0], "production projector coordinate changed")
    require(locality["coupling_support_lead_sites"] == [0], "production coupling support widened")
    require(locality["coupling_localized_at_site_zero"], "production coupling is not local")
    require((locality["total_dimension"], locality["total_hamiltonian_nnz"], locality["coupling_term_nnz"]) == (10256, 71760, 16), "production sparse geometry changed")
    reference_projector = fixture["site_zero_projector"]
    require(reference_projector["nnz"] == 1 and reference_projector["only_coordinate"] == [0, 0], "reference projector changed")
    require(reference_projector["support_assertion_pass"], "reference projector assertion failed")

    production_packets = production["packet_preflight"]
    reference_packets = reference["finite_bandwidth"]
    for production_key, reference_key in (
        ("query_A", "query_A"),
        ("query_B", "query_B"),
        ("above_threshold_control", "above_threshold_control"),
    ):
        case = production_packets[production_key]
        oracle = reference_packets[reference_key]
        close(case["k"], oracle["momentum"], absolute=1.0e-12, label=f"{production_key} momentum")
        close(case["nominal_energy"], oracle["central_energy"], absolute=1.0e-12, label=f"{production_key} energy")
        close(case["discrete_open_lead_mean_energy"], oracle["dst_mean_energy"], absolute=2.0e-12, label=f"{production_key} mean energy")
        close(case["discrete_open_lead_energy_standard_deviation"], oracle["dst_std_energy"], absolute=2.0e-12, label=f"{production_key} energy width")
        close(case["above_target_gap_spectral_weight"], oracle["dst_weight_at_or_above_target_gap"], absolute=1.0e-12, label=f"{production_key} spectral tail")
        close(
            case["nominal_group_velocity"],
            -2.0 * math.sin(case["k"]),
            absolute=1.0e-15,
            label=f"{production_key} leftward group velocity",
        )
        require(case["nominal_group_velocity"] < 0.0, f"{production_key} packet is not directed toward contact")
    require(production_packets["query_A"]["effectively_subgap_at_declared_tail"], "query A tail failed")
    require(production_packets["query_B"]["effectively_subgap_at_declared_tail"], "query B tail failed")
    require(not production_packets["above_threshold_control"]["effectively_subgap_at_declared_tail"], "above-threshold control became subgap")
    require(reference_packets["query_A"]["dst_weight_at_or_above_target_gap"] < 1.0e-9, "reference A tail failed")
    require(reference_packets["query_B"]["dst_weight_at_or_above_target_gap"] < 1.0e-9, "reference B tail failed")
    require(reference_packets["above_threshold_control"]["dst_weight_at_or_above_target_gap"] > 0.9, "reference above-threshold control changed")
    require(production_packets["finite_bandwidth_packets_are_not_exact_energy_eigenstates"], "finite bandwidth caveat lost")


def preparation_and_return_parity(production: dict[str, Any], reference: dict[str, Any]) -> None:
    preparation = production["preparation"]
    oracle_preparation = reference["preparation"]
    require((preparation["duration"], preparation["steps"]) == (120.0, 480), "production preparation changed")
    require((oracle_preparation["duration"], oracle_preparation["public_midpoint_steps"]) == (120.0, 480), "reference preparation changed")
    close(preparation["ground_infidelity"], oracle_preparation["midpoint_ground_infidelity"], absolute=1.0e-12, label="T120 preparation infidelity")
    close(
        preparation["T480_nonclaim_bearing_diagnostic"]["ground_infidelity"],
        oracle_preparation["T480_duration_control"]["ground_infidelity"],
        absolute=1.0e-13,
        label="T480 preparation infidelity",
    )
    require(not preparation["T480_nonclaim_bearing_diagnostic"]["used_as_production_query_input"], "T480 repaired production")
    require(preparation["T480_nonclaim_bearing_diagnostic"]["may_not_repair_the_frozen_T120_outcome"], "T480 scope widened")
    require(preparation["exact_ground_was_not_injected_into_production_queries"], "exact ground entered production")
    require(oracle_preparation["exact_ground_injection_is_control_only"], "reference exact ground promoted")
    require(oracle_preparation["T480_duration_control"]["role"] == "DECOMPOSITION_CONTROL_ONLY_NOT_A_REPAIR_OR_PROMOTION", "reference T480 role changed")

    production_a = production["query_A"]
    production_b_clean = production["query_B_clean"]
    production_b_reuse = production["query_B_reuse"]
    reference_a = reference["time_domain"]["query_A"]
    reference_b_clean = reference["time_domain"]["query_B_clean"]
    reference_b_reuse = reference["executed_distinct_energy_reuse"]["query_B_returned_controlled_mixed_density"]

    comparisons = (
        (production_a, reference_a, 5.0e-5, "query A"),
        (production_b_clean, reference_b_clean, 2.0e-4, "query B clean"),
        (production_b_reuse, reference_b_reuse, 2.0e-4, "query B returned"),
    )
    for case, oracle, transient_tolerance, label in comparisons:
        close(case["maximum_transient_target_excitation"], oracle["max_transient_target_excitation"], absolute=transient_tolerance, label=f"{label} transient")
        close(case["final_target_excitation"], oracle["final_target_excitation"], absolute=5.0e-10, label=f"{label} final excitation")
        close(case["final_contact_probability"], oracle["final_site_zero_probability"], absolute=1.0e-13, label=f"{label} contact drain")
        require(case["final_contact_probability"] < 1.0e-8, f"{label} contact did not drain")
        require(case["final_near_target_probability"] < 1.0e-7, f"{label} near-target region did not drain")
        require(case["maximum_norm_error"] < 1.0e-9, f"{label} production norm failed")
        require(oracle["norm_error"] < 1.0e-9, f"{label} reference norm failed")

    density_thresholds = production["thresholds"]
    for case, label in (
        (production_a, "query A"),
        (production_b_reuse, "query B returned"),
    ):
        diagnostics = case["target_return"]
        require(
            diagnostics["trace_error"] <= density_thresholds["norm_or_trace_error_max"],
            f"{label} density trace gate failed",
        )
        require(
            diagnostics["hermiticity_max_abs"]
            <= density_thresholds["density_hermiticity_error_max"],
            f"{label} density Hermiticity gate failed",
        )
        require(
            diagnostics["minimum_eigenvalue"]
            >= density_thresholds["density_minimum_eigenvalue_min"],
            f"{label} density PSD gate failed",
        )
        fidelity = diagnostics["fidelity_to_interaction_picture_free_target"]
        require(0.0 <= fidelity <= 1.0, f"{label} Uhlmann fidelity is outside [0,1]")

    production_a_return = production_a["target_return"]["trace_distance_to_interaction_picture_free_target"]
    production_b_clean_return = production_b_clean["target_return"]["trace_distance_to_interaction_picture_free_target"]
    production_b_reuse_return = production_b_reuse["target_return"]["trace_distance_to_interaction_picture_free_target"]
    reference_a_return = reference_a["final_target_trace_distance_from_matched_free_target"]
    reference_b_clean_return = reference_b_clean["final_target_trace_distance_from_matched_free_target"]
    reference_b_reuse_return = reference_b_reuse["final_target_trace_distance_from_matched_free_returned_density"]
    close(production_a_return, reference_a_return, absolute=2.0e-8, label="query A return obstruction")
    close(production_b_clean_return, reference_b_clean_return, absolute=1.0e-7, label="query B clean return obstruction")
    close(production_b_reuse_return, reference_b_reuse_return, absolute=3.0e-8, label="query B returned obstruction")
    for value, label in (
        (production_a_return, "production A"),
        (production_b_reuse_return, "production B returned"),
        (reference_a_return, "reference A"),
        (reference_b_reuse_return, "reference B returned"),
    ):
        require(value > 1.0e-7, f"{label} unexpectedly passed return gate")
    close(reference["package_qualification"]["query_A_observed"], reference_a_return, absolute=1.0e-15, label="reference qualification A")
    close(reference["package_qualification"]["query_B_observed"], reference_b_clean_return, absolute=1.0e-15, label="reference qualification B")


def delays_modes_and_convergence(production: dict[str, Any], reference: dict[str, Any]) -> None:
    delays = production["delay"]
    reference_time = reference["time_domain"]
    pairs = (
        (delays["query_A_T340"]["relative_current_centroid_delay"], reference_time["query_A"]["current_centroid_delay"], 2.0e-3, "A T340 delay"),
        (delays["query_B_clean_T340"]["relative_current_centroid_delay"], reference_time["query_B_clean"]["current_centroid_delay"], 3.0e-2, "B T340 delay"),
        (delays["query_A_T370_control"]["relative_current_centroid_delay"], reference_time["T370_delay_control"]["query_A"]["current_centroid_delay"], 2.0e-3, "A T370 delay"),
        (delays["query_B_clean_T370_control"]["relative_current_centroid_delay"], reference_time["T370_delay_control"]["query_B"]["current_centroid_delay"], 3.0e-2, "B T370 delay"),
    )
    for first, second, tolerance, label in pairs:
        close(first, second, absolute=tolerance, label=label)

    production_stationary_a = delays["stationary_16_channel_query_A"]
    production_stationary_b = delays["stationary_16_channel_query_B"]
    reference_stationary_a = reference["stationary_dense_channel_reference"]["query_A"]
    reference_stationary_b = reference["stationary_dense_channel_reference"]["query_B"]
    for case, oracle, label in (
        (production_stationary_a, reference_stationary_a, "A"),
        (production_stationary_b, reference_stationary_b, "B"),
    ):
        close(case["relative_phase_radians"], oracle["relative_phase"], absolute=1.0e-10, label=f"{label} stationary phase")
        close(case["wigner_delay"], oracle["relative_wigner_delay"], absolute=1.0e-7, label=f"{label} stationary delay")
        require(case["unit_modulus_error"] < 1.0e-10, f"{label} production stationary modulus failed")
        require(oracle["unit_modulus_residual"] < 1.0e-10, f"{label} reference stationary modulus failed")
    require(production_stationary_b["wigner_delay"] - production_stationary_a["wigner_delay"] >= 5.5, "production delay ordering failed")
    require(reference_stationary_b["relative_wigner_delay"] - reference_stationary_a["relative_wigner_delay"] >= 5.5, "reference delay ordering failed")

    weighted_a = reference["stationary_dense_channel_reference"]["query_A_packet_weighted_relative_delay"]
    weighted_b = reference["stationary_dense_channel_reference"]["query_B_packet_weighted_relative_delay"]
    require(abs(delays["query_A_T370_control"]["relative_current_centroid_delay"] - weighted_a) < abs(delays["query_A_T340"]["relative_current_centroid_delay"] - weighted_a), "A T370 did not improve finite-window delay")
    require(abs(delays["query_B_clean_T370_control"]["relative_current_centroid_delay"] - weighted_b) < abs(delays["query_B_clean_T340"]["relative_current_centroid_delay"] - weighted_b), "B T370 did not improve finite-window delay")
    require(delays["T340_is_finite_window_not_a_tight_wigner_estimator"], "T340 Wigner caveat lost")
    require(production["convergence"]["T370_detector_flux_control_executed"], "production T370 control missing")
    reuse_t370 = delays["query_B_reuse_T370"]
    require(
        min(
            reuse_t370["coupled"]["incoming_flux"],
            reuse_t370["coupled"]["outgoing_flux"],
        )
        >= production["thresholds"]["detector_outgoing_flux_min"],
        "production B-reuse T370 flux failed",
    )
    require(
        abs(
            reuse_t370["relative_current_centroid_delay"]
            - delays["query_B_clean_T370_control"]["relative_current_centroid_delay"]
        )
        < 1.0e-4,
        "production B-reuse T370 delay diverged from clean-B comparator",
    )
    require(production["convergence"]["query_A_one_shot_endpoint"]["phase_aligned_endpoint_l2"] <= 1.0e-8, "one-shot/chunked endpoint failed")
    require(reference_time["step_halving_control_A"]["fine_delay_Richardson_error_estimate"] < 0.03, "reference delay convergence failed")
    require(reference_time["step_halving_control_A"]["fine_phase_Richardson_error_estimate"] < 2.0e-3, "reference phase convergence failed")
    require(reference_time["finite_lead_control_A"]["current_delay_difference"] < 1.0e-4, "reference lead-size control failed")
    for query_key in ("query_A", "query_B"):
        control = reference_time["T370_delay_control"][query_key]
        require(min(control["candidate_integrated_incoming_current"], control["candidate_integrated_outgoing_current"]) >= 0.99, f"reference {query_key} T370 flux failed")

    production_a_iq = complex_value(production["query_A"]["boundary_mode"]["matched_g0_arm_density_coherence_iq"])
    production_b_iq = complex_value(production["query_B_clean"]["boundary_mode"]["matched_g0_arm_density_coherence_iq"])
    reference_a_iq = complex_value(reference_time["query_A"]["iq"]["same_time_iq"])
    reference_b_iq = complex_value(reference_time["query_B_clean"]["iq"]["same_time_iq"])
    require(abs(production_a_iq - reference_a_iq) <= 2.0e-3, "query A IQ parity failed")
    require(abs(production_b_iq - reference_b_iq) <= 4.0e-3, "query B IQ parity failed")
    close(
        production["query_A"]["boundary_mode"]["matched_g0_arm_density_coherence_modulus_squared"],
        reference_time["query_A"]["iq"]["same_time_visibility"] ** 2,
        absolute=1.0e-5,
        label="query A mode visibility",
    )
    close(
        production["query_B_clean"]["boundary_mode"]["matched_g0_arm_density_coherence_modulus_squared"],
        reference_time["query_B_clean"]["iq"]["same_time_visibility"] ** 2,
        absolute=1.0e-4,
        label="query B mode visibility",
    )


def handoff_controls_and_ceilings(production: dict[str, Any], reference: dict[str, Any]) -> None:
    handoff = production["returned_rho_A_handoff"]
    require(handoff["all_16_spectral_components_propagated"], "production did not propagate all returned components")
    require(handoff["spectral_component_count"] == 16, "production handoff component count changed")
    require(handoff["discarded_positive_eigenvalue_weight"] == 0.0, "production discarded returned density")
    require(handoff["negative_eigenvalue_trace_clipped"] <= 1.0e-12, "production PSD clip exceeded gate")
    require(handoff["reconstruction_max_abs"] <= 1.0e-12, "production density reconstruction failed")
    require(handoff["returned_state_rematerialization_used"], "production hides rematerialization")
    require(handoff["generic_target_state_repreparation_via_density_rematerialization_used"], "production hides generic repreparation")
    require(not handoff["target_ground_reload_used"], "production reloaded ground")
    require(not handoff["same_backing_reuse_established"], "production same-backing promoted")
    require(not handoff["physical_reuse_established"], "production physical reuse promoted")

    reuse = reference["executed_distinct_energy_reuse"]
    require(reuse["returned_state_rematerialization_used"], "reference hides rematerialization")
    require(reuse["controlled_spectral_truncation_used"], "reference truncation disclosure lost")
    require(not reuse["full_returned_density_matrix_propagated"], "reference falsely claims full implementation")
    require(reuse["discarded_density_trace_limit"] == 1.0e-12, "reference discard limit changed")
    discarded = reuse["query_B_returned_controlled_mixed_density"]["discarded_input_density_trace_bound"]
    require(0.0 <= discarded <= 1.0e-12, "reference discarded trace exceeded bound")
    require(not reuse["baseline_or_exact_ground_reload_used"], "reference reloaded baseline")
    require(not reuse["same_backing"], "reference same-backing promoted")
    close(handoff["dominant_weight"], reuse["query_A_final_target_principal_weight"], absolute=1.0e-12, label="returned density dominant weight")
    close(handoff["non_dominant_trace"], reuse["query_A_nonprincipal_trace"], absolute=1.0e-12, label="returned density residual weight")
    require(reuse["clean_vs_returned_current_delay_difference"] < 1.0e-4, "returned B delay parity failed")
    require(reuse["clean_vs_returned_final_target_density_trace_distance"] < 1.0e-4, "returned B density parity failed")

    exact_ground = production["controls"]["exact_ground_injection"]
    require(exact_ground["classification"] == "ANSWER_BEARING_PREPARATION_SHAM_NOT_PRODUCTION", "exact-ground control promoted")
    require(exact_ground["query_B"]["target_return"]["trace_distance_to_interaction_picture_free_target"] < 1.0e-7, "production exact-ground diagnostic changed")
    require(reference["controls"]["exact_ground_query_B"]["final_target_trace_distance_from_matched_free_target"] < 1.0e-7, "reference exact-ground diagnostic changed")
    above_production = production["controls"]["above_threshold"]["query"]
    reference_above_control = reference["controls"]["above_threshold"]
    require(
        reference_above_control["initial_target"] == "EXACT_GROUND",
        "reference above-threshold control does not match production input",
    )
    above_reference = reference_above_control["query"]
    require(above_production["final_target_excitation"] > 0.9, "production above-threshold nonreturn lost")
    require(above_reference["final_target_excitation"] > 0.9, "reference above-threshold nonreturn lost")
    close(above_production["final_target_excitation"], above_reference["final_target_excitation"], absolute=5.0e-4, label="above-threshold excitation")
    require(above_production["target_return"]["trace_distance_to_interaction_picture_free_target"] > 0.9, "production above-threshold target returned")
    require(above_reference["final_target_trace_distance_from_matched_free_target"] > 0.9, "reference above-threshold target returned")

    resources = production["resource_accounting"]
    require((resources["target_dimension"], resources["lead_length"], resources["joint_complex_amplitudes"]) == (16, 641, 10256), "production resource geometry changed")
    require(resources["raw_joint_state_bytes_complex128"] == 164096, "production raw state bytes changed")
    require(resources["full_171_state_history_bytes_if_retained"] == 28060416, "production history bytes changed")
    require(not resources["history_retained_by_production"], "production retained full history")
    require(resources["query_B_returned_density_component_count"] == 16, "production returned rank changed")
    require(resources["query_B_raw_block_state_bytes"] == 2625536, "production block bytes changed")
    require(resources["strongest_honest_classical_comparator"] == "IDENTICAL_O_16L_SPARSE_COORDINATE_TIME_EVOLUTION", "classical comparator changed")
    require(resources["one_particle_lead_cut_mps_bond_upper_bound"] == 17, "MPS ceiling changed")
    require(not resources["tensor_network_crossover_established"], "tensor-network crossover promoted")
    require(not resources["resource_advantage_established"], "resource advantage promoted")
    required_uninstrumented = {
        "PYTHON_NUMPY_SCIPY_OBJECT_AND_ALLOCATOR_OVERHEAD",
        "SCIPY_INTERNAL_KRYLOV_BASIS_PAYLOAD_AND_EXACT_PEAK_RSS",
        "DESCRIPTOR_COPIES_AND_CONTROLLER_DETECTOR_OBJECT_STATE",
        "WHOLE_PROCESS_LIVENESS_OUTSIDE_DECLARED_STATE_CHUNKS",
        "PHYSICAL_ENERGY_LOSS_BANDWIDTH_LATENCY_AND_CONTROL_WORK",
        "FINITE_PRECISION_REPETITION_OR_SHOT_COUNT_FOR_A_PHYSICAL_ESTIMATOR",
    }
    require(required_uninstrumented.issubset(set(resources["not_instrumented"])), "production resource caveats incomplete")

    reference_resources = reference["resource_accounting"]
    require((reference_resources["complex_state_amplitudes"], reference_resources["complex128_state_bytes"]) == (10256, 164096), "reference state accounting changed")
    require(reference_resources["full_rank_16_mixed_complex128_bytes"] == 2625536, "reference full-rank bytes changed")
    require(reference_resources["fixed_fixture_tensor_network_crossover_claim"] == "NONE", "reference tensor crossover promoted")
    require(reference_resources["asymptotic_resource_advantage_claim"] == "NONE", "reference asymptotic advantage promoted")
    require(not reference_resources["growing_exact_rank_is_approximation_lower_bound"], "reference rank lower bound promoted")
    require(reference_resources["forward_only_shadow_may_omit_restoration_tail"], "reference forward shadow weakened")
    require(reference_resources["promotion_requires_scaling_family"], "reference scaling requirement lost")

    forward_shadow = production["controls"]["forward_only_shadow"]
    require(forward_shadow["implemented_constructively"], "forward shadow missing")
    require(forward_shadow["state_no_greater_than_phase_model"], "forward-shadow state ceiling lost")
    require(forward_shadow["work_no_greater_than_phase_model"], "forward-shadow work ceiling lost")
    require(forward_shadow["m257_intact"], "M257 guardrail lost")
    require(set(production["global_not_established"]) == GLOBAL_NOT_ESTABLISHED, "global claim exclusions changed")
    require(not production["backend"]["qemu_device_executed"], "QEMU execution promoted")
    require(not production["backend"]["physical_execution"], "physical execution promoted")
    require(production["access_model"]["forward_shadow_may_execute_identical_sparse_recurrence"], "equal-access shadow hidden")
    require(not production["access_model"]["oracle_or_restricted_access_advantage_claimed"], "restricted-access advantage promoted")
    reference_claims = reference["metadata"]["claims"]
    for key in ("physical_observation", "physical_restoration", "same_backing_reuse", "resource_advantage", "m257_escape", "package_qualification"):
        require(not reference_claims[key], f"reference promoted {key}")


def compare_or_write_seals(
    generated_production: bytes,
    generated_reference: bytes,
    *,
    write_seals: bool,
) -> None:
    production_bytes = seal_bytes(generated_production)
    reference_bytes = seal_bytes(generated_reference)
    if write_seals:
        PRODUCTION_SEAL.parent.mkdir(parents=True, exist_ok=True)
        PRODUCTION_SEAL.write_bytes(production_bytes)
        REFERENCE_SEAL.write_bytes(reference_bytes)
        return
    require(PRODUCTION_SEAL.is_file(), "production evidence seal missing")
    require(REFERENCE_SEAL.is_file(), "reference evidence seal missing")
    require(PRODUCTION_SEAL.read_bytes() == production_bytes, "production evidence seal is stale")
    require(REFERENCE_SEAL.read_bytes() == reference_bytes, "reference evidence seal is stale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-seals",
        action="store_true",
        help="write canonical seals only after every strict-scope gate passes",
    )
    arguments = parser.parse_args()

    source_and_document_audit()
    generated_production = regenerate(PRODUCTION)
    generated_reference = regenerate(REFERENCE)
    production = json.loads(generated_production)
    reference = json.loads(generated_reference)
    metadata_and_obstruction(production, reference)
    geometry_locality_and_packets(production, reference)
    preparation_and_return_parity(production, reference)
    delays_modes_and_convergence(production, reference)
    handoff_controls_and_ceilings(production, reference)
    compare_or_write_seals(
        generated_production,
        generated_reference,
        write_seals=arguments.write_seals,
    )
    print(
        "PASS_STRICT_SCOPE M264_TIME_DOMAIN_SCATTERING_OBSTRUCTION "
        "SCIENCE=SEPARATE_REFERENCE_PARITY RESTORATION=NO_RESTORATION_CLAIM "
        "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
    )


if __name__ == "__main__":
    main()
